"""Unit tests for CQROS historical funding-rate downloader."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from dataclasses import FrozenInstanceError, is_dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.core.constants import (
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
    MILLISECONDS_PER_HOUR,
)
from cqros.core.exceptions import ValidationError
from cqros.ingestion import (
    DEFAULT_FUNDING_CHUNK_SIZE_MS,
    DEFAULT_FUNDING_INTERVAL_MS,
    DEFAULT_FUNDING_REQUEST_LIMIT,
    DEFAULT_FUNDING_TIMEFRAME,
    FundingDownloader,
    FundingDownloadPlanner,
    FundingDownloadTask,
)
from cqros.ingestion.chunk_sizing import DEFAULT_CHUNK_SAFETY_FACTOR
from cqros.ingestion.funding import (
    FundingDownloader as FundingDownloaderDirect,
)
from cqros.ingestion.funding import (
    FundingDownloadPlanner as FundingDownloadPlannerDirect,
)
from cqros.ingestion.funding import (
    FundingDownloadTask as FundingDownloadTaskDirect,
)

_SYMBOL = "BTCUSDT"
_START = 1_700_000_000_000
_EIGHT_HOURS = 8 * MILLISECONDS_PER_HOUR
_CHUNK = 30 * _EIGHT_HOURS


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def _funding(
    funding_time: int,
    *,
    symbol: str = _SYMBOL,
    funding_rate: str = "0.00010000",
    mark_price: str | None = "42000.50",
) -> dict[str, Any]:
    """Build a Binance-shaped funding-rate object."""
    row: dict[str, Any] = {
        "symbol": symbol,
        "fundingRate": funding_rate,
        "fundingTime": funding_time,
    }
    if mark_price is not None:
        row["markPrice"] = mark_price
    return row


def test_exports_match_module_symbols() -> None:
    """Package exports match the funding module classes and defaults."""
    assert FundingDownloadTask is FundingDownloadTaskDirect
    assert FundingDownloadPlanner is FundingDownloadPlannerDirect
    assert FundingDownloader is FundingDownloaderDirect
    assert DEFAULT_FUNDING_INTERVAL_MS == _EIGHT_HOURS
    assert DEFAULT_FUNDING_REQUEST_LIMIT == 1_000
    assert DEFAULT_FUNDING_TIMEFRAME == "8h"
    assert DEFAULT_FUNDING_CHUNK_SIZE_MS == int(
        DEFAULT_FUNDING_REQUEST_LIMIT * DEFAULT_CHUNK_SAFETY_FACTOR * DEFAULT_FUNDING_INTERVAL_MS
    )


def test_funding_download_task_is_immutable() -> None:
    """FundingDownloadTask is a frozen slotted dataclass."""
    task = FundingDownloadTask(
        symbol=_SYMBOL,
        start_time=_START,
        end_time=_START + 1_000,
    )
    assert is_dataclass(task)
    with pytest.raises(FrozenInstanceError):
        task.symbol = "ETHUSDT"  # type: ignore[misc]


def test_planner_rejects_non_positive_chunk_size() -> None:
    """Planner construction fails fast on invalid chunk sizes."""
    with pytest.raises(ValidationError) as exc_info:
        FundingDownloadPlanner(chunk_size_ms=0)
    assert exc_info.value.error_code == "INGESTION-FUNDING-001"


def test_planner_rejects_non_integer_timestamps() -> None:
    """Planner rejects non-integer Unix millisecond timestamps."""
    planner = FundingDownloadPlanner(chunk_size_ms=_CHUNK)
    with pytest.raises(ValidationError) as exc_info:
        planner.plan(
            symbol=_SYMBOL,
            start_time="1700000000000",  # type: ignore[arg-type]
            end_time=_START + 1_000,
        )
    assert exc_info.value.error_code == "INGESTION-FUNDING-003"


def test_planner_splits_range_into_contiguous_tasks() -> None:
    """Long ranges are split into contiguous inclusive chunks."""
    planner = FundingDownloadPlanner(chunk_size_ms=_CHUNK)
    tasks = planner.plan(
        symbol=_SYMBOL,
        start_time=_START,
        end_time=_START + (2 * _CHUNK) + 100,
    )

    assert len(tasks) == 3
    assert all(isinstance(task, FundingDownloadTask) for task in tasks)
    assert tasks[0] == FundingDownloadTask(
        symbol=_SYMBOL,
        start_time=_START,
        end_time=_START + _CHUNK - 1,
    )
    assert tasks[1].start_time == tasks[0].end_time + 1
    assert tasks[2].end_time == _START + (2 * _CHUNK) + 100
    assert tasks[0].end_time + 1 == tasks[1].start_time
    assert tasks[1].end_time + 1 == tasks[2].start_time


def test_planner_returns_empty_tuple_when_start_after_end() -> None:
    """Inverted ranges produce no tasks."""
    planner = FundingDownloadPlanner(chunk_size_ms=_CHUNK)
    assert (
        planner.plan(
            symbol=_SYMBOL,
            start_time=_START + 1,
            end_time=_START,
        )
        == ()
    )


def test_planner_single_task_when_range_fits_one_chunk() -> None:
    """Ranges within one chunk yield a single task."""
    planner = FundingDownloadPlanner(chunk_size_ms=_CHUNK)
    tasks = planner.plan(
        symbol=_SYMBOL,
        start_time=_START,
        end_time=_START + 1_000,
    )
    assert tasks == (
        FundingDownloadTask(
            symbol=_SYMBOL,
            start_time=_START,
            end_time=_START + 1_000,
        ),
    )


def test_downloader_rejects_non_positive_funding_limit() -> None:
    """Downloader construction fails fast on invalid request limits."""
    with pytest.raises(ValidationError) as exc_info:
        FundingDownloader(
            client=MagicMock(),
            repository=MagicMock(),
            planner=FundingDownloadPlanner(),
            funding_limit=0,
        )
    assert exc_info.value.error_code == "INGESTION-FUNDING-002"


def test_download_symbol_fetches_persists_and_avoids_paths() -> None:
    """Symbol download paginates funding and persists year partitions."""
    client = MagicMock()
    client.get_funding_rates = AsyncMock(
        side_effect=[
            [
                _funding(_START),
                _funding(_START + _EIGHT_HOURS, funding_rate="0.00020000"),
            ],
            [],
        ]
    )
    repository = MagicMock()
    planner = FundingDownloadPlanner(chunk_size_ms=_CHUNK)
    downloader = FundingDownloader(
        client,
        repository,
        planner,
        funding_limit=2,
    )

    _run(
        downloader.download_symbol(
            symbol=_SYMBOL,
            start_time=_START,
            end_time=_START + (2 * _EIGHT_HOURS),
        )
    )

    assert client.get_funding_rates.await_count >= 1
    repository.save_funding.assert_called_once()
    args, kwargs = repository.save_funding.call_args
    frame = args[0]
    assert isinstance(frame, pl.DataFrame)
    assert frame.height == 2
    assert kwargs["exchange"] == EXCHANGE_BINANCE
    assert kwargs["market"] == MARKET_USDT_PERPETUAL
    assert kwargs["symbol"] == _SYMBOL
    assert kwargs["timeframe"] == DEFAULT_FUNDING_TIMEFRAME
    assert kwargs["year"] == 2023
    assert "path" not in kwargs


def test_download_symbol_paginates_until_page_under_limit() -> None:
    """Full pages continue pagination; a short page ends the task fetch."""
    client = MagicMock()
    client.get_funding_rates = AsyncMock(
        side_effect=[
            [
                _funding(_START),
                _funding(_START + _EIGHT_HOURS),
            ],
            [_funding(_START + (2 * _EIGHT_HOURS))],
        ]
    )
    repository = MagicMock()
    downloader = FundingDownloader(
        client,
        repository,
        FundingDownloadPlanner(chunk_size_ms=_CHUNK),
        funding_limit=2,
    )

    _run(
        downloader.download_symbol(
            symbol=_SYMBOL,
            start_time=_START,
            end_time=_START + (3 * _EIGHT_HOURS),
        )
    )

    assert client.get_funding_rates.await_count == 2
    first_call = client.get_funding_rates.await_args_list[0]
    second_call = client.get_funding_rates.await_args_list[1]
    assert first_call.kwargs["start_time"] == _START
    assert second_call.kwargs["start_time"] == _START + _EIGHT_HOURS + 1
    frame = repository.save_funding.call_args.args[0]
    assert frame.height == 3


def test_download_symbol_skips_persist_when_empty() -> None:
    """Empty exchange responses do not write partitions."""
    client = MagicMock()
    client.get_funding_rates = AsyncMock(return_value=[])
    repository = MagicMock()
    downloader = FundingDownloader(
        client,
        repository,
        FundingDownloadPlanner(chunk_size_ms=_CHUNK),
    )

    _run(
        downloader.download_symbol(
            symbol=_SYMBOL,
            start_time=_START,
            end_time=_START + 1_000,
        )
    )

    repository.save_funding.assert_not_called()


def test_fetch_symbol_returns_empty_frame_for_inverted_range() -> None:
    """Inverted ranges return an empty canonical funding frame."""
    client = MagicMock()
    client.get_funding_rates = AsyncMock()
    downloader = FundingDownloader(
        client,
        MagicMock(),
        FundingDownloadPlanner(chunk_size_ms=_CHUNK),
    )

    frame = _run(
        downloader.fetch_symbol(
            symbol=_SYMBOL,
            start_time=_START + 1,
            end_time=_START,
        )
    )

    assert frame.height == 0
    assert frame.columns == [
        "symbol",
        "funding_time",
        "funding_rate",
        "mark_price",
    ]
    client.get_funding_rates.assert_not_called()


def test_download_universe_processes_symbols_sequentially() -> None:
    """Universe download invokes symbol download for each symbol in order."""
    client = MagicMock()
    client.get_funding_rates = AsyncMock(
        side_effect=[
            [_funding(_START)],
            [_funding(_START, symbol="ETHUSDT")],
        ]
    )
    repository = MagicMock()
    downloader = FundingDownloader(
        client,
        repository,
        FundingDownloadPlanner(chunk_size_ms=_CHUNK),
    )

    _run(
        downloader.download_universe(
            ["BTCUSDT", "ETHUSDT"],
            start_time=_START,
            end_time=_START + 1_000,
        )
    )

    assert repository.save_funding.call_count == 2
    symbols = [call.kwargs["symbol"] for call in repository.save_funding.call_args_list]
    assert symbols == ["BTCUSDT", "ETHUSDT"]


def test_invalid_funding_payload_raises_validation_error() -> None:
    """Malformed funding payloads fail with a validation error."""
    client = MagicMock()
    client.get_funding_rates = AsyncMock(return_value={"not": "a list"})
    downloader = FundingDownloader(
        client,
        MagicMock(),
        FundingDownloadPlanner(chunk_size_ms=_CHUNK),
    )

    with pytest.raises(ValidationError) as exc_info:
        _run(
            downloader.download_symbol(
                symbol=_SYMBOL,
                start_time=_START,
                end_time=_START + 1_000,
            )
        )
    assert exc_info.value.error_code == "INGESTION-FUNDING-004"


def test_non_mapping_funding_row_raises_validation_error() -> None:
    """Non-object funding rows fail validation."""
    client = MagicMock()
    client.get_funding_rates = AsyncMock(return_value=[["not", "an", "object"]])
    downloader = FundingDownloader(
        client,
        MagicMock(),
        FundingDownloadPlanner(chunk_size_ms=_CHUNK),
    )

    with pytest.raises(ValidationError) as exc_info:
        _run(
            downloader.download_symbol(
                symbol=_SYMBOL,
                start_time=_START,
                end_time=_START + 1_000,
            )
        )
    assert exc_info.value.error_code == "INGESTION-FUNDING-005"


def test_missing_required_funding_fields_raise_validation_error() -> None:
    """Rows missing required vendor fields fail validation."""
    client = MagicMock()
    client.get_funding_rates = AsyncMock(return_value=[{"symbol": _SYMBOL, "fundingTime": _START}])
    downloader = FundingDownloader(
        client,
        MagicMock(),
        FundingDownloadPlanner(chunk_size_ms=_CHUNK),
    )

    with pytest.raises(ValidationError) as exc_info:
        _run(
            downloader.download_symbol(
                symbol=_SYMBOL,
                start_time=_START,
                end_time=_START + 1_000,
            )
        )
    assert exc_info.value.error_code == "INGESTION-FUNDING-006"
    assert "fundingRate" in exc_info.value.details["missing"]


def test_mismatched_funding_symbol_raises_validation_error() -> None:
    """Payload symbols that disagree with the request fail validation."""
    client = MagicMock()
    client.get_funding_rates = AsyncMock(return_value=[_funding(_START, symbol="ETHUSDT")])
    downloader = FundingDownloader(
        client,
        MagicMock(),
        FundingDownloadPlanner(chunk_size_ms=_CHUNK),
    )

    with pytest.raises(ValidationError) as exc_info:
        _run(
            downloader.download_symbol(
                symbol=_SYMBOL,
                start_time=_START,
                end_time=_START + 1_000,
            )
        )
    assert exc_info.value.error_code == "INGESTION-FUNDING-007"


def test_persisted_frame_schema_matches_canonical_columns() -> None:
    """Persisted frames use the canonical raw funding column set."""
    client = MagicMock()
    client.get_funding_rates = AsyncMock(
        return_value=[_funding(_START, funding_rate="0.00012345", mark_price="50123.45")]
    )
    repository = MagicMock()
    downloader = FundingDownloader(
        client,
        repository,
        FundingDownloadPlanner(chunk_size_ms=_CHUNK),
    )

    _run(
        downloader.download_symbol(
            symbol=_SYMBOL,
            start_time=_START,
            end_time=_START + 1_000,
        )
    )

    frame = repository.save_funding.call_args.args[0]
    expected = pl.DataFrame(
        {
            "symbol": [_SYMBOL],
            "funding_time": [_START],
            "funding_rate": [0.00012345],
            "mark_price": [50123.45],
        }
    )
    assert_frame_equal(frame, expected)


def test_missing_mark_price_persists_as_null() -> None:
    """Absent or blank markPrice values become null floats."""
    client = MagicMock()
    client.get_funding_rates = AsyncMock(
        side_effect=[
            [
                _funding(_START, mark_price=None),
                {
                    "symbol": _SYMBOL,
                    "fundingRate": "0.00020000",
                    "fundingTime": _START + _EIGHT_HOURS,
                    "markPrice": "",
                },
            ]
        ]
    )
    repository = MagicMock()
    downloader = FundingDownloader(
        client,
        repository,
        FundingDownloadPlanner(chunk_size_ms=_CHUNK),
        funding_limit=10,
    )

    _run(
        downloader.download_symbol(
            symbol=_SYMBOL,
            start_time=_START,
            end_time=_START + (2 * _EIGHT_HOURS),
        )
    )

    frame = repository.save_funding.call_args.args[0]
    assert frame["mark_price"].null_count() == 2
    assert frame["funding_rate"].to_list() == pytest.approx([0.0001, 0.0002])


def test_fetch_symbol_does_not_persist() -> None:
    """fetch_symbol returns data without writing repository partitions."""
    client = MagicMock()
    client.get_funding_rates = AsyncMock(return_value=[_funding(_START)])
    repository = MagicMock()
    downloader = FundingDownloader(
        client,
        repository,
        FundingDownloadPlanner(chunk_size_ms=_CHUNK),
    )

    frame = _run(
        downloader.fetch_symbol(
            symbol=_SYMBOL,
            start_time=_START,
            end_time=_START + 1_000,
        )
    )

    assert frame.height == 1
    repository.save_funding.assert_not_called()


def test_pagination_stops_when_cursor_does_not_advance() -> None:
    """Stuck pagination cursors terminate instead of looping forever."""
    client = MagicMock()
    client.get_funding_rates = AsyncMock(
        return_value=[
            _funding(_START),
            _funding(_START),
        ]
    )
    repository = MagicMock()
    downloader = FundingDownloader(
        client,
        repository,
        FundingDownloadPlanner(chunk_size_ms=_CHUNK),
        funding_limit=2,
    )

    _run(
        downloader.download_symbol(
            symbol=_SYMBOL,
            start_time=_START,
            end_time=_START + _EIGHT_HOURS,
        )
    )

    # First full page advances the cursor to last_funding_time + 1. The second
    # identical page then fails the cursor-advance guard and terminates.
    assert client.get_funding_rates.await_count == 2
    frame = repository.save_funding.call_args.args[0]
    assert frame.height == 4


def test_invalid_funding_rate_value_raises_validation_error() -> None:
    """Non-numeric fundingRate values fail validation."""
    client = MagicMock()
    client.get_funding_rates = AsyncMock(
        return_value=[
            {
                "symbol": _SYMBOL,
                "fundingRate": "not-a-number",
                "fundingTime": _START,
                "markPrice": "1.0",
            }
        ]
    )
    downloader = FundingDownloader(
        client,
        MagicMock(),
        FundingDownloadPlanner(chunk_size_ms=_CHUNK),
    )

    with pytest.raises(ValidationError) as exc_info:
        _run(
            downloader.download_symbol(
                symbol=_SYMBOL,
                start_time=_START,
                end_time=_START + 1_000,
            )
        )
    assert exc_info.value.error_code == "INGESTION-FUNDING-010"
