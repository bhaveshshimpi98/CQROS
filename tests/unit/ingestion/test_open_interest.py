"""Unit tests for CQROS historical open-interest downloader."""

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
    MILLISECONDS_PER_MINUTE,
)
from cqros.core.exceptions import ValidationError
from cqros.ingestion import (
    DEFAULT_OPEN_INTEREST_CHUNK_SAFETY_FACTOR,
    DEFAULT_OPEN_INTEREST_PERIOD,
    DEFAULT_OPEN_INTEREST_REQUEST_LIMIT,
    OPEN_INTEREST_PERIODS,
    OpenInterestDownloader,
    OpenInterestDownloadPlanner,
    OpenInterestDownloadTask,
)
from cqros.ingestion.chunk_sizing import DEFAULT_CHUNK_SAFETY_FACTOR
from cqros.ingestion.open_interest import (
    OpenInterestDownloader as OpenInterestDownloaderDirect,
)
from cqros.ingestion.open_interest import (
    OpenInterestDownloadPlanner as OpenInterestDownloadPlannerDirect,
)
from cqros.ingestion.open_interest import (
    OpenInterestDownloadTask as OpenInterestDownloadTaskDirect,
)

_SYMBOL = "BTCUSDT"
_PERIOD = "5m"
_START = 1_700_000_000_000
_FIVE_MINUTES = 5 * MILLISECONDS_PER_MINUTE
_CHUNK = 30 * _FIVE_MINUTES


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def _open_interest(
    timestamp: int,
    *,
    symbol: str = _SYMBOL,
    open_interest: str = "20403.63700000",
) -> dict[str, Any]:
    """Build a Binance-shaped open-interest history object."""
    return {
        "symbol": symbol,
        "sumOpenInterest": open_interest,
        "sumOpenInterestValue": "150570784.07809979",
        "timestamp": str(timestamp),
    }


def test_exports_match_module_symbols() -> None:
    """Package exports match the open-interest module classes and defaults."""
    assert OpenInterestDownloadTask is OpenInterestDownloadTaskDirect
    assert OpenInterestDownloadPlanner is OpenInterestDownloadPlannerDirect
    assert OpenInterestDownloader is OpenInterestDownloaderDirect
    assert DEFAULT_OPEN_INTEREST_PERIOD == "5m"
    assert DEFAULT_OPEN_INTEREST_REQUEST_LIMIT == 500
    assert DEFAULT_OPEN_INTEREST_CHUNK_SAFETY_FACTOR == DEFAULT_CHUNK_SAFETY_FACTOR
    assert "5m" in OPEN_INTEREST_PERIODS
    assert "1d" in OPEN_INTEREST_PERIODS


def test_open_interest_download_task_is_immutable() -> None:
    """OpenInterestDownloadTask is a frozen slotted dataclass."""
    task = OpenInterestDownloadTask(
        symbol=_SYMBOL,
        period=_PERIOD,
        start_time=_START,
        end_time=_START + 1_000,
    )
    assert is_dataclass(task)
    with pytest.raises(FrozenInstanceError):
        task.symbol = "ETHUSDT"  # type: ignore[misc]


def test_planner_rejects_non_positive_chunk_size() -> None:
    """Planner construction fails fast on invalid chunk sizes."""
    with pytest.raises(ValidationError) as exc_info:
        OpenInterestDownloadPlanner(chunk_size_ms=0)
    assert exc_info.value.error_code == "INGESTION-OPEN-INTEREST-001"


def test_planner_rejects_invalid_safety_factor() -> None:
    """Planner construction fails fast on invalid safety factors."""
    with pytest.raises(ValidationError) as exc_info:
        OpenInterestDownloadPlanner(safety_factor=0.0)
    assert exc_info.value.error_code == "INGESTION-OPEN-INTEREST-003"


def test_planner_rejects_unsupported_period() -> None:
    """Planner rejects periods outside the Binance open-interest allowlist."""
    planner = OpenInterestDownloadPlanner(chunk_size_ms=_CHUNK)
    with pytest.raises(ValidationError) as exc_info:
        planner.plan(
            symbol=_SYMBOL,
            period="8h",
            start_time=_START,
            end_time=_START + 1_000,
        )
    assert exc_info.value.error_code == "INGESTION-OPEN-INTEREST-008"


def test_planner_rejects_non_integer_timestamps() -> None:
    """Planner rejects non-integer Unix millisecond timestamps."""
    planner = OpenInterestDownloadPlanner(chunk_size_ms=_CHUNK)
    with pytest.raises(ValidationError) as exc_info:
        planner.plan(
            symbol=_SYMBOL,
            period=_PERIOD,
            start_time="1700000000000",  # type: ignore[arg-type]
            end_time=_START + 1_000,
        )
    assert exc_info.value.error_code == "INGESTION-OPEN-INTEREST-006"


def test_planner_splits_range_into_contiguous_tasks() -> None:
    """Long ranges are split into contiguous inclusive chunks."""
    planner = OpenInterestDownloadPlanner(chunk_size_ms=_CHUNK)
    tasks = planner.plan(
        symbol=_SYMBOL,
        period=_PERIOD,
        start_time=_START,
        end_time=_START + (2 * _CHUNK) + 100,
    )

    assert len(tasks) == 3
    assert all(isinstance(task, OpenInterestDownloadTask) for task in tasks)
    assert tasks[0] == OpenInterestDownloadTask(
        symbol=_SYMBOL,
        period=_PERIOD,
        start_time=_START,
        end_time=_START + _CHUNK - 1,
    )
    assert tasks[1].start_time == tasks[0].end_time + 1
    assert tasks[2].end_time == _START + (2 * _CHUNK) + 100
    assert tasks[0].end_time + 1 == tasks[1].start_time
    assert tasks[1].end_time + 1 == tasks[2].start_time


def test_planner_uses_period_aware_chunk_sizing_by_default() -> None:
    """Without a fixed chunk size, period duration drives task windows."""
    planner = OpenInterestDownloadPlanner(request_limit=10, safety_factor=1.0)
    tasks = planner.plan(
        symbol=_SYMBOL,
        period=_PERIOD,
        start_time=_START,
        end_time=_START + (15 * _FIVE_MINUTES),
    )
    assert len(tasks) == 2
    assert tasks[0].end_time - tasks[0].start_time + 1 == 10 * _FIVE_MINUTES


def test_planner_returns_empty_tuple_when_start_after_end() -> None:
    """Inverted ranges produce no tasks."""
    planner = OpenInterestDownloadPlanner(chunk_size_ms=_CHUNK)
    assert (
        planner.plan(
            symbol=_SYMBOL,
            period=_PERIOD,
            start_time=_START + 1,
            end_time=_START,
        )
        == ()
    )


def test_planner_single_task_when_range_fits_one_chunk() -> None:
    """Ranges within one chunk yield a single task."""
    planner = OpenInterestDownloadPlanner(chunk_size_ms=_CHUNK)
    tasks = planner.plan(
        symbol=_SYMBOL,
        period=_PERIOD,
        start_time=_START,
        end_time=_START + 1_000,
    )
    assert tasks == (
        OpenInterestDownloadTask(
            symbol=_SYMBOL,
            period=_PERIOD,
            start_time=_START,
            end_time=_START + 1_000,
        ),
    )


def test_downloader_rejects_non_positive_request_limit() -> None:
    """Downloader construction fails fast on invalid request limits."""
    with pytest.raises(ValidationError) as exc_info:
        OpenInterestDownloader(
            client=MagicMock(),
            repository=MagicMock(),
            planner=OpenInterestDownloadPlanner(chunk_size_ms=_CHUNK),
            request_limit=0,
        )
    assert exc_info.value.error_code == "INGESTION-OPEN-INTEREST-005"


def test_download_symbol_fetches_persists_and_avoids_paths() -> None:
    """Symbol download paginates open interest and persists year partitions."""
    client = MagicMock()
    client.get_open_interest_history = AsyncMock(
        side_effect=[
            [
                _open_interest(_START),
                _open_interest(_START + _FIVE_MINUTES, open_interest="21000.0"),
            ],
            [],
        ]
    )
    repository = MagicMock()
    planner = OpenInterestDownloadPlanner(chunk_size_ms=_CHUNK)
    downloader = OpenInterestDownloader(
        client,
        repository,
        planner,
        request_limit=2,
    )

    _run(
        downloader.download_symbol(
            symbol=_SYMBOL,
            period=_PERIOD,
            start_time=_START,
            end_time=_START + (2 * _FIVE_MINUTES),
        )
    )

    assert client.get_open_interest_history.await_count >= 1
    repository.save_open_interest.assert_called_once()
    args, kwargs = repository.save_open_interest.call_args
    frame = args[0]
    assert isinstance(frame, pl.DataFrame)
    assert frame.height == 2
    assert kwargs["exchange"] == EXCHANGE_BINANCE
    assert kwargs["market"] == MARKET_USDT_PERPETUAL
    assert kwargs["symbol"] == _SYMBOL
    assert kwargs["timeframe"] == _PERIOD
    assert kwargs["year"] == 2023
    assert "path" not in kwargs


def test_download_symbol_paginates_until_page_under_limit() -> None:
    """Full pages continue pagination; a short page ends the task fetch."""
    client = MagicMock()
    client.get_open_interest_history = AsyncMock(
        side_effect=[
            [
                _open_interest(_START),
                _open_interest(_START + _FIVE_MINUTES),
            ],
            [_open_interest(_START + (2 * _FIVE_MINUTES))],
        ]
    )
    repository = MagicMock()
    downloader = OpenInterestDownloader(
        client,
        repository,
        OpenInterestDownloadPlanner(chunk_size_ms=_CHUNK),
        request_limit=2,
    )

    _run(
        downloader.download_symbol(
            symbol=_SYMBOL,
            period=_PERIOD,
            start_time=_START,
            end_time=_START + (3 * _FIVE_MINUTES),
        )
    )

    assert client.get_open_interest_history.await_count == 2
    first_call = client.get_open_interest_history.await_args_list[0]
    second_call = client.get_open_interest_history.await_args_list[1]
    assert first_call.args == (_SYMBOL, _PERIOD)
    assert first_call.kwargs["start_time"] == _START
    assert second_call.kwargs["start_time"] == _START + _FIVE_MINUTES + 1
    frame = repository.save_open_interest.call_args.args[0]
    assert frame.height == 3


def test_download_symbol_skips_persist_when_empty() -> None:
    """Empty exchange responses do not write partitions."""
    client = MagicMock()
    client.get_open_interest_history = AsyncMock(return_value=[])
    repository = MagicMock()
    downloader = OpenInterestDownloader(
        client,
        repository,
        OpenInterestDownloadPlanner(chunk_size_ms=_CHUNK),
    )

    _run(
        downloader.download_symbol(
            symbol=_SYMBOL,
            period=_PERIOD,
            start_time=_START,
            end_time=_START + 1_000,
        )
    )

    repository.save_open_interest.assert_not_called()


def test_fetch_symbol_returns_empty_frame_for_inverted_range() -> None:
    """Inverted ranges return an empty canonical open-interest frame."""
    client = MagicMock()
    client.get_open_interest_history = AsyncMock()
    downloader = OpenInterestDownloader(
        client,
        MagicMock(),
        OpenInterestDownloadPlanner(chunk_size_ms=_CHUNK),
    )

    frame = _run(
        downloader.fetch_symbol(
            symbol=_SYMBOL,
            period=_PERIOD,
            start_time=_START + 1,
            end_time=_START,
        )
    )

    assert frame.height == 0
    assert frame.columns == ["symbol", "timestamp", "open_interest"]
    client.get_open_interest_history.assert_not_called()


def test_download_universe_processes_symbols_sequentially() -> None:
    """Universe download invokes symbol download for each symbol in order."""
    client = MagicMock()
    client.get_open_interest_history = AsyncMock(
        side_effect=[
            [_open_interest(_START)],
            [_open_interest(_START, symbol="ETHUSDT")],
        ]
    )
    repository = MagicMock()
    downloader = OpenInterestDownloader(
        client,
        repository,
        OpenInterestDownloadPlanner(chunk_size_ms=_CHUNK),
    )

    _run(
        downloader.download_universe(
            ["BTCUSDT", "ETHUSDT"],
            period=_PERIOD,
            start_time=_START,
            end_time=_START + 1_000,
        )
    )

    assert repository.save_open_interest.call_count == 2
    symbols = [call.kwargs["symbol"] for call in repository.save_open_interest.call_args_list]
    assert symbols == ["BTCUSDT", "ETHUSDT"]


def test_invalid_open_interest_payload_raises_validation_error() -> None:
    """Malformed open-interest payloads fail with a validation error."""
    client = MagicMock()
    client.get_open_interest_history = AsyncMock(return_value={"not": "a list"})
    downloader = OpenInterestDownloader(
        client,
        MagicMock(),
        OpenInterestDownloadPlanner(chunk_size_ms=_CHUNK),
    )

    with pytest.raises(ValidationError) as exc_info:
        _run(
            downloader.download_symbol(
                symbol=_SYMBOL,
                period=_PERIOD,
                start_time=_START,
                end_time=_START + 1_000,
            )
        )
    assert exc_info.value.error_code == "INGESTION-OPEN-INTEREST-009"


def test_non_mapping_open_interest_row_raises_validation_error() -> None:
    """Non-object open-interest rows fail validation."""
    client = MagicMock()
    client.get_open_interest_history = AsyncMock(return_value=[["not", "an", "object"]])
    downloader = OpenInterestDownloader(
        client,
        MagicMock(),
        OpenInterestDownloadPlanner(chunk_size_ms=_CHUNK),
    )

    with pytest.raises(ValidationError) as exc_info:
        _run(
            downloader.download_symbol(
                symbol=_SYMBOL,
                period=_PERIOD,
                start_time=_START,
                end_time=_START + 1_000,
            )
        )
    assert exc_info.value.error_code == "INGESTION-OPEN-INTEREST-010"


def test_missing_required_open_interest_fields_raise_validation_error() -> None:
    """Rows missing required vendor fields fail validation."""
    client = MagicMock()
    client.get_open_interest_history = AsyncMock(
        return_value=[{"symbol": _SYMBOL, "timestamp": _START}]
    )
    downloader = OpenInterestDownloader(
        client,
        MagicMock(),
        OpenInterestDownloadPlanner(chunk_size_ms=_CHUNK),
    )

    with pytest.raises(ValidationError) as exc_info:
        _run(
            downloader.download_symbol(
                symbol=_SYMBOL,
                period=_PERIOD,
                start_time=_START,
                end_time=_START + 1_000,
            )
        )
    assert exc_info.value.error_code == "INGESTION-OPEN-INTEREST-011"
    assert "sumOpenInterest" in exc_info.value.details["missing"]


def test_mismatched_open_interest_symbol_raises_validation_error() -> None:
    """Payload symbols that disagree with the request fail validation."""
    client = MagicMock()
    client.get_open_interest_history = AsyncMock(
        return_value=[_open_interest(_START, symbol="ETHUSDT")]
    )
    downloader = OpenInterestDownloader(
        client,
        MagicMock(),
        OpenInterestDownloadPlanner(chunk_size_ms=_CHUNK),
    )

    with pytest.raises(ValidationError) as exc_info:
        _run(
            downloader.download_symbol(
                symbol=_SYMBOL,
                period=_PERIOD,
                start_time=_START,
                end_time=_START + 1_000,
            )
        )
    assert exc_info.value.error_code == "INGESTION-OPEN-INTEREST-012"


def test_persisted_frame_schema_matches_canonical_columns() -> None:
    """Persisted frames use the canonical raw open-interest column set."""
    client = MagicMock()
    client.get_open_interest_history = AsyncMock(
        return_value=[_open_interest(_START, open_interest="12345.67")]
    )
    repository = MagicMock()
    downloader = OpenInterestDownloader(
        client,
        repository,
        OpenInterestDownloadPlanner(chunk_size_ms=_CHUNK),
    )

    _run(
        downloader.download_symbol(
            symbol=_SYMBOL,
            period=_PERIOD,
            start_time=_START,
            end_time=_START + 1_000,
        )
    )

    frame = repository.save_open_interest.call_args.args[0]
    expected = pl.DataFrame(
        {
            "symbol": [_SYMBOL],
            "timestamp": [_START],
            "open_interest": [12345.67],
        }
    )
    assert_frame_equal(frame, expected)


def test_fetch_symbol_does_not_persist() -> None:
    """fetch_symbol returns data without writing repository partitions."""
    client = MagicMock()
    client.get_open_interest_history = AsyncMock(return_value=[_open_interest(_START)])
    repository = MagicMock()
    downloader = OpenInterestDownloader(
        client,
        repository,
        OpenInterestDownloadPlanner(chunk_size_ms=_CHUNK),
    )

    frame = _run(
        downloader.fetch_symbol(
            symbol=_SYMBOL,
            period=_PERIOD,
            start_time=_START,
            end_time=_START + 1_000,
        )
    )

    assert frame.height == 1
    repository.save_open_interest.assert_not_called()


def test_pagination_stops_when_cursor_does_not_advance() -> None:
    """Stuck pagination cursors terminate instead of looping forever."""
    client = MagicMock()
    client.get_open_interest_history = AsyncMock(
        return_value=[
            _open_interest(_START),
            _open_interest(_START),
        ]
    )
    repository = MagicMock()
    downloader = OpenInterestDownloader(
        client,
        repository,
        OpenInterestDownloadPlanner(chunk_size_ms=_CHUNK),
        request_limit=2,
    )

    _run(
        downloader.download_symbol(
            symbol=_SYMBOL,
            period=_PERIOD,
            start_time=_START,
            end_time=_START + _FIVE_MINUTES,
        )
    )

    assert client.get_open_interest_history.await_count == 2
    frame = repository.save_open_interest.call_args.args[0]
    assert frame.height == 4


def test_invalid_open_interest_value_raises_validation_error() -> None:
    """Non-numeric sumOpenInterest values fail validation."""
    client = MagicMock()
    client.get_open_interest_history = AsyncMock(
        return_value=[
            {
                "symbol": _SYMBOL,
                "sumOpenInterest": "not-a-number",
                "timestamp": str(_START),
            }
        ]
    )
    downloader = OpenInterestDownloader(
        client,
        MagicMock(),
        OpenInterestDownloadPlanner(chunk_size_ms=_CHUNK),
    )

    with pytest.raises(ValidationError) as exc_info:
        _run(
            downloader.download_symbol(
                symbol=_SYMBOL,
                period=_PERIOD,
                start_time=_START,
                end_time=_START + 1_000,
            )
        )
    assert exc_info.value.error_code == "INGESTION-OPEN-INTEREST-015"
