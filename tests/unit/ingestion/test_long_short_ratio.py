"""Unit tests for CQROS historical long/short ratio downloader."""

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
    DEFAULT_LONG_SHORT_CHUNK_SAFETY_FACTOR,
    DEFAULT_LONG_SHORT_PERIOD,
    DEFAULT_LONG_SHORT_REQUEST_LIMIT,
    LONG_SHORT_PERIODS,
    LongShortDownloader,
    LongShortDownloadPlanner,
    LongShortDownloadTask,
    LongShortRatioKind,
)
from cqros.ingestion.chunk_sizing import DEFAULT_CHUNK_SAFETY_FACTOR
from cqros.ingestion.long_short_ratio import (
    LongShortDownloader as LongShortDownloaderDirect,
)
from cqros.ingestion.long_short_ratio import (
    LongShortDownloadPlanner as LongShortDownloadPlannerDirect,
)
from cqros.ingestion.long_short_ratio import (
    LongShortDownloadTask as LongShortDownloadTaskDirect,
)
from cqros.ingestion.long_short_ratio import (
    LongShortRatioKind as LongShortRatioKindDirect,
)

_SYMBOL = "BTCUSDT"
_PERIOD = "5m"
_KIND = LongShortRatioKind.GLOBAL_ACCOUNT
_START = 1_700_000_000_000
_FIVE_MINUTES = 5 * MILLISECONDS_PER_MINUTE
_CHUNK = 30 * _FIVE_MINUTES


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def _long_short(
    timestamp: int,
    *,
    symbol: str = _SYMBOL,
    long_account: str = "0.6622",
    short_account: str = "0.3378",
    long_short_ratio: str = "1.9600",
) -> dict[str, Any]:
    """Build a Binance-shaped long/short ratio object."""
    return {
        "symbol": symbol,
        "longAccount": long_account,
        "shortAccount": short_account,
        "longShortRatio": long_short_ratio,
        "timestamp": str(timestamp),
    }


def _configure_client(client: MagicMock, *, kind: LongShortRatioKind) -> AsyncMock:
    """Attach AsyncMocks for all long/short endpoints; return the active one."""
    active = AsyncMock(return_value=[_long_short(_START)])
    client.get_global_long_short_account_ratio = (
        active if kind is LongShortRatioKind.GLOBAL_ACCOUNT else AsyncMock()
    )
    client.get_top_long_short_account_ratio = (
        active if kind is LongShortRatioKind.TOP_TRADER_ACCOUNT else AsyncMock()
    )
    client.get_top_long_short_position_ratio = (
        active if kind is LongShortRatioKind.TOP_TRADER_POSITION else AsyncMock()
    )
    return active


def test_exports_match_module_symbols() -> None:
    """Package exports match the long/short module classes and defaults."""
    assert LongShortDownloadTask is LongShortDownloadTaskDirect
    assert LongShortDownloadPlanner is LongShortDownloadPlannerDirect
    assert LongShortDownloader is LongShortDownloaderDirect
    assert LongShortRatioKind is LongShortRatioKindDirect
    assert DEFAULT_LONG_SHORT_PERIOD == "5m"
    assert DEFAULT_LONG_SHORT_REQUEST_LIMIT == 500
    assert DEFAULT_LONG_SHORT_CHUNK_SAFETY_FACTOR == DEFAULT_CHUNK_SAFETY_FACTOR
    assert "5m" in LONG_SHORT_PERIODS
    assert LongShortRatioKind.GLOBAL_ACCOUNT.value == "global_long_short_account_ratio"


def test_long_short_download_task_is_immutable() -> None:
    """LongShortDownloadTask is a frozen slotted dataclass."""
    task = LongShortDownloadTask(
        symbol=_SYMBOL,
        period=_PERIOD,
        kind=_KIND,
        start_time=_START,
        end_time=_START + 1_000,
    )
    assert is_dataclass(task)
    with pytest.raises(FrozenInstanceError):
        task.symbol = "ETHUSDT"  # type: ignore[misc]


def test_planner_rejects_non_positive_chunk_size() -> None:
    """Planner construction fails fast on invalid chunk sizes."""
    with pytest.raises(ValidationError) as exc_info:
        LongShortDownloadPlanner(chunk_size_ms=0)
    assert exc_info.value.error_code == "INGESTION-LONG-SHORT-001"


def test_planner_rejects_unsupported_period_and_kind() -> None:
    """Planner rejects unsupported periods and dataset kinds."""
    planner = LongShortDownloadPlanner(chunk_size_ms=_CHUNK)
    with pytest.raises(ValidationError) as period_exc:
        planner.plan(
            symbol=_SYMBOL,
            period="8h",
            kind=_KIND,
            start_time=_START,
            end_time=_START + 1_000,
        )
    assert period_exc.value.error_code == "INGESTION-LONG-SHORT-008"

    with pytest.raises(ValidationError) as kind_exc:
        planner.plan(
            symbol=_SYMBOL,
            period=_PERIOD,
            kind="not-a-kind",  # type: ignore[arg-type]
            start_time=_START,
            end_time=_START + 1_000,
        )
    assert kind_exc.value.error_code == "INGESTION-LONG-SHORT-009"


def test_planner_splits_range_into_contiguous_tasks() -> None:
    """Long ranges are split into contiguous inclusive chunks."""
    planner = LongShortDownloadPlanner(chunk_size_ms=_CHUNK)
    tasks = planner.plan(
        symbol=_SYMBOL,
        period=_PERIOD,
        kind=_KIND,
        start_time=_START,
        end_time=_START + (2 * _CHUNK) + 100,
    )

    assert len(tasks) == 3
    assert all(isinstance(task, LongShortDownloadTask) for task in tasks)
    assert tasks[0] == LongShortDownloadTask(
        symbol=_SYMBOL,
        period=_PERIOD,
        kind=_KIND,
        start_time=_START,
        end_time=_START + _CHUNK - 1,
    )
    assert tasks[1].start_time == tasks[0].end_time + 1
    assert tasks[2].end_time == _START + (2 * _CHUNK) + 100


def test_downloader_rejects_non_positive_request_limit() -> None:
    """Downloader construction fails fast on invalid request limits."""
    with pytest.raises(ValidationError) as exc_info:
        LongShortDownloader(
            client=MagicMock(),
            repository=MagicMock(),
            planner=LongShortDownloadPlanner(chunk_size_ms=_CHUNK),
            request_limit=0,
        )
    assert exc_info.value.error_code == "INGESTION-LONG-SHORT-005"


@pytest.mark.parametrize(
    ("kind", "save_name", "client_attr"),
    [
        (
            LongShortRatioKind.GLOBAL_ACCOUNT,
            "save_global_long_short_account_ratio",
            "get_global_long_short_account_ratio",
        ),
        (
            LongShortRatioKind.TOP_TRADER_ACCOUNT,
            "save_top_long_short_account_ratio",
            "get_top_long_short_account_ratio",
        ),
        (
            LongShortRatioKind.TOP_TRADER_POSITION,
            "save_top_long_short_position_ratio",
            "get_top_long_short_position_ratio",
        ),
    ],
)
def test_download_symbol_routes_to_matching_endpoint_and_storage(
    kind: LongShortRatioKind,
    save_name: str,
    client_attr: str,
) -> None:
    """Each dataset kind uses its own client endpoint and repository method."""
    client = MagicMock()
    active = _configure_client(client, kind=kind)
    active.side_effect = [
        [
            _long_short(_START),
            _long_short(_START + _FIVE_MINUTES, long_short_ratio="2.0000"),
        ],
        [],
    ]
    repository = MagicMock()
    downloader = LongShortDownloader(
        client,
        repository,
        LongShortDownloadPlanner(chunk_size_ms=_CHUNK),
        request_limit=2,
    )

    _run(
        downloader.download_symbol(
            symbol=_SYMBOL,
            kind=kind,
            period=_PERIOD,
            start_time=_START,
            end_time=_START + (2 * _FIVE_MINUTES),
        )
    )

    assert getattr(client, client_attr).await_count >= 1
    save = getattr(repository, save_name)
    save.assert_called_once()
    args, kwargs = save.call_args
    frame = args[0]
    assert isinstance(frame, pl.DataFrame)
    assert frame.height == 2
    assert kwargs["exchange"] == EXCHANGE_BINANCE
    assert kwargs["market"] == MARKET_USDT_PERPETUAL
    assert kwargs["symbol"] == _SYMBOL
    assert kwargs["timeframe"] == _PERIOD
    assert kwargs["year"] == 2023
    assert "path" not in kwargs

    # Other dataset savers must remain untouched.
    for other in (
        "save_global_long_short_account_ratio",
        "save_top_long_short_account_ratio",
        "save_top_long_short_position_ratio",
    ):
        if other != save_name:
            getattr(repository, other).assert_not_called()


def test_download_symbol_paginates_until_page_under_limit() -> None:
    """Full pages continue pagination; a short page ends the task fetch."""
    client = MagicMock()
    active = _configure_client(client, kind=_KIND)
    active.side_effect = [
        [_long_short(_START), _long_short(_START + _FIVE_MINUTES)],
        [_long_short(_START + (2 * _FIVE_MINUTES))],
    ]
    repository = MagicMock()
    downloader = LongShortDownloader(
        client,
        repository,
        LongShortDownloadPlanner(chunk_size_ms=_CHUNK),
        request_limit=2,
    )

    _run(
        downloader.download_symbol(
            symbol=_SYMBOL,
            kind=_KIND,
            period=_PERIOD,
            start_time=_START,
            end_time=_START + (3 * _FIVE_MINUTES),
        )
    )

    assert active.await_count == 2
    assert active.await_args_list[1].kwargs["start_time"] == _START + _FIVE_MINUTES + 1
    frame = repository.save_global_long_short_account_ratio.call_args.args[0]
    assert frame.height == 3


def test_download_symbol_skips_persist_when_empty() -> None:
    """Empty exchange responses do not write partitions."""
    client = MagicMock()
    active = _configure_client(client, kind=_KIND)
    active.return_value = []
    repository = MagicMock()
    downloader = LongShortDownloader(
        client,
        repository,
        LongShortDownloadPlanner(chunk_size_ms=_CHUNK),
    )

    _run(
        downloader.download_symbol(
            symbol=_SYMBOL,
            kind=_KIND,
            period=_PERIOD,
            start_time=_START,
            end_time=_START + 1_000,
        )
    )

    repository.save_global_long_short_account_ratio.assert_not_called()


def test_fetch_symbol_returns_empty_frame_for_inverted_range() -> None:
    """Inverted ranges return an empty canonical long/short frame."""
    client = MagicMock()
    _configure_client(client, kind=_KIND)
    downloader = LongShortDownloader(
        client,
        MagicMock(),
        LongShortDownloadPlanner(chunk_size_ms=_CHUNK),
    )

    frame = _run(
        downloader.fetch_symbol(
            symbol=_SYMBOL,
            kind=_KIND,
            period=_PERIOD,
            start_time=_START + 1,
            end_time=_START,
        )
    )

    assert frame.height == 0
    assert frame.columns == [
        "symbol",
        "timestamp",
        "long_account",
        "short_account",
        "long_short_ratio",
    ]
    client.get_global_long_short_account_ratio.assert_not_called()


def test_download_universe_processes_symbols_sequentially() -> None:
    """Universe download invokes symbol download for each symbol in order."""
    client = MagicMock()
    active = _configure_client(client, kind=_KIND)
    active.side_effect = [
        [_long_short(_START)],
        [_long_short(_START, symbol="ETHUSDT")],
    ]
    repository = MagicMock()
    downloader = LongShortDownloader(
        client,
        repository,
        LongShortDownloadPlanner(chunk_size_ms=_CHUNK),
    )

    _run(
        downloader.download_universe(
            ["BTCUSDT", "ETHUSDT"],
            kind=_KIND,
            period=_PERIOD,
            start_time=_START,
            end_time=_START + 1_000,
        )
    )

    assert repository.save_global_long_short_account_ratio.call_count == 2
    symbols = [
        call.kwargs["symbol"]
        for call in repository.save_global_long_short_account_ratio.call_args_list
    ]
    assert symbols == ["BTCUSDT", "ETHUSDT"]


def test_invalid_payload_and_row_raise_validation_errors() -> None:
    """Malformed payloads and rows fail with structured validation errors."""
    client = MagicMock()
    active = _configure_client(client, kind=_KIND)
    downloader = LongShortDownloader(
        client,
        MagicMock(),
        LongShortDownloadPlanner(chunk_size_ms=_CHUNK),
    )

    active.return_value = {"not": "a list"}
    with pytest.raises(ValidationError) as payload_exc:
        _run(
            downloader.download_symbol(
                symbol=_SYMBOL,
                kind=_KIND,
                period=_PERIOD,
                start_time=_START,
                end_time=_START + 1_000,
            )
        )
    assert payload_exc.value.error_code == "INGESTION-LONG-SHORT-010"

    active.return_value = [["not", "an", "object"]]
    with pytest.raises(ValidationError) as row_exc:
        _run(
            downloader.download_symbol(
                symbol=_SYMBOL,
                kind=_KIND,
                period=_PERIOD,
                start_time=_START,
                end_time=_START + 1_000,
            )
        )
    assert row_exc.value.error_code == "INGESTION-LONG-SHORT-011"

    active.return_value = [{"symbol": _SYMBOL, "timestamp": _START}]
    with pytest.raises(ValidationError) as missing_exc:
        _run(
            downloader.download_symbol(
                symbol=_SYMBOL,
                kind=_KIND,
                period=_PERIOD,
                start_time=_START,
                end_time=_START + 1_000,
            )
        )
    assert missing_exc.value.error_code == "INGESTION-LONG-SHORT-012"


def test_mismatched_symbol_raises_validation_error() -> None:
    """Payload symbols that disagree with the request fail validation."""
    client = MagicMock()
    active = _configure_client(client, kind=_KIND)
    active.return_value = [_long_short(_START, symbol="ETHUSDT")]
    downloader = LongShortDownloader(
        client,
        MagicMock(),
        LongShortDownloadPlanner(chunk_size_ms=_CHUNK),
    )

    with pytest.raises(ValidationError) as exc_info:
        _run(
            downloader.download_symbol(
                symbol=_SYMBOL,
                kind=_KIND,
                period=_PERIOD,
                start_time=_START,
                end_time=_START + 1_000,
            )
        )
    assert exc_info.value.error_code == "INGESTION-LONG-SHORT-013"


def test_persisted_frame_schema_matches_canonical_columns() -> None:
    """Persisted frames use the canonical raw long/short column set."""
    client = MagicMock()
    active = _configure_client(client, kind=_KIND)
    active.return_value = [
        _long_short(
            _START,
            long_account="0.6000",
            short_account="0.4000",
            long_short_ratio="1.5000",
        )
    ]
    repository = MagicMock()
    downloader = LongShortDownloader(
        client,
        repository,
        LongShortDownloadPlanner(chunk_size_ms=_CHUNK),
    )

    _run(
        downloader.download_symbol(
            symbol=_SYMBOL,
            kind=_KIND,
            period=_PERIOD,
            start_time=_START,
            end_time=_START + 1_000,
        )
    )

    frame = repository.save_global_long_short_account_ratio.call_args.args[0]
    expected = pl.DataFrame(
        {
            "symbol": [_SYMBOL],
            "timestamp": [_START],
            "long_account": [0.6],
            "short_account": [0.4],
            "long_short_ratio": [1.5],
        }
    )
    assert_frame_equal(frame, expected)


def test_fetch_symbol_does_not_persist() -> None:
    """fetch_symbol returns data without writing repository partitions."""
    client = MagicMock()
    _configure_client(client, kind=_KIND)
    repository = MagicMock()
    downloader = LongShortDownloader(
        client,
        repository,
        LongShortDownloadPlanner(chunk_size_ms=_CHUNK),
    )

    frame = _run(
        downloader.fetch_symbol(
            symbol=_SYMBOL,
            kind=_KIND,
            period=_PERIOD,
            start_time=_START,
            end_time=_START + 1_000,
        )
    )

    assert frame.height == 1
    repository.save_global_long_short_account_ratio.assert_not_called()


def test_pagination_stops_when_cursor_does_not_advance() -> None:
    """Stuck pagination cursors terminate instead of looping forever."""
    client = MagicMock()
    active = _configure_client(client, kind=_KIND)
    active.return_value = [_long_short(_START), _long_short(_START)]
    repository = MagicMock()
    downloader = LongShortDownloader(
        client,
        repository,
        LongShortDownloadPlanner(chunk_size_ms=_CHUNK),
        request_limit=2,
    )

    _run(
        downloader.download_symbol(
            symbol=_SYMBOL,
            kind=_KIND,
            period=_PERIOD,
            start_time=_START,
            end_time=_START + _FIVE_MINUTES,
        )
    )

    assert active.await_count == 2
    frame = repository.save_global_long_short_account_ratio.call_args.args[0]
    assert frame.height == 4
