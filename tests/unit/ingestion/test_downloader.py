"""Unit tests for CQROS historical market-data downloader."""

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
    MILLISECONDS_PER_DAY,
)
from cqros.core.exceptions import ValidationError
from cqros.ingestion import (
    DEFAULT_DOWNLOAD_CHUNK_SIZE_MS,
    DEFAULT_KLINE_REQUEST_LIMIT,
    DownloadPlanner,
    DownloadTask,
    HistoricalDownloader,
)
from cqros.ingestion.downloader import (
    DownloadPlanner as DownloadPlannerDirect,
)
from cqros.ingestion.downloader import (
    DownloadTask as DownloadTaskDirect,
)
from cqros.ingestion.downloader import (
    HistoricalDownloader as HistoricalDownloaderDirect,
)

_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1m"
_START = 1_700_000_000_000
_DAY = MILLISECONDS_PER_DAY


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def _kline(
    open_time: int,
    *,
    close: str = "100.0",
    close_time: int | None = None,
) -> list[Any]:
    """Build a Binance-shaped kline array."""
    return [
        open_time,
        "100.0",
        "101.0",
        "99.0",
        close,
        "10.0",
        close_time if close_time is not None else open_time + 59_999,
        "1000.0",
        42,
        "5.0",
        "500.0",
        "0",
    ]


def test_exports_match_module_symbols() -> None:
    """Package exports match the downloader module classes."""
    assert DownloadTask is DownloadTaskDirect
    assert DownloadPlanner is DownloadPlannerDirect
    assert HistoricalDownloader is HistoricalDownloaderDirect
    assert DEFAULT_DOWNLOAD_CHUNK_SIZE_MS == MILLISECONDS_PER_DAY
    assert DEFAULT_KLINE_REQUEST_LIMIT == 1_500


def test_download_task_is_immutable() -> None:
    """DownloadTask is a frozen slotted dataclass."""
    task = DownloadTask(
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        start_time=_START,
        end_time=_START + 1_000,
    )
    assert is_dataclass(task)
    with pytest.raises(FrozenInstanceError):
        task.symbol = "ETHUSDT"  # type: ignore[misc]


def test_planner_rejects_non_positive_chunk_size() -> None:
    """Planner construction fails fast on invalid chunk sizes."""
    with pytest.raises(ValidationError) as exc_info:
        DownloadPlanner(chunk_size_ms=0)
    assert exc_info.value.error_code == "INGESTION-DOWNLOADER-001"


def test_planner_splits_range_into_contiguous_tasks() -> None:
    """Long ranges are split into contiguous inclusive chunks."""
    planner = DownloadPlanner(chunk_size_ms=_DAY)
    tasks = planner.plan(
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        start_time=_START,
        end_time=_START + (2 * _DAY) + 100,
    )

    assert len(tasks) == 3
    assert all(isinstance(task, DownloadTask) for task in tasks)
    assert tasks[0] == DownloadTask(
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        start_time=_START,
        end_time=_START + _DAY - 1,
    )
    assert tasks[1].start_time == tasks[0].end_time + 1
    assert tasks[2].end_time == _START + (2 * _DAY) + 100
    assert tasks[0].end_time + 1 == tasks[1].start_time
    assert tasks[1].end_time + 1 == tasks[2].start_time


def test_planner_returns_empty_tuple_when_start_after_end() -> None:
    """Inverted ranges produce no tasks."""
    planner = DownloadPlanner(chunk_size_ms=_DAY)
    assert (
        planner.plan(
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            start_time=_START + 1,
            end_time=_START,
        )
        == ()
    )


def test_planner_single_task_when_range_fits_one_chunk() -> None:
    """Ranges within one chunk yield a single task."""
    planner = DownloadPlanner(chunk_size_ms=_DAY)
    tasks = planner.plan(
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        start_time=_START,
        end_time=_START + 1_000,
    )
    assert tasks == (
        DownloadTask(
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            start_time=_START,
            end_time=_START + 1_000,
        ),
    )


def test_downloader_rejects_non_positive_kline_limit() -> None:
    """Downloader construction fails fast on invalid request limits."""
    with pytest.raises(ValidationError) as exc_info:
        HistoricalDownloader(
            client=MagicMock(),
            repository=MagicMock(),
            planner=DownloadPlanner(),
            kline_limit=0,
        )
    assert exc_info.value.error_code == "INGESTION-DOWNLOADER-002"


def test_downloader_rejects_non_positive_workers() -> None:
    """Downloader construction fails fast on invalid workers."""
    with pytest.raises(ValidationError) as exc_info:
        HistoricalDownloader(
            client=MagicMock(),
            repository=MagicMock(),
            planner=DownloadPlanner(),
            workers=0,
        )
    assert exc_info.value.error_code == "INGESTION-DOWNLOADER-011"


def test_downloader_rejects_non_positive_batch_size() -> None:
    """Downloader construction fails fast on invalid batch_size."""
    with pytest.raises(ValidationError) as exc_info:
        HistoricalDownloader(
            client=MagicMock(),
            repository=MagicMock(),
            planner=DownloadPlanner(),
            batch_size=0,
        )
    assert exc_info.value.error_code == "INGESTION-DOWNLOADER-012"


def test_downloader_stores_execution_configuration() -> None:
    """Downloader retains configured workers and batch_size."""
    downloader = HistoricalDownloader(
        client=MagicMock(),
        repository=MagicMock(),
        planner=DownloadPlanner(),
        workers=5,
        batch_size=20,
    )
    assert downloader.workers == 5
    assert downloader.batch_size == 20


def test_download_symbol_fetches_persists_and_avoids_paths() -> None:
    """Symbol download paginates klines and persists year partitions."""
    client = MagicMock()
    client.get_klines = AsyncMock(
        side_effect=[
            [_kline(_START), _kline(_START + 60_000)],
            [],
        ]
    )
    repository = MagicMock()
    planner = DownloadPlanner(chunk_size_ms=_DAY)
    downloader = HistoricalDownloader(
        client,
        repository,
        planner,
        kline_limit=2,
    )

    _run(
        downloader.download_symbol(
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            start_time=_START,
            end_time=_START + 120_000,
        )
    )

    assert client.get_klines.await_count >= 1
    repository.save_ohlcv.assert_called_once()
    args, kwargs = repository.save_ohlcv.call_args
    frame = args[0]
    assert isinstance(frame, pl.DataFrame)
    assert frame.height == 2
    assert kwargs["exchange"] == EXCHANGE_BINANCE
    assert kwargs["market"] == MARKET_USDT_PERPETUAL
    assert kwargs["symbol"] == _SYMBOL
    assert kwargs["timeframe"] == _TIMEFRAME
    assert kwargs["year"] == 2023
    assert "path" not in kwargs


def test_download_symbol_paginates_until_page_under_limit() -> None:
    """Full pages continue pagination; a short page ends the task fetch."""
    client = MagicMock()
    client.get_klines = AsyncMock(
        side_effect=[
            [_kline(_START), _kline(_START + 60_000)],
            [_kline(_START + 120_000)],
        ]
    )
    repository = MagicMock()
    downloader = HistoricalDownloader(
        client,
        repository,
        DownloadPlanner(chunk_size_ms=_DAY),
        kline_limit=2,
    )

    _run(
        downloader.download_symbol(
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            start_time=_START,
            end_time=_START + 180_000,
        )
    )

    assert client.get_klines.await_count == 2
    frame = repository.save_ohlcv.call_args.args[0]
    assert frame.height == 3


def test_download_symbol_skips_persist_when_empty() -> None:
    """Empty exchange responses do not write partitions."""
    client = MagicMock()
    client.get_klines = AsyncMock(return_value=[])
    repository = MagicMock()
    downloader = HistoricalDownloader(
        client,
        repository,
        DownloadPlanner(chunk_size_ms=_DAY),
    )

    _run(
        downloader.download_symbol(
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            start_time=_START,
            end_time=_START + 1_000,
        )
    )

    repository.save_ohlcv.assert_not_called()


def test_download_universe_processes_symbols_sequentially() -> None:
    """Universe download invokes symbol download for each symbol in order."""
    client = MagicMock()
    client.get_klines = AsyncMock(
        side_effect=[
            [_kline(_START)],
            [_kline(_START)],
        ]
    )
    repository = MagicMock()
    downloader = HistoricalDownloader(
        client,
        repository,
        DownloadPlanner(chunk_size_ms=_DAY),
    )

    _run(
        downloader.download_universe(
            ["BTCUSDT", "ETHUSDT"],
            timeframe=_TIMEFRAME,
            start_time=_START,
            end_time=_START + 1_000,
        )
    )

    assert repository.save_ohlcv.call_count == 2
    symbols = [call.kwargs["symbol"] for call in repository.save_ohlcv.call_args_list]
    assert symbols == ["BTCUSDT", "ETHUSDT"]


def test_invalid_klines_payload_raises_validation_error() -> None:
    """Malformed kline payloads fail with a validation error."""
    client = MagicMock()
    client.get_klines = AsyncMock(return_value={"not": "a list"})
    downloader = HistoricalDownloader(
        client,
        MagicMock(),
        DownloadPlanner(chunk_size_ms=_DAY),
    )

    with pytest.raises(ValidationError) as exc_info:
        _run(
            downloader.download_symbol(
                symbol=_SYMBOL,
                timeframe=_TIMEFRAME,
                start_time=_START,
                end_time=_START + 1_000,
            )
        )
    assert exc_info.value.error_code == "INGESTION-DOWNLOADER-004"


def test_persisted_frame_schema_matches_canonical_columns() -> None:
    """Persisted frames use the canonical raw OHLCV column set."""
    client = MagicMock()
    client.get_klines = AsyncMock(return_value=[_kline(_START, close="123.45")])
    repository = MagicMock()
    downloader = HistoricalDownloader(
        client,
        repository,
        DownloadPlanner(chunk_size_ms=_DAY),
    )

    _run(
        downloader.download_symbol(
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            start_time=_START,
            end_time=_START + 1_000,
        )
    )

    frame = repository.save_ohlcv.call_args.args[0]
    expected = pl.DataFrame(
        {
            "symbol": [_SYMBOL],
            "timeframe": [_TIMEFRAME],
            "open_time": [_START],
            "close_time": [_START + 59_999],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [123.45],
            "volume": [10.0],
            "quote_volume": [1000.0],
            "trade_count": [42],
        }
    )
    assert_frame_equal(frame, expected)
