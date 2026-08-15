"""Unit tests for CQROS incremental market-data updater."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from pathlib import Path
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
from cqros.core.exceptions import DataValidationError, MissingDataError, ValidationError
from cqros.core.types import FilePath
from cqros.ingestion import IncrementalUpdater
from cqros.ingestion.downloader import DownloadPlanner, HistoricalDownloader
from cqros.ingestion.updater import IncrementalUpdater as IncrementalUpdaterDirect
from cqros.ingestion.validator import (
    MarketDataValidator,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)
from cqros.storage import DatasetNotFoundError, MarketDataRepository, StorageLayout

_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1m"
_START = 1_700_000_000_000
_DAY = MILLISECONDS_PER_DAY


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def _ohlcv_row(
    open_time: int,
    *,
    close: float = 100.0,
) -> dict[str, object]:
    """Build a canonical OHLCV row mapping."""
    return {
        "symbol": _SYMBOL,
        "timeframe": _TIMEFRAME,
        "open_time": open_time,
        "close_time": open_time + 59_999,
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": close,
        "volume": 10.0,
        "quote_volume": 1000.0,
        "trade_count": 42,
    }


def _ohlcv_frame(*open_times: int) -> pl.DataFrame:
    """Build an OHLCV DataFrame for the given open times."""
    return pl.DataFrame([_ohlcv_row(open_time) for open_time in open_times])


class _InMemoryDataStore:
    """Minimal ``IDataStore`` stub that records paths and frames."""

    def __init__(self) -> None:
        self.frames: dict[Path, pl.DataFrame] = {}
        self.write_paths: list[Path] = []
        self.read_paths: list[Path] = []

    def write(self, path: FilePath, dataframe: pl.DataFrame) -> None:
        target = Path(path)
        self.write_paths.append(target)
        self.frames[target] = dataframe

    def read(self, path: FilePath) -> pl.DataFrame:
        target = Path(path)
        self.read_paths.append(target)
        try:
            return self.frames[target]
        except KeyError as exc:
            raise DatasetNotFoundError(
                "Dataset not found",
                error_code="STORAGE-TEST-001",
                details={"path": str(target)},
            ) from exc

    def scan(self, path: FilePath) -> pl.LazyFrame:
        return self.read(path).lazy()

    def exists(self, path: FilePath) -> bool:
        return Path(path) in self.frames

    def delete(self, path: FilePath) -> None:
        del self.frames[Path(path)]

    def schema(self, path: FilePath) -> pl.Schema:
        return self.read(path).schema

    def row_count(self, path: FilePath) -> int:
        return self.read(path).height


@pytest.fixture
def layout(tmp_path: Path) -> StorageLayout:
    """Return a layout rooted at a temporary directory."""
    return StorageLayout(tmp_path)


@pytest.fixture
def datastore() -> _InMemoryDataStore:
    """Return an in-memory datastore stub."""
    return _InMemoryDataStore()


@pytest.fixture
def repository(
    layout: StorageLayout,
    datastore: _InMemoryDataStore,
) -> MarketDataRepository:
    """Return a repository wired to the test layout and datastore."""
    return MarketDataRepository(layout, datastore)


def test_exports_match_module_symbol() -> None:
    """Package export matches the updater module class."""
    assert IncrementalUpdater is IncrementalUpdaterDirect


def test_update_symbol_raises_when_no_stored_data(
    repository: MarketDataRepository,
) -> None:
    """Incremental update requires seeded historical storage."""
    client = MagicMock()
    client.open = AsyncMock()
    downloader = MagicMock()
    downloader.fetch_symbol = AsyncMock()
    updater = IncrementalUpdater(
        client,
        repository,
        MarketDataValidator(),
        downloader,
    )

    with pytest.raises(MissingDataError) as exc_info:
        _run(
            updater.update_symbol(
                symbol=_SYMBOL,
                timeframe=_TIMEFRAME,
                end_time=_START + 120_000,
            )
        )

    assert exc_info.value.error_code == "INGESTION-UPDATER-001"
    downloader.fetch_symbol.assert_not_awaited()


def test_update_symbol_downloads_only_missing_range_and_merges(
    repository: MarketDataRepository,
    layout: StorageLayout,
    datastore: _InMemoryDataStore,
) -> None:
    """Updater fetches after the latest open_time and rewrites merged partitions."""
    existing = _ohlcv_frame(_START, _START + 60_000)
    repository.save_ohlcv(
        existing,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=2023,
    )
    datastore.write_paths.clear()

    new_rows = _ohlcv_frame(_START + 120_000, _START + 180_000)
    client = MagicMock()
    client.open = AsyncMock()
    downloader = MagicMock()
    downloader.fetch_symbol = AsyncMock(return_value=new_rows)
    validator = MagicMock()
    validator.validate.return_value = ValidationReport(
        timeframe=_TIMEFRAME,
        row_count=new_rows.height,
        issues=(),
    )
    updater = IncrementalUpdater(client, repository, validator, downloader)

    _run(
        updater.update_symbol(
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            end_time=_START + 180_000,
        )
    )

    client.open.assert_awaited_once()
    downloader.fetch_symbol.assert_awaited_once_with(
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        start_time=_START + 60_000 + 1,
        end_time=_START + 180_000,
    )
    validator.validate.assert_called_once()
    assert len(datastore.write_paths) == 1

    path = layout.raw_ohlcv_path(
        EXCHANGE_BINANCE,
        MARKET_USDT_PERPETUAL,
        _SYMBOL,
        _TIMEFRAME,
        2023,
    )
    merged = datastore.frames[path]
    assert merged.height == 4
    assert merged.get_column("open_time").to_list() == [
        _START,
        _START + 60_000,
        _START + 120_000,
        _START + 180_000,
    ]


def test_update_symbol_noop_when_already_current(
    repository: MarketDataRepository,
) -> None:
    """No download occurs when storage already covers end_time."""
    repository.save_ohlcv(
        _ohlcv_frame(_START + 120_000),
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=2023,
    )
    client = MagicMock()
    client.open = AsyncMock()
    downloader = MagicMock()
    downloader.fetch_symbol = AsyncMock()
    updater = IncrementalUpdater(
        client,
        repository,
        MarketDataValidator(),
        downloader,
    )

    _run(
        updater.update_symbol(
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            end_time=_START + 120_000,
        )
    )

    client.open.assert_not_awaited()
    downloader.fetch_symbol.assert_not_awaited()


def test_update_symbol_deduplicates_overlapping_open_times(
    repository: MarketDataRepository,
    layout: StorageLayout,
    datastore: _InMemoryDataStore,
) -> None:
    """Overlapping open times keep the downloaded row and stay sorted."""
    repository.save_ohlcv(
        _ohlcv_frame(_START),
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=2023,
    )
    overlap = pl.DataFrame([_ohlcv_row(_START, close=200.0), _ohlcv_row(_START + 60_000)])
    client = MagicMock()
    client.open = AsyncMock()
    downloader = MagicMock()
    downloader.fetch_symbol = AsyncMock(return_value=overlap)
    validator = MagicMock()
    validator.validate.return_value = ValidationReport(
        timeframe=_TIMEFRAME,
        row_count=overlap.height,
        issues=(),
    )
    updater = IncrementalUpdater(client, repository, validator, downloader)

    _run(
        updater.update_symbol(
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            end_time=_START + 60_000,
        )
    )

    path = layout.raw_ohlcv_path(
        EXCHANGE_BINANCE,
        MARKET_USDT_PERPETUAL,
        _SYMBOL,
        _TIMEFRAME,
        2023,
    )
    merged = datastore.frames[path]
    assert merged.height == 2
    assert merged.get_column("close").to_list() == [200.0, 100.0]


def test_update_symbol_rewrites_only_affected_year_partitions(
    repository: MarketDataRepository,
    layout: StorageLayout,
    datastore: _InMemoryDataStore,
) -> None:
    """Unaffected year partitions are not rewritten."""
    repository.save_ohlcv(
        _ohlcv_frame(_START),
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=2023,
    )
    # Year-boundary candle in 2024.
    new_open = 1_704_067_200_000
    new_rows = _ohlcv_frame(new_open)
    client = MagicMock()
    client.open = AsyncMock()
    downloader = MagicMock()
    downloader.fetch_symbol = AsyncMock(return_value=new_rows)
    validator = MagicMock()
    validator.validate.return_value = ValidationReport(
        timeframe=_TIMEFRAME,
        row_count=1,
        issues=(),
    )
    updater = IncrementalUpdater(client, repository, validator, downloader)
    datastore.write_paths.clear()

    _run(
        updater.update_symbol(
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            end_time=new_open,
        )
    )

    written_years = [path.stem for path in datastore.write_paths]
    assert written_years == ["2024"]
    assert (
        layout.raw_ohlcv_path(
            EXCHANGE_BINANCE,
            MARKET_USDT_PERPETUAL,
            _SYMBOL,
            _TIMEFRAME,
            2023,
        )
        in datastore.frames
    )
    assert_frame_equal(
        datastore.frames[
            layout.raw_ohlcv_path(
                EXCHANGE_BINANCE,
                MARKET_USDT_PERPETUAL,
                _SYMBOL,
                _TIMEFRAME,
                2023,
            )
        ],
        _ohlcv_frame(_START),
    )


def test_update_symbol_rejects_invalid_download(
    repository: MarketDataRepository,
) -> None:
    """Validation failures abort before any merge rewrite."""
    repository.save_ohlcv(
        _ohlcv_frame(_START),
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=2023,
    )
    client = MagicMock()
    client.open = AsyncMock()
    downloader = MagicMock()
    downloader.fetch_symbol = AsyncMock(return_value=_ohlcv_frame(_START + 60_000))
    validator = MagicMock()
    validator.validate.return_value = ValidationReport(
        timeframe=_TIMEFRAME,
        row_count=1,
        issues=(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                check="schema",
                message="broken",
            ),
        ),
    )
    updater = IncrementalUpdater(client, repository, validator, downloader)

    with pytest.raises(DataValidationError) as exc_info:
        _run(
            updater.update_symbol(
                symbol=_SYMBOL,
                timeframe=_TIMEFRAME,
                end_time=_START + 60_000,
            )
        )

    assert exc_info.value.error_code == "INGESTION-UPDATER-003"


def test_update_symbol_rejects_non_int_end_time(
    repository: MarketDataRepository,
) -> None:
    """end_time must be an int Unix millisecond timestamp."""
    updater = IncrementalUpdater(
        MagicMock(),
        repository,
        MarketDataValidator(),
        MagicMock(),
    )
    with pytest.raises(ValidationError) as exc_info:
        _run(
            updater.update_symbol(
                symbol=_SYMBOL,
                timeframe=_TIMEFRAME,
                end_time="1700000000000",  # type: ignore[arg-type]
            )
        )
    assert exc_info.value.error_code == "INGESTION-UPDATER-002"


def test_update_universe_processes_symbols_sequentially(
    repository: MarketDataRepository,
) -> None:
    """Universe update invokes symbol update for each symbol in order."""
    for symbol in ("BTCUSDT", "ETHUSDT"):
        frame = pl.DataFrame(
            [
                {
                    **_ohlcv_row(_START),
                    "symbol": symbol,
                }
            ]
        )
        repository.save_ohlcv(
            frame,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            symbol=symbol,
            timeframe=_TIMEFRAME,
            year=2023,
        )

    client = MagicMock()
    client.open = AsyncMock()
    downloader = MagicMock()
    downloader.fetch_symbol = AsyncMock(
        side_effect=[
            pl.DataFrame([{**_ohlcv_row(_START + 60_000), "symbol": "BTCUSDT"}]),
            pl.DataFrame([{**_ohlcv_row(_START + 60_000), "symbol": "ETHUSDT"}]),
        ]
    )
    validator = MagicMock()
    validator.validate.side_effect = [
        ValidationReport(timeframe=_TIMEFRAME, row_count=1, issues=()),
        ValidationReport(timeframe=_TIMEFRAME, row_count=1, issues=()),
    ]
    updater = IncrementalUpdater(client, repository, validator, downloader)

    _run(
        updater.update_universe(
            ["BTCUSDT", "ETHUSDT"],
            timeframe=_TIMEFRAME,
            end_time=_START + 60_000,
        )
    )

    symbols = [call.kwargs["symbol"] for call in downloader.fetch_symbol.await_args_list]
    assert symbols == ["BTCUSDT", "ETHUSDT"]


def test_update_symbol_with_live_downloader_does_not_redownload_history(
    repository: MarketDataRepository,
) -> None:
    """Live downloader fetch starts after the latest stored candle only."""
    aligned = 1_700_000_040_000
    repository.save_ohlcv(
        _ohlcv_frame(aligned),
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=2023,
    )
    client = MagicMock()
    client.open = AsyncMock()
    client.get_klines = AsyncMock(
        return_value=[
            [
                aligned + 60_000,
                "100.0",
                "101.0",
                "99.0",
                "100.0",
                "10.0",
                aligned + 119_999,
                "1000.0",
                42,
                "5.0",
                "500.0",
                "0",
            ]
        ]
    )
    downloader = HistoricalDownloader(
        client,
        repository,
        DownloadPlanner(chunk_size_ms=_DAY),
    )
    updater = IncrementalUpdater(
        client,
        repository,
        MarketDataValidator(),
        downloader,
    )

    _run(
        updater.update_symbol(
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            end_time=aligned + 60_000,
        )
    )

    assert client.get_klines.await_count == 1
    kwargs: dict[str, Any] = client.get_klines.await_args.kwargs
    assert kwargs["start_time"] == aligned + 1
    assert kwargs["end_time"] == aligned + 60_000


def test_update_symbol_requests_only_newest_candles(
    repository: MarketDataRepository,
) -> None:
    """Fetch window starts at latest stored open_time + 1, never before it."""
    historical = (_START, _START + 60_000, _START + 120_000)
    repository.save_ohlcv(
        _ohlcv_frame(*historical),
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=2023,
    )
    latest = historical[-1]
    end_time = latest + 180_000
    new_rows = _ohlcv_frame(latest + 60_000, latest + 120_000)

    client = MagicMock()
    client.open = AsyncMock()
    downloader = MagicMock()
    downloader.fetch_symbol = AsyncMock(return_value=new_rows)
    validator = MagicMock()
    validator.validate.return_value = ValidationReport(
        timeframe=_TIMEFRAME,
        row_count=new_rows.height,
        issues=(),
    )
    updater = IncrementalUpdater(client, repository, validator, downloader)

    _run(
        updater.update_symbol(
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            end_time=end_time,
        )
    )

    downloader.fetch_symbol.assert_awaited_once_with(
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        start_time=latest + 1,
        end_time=end_time,
    )
    fetch_kwargs = downloader.fetch_symbol.await_args.kwargs
    assert fetch_kwargs["start_time"] > latest
    assert fetch_kwargs["start_time"] == latest + 1


def test_update_symbol_never_redownloads_historical_candles(
    repository: MarketDataRepository,
) -> None:
    """Completed history open times are excluded from the download request."""
    history_start = _START
    history_end = _START + (5 * 60_000)
    historical_times = tuple(history_start + (i * 60_000) for i in range(6))
    repository.save_ohlcv(
        _ohlcv_frame(*historical_times),
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=2023,
    )

    client = MagicMock()
    client.open = AsyncMock()
    downloader = MagicMock()
    downloader.fetch_symbol = AsyncMock(return_value=_ohlcv_frame(history_end + 60_000))
    validator = MagicMock()
    validator.validate.return_value = ValidationReport(
        timeframe=_TIMEFRAME,
        row_count=1,
        issues=(),
    )
    updater = IncrementalUpdater(client, repository, validator, downloader)

    _run(
        updater.update_symbol(
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            end_time=history_end + 60_000,
        )
    )

    fetch_kwargs = downloader.fetch_symbol.await_args.kwargs
    assert fetch_kwargs["start_time"] == history_end + 1
    assert fetch_kwargs["start_time"] > history_start
    for open_time in historical_times:
        assert fetch_kwargs["start_time"] > open_time
    assert downloader.fetch_symbol.await_count == 1


def test_update_symbol_multiple_reruns_request_zero_completed_history(
    repository: MarketDataRepository,
) -> None:
    """Reruns after a successful update never re-request completed history."""
    seed_latest = _START + 60_000
    repository.save_ohlcv(
        _ohlcv_frame(_START, seed_latest),
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=2023,
    )

    first_new = _START + 120_000
    second_new = _START + 180_000
    fetch_returns = [
        _ohlcv_frame(first_new),
        _ohlcv_frame(second_new),
    ]
    client = MagicMock()
    client.open = AsyncMock()
    downloader = MagicMock()
    downloader.fetch_symbol = AsyncMock(side_effect=fetch_returns)
    validator = MagicMock()
    validator.validate.side_effect = [
        ValidationReport(timeframe=_TIMEFRAME, row_count=1, issues=()),
        ValidationReport(timeframe=_TIMEFRAME, row_count=1, issues=()),
    ]
    updater = IncrementalUpdater(client, repository, validator, downloader)

    _run(
        updater.update_symbol(
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            end_time=first_new,
        )
    )
    _run(
        updater.update_symbol(
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            end_time=second_new,
        )
    )
    # Third run: storage already covers second_new; no fetch at all.
    _run(
        updater.update_symbol(
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            end_time=second_new,
        )
    )

    assert downloader.fetch_symbol.await_count == 2
    first_call = downloader.fetch_symbol.await_args_list[0].kwargs
    second_call = downloader.fetch_symbol.await_args_list[1].kwargs

    assert first_call["start_time"] == seed_latest + 1
    assert first_call["end_time"] == first_new
    assert second_call["start_time"] == first_new + 1
    assert second_call["end_time"] == second_new

    # Each request starts strictly after every previously completed candle.
    assert first_call["start_time"] > _START
    assert first_call["start_time"] > seed_latest
    assert second_call["start_time"] > _START
    assert second_call["start_time"] > seed_latest
    assert second_call["start_time"] > first_new
