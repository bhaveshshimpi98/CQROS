"""Unit tests for CQROS market-dataset repair engine."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import polars as pl
import pytest

from cqros.core.constants import (
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
)
from cqros.core.exceptions import DataValidationError, ValidationError
from cqros.core.types import FilePath
from cqros.ingestion import (
    DatasetRepairEngine,
    RepairIssue,
    RepairReport,
    RepairSeverity,
)
from cqros.ingestion.manifest import ManifestRepository
from cqros.ingestion.repair import DatasetRepairEngine as DatasetRepairEngineDirect
from cqros.ingestion.validator import (
    MarketDataValidator,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)
from cqros.storage import DatasetNotFoundError, MarketDataRepository, StorageLayout

_SYMBOL = "BTCUSDT"
_SYMBOL_ETH = "ETHUSDT"
_TIMEFRAME = "1m"
# Aligned to the 1-minute grid (Unix ms divisible by 60_000).
_START = 1_699_999_980_000
_INTERVAL = 60_000


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def _ohlcv_row(
    open_time: int,
    *,
    symbol: str = _SYMBOL,
    close: float = 100.0,
) -> dict[str, object]:
    """Build a canonical OHLCV row mapping."""
    return {
        "symbol": symbol,
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


def _ohlcv_frame(*open_times: int, symbol: str = _SYMBOL) -> pl.DataFrame:
    """Build an OHLCV DataFrame for the given open times."""
    return pl.DataFrame([_ohlcv_row(open_time, symbol=symbol) for open_time in open_times])


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


def _manifest_repository_for(
    layout: StorageLayout,
    *,
    symbol: str = _SYMBOL,
    timeframe: str = _TIMEFRAME,
) -> ManifestRepository:
    """Build a manifest repository for an OHLCV dataset directory."""
    dataset_dir = layout.raw_ohlcv_path(
        EXCHANGE_BINANCE,
        MARKET_USDT_PERPETUAL,
        symbol,
        timeframe,
        year=2019,
    ).parent
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return ManifestRepository(dataset_dir)


def _engine(
    repository: MarketDataRepository,
    layout: StorageLayout,
    *,
    downloader: Any | None = None,
    validator: Any | None = None,
    symbol: str = _SYMBOL,
) -> tuple[DatasetRepairEngine, Any]:
    """Build a repair engine with mocked downloader by default."""
    if downloader is None:
        downloader = MagicMock()
        downloader.fetch_symbol = AsyncMock(return_value=pl.DataFrame())
    if validator is None:
        validator = MarketDataValidator()
    manifest_repository = _manifest_repository_for(layout, symbol=symbol)
    engine = DatasetRepairEngine(
        repository,
        downloader,
        validator,
        manifest_repository,
    )
    return engine, downloader


def test_repair_types_are_exported_and_frozen() -> None:
    """Package exports match module symbols and report types are frozen."""
    assert DatasetRepairEngine is DatasetRepairEngineDirect
    issue = RepairIssue(severity=RepairSeverity.ERROR, check="x", message="y")
    report = RepairReport(
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        start_time_ms=_START,
        end_time_ms=_START + _INTERVAL,
        issues=(issue,),
        repaired_ranges=(),
        rewritten_years=(),
        downloaded_rows=0,
        manifest_updated=False,
    )
    assert is_dataclass(issue)
    assert is_dataclass(report)
    assert report.has_errors is True
    assert report.errors() == (issue,)
    with pytest.raises(FrozenInstanceError):
        issue.message = "changed"  # type: ignore[misc]


def test_repair_symbol_rejects_inverted_time_range(
    repository: MarketDataRepository,
    layout: StorageLayout,
) -> None:
    """repair_symbol rejects start_time greater than end_time."""
    engine, downloader = _engine(repository, layout)

    with pytest.raises(ValidationError) as exc_info:
        _run(
            engine.repair_symbol(
                symbol=_SYMBOL,
                timeframe=_TIMEFRAME,
                start_time=_START + 1,
                end_time=_START,
            )
        )

    assert exc_info.value.error_code == "INGESTION-REPAIR-001"
    downloader.fetch_symbol.assert_not_awaited()


def test_repair_symbol_downloads_missing_partition(
    repository: MarketDataRepository,
    layout: StorageLayout,
    datastore: _InMemoryDataStore,
) -> None:
    """Missing year partitions are downloaded and rewritten."""
    repaired = _ohlcv_frame(_START, _START + _INTERVAL)
    downloader = MagicMock()
    downloader.fetch_symbol = AsyncMock(return_value=repaired)
    validator = MagicMock()
    validator.validate.return_value = ValidationReport(
        timeframe=_TIMEFRAME,
        row_count=repaired.height,
        issues=(),
    )
    engine, _ = _engine(repository, layout, downloader=downloader, validator=validator)

    report = _run(
        engine.repair_symbol(
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            start_time=_START,
            end_time=_START + _INTERVAL,
        )
    )

    assert report.has_errors is True
    assert any(issue.check == "missing_partition" for issue in report.issues)
    assert report.repaired_ranges == ((_START, _START + _INTERVAL),)
    assert report.rewritten_years == (2023,)
    assert report.downloaded_rows == 2
    assert report.manifest_updated is True
    downloader.fetch_symbol.assert_awaited_once_with(
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        start_time=_START,
        end_time=_START + _INTERVAL,
    )
    path = layout.raw_ohlcv_path(
        EXCHANGE_BINANCE,
        MARKET_USDT_PERPETUAL,
        _SYMBOL,
        _TIMEFRAME,
        2023,
    )
    assert path in datastore.frames
    assert datastore.frames[path].height == 2


def test_repair_symbol_downloads_only_coverage_gaps(
    repository: MarketDataRepository,
    layout: StorageLayout,
) -> None:
    """Trailing coverage holes download only the missing forward range."""
    existing = _ohlcv_frame(_START, _START + _INTERVAL)
    repository.save_ohlcv(
        existing,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=2023,
    )
    gap_rows = _ohlcv_frame(_START + 2 * _INTERVAL, _START + 3 * _INTERVAL)
    downloader = MagicMock()
    downloader.fetch_symbol = AsyncMock(return_value=gap_rows)
    engine, _ = _engine(repository, layout, downloader=downloader)

    report = _run(
        engine.repair_symbol(
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            start_time=_START,
            end_time=_START + 3 * _INTERVAL,
        )
    )

    assert any(issue.check == "coverage_gap" for issue in report.errors())
    assert report.repaired_ranges == ((_START + 2 * _INTERVAL, _START + 3 * _INTERVAL),)
    downloader.fetch_symbol.assert_awaited_once_with(
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        start_time=_START + 2 * _INTERVAL,
        end_time=_START + 3 * _INTERVAL,
    )
    loaded = repository.load_ohlcv(
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=2023,
    )
    assert loaded.get_column("open_time").to_list() == [
        _START,
        _START + _INTERVAL,
        _START + 2 * _INTERVAL,
        _START + 3 * _INTERVAL,
    ]


def test_repair_symbol_downloads_only_internal_timestamp_gaps(
    repository: MarketDataRepository,
    layout: StorageLayout,
) -> None:
    """Validator gap-only failures redownload solely the missing bars."""
    existing = _ohlcv_frame(_START, _START + 3 * _INTERVAL)
    repository.save_ohlcv(
        existing,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=2023,
    )
    gap_rows = _ohlcv_frame(_START + _INTERVAL, _START + 2 * _INTERVAL)
    downloader = MagicMock()
    downloader.fetch_symbol = AsyncMock(return_value=gap_rows)
    engine, _ = _engine(repository, layout, downloader=downloader)

    report = _run(
        engine.repair_symbol(
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            start_time=_START,
            end_time=_START + 3 * _INTERVAL,
        )
    )

    assert any(issue.check == "corrupted_dataset" for issue in report.errors())
    assert report.repaired_ranges == ((_START + _INTERVAL, _START + 3 * _INTERVAL - 1),)
    downloader.fetch_symbol.assert_awaited_once_with(
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        start_time=_START + _INTERVAL,
        end_time=_START + 3 * _INTERVAL - 1,
    )


def test_repair_symbol_redownloads_fully_corrupted_partition(
    repository: MarketDataRepository,
    layout: StorageLayout,
) -> None:
    """Non-gap validation failures redownload the full expected year range."""
    bad = _ohlcv_frame(_START)
    bad = bad.with_columns(pl.lit(-1.0).alias("volume"))  # pyright: ignore[reportUnknownMemberType]
    repository.save_ohlcv(
        bad,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=2023,
    )
    repaired = _ohlcv_frame(_START, _START + _INTERVAL)
    downloader = MagicMock()
    downloader.fetch_symbol = AsyncMock(return_value=repaired)
    engine, _ = _engine(repository, layout, downloader=downloader)

    report = _run(
        engine.repair_symbol(
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            start_time=_START,
            end_time=_START + _INTERVAL,
        )
    )

    assert any(issue.check == "corrupted_dataset" for issue in report.errors())
    assert report.repaired_ranges == ((_START, _START + _INTERVAL),)
    downloader.fetch_symbol.assert_awaited_once()


def test_repair_symbol_noop_when_healthy(
    repository: MarketDataRepository,
    layout: StorageLayout,
) -> None:
    """Healthy continuous coverage does not download data."""
    existing = _ohlcv_frame(_START, _START + _INTERVAL, _START + 2 * _INTERVAL)
    repository.save_ohlcv(
        existing,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=2023,
    )
    engine, downloader = _engine(repository, layout)

    report = _run(
        engine.repair_symbol(
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            start_time=_START,
            end_time=_START + 2 * _INTERVAL,
        )
    )

    downloader.fetch_symbol.assert_not_awaited()
    assert report.repaired_ranges == ()
    assert report.rewritten_years == ()
    assert report.downloaded_rows == 0
    assert report.manifest_updated is True
    assert any(issue.check == "missing_manifest" for issue in report.warnings())


def test_repair_symbol_rebuilds_invalid_manifest(
    repository: MarketDataRepository,
    layout: StorageLayout,
) -> None:
    """Corrupt manifest JSON is replaced after a healthy partition scan."""
    existing = _ohlcv_frame(_START, _START + _INTERVAL)
    repository.save_ohlcv(
        existing,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=2023,
    )
    manifest_repository = _manifest_repository_for(layout)
    manifest_repository.path.write_text("{not-json", encoding="utf-8")
    downloader = MagicMock()
    downloader.fetch_symbol = AsyncMock()
    engine = DatasetRepairEngine(
        repository,
        downloader,
        MarketDataValidator(),
        manifest_repository,
    )

    report = _run(
        engine.repair_symbol(
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            start_time=_START,
            end_time=_START + _INTERVAL,
        )
    )

    assert any(issue.check == "invalid_manifest" for issue in report.errors())
    assert report.manifest_updated is True
    loaded = manifest_repository.load()
    assert loaded.years == (2023,)
    assert loaded.total_rows == 2
    downloader.fetch_symbol.assert_not_awaited()


def test_repair_symbol_raises_when_download_fails_validation(
    repository: MarketDataRepository,
    layout: StorageLayout,
) -> None:
    """Invalid downloaded repair frames abort before rewrite."""
    downloader = MagicMock()
    downloader.fetch_symbol = AsyncMock(return_value=_ohlcv_frame(_START))
    validator = MagicMock()
    validator.validate.return_value = ValidationReport(
        timeframe=_TIMEFRAME,
        row_count=1,
        issues=(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                check="nulls",
                message="bad",
            ),
        ),
    )
    engine, _ = _engine(repository, layout, downloader=downloader, validator=validator)

    with pytest.raises(DataValidationError) as exc_info:
        _run(
            engine.repair_symbol(
                symbol=_SYMBOL,
                timeframe=_TIMEFRAME,
                start_time=_START,
                end_time=_START,
            )
        )

    assert exc_info.value.error_code == "INGESTION-REPAIR-004"


def test_repair_universe_processes_symbols_sequentially(
    repository: MarketDataRepository,
    layout: StorageLayout,
    datastore: _InMemoryDataStore,
) -> None:
    """Universe repair returns one report per symbol and fetches each gap."""
    btc = _ohlcv_frame(_START)
    eth = _ohlcv_frame(_START, symbol=_SYMBOL_ETH)
    downloader = MagicMock()
    downloader.fetch_symbol = AsyncMock(side_effect=[btc, eth])
    validator = MagicMock()
    validator.validate.return_value = ValidationReport(
        timeframe=_TIMEFRAME,
        row_count=1,
        issues=(),
    )
    # Seed template manifest repo under BTC path; ETH uses a sibling dir.
    engine, _ = _engine(repository, layout, downloader=downloader, validator=validator)

    reports = _run(
        engine.repair_universe(
            [_SYMBOL, _SYMBOL_ETH],
            timeframe=_TIMEFRAME,
            start_time=_START,
            end_time=_START,
        )
    )

    assert len(reports) == 2
    assert reports[0].symbol == _SYMBOL
    assert reports[1].symbol == _SYMBOL_ETH
    assert downloader.fetch_symbol.await_count == 2
    assert len(datastore.write_paths) == 2
