"""Unit tests for CQROS processed market-data repository."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.core.constants import (
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
    STORAGE_DIR_PROCESSED,
)
from cqros.core.types import FilePath
from cqros.storage import (
    DatasetNotFoundError,
    ParquetStore,
    ProcessedMarketDataRepository,
    StorageLayout,
)
from cqros.storage.processed_repository import (
    ProcessedMarketDataRepository as ProcessedMarketDataRepositoryDirect,
)

_EXCHANGE = "binance"
_MARKET = "perpetual"
_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1m"
_YEAR = 2026
_OHLCV_EXCHANGE = EXCHANGE_BINANCE
_OHLCV_MARKET = MARKET_USDT_PERPETUAL


class _InMemoryDataStore:
    """Minimal ``IDataStore`` stub that records paths and frames."""

    def __init__(self) -> None:
        self.frames: dict[Path, pl.DataFrame] = {}
        self.write_paths: list[Path] = []
        self.read_paths: list[Path] = []
        self.exists_paths: list[Path] = []
        self.scan_paths: list[Path] = []

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
        target = Path(path)
        self.scan_paths.append(target)
        return self.read(path).lazy()

    def exists(self, path: FilePath) -> bool:
        target = Path(path)
        self.exists_paths.append(target)
        return target in self.frames

    def delete(self, path: FilePath) -> None:
        del self.frames[Path(path)]

    def schema(self, path: FilePath) -> pl.Schema:
        return self.read(path).schema

    def row_count(self, path: FilePath) -> int:
        return self.read(path).height


@pytest.fixture
def sample_frame() -> pl.DataFrame:
    """Return a deterministic sample DataFrame."""
    return pl.DataFrame(
        {
            "timestamp_ms": [1_700_000_000_000, 1_700_000_060_000],
            "value": [0.01, 0.02],
        }
    )


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
) -> ProcessedMarketDataRepository:
    """Return a processed repository wired to the test layout and datastore."""
    return ProcessedMarketDataRepository(layout, datastore)


def test_processed_market_data_repository_is_exported_from_package() -> None:
    """Package export matches the processed repository module class."""
    assert ProcessedMarketDataRepository is ProcessedMarketDataRepositoryDirect


def test_save_and_load_ohlcv_uses_processed_layout_path(
    repository: ProcessedMarketDataRepository,
    layout: StorageLayout,
    datastore: _InMemoryDataStore,
    sample_frame: pl.DataFrame,
) -> None:
    """Processed OHLCV save/load uses the processed layout path."""
    expected = layout.processed_ohlcv_path(
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    assert STORAGE_DIR_PROCESSED in expected.parts

    repository.save_ohlcv(
        sample_frame,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    loaded = repository.load_ohlcv(
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )

    assert datastore.write_paths == [expected]
    assert datastore.read_paths == [expected]
    assert_frame_equal(loaded, sample_frame)


@pytest.mark.parametrize(
    ("save_name", "load_name", "path_method"),
    [
        ("save_ohlcv", "load_ohlcv", "processed_ohlcv_path"),
        ("save_funding", "load_funding", "processed_funding_path"),
        ("save_open_interest", "load_open_interest", "processed_open_interest_path"),
        ("save_taker_volume", "load_taker_volume", "processed_taker_volume_path"),
        (
            "save_global_long_short_account_ratio",
            "load_global_long_short_account_ratio",
            "processed_global_long_short_account_ratio_path",
        ),
        (
            "save_top_long_short_account_ratio",
            "load_top_long_short_account_ratio",
            "processed_top_long_short_account_ratio_path",
        ),
        (
            "save_top_long_short_position_ratio",
            "load_top_long_short_position_ratio",
            "processed_top_long_short_position_ratio_path",
        ),
    ],
)
def test_each_processed_dataset_type_uses_distinct_processed_path(
    repository: ProcessedMarketDataRepository,
    layout: StorageLayout,
    datastore: _InMemoryDataStore,
    sample_frame: pl.DataFrame,
    save_name: str,
    load_name: str,
    path_method: str,
) -> None:
    """Each processed dataset type resolves a distinct processed layout path."""
    expected = getattr(layout, path_method)(
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    assert STORAGE_DIR_PROCESSED in expected.parts
    save = getattr(repository, save_name)
    load = getattr(repository, load_name)

    save(
        sample_frame,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    loaded = load(
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )

    assert datastore.write_paths[-1] == expected
    assert datastore.read_paths[-1] == expected
    assert_frame_equal(loaded, sample_frame)


def test_processed_paths_differ_from_raw_paths(layout: StorageLayout) -> None:
    """Processed and raw OHLCV partitions resolve to different locations."""
    raw = layout.raw_ohlcv_path(_EXCHANGE, _MARKET, _SYMBOL, _TIMEFRAME, _YEAR)
    processed = layout.processed_ohlcv_path(
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    assert raw != processed
    assert STORAGE_DIR_PROCESSED in processed.parts


def test_public_api_does_not_return_filesystem_paths(
    repository: ProcessedMarketDataRepository,
    sample_frame: pl.DataFrame,
) -> None:
    """Save returns None and load returns a DataFrame, never a Path."""
    result = repository.save_ohlcv(
        sample_frame,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    loaded = repository.load_ohlcv(
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert result is None
    assert isinstance(loaded, pl.DataFrame)
    assert not isinstance(loaded, Path)


def test_load_propagates_datastore_not_found(
    repository: ProcessedMarketDataRepository,
) -> None:
    """Missing datasets surface the datastore ``DatasetNotFoundError``."""
    with pytest.raises(DatasetNotFoundError):
        repository.load_ohlcv(
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )


def test_round_trip_with_parquet_store(
    layout: StorageLayout,
    sample_frame: pl.DataFrame,
) -> None:
    """Processed repository round-trips through a real ``ParquetStore``."""
    repository = ProcessedMarketDataRepository(layout, ParquetStore())
    repository.save_ohlcv(
        sample_frame,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    loaded = repository.load_ohlcv(
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert_frame_equal(loaded, sample_frame)
    assert layout.processed_ohlcv_path(
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    ).is_file()


def test_has_ohlcv_empty_returns_false(
    repository: ProcessedMarketDataRepository,
    datastore: _InMemoryDataStore,
) -> None:
    """has_ohlcv returns False when no yearly processed partitions exist."""
    assert repository.has_ohlcv(_SYMBOL, _TIMEFRAME) is False
    assert datastore.read_paths == []
    assert datastore.exists_paths


def test_has_ohlcv_single_year_returns_true(
    repository: ProcessedMarketDataRepository,
    layout: StorageLayout,
    datastore: _InMemoryDataStore,
    sample_frame: pl.DataFrame,
) -> None:
    """has_ohlcv returns True when a single processed year partition exists."""
    repository.save_ohlcv(
        sample_frame,
        exchange=_OHLCV_EXCHANGE,
        market=_OHLCV_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=2023,
    )
    datastore.read_paths.clear()
    datastore.exists_paths.clear()

    assert repository.has_ohlcv(_SYMBOL, _TIMEFRAME) is True
    assert datastore.read_paths == []
    assert (
        layout.processed_ohlcv_path(
            _OHLCV_EXCHANGE,
            _OHLCV_MARKET,
            _SYMBOL,
            _TIMEFRAME,
            2023,
        )
        in datastore.exists_paths
    )


def test_discover_partitions_finds_year_files(tmp_path: Path) -> None:
    """Discovery walks processed trees without returning filesystem paths."""
    path = (
        tmp_path
        / STORAGE_DIR_PROCESSED
        / "ohlcv"
        / EXCHANGE_BINANCE
        / MARKET_USDT_PERPETUAL
        / "BTCUSDT"
        / "1h"
        / "2024.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    eth = (
        tmp_path
        / STORAGE_DIR_PROCESSED
        / "ohlcv"
        / EXCHANGE_BINANCE
        / MARKET_USDT_PERPETUAL
        / "ETHUSDT"
        / "1h"
        / "2023.parquet"
    )
    eth.parent.mkdir(parents=True, exist_ok=True)
    eth.write_bytes(b"")

    repository = ProcessedMarketDataRepository(StorageLayout(tmp_path), ParquetStore())
    partitions = repository.discover_partitions(datasets=("ohlcv",))

    assert [(p.symbol, p.timeframe, p.year) for p in partitions] == [
        ("BTCUSDT", "1h", 2024),
        ("ETHUSDT", "1h", 2023),
    ]
    assert repository.discover_datasets() == ("ohlcv",)
    assert repository.discover_symbols(dataset="ohlcv") == ("BTCUSDT", "ETHUSDT")
    assert repository.discover_timeframes(dataset="ohlcv", symbol="BTCUSDT") == ("1h",)


def test_discover_partitions_applies_filters(tmp_path: Path) -> None:
    """Discovery filters by dataset, symbol, and timeframe allowlists."""
    for symbol, timeframe, year in (
        ("BTCUSDT", "1h", 2024),
        ("BTCUSDT", "4h", 2024),
        ("ETHUSDT", "1h", 2024),
    ):
        path = (
            tmp_path
            / STORAGE_DIR_PROCESSED
            / "ohlcv"
            / EXCHANGE_BINANCE
            / MARKET_USDT_PERPETUAL
            / symbol
            / timeframe
            / f"{year}.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

    funding = (
        tmp_path
        / STORAGE_DIR_PROCESSED
        / "funding"
        / EXCHANGE_BINANCE
        / MARKET_USDT_PERPETUAL
        / "BTCUSDT"
        / "8h"
        / "2024.parquet"
    )
    funding.parent.mkdir(parents=True, exist_ok=True)
    funding.write_bytes(b"")

    repository = ProcessedMarketDataRepository(StorageLayout(tmp_path), ParquetStore())
    partitions = repository.discover_partitions(
        datasets=("ohlcv",),
        symbols=("BTCUSDT",),
        timeframes=("1h",),
    )

    assert len(partitions) == 1
    assert partitions[0].dataset == "ohlcv"
    assert partitions[0].symbol == "BTCUSDT"
    assert partitions[0].timeframe == "1h"
