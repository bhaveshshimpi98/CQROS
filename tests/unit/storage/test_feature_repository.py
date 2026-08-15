"""Unit tests for CQROS feature dataset repository."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.core.constants import (
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
    STORAGE_DIR_FEATURES,
    STORAGE_DIR_PROCESSED,
)
from cqros.core.types import FilePath
from cqros.storage import (
    DatasetNotFoundError,
    FeaturePartitionRef,
    FeatureRepository,
    ParquetStore,
    StorageLayout,
)
from cqros.storage.feature_repository import FeatureRepository as FeatureRepositoryDirect

_EXCHANGE = "binance"
_MARKET = "perpetual"
_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1h"
_YEAR = 2026


class _InMemoryDataStore:
    """Minimal ``IDataStore`` stub that records paths and frames."""

    def __init__(self) -> None:
        self.frames: dict[Path, pl.DataFrame] = {}
        self.write_paths: list[Path] = []
        self.read_paths: list[Path] = []
        self.exists_paths: list[Path] = []
        self.delete_paths: list[Path] = []
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
        target = Path(path)
        self.delete_paths.append(target)
        try:
            del self.frames[target]
        except KeyError as exc:
            raise DatasetNotFoundError(
                "Dataset not found",
                error_code="STORAGE-TEST-001",
                details={"path": str(target)},
            ) from exc

    def schema(self, path: FilePath) -> pl.Schema:
        return self.read(path).schema

    def row_count(self, path: FilePath) -> int:
        return self.read(path).height


@pytest.fixture
def sample_frame() -> pl.DataFrame:
    """Return a deterministic sample feature DataFrame."""
    return pl.DataFrame(
        {
            "timestamp_ms": [1_700_000_000_000, 1_700_000_060_000],
            "returns": [0.01, -0.02],
            "rolling_mean": [100.0, 101.0],
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
) -> FeatureRepository:
    """Return a feature repository wired to the test layout and datastore."""
    return FeatureRepository(layout, datastore)


def test_feature_repository_is_exported_from_package() -> None:
    """Package export matches the feature repository module class."""
    assert FeatureRepository is FeatureRepositoryDirect


def test_save_and_load_uses_feature_layout_path(
    repository: FeatureRepository,
    layout: StorageLayout,
    datastore: _InMemoryDataStore,
    sample_frame: pl.DataFrame,
) -> None:
    """Feature save/load uses StorageLayout.feature_path."""
    expected = layout.feature_path(
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    assert STORAGE_DIR_FEATURES in expected.parts
    assert STORAGE_DIR_PROCESSED not in expected.parts

    repository.save(
        sample_frame,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    loaded = repository.load(
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )

    assert datastore.write_paths == [expected]
    assert datastore.read_paths == [expected]
    assert_frame_equal(loaded, sample_frame)


def test_feature_path_partitioning_matches_layout_contract(layout: StorageLayout) -> None:
    """Feature partitions follow exchange/market/symbol/timeframe/year.parquet."""
    path = layout.feature_path(
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    assert path.name == f"{_YEAR}.parquet"
    assert path.parent.name == _TIMEFRAME
    assert path.parent.parent.name == _SYMBOL
    assert path.parent.parent.parent.name == _MARKET
    assert path.parent.parent.parent.parent.name == _EXCHANGE
    assert path.parent.parent.parent.parent.parent.name == STORAGE_DIR_FEATURES
    assert STORAGE_DIR_FEATURES in path.parts


def test_public_api_does_not_return_filesystem_paths(
    repository: FeatureRepository,
    sample_frame: pl.DataFrame,
) -> None:
    """Save returns None and load returns a DataFrame, never a Path."""
    result = repository.save(
        sample_frame,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    loaded = repository.load(
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
    repository: FeatureRepository,
) -> None:
    """Missing datasets surface the datastore ``DatasetNotFoundError``."""
    with pytest.raises(DatasetNotFoundError):
        repository.load(
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )


def test_exists_false_when_missing(
    repository: FeatureRepository,
    layout: StorageLayout,
    datastore: _InMemoryDataStore,
) -> None:
    """exists returns False and never reads Parquet contents."""
    assert (
        repository.exists(
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )
        is False
    )
    assert datastore.read_paths == []
    assert datastore.exists_paths == [
        layout.feature_path(_EXCHANGE, _MARKET, _SYMBOL, _TIMEFRAME, _YEAR)
    ]


def test_exists_true_when_partition_saved(
    repository: FeatureRepository,
    layout: StorageLayout,
    datastore: _InMemoryDataStore,
    sample_frame: pl.DataFrame,
) -> None:
    """exists returns True after a partition is saved."""
    repository.save(
        sample_frame,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    datastore.read_paths.clear()
    datastore.exists_paths.clear()

    assert (
        repository.exists(
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )
        is True
    )
    assert datastore.read_paths == []
    assert (
        layout.feature_path(
            _EXCHANGE,
            _MARKET,
            _SYMBOL,
            _TIMEFRAME,
            _YEAR,
        )
        in datastore.exists_paths
    )


def test_delete_removes_partition(
    repository: FeatureRepository,
    layout: StorageLayout,
    datastore: _InMemoryDataStore,
    sample_frame: pl.DataFrame,
) -> None:
    """delete removes a saved partition through the datastore."""
    repository.save(
        sample_frame,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    expected = layout.feature_path(
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )

    repository.delete(
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )

    assert datastore.delete_paths == [expected]
    assert (
        repository.exists(
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )
        is False
    )


def test_delete_missing_propagates_not_found(
    repository: FeatureRepository,
) -> None:
    """delete surfaces DatasetNotFoundError when the partition is absent."""
    with pytest.raises(DatasetNotFoundError):
        repository.delete(
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )


def test_list_years_returns_sorted_partition_years(tmp_path: Path) -> None:
    """list_years discovers existing year parquet files in sorted order."""
    base = (
        tmp_path
        / STORAGE_DIR_FEATURES
        / EXCHANGE_BINANCE
        / MARKET_USDT_PERPETUAL
        / "BTCUSDT"
        / "1h"
    )
    base.mkdir(parents=True, exist_ok=True)
    (base / "2023.parquet").write_bytes(b"")
    (base / "2025.parquet").write_bytes(b"")
    (base / "2024.parquet").write_bytes(b"")
    (base / "notes.txt").write_text("ignore", encoding="utf-8")

    repository = FeatureRepository(StorageLayout(tmp_path), ParquetStore())
    years = repository.list_years(
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol="BTCUSDT",
        timeframe="1h",
    )

    assert years == (2023, 2024, 2025)


def test_list_years_empty_when_missing(tmp_path: Path) -> None:
    """list_years returns an empty tuple when no partitions exist."""
    repository = FeatureRepository(StorageLayout(tmp_path), ParquetStore())
    assert (
        repository.list_years(
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            symbol="BTCUSDT",
            timeframe="1h",
        )
        == ()
    )


def test_discover_partitions_finds_year_files(tmp_path: Path) -> None:
    """Discovery walks feature trees without returning filesystem paths."""
    path = (
        tmp_path
        / STORAGE_DIR_FEATURES
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
        / STORAGE_DIR_FEATURES
        / EXCHANGE_BINANCE
        / MARKET_USDT_PERPETUAL
        / "ETHUSDT"
        / "1h"
        / "2023.parquet"
    )
    eth.parent.mkdir(parents=True, exist_ok=True)
    eth.write_bytes(b"")

    repository = FeatureRepository(StorageLayout(tmp_path), ParquetStore())
    partitions = repository.discover_partitions()

    assert partitions == (
        FeaturePartitionRef(symbol="BTCUSDT", timeframe="1h", year=2024),
        FeaturePartitionRef(symbol="ETHUSDT", timeframe="1h", year=2023),
    )
    assert repository.discover_symbols() == ("BTCUSDT", "ETHUSDT")
    assert repository.discover_timeframes(symbol="BTCUSDT") == ("1h",)


def test_discover_partitions_applies_filters(tmp_path: Path) -> None:
    """Discovery filters by symbol and timeframe allowlists."""
    for symbol, timeframe, year in (
        ("BTCUSDT", "1h", 2024),
        ("BTCUSDT", "4h", 2024),
        ("ETHUSDT", "1h", 2024),
    ):
        path = (
            tmp_path
            / STORAGE_DIR_FEATURES
            / EXCHANGE_BINANCE
            / MARKET_USDT_PERPETUAL
            / symbol
            / timeframe
            / f"{year}.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

    repository = FeatureRepository(StorageLayout(tmp_path), ParquetStore())
    partitions = repository.discover_partitions(
        symbols=("BTCUSDT",),
        timeframes=("1h",),
    )

    assert partitions == (FeaturePartitionRef(symbol="BTCUSDT", timeframe="1h", year=2024),)


def test_round_trip_with_parquet_store(
    layout: StorageLayout,
    sample_frame: pl.DataFrame,
) -> None:
    """Feature repository round-trips through a real ``ParquetStore``."""
    repository = FeatureRepository(layout, ParquetStore())
    repository.save(
        sample_frame,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    loaded = repository.load(
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert_frame_equal(loaded, sample_frame)
    assert layout.feature_path(
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    ).is_file()


def test_feature_paths_differ_from_processed_paths(layout: StorageLayout) -> None:
    """Feature and processed partitions resolve to different locations."""
    feature = layout.feature_path(_EXCHANGE, _MARKET, _SYMBOL, _TIMEFRAME, _YEAR)
    processed = layout.processed_ohlcv_path(
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    assert feature != processed
    assert STORAGE_DIR_FEATURES in feature.parts
    assert STORAGE_DIR_PROCESSED in processed.parts
