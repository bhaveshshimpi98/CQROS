"""Unit tests for CQROS signal dataset repository."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.core.constants import (
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
    STORAGE_DIR_FEATURES,
    STORAGE_DIR_LABELS,
    STORAGE_DIR_SIGNALS,
    STORAGE_DIR_TRAINING,
)
from cqros.core.types import FilePath
from cqros.storage import (
    DatasetNotFoundError,
    ParquetStore,
    SignalPartitionRef,
    SignalRepository,
    StorageLayout,
)
from cqros.storage.signal_repository import SignalRepository as SignalRepositoryDirect

_EXCHANGE = "binance"
_MARKET = "perpetual"
_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1h"
_YEAR = 2026
_MODEL_NAME = "alpha-lgbm"
_MODEL_VERSION = "1.0.0"

_CANONICAL_COLUMNS = (
    "symbol",
    "timeframe",
    "open_time",
    "model_name",
    "model_version",
    "signal",
)

_CANONICAL_DTYPES = {
    "symbol": pl.String,
    "timeframe": pl.String,
    "open_time": pl.Int64,
    "model_name": pl.String,
    "model_version": pl.String,
    "signal": pl.String,
}


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
    """Return a deterministic canonical sample signal DataFrame."""
    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "timeframe": ["1h", "1h"],
            "open_time": [1_700_000_000_000, 1_700_000_060_000],
            "model_name": [_MODEL_NAME, _MODEL_NAME],
            "model_version": [_MODEL_VERSION, _MODEL_VERSION],
            "signal": ["BUY", "SELL"],
        },
        schema=_CANONICAL_DTYPES,
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
) -> SignalRepository:
    """Return a signal repository wired to the test layout and datastore."""
    return SignalRepository(layout, datastore)


def _signal_path(layout: StorageLayout) -> Path:
    """Compose the canonical sample signal partition path."""
    return layout.signal_path(
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )


def test_signal_repository_is_exported_from_package() -> None:
    """Package export matches the signal repository module class."""
    assert SignalRepository is SignalRepositoryDirect


def test_signal_partition_ref_is_frozen_dataclass() -> None:
    """SignalPartitionRef is an immutable slotted dataclass."""
    ref = SignalPartitionRef(
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert is_dataclass(ref)
    assert ref.exchange == _EXCHANGE
    assert ref.market == _MARKET
    assert ref.symbol == _SYMBOL
    assert ref.timeframe == _TIMEFRAME
    assert ref.year == _YEAR
    assert ref == SignalPartitionRef(
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    with pytest.raises(FrozenInstanceError):
        ref.year = 2025  # type: ignore[misc]


def test_save_and_load_uses_signal_layout_path(
    repository: SignalRepository,
    layout: StorageLayout,
    datastore: _InMemoryDataStore,
    sample_frame: pl.DataFrame,
) -> None:
    """Signal save/load uses StorageLayout.signal_path."""
    expected = _signal_path(layout)
    assert STORAGE_DIR_SIGNALS in expected.parts
    assert STORAGE_DIR_FEATURES not in expected.parts
    assert STORAGE_DIR_LABELS not in expected.parts
    assert STORAGE_DIR_TRAINING not in expected.parts

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


def test_save_overwrites_existing_partition(
    repository: SignalRepository,
    sample_frame: pl.DataFrame,
) -> None:
    """Saving the same partition twice replaces the stored frame."""
    replacement = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timeframe": ["1h"],
            "open_time": [1_800_000_000_000],
            "model_name": [_MODEL_NAME],
            "model_version": [_MODEL_VERSION],
            "signal": ["HOLD"],
        },
        schema=_CANONICAL_DTYPES,
    )
    repository.save(
        sample_frame,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    repository.save(
        replacement,
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
    assert_frame_equal(loaded, replacement)


def test_canonical_schema_round_trip(
    repository: SignalRepository,
    sample_frame: pl.DataFrame,
) -> None:
    """Persisted frames retain the canonical Signal column set."""
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
    assert tuple(loaded.columns) == _CANONICAL_COLUMNS
    assert "prediction" not in loaded.columns
    assert "confidence" not in loaded.columns


def test_canonical_column_order_preserved(
    repository: SignalRepository,
    sample_frame: pl.DataFrame,
) -> None:
    """Repository persists column order exactly as provided."""
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
    assert loaded.columns == list(_CANONICAL_COLUMNS)
    assert loaded.columns == sample_frame.columns


def test_dtype_preservation(
    repository: SignalRepository,
    sample_frame: pl.DataFrame,
) -> None:
    """Repository preserves provided Signal column dtypes on round-trip."""
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
    for column, dtype in _CANONICAL_DTYPES.items():
        assert loaded.schema[column] == dtype
        assert loaded.schema[column] == sample_frame.schema[column]


def test_signal_path_partitioning_matches_layout_contract(layout: StorageLayout) -> None:
    """Signal partitions follow exchange/market/symbol/timeframe/year.parquet."""
    path = _signal_path(layout)
    assert path.name == f"{_YEAR}.parquet"
    assert path.parent.name == _TIMEFRAME
    assert path.parent.parent.name == _SYMBOL
    assert path.parent.parent.parent.name == _MARKET
    assert path.parent.parent.parent.parent.name == _EXCHANGE
    assert path.parent.parent.parent.parent.parent.name == STORAGE_DIR_SIGNALS
    assert STORAGE_DIR_SIGNALS in path.parts


def test_public_api_does_not_return_filesystem_paths(
    repository: SignalRepository,
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
    repository: SignalRepository,
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
    repository: SignalRepository,
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
    assert datastore.exists_paths == [_signal_path(layout)]


def test_exists_true_when_partition_saved(
    repository: SignalRepository,
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
    assert _signal_path(layout) in datastore.exists_paths


def test_delete_removes_partition(
    repository: SignalRepository,
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
    expected = _signal_path(layout)

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
    repository: SignalRepository,
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
        tmp_path / STORAGE_DIR_SIGNALS / EXCHANGE_BINANCE / MARKET_USDT_PERPETUAL / "BTCUSDT" / "1h"
    )
    base.mkdir(parents=True, exist_ok=True)
    (base / "2023.parquet").write_bytes(b"")
    (base / "2025.parquet").write_bytes(b"")
    (base / "2024.parquet").write_bytes(b"")
    (base / "notes.txt").write_text("ignore", encoding="utf-8")

    repository = SignalRepository(StorageLayout(tmp_path), ParquetStore())
    years = repository.list_years(
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol="BTCUSDT",
        timeframe="1h",
    )

    assert years == (2023, 2024, 2025)


def test_list_years_empty_when_missing(tmp_path: Path) -> None:
    """list_years returns an empty tuple when no partitions exist."""
    repository = SignalRepository(StorageLayout(tmp_path), ParquetStore())
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
    """Discovery walks signal trees without returning filesystem paths."""
    path = (
        tmp_path
        / STORAGE_DIR_SIGNALS
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
        / STORAGE_DIR_SIGNALS
        / EXCHANGE_BINANCE
        / MARKET_USDT_PERPETUAL
        / "ETHUSDT"
        / "1h"
        / "2023.parquet"
    )
    eth.parent.mkdir(parents=True, exist_ok=True)
    eth.write_bytes(b"")

    repository = SignalRepository(StorageLayout(tmp_path), ParquetStore())
    partitions = repository.discover_partitions()

    assert partitions == (
        SignalPartitionRef(
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            symbol="BTCUSDT",
            timeframe="1h",
            year=2024,
        ),
        SignalPartitionRef(
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            symbol="ETHUSDT",
            timeframe="1h",
            year=2023,
        ),
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
            / STORAGE_DIR_SIGNALS
            / EXCHANGE_BINANCE
            / MARKET_USDT_PERPETUAL
            / symbol
            / timeframe
            / f"{year}.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

    repository = SignalRepository(StorageLayout(tmp_path), ParquetStore())
    partitions = repository.discover_partitions(
        symbols=("BTCUSDT",),
        timeframes=("1h",),
    )

    assert partitions == (
        SignalPartitionRef(
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            symbol="BTCUSDT",
            timeframe="1h",
            year=2024,
        ),
    )


def test_round_trip_with_parquet_store(
    layout: StorageLayout,
    sample_frame: pl.DataFrame,
) -> None:
    """Signal repository round-trips through a real ``ParquetStore``."""
    repository = SignalRepository(layout, ParquetStore())
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
    assert tuple(loaded.columns) == _CANONICAL_COLUMNS
    for column, dtype in _CANONICAL_DTYPES.items():
        assert loaded.schema[column] == dtype
    assert _signal_path(layout).is_file()


def test_signal_paths_differ_from_other_dataset_paths(layout: StorageLayout) -> None:
    """Signal partitions resolve to a location distinct from other tiers."""
    signal = _signal_path(layout)
    training = layout.training_path(_EXCHANGE, _MARKET, _SYMBOL, _TIMEFRAME, _YEAR)
    feature = layout.feature_path(_EXCHANGE, _MARKET, _SYMBOL, _TIMEFRAME, _YEAR)
    label = layout.label_path(_EXCHANGE, _MARKET, _SYMBOL, _TIMEFRAME, _YEAR)
    assert signal != training
    assert signal != feature
    assert signal != label
    assert STORAGE_DIR_SIGNALS in signal.parts
    assert STORAGE_DIR_TRAINING in training.parts
    assert STORAGE_DIR_FEATURES in feature.parts
    assert STORAGE_DIR_LABELS in label.parts
