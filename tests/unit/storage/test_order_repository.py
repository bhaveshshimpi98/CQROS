"""Unit tests for CQROS OMS order dataset repository."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.core.constants import (
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
    STORAGE_DIR_FEATURES,
    STORAGE_DIR_ORDERS,
    STORAGE_DIR_PORTFOLIOS,
    STORAGE_DIR_RISKS,
    STORAGE_DIR_SIGNALS,
    STORAGE_DIR_TRAINING,
)
from cqros.core.types import FilePath
from cqros.storage import (
    DatasetNotFoundError,
    OrderPartitionRef,
    OrderRepository,
    ParquetStore,
    StorageLayout,
)
from cqros.storage.order_repository import OrderRepository as OrderRepositoryDirect

_MANAGER = "simple"
_EXCHANGE = "binance"
_MARKET = "perpetual"
_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1h"
_YEAR = 2026
_MODEL_NAME = "alpha-lgbm"
_MODEL_VERSION = "1.0.0"
_POLICY = "fixed_risk"
_OPTIMIZER = "equal_weight"

_CANONICAL_COLUMNS = (
    "symbol",
    "timeframe",
    "open_time",
    "order_id",
    "parent_order_id",
    "model_name",
    "model_version",
    "policy",
    "optimizer",
    "side",
    "order_type",
    "quantity",
    "limit_price",
    "stop_price",
    "filled_quantity",
    "average_fill_price",
    "status",
    "created_at",
    "updated_at",
)

_CANONICAL_DTYPES = {
    "symbol": pl.Utf8,
    "timeframe": pl.Utf8,
    "open_time": pl.Datetime("us", "UTC"),
    "order_id": pl.Utf8,
    "parent_order_id": pl.Utf8,
    "model_name": pl.Utf8,
    "model_version": pl.Utf8,
    "policy": pl.Utf8,
    "optimizer": pl.Utf8,
    "side": pl.Utf8,
    "order_type": pl.Utf8,
    "quantity": pl.Float64,
    "limit_price": pl.Float64,
    "stop_price": pl.Float64,
    "filled_quantity": pl.Float64,
    "average_fill_price": pl.Float64,
    "status": pl.Utf8,
    "created_at": pl.Datetime("us", "UTC"),
    "updated_at": pl.Datetime("us", "UTC"),
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
    """Return a deterministic canonical sample order DataFrame."""
    created = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT", "ETHUSDT"],
            "timeframe": ["1h", "1h"],
            "open_time": [
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 1, microsecond=1, tzinfo=UTC),
            ],
            "order_id": ["aaa", "bbb"],
            "parent_order_id": ["aaa", "bbb"],
            "model_name": [_MODEL_NAME, _MODEL_NAME],
            "model_version": [_MODEL_VERSION, _MODEL_VERSION],
            "policy": [_POLICY, _POLICY],
            "optimizer": [_OPTIMIZER, _OPTIMIZER],
            "side": ["BUY", "SELL"],
            "order_type": ["MARKET", "MARKET"],
            "quantity": [0.25, 0.5],
            "limit_price": [None, None],
            "stop_price": [None, None],
            "filled_quantity": [0.0, 0.0],
            "average_fill_price": [None, None],
            "status": ["PENDING", "PENDING"],
            "created_at": [created, created],
            "updated_at": [created, created],
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
) -> OrderRepository:
    """Return an order repository wired to the test layout and datastore."""
    return OrderRepository(layout, datastore)


def _order_path(layout: StorageLayout) -> Path:
    """Compose the canonical sample order partition path."""
    return layout.order_path(
        _MANAGER,
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )


def test_order_repository_is_exported_from_package() -> None:
    """Package export matches the order repository module class."""
    assert OrderRepository is OrderRepositoryDirect


def test_order_partition_ref_is_frozen_dataclass() -> None:
    """OrderPartitionRef is an immutable slotted dataclass."""
    ref = OrderPartitionRef(
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert is_dataclass(ref)
    assert ref.manager == _MANAGER
    assert ref.exchange == _EXCHANGE
    assert ref.market == _MARKET
    assert ref.symbol == _SYMBOL
    assert ref.timeframe == _TIMEFRAME
    assert ref.year == _YEAR
    assert ref == OrderPartitionRef(
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    with pytest.raises(FrozenInstanceError):
        ref.year = 2025  # type: ignore[misc]


def test_save_and_load_uses_order_layout_path(
    repository: OrderRepository,
    layout: StorageLayout,
    datastore: _InMemoryDataStore,
    sample_frame: pl.DataFrame,
) -> None:
    """Order save/load uses StorageLayout.order_path."""
    expected = _order_path(layout)
    assert STORAGE_DIR_ORDERS in expected.parts
    assert STORAGE_DIR_RISKS not in expected.parts
    assert STORAGE_DIR_PORTFOLIOS not in expected.parts
    assert STORAGE_DIR_SIGNALS not in expected.parts
    assert STORAGE_DIR_FEATURES not in expected.parts
    assert STORAGE_DIR_TRAINING not in expected.parts

    repository.save(
        sample_frame,
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    loaded = repository.load(
        manager=_MANAGER,
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
    repository: OrderRepository,
    sample_frame: pl.DataFrame,
) -> None:
    """Saving the same partition twice replaces the stored frame."""
    created = datetime(2024, 7, 1, tzinfo=UTC)
    replacement = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timeframe": ["1h"],
            "open_time": [datetime(2024, 6, 1, tzinfo=UTC)],
            "order_id": ["ccc"],
            "parent_order_id": ["ccc"],
            "model_name": [_MODEL_NAME],
            "model_version": [_MODEL_VERSION],
            "policy": [_POLICY],
            "optimizer": [_OPTIMIZER],
            "side": ["BUY"],
            "order_type": ["MARKET"],
            "quantity": [0.1],
            "limit_price": [None],
            "stop_price": [None],
            "filled_quantity": [0.0],
            "average_fill_price": [None],
            "status": ["PENDING"],
            "created_at": [created],
            "updated_at": [created],
        },
        schema=_CANONICAL_DTYPES,
    )
    repository.save(
        sample_frame,
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    repository.save(
        replacement,
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    loaded = repository.load(
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert_frame_equal(loaded, replacement)


def test_canonical_schema_round_trip(
    repository: OrderRepository,
    sample_frame: pl.DataFrame,
) -> None:
    """Persisted frames retain the canonical Order column set."""
    repository.save(
        sample_frame,
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    loaded = repository.load(
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert tuple(loaded.columns) == _CANONICAL_COLUMNS
    assert "decision" not in loaded.columns
    assert "approved_weight" not in loaded.columns


def test_canonical_column_order_preserved(
    repository: OrderRepository,
    sample_frame: pl.DataFrame,
) -> None:
    """Repository persists column order exactly as provided."""
    repository.save(
        sample_frame,
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    loaded = repository.load(
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert loaded.columns == list(_CANONICAL_COLUMNS)
    assert loaded.columns == sample_frame.columns


def test_dtype_preservation(
    repository: OrderRepository,
    sample_frame: pl.DataFrame,
) -> None:
    """Repository preserves provided Order column dtypes on round-trip."""
    repository.save(
        sample_frame,
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    loaded = repository.load(
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    for column, dtype in _CANONICAL_DTYPES.items():
        assert loaded.schema[column] == dtype
        assert loaded.schema[column] == sample_frame.schema[column]


def test_order_path_partitioning_matches_layout_contract(
    layout: StorageLayout,
) -> None:
    """Order partitions follow manager/exchange/market/symbol/timeframe/year."""
    path = _order_path(layout)
    assert path.name == f"{_YEAR}.parquet"
    assert path.parent.name == _TIMEFRAME
    assert path.parent.parent.name == _SYMBOL
    assert path.parent.parent.parent.name == _MARKET
    assert path.parent.parent.parent.parent.name == _EXCHANGE
    assert path.parent.parent.parent.parent.parent.name == _MANAGER
    assert path.parent.parent.parent.parent.parent.parent.name == STORAGE_DIR_ORDERS
    assert STORAGE_DIR_ORDERS in path.parts


def test_public_api_does_not_return_filesystem_paths(
    repository: OrderRepository,
    sample_frame: pl.DataFrame,
) -> None:
    """Save returns None and load returns a DataFrame, never a Path."""
    result = repository.save(
        sample_frame,
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    loaded = repository.load(
        manager=_MANAGER,
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
    repository: OrderRepository,
) -> None:
    """Missing datasets surface the datastore ``DatasetNotFoundError``."""
    with pytest.raises(DatasetNotFoundError):
        repository.load(
            manager=_MANAGER,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )


def test_exists_false_when_missing(
    repository: OrderRepository,
    layout: StorageLayout,
    datastore: _InMemoryDataStore,
) -> None:
    """exists returns False and never reads Parquet contents."""
    assert (
        repository.exists(
            manager=_MANAGER,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )
        is False
    )
    assert datastore.read_paths == []
    assert datastore.exists_paths == [_order_path(layout)]


def test_exists_true_when_partition_saved(
    repository: OrderRepository,
    layout: StorageLayout,
    datastore: _InMemoryDataStore,
    sample_frame: pl.DataFrame,
) -> None:
    """exists returns True after a partition is saved."""
    repository.save(
        sample_frame,
        manager=_MANAGER,
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
            manager=_MANAGER,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )
        is True
    )
    assert datastore.read_paths == []
    assert _order_path(layout) in datastore.exists_paths


def test_delete_removes_partition(
    repository: OrderRepository,
    layout: StorageLayout,
    datastore: _InMemoryDataStore,
    sample_frame: pl.DataFrame,
) -> None:
    """delete removes a saved partition through the datastore."""
    repository.save(
        sample_frame,
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    expected = _order_path(layout)

    repository.delete(
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )

    assert datastore.delete_paths == [expected]
    assert (
        repository.exists(
            manager=_MANAGER,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )
        is False
    )


def test_delete_missing_propagates_not_found(
    repository: OrderRepository,
) -> None:
    """delete surfaces DatasetNotFoundError when the partition is absent."""
    with pytest.raises(DatasetNotFoundError):
        repository.delete(
            manager=_MANAGER,
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
        / STORAGE_DIR_ORDERS
        / _MANAGER
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

    repository = OrderRepository(StorageLayout(tmp_path), ParquetStore())
    years = repository.list_years(
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol="BTCUSDT",
        timeframe="1h",
    )

    assert years == (2023, 2024, 2025)


def test_list_years_empty_when_missing(tmp_path: Path) -> None:
    """list_years returns an empty tuple when no partitions exist."""
    repository = OrderRepository(StorageLayout(tmp_path), ParquetStore())
    assert (
        repository.list_years(
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            symbol="BTCUSDT",
            timeframe="1h",
        )
        == ()
    )


def test_discover_partitions_finds_year_files(tmp_path: Path) -> None:
    """Discovery walks order trees without returning filesystem paths."""
    path = (
        tmp_path
        / STORAGE_DIR_ORDERS
        / _MANAGER
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
        / STORAGE_DIR_ORDERS
        / _MANAGER
        / EXCHANGE_BINANCE
        / MARKET_USDT_PERPETUAL
        / "ETHUSDT"
        / "1h"
        / "2023.parquet"
    )
    eth.parent.mkdir(parents=True, exist_ok=True)
    eth.write_bytes(b"")

    repository = OrderRepository(StorageLayout(tmp_path), ParquetStore())
    partitions = repository.discover_partitions()

    assert partitions == (
        OrderPartitionRef(
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            symbol="BTCUSDT",
            timeframe="1h",
            year=2024,
        ),
        OrderPartitionRef(
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            symbol="ETHUSDT",
            timeframe="1h",
            year=2023,
        ),
    )
    assert repository.discover_managers() == (_MANAGER,)
    assert repository.discover_symbols(manager=_MANAGER) == ("BTCUSDT", "ETHUSDT")
    assert repository.discover_timeframes(manager=_MANAGER, symbol="BTCUSDT") == ("1h",)


def test_discover_partitions_applies_filters(tmp_path: Path) -> None:
    """Discovery filters by manager, symbol, and timeframe allowlists."""
    for manager, symbol, timeframe, year in (
        ("simple", "BTCUSDT", "1h", 2024),
        ("simple", "BTCUSDT", "4h", 2024),
        ("simple", "ETHUSDT", "1h", 2024),
        ("twap", "BTCUSDT", "1h", 2024),
    ):
        path = (
            tmp_path
            / STORAGE_DIR_ORDERS
            / manager
            / EXCHANGE_BINANCE
            / MARKET_USDT_PERPETUAL
            / symbol
            / timeframe
            / f"{year}.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

    repository = OrderRepository(StorageLayout(tmp_path), ParquetStore())
    partitions = repository.discover_partitions(
        managers=("simple",),
        symbols=("BTCUSDT",),
        timeframes=("1h",),
    )

    assert partitions == (
        OrderPartitionRef(
            manager="simple",
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
    """Order repository round-trips through a real ``ParquetStore``."""
    repository = OrderRepository(layout, ParquetStore())
    repository.save(
        sample_frame,
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    loaded = repository.load(
        manager=_MANAGER,
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
    assert _order_path(layout).is_file()


def test_order_paths_differ_from_other_dataset_paths(layout: StorageLayout) -> None:
    """Order partitions resolve to a location distinct from other tiers."""
    order = _order_path(layout)
    risk = layout.risk_path("fixed_risk", _EXCHANGE, _MARKET, _SYMBOL, _TIMEFRAME, _YEAR)
    portfolio = layout.portfolio_path(_EXCHANGE, _MARKET, _SYMBOL, _TIMEFRAME, _YEAR)
    signal = layout.signal_path(_EXCHANGE, _MARKET, _SYMBOL, _TIMEFRAME, _YEAR)
    training = layout.training_path(_EXCHANGE, _MARKET, _SYMBOL, _TIMEFRAME, _YEAR)
    feature = layout.feature_path(_EXCHANGE, _MARKET, _SYMBOL, _TIMEFRAME, _YEAR)
    assert order != risk
    assert order != portfolio
    assert order != signal
    assert order != training
    assert order != feature
    assert STORAGE_DIR_ORDERS in order.parts
    assert STORAGE_DIR_RISKS in risk.parts
    assert STORAGE_DIR_PORTFOLIOS in portfolio.parts
    assert STORAGE_DIR_SIGNALS in signal.parts
    assert STORAGE_DIR_TRAINING in training.parts
    assert STORAGE_DIR_FEATURES in feature.parts
