"""Unit tests for CQROS analytics dataset repository."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.analytics import AnalyticsRepository, AnalyticsStatus, AnalyticsValidationError
from cqros.analytics.repository import AnalyticsPartitionRef
from cqros.analytics.schema import ANALYTICS_SCHEMA, CANONICAL_COLUMN_ORDER, COLUMN_DTYPES
from cqros.core.constants import (
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
    STORAGE_DIR_ANALYTICS,
)
from cqros.core.types import FilePath
from cqros.storage import DatasetNotFoundError, ParquetStore, StorageLayout

_MANAGER = "simple"
_EXCHANGE = EXCHANGE_BINANCE
_MARKET = MARKET_USDT_PERPETUAL
_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1h"
_YEAR = 2026
_OPEN_TIME_MS = int(datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC).timestamp() * 1000.0)


class _InMemoryDataStore:
    """Minimal ``IDataStore`` stub that stores frames in memory."""

    def __init__(self) -> None:
        self.frames: dict[Path, pl.DataFrame] = {}
        self.write_paths: list[Path] = []
        self.read_paths: list[Path] = []
        self.exists_paths: list[Path] = []
        self.delete_paths: list[Path] = []

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
                error_code="STORAGE-TEST-002",
                details={"path": str(target)},
            ) from exc

    def schema(self, path: FilePath) -> pl.Schema:
        return self.read(path).schema

    def row_count(self, path: FilePath) -> int:
        return self.read(path).height


def _canonical_frame(
    *,
    symbol: str = _SYMBOL,
    status: str = AnalyticsStatus.FINISHED.value,
    rolling_return: float = 0.05,
) -> pl.DataFrame:
    """Build a canonical analytics frame for repository tests."""
    return pl.DataFrame(
        {
            "symbol": [symbol],
            "timeframe": [_TIMEFRAME],
            "open_time": [_OPEN_TIME_MS],
            "manager": [_MANAGER],
            "rolling_return": [rolling_return],
            "rolling_volatility": [0.1],
            "rolling_sharpe": [1.0],
            "rolling_sortino": [1.2],
            "rolling_max_drawdown": [0.02],
            "rolling_win_rate": [0.6],
            "rolling_profit_factor": [1.5],
            "rolling_expectancy": [10.0],
            "rolling_cagr": [0.08],
            "rolling_calmar": [4.0],
            "rolling_recovery_factor": [25000.0],
            "benchmark_return": [0.0],
            "benchmark_alpha": [0.0],
            "benchmark_beta": [0.0],
            "benchmark_correlation": [0.0],
            "benchmark_tracking_error": [0.0],
            "benchmark_information_ratio": [0.0],
            "status": [status],
        },
        schema=dict(COLUMN_DTYPES),
    ).select(list(CANONICAL_COLUMN_ORDER))


def _partition_kwargs() -> dict[str, object]:
    """Return common partition identity kwargs for repository calls."""
    return {
        "manager": _MANAGER,
        "exchange": _EXCHANGE,
        "market": _MARKET,
        "symbol": _SYMBOL,
        "timeframe": _TIMEFRAME,
        "year": _YEAR,
    }


# ---------------------------------------------------------------------------
# AnalyticsPartitionRef dataclass
# ---------------------------------------------------------------------------


def test_partition_ref_is_frozen_dataclass() -> None:
    """AnalyticsPartitionRef is a frozen immutable dataclass."""
    ref = AnalyticsPartitionRef(
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert is_dataclass(ref)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        ref.symbol = "ETHUSDT"  # type: ignore[misc]


def test_partition_ref_equality() -> None:
    """Two AnalyticsPartitionRef instances with identical fields are equal."""
    ref1 = AnalyticsPartitionRef(
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    ref2 = AnalyticsPartitionRef(
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert ref1 == ref2


# ---------------------------------------------------------------------------
# save / exists / load round-trip
# ---------------------------------------------------------------------------


def test_save_exists_load_round_trip() -> None:
    """Saved frames can be retrieved by exists() and load()."""
    store = _InMemoryDataStore()
    repository = AnalyticsRepository(StorageLayout(Path("/data")), store)
    frame = _canonical_frame()
    kwargs = _partition_kwargs()

    assert not repository.exists(**kwargs)  # type: ignore[arg-type]

    repository.save(frame, **kwargs)  # type: ignore[arg-type]

    assert repository.exists(**kwargs)  # type: ignore[arg-type]

    loaded = repository.load(**kwargs)  # type: ignore[arg-type]
    assert_frame_equal(loaded, frame)
    assert loaded.schema == ANALYTICS_SCHEMA


def test_save_overwrites_existing_partition() -> None:
    """Saving twice overwrites the existing partition with the new frame."""
    store = _InMemoryDataStore()
    repository = AnalyticsRepository(StorageLayout(Path("/data")), store)
    kwargs = _partition_kwargs()

    frame_v1 = _canonical_frame(rolling_return=0.05)
    repository.save(frame_v1, **kwargs)  # type: ignore[arg-type]

    frame_v2 = _canonical_frame(rolling_return=0.10)
    repository.save(frame_v2, **kwargs)  # type: ignore[arg-type]

    loaded = repository.load(**kwargs)  # type: ignore[arg-type]
    assert loaded["rolling_return"].to_list() == [pytest.approx(0.10)]


def test_save_rejects_non_dataframe() -> None:
    """save() rejects non-DataFrame inputs with ANA_REPO_FRAME_TYPE."""
    repository = AnalyticsRepository(StorageLayout(Path("/data")), _InMemoryDataStore())
    with pytest.raises(AnalyticsValidationError) as exc_info:
        repository.save(
            "not-a-frame",  # type: ignore[arg-type]
            **_partition_kwargs(),  # type: ignore[arg-type]
        )
    assert exc_info.value.error_code == "ANA_REPO_FRAME_TYPE"


def test_save_rejects_missing_required_columns() -> None:
    """save() rejects frames missing required columns."""
    repository = AnalyticsRepository(StorageLayout(Path("/data")), _InMemoryDataStore())
    with pytest.raises(AnalyticsValidationError) as exc_info:
        repository.save(
            pl.DataFrame({"symbol": ["BTCUSDT"]}),
            **_partition_kwargs(),  # type: ignore[arg-type]
        )
    assert exc_info.value.error_code == "ANA_REPO_MISSING_COLUMNS"


def test_save_rejects_schema_cast_failure() -> None:
    """save() rejects frames that cannot cast to ANALYTICS_SCHEMA."""
    repository = AnalyticsRepository(StorageLayout(Path("/data")), _InMemoryDataStore())
    frame = _canonical_frame().with_columns(pl.lit("not-a-float").alias("rolling_return"))
    with pytest.raises(AnalyticsValidationError) as exc_info:
        repository.save(frame, **_partition_kwargs())  # type: ignore[arg-type]
    assert exc_info.value.error_code == "ANA_REPO_SCHEMA_CAST"


def test_load_casts_to_analytics_schema() -> None:
    """load() returns frames cast to ANALYTICS_SCHEMA."""
    store = _InMemoryDataStore()
    repository = AnalyticsRepository(StorageLayout(Path("/data")), store)
    kwargs = _partition_kwargs()
    repository.save(_canonical_frame(), **kwargs)  # type: ignore[arg-type]
    loaded = repository.load(**kwargs)  # type: ignore[arg-type]
    assert loaded.schema == ANALYTICS_SCHEMA
    assert tuple(loaded.columns) == CANONICAL_COLUMN_ORDER


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


def test_delete_removes_partition() -> None:
    """delete() removes the partition so subsequent exists() returns False."""
    store = _InMemoryDataStore()
    repository = AnalyticsRepository(StorageLayout(Path("/data")), store)
    kwargs = _partition_kwargs()
    repository.save(_canonical_frame(), **kwargs)  # type: ignore[arg-type]
    assert repository.exists(**kwargs)  # type: ignore[arg-type]

    repository.delete(**kwargs)  # type: ignore[arg-type]
    assert not repository.exists(**kwargs)  # type: ignore[arg-type]


def test_delete_nonexistent_raises() -> None:
    """Deleting a partition that does not exist propagates DatasetNotFoundError."""
    store = _InMemoryDataStore()
    repository = AnalyticsRepository(StorageLayout(Path("/data")), store)
    with pytest.raises(DatasetNotFoundError):
        repository.delete(**_partition_kwargs())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ParquetStore on-disk round-trip
# ---------------------------------------------------------------------------


def test_parquet_store_round_trip(tmp_path: Path) -> None:
    """save/load with ParquetStore writes and reads canonical frames correctly."""
    layout = StorageLayout(tmp_path)
    repository = AnalyticsRepository(layout, ParquetStore())
    frame = _canonical_frame()
    kwargs = _partition_kwargs()

    repository.save(frame, **kwargs)  # type: ignore[arg-type]
    loaded = repository.load(**kwargs)  # type: ignore[arg-type]
    assert_frame_equal(loaded, frame)


def test_parquet_store_path_contains_analytics_directory(tmp_path: Path) -> None:
    """Saved partitions reside under the analytics storage directory."""
    layout = StorageLayout(tmp_path)
    repository = AnalyticsRepository(layout, ParquetStore())
    repository.save(_canonical_frame(), **_partition_kwargs())  # type: ignore[arg-type]
    assert (tmp_path / STORAGE_DIR_ANALYTICS).is_dir()


# ---------------------------------------------------------------------------
# discover_managers
# ---------------------------------------------------------------------------


def test_discover_managers_returns_empty_when_no_partitions(tmp_path: Path) -> None:
    """discover_managers returns empty tuple when the analytics tier is absent."""
    repository = AnalyticsRepository(StorageLayout(tmp_path), ParquetStore())
    assert repository.discover_managers() == ()


def test_discover_managers_returns_sorted_names(tmp_path: Path) -> None:
    """discover_managers returns alphabetically sorted manager names."""
    layout = StorageLayout(tmp_path)
    repository = AnalyticsRepository(layout, ParquetStore())
    for manager in ("zebra", "alpha", "beta"):
        repository.save(
            _canonical_frame(),
            manager=manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )
    assert repository.discover_managers() == ("alpha", "beta", "zebra")


# ---------------------------------------------------------------------------
# discover_symbols / discover_timeframes
# ---------------------------------------------------------------------------


def test_discover_symbols_returns_sorted(tmp_path: Path) -> None:
    """discover_symbols returns sorted symbol names for a given manager."""
    layout = StorageLayout(tmp_path)
    repository = AnalyticsRepository(layout, ParquetStore())
    for symbol in ("SOLUSDT", "BTCUSDT", "ETHUSDT"):
        repository.save(
            _canonical_frame(symbol=symbol),
            manager=_MANAGER,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=symbol,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )
    assert repository.discover_symbols(manager=_MANAGER) == ("BTCUSDT", "ETHUSDT", "SOLUSDT")


def test_discover_timeframes_returns_sorted(tmp_path: Path) -> None:
    """discover_timeframes returns sorted timeframe names for a given symbol."""
    layout = StorageLayout(tmp_path)
    repository = AnalyticsRepository(layout, ParquetStore())
    for timeframe in ("4h", "1d", "1h"):
        repository.save(
            _canonical_frame(),
            manager=_MANAGER,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=_SYMBOL,
            timeframe=timeframe,
            year=_YEAR,
        )
    assert repository.discover_timeframes(manager=_MANAGER, symbol=_SYMBOL) == ("1d", "1h", "4h")


# ---------------------------------------------------------------------------
# list_years
# ---------------------------------------------------------------------------


def test_list_years_returns_sorted_years(tmp_path: Path) -> None:
    """list_years returns sorted calendar years as integers."""
    layout = StorageLayout(tmp_path)
    repository = AnalyticsRepository(layout, ParquetStore())
    for year in (2026, 2024, 2025):
        repository.save(
            _canonical_frame(),
            manager=_MANAGER,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            year=year,
        )
    years = repository.list_years(
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
    )
    assert years == (2024, 2025, 2026)


# ---------------------------------------------------------------------------
# discover / discover_partitions
# ---------------------------------------------------------------------------


def test_discover_partitions_returns_empty_on_missing_root(tmp_path: Path) -> None:
    """discover_partitions returns empty tuple when no partitions exist."""
    repository = AnalyticsRepository(StorageLayout(tmp_path), ParquetStore())
    assert repository.discover_partitions() == ()
    assert repository.discover() == ()


def test_discover_partitions_returns_sorted_refs(tmp_path: Path) -> None:
    """discover_partitions returns deterministically sorted partition references."""
    layout = StorageLayout(tmp_path)
    repository = AnalyticsRepository(layout, ParquetStore())
    for symbol, year in (("ETHUSDT", 2025), ("BTCUSDT", 2026), ("BTCUSDT", 2025)):
        repository.save(
            _canonical_frame(symbol=symbol),
            manager=_MANAGER,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=symbol,
            timeframe=_TIMEFRAME,
            year=year,
        )
    partitions = repository.discover_partitions()
    assert len(partitions) == 3
    assert partitions[0].symbol == "BTCUSDT"
    assert partitions[0].year == 2025
    assert partitions[1].symbol == "BTCUSDT"
    assert partitions[1].year == 2026
    assert partitions[2].symbol == "ETHUSDT"
    assert partitions[2].year == 2025


def test_discover_filters_by_manager(tmp_path: Path) -> None:
    """discover() respects manager filter."""
    layout = StorageLayout(tmp_path)
    repository = AnalyticsRepository(layout, ParquetStore())
    repository.save(
        _canonical_frame(),
        manager="manager-a",
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    repository.save(
        _canonical_frame(),
        manager="manager-b",
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    partitions = repository.discover(managers=["manager-a"])
    assert len(partitions) == 1
    assert partitions[0].manager == "manager-a"


def test_discover_partitions_filters_by_symbol(tmp_path: Path) -> None:
    """discover_partitions respects symbol filter."""
    layout = StorageLayout(tmp_path)
    repository = AnalyticsRepository(layout, ParquetStore())
    for symbol in ("BTCUSDT", "ETHUSDT"):
        repository.save(
            _canonical_frame(symbol=symbol),
            manager=_MANAGER,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=symbol,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )
    partitions = repository.discover_partitions(symbols=["BTCUSDT"])
    assert all(p.symbol == "BTCUSDT" for p in partitions)
    assert len(partitions) == 1


def test_discover_partitions_filters_by_timeframe(tmp_path: Path) -> None:
    """discover_partitions respects timeframe filter."""
    layout = StorageLayout(tmp_path)
    repository = AnalyticsRepository(layout, ParquetStore())
    for timeframe in ("1h", "4h"):
        repository.save(
            _canonical_frame(),
            manager=_MANAGER,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=_SYMBOL,
            timeframe=timeframe,
            year=_YEAR,
        )
    partitions = repository.discover_partitions(timeframes=["1h"])
    assert len(partitions) == 1
    assert partitions[0].timeframe == "1h"
