"""Unit tests for CQROS exit-engine dataset repository."""

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
    STORAGE_DIR_EXIT_ENGINE,
)
from cqros.core.types import FilePath
from cqros.exit_engine import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    ExitAction,
    ExitEnginePartitionRef,
    ExitReason,
    ExitRepository,
)
from cqros.storage import DatasetNotFoundError, ParquetStore, StorageLayout

_MANAGER = "simple"
_EXCHANGE = EXCHANGE_BINANCE
_MARKET = MARKET_USDT_PERPETUAL
_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1h"
_YEAR = 2026
_OPEN_TIME = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)


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


def _canonical_frame(
    *,
    symbol: str = _SYMBOL,
    position_id: str = "pos-00000001",
    exit_action: str = ExitAction.HOLD.value,
    exit_reason: str = ExitReason.NONE.value,
) -> pl.DataFrame:
    """Build a canonical exit-engine frame for repository tests."""
    return pl.DataFrame(
        {
            "symbol": [symbol],
            "timeframe": [_TIMEFRAME],
            "open_time": [_OPEN_TIME],
            "position_id": [position_id],
            "manager": [_MANAGER],
            "entry_price": [100.0],
            "current_price": [102.0],
            "quantity": [1.0],
            "risk_reward_ratio": [0.4],
            "risk_state": ["NORMAL"],
            "trade_state": ["NONE"],
            "pyramid_state": ["INSUFFICIENT_PROFIT"],
            "exit_action": [exit_action],
            "exit_reason": [exit_reason],
            "recommended_quantity": [0.0],
            "recommended_percent": [0.0],
            "priority": [0],
            "created_at": [_OPEN_TIME],
        },
        schema=dict(COLUMN_DTYPES),
    ).select(list(CANONICAL_COLUMN_ORDER))


# ---------------------------------------------------------------------------
# ExitEnginePartitionRef dataclass
# ---------------------------------------------------------------------------


def test_partition_ref_is_frozen_dataclass() -> None:
    """ExitEnginePartitionRef is a frozen immutable dataclass."""
    ref = ExitEnginePartitionRef(
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
    """Two ExitEnginePartitionRef instances with identical fields are equal."""
    ref1 = ExitEnginePartitionRef(
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    ref2 = ExitEnginePartitionRef(
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
    repository = ExitRepository(StorageLayout(Path("/data")), store)
    frame = _canonical_frame()

    assert not repository.exists(
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )

    repository.save(
        frame,
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )

    assert repository.exists(
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
    assert_frame_equal(loaded, frame)


def test_save_overwrites_existing_partition() -> None:
    """Saving twice overwrites the existing partition with the new frame."""
    store = _InMemoryDataStore()
    repository = ExitRepository(StorageLayout(Path("/data")), store)
    kwargs = dict(
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )

    frame_v1 = _canonical_frame(exit_action=ExitAction.HOLD.value)
    repository.save(frame_v1, **kwargs)  # type: ignore[arg-type]

    frame_v2 = _canonical_frame(exit_action=ExitAction.FULL_EXIT.value)
    repository.save(frame_v2, **kwargs)  # type: ignore[arg-type]

    loaded = repository.load(**kwargs)  # type: ignore[arg-type]
    assert loaded["exit_action"].to_list() == [ExitAction.FULL_EXIT.value]


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


def test_delete_removes_partition() -> None:
    """delete() removes the partition so subsequent exists() returns False."""
    store = _InMemoryDataStore()
    repository = ExitRepository(StorageLayout(Path("/data")), store)
    frame = _canonical_frame()
    kwargs = dict(
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    repository.save(frame, **kwargs)  # type: ignore[arg-type]
    assert repository.exists(**kwargs)  # type: ignore[arg-type]

    repository.delete(**kwargs)  # type: ignore[arg-type]
    assert not repository.exists(**kwargs)  # type: ignore[arg-type]


def test_delete_nonexistent_raises() -> None:
    """Deleting a partition that does not exist propagates DatasetNotFoundError."""
    store = _InMemoryDataStore()
    repository = ExitRepository(StorageLayout(Path("/data")), store)
    with pytest.raises(DatasetNotFoundError):
        repository.delete(
            manager=_MANAGER,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )


# ---------------------------------------------------------------------------
# ParquetStore on-disk round-trip
# ---------------------------------------------------------------------------


def test_parquet_store_round_trip(tmp_path: Path) -> None:
    """save/load with ParquetStore writes and reads canonical frames correctly."""
    layout = StorageLayout(tmp_path)
    repository = ExitRepository(layout, ParquetStore())
    frame = _canonical_frame()

    repository.save(
        frame,
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
    assert_frame_equal(loaded, frame)


def test_parquet_store_path_contains_exit_engine_directory(tmp_path: Path) -> None:
    """Saved partitions reside under the exit_engine storage directory."""
    layout = StorageLayout(tmp_path)
    repository = ExitRepository(layout, ParquetStore())
    repository.save(
        _canonical_frame(),
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert (tmp_path / STORAGE_DIR_EXIT_ENGINE).is_dir()


# ---------------------------------------------------------------------------
# discover_managers
# ---------------------------------------------------------------------------


def test_discover_managers_returns_empty_when_no_partitions(tmp_path: Path) -> None:
    """discover_managers returns empty tuple when the exit-engine tier is absent."""
    repository = ExitRepository(StorageLayout(tmp_path), ParquetStore())
    assert repository.discover_managers() == ()


def test_discover_managers_returns_sorted_names(tmp_path: Path) -> None:
    """discover_managers returns alphabetically sorted manager names."""
    layout = StorageLayout(tmp_path)
    repository = ExitRepository(layout, ParquetStore())
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
    repository = ExitRepository(layout, ParquetStore())
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
    repository = ExitRepository(layout, ParquetStore())
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
    repository = ExitRepository(layout, ParquetStore())
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
# discover_partitions
# ---------------------------------------------------------------------------


def test_discover_partitions_returns_empty_on_missing_root(tmp_path: Path) -> None:
    """discover_partitions returns empty tuple when no partitions exist."""
    repository = ExitRepository(StorageLayout(tmp_path), ParquetStore())
    assert repository.discover_partitions() == ()


def test_discover_partitions_returns_sorted_refs(tmp_path: Path) -> None:
    """discover_partitions returns deterministically sorted partition references."""
    layout = StorageLayout(tmp_path)
    repository = ExitRepository(layout, ParquetStore())
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


def test_discover_partitions_filters_by_manager(tmp_path: Path) -> None:
    """discover_partitions respects manager filter."""
    layout = StorageLayout(tmp_path)
    repository = ExitRepository(layout, ParquetStore())
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
    partitions = repository.discover_partitions(managers=["manager-a"])
    assert len(partitions) == 1
    assert partitions[0].manager == "manager-a"


def test_discover_partitions_filters_by_symbol(tmp_path: Path) -> None:
    """discover_partitions respects symbol filter."""
    layout = StorageLayout(tmp_path)
    repository = ExitRepository(layout, ParquetStore())
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
