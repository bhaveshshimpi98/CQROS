"""Unit tests for CQROS position dataset repository."""

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
    STORAGE_DIR_POSITIONS,
)
from cqros.core.types import FilePath
from cqros.positions import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    PositionPartitionRef,
    PositionRepository,
    PositionStatus,
)
from cqros.positions.repository import PositionRepository as PositionRepositoryDirect
from cqros.storage import DatasetNotFoundError, ParquetStore, StorageLayout

_MANAGER = "simple"
_EXCHANGE = EXCHANGE_BINANCE
_MARKET = MARKET_USDT_PERPETUAL
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
                error_code="STORAGE-TEST-001",
                details={"path": str(target)},
            ) from exc

    def schema(self, path: FilePath) -> pl.Schema:
        return self.read(path).schema

    def row_count(self, path: FilePath) -> int:
        return self.read(path).height


@pytest.fixture
def sample_frame() -> pl.DataFrame:
    """Return a deterministic canonical sample position DataFrame."""
    opened_at = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
    return pl.DataFrame(
        {
            "symbol": [_SYMBOL],
            "timeframe": [_TIMEFRAME],
            "position_id": ["pos-00000001"],
            "side": ["LONG"],
            "status": [PositionStatus.OPEN.value],
            "quantity": [1.0],
            "average_entry_price": [100.0],
            "market_price": [100.0],
            "realized_pnl": [0.0],
            "unrealized_pnl": [0.0],
            "fees_paid": [0.0],
            "opened_at": [opened_at],
            "updated_at": [opened_at],
            "closed_at": [None],
            "model_name": ["alpha-lgbm"],
            "model_version": ["1.0.0"],
            "optimizer": ["equal_weight"],
            "policy": ["fixed_risk"],
            "manager": [_MANAGER],
        },
        schema=dict(COLUMN_DTYPES),
    ).select(list(CANONICAL_COLUMN_ORDER))


def test_position_repository_is_exported_from_package() -> None:
    """Package export matches the repository module by identity."""
    assert PositionRepository is PositionRepositoryDirect


def test_position_partition_ref_is_frozen_dataclass() -> None:
    """PositionPartitionRef is an immutable slotted dataclass."""
    assert is_dataclass(PositionPartitionRef)
    ref = PositionPartitionRef(
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    with pytest.raises(FrozenInstanceError):
        ref.year = 2025  # type: ignore[misc]


def test_save_load_exists_delete_round_trip(
    tmp_path: Path,
    sample_frame: pl.DataFrame,
) -> None:
    """Repository persists and reloads frames through the layout path."""
    layout = StorageLayout(tmp_path)
    datastore = _InMemoryDataStore()
    repository = PositionRepository(layout, datastore)
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
    repository.save(
        sample_frame,
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    expected_path = layout.position_path(
        _MANAGER,
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    assert datastore.write_paths == [expected_path]
    assert STORAGE_DIR_POSITIONS in expected_path.parts
    loaded = repository.load(
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert_frame_equal(loaded, sample_frame)
    repository.delete(
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert datastore.delete_paths == [expected_path]


def test_discover_partitions_from_disk(tmp_path: Path, sample_frame: pl.DataFrame) -> None:
    """Discovery walks the positions tier and returns sorted partition refs."""
    layout = StorageLayout(tmp_path)
    repository = PositionRepository(layout, ParquetStore())
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
        sample_frame,
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol="ETHUSDT",
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert repository.discover_managers() == (_MANAGER,)
    assert repository.discover_symbols(manager=_MANAGER) == ("BTCUSDT", "ETHUSDT")
    partitions = repository.discover_partitions(managers=(_MANAGER,))
    assert partitions == (
        PositionPartitionRef(
            manager=_MANAGER,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol="BTCUSDT",
            timeframe=_TIMEFRAME,
            year=_YEAR,
        ),
        PositionPartitionRef(
            manager=_MANAGER,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol="ETHUSDT",
            timeframe=_TIMEFRAME,
            year=_YEAR,
        ),
    )
