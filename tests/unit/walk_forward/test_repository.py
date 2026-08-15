"""Unit tests for CQROS walk-forward dataset repository."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.core.constants import (
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
    STORAGE_DIR_WALK_FORWARD,
)
from cqros.core.types import FilePath
from cqros.storage import DatasetNotFoundError, ParquetStore, StorageLayout
from cqros.walk_forward import WalkForwardError, WalkForwardRepository, WalkForwardStatus
from cqros.walk_forward.repository import WalkForwardPartitionRef
from cqros.walk_forward.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    WALK_FORWARD_SCHEMA,
)

_MANAGER = "simple"
_EXCHANGE = EXCHANGE_BINANCE
_MARKET = MARKET_USDT_PERPETUAL
_TIMEFRAME = "1h"
_YEAR = 2026
_SELECTION_TIME = 1_704_067_200_000


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


def _partition_kwargs(
    *,
    manager: str = _MANAGER,
    timeframe: str = _TIMEFRAME,
    year: int = _YEAR,
) -> dict[str, object]:
    """Build keyword arguments identifying one partition."""
    return {
        "manager": manager,
        "exchange": _EXCHANGE,
        "market": _MARKET,
        "timeframe": timeframe,
        "year": year,
    }


def _canonical_frame(
    *,
    timeframe: str = _TIMEFRAME,
    status: str = WalkForwardStatus.PASS.value,
    train_score: float = 0.0,
    selected_factors: int = 1,
) -> pl.DataFrame:
    """Build a canonical walk-forward frame for repository tests."""
    return pl.DataFrame(
        {
            "strategy_name": ["default_strategy"],
            "strategy_version": ["v1"],
            "timeframe": [timeframe],
            "fold_id": [1],
            "train_start": [_SELECTION_TIME],
            "train_end": [_SELECTION_TIME],
            "test_start": [_SELECTION_TIME],
            "test_end": [_SELECTION_TIME],
            "train_rows": [1],
            "test_rows": [1],
            "selected_factors": [selected_factors],
            "model_version": ["v1"],
            "train_score": [train_score],
            "test_score": [0.0],
            "overfit_gap": [0.0],
            "status": [status],
        },
        schema=dict(COLUMN_DTYPES),
    ).select(list(CANONICAL_COLUMN_ORDER))


# ---------------------------------------------------------------------------
# WalkForwardPartitionRef dataclass
# ---------------------------------------------------------------------------


def test_partition_ref_is_frozen_dataclass() -> None:
    """WalkForwardPartitionRef is a frozen immutable dataclass."""
    ref = WalkForwardPartitionRef(
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert is_dataclass(ref)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        ref.timeframe = "4h"  # type: ignore[misc]


def test_partition_ref_equality() -> None:
    """Two WalkForwardPartitionRef instances with identical fields are equal."""
    ref1 = WalkForwardPartitionRef(
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    ref2 = WalkForwardPartitionRef(
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
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
    repository = WalkForwardRepository(StorageLayout(Path("/data")), store)
    frame = _canonical_frame()
    kwargs = _partition_kwargs()

    assert not repository.exists(**kwargs)  # type: ignore[arg-type]

    repository.save(frame, **kwargs)  # type: ignore[arg-type]

    assert repository.exists(**kwargs)  # type: ignore[arg-type]

    loaded = repository.load(**kwargs)  # type: ignore[arg-type]
    assert_frame_equal(loaded, frame)
    assert loaded.schema == WALK_FORWARD_SCHEMA


def test_save_overwrites_existing_partition() -> None:
    """Saving twice overwrites the existing partition with the new frame."""
    store = _InMemoryDataStore()
    repository = WalkForwardRepository(StorageLayout(Path("/data")), store)
    kwargs = _partition_kwargs()

    frame_v1 = _canonical_frame(train_score=0.05)
    repository.save(frame_v1, **kwargs)  # type: ignore[arg-type]

    frame_v2 = _canonical_frame(train_score=0.10)
    repository.save(frame_v2, **kwargs)  # type: ignore[arg-type]

    loaded = repository.load(**kwargs)  # type: ignore[arg-type]
    assert loaded["train_score"].to_list() == [pytest.approx(0.10)]


def test_save_rejects_non_dataframe() -> None:
    """save() rejects non-DataFrame inputs with WF_REPO_FRAME_TYPE."""
    repository = WalkForwardRepository(StorageLayout(Path("/data")), _InMemoryDataStore())
    with pytest.raises(WalkForwardError) as exc_info:
        repository.save(
            "not-a-frame",  # type: ignore[arg-type]
            **_partition_kwargs(),  # type: ignore[arg-type]
        )
    assert exc_info.value.error_code == "WF_REPO_FRAME_TYPE"


def test_save_rejects_missing_required_columns() -> None:
    """save() rejects frames missing required columns."""
    repository = WalkForwardRepository(StorageLayout(Path("/data")), _InMemoryDataStore())
    with pytest.raises(WalkForwardError) as exc_info:
        repository.save(
            pl.DataFrame({"strategy_name": ["default_strategy"]}),
            **_partition_kwargs(),  # type: ignore[arg-type]
        )
    assert exc_info.value.error_code == "WF_REPO_MISSING_COLUMNS"


def test_save_rejects_schema_cast_failure() -> None:
    """save() rejects frames that cannot cast to WALK_FORWARD_SCHEMA."""
    repository = WalkForwardRepository(StorageLayout(Path("/data")), _InMemoryDataStore())
    frame = _canonical_frame().with_columns(pl.lit("not-a-float").alias("train_score"))
    with pytest.raises(WalkForwardError) as exc_info:
        repository.save(frame, **_partition_kwargs())  # type: ignore[arg-type]
    assert exc_info.value.error_code == "WF_REPO_SCHEMA_CAST"


def test_load_casts_to_walk_forward_schema() -> None:
    """load() returns frames cast to WALK_FORWARD_SCHEMA."""
    store = _InMemoryDataStore()
    repository = WalkForwardRepository(StorageLayout(Path("/data")), store)
    kwargs = _partition_kwargs()
    repository.save(_canonical_frame(), **kwargs)  # type: ignore[arg-type]
    loaded = repository.load(**kwargs)  # type: ignore[arg-type]
    assert loaded.schema == WALK_FORWARD_SCHEMA
    assert tuple(loaded.columns) == CANONICAL_COLUMN_ORDER


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


def test_delete_removes_partition() -> None:
    """delete() removes the partition so subsequent exists() returns False."""
    store = _InMemoryDataStore()
    repository = WalkForwardRepository(StorageLayout(Path("/data")), store)
    kwargs = _partition_kwargs()
    repository.save(_canonical_frame(), **kwargs)  # type: ignore[arg-type]
    assert repository.exists(**kwargs)  # type: ignore[arg-type]

    repository.delete(**kwargs)  # type: ignore[arg-type]
    assert not repository.exists(**kwargs)  # type: ignore[arg-type]


def test_delete_nonexistent_raises() -> None:
    """Deleting a partition that does not exist propagates DatasetNotFoundError."""
    store = _InMemoryDataStore()
    repository = WalkForwardRepository(StorageLayout(Path("/data")), store)
    with pytest.raises(DatasetNotFoundError):
        repository.delete(**_partition_kwargs())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ParquetStore on-disk round-trip
# ---------------------------------------------------------------------------


def test_parquet_store_round_trip(tmp_path: Path) -> None:
    """save/load with ParquetStore writes and reads canonical frames correctly."""
    layout = StorageLayout(tmp_path)
    repository = WalkForwardRepository(layout, ParquetStore())
    frame = _canonical_frame()

    repository.save(frame, **_partition_kwargs())  # type: ignore[arg-type]
    loaded = repository.load(**_partition_kwargs())  # type: ignore[arg-type]
    assert_frame_equal(loaded, frame)


def test_parquet_store_path_contains_walk_forward_directory(tmp_path: Path) -> None:
    """Saved partitions reside under the walk-forward storage directory."""
    layout = StorageLayout(tmp_path)
    repository = WalkForwardRepository(layout, ParquetStore())
    repository.save(_canonical_frame(), **_partition_kwargs())  # type: ignore[arg-type]
    assert (tmp_path / STORAGE_DIR_WALK_FORWARD).is_dir()


# ---------------------------------------------------------------------------
# discover_managers
# ---------------------------------------------------------------------------


def test_discover_managers_returns_empty_when_no_partitions(tmp_path: Path) -> None:
    """discover_managers returns empty tuple when the walk-forward tier is absent."""
    repository = WalkForwardRepository(StorageLayout(tmp_path), ParquetStore())
    assert repository.discover_managers() == ()


def test_discover_managers_returns_sorted_names(tmp_path: Path) -> None:
    """discover_managers returns alphabetically sorted manager names."""
    layout = StorageLayout(tmp_path)
    repository = WalkForwardRepository(layout, ParquetStore())
    for manager in ("zebra", "alpha", "beta"):
        repository.save(
            _canonical_frame(),
            **_partition_kwargs(manager=manager),  # type: ignore[arg-type]
        )
    assert repository.discover_managers() == ("alpha", "beta", "zebra")


# ---------------------------------------------------------------------------
# discover_timeframes
# ---------------------------------------------------------------------------


def test_discover_timeframes_returns_sorted(tmp_path: Path) -> None:
    """discover_timeframes returns sorted timeframe names for a given manager."""
    layout = StorageLayout(tmp_path)
    repository = WalkForwardRepository(layout, ParquetStore())
    for timeframe in ("4h", "1d", "1h"):
        repository.save(
            _canonical_frame(timeframe=timeframe),
            **_partition_kwargs(timeframe=timeframe),  # type: ignore[arg-type]
        )
    assert repository.discover_timeframes(manager=_MANAGER) == ("1d", "1h", "4h")


# ---------------------------------------------------------------------------
# list_years
# ---------------------------------------------------------------------------


def test_list_years_returns_sorted_years(tmp_path: Path) -> None:
    """list_years returns sorted calendar years as integers."""
    layout = StorageLayout(tmp_path)
    repository = WalkForwardRepository(layout, ParquetStore())
    for year in (2026, 2024, 2025):
        repository.save(
            _canonical_frame(),
            **_partition_kwargs(year=year),  # type: ignore[arg-type]
        )
    years = repository.list_years(
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        timeframe=_TIMEFRAME,
    )
    assert years == (2024, 2025, 2026)


# ---------------------------------------------------------------------------
# discover / discover_partitions
# ---------------------------------------------------------------------------


def test_discover_partitions_returns_empty_on_missing_root(tmp_path: Path) -> None:
    """discover_partitions returns empty tuple when no partitions exist."""
    repository = WalkForwardRepository(StorageLayout(tmp_path), ParquetStore())
    assert repository.discover_partitions() == ()


def test_discover_returns_same_as_discover_partitions(tmp_path: Path) -> None:
    """discover() delegates to discover_partitions()."""
    layout = StorageLayout(tmp_path)
    repository = WalkForwardRepository(layout, ParquetStore())
    repository.save(_canonical_frame(), **_partition_kwargs())  # type: ignore[arg-type]
    assert repository.discover() == repository.discover_partitions()


def test_discover_partitions_returns_sorted_refs(tmp_path: Path) -> None:
    """discover_partitions returns deterministically sorted partition references."""
    layout = StorageLayout(tmp_path)
    repository = WalkForwardRepository(layout, ParquetStore())
    for timeframe, year in (("4h", 2025), ("1h", 2026), ("1h", 2025)):
        repository.save(
            _canonical_frame(timeframe=timeframe),
            **_partition_kwargs(timeframe=timeframe, year=year),  # type: ignore[arg-type]
        )
    partitions = repository.discover_partitions()
    assert len(partitions) == 3
    assert partitions[0].timeframe == "1h"
    assert partitions[0].year == 2025
    assert partitions[1].timeframe == "1h"
    assert partitions[1].year == 2026
    assert partitions[2].timeframe == "4h"
    assert partitions[2].year == 2025


def test_discover_partitions_filters_by_manager(tmp_path: Path) -> None:
    """discover_partitions respects manager filter."""
    layout = StorageLayout(tmp_path)
    repository = WalkForwardRepository(layout, ParquetStore())
    repository.save(
        _canonical_frame(),
        **_partition_kwargs(manager="manager-a"),  # type: ignore[arg-type]
    )
    repository.save(
        _canonical_frame(),
        **_partition_kwargs(manager="manager-b"),  # type: ignore[arg-type]
    )
    partitions = repository.discover_partitions(managers=["manager-a"])
    assert len(partitions) == 1
    assert partitions[0].manager == "manager-a"


def test_discover_partitions_filters_by_timeframe(tmp_path: Path) -> None:
    """discover_partitions respects timeframe filter."""
    layout = StorageLayout(tmp_path)
    repository = WalkForwardRepository(layout, ParquetStore())
    for timeframe in ("1h", "4h"):
        repository.save(
            _canonical_frame(timeframe=timeframe),
            **_partition_kwargs(timeframe=timeframe),  # type: ignore[arg-type]
        )
    partitions = repository.discover_partitions(timeframes=["1h"])
    assert len(partitions) == 1
    assert partitions[0].timeframe == "1h"
