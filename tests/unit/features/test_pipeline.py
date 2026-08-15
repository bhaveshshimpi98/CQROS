"""Unit tests for CQROS Feature Engine ``FeaturePipeline``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.core.types import FilePath
from cqros.features.base import BaseFeature
from cqros.features.exceptions import (
    FeatureDependencyError,
    FeatureExecutionError,
    FeatureValidationError,
    UnknownFeatureError,
)
from cqros.features.pipeline import FeaturePipeline
from cqros.features.registry import FeatureRegistry
from cqros.features.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    FEATURE_COLUMNS,
    REQUIRED_COLUMNS,
)
from cqros.storage import DatasetNotFoundError, FeatureRepository, StorageLayout

_execution_log: list[str] = []

_EXCHANGE = "binance"
_MARKET = "perpetual"
_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1h"
_YEAR = 2026
_PARTITION_KWARGS: dict[str, Any] = {
    "exchange": _EXCHANGE,
    "market": _MARKET,
    "symbol": _SYMBOL,
    "timeframe": _TIMEFRAME,
    "year": _YEAR,
}


class _RecordingRepository:
    """Minimal feature repository stub that records save calls."""

    def __init__(self) -> None:
        self.saved: list[pl.DataFrame] = []
        self.save_kwargs: list[dict[str, object]] = []

    def save(
        self,
        dataframe: pl.DataFrame,
        *,
        exchange: str,
        market: str,
        symbol: str,
        timeframe: str,
        year: int,
    ) -> None:
        self.saved.append(dataframe)
        self.save_kwargs.append(
            {
                "exchange": exchange,
                "market": market,
                "symbol": symbol,
                "timeframe": timeframe,
                "year": year,
            }
        )


class _InMemoryDataStore:
    """Minimal ``IDataStore`` stub for real ``FeatureRepository`` wiring."""

    def __init__(self) -> None:
        self.frames: dict[Path, pl.DataFrame] = {}

    def write(self, path: FilePath, dataframe: pl.DataFrame) -> None:
        self.frames[Path(path)] = dataframe

    def read(self, path: FilePath) -> pl.DataFrame:
        target = Path(path)
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


@dataclass(frozen=True, slots=True)
class _AddColumnFeature(BaseFeature):
    """Appends a constant column named after the feature."""

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Return ``frame`` with ``produced_columns[0]`` added."""
        column: str = self.produced_columns[0]
        series = pl.Series(column, [1.0] * frame.height)
        return frame.hstack([series])


@dataclass(frozen=True, slots=True)
class _RecordingFeature(BaseFeature):
    """Appends a column and records execution into ``_execution_log``."""

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Record execution order and append the produced column."""
        _execution_log.append(self.name)
        column: str = self.produced_columns[0]
        series = pl.Series(column, [float(len(_execution_log))] * frame.height)
        return frame.hstack([series])


@dataclass(frozen=True, slots=True)
class _FailingFeature(BaseFeature):
    """Raises a RuntimeError from transform."""

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Always fail."""
        raise RuntimeError("boom")


@dataclass(frozen=True, slots=True)
class _TypedColumnFeature(BaseFeature):
    """Appends an integer column that must be cast to Float64."""

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Return ``frame`` with an Int64 produced column."""
        column: str = self.produced_columns[0]
        series = pl.Series(column, [1] * frame.height, dtype=pl.Int64)
        return frame.hstack([series])


@dataclass(frozen=True, slots=True)
class _ShiftWarmupFeature(_AddColumnFeature):
    """Stub whose warm-up equals lookback (shift semantics)."""

    @property
    def warmup_rows(self) -> int:
        return self.lookback


def _feature(
    name: str,
    *,
    dependencies: tuple[str, ...] = (),
    required_columns: tuple[str, ...] = (),
    lookback: int = 0,
    cls: (
        type[_AddColumnFeature]
        | type[_RecordingFeature]
        | type[_FailingFeature]
        | type[_TypedColumnFeature]
        | type[_ShiftWarmupFeature]
    ) = _AddColumnFeature,
) -> BaseFeature:
    """Build a concrete feature for pipeline tests."""
    return cls(
        name=name,
        version="1.0.0",
        category="test",
        description=f"{name} stub",
        required_columns=required_columns,
        produced_columns=(name,),
        lookback=lookback,
        dependencies=dependencies,
    )


def _base_frame(*, rows: int = 5) -> pl.DataFrame:
    """Build a primary-key input frame for merged-schema runs."""
    return pl.DataFrame(
        {
            "symbol": [_SYMBOL] * rows,
            "timeframe": [_TIMEFRAME] * rows,
            "open_time": list(range(rows)),
        }
    )


def _catalog_features(
    *,
    lookback: int = 0,
    cls: type[_AddColumnFeature] | type[_TypedColumnFeature] = _AddColumnFeature,
) -> tuple[BaseFeature, ...]:
    """Return stub features covering every merged feature column."""
    return tuple(_feature(name, lookback=lookback, cls=cls) for name in FEATURE_COLUMNS)


def _pipeline_with(
    features: tuple[BaseFeature, ...],
    repository: _RecordingRepository | FeatureRepository | None = None,
) -> tuple[FeaturePipeline, _RecordingRepository | FeatureRepository]:
    """Create a pipeline whose registry contains ``features``."""
    registry = FeatureRegistry()
    registry.register_many(features)
    repo: _RecordingRepository | FeatureRepository
    if repository is None:
        repo = _RecordingRepository()
    else:
        repo = repository
    return FeaturePipeline(registry, repo), repo  # type: ignore[arg-type]


def _reset_execution_log() -> None:
    """Clear the shared execution-order log."""
    _execution_log.clear()


def test_run_full_catalog_persists_merged_schema() -> None:
    """Full catalog execution finalizes schema columns and persists."""
    pipeline, repository = _pipeline_with(_catalog_features())
    assert isinstance(repository, _RecordingRepository)
    result = pipeline.run(_base_frame(), FEATURE_COLUMNS, **_PARTITION_KWARGS)

    assert result.columns == list(CANONICAL_COLUMN_ORDER)
    assert len(repository.saved) == 1
    assert_frame_equal(repository.saved[0], result)
    assert repository.save_kwargs == [_PARTITION_KWARGS]


def test_canonical_column_ordering() -> None:
    """Finalized output columns follow CANONICAL_COLUMN_ORDER exactly."""
    # Register features in reverse order so execution append order differs.
    reversed_features = tuple(_feature(name) for name in reversed(FEATURE_COLUMNS))
    pipeline, _repository = _pipeline_with(reversed_features)
    result = pipeline.run(
        _base_frame(),
        tuple(reversed(FEATURE_COLUMNS)),
        **_PARTITION_KWARGS,
    )
    assert result.columns == list(CANONICAL_COLUMN_ORDER)


def test_dtype_enforcement() -> None:
    """Finalized feature columns are cast to the schema Float64 dtype."""
    pipeline, _repository = _pipeline_with(_catalog_features(cls=_TypedColumnFeature))
    result = pipeline.run(_base_frame(), FEATURE_COLUMNS, **_PARTITION_KWARGS)

    assert result.schema["symbol"] == pl.String
    assert result.schema["timeframe"] == pl.String
    assert result.schema["open_time"] == pl.Int64
    for column in FEATURE_COLUMNS:
        assert result.schema[column] == pl.Float64
        assert result.schema[column] == COLUMN_DTYPES[column]


def test_warmup_trimming_uses_max_warmup_rows_from_metadata() -> None:
    """Warm-up removes the first max(warmup_rows) rows (default lookback - 1)."""
    pipeline, repository = _pipeline_with(_catalog_features(lookback=20))
    assert isinstance(repository, _RecordingRepository)
    result = pipeline.run(
        _base_frame(rows=25),
        FEATURE_COLUMNS,
        **_PARTITION_KWARGS,
    )

    # Default warmup_rows = lookback - 1 = 19 → 6 remain.
    assert result.height == 6
    assert result.get_column("open_time").to_list() == list(range(19, 25))
    assert repository.saved[0].height == 6


def test_warmup_trimming_with_lookback_one_keeps_all_rows() -> None:
    """Default warmup_rows for lookback 1 is 0, so no rows are removed."""
    pipeline, _repository = _pipeline_with(_catalog_features(lookback=1))
    result = pipeline.run(
        _base_frame(rows=4),
        FEATURE_COLUMNS,
        **_PARTITION_KWARGS,
    )
    assert result.height == 4


def test_warmup_trimming_uses_overridden_warmup_rows() -> None:
    """Explicit warmup_rows overrides the default lookback - 1 formula."""
    features = tuple(
        _feature(name, lookback=20, cls=_ShiftWarmupFeature) for name in FEATURE_COLUMNS
    )
    pipeline, repository = _pipeline_with(features)
    assert isinstance(repository, _RecordingRepository)
    result = pipeline.run(
        _base_frame(rows=25),
        FEATURE_COLUMNS,
        **_PARTITION_KWARGS,
    )

    # Overridden warmup_rows = 20 → 5 remain.
    assert result.height == 5
    assert result.get_column("open_time").to_list() == list(range(20, 25))
    assert repository.saved[0].height == 5


def test_warmup_trimming_mixed_catalog_uses_maximum_warmup_rows() -> None:
    """Mixed default and overridden warm-ups trim by the catalog maximum."""
    features = tuple(
        (
            _feature(name, lookback=20)
            if name != "funding_momentum"
            else _feature(
                name,
                lookback=20,
                cls=_ShiftWarmupFeature,
            )
        )
        for name in FEATURE_COLUMNS
    )
    pipeline, _repository = _pipeline_with(features)
    result = pipeline.run(
        _base_frame(rows=25),
        FEATURE_COLUMNS,
        **_PARTITION_KWARGS,
    )
    # Rolling stubs → warmup 19; momentum stub → warmup 20; trim 20.
    assert result.height == 5
    assert result.get_column("open_time").to_list() == list(range(20, 25))


def test_full_catalog_generation_has_no_residual_nulls() -> None:
    """Real feature catalog trims enough rows for zero residual NULLs."""
    from cqros.cli.generate_features import build_default_registry
    from cqros.features.schema import FEATURE_NAMES
    from cqros.features.verification import FeatureVerifier

    registry = build_default_registry()
    assert max(feature.warmup_rows for feature in registry.list()) == 20

    rows = 40
    frame = pl.DataFrame(
        {
            "symbol": [_SYMBOL] * rows,
            "timeframe": [_TIMEFRAME] * rows,
            "open_time": list(range(rows)),
            "close": [100.0 + i * 0.5 for i in range(rows)],
            "high": [101.0 + i * 0.5 for i in range(rows)],
            "low": [99.0 + i * 0.5 for i in range(rows)],
            "volume": [10.0 + i for i in range(rows)],
            "funding_rate": [0.0001 + 0.00001 * i for i in range(rows)],
            "open_interest": [1_000.0 + i for i in range(rows)],
            "buy_volume": [5.0 + i for i in range(rows)],
            "sell_volume": [4.0 + i for i in range(rows)],
            "long_short_ratio": [1.2 + 0.01 * i for i in range(rows)],
        }
    )
    pipeline, _repository = _pipeline_with(tuple(registry.list()))
    result = pipeline.run(frame, FEATURE_NAMES, **_PARTITION_KWARGS)

    assert result.height == rows - 20
    report = FeatureVerifier().verify(result)
    assert report.null_rows == 0
    assert report.passed is True
    for column in FEATURE_COLUMNS:
        assert result.get_column(column).null_count() == 0


def test_repository_save_invocation() -> None:
    """Pipeline invokes FeatureRepository.save with partition identity."""
    layout = StorageLayout(Path("unused-root"))
    datastore = _InMemoryDataStore()
    repository = FeatureRepository(layout, datastore)
    pipeline, _repo = _pipeline_with(_catalog_features(), repository=repository)

    result = pipeline.run(_base_frame(), FEATURE_COLUMNS, **_PARTITION_KWARGS)
    expected_path = layout.feature_path(
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )

    assert expected_path in datastore.frames
    assert_frame_equal(datastore.frames[expected_path], result)


def test_missing_feature_column_failure() -> None:
    """Missing required schema columns raise FeatureValidationError."""
    pipeline, repository = _pipeline_with((_feature("returns"),))
    assert isinstance(repository, _RecordingRepository)

    with pytest.raises(FeatureValidationError, match="missing required columns") as exc_info:
        pipeline.run(_base_frame(), ("returns",), **_PARTITION_KWARGS)

    error = exc_info.value
    assert error.error_code == "FEATURE-PIPE-004"
    assert "rolling_mean" in error.details["missing_columns"]
    assert error.details["required_columns"] == REQUIRED_COLUMNS
    assert repository.saved == []


def test_dependency_execution() -> None:
    """A feature's direct dependencies are executed before it."""
    _reset_execution_log()
    pipeline, _repository = _pipeline_with(
        (
            _feature("base", cls=_RecordingFeature),
            _feature("child", dependencies=("base",), cls=_RecordingFeature),
        )
    )
    order = pipeline._topological_order(("child",))
    result = pipeline._execute(_base_frame(), order)
    assert _execution_log == ["base", "child"]
    assert "base" in result.columns
    assert "child" in result.columns


def test_recursive_dependency_execution() -> None:
    """Nested dependencies are resolved recursively."""
    _reset_execution_log()
    pipeline, _repository = _pipeline_with(
        (
            _feature("a", cls=_RecordingFeature),
            _feature("b", dependencies=("a",), cls=_RecordingFeature),
            _feature("c", dependencies=("b",), cls=_RecordingFeature),
        )
    )
    order = pipeline._topological_order(("c",))
    result = pipeline._execute(_base_frame(), order)
    assert _execution_log == ["a", "b", "c"]
    assert set(result.columns) >= {"symbol", "a", "b", "c"}


def test_duplicate_requested_names_execute_once() -> None:
    """Duplicate names in the request list execute the feature once."""
    _reset_execution_log()
    pipeline, _repository = _pipeline_with((_feature("returns", cls=_RecordingFeature),))
    order = pipeline._topological_order(("returns", "returns", "returns"))
    result = pipeline._execute(_base_frame(), order)
    assert _execution_log == ["returns"]
    assert result.columns.count("returns") == 1


def test_unknown_feature_raises() -> None:
    """Missing registry entries raise UnknownFeatureError."""
    pipeline, _repository = _pipeline_with((_feature("returns"),))
    with pytest.raises(UnknownFeatureError, match="not registered"):
        pipeline.run(_base_frame(), ("missing",), **_PARTITION_KWARGS)


def test_unknown_dependency_raises() -> None:
    """Missing dependency names raise UnknownFeatureError."""
    pipeline, _repository = _pipeline_with((_feature("child", dependencies=("missing",)),))
    with pytest.raises(UnknownFeatureError, match="not registered"):
        pipeline.run(_base_frame(), ("child",), **_PARTITION_KWARGS)


def test_circular_dependency_detection() -> None:
    """Cycles in the dependency graph raise FeatureDependencyError."""
    pipeline, _repository = _pipeline_with(
        (
            _feature("a", dependencies=("b",)),
            _feature("b", dependencies=("a",)),
        )
    )
    with pytest.raises(FeatureDependencyError, match="circular"):
        pipeline.run(_base_frame(), ("a",), **_PARTITION_KWARGS)


def test_execution_order_is_deterministic() -> None:
    """Independent requested features preserve request order after deps."""
    _reset_execution_log()
    pipeline, _repository = _pipeline_with(
        (
            _feature("shared", cls=_RecordingFeature),
            _feature("left", dependencies=("shared",), cls=_RecordingFeature),
            _feature("right", dependencies=("shared",), cls=_RecordingFeature),
        )
    )
    order_left = pipeline._topological_order(("left", "right"))
    pipeline._execute(_base_frame(), order_left)
    assert _execution_log == ["shared", "left", "right"]
    _reset_execution_log()
    order_right = pipeline._topological_order(("right", "left"))
    pipeline._execute(_base_frame(), order_right)
    assert _execution_log == ["shared", "right", "left"]


def test_transform_exception_wrapping() -> None:
    """Unexpected transform errors are wrapped in FeatureExecutionError."""
    pipeline, _repository = _pipeline_with((_feature("broken", cls=_FailingFeature),))
    with pytest.raises(FeatureExecutionError, match="feature transform failed") as exc_info:
        pipeline.run(_base_frame(), ("broken",), **_PARTITION_KWARGS)
    error = exc_info.value
    assert error.error_code == "FEATURE-PIPE-003"
    assert error.details["feature"] == "broken"
    assert error.details["exception_type"] == "RuntimeError"
    assert error.details["exception_message"] == "boom"
    assert isinstance(error.__cause__, RuntimeError)


def test_input_dataframe_immutability() -> None:
    """Pipeline execution does not mutate the caller-supplied DataFrame."""
    pipeline, _repository = _pipeline_with(_catalog_features())
    original = _base_frame()
    original_columns = list(original.columns)
    original_values = original.get_column("open_time").to_list()
    result = pipeline.run(original, FEATURE_COLUMNS, **_PARTITION_KWARGS)
    assert list(original.columns) == original_columns
    assert original.get_column("open_time").to_list() == original_values
    assert "returns" not in original.columns
    assert "returns" in result.columns
    assert result is not original


def test_blank_requested_name_raises() -> None:
    """Blank requested feature names are rejected."""
    pipeline, _repository = _pipeline_with((_feature("returns"),))
    with pytest.raises(FeatureValidationError, match="non-blank"):
        pipeline.run(_base_frame(), ("",), **_PARTITION_KWARGS)


def test_package_exports_concrete_feature_pipeline() -> None:
    """Package FeaturePipeline export is the concrete orchestrator."""
    import cqros.features as features_package

    assert features_package.FeaturePipeline is FeaturePipeline
    assert "FeaturePipeline" in features_package.__all__
