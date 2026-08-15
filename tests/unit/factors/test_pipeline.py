"""Unit tests for CQROS Factor Research Engine ``FactorPipeline``."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.factors.base import BaseFactor
from cqros.factors.default_registry import build_default_registry
from cqros.factors.exceptions import (
    FactorExecutionError,
    FactorRegistrationError,
    FactorValidationError,
)
from cqros.factors.interfaces import Factor
from cqros.factors.interfaces import FactorPipeline as FactorPipelineProtocol
from cqros.factors.pipeline import FactorPipeline
from cqros.factors.price import MomentumFactor
from cqros.factors.registry import FactorRegistry
from cqros.factors.schema import PRIMARY_KEY_COLUMNS
from cqros.factors.volume import RelativeVolumeFactor

_PRODUCTION_FACTOR_COUNT = 111
_ROW_COUNT = 24


@dataclass(frozen=True, slots=True)
class _StubFactor(BaseFactor):
    """Minimal deterministic factor used only for pipeline unit tests."""

    name: str = "stub_alpha"
    version: str = "1.0.0"
    description: str = "Stub factor for pipeline tests"
    category: str = "test"
    required_features: tuple[str, ...] = ("close",)
    produced_columns: tuple[str, ...] = ("stub_alpha",)
    lookback: int = 0

    def __post_init__(self) -> None:
        BaseFactor.__post_init__(self)

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(pl.col("close").alias(self.produced_columns[0]))


@dataclass(frozen=True, slots=True)
class _SecondStubFactor(BaseFactor):
    """Second stub factor with a distinct produced column."""

    name: str = "stub_beta"
    version: str = "1.0.0"
    description: str = "Second stub factor for pipeline tests"
    category: str = "test"
    required_features: tuple[str, ...] = ("close",)
    produced_columns: tuple[str, ...] = ("stub_beta",)
    lookback: int = 0

    def __post_init__(self) -> None:
        BaseFactor.__post_init__(self)

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns((pl.col("close") * 2.0).alias(self.produced_columns[0]))


@dataclass(frozen=True, slots=True)
class _PrimaryKeyCollidingFactor(BaseFactor):
    """Stub that produces a primary-key column name to force merge conflicts."""

    name: str = "pk_collision"
    version: str = "1.0.0"
    description: str = "Produces symbol to collide with primary keys"
    category: str = "test"
    required_features: tuple[str, ...] = ("close",)
    produced_columns: tuple[str, ...] = ("symbol",)
    lookback: int = 0

    def __post_init__(self) -> None:
        BaseFactor.__post_init__(self)

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(pl.lit("COLLIDE").alias("symbol"))


@dataclass(frozen=True, slots=True)
class _FailingFactor(BaseFactor):
    """Stub whose compute always raises."""

    name: str = "failing"
    version: str = "1.0.0"
    description: str = "Always fails"
    category: str = "test"
    required_features: tuple[str, ...] = ("close",)
    produced_columns: tuple[str, ...] = ("failing",)
    lookback: int = 0

    def __post_init__(self) -> None:
        BaseFactor.__post_init__(self)

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        raise RuntimeError("intentional compute failure")


def _synthetic_training_frame(
    *,
    required_features: Sequence[str] | None = None,
    row_count: int = _ROW_COUNT,
) -> pl.DataFrame:
    """Build a small synthetic training frame for pipeline tests."""
    data: dict[str, object] = {
        "symbol": ["BTCUSDT"] * row_count,
        "timeframe": ["1h"] * row_count,
        "open_time": list(range(row_count)),
    }
    features = (
        tuple(required_features)
        if required_features is not None
        else tuple(
            sorted(
                {
                    feature
                    for factor in build_default_registry().list()
                    for feature in factor.required_features
                }
            )
        )
    )
    for feature in features:
        data[feature] = [float((index % 10) + 1) for index in range(row_count)]
    return pl.DataFrame(data)


def _registry_with(*factors: Factor) -> FactorRegistry:
    """Return a registry containing only the supplied factors."""
    registry = FactorRegistry()
    registry.register_many(factors)
    return registry


class _ListOverrideRegistry(FactorRegistry):
    """FactorRegistry test double that returns a fixed ``list`` result."""

    __slots__ = ("_override",)

    def __init__(self, factors: tuple[Factor, ...]) -> None:
        super().__init__()
        self._override = factors

    def list(self) -> tuple[Factor, ...]:
        return self._override


def test_pipeline_loads_production_registry_by_default() -> None:
    """FactorPipeline loads the production catalog when no registry is given."""
    pipeline = FactorPipeline()
    assert isinstance(pipeline, FactorPipelineProtocol)
    assert len(pipeline.registry.list()) == _PRODUCTION_FACTOR_COUNT
    assert set(pipeline.registry.names()) == set(build_default_registry().names())


def test_pipeline_executes_all_production_factors_and_merges() -> None:
    """Production catalog execution returns PK columns plus every factor column."""
    registry = build_default_registry()
    pipeline = FactorPipeline(registry)
    frame = _synthetic_training_frame()
    original_columns = list(frame.columns)
    original_close = frame.get_column("close").to_list()

    result = pipeline.run(frame)

    expected_factor_columns = [
        column for factor in registry.list() for column in factor.produced_columns
    ]
    assert result.columns == [*PRIMARY_KEY_COLUMNS, *expected_factor_columns]
    assert result.height == frame.height
    assert set(PRIMARY_KEY_COLUMNS).issubset(result.columns)
    assert all(column in result.columns for column in expected_factor_columns)
    assert len(result.columns) == len(PRIMARY_KEY_COLUMNS) + len(expected_factor_columns)
    assert list(frame.columns) == original_columns
    assert frame.get_column("close").to_list() == original_close


def test_pipeline_merge_is_deterministic() -> None:
    """Repeated runs with identical inputs produce identical column order and values."""
    registry = _registry_with(MomentumFactor(), RelativeVolumeFactor())
    pipeline = FactorPipeline(registry)
    frame = _synthetic_training_frame(
        required_features=("close", "volume"),
        row_count=12,
    )
    first = pipeline.run(frame)
    second = FactorPipeline(registry).run(frame)
    assert first.columns == second.columns
    assert_frame_equal(first, second)


def test_pipeline_column_ordering_follows_registry_order() -> None:
    """Merged factor columns follow alphabetical registry factor order."""
    registry = _registry_with(RelativeVolumeFactor(), MomentumFactor())
    pipeline = FactorPipeline(registry)
    frame = _synthetic_training_frame(required_features=("close", "volume"), row_count=8)
    result = pipeline.run(frame)
    assert result.columns == [
        "symbol",
        "timeframe",
        "open_time",
        "momentum",
        "relative_volume",
    ]


def test_pipeline_successful_merge_with_stub_factors() -> None:
    """Independent stub factors merge onto primary keys without chaining."""
    registry = _registry_with(_StubFactor(), _SecondStubFactor())
    pipeline = FactorPipeline(registry)
    frame = _synthetic_training_frame(required_features=("close",), row_count=5)
    result = pipeline.run(frame)
    assert result.columns == ["symbol", "timeframe", "open_time", "stub_alpha", "stub_beta"]
    assert result.get_column("stub_alpha").to_list() == frame.get_column("close").to_list()
    assert result.get_column("stub_beta").to_list() == [
        value * 2.0 for value in frame.get_column("close").to_list()
    ]


def test_pipeline_detects_missing_required_features() -> None:
    """Missing required features fail before factor execution."""
    registry = _registry_with(MomentumFactor())
    pipeline = FactorPipeline(registry)
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timeframe": ["1h"],
            "open_time": [0],
            "volume": [1.0],
        }
    )
    with pytest.raises(
        FactorValidationError,
        match="training frame is missing required factor features",
    ) as exc_info:
        pipeline.run(frame)
    error = exc_info.value
    assert error.error_code == "FACTOR-PIPE-002"
    assert error.details["missing_columns"] == ("close",)


def test_pipeline_detects_missing_primary_keys() -> None:
    """Missing primary-key columns fail immediately."""
    registry = _registry_with(_StubFactor())
    pipeline = FactorPipeline(registry)
    frame = pl.DataFrame({"close": [1.0, 2.0], "open_time": [0, 1]})
    with pytest.raises(
        FactorValidationError,
        match="training frame is missing primary key columns",
    ) as exc_info:
        pipeline.run(frame)
    error = exc_info.value
    assert error.error_code == "FACTOR-PIPE-001"
    missing_columns = error.details["missing_columns"]
    assert isinstance(missing_columns, tuple)
    assert "symbol" in missing_columns
    assert "timeframe" in missing_columns


def test_pipeline_duplicate_produced_column_protection() -> None:
    """Duplicate produced columns in the catalog fail before execution."""
    first = _StubFactor()
    colliding = _StubFactor(
        name="stub_beta_conflict",
        produced_columns=("stub_alpha",),
    )
    pipeline = FactorPipeline(_ListOverrideRegistry((first, colliding)))
    frame = _synthetic_training_frame(required_features=("close",), row_count=3)
    with pytest.raises(
        FactorRegistrationError,
        match="duplicate produced column in catalog: stub_alpha",
    ) as exc_info:
        pipeline.run(frame)
    assert exc_info.value.error_code == "FACTOR-PIPE-004"


def test_pipeline_duplicate_factor_name_protection() -> None:
    """Duplicate factor names in the catalog fail before execution."""
    factor = _StubFactor()
    pipeline = FactorPipeline(_ListOverrideRegistry((factor, factor)))
    frame = _synthetic_training_frame(required_features=("close",), row_count=3)
    with pytest.raises(
        FactorRegistrationError,
        match="duplicate factor name in catalog: stub_alpha",
    ) as exc_info:
        pipeline.run(frame)
    assert exc_info.value.error_code == "FACTOR-PIPE-003"


def test_pipeline_merged_duplicate_column_protection() -> None:
    """Merged frames reject produced columns that collide with primary keys."""
    registry = _registry_with(_PrimaryKeyCollidingFactor())
    pipeline = FactorPipeline(registry)
    frame = _synthetic_training_frame(required_features=("close",), row_count=3)
    with pytest.raises(
        FactorValidationError,
        match="merged factor matrix contains duplicate column names",
    ) as exc_info:
        pipeline.run(frame)
    assert exc_info.value.error_code == "FACTOR-PIPE-007"
    duplicate_columns = exc_info.value.details["duplicate_columns"]
    assert isinstance(duplicate_columns, tuple)
    assert "symbol" in duplicate_columns


def test_pipeline_wraps_factor_execution_failures() -> None:
    """Compute exceptions are wrapped as FactorExecutionError."""
    registry = _registry_with(_FailingFactor())
    pipeline = FactorPipeline(registry)
    frame = _synthetic_training_frame(required_features=("close",), row_count=3)
    with pytest.raises(FactorExecutionError, match="factor compute failed: failing") as exc_info:
        pipeline.run(frame)
    error = exc_info.value
    assert error.error_code == "FACTOR-PIPE-005"
    assert error.details["factor"] == "failing"
    assert error.details["exception_type"] == "RuntimeError"
