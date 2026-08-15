"""Unit tests for CQROS ``FactorValidationPipeline``."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.core.constants import EXCHANGE_BINANCE, MARKET_USDT_PERPETUAL
from cqros.factor_validation import (
    FACTOR_VALIDATION_SCHEMA,
    FactorValidationEngineRegistry,
    FactorValidationError,
    FactorValidationExecutionConfig,
    FactorValidationExecutionMode,
    FactorValidationPipeline,
    FactorValidationStatus,
    SimpleFactorValidationEngine,
)
from cqros.factor_validation.schema import CANONICAL_COLUMN_ORDER
from cqros.factors import FactorStatus

_TIMEFRAME = "1h"
_SYMBOL = "BTCUSDT"
_MANAGER = "default"
_EXCHANGE = EXCHANGE_BINANCE
_MARKET = MARKET_USDT_PERPETUAL
_YEAR = 2026
_FACTOR_NAME = "momentum"
_FACTOR_VERSION = "1.0.0"
_FACTOR_CATEGORY = "price"
_DATASET_VERSION = "dataset-v1"
_LABEL_VERSION = "label-v1"
_OPEN_TIME = datetime(2024, 1, 1, tzinfo=UTC)
_OPEN_TIME_MS = int(_OPEN_TIME.timestamp() * 1000.0)
_VALIDATION_START_TIME = _OPEN_TIME_MS
_VALIDATION_END_TIME = _OPEN_TIME_MS

_PARTITION_KWARGS = {
    "manager": _MANAGER,
    "exchange": _EXCHANGE,
    "market": _MARKET,
    "timeframe": _TIMEFRAME,
    "year": _YEAR,
}


def _validation_dataset(
    *,
    symbol: str = _SYMBOL,
    factor_name: str = _FACTOR_NAME,
) -> pl.DataFrame:
    """Build an assembled validation dataset expected by the engine."""
    return pl.DataFrame(
        {
            "symbol": [symbol],
            "timeframe": [_TIMEFRAME],
            "open_time": [_OPEN_TIME],
            "factor_name": [factor_name],
            "factor_version": [_FACTOR_VERSION],
            "factor_category": [_FACTOR_CATEGORY],
            "factor_group": ["alpha"],
            "factor_value": [0.1],
            "lookback": [20],
            "prediction_horizon": [1],
            "enabled": [True],
            "status": [FactorStatus.ACTIVE.value],
            "future_return_1": [0.2],
        }
    )


def _canonical_validation_row(
    *,
    status: str = FactorValidationStatus.PASS.value,
) -> pl.DataFrame:
    """Build one canonical factor-validation row for synthetic engine outputs."""
    return pl.DataFrame(
        {
            "factor_name": [_FACTOR_NAME],
            "factor_version": [_FACTOR_VERSION],
            "timeframe": [_TIMEFRAME],
            "validation_time": [_OPEN_TIME_MS],
            "factor_category": [_FACTOR_CATEGORY],
            "dataset_version": [_DATASET_VERSION],
            "label_version": [_LABEL_VERSION],
            "validation_start_time": [_VALIDATION_START_TIME],
            "validation_end_time": [_VALIDATION_END_TIME],
            "information_coefficient": [0.0],
            "rank_information_coefficient": [0.0],
            "ic_information_ratio": [0.0],
            "ic_std": [0.0],
            "ic_p_value": [1.0],
            "ic_t_stat": [0.0],
            "ic_decay": [0.0],
            "turnover": [0.0],
            "monotonicity_score": [0.0],
            "quantile_spread": [0.0],
            "observations": [1],
            "ic_observations": [1],
            "status": [status],
        },
        schema=FACTOR_VALIDATION_SCHEMA,
    ).select(list(CANONICAL_COLUMN_ORDER))


class _StubBuilder:
    """Test double that returns a fixed assembled validation dataset."""

    def __init__(self, frame: pl.DataFrame | None = None) -> None:
        self.frame = frame if frame is not None else _validation_dataset()
        self.build_calls: list[dict[str, Any]] = []

    def build(self, **kwargs: Any) -> pl.DataFrame:
        self.build_calls.append(dict(kwargs))
        return self.frame


class _RecordingEngine:
    """Engine test double that records the assembled dataset it receives."""

    def __init__(self, output: pl.DataFrame | None = None) -> None:
        self.received: pl.DataFrame | None = None
        self._output = output if output is not None else _canonical_validation_row()

    def build(self, factors: pl.DataFrame) -> pl.DataFrame:
        self.received = factors.clone()
        return self._output


_FULL_PANEL_CONFIG = FactorValidationExecutionConfig(mode=FactorValidationExecutionMode.FULL_PANEL)


def _pipeline(
    registry: FactorValidationEngineRegistry,
    builder: object,
) -> FactorValidationPipeline:
    """Build a pipeline using full-panel mode for stub builders."""
    return FactorValidationPipeline(
        registry,
        builder,  # type: ignore[arg-type]
        execution_config=_FULL_PANEL_CONFIG,
    )


def _build_registry(engine_name: str = "simple") -> FactorValidationEngineRegistry:
    """Build a registry with SimpleFactorValidationEngine under engine_name."""
    registry = FactorValidationEngineRegistry()
    registry.register(engine_name, SimpleFactorValidationEngine())
    return registry


def _run(
    pipeline: FactorValidationPipeline,
    *,
    engine_name: str = "simple",
) -> pl.DataFrame:
    """Run the pipeline with default partition identity."""
    return pipeline.run(engine_name, **_PARTITION_KWARGS)


# ---------------------------------------------------------------------------
# Engine-name validation
# ---------------------------------------------------------------------------


def test_run_rejects_blank_engine_name() -> None:
    """Blank or whitespace engine names raise FVAL_PIPE_NAME_BLANK."""
    pipeline = _pipeline(_build_registry(), _StubBuilder())
    for blank in ("", "   "):
        with pytest.raises(FactorValidationError) as exc_info:
            pipeline.run(blank, **_PARTITION_KWARGS)
        assert exc_info.value.error_code == "FVAL_PIPE_NAME_BLANK"


def test_run_rejects_unknown_engine_name() -> None:
    """Unregistered engine name causes FVAL_REG_UNKNOWN from registry lookup."""
    pipeline = _pipeline(_build_registry(), _StubBuilder())
    with pytest.raises(FactorValidationError) as exc_info:
        _run(pipeline, engine_name="unknown-engine")
    assert exc_info.value.error_code == "FVAL_REG_UNKNOWN"


# ---------------------------------------------------------------------------
# Builder integration
# ---------------------------------------------------------------------------


def test_run_calls_builder_before_engine() -> None:
    """Pipeline assembles the validation dataset through the builder first."""
    builder = _StubBuilder(_validation_dataset())
    engine = _RecordingEngine()
    registry = FactorValidationEngineRegistry()
    registry.register("recording", engine)  # type: ignore[arg-type]
    pipeline = _pipeline(registry, builder)

    result = pipeline.run("recording", **_PARTITION_KWARGS)

    assert builder.build_calls == [{**_PARTITION_KWARGS, "symbols": None}]
    assert engine.received is not None
    assert_frame_equal(engine.received, builder.frame)
    assert result.schema == FACTOR_VALIDATION_SCHEMA


def test_run_passes_assembled_dataset_to_engine() -> None:
    """Engine receives the builder-assembled frame, not a raw Factors frame."""
    assembled = _validation_dataset().with_columns(pl.lit(0.55).alias("future_return_1"))
    builder = _StubBuilder(assembled)
    engine = _RecordingEngine()
    registry = FactorValidationEngineRegistry()
    registry.register("recording", engine)  # type: ignore[arg-type]
    pipeline = _pipeline(registry, builder)

    pipeline.run("recording", **_PARTITION_KWARGS)

    assert engine.received is not None
    assert "future_return_1" in engine.received.columns
    assert engine.received.get_column("future_return_1").to_list() == [0.55]


# ---------------------------------------------------------------------------
# Duplicate / missing primary key rejection
# ---------------------------------------------------------------------------


def test_run_rejects_duplicate_primary_keys() -> None:
    """Duplicate primary keys in engine output raise FVAL_PIPE_DUPLICATE_KEYS."""

    class _DuplicatingEngine:
        """Test engine that returns two rows with identical primary keys."""

        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            row = _canonical_validation_row()
            return pl.concat([row, row])

    registry = FactorValidationEngineRegistry()
    registry.register("duplicating", _DuplicatingEngine())  # type: ignore[arg-type]
    pipeline = _pipeline(registry, _StubBuilder())

    with pytest.raises(FactorValidationError) as exc_info:
        pipeline.run("duplicating", **_PARTITION_KWARGS)
    assert exc_info.value.error_code == "FVAL_PIPE_DUPLICATE_KEYS"


def test_run_rejects_missing_primary_key_columns() -> None:
    """Engine output missing primary-key columns raises FVAL_PIPE_MISSING_PRIMARY_KEYS."""

    class _MissingPrimaryKeyEngine:
        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            return _canonical_validation_row().drop("validation_time")

    registry = FactorValidationEngineRegistry()
    registry.register("missing-pk", _MissingPrimaryKeyEngine())  # type: ignore[arg-type]
    pipeline = _pipeline(registry, _StubBuilder())

    with pytest.raises(FactorValidationError) as exc_info:
        pipeline.run("missing-pk", **_PARTITION_KWARGS)
    assert exc_info.value.error_code == "FVAL_PIPE_MISSING_PRIMARY_KEYS"


# ---------------------------------------------------------------------------
# Invalid engine output types
# ---------------------------------------------------------------------------


def test_run_rejects_non_dataframe_engine_output() -> None:
    """Engine returning a non-DataFrame raises FVAL_PIPE_INVALID_OUTPUT."""

    class _BadEngine:
        def build(self, *args: object, **kwargs: object) -> object:
            return "not-a-dataframe"

    registry = FactorValidationEngineRegistry()
    registry.register("bad", _BadEngine())  # type: ignore[arg-type]
    pipeline = _pipeline(registry, _StubBuilder())

    with pytest.raises(FactorValidationError) as exc_info:
        pipeline.run("bad", **_PARTITION_KWARGS)
    assert exc_info.value.error_code == "FVAL_PIPE_INVALID_OUTPUT"


def test_run_rejects_empty_engine_output() -> None:
    """Engine returning an empty DataFrame raises FVAL_PIPE_OUTPUT_EMPTY."""

    class _EmptyEngine:
        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            return pl.DataFrame({"factor_name": []})

    registry = FactorValidationEngineRegistry()
    registry.register("empty", _EmptyEngine())  # type: ignore[arg-type]
    pipeline = _pipeline(registry, _StubBuilder())

    with pytest.raises(FactorValidationError) as exc_info:
        pipeline.run("empty", **_PARTITION_KWARGS)
    assert exc_info.value.error_code == "FVAL_PIPE_OUTPUT_EMPTY"


def test_run_rejects_missing_schema_columns_in_engine_output() -> None:
    """Engine returning a DataFrame missing required columns raises FVAL_PIPE_MISSING_COLUMNS."""

    class _IncompleteEngine:
        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            return pl.DataFrame(
                {
                    "factor_name": [_FACTOR_NAME],
                    "factor_version": [_FACTOR_VERSION],
                    "timeframe": [_TIMEFRAME],
                    "validation_time": [_OPEN_TIME_MS],
                    "information_coefficient": [0.0],
                }
            )

    registry = FactorValidationEngineRegistry()
    registry.register("incomplete", _IncompleteEngine())  # type: ignore[arg-type]
    pipeline = _pipeline(registry, _StubBuilder())

    with pytest.raises(FactorValidationError) as exc_info:
        pipeline.run("incomplete", **_PARTITION_KWARGS)
    assert exc_info.value.error_code == "FVAL_PIPE_MISSING_COLUMNS"


def test_run_rejects_schema_cast_failure() -> None:
    """Engine output that cannot cast to FACTOR_VALIDATION_SCHEMA raises FVAL_PIPE_SCHEMA_CAST."""

    class _UncastableEngine:
        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            row = _canonical_validation_row()
            return row.with_columns(pl.lit("not-a-float").alias("information_coefficient"))

    registry = FactorValidationEngineRegistry()
    registry.register("uncastable", _UncastableEngine())  # type: ignore[arg-type]
    pipeline = _pipeline(registry, _StubBuilder())

    with pytest.raises(FactorValidationError) as exc_info:
        pipeline.run("uncastable", **_PARTITION_KWARGS)
    assert exc_info.value.error_code == "FVAL_PIPE_SCHEMA_CAST"


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_run_produces_canonical_validation_row() -> None:
    """Pipeline with default inputs produces one canonical validation output row."""
    pipeline = _pipeline(_build_registry(), _StubBuilder())
    result = _run(pipeline)
    assert result.height == 1
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result.schema == FACTOR_VALIDATION_SCHEMA
    assert result["status"].to_list() == [FactorValidationStatus.FAIL.value]
    assert result["validation_time"].to_list() == [_OPEN_TIME_MS]
    assert result["information_coefficient"].to_list() == [None]
    assert result["observations"].to_list() == [1]
    assert result["dataset_version"].to_list() == ["default"]
    assert result["label_version"].to_list() == ["default"]
    assert result["validation_start_time"].to_list() == [_VALIDATION_START_TIME]
    assert result["validation_end_time"].to_list() == [_VALIDATION_END_TIME]
    assert result["ic_observations"].to_list() == [None]
    assert result["ic_std"].to_list() == [None]


def test_run_resolves_engine_from_registry() -> None:
    """Pipeline resolves and executes the engine registered under engine_name."""
    registry = FactorValidationEngineRegistry()
    engine = SimpleFactorValidationEngine()
    registry.register("custom", engine)
    pipeline = _pipeline(registry, _StubBuilder())
    result = _run(pipeline, engine_name="custom")
    assert result.schema == FACTOR_VALIDATION_SCHEMA
    assert result.height == 1


def test_run_reorders_columns_to_canonical_order() -> None:
    """Pipeline reorders engine output columns to CANONICAL_COLUMN_ORDER."""

    class _ShuffledEngine:
        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            row = _canonical_validation_row()
            shuffled = list(reversed(row.columns))
            return row.select(shuffled)

    registry = FactorValidationEngineRegistry()
    registry.register("shuffled", _ShuffledEngine())  # type: ignore[arg-type]
    pipeline = _pipeline(registry, _StubBuilder())
    result = pipeline.run("shuffled", **_PARTITION_KWARGS)
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result.schema == FACTOR_VALIDATION_SCHEMA
