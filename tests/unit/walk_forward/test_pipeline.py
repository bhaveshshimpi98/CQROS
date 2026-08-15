"""Unit tests for CQROS ``WalkForwardPipeline``."""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.walk_forward import (
    WALK_FORWARD_SCHEMA,
    SimpleWalkForwardEngine,
    WalkForwardEngineRegistry,
    WalkForwardError,
    WalkForwardPipeline,
    WalkForwardStatus,
)
from cqros.walk_forward.schema import CANONICAL_COLUMN_ORDER, COLUMN_DTYPES

_TIMEFRAME = "1h"
_FACTOR_NAME = "momentum"
_FACTOR_VERSION = "1.0.0"
_FACTOR_CATEGORY = "price"
_SELECTION_TIME = 1_704_067_200_000
_SELECTION_SCORE = 0.12


def _factor_selection_frame(
    *,
    factor_name: str = _FACTOR_NAME,
    selection_score: float = _SELECTION_SCORE,
) -> pl.DataFrame:
    """Build walk-forward evaluation input for pipeline tests.

    Includes ``future_return_1`` as produced by ``assemble_walk_forward_input``;
    the pipeline receives evaluation-enriched frames, not raw Factor Selection.
    """
    return pl.DataFrame(
        {
            "factor_name": [factor_name],
            "factor_version": [_FACTOR_VERSION],
            "timeframe": [_TIMEFRAME],
            "selection_time": [_SELECTION_TIME],
            "factor_category": [_FACTOR_CATEGORY],
            "selected": [True],
            "selection_score": [selection_score],
            "selection_rank": [1],
            "selection_reason": ["v1_default_selection"],
            "selection_ic": [0.08],
            "selected_direction": [1],
            "orientation_policy": ["signed_ic_v1"],
            "status": ["SELECTED"],
            "future_return_1": [0.01],
        }
    )


def _canonical_walk_forward_row(
    *,
    status: str = WalkForwardStatus.PASS.value,
) -> pl.DataFrame:
    """Build one canonical walk-forward row for synthetic engine outputs."""
    return pl.DataFrame(
        {
            "strategy_name": ["default_strategy"],
            "strategy_version": ["v1"],
            "timeframe": [_TIMEFRAME],
            "fold_id": [1],
            "train_start": [_SELECTION_TIME],
            "train_end": [_SELECTION_TIME],
            "test_start": [_SELECTION_TIME],
            "test_end": [_SELECTION_TIME],
            "train_rows": [1],
            "test_rows": [1],
            "selected_factors": [1],
            "model_version": ["v1"],
            "train_score": [0.0],
            "test_score": [0.0],
            "overfit_gap": [0.0],
            "status": [status],
        },
        schema=dict(COLUMN_DTYPES),
    ).select(list(CANONICAL_COLUMN_ORDER))


def _build_registry(engine_name: str = "simple") -> WalkForwardEngineRegistry:
    """Build a registry with SimpleWalkForwardEngine under engine_name."""
    registry = WalkForwardEngineRegistry()
    registry.register(engine_name, SimpleWalkForwardEngine())
    return registry


def _run(
    pipeline: WalkForwardPipeline,
    *,
    engine_name: str = "simple",
    factor_selection: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Run the pipeline with a default Factor Selection frame."""
    return pipeline.run(
        engine_name,
        (factor_selection if factor_selection is not None else _factor_selection_frame()),
    )


# ---------------------------------------------------------------------------
# Engine-name validation
# ---------------------------------------------------------------------------


def test_run_rejects_blank_engine_name() -> None:
    """Blank or whitespace engine names raise WF_PIPE_NAME_BLANK."""
    pipeline = WalkForwardPipeline(_build_registry())
    for blank in ("", "   "):
        with pytest.raises(WalkForwardError) as exc_info:
            pipeline.run(blank, _factor_selection_frame())
        assert exc_info.value.error_code == "WF_PIPE_NAME_BLANK"


def test_run_rejects_unknown_engine_name() -> None:
    """Unregistered engine name causes WF_REG_UNKNOWN from registry lookup."""
    pipeline = WalkForwardPipeline(_build_registry())
    with pytest.raises(WalkForwardError) as exc_info:
        _run(pipeline, engine_name="unknown-engine")
    assert exc_info.value.error_code == "WF_REG_UNKNOWN"


# ---------------------------------------------------------------------------
# Duplicate / missing primary key rejection
# ---------------------------------------------------------------------------


def test_run_rejects_duplicate_primary_keys() -> None:
    """Duplicate primary keys in engine output raise WF_PIPE_DUPLICATE_KEYS."""

    class _DuplicatingEngine:
        """Test engine that returns two rows with identical primary keys."""

        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            row = _canonical_walk_forward_row()
            return pl.concat([row, row])

    registry = WalkForwardEngineRegistry()
    registry.register("duplicating", _DuplicatingEngine())  # type: ignore[arg-type]
    pipeline = WalkForwardPipeline(registry)

    with pytest.raises(WalkForwardError) as exc_info:
        pipeline.run("duplicating", _factor_selection_frame())
    assert exc_info.value.error_code == "WF_PIPE_DUPLICATE_KEYS"


def test_run_rejects_missing_primary_key_columns() -> None:
    """Engine output missing primary-key columns raises WF_PIPE_MISSING_PRIMARY_KEYS."""

    class _MissingPrimaryKeyEngine:
        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            return _canonical_walk_forward_row().drop("fold_id")

    registry = WalkForwardEngineRegistry()
    registry.register("missing-pk", _MissingPrimaryKeyEngine())  # type: ignore[arg-type]
    pipeline = WalkForwardPipeline(registry)

    with pytest.raises(WalkForwardError) as exc_info:
        pipeline.run("missing-pk", _factor_selection_frame())
    assert exc_info.value.error_code == "WF_PIPE_MISSING_PRIMARY_KEYS"


# ---------------------------------------------------------------------------
# Invalid engine output types
# ---------------------------------------------------------------------------


def test_run_rejects_non_dataframe_engine_output() -> None:
    """Engine returning a non-DataFrame raises WF_PIPE_INVALID_OUTPUT."""

    class _BadEngine:
        def build(self, *args: object, **kwargs: object) -> object:
            return "not-a-dataframe"

    registry = WalkForwardEngineRegistry()
    registry.register("bad", _BadEngine())  # type: ignore[arg-type]
    pipeline = WalkForwardPipeline(registry)

    with pytest.raises(WalkForwardError) as exc_info:
        pipeline.run("bad", _factor_selection_frame())
    assert exc_info.value.error_code == "WF_PIPE_INVALID_OUTPUT"


def test_run_rejects_empty_engine_output() -> None:
    """Engine returning an empty DataFrame raises WF_PIPE_OUTPUT_EMPTY."""

    class _EmptyEngine:
        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            return pl.DataFrame({"strategy_name": []})

    registry = WalkForwardEngineRegistry()
    registry.register("empty", _EmptyEngine())  # type: ignore[arg-type]
    pipeline = WalkForwardPipeline(registry)

    with pytest.raises(WalkForwardError) as exc_info:
        pipeline.run("empty", _factor_selection_frame())
    assert exc_info.value.error_code == "WF_PIPE_OUTPUT_EMPTY"


def test_run_rejects_missing_schema_columns_in_engine_output() -> None:
    """Engine returning a DataFrame missing required columns raises WF_PIPE_MISSING_COLUMNS."""

    class _IncompleteEngine:
        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            # Include primary keys so the failure is specifically missing required fields.
            return pl.DataFrame(
                {
                    "strategy_name": ["default_strategy"],
                    "strategy_version": ["v1"],
                    "timeframe": [_TIMEFRAME],
                    "fold_id": [1],
                    "train_score": [0.0],
                }
            )

    registry = WalkForwardEngineRegistry()
    registry.register("incomplete", _IncompleteEngine())  # type: ignore[arg-type]
    pipeline = WalkForwardPipeline(registry)

    with pytest.raises(WalkForwardError) as exc_info:
        pipeline.run("incomplete", _factor_selection_frame())
    assert exc_info.value.error_code == "WF_PIPE_MISSING_COLUMNS"


def test_run_rejects_schema_cast_failure() -> None:
    """Engine output that cannot cast to WALK_FORWARD_SCHEMA raises WF_PIPE_SCHEMA_CAST."""

    class _UncastableEngine:
        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            row = _canonical_walk_forward_row()
            return row.with_columns(pl.lit("not-a-float").alias("train_score"))

    registry = WalkForwardEngineRegistry()
    registry.register("uncastable", _UncastableEngine())  # type: ignore[arg-type]
    pipeline = WalkForwardPipeline(registry)

    with pytest.raises(WalkForwardError) as exc_info:
        pipeline.run("uncastable", _factor_selection_frame())
    assert exc_info.value.error_code == "WF_PIPE_SCHEMA_CAST"


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_run_produces_canonical_walk_forward_row() -> None:
    """Pipeline with default inputs produces one canonical walk-forward output row."""
    pipeline = WalkForwardPipeline(_build_registry())
    result = _run(pipeline)
    assert result.height == 1
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result.schema == WALK_FORWARD_SCHEMA
    assert result["status"].to_list() == [WalkForwardStatus.PASS.value]
    assert result["strategy_name"].to_list() == ["default_strategy"]
    assert result["strategy_version"].to_list() == ["v1"]
    assert result["fold_id"].to_list() == [1]
    assert result["train_start"].to_list() == [_SELECTION_TIME]
    assert result["selected_factors"].to_list() == [1]
    assert result["train_score"].to_list()[0] == pytest.approx(0.01)


def test_run_resolves_engine_from_registry() -> None:
    """Pipeline resolves and executes the engine registered under engine_name."""
    registry = WalkForwardEngineRegistry()
    engine = SimpleWalkForwardEngine()
    registry.register("custom", engine)
    pipeline = WalkForwardPipeline(registry)
    result = _run(pipeline, engine_name="custom")
    assert result.schema == WALK_FORWARD_SCHEMA
    assert result.height == 1


def test_run_reorders_columns_to_canonical_order() -> None:
    """Pipeline reorders engine output columns to CANONICAL_COLUMN_ORDER."""

    class _ShuffledEngine:
        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            row = _canonical_walk_forward_row()
            shuffled = list(reversed(row.columns))
            return row.select(shuffled)

    registry = WalkForwardEngineRegistry()
    registry.register("shuffled", _ShuffledEngine())  # type: ignore[arg-type]
    pipeline = WalkForwardPipeline(registry)
    result = pipeline.run("shuffled", _factor_selection_frame())
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result.schema == WALK_FORWARD_SCHEMA


def test_run_does_not_mutate_input_frame() -> None:
    """Pipeline run must not mutate the caller-supplied Factor Selection frame."""
    pipeline = WalkForwardPipeline(_build_registry())
    factor_selection = _factor_selection_frame()
    before = factor_selection.clone()
    pipeline.run("simple", factor_selection)
    assert_frame_equal(factor_selection, before)
