"""Unit tests for CQROS ``PurgedCVPipeline``."""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.purged_cv import (
    PURGED_CV_SCHEMA,
    PurgedCVEngineRegistry,
    PurgedCVError,
    PurgedCVPipeline,
    PurgedCVStatus,
    SimplePurgedCVEngine,
)
from cqros.purged_cv.schema import CANONICAL_COLUMN_ORDER, COLUMN_DTYPES

_STRATEGY_NAME = "default_strategy"
_STRATEGY_VERSION = "v1"
_TIMEFRAME = "1h"
_HOUR_MS = 3_600_000
_BASE_TIME = 1_704_067_200_000


def _test_starts(count: int, *, start: int = _BASE_TIME) -> list[int]:
    """Build ``count`` ascending walk-forward ``test_start`` timestamps."""
    return [start + (index * _HOUR_MS) for index in range(count)]


def _walk_forward_frame(*, row_count: int = 10) -> pl.DataFrame:
    """Build a minimal Walk-Forward frame for pipeline tests."""
    test_starts = _test_starts(row_count)
    return pl.DataFrame(
        {
            "strategy_name": [_STRATEGY_NAME] * row_count,
            "strategy_version": [_STRATEGY_VERSION] * row_count,
            "timeframe": [_TIMEFRAME] * row_count,
            "fold_id": list(range(1, row_count + 1)),
            "train_start": test_starts,
            "train_end": test_starts,
            "test_start": test_starts,
            "test_end": test_starts,
            "train_rows": [10] * row_count,
            "test_rows": [5] * row_count,
            "selected_factors": [1] * row_count,
            "model_version": ["v1"] * row_count,
            "train_score": [0.10 + (0.01 * index) for index in range(row_count)],
            "test_score": [0.05 + (0.01 * index) for index in range(row_count)],
            "overfit_gap": [None] * row_count,
            "status": ["PASS"] * row_count,
        }
    )


def _canonical_purged_cv_row(
    *,
    status: str = PurgedCVStatus.PASS.value,
    fold_id: int = 1,
) -> pl.DataFrame:
    """Build one canonical purged-CV row for synthetic engine outputs."""
    return pl.DataFrame(
        {
            "strategy_name": [_STRATEGY_NAME],
            "strategy_version": [_STRATEGY_VERSION],
            "timeframe": [_TIMEFRAME],
            "fold_id": [fold_id],
            "train_start_time": [_BASE_TIME],
            "train_end_time": [_BASE_TIME + _HOUR_MS],
            "test_start_time": [_BASE_TIME + (2 * _HOUR_MS)],
            "test_end_time": [_BASE_TIME + (3 * _HOUR_MS)],
            "purge_size": [1],
            "embargo_size": [1],
            "train_rows": [2],
            "test_rows": [2],
            "train_score": [0.10],
            "test_score": [0.05],
            "overfit_gap": [0.05],
            "status": [status],
        },
        schema=dict(COLUMN_DTYPES),
    ).select(list(CANONICAL_COLUMN_ORDER))


def _build_registry(engine_name: str = "simple") -> PurgedCVEngineRegistry:
    """Build a registry with a compact SimplePurgedCVEngine under engine_name."""
    registry = PurgedCVEngineRegistry()
    registry.register(
        engine_name,
        SimplePurgedCVEngine(n_folds=2, purge_size=0, embargo_size=0),
    )
    return registry


def _build(
    pipeline: PurgedCVPipeline,
    *,
    engine: str = "simple",
    walk_forward: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build purged-CV rows through the pipeline with a default frame."""
    return pipeline.build(
        walk_forward if walk_forward is not None else _walk_forward_frame(),
        engine=engine,
    )


# ---------------------------------------------------------------------------
# Engine-name validation
# ---------------------------------------------------------------------------


def test_build_rejects_blank_engine_name() -> None:
    """Blank or whitespace engine names raise PCV_PIPE_NAME_BLANK."""
    pipeline = PurgedCVPipeline(_build_registry())
    for blank in ("", "   "):
        with pytest.raises(PurgedCVError) as exc_info:
            pipeline.build(_walk_forward_frame(), engine=blank)
        assert exc_info.value.error_code == "PCV_PIPE_NAME_BLANK"


def test_build_rejects_unknown_engine_name() -> None:
    """Unregistered engine name causes PCV_REG_UNKNOWN from registry lookup."""
    pipeline = PurgedCVPipeline(_build_registry())
    with pytest.raises(PurgedCVError) as exc_info:
        _build(pipeline, engine="unknown-engine")
    assert exc_info.value.error_code == "PCV_REG_UNKNOWN"


# ---------------------------------------------------------------------------
# Duplicate / missing primary key rejection
# ---------------------------------------------------------------------------


def test_build_rejects_duplicate_primary_keys() -> None:
    """Duplicate primary keys in engine output raise PCV_PIPE_DUPLICATE_KEYS."""

    class _DuplicatingEngine:
        """Test engine that returns two rows with identical primary keys."""

        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            row = _canonical_purged_cv_row()
            return pl.concat([row, row])

    registry = PurgedCVEngineRegistry()
    registry.register("duplicating", _DuplicatingEngine())  # type: ignore[arg-type]
    pipeline = PurgedCVPipeline(registry)

    with pytest.raises(PurgedCVError) as exc_info:
        pipeline.build(_walk_forward_frame(), engine="duplicating")
    assert exc_info.value.error_code == "PCV_PIPE_DUPLICATE_KEYS"


def test_build_rejects_missing_primary_key_columns() -> None:
    """Engine output missing primary-key columns raises PCV_PIPE_MISSING_PRIMARY_KEYS."""

    class _MissingPrimaryKeyEngine:
        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            return _canonical_purged_cv_row().drop("fold_id")

    registry = PurgedCVEngineRegistry()
    registry.register("missing-pk", _MissingPrimaryKeyEngine())  # type: ignore[arg-type]
    pipeline = PurgedCVPipeline(registry)

    with pytest.raises(PurgedCVError) as exc_info:
        pipeline.build(_walk_forward_frame(), engine="missing-pk")
    assert exc_info.value.error_code == "PCV_PIPE_MISSING_PRIMARY_KEYS"


# ---------------------------------------------------------------------------
# Invalid engine output types
# ---------------------------------------------------------------------------


def test_build_rejects_non_dataframe_engine_output() -> None:
    """Engine returning a non-DataFrame raises PCV_PIPE_INVALID_OUTPUT."""

    class _BadEngine:
        def build(self, *args: object, **kwargs: object) -> object:
            return "not-a-dataframe"

    registry = PurgedCVEngineRegistry()
    registry.register("bad", _BadEngine())  # type: ignore[arg-type]
    pipeline = PurgedCVPipeline(registry)

    with pytest.raises(PurgedCVError) as exc_info:
        pipeline.build(_walk_forward_frame(), engine="bad")
    assert exc_info.value.error_code == "PCV_PIPE_INVALID_OUTPUT"


def test_build_rejects_empty_engine_output() -> None:
    """Engine returning an empty DataFrame raises PCV_PIPE_OUTPUT_EMPTY."""

    class _EmptyEngine:
        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            return pl.DataFrame({"strategy_name": []})

    registry = PurgedCVEngineRegistry()
    registry.register("empty", _EmptyEngine())  # type: ignore[arg-type]
    pipeline = PurgedCVPipeline(registry)

    with pytest.raises(PurgedCVError) as exc_info:
        pipeline.build(_walk_forward_frame(), engine="empty")
    assert exc_info.value.error_code == "PCV_PIPE_OUTPUT_EMPTY"


def test_build_rejects_missing_schema_columns_in_engine_output() -> None:
    """Engine returning a DataFrame missing required columns raises PCV_PIPE_MISSING_COLUMNS."""

    class _IncompleteEngine:
        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            # Include primary keys so the failure is specifically missing required fields.
            return pl.DataFrame(
                {
                    "strategy_name": [_STRATEGY_NAME],
                    "strategy_version": [_STRATEGY_VERSION],
                    "timeframe": [_TIMEFRAME],
                    "fold_id": [1],
                    "train_score": [0.0],
                }
            )

    registry = PurgedCVEngineRegistry()
    registry.register("incomplete", _IncompleteEngine())  # type: ignore[arg-type]
    pipeline = PurgedCVPipeline(registry)

    with pytest.raises(PurgedCVError) as exc_info:
        pipeline.build(_walk_forward_frame(), engine="incomplete")
    assert exc_info.value.error_code == "PCV_PIPE_MISSING_COLUMNS"


def test_build_rejects_schema_cast_failure() -> None:
    """Engine output that cannot cast to PURGED_CV_SCHEMA raises PCV_PIPE_SCHEMA_CAST."""

    class _UncastableEngine:
        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            row = _canonical_purged_cv_row()
            return row.with_columns(pl.lit("not-a-float").alias("train_score"))

    registry = PurgedCVEngineRegistry()
    registry.register("uncastable", _UncastableEngine())  # type: ignore[arg-type]
    pipeline = PurgedCVPipeline(registry)

    with pytest.raises(PurgedCVError) as exc_info:
        pipeline.build(_walk_forward_frame(), engine="uncastable")
    assert exc_info.value.error_code == "PCV_PIPE_SCHEMA_CAST"


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_build_produces_canonical_purged_cv_rows() -> None:
    """Pipeline with default inputs produces canonical purged-CV output rows."""
    pipeline = PurgedCVPipeline(_build_registry())
    result = _build(pipeline)
    assert result.height == 2
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result.schema == PURGED_CV_SCHEMA
    assert result["status"].to_list() == [
        PurgedCVStatus.PASS.value,
        PurgedCVStatus.PASS.value,
    ]
    assert result["strategy_name"].to_list() == [_STRATEGY_NAME, _STRATEGY_NAME]
    assert result["strategy_version"].to_list() == [_STRATEGY_VERSION, _STRATEGY_VERSION]
    assert result["fold_id"].to_list() == [1, 2]
    assert result["purge_size"].to_list() == [0, 0]
    assert result["embargo_size"].to_list() == [0, 0]


def test_build_resolves_engine_from_registry() -> None:
    """Pipeline resolves and executes the engine registered under engine."""
    registry = PurgedCVEngineRegistry()
    engine = SimplePurgedCVEngine(n_folds=2, purge_size=0, embargo_size=0)
    registry.register("custom", engine)
    pipeline = PurgedCVPipeline(registry)
    result = _build(pipeline, engine="custom")
    assert result.schema == PURGED_CV_SCHEMA
    assert result.height == 2


def test_default_registry_registers_simple_engine() -> None:
    """PurgedCVPipeline() without a registry resolves the default simple engine."""
    pipeline = PurgedCVPipeline()
    result = pipeline.build(_walk_forward_frame(row_count=20))
    assert result.schema == PURGED_CV_SCHEMA
    assert result.height == 5
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER


def test_build_reorders_columns_to_canonical_order() -> None:
    """Pipeline reorders engine output columns to CANONICAL_COLUMN_ORDER."""

    class _ShuffledEngine:
        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            row = _canonical_purged_cv_row()
            shuffled = list(reversed(row.columns))
            return row.select(shuffled)

    registry = PurgedCVEngineRegistry()
    registry.register("shuffled", _ShuffledEngine())  # type: ignore[arg-type]
    pipeline = PurgedCVPipeline(registry)
    result = pipeline.build(_walk_forward_frame(), engine="shuffled")
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result.schema == PURGED_CV_SCHEMA


def test_build_does_not_mutate_input_frame() -> None:
    """Pipeline build must not mutate the caller-supplied Walk-Forward frame."""
    pipeline = PurgedCVPipeline(_build_registry())
    walk_forward = _walk_forward_frame()
    before = walk_forward.clone()
    pipeline.build(walk_forward, engine="simple")
    assert_frame_equal(walk_forward, before)


def test_build_output_is_deterministic() -> None:
    """Identical Walk-Forward inputs produce identical purged-CV outputs."""
    pipeline = PurgedCVPipeline(_build_registry())
    walk_forward = _walk_forward_frame()
    first = pipeline.build(walk_forward, engine="simple")
    second = pipeline.build(walk_forward, engine="simple")
    assert_frame_equal(first, second)
