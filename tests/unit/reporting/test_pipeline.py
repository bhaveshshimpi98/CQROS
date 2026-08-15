"""Unit tests for CQROS ``ReportingPipeline``."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.reporting import (
    REPORTING_SCHEMA,
    ReportingEngineRegistry,
    ReportingPipeline,
    ReportingStatus,
    ReportingValidationError,
    SimpleReportingEngine,
)
from cqros.reporting.schema import CANONICAL_COLUMN_ORDER, COLUMN_DTYPES

_TIMEFRAME = "1h"
_MANAGER = "simple"
_OPEN_TIME = datetime(2024, 1, 1, tzinfo=UTC)
_OPEN_TIME_MS = int(_OPEN_TIME.timestamp() * 1000.0)


def _analytics_frame(
    *,
    symbol: str = "BTCUSDT",
    manager: str = _MANAGER,
) -> pl.DataFrame:
    """Build a minimal analytics frame for pipeline tests."""
    return pl.DataFrame(
        {
            "symbol": [symbol],
            "timeframe": [_TIMEFRAME],
            "open_time": [_OPEN_TIME],
            "manager": [manager],
        }
    )


def _canonical_reporting_row(*, status: str = ReportingStatus.GENERATED.value) -> pl.DataFrame:
    """Build one canonical reporting row for synthetic engine outputs."""
    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timeframe": [_TIMEFRAME],
            "open_time": [_OPEN_TIME_MS],
            "manager": [_MANAGER],
            "report_name": ["performance_report"],
            "report_type": ["analytics"],
            "report_format": ["parquet"],
            "report_version": ["v1"],
            "report_path": [""],
            "generated_at": [_OPEN_TIME_MS],
            "status": [status],
        },
        schema=dict(COLUMN_DTYPES),
    ).select(list(CANONICAL_COLUMN_ORDER))


def _build_registry(engine_name: str = "simple") -> ReportingEngineRegistry:
    """Build a registry with SimpleReportingEngine under engine_name."""
    registry = ReportingEngineRegistry()
    registry.register(engine_name, SimpleReportingEngine())
    return registry


def _run(
    pipeline: ReportingPipeline,
    *,
    engine_name: str = "simple",
    analytics: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Run the pipeline with a default analytics frame."""
    return pipeline.run(
        engine_name,
        analytics if analytics is not None else _analytics_frame(),
    )


# ---------------------------------------------------------------------------
# Engine-name validation
# ---------------------------------------------------------------------------


def test_run_rejects_blank_engine_name() -> None:
    """Blank or whitespace engine names raise REP_PIPE_NAME_BLANK."""
    pipeline = ReportingPipeline(_build_registry())
    for blank in ("", "   "):
        with pytest.raises(ReportingValidationError) as exc_info:
            pipeline.run(blank, _analytics_frame())
        assert exc_info.value.error_code == "REP_PIPE_NAME_BLANK"


def test_run_rejects_unknown_engine_name() -> None:
    """Unregistered engine name causes REP_REG_UNKNOWN from registry lookup."""
    pipeline = ReportingPipeline(_build_registry())
    with pytest.raises(ReportingValidationError) as exc_info:
        _run(pipeline, engine_name="unknown-engine")
    assert exc_info.value.error_code == "REP_REG_UNKNOWN"


# ---------------------------------------------------------------------------
# Duplicate / missing primary key rejection
# ---------------------------------------------------------------------------


def test_run_rejects_duplicate_primary_keys() -> None:
    """Duplicate primary keys in engine output raise REP_PIPE_DUPLICATE_KEYS."""

    class _DuplicatingEngine:
        """Test engine that returns two rows with identical primary keys."""

        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            row = _canonical_reporting_row()
            return pl.concat([row, row])

    registry = ReportingEngineRegistry()
    registry.register("duplicating", _DuplicatingEngine())  # type: ignore[arg-type]
    pipeline = ReportingPipeline(registry)

    with pytest.raises(ReportingValidationError) as exc_info:
        pipeline.run("duplicating", _analytics_frame())
    assert exc_info.value.error_code == "REP_PIPE_DUPLICATE_KEYS"


def test_run_rejects_missing_primary_key_columns() -> None:
    """Engine output missing primary-key columns raises REP_PIPE_MISSING_PRIMARY_KEYS."""

    class _MissingPrimaryKeyEngine:
        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            row = _canonical_reporting_row().drop("open_time")
            return row

    registry = ReportingEngineRegistry()
    registry.register("missing-pk", _MissingPrimaryKeyEngine())  # type: ignore[arg-type]
    pipeline = ReportingPipeline(registry)

    with pytest.raises(ReportingValidationError) as exc_info:
        pipeline.run("missing-pk", _analytics_frame())
    assert exc_info.value.error_code == "REP_PIPE_MISSING_PRIMARY_KEYS"


# ---------------------------------------------------------------------------
# Invalid engine output types
# ---------------------------------------------------------------------------


def test_run_rejects_non_dataframe_engine_output() -> None:
    """Engine returning a non-DataFrame raises REP_PIPE_INVALID_OUTPUT."""

    class _BadEngine:
        def build(self, *args: object, **kwargs: object) -> object:
            return "not-a-dataframe"

    registry = ReportingEngineRegistry()
    registry.register("bad", _BadEngine())  # type: ignore[arg-type]
    pipeline = ReportingPipeline(registry)

    with pytest.raises(ReportingValidationError) as exc_info:
        pipeline.run("bad", _analytics_frame())
    assert exc_info.value.error_code == "REP_PIPE_INVALID_OUTPUT"


def test_run_rejects_empty_engine_output() -> None:
    """Engine returning an empty DataFrame raises REP_PIPE_OUTPUT_EMPTY."""

    class _EmptyEngine:
        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            return pl.DataFrame({"symbol": []})

    registry = ReportingEngineRegistry()
    registry.register("empty", _EmptyEngine())  # type: ignore[arg-type]
    pipeline = ReportingPipeline(registry)

    with pytest.raises(ReportingValidationError) as exc_info:
        pipeline.run("empty", _analytics_frame())
    assert exc_info.value.error_code == "REP_PIPE_OUTPUT_EMPTY"


def test_run_rejects_missing_schema_columns_in_engine_output() -> None:
    """Engine returning a DataFrame missing required columns raises REP_PIPE_MISSING_COLUMNS."""

    class _IncompleteEngine:
        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            # Include primary keys so the failure is specifically missing required fields.
            return pl.DataFrame(
                {
                    "symbol": ["BTCUSDT"],
                    "timeframe": [_TIMEFRAME],
                    "open_time": [_OPEN_TIME_MS],
                    "report_name": ["performance_report"],
                }
            )

    registry = ReportingEngineRegistry()
    registry.register("incomplete", _IncompleteEngine())  # type: ignore[arg-type]
    pipeline = ReportingPipeline(registry)

    with pytest.raises(ReportingValidationError) as exc_info:
        pipeline.run("incomplete", _analytics_frame())
    assert exc_info.value.error_code == "REP_PIPE_MISSING_COLUMNS"


def test_run_rejects_schema_cast_failure() -> None:
    """Engine output that cannot cast to REPORTING_SCHEMA raises REP_PIPE_SCHEMA_CAST."""

    class _UncastableEngine:
        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            row = _canonical_reporting_row()
            return row.with_columns(pl.lit("not-an-int").alias("generated_at"))

    registry = ReportingEngineRegistry()
    registry.register("uncastable", _UncastableEngine())  # type: ignore[arg-type]
    pipeline = ReportingPipeline(registry)

    with pytest.raises(ReportingValidationError) as exc_info:
        pipeline.run("uncastable", _analytics_frame())
    assert exc_info.value.error_code == "REP_PIPE_SCHEMA_CAST"


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_run_produces_canonical_reporting_row() -> None:
    """Pipeline with default inputs produces one canonical reporting output row."""
    pipeline = ReportingPipeline(_build_registry())
    result = _run(pipeline)
    assert result.height == 1
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result.schema == REPORTING_SCHEMA
    assert result["status"].to_list() == [ReportingStatus.GENERATED.value]
    assert result["open_time"].to_list() == [_OPEN_TIME_MS]
    assert result["generated_at"].to_list() == [_OPEN_TIME_MS]
    assert result["report_name"].to_list() == ["performance_report"]
    assert result["report_type"].to_list() == ["analytics"]


def test_run_resolves_engine_from_registry() -> None:
    """Pipeline resolves and executes the engine registered under engine_name."""
    registry = ReportingEngineRegistry()
    engine = SimpleReportingEngine()
    registry.register("custom", engine)
    pipeline = ReportingPipeline(registry)
    result = _run(pipeline, engine_name="custom")
    assert result.schema == REPORTING_SCHEMA
    assert result.height == 1


def test_run_reorders_columns_to_canonical_order() -> None:
    """Pipeline reorders engine output columns to CANONICAL_COLUMN_ORDER."""

    class _ShuffledEngine:
        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            row = _canonical_reporting_row()
            shuffled = list(reversed(row.columns))
            return row.select(shuffled)

    registry = ReportingEngineRegistry()
    registry.register("shuffled", _ShuffledEngine())  # type: ignore[arg-type]
    pipeline = ReportingPipeline(registry)
    result = pipeline.run("shuffled", _analytics_frame())
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result.schema == REPORTING_SCHEMA


def test_run_does_not_mutate_input_frame() -> None:
    """Pipeline run must not mutate the caller-supplied analytics frame."""
    pipeline = ReportingPipeline(_build_registry())
    analytics = _analytics_frame()
    before = analytics.clone()
    pipeline.run("simple", analytics)
    assert_frame_equal(analytics, before)
