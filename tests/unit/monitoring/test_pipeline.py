"""Unit tests for CQROS ``MonitoringPipeline``."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.monitoring import (
    MONITORING_SCHEMA,
    MonitoringEngineRegistry,
    MonitoringPipeline,
    MonitoringStatus,
    MonitoringValidationError,
    SimpleMonitoringEngine,
)
from cqros.monitoring.schema import CANONICAL_COLUMN_ORDER, COLUMN_DTYPES

_TIMEFRAME = "1h"
_MANAGER = "simple"
_OPEN_TIME = datetime(2024, 1, 1, tzinfo=UTC)
_OPEN_TIME_MS = int(_OPEN_TIME.timestamp() * 1000.0)


def _reporting_frame(
    *,
    symbol: str = "BTCUSDT",
    manager: str = _MANAGER,
) -> pl.DataFrame:
    """Build a minimal reporting frame for pipeline tests."""
    return pl.DataFrame(
        {
            "symbol": [symbol],
            "timeframe": [_TIMEFRAME],
            "open_time": [_OPEN_TIME],
            "manager": [manager],
        }
    )


def _canonical_monitoring_row(*, status: str = MonitoringStatus.NORMAL.value) -> pl.DataFrame:
    """Build one canonical monitoring row for synthetic engine outputs."""
    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timeframe": [_TIMEFRAME],
            "open_time": [_OPEN_TIME_MS],
            "manager": [_MANAGER],
            "monitor_type": ["system"],
            "monitor_name": ["report_monitor"],
            "severity": ["NORMAL"],
            "metric_name": ["report_generation"],
            "metric_value": [1.0],
            "threshold": [1.0],
            "alert": [False],
            "status": [status],
        },
        schema=dict(COLUMN_DTYPES),
    ).select(list(CANONICAL_COLUMN_ORDER))


def _build_registry(engine_name: str = "simple") -> MonitoringEngineRegistry:
    """Build a registry with SimpleMonitoringEngine under engine_name."""
    registry = MonitoringEngineRegistry()
    registry.register(engine_name, SimpleMonitoringEngine())
    return registry


def _run(
    pipeline: MonitoringPipeline,
    *,
    engine_name: str = "simple",
    reporting: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Run the pipeline with a default reporting frame."""
    return pipeline.run(
        engine_name,
        reporting if reporting is not None else _reporting_frame(),
    )


# ---------------------------------------------------------------------------
# Engine-name validation
# ---------------------------------------------------------------------------


def test_run_rejects_blank_engine_name() -> None:
    """Blank or whitespace engine names raise MON_PIPE_NAME_BLANK."""
    pipeline = MonitoringPipeline(_build_registry())
    for blank in ("", "   "):
        with pytest.raises(MonitoringValidationError) as exc_info:
            pipeline.run(blank, _reporting_frame())
        assert exc_info.value.error_code == "MON_PIPE_NAME_BLANK"


def test_run_rejects_unknown_engine_name() -> None:
    """Unregistered engine name causes MON_REG_UNKNOWN from registry lookup."""
    pipeline = MonitoringPipeline(_build_registry())
    with pytest.raises(MonitoringValidationError) as exc_info:
        _run(pipeline, engine_name="unknown-engine")
    assert exc_info.value.error_code == "MON_REG_UNKNOWN"


# ---------------------------------------------------------------------------
# Duplicate / missing primary key rejection
# ---------------------------------------------------------------------------


def test_run_rejects_duplicate_primary_keys() -> None:
    """Duplicate primary keys in engine output raise MON_PIPE_DUPLICATE_KEYS."""

    class _DuplicatingEngine:
        """Test engine that returns two rows with identical primary keys."""

        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            row = _canonical_monitoring_row()
            return pl.concat([row, row])

    registry = MonitoringEngineRegistry()
    registry.register("duplicating", _DuplicatingEngine())  # type: ignore[arg-type]
    pipeline = MonitoringPipeline(registry)

    with pytest.raises(MonitoringValidationError) as exc_info:
        pipeline.run("duplicating", _reporting_frame())
    assert exc_info.value.error_code == "MON_PIPE_DUPLICATE_KEYS"


def test_run_rejects_missing_primary_key_columns() -> None:
    """Engine output missing primary-key columns raises MON_PIPE_MISSING_PRIMARY_KEYS."""

    class _MissingPrimaryKeyEngine:
        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            row = _canonical_monitoring_row().drop("open_time")
            return row

    registry = MonitoringEngineRegistry()
    registry.register("missing-pk", _MissingPrimaryKeyEngine())  # type: ignore[arg-type]
    pipeline = MonitoringPipeline(registry)

    with pytest.raises(MonitoringValidationError) as exc_info:
        pipeline.run("missing-pk", _reporting_frame())
    assert exc_info.value.error_code == "MON_PIPE_MISSING_PRIMARY_KEYS"


# ---------------------------------------------------------------------------
# Invalid engine output types
# ---------------------------------------------------------------------------


def test_run_rejects_non_dataframe_engine_output() -> None:
    """Engine returning a non-DataFrame raises MON_PIPE_INVALID_OUTPUT."""

    class _BadEngine:
        def build(self, *args: object, **kwargs: object) -> object:
            return "not-a-dataframe"

    registry = MonitoringEngineRegistry()
    registry.register("bad", _BadEngine())  # type: ignore[arg-type]
    pipeline = MonitoringPipeline(registry)

    with pytest.raises(MonitoringValidationError) as exc_info:
        pipeline.run("bad", _reporting_frame())
    assert exc_info.value.error_code == "MON_PIPE_INVALID_OUTPUT"


def test_run_rejects_empty_engine_output() -> None:
    """Engine returning an empty DataFrame raises MON_PIPE_OUTPUT_EMPTY."""

    class _EmptyEngine:
        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            return pl.DataFrame({"symbol": []})

    registry = MonitoringEngineRegistry()
    registry.register("empty", _EmptyEngine())  # type: ignore[arg-type]
    pipeline = MonitoringPipeline(registry)

    with pytest.raises(MonitoringValidationError) as exc_info:
        pipeline.run("empty", _reporting_frame())
    assert exc_info.value.error_code == "MON_PIPE_OUTPUT_EMPTY"


def test_run_rejects_missing_schema_columns_in_engine_output() -> None:
    """Engine returning a DataFrame missing required columns raises MON_PIPE_MISSING_COLUMNS."""

    class _IncompleteEngine:
        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            # Include primary keys so the failure is specifically missing required fields.
            return pl.DataFrame(
                {
                    "symbol": ["BTCUSDT"],
                    "timeframe": [_TIMEFRAME],
                    "open_time": [_OPEN_TIME_MS],
                    "monitor_name": ["report_monitor"],
                }
            )

    registry = MonitoringEngineRegistry()
    registry.register("incomplete", _IncompleteEngine())  # type: ignore[arg-type]
    pipeline = MonitoringPipeline(registry)

    with pytest.raises(MonitoringValidationError) as exc_info:
        pipeline.run("incomplete", _reporting_frame())
    assert exc_info.value.error_code == "MON_PIPE_MISSING_COLUMNS"


def test_run_rejects_schema_cast_failure() -> None:
    """Engine output that cannot cast to MONITORING_SCHEMA raises MON_PIPE_SCHEMA_CAST."""

    class _UncastableEngine:
        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            row = _canonical_monitoring_row()
            return row.with_columns(pl.lit("not-a-float").alias("metric_value"))

    registry = MonitoringEngineRegistry()
    registry.register("uncastable", _UncastableEngine())  # type: ignore[arg-type]
    pipeline = MonitoringPipeline(registry)

    with pytest.raises(MonitoringValidationError) as exc_info:
        pipeline.run("uncastable", _reporting_frame())
    assert exc_info.value.error_code == "MON_PIPE_SCHEMA_CAST"


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_run_produces_canonical_monitoring_row() -> None:
    """Pipeline with default inputs produces one canonical monitoring output row."""
    pipeline = MonitoringPipeline(_build_registry())
    result = _run(pipeline)
    assert result.height == 1
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result.schema == MONITORING_SCHEMA
    assert result["status"].to_list() == [MonitoringStatus.NORMAL.value]
    assert result["open_time"].to_list() == [_OPEN_TIME_MS]
    assert result["monitor_type"].to_list() == ["system"]
    assert result["monitor_name"].to_list() == ["report_monitor"]
    assert result["metric_name"].to_list() == ["report_generation"]
    assert result["alert"].to_list() == [False]


def test_run_resolves_engine_from_registry() -> None:
    """Pipeline resolves and executes the engine registered under engine_name."""
    registry = MonitoringEngineRegistry()
    engine = SimpleMonitoringEngine()
    registry.register("custom", engine)
    pipeline = MonitoringPipeline(registry)
    result = _run(pipeline, engine_name="custom")
    assert result.schema == MONITORING_SCHEMA
    assert result.height == 1


def test_run_reorders_columns_to_canonical_order() -> None:
    """Pipeline reorders engine output columns to CANONICAL_COLUMN_ORDER."""

    class _ShuffledEngine:
        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            row = _canonical_monitoring_row()
            shuffled = list(reversed(row.columns))
            return row.select(shuffled)

    registry = MonitoringEngineRegistry()
    registry.register("shuffled", _ShuffledEngine())  # type: ignore[arg-type]
    pipeline = MonitoringPipeline(registry)
    result = pipeline.run("shuffled", _reporting_frame())
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result.schema == MONITORING_SCHEMA


def test_run_does_not_mutate_input_frame() -> None:
    """Pipeline run must not mutate the caller-supplied reporting frame."""
    pipeline = MonitoringPipeline(_build_registry())
    reporting = _reporting_frame()
    before = reporting.clone()
    pipeline.run("simple", reporting)
    assert_frame_equal(reporting, before)
