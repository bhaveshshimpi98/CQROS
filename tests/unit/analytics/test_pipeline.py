"""Unit tests for CQROS ``AnalyticsPipeline``."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.analytics import (
    ANALYTICS_SCHEMA,
    AnalyticsEngineRegistry,
    AnalyticsPipeline,
    AnalyticsStatus,
    AnalyticsValidationError,
    SimpleAnalyticsEngine,
)
from cqros.analytics.schema import CANONICAL_COLUMN_ORDER, COLUMN_DTYPES
from cqros.performance.schema import PerformanceStatus

_TIMEFRAME = "1h"
_MANAGER = "simple"
_OPEN_TIME = datetime(2024, 1, 1, tzinfo=UTC)
_OPEN_TIME_MS = int(_OPEN_TIME.timestamp() * 1000.0)


def _performance_frame(
    *,
    symbol: str = "BTCUSDT",
    manager: str = _MANAGER,
) -> pl.DataFrame:
    """Build a minimal performance frame for pipeline tests."""
    return pl.DataFrame(
        {
            "symbol": [symbol],
            "timeframe": [_TIMEFRAME],
            "open_time": [_OPEN_TIME],
            "manager": [manager],
            "total_return": [0.05],
            "volatility": [0.1],
            "sharpe_ratio": [1.0],
            "sortino_ratio": [1.2],
            "max_drawdown": [0.02],
            "win_rate": [0.6],
            "profit_factor": [1.5],
            "expectancy": [10.0],
            "cagr": [0.08],
            "calmar_ratio": [4.0],
            "net_profit": [500.0],
            "status": [PerformanceStatus.FINISHED.value],
        }
    )


def _canonical_analytics_row(*, status: str = AnalyticsStatus.FINISHED.value) -> pl.DataFrame:
    """Build one canonical analytics row for synthetic engine outputs."""
    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timeframe": [_TIMEFRAME],
            "open_time": [_OPEN_TIME_MS],
            "manager": [_MANAGER],
            "rolling_return": [0.05],
            "rolling_volatility": [0.1],
            "rolling_sharpe": [1.0],
            "rolling_sortino": [1.2],
            "rolling_max_drawdown": [0.02],
            "rolling_win_rate": [0.6],
            "rolling_profit_factor": [1.5],
            "rolling_expectancy": [10.0],
            "rolling_cagr": [0.08],
            "rolling_calmar": [4.0],
            "rolling_recovery_factor": [25000.0],
            "benchmark_return": [0.0],
            "benchmark_alpha": [0.0],
            "benchmark_beta": [0.0],
            "benchmark_correlation": [0.0],
            "benchmark_tracking_error": [0.0],
            "benchmark_information_ratio": [0.0],
            "status": [status],
        },
        schema=dict(COLUMN_DTYPES),
    ).select(list(CANONICAL_COLUMN_ORDER))


def _build_registry(engine_name: str = "simple") -> AnalyticsEngineRegistry:
    """Build a registry with SimpleAnalyticsEngine under engine_name."""
    registry = AnalyticsEngineRegistry()
    registry.register(engine_name, SimpleAnalyticsEngine())
    return registry


def _run(
    pipeline: AnalyticsPipeline,
    *,
    engine_name: str = "simple",
    performance: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Run the pipeline with a default performance frame."""
    return pipeline.run(
        engine_name,
        performance if performance is not None else _performance_frame(),
    )


# ---------------------------------------------------------------------------
# Engine-name validation
# ---------------------------------------------------------------------------


def test_run_rejects_blank_engine_name() -> None:
    """Blank or whitespace engine names raise ANA_PIPE_NAME_BLANK."""
    pipeline = AnalyticsPipeline(_build_registry())
    for blank in ("", "   "):
        with pytest.raises(AnalyticsValidationError) as exc_info:
            pipeline.run(blank, _performance_frame())
        assert exc_info.value.error_code == "ANA_PIPE_NAME_BLANK"


def test_run_rejects_unknown_engine_name() -> None:
    """Unregistered engine name causes ANA_REG_UNKNOWN from registry lookup."""
    pipeline = AnalyticsPipeline(_build_registry())
    with pytest.raises(AnalyticsValidationError) as exc_info:
        _run(pipeline, engine_name="unknown-engine")
    assert exc_info.value.error_code == "ANA_REG_UNKNOWN"


# ---------------------------------------------------------------------------
# Duplicate / missing primary key rejection
# ---------------------------------------------------------------------------


def test_run_rejects_duplicate_primary_keys() -> None:
    """Duplicate primary keys in engine output raise ANA_PIPE_DUPLICATE_KEYS."""

    class _DuplicatingEngine:
        """Test engine that returns two rows with identical primary keys."""

        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            row = _canonical_analytics_row()
            return pl.concat([row, row])

    registry = AnalyticsEngineRegistry()
    registry.register("duplicating", _DuplicatingEngine())  # type: ignore[arg-type]
    pipeline = AnalyticsPipeline(registry)

    with pytest.raises(AnalyticsValidationError) as exc_info:
        pipeline.run("duplicating", _performance_frame())
    assert exc_info.value.error_code == "ANA_PIPE_DUPLICATE_KEYS"


def test_run_rejects_missing_primary_key_columns() -> None:
    """Engine output missing primary-key columns raises ANA_PIPE_MISSING_PRIMARY_KEYS."""

    class _MissingPrimaryKeyEngine:
        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            row = _canonical_analytics_row().drop("open_time")
            return row

    registry = AnalyticsEngineRegistry()
    registry.register("missing-pk", _MissingPrimaryKeyEngine())  # type: ignore[arg-type]
    pipeline = AnalyticsPipeline(registry)

    with pytest.raises(AnalyticsValidationError) as exc_info:
        pipeline.run("missing-pk", _performance_frame())
    assert exc_info.value.error_code == "ANA_PIPE_MISSING_PRIMARY_KEYS"


# ---------------------------------------------------------------------------
# Invalid engine output types
# ---------------------------------------------------------------------------


def test_run_rejects_non_dataframe_engine_output() -> None:
    """Engine returning a non-DataFrame raises ANA_PIPE_INVALID_OUTPUT."""

    class _BadEngine:
        def build(self, *args: object, **kwargs: object) -> object:
            return "not-a-dataframe"

    registry = AnalyticsEngineRegistry()
    registry.register("bad", _BadEngine())  # type: ignore[arg-type]
    pipeline = AnalyticsPipeline(registry)

    with pytest.raises(AnalyticsValidationError) as exc_info:
        pipeline.run("bad", _performance_frame())
    assert exc_info.value.error_code == "ANA_PIPE_INVALID_OUTPUT"


def test_run_rejects_empty_engine_output() -> None:
    """Engine returning an empty DataFrame raises ANA_PIPE_OUTPUT_EMPTY."""

    class _EmptyEngine:
        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            return pl.DataFrame({"symbol": []})

    registry = AnalyticsEngineRegistry()
    registry.register("empty", _EmptyEngine())  # type: ignore[arg-type]
    pipeline = AnalyticsPipeline(registry)

    with pytest.raises(AnalyticsValidationError) as exc_info:
        pipeline.run("empty", _performance_frame())
    assert exc_info.value.error_code == "ANA_PIPE_OUTPUT_EMPTY"


def test_run_rejects_missing_schema_columns_in_engine_output() -> None:
    """Engine returning a DataFrame missing required columns raises ANA_PIPE_MISSING_COLUMNS."""

    class _IncompleteEngine:
        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            # Include primary keys so the failure is specifically missing required metrics.
            return pl.DataFrame(
                {
                    "symbol": ["BTCUSDT"],
                    "timeframe": [_TIMEFRAME],
                    "open_time": [_OPEN_TIME_MS],
                    "rolling_return": [0.0],
                }
            )

    registry = AnalyticsEngineRegistry()
    registry.register("incomplete", _IncompleteEngine())  # type: ignore[arg-type]
    pipeline = AnalyticsPipeline(registry)

    with pytest.raises(AnalyticsValidationError) as exc_info:
        pipeline.run("incomplete", _performance_frame())
    assert exc_info.value.error_code == "ANA_PIPE_MISSING_COLUMNS"


def test_run_rejects_schema_cast_failure() -> None:
    """Engine output that cannot cast to ANALYTICS_SCHEMA raises ANA_PIPE_SCHEMA_CAST."""

    class _UncastableEngine:
        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            row = _canonical_analytics_row()
            return row.with_columns(pl.lit("not-a-float").alias("rolling_return"))

    registry = AnalyticsEngineRegistry()
    registry.register("uncastable", _UncastableEngine())  # type: ignore[arg-type]
    pipeline = AnalyticsPipeline(registry)

    with pytest.raises(AnalyticsValidationError) as exc_info:
        pipeline.run("uncastable", _performance_frame())
    assert exc_info.value.error_code == "ANA_PIPE_SCHEMA_CAST"


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_run_produces_canonical_analytics_row() -> None:
    """Pipeline with default inputs produces one canonical analytics output row."""
    pipeline = AnalyticsPipeline(_build_registry())
    result = _run(pipeline)
    assert result.height == 1
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result.schema == ANALYTICS_SCHEMA
    assert result["status"].to_list() == [AnalyticsStatus.FINISHED.value]
    assert result["open_time"].to_list() == [_OPEN_TIME_MS]
    assert result["rolling_return"].to_list()[0] == pytest.approx(0.05)
    assert result["benchmark_return"].to_list()[0] == pytest.approx(0.0)


def test_run_resolves_engine_from_registry() -> None:
    """Pipeline resolves and executes the engine registered under engine_name."""
    registry = AnalyticsEngineRegistry()
    engine = SimpleAnalyticsEngine()
    registry.register("custom", engine)
    pipeline = AnalyticsPipeline(registry)
    result = _run(pipeline, engine_name="custom")
    assert result.schema == ANALYTICS_SCHEMA
    assert result.height == 1


def test_run_reorders_columns_to_canonical_order() -> None:
    """Pipeline reorders engine output columns to CANONICAL_COLUMN_ORDER."""

    class _ShuffledEngine:
        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            row = _canonical_analytics_row()
            shuffled = list(reversed(row.columns))
            return row.select(shuffled)

    registry = AnalyticsEngineRegistry()
    registry.register("shuffled", _ShuffledEngine())  # type: ignore[arg-type]
    pipeline = AnalyticsPipeline(registry)
    result = pipeline.run("shuffled", _performance_frame())
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result.schema == ANALYTICS_SCHEMA


def test_run_does_not_mutate_input_frame() -> None:
    """Pipeline run must not mutate the caller-supplied performance frame."""
    pipeline = AnalyticsPipeline(_build_registry())
    performance = _performance_frame()
    before = performance.clone()
    pipeline.run("simple", performance)
    assert_frame_equal(performance, before)
