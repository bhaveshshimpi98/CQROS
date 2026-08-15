"""Unit tests for CQROS ``PerformancePipeline``."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from cqros.performance import (
    PERFORMANCE_SCHEMA,
    PerformanceEngineRegistry,
    PerformancePipeline,
    PerformanceStatus,
    PerformanceValidationError,
    SimplePerformanceEngine,
)
from cqros.performance.schema import CANONICAL_COLUMN_ORDER, COLUMN_DTYPES

_TIMEFRAME = "1h"
_MANAGER = "simple"
_OPEN_TIME = datetime(2024, 1, 1, tzinfo=UTC)


def _backtesting_frame(
    *,
    symbol: str = "BTCUSDT",
    equity: float = 10500.0,
    manager: str = _MANAGER,
) -> pl.DataFrame:
    """Build a minimal backtesting frame for pipeline tests."""
    return pl.DataFrame(
        {
            "symbol": [symbol],
            "timeframe": [_TIMEFRAME],
            "open_time": [_OPEN_TIME],
            "manager": [manager],
            "equity": [equity],
            "daily_return": [0.0],
            "drawdown": [0.0],
            "realized_pnl": [0.0],
            "trade_count": [0],
        }
    )


def _canonical_performance_row(*, status: str = PerformanceStatus.FINISHED.value) -> pl.DataFrame:
    """Build one canonical performance row for synthetic engine outputs."""
    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timeframe": [_TIMEFRAME],
            "open_time": [_OPEN_TIME],
            "manager": [_MANAGER],
            "total_return": [0.0],
            "cagr": [0.0],
            "volatility": [0.0],
            "downside_volatility": [0.0],
            "max_drawdown": [0.0],
            "drawdown_duration": [0],
            "sharpe_ratio": [None],
            "sortino_ratio": [None],
            "calmar_ratio": [None],
            "total_trades": [0],
            "winning_trades": [0],
            "losing_trades": [0],
            "win_rate": [0.0],
            "average_win": [None],
            "average_loss": [None],
            "profit_factor": [None],
            "expectancy": [0.0],
            "starting_equity": [10500.0],
            "ending_equity": [10500.0],
            "net_profit": [0.0],
            "gross_profit": [0.0],
            "gross_loss": [0.0],
            "first_trade_time": [None],
            "last_trade_time": [None],
            "status": [status],
        },
        schema=dict(COLUMN_DTYPES),
    ).select(list(CANONICAL_COLUMN_ORDER))


def _build_registry(engine_name: str = "simple") -> PerformanceEngineRegistry:
    """Build a registry with SimplePerformanceEngine under engine_name."""
    registry = PerformanceEngineRegistry()
    registry.register(engine_name, SimplePerformanceEngine())
    return registry


def _run(
    pipeline: PerformancePipeline,
    *,
    engine_name: str = "simple",
    backtesting: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Run the pipeline with a default backtesting frame."""
    return pipeline.run(
        engine_name,
        backtesting if backtesting is not None else _backtesting_frame(),
    )


# ---------------------------------------------------------------------------
# Engine-name validation
# ---------------------------------------------------------------------------


def test_run_rejects_blank_engine_name() -> None:
    """Blank or whitespace engine names raise PERF_PIPE_NAME_BLANK."""
    pipeline = PerformancePipeline(_build_registry())
    for blank in ("", "   "):
        with pytest.raises(PerformanceValidationError) as exc_info:
            pipeline.run(blank, _backtesting_frame())
        assert exc_info.value.error_code == "PERF_PIPE_NAME_BLANK"


def test_run_rejects_unknown_engine_name() -> None:
    """Unregistered engine name causes PERF_REG_UNKNOWN from registry lookup."""
    pipeline = PerformancePipeline(_build_registry())
    with pytest.raises(PerformanceValidationError) as exc_info:
        _run(pipeline, engine_name="unknown-engine")
    assert exc_info.value.error_code == "PERF_REG_UNKNOWN"


# ---------------------------------------------------------------------------
# Duplicate primary key rejection
# ---------------------------------------------------------------------------


def test_run_rejects_duplicate_primary_keys() -> None:
    """Duplicate primary keys in engine output raise PERF_PIPE_DUPLICATE_KEYS."""

    class _DuplicatingEngine:
        """Test engine that returns two rows with identical primary keys."""

        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            row = _canonical_performance_row()
            return pl.concat([row, row])

    registry = PerformanceEngineRegistry()
    registry.register("duplicating", _DuplicatingEngine())  # type: ignore[arg-type]
    pipeline = PerformancePipeline(registry)

    with pytest.raises(PerformanceValidationError) as exc_info:
        pipeline.run("duplicating", _backtesting_frame())
    assert exc_info.value.error_code == "PERF_PIPE_DUPLICATE_KEYS"


# ---------------------------------------------------------------------------
# Invalid engine output types
# ---------------------------------------------------------------------------


def test_run_rejects_non_dataframe_engine_output() -> None:
    """Engine returning a non-DataFrame raises PERF_PIPE_INVALID_OUTPUT."""

    class _BadEngine:
        def build(self, *args: object, **kwargs: object) -> object:
            return "not-a-dataframe"

    registry = PerformanceEngineRegistry()
    registry.register("bad", _BadEngine())  # type: ignore[arg-type]
    pipeline = PerformancePipeline(registry)

    with pytest.raises(PerformanceValidationError) as exc_info:
        pipeline.run("bad", _backtesting_frame())
    assert exc_info.value.error_code == "PERF_PIPE_INVALID_OUTPUT"


def test_run_rejects_empty_engine_output() -> None:
    """Engine returning an empty DataFrame raises PERF_PIPE_OUTPUT_EMPTY."""

    class _EmptyEngine:
        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            return pl.DataFrame({"symbol": []})

    registry = PerformanceEngineRegistry()
    registry.register("empty", _EmptyEngine())  # type: ignore[arg-type]
    pipeline = PerformancePipeline(registry)

    with pytest.raises(PerformanceValidationError) as exc_info:
        pipeline.run("empty", _backtesting_frame())
    assert exc_info.value.error_code == "PERF_PIPE_OUTPUT_EMPTY"


def test_run_rejects_missing_schema_columns_in_engine_output() -> None:
    """Engine returning a DataFrame missing required columns raises PERF_PIPE_MISSING_COLUMNS."""

    class _IncompleteEngine:
        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            return pl.DataFrame({"symbol": ["BTCUSDT"], "total_return": [0.0]})

    registry = PerformanceEngineRegistry()
    registry.register("incomplete", _IncompleteEngine())  # type: ignore[arg-type]
    pipeline = PerformancePipeline(registry)

    with pytest.raises(PerformanceValidationError) as exc_info:
        pipeline.run("incomplete", _backtesting_frame())
    assert exc_info.value.error_code == "PERF_PIPE_MISSING_COLUMNS"


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_run_produces_canonical_performance_row() -> None:
    """Pipeline with default inputs produces one canonical performance output row."""
    pipeline = PerformancePipeline(_build_registry())
    result = _run(pipeline)
    assert result.height == 1
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result.schema == PERFORMANCE_SCHEMA
    assert result["status"].to_list() == [PerformanceStatus.FINISHED.value]
    assert result["ending_equity"].to_list()[0] == pytest.approx(10500.0)


def test_run_does_not_mutate_input_frame() -> None:
    """Pipeline run must not mutate the caller-supplied backtesting frame."""
    from polars.testing import assert_frame_equal

    pipeline = PerformancePipeline(_build_registry())
    backtesting = _backtesting_frame()
    before = backtesting.clone()
    pipeline.run("simple", backtesting)
    assert_frame_equal(backtesting, before)
