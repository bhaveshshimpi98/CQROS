"""Unit tests for CQROS ``SimpleAnalyticsEngine``."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.analytics import (
    ANALYTICS_SCHEMA,
    AnalyticsStatus,
    AnalyticsValidationError,
    SimpleAnalyticsEngine,
)
from cqros.analytics.engine import PERFORMANCE_INPUT_COLUMNS, validate_performance_frame
from cqros.analytics.schema import CANONICAL_COLUMN_ORDER
from cqros.performance.schema import PerformanceStatus

_TIMEFRAME = "1h"
_MANAGER = "simple"
_SYMBOL = "BTCUSDT"


def _open_time(index: int = 0) -> datetime:
    """Build a deterministic UTC open_time for row ``index``."""
    return datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=index)


def _epoch_ms(value: datetime) -> int:
    """Convert a UTC datetime to epoch milliseconds."""
    return int(value.timestamp() * 1000.0)


def _performance_frame(
    *,
    open_times: list[datetime] | None = None,
    total_returns: list[float] | None = None,
    volatilities: list[float] | None = None,
    sharpe_ratios: list[float | None] | None = None,
    sortino_ratios: list[float | None] | None = None,
    max_drawdowns: list[float] | None = None,
    win_rates: list[float] | None = None,
    profit_factors: list[float | None] | None = None,
    expectancies: list[float] | None = None,
    cagrs: list[float] | None = None,
    calmar_ratios: list[float | None] | None = None,
    net_profits: list[float] | None = None,
    statuses: list[str] | None = None,
    manager: str = _MANAGER,
    symbol: str = _SYMBOL,
) -> pl.DataFrame:
    """Build a minimal performance frame for analytics engine tests."""
    open_times = open_times if open_times is not None else [_open_time(0)]
    row_count = len(open_times)
    total_returns = total_returns if total_returns is not None else [0.05] * row_count
    volatilities = volatilities if volatilities is not None else [0.1] * row_count
    resolved_sharpe: list[float | None] = (
        list(sharpe_ratios) if sharpe_ratios is not None else [1.0] * row_count
    )
    resolved_sortino: list[float | None] = (
        list(sortino_ratios) if sortino_ratios is not None else [1.2] * row_count
    )
    max_drawdowns = max_drawdowns if max_drawdowns is not None else [0.02] * row_count
    win_rates = win_rates if win_rates is not None else [0.6] * row_count
    resolved_profit_factor: list[float | None] = (
        list(profit_factors) if profit_factors is not None else [1.5] * row_count
    )
    expectancies = expectancies if expectancies is not None else [10.0] * row_count
    cagrs = cagrs if cagrs is not None else [0.08] * row_count
    resolved_calmar: list[float | None] = (
        list(calmar_ratios) if calmar_ratios is not None else [4.0] * row_count
    )
    net_profits = net_profits if net_profits is not None else [500.0] * row_count
    if statuses is None:
        if row_count == 1:
            statuses = [PerformanceStatus.FINISHED.value]
        else:
            statuses = [PerformanceStatus.ACTIVE.value] * (row_count - 1) + [
                PerformanceStatus.FINISHED.value
            ]
    return pl.DataFrame(
        {
            "symbol": [symbol] * row_count,
            "timeframe": [_TIMEFRAME] * row_count,
            "open_time": open_times,
            "manager": [manager] * row_count,
            "total_return": total_returns,
            "volatility": volatilities,
            "sharpe_ratio": resolved_sharpe,
            "sortino_ratio": resolved_sortino,
            "max_drawdown": max_drawdowns,
            "win_rate": win_rates,
            "profit_factor": resolved_profit_factor,
            "expectancy": expectancies,
            "cagr": cagrs,
            "calmar_ratio": resolved_calmar,
            "net_profit": net_profits,
            "status": statuses,
        }
    )


def _build(
    engine: SimpleAnalyticsEngine,
    *,
    performance: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build analytics metrics with a default performance frame."""
    return engine.build(performance if performance is not None else _performance_frame())


# ---------------------------------------------------------------------------
# Input column contracts
# ---------------------------------------------------------------------------


def test_input_columns_contract() -> None:
    """PERFORMANCE_INPUT_COLUMNS enumerates every column the engine consumes."""
    for column in (
        "symbol",
        "timeframe",
        "open_time",
        "manager",
        "total_return",
        "volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "win_rate",
        "profit_factor",
        "expectancy",
        "cagr",
        "calmar_ratio",
        "net_profit",
        "status",
    ):
        assert column in PERFORMANCE_INPUT_COLUMNS


# ---------------------------------------------------------------------------
# Frame validator tests
# ---------------------------------------------------------------------------


def test_validate_performance_frame_rejects_non_dataframe() -> None:
    """validate_performance_frame rejects non-DataFrame inputs with ANA_FRAME_TYPE."""
    with pytest.raises(AnalyticsValidationError) as exc_info:
        validate_performance_frame("not-a-frame")  # type: ignore[arg-type]
    assert exc_info.value.error_code == "ANA_FRAME_TYPE"


def test_validate_performance_frame_rejects_empty_dataframe() -> None:
    """validate_performance_frame rejects DataFrames with zero rows."""
    empty = pl.DataFrame({"symbol": []})
    with pytest.raises(AnalyticsValidationError) as exc_info:
        validate_performance_frame(empty)
    assert exc_info.value.error_code == "ANA_FRAME_EMPTY"


def test_build_rejects_empty_dataframe() -> None:
    """build rejects empty performance frames."""
    empty = pl.DataFrame(schema={column: pl.Float64 for column in ("total_return",)}).clear()
    with pytest.raises(AnalyticsValidationError) as exc_info:
        SimpleAnalyticsEngine().build(empty)
    assert exc_info.value.error_code == "ANA_FRAME_EMPTY"


# ---------------------------------------------------------------------------
# Missing column validation
# ---------------------------------------------------------------------------


def test_build_rejects_missing_performance_columns() -> None:
    """Missing required performance columns raise ANA_MISSING_COLUMNS."""
    engine = SimpleAnalyticsEngine()
    with pytest.raises(AnalyticsValidationError) as exc_info:
        _build(engine, performance=_performance_frame().drop("total_return"))
    assert exc_info.value.error_code == "ANA_MISSING_COLUMNS"


def test_build_rejects_non_finite_metrics() -> None:
    """Non-finite performance metric values raise ANA_NON_FINITE."""
    with pytest.raises(AnalyticsValidationError) as exc_info:
        _build(
            SimpleAnalyticsEngine(),
            performance=_performance_frame(total_returns=[float("nan")]),
        )
    assert exc_info.value.error_code == "ANA_NON_FINITE"


# ---------------------------------------------------------------------------
# Rolling metric generation
# ---------------------------------------------------------------------------


def test_rolling_metrics_map_from_performance_expanding_window() -> None:
    """Rolling metrics at each timestamp use the expanding performance values."""
    t0, t1 = _open_time(0), _open_time(1)
    result = _build(
        SimpleAnalyticsEngine(),
        performance=_performance_frame(
            open_times=[t0, t1],
            total_returns=[0.0, 0.05],
            volatilities=[0.0, 0.12],
            sharpe_ratios=[None, 1.5],
            sortino_ratios=[None, 1.8],
            max_drawdowns=[0.0, 0.04],
            win_rates=[0.0, 0.55],
            profit_factors=[None, 1.25],
            expectancies=[0.0, 12.0],
            cagrs=[0.0, 0.1],
            calmar_ratios=[None, 2.5],
            net_profits=[0.0, 400.0],
        ),
    )
    assert result["rolling_return"].to_list() == [pytest.approx(0.0), pytest.approx(0.05)]
    assert result["rolling_volatility"].to_list() == [pytest.approx(0.0), pytest.approx(0.12)]
    assert result["rolling_sharpe"].to_list() == [None, pytest.approx(1.5)]
    assert result["rolling_sortino"].to_list() == [None, pytest.approx(1.8)]
    assert result["rolling_max_drawdown"].to_list() == [
        pytest.approx(0.0),
        pytest.approx(0.04),
    ]
    assert result["rolling_win_rate"].to_list() == [pytest.approx(0.0), pytest.approx(0.55)]
    assert result["rolling_profit_factor"].to_list() == [None, pytest.approx(1.25)]
    assert result["rolling_expectancy"].to_list() == [pytest.approx(0.0), pytest.approx(12.0)]
    assert result["rolling_cagr"].to_list() == [pytest.approx(0.0), pytest.approx(0.1)]
    assert result["rolling_calmar"].to_list() == [None, pytest.approx(2.5)]


def test_rolling_recovery_factor_from_net_profit_and_drawdown() -> None:
    """rolling_recovery_factor = net_profit / max_drawdown when drawdown is positive."""
    t0, t1 = _open_time(0), _open_time(1)
    result = _build(
        SimpleAnalyticsEngine(),
        performance=_performance_frame(
            open_times=[t0, t1],
            max_drawdowns=[0.0, 0.05],
            net_profits=[0.0, 250.0],
        ),
    )
    assert result["rolling_recovery_factor"].to_list()[0] is None
    assert result["rolling_recovery_factor"].to_list()[1] == pytest.approx(5000.0)


# ---------------------------------------------------------------------------
# Benchmark defaults
# ---------------------------------------------------------------------------


def test_benchmark_fields_are_deterministic_zeros() -> None:
    """Benchmark analytics fields are deterministic 0.0 stubs in v1."""
    result = _build(SimpleAnalyticsEngine())
    for column in (
        "benchmark_return",
        "benchmark_alpha",
        "benchmark_beta",
        "benchmark_correlation",
        "benchmark_tracking_error",
        "benchmark_information_ratio",
    ):
        assert result[column].to_list() == [pytest.approx(0.0)]


# ---------------------------------------------------------------------------
# Status propagation
# ---------------------------------------------------------------------------


def test_status_is_copied_from_performance_rows() -> None:
    """status is copied from the corresponding performance row."""
    t0, t1 = _open_time(0), _open_time(1)
    result = _build(
        SimpleAnalyticsEngine(),
        performance=_performance_frame(
            open_times=[t0, t1],
            statuses=[
                PerformanceStatus.ACTIVE.value,
                PerformanceStatus.FINISHED.value,
            ],
        ),
    )
    assert result["status"].to_list() == [
        AnalyticsStatus.ACTIVE.value,
        AnalyticsStatus.FINISHED.value,
    ]


def test_single_row_preserves_finished_status() -> None:
    """A single-row performance ledger preserves FINISHED status."""
    result = _build(SimpleAnalyticsEngine())
    assert result["status"].to_list() == [AnalyticsStatus.FINISHED.value]


# ---------------------------------------------------------------------------
# Output schema, invariants, and immutability
# ---------------------------------------------------------------------------


def test_output_canonical_ordering_and_dtype_schema() -> None:
    """Engine output enforces canonical column order and ANALYTICS_SCHEMA dtypes."""
    result = _build(SimpleAnalyticsEngine())
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result.schema == ANALYTICS_SCHEMA
    assert result.schema["open_time"] == pl.Int64
    assert result.schema["rolling_return"] == pl.Float64
    assert result.schema["status"] == pl.Utf8


def test_open_time_converted_to_epoch_milliseconds() -> None:
    """open_time is emitted as Int64 epoch milliseconds."""
    t0, t1 = _open_time(0), _open_time(1)
    result = _build(
        SimpleAnalyticsEngine(),
        performance=_performance_frame(open_times=[t0, t1]),
    )
    assert result["open_time"].to_list() == [_epoch_ms(t0), _epoch_ms(t1)]


def test_manager_is_preserved_on_every_row() -> None:
    """manager column preserves upstream lineage on every row."""
    t0, t1 = _open_time(0), _open_time(1)
    result = _build(
        SimpleAnalyticsEngine(),
        performance=_performance_frame(
            manager="custom-manager",
            open_times=[t0, t1],
        ),
    )
    assert result["manager"].to_list() == ["custom-manager", "custom-manager"]


def test_inputs_are_immutable() -> None:
    """build must not mutate the caller-supplied performance frame."""
    performance = _performance_frame()
    before = performance.clone()
    SimpleAnalyticsEngine().build(performance)
    assert_frame_equal(performance, before)


def test_output_is_deterministic() -> None:
    """Identical performance inputs produce identical analytics outputs."""
    performance = _performance_frame(
        open_times=[_open_time(0), _open_time(1)],
        total_returns=[0.0, 0.03],
        max_drawdowns=[0.0, 0.01],
        net_profits=[0.0, 100.0],
    )
    engine = SimpleAnalyticsEngine()
    first = engine.build(performance)
    second = engine.build(performance)
    assert_frame_equal(first, second)


def test_multiple_timestamps_sorted_by_open_time() -> None:
    """Output rows are sorted by open_time ascending."""
    t0, t2, t1 = _open_time(0), _open_time(2), _open_time(1)
    result = _build(
        SimpleAnalyticsEngine(),
        performance=_performance_frame(
            open_times=[t2, t0, t1],
            total_returns=[0.04, 0.0, 0.02],
            statuses=[
                PerformanceStatus.FINISHED.value,
                PerformanceStatus.ACTIVE.value,
                PerformanceStatus.ACTIVE.value,
            ],
        ),
    )
    assert result["open_time"].to_list() == [_epoch_ms(t0), _epoch_ms(t1), _epoch_ms(t2)]
    assert result["rolling_return"].to_list() == [
        pytest.approx(0.0),
        pytest.approx(0.02),
        pytest.approx(0.04),
    ]
