"""Unit tests for CQROS ``SimplePerformanceEngine``."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.core.constants import DAYS_PER_YEAR, SECONDS_PER_DAY
from cqros.performance import (
    PERFORMANCE_SCHEMA,
    PerformanceStatus,
    PerformanceValidationError,
    SimplePerformanceEngine,
)
from cqros.performance.engine import BACKTESTING_INPUT_COLUMNS, validate_backtesting_frame
from cqros.performance.schema import CANONICAL_COLUMN_ORDER

_TIMEFRAME = "1h"
_MANAGER = "simple"
_SYMBOL = "BTCUSDT"
_SECONDS_PER_YEAR = float(SECONDS_PER_DAY * DAYS_PER_YEAR)


def _open_time(index: int = 0) -> datetime:
    """Build a deterministic UTC open_time for row ``index``."""
    return datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=index)


def _backtesting_frame(
    *,
    open_times: list[datetime] | None = None,
    equities: list[float] | None = None,
    daily_returns: list[float] | None = None,
    drawdowns: list[float] | None = None,
    realized_pnls: list[float] | None = None,
    trade_counts: list[int] | None = None,
    manager: str = _MANAGER,
    symbol: str = _SYMBOL,
) -> pl.DataFrame:
    """Build a minimal backtesting frame for performance engine tests."""
    open_times = open_times if open_times is not None else [_open_time(0)]
    row_count = len(open_times)
    equities = equities if equities is not None else [10500.0] * row_count
    daily_returns = daily_returns if daily_returns is not None else [0.0] * row_count
    drawdowns = drawdowns if drawdowns is not None else [0.0] * row_count
    realized_pnls = realized_pnls if realized_pnls is not None else [0.0] * row_count
    trade_counts = trade_counts if trade_counts is not None else [0] * row_count
    return pl.DataFrame(
        {
            "symbol": [symbol] * row_count,
            "timeframe": [_TIMEFRAME] * row_count,
            "open_time": open_times,
            "manager": [manager] * row_count,
            "equity": equities,
            "daily_return": daily_returns,
            "drawdown": drawdowns,
            "realized_pnl": realized_pnls,
            "trade_count": trade_counts,
        }
    )


def _build(
    engine: SimplePerformanceEngine,
    *,
    backtesting: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build performance metrics with a default backtesting frame."""
    return engine.build(backtesting if backtesting is not None else _backtesting_frame())


# ---------------------------------------------------------------------------
# Input column contracts
# ---------------------------------------------------------------------------


def test_input_columns_contract() -> None:
    """BACKTESTING_INPUT_COLUMNS enumerates every column the engine consumes."""
    for column in (
        "symbol",
        "timeframe",
        "open_time",
        "manager",
        "equity",
        "daily_return",
        "drawdown",
        "realized_pnl",
        "trade_count",
    ):
        assert column in BACKTESTING_INPUT_COLUMNS


# ---------------------------------------------------------------------------
# Frame validator tests
# ---------------------------------------------------------------------------


def test_validate_backtesting_frame_rejects_non_dataframe() -> None:
    """validate_backtesting_frame rejects non-DataFrame inputs with PERF_FRAME_TYPE."""
    with pytest.raises(PerformanceValidationError) as exc_info:
        validate_backtesting_frame("not-a-frame")  # type: ignore[arg-type]
    assert exc_info.value.error_code == "PERF_FRAME_TYPE"


def test_validate_backtesting_frame_rejects_empty_dataframe() -> None:
    """validate_backtesting_frame rejects DataFrames with zero rows."""
    empty = pl.DataFrame({"symbol": []})
    with pytest.raises(PerformanceValidationError) as exc_info:
        validate_backtesting_frame(empty)
    assert exc_info.value.error_code == "PERF_FRAME_EMPTY"


# ---------------------------------------------------------------------------
# Missing column validation
# ---------------------------------------------------------------------------


def test_build_rejects_missing_backtesting_columns() -> None:
    """Missing required backtesting columns raise PERF_MISSING_COLUMNS."""
    engine = SimplePerformanceEngine()
    with pytest.raises(PerformanceValidationError) as exc_info:
        _build(engine, backtesting=_backtesting_frame().drop("equity"))
    assert exc_info.value.error_code == "PERF_MISSING_COLUMNS"


def test_build_rejects_decreasing_trade_count() -> None:
    """Decreasing trade_count across open_time raises PERF_TRADE_COUNT."""
    t0, t1 = _open_time(0), _open_time(1)
    with pytest.raises(PerformanceValidationError) as exc_info:
        _build(
            SimplePerformanceEngine(),
            backtesting=_backtesting_frame(
                open_times=[t0, t1],
                equities=[10000.0, 10100.0],
                daily_returns=[0.0, 0.01],
                trade_counts=[2, 1],
            ),
        )
    assert exc_info.value.error_code == "PERF_TRADE_COUNT"


def test_build_rejects_non_finite_equity_metrics() -> None:
    """Non-finite equity values raise PERF_NON_FINITE."""
    with pytest.raises(PerformanceValidationError) as exc_info:
        _build(
            SimplePerformanceEngine(),
            backtesting=_backtesting_frame(equities=[float("nan")]),
        )
    assert exc_info.value.error_code == "PERF_NON_FINITE"


# ---------------------------------------------------------------------------
# Equity and return metrics
# ---------------------------------------------------------------------------


def test_starting_and_ending_equity() -> None:
    """starting_equity is first equity; ending_equity is current equity."""
    t0 = _open_time(0)
    t1 = t0 + timedelta(days=DAYS_PER_YEAR)
    result = _build(
        SimplePerformanceEngine(),
        backtesting=_backtesting_frame(
            open_times=[t0, t1],
            equities=[10000.0, 11000.0],
            daily_returns=[0.0, 0.1],
        ),
    )
    assert result["starting_equity"].to_list() == [pytest.approx(10000.0), pytest.approx(10000.0)]
    assert result["ending_equity"].to_list() == [pytest.approx(10000.0), pytest.approx(11000.0)]
    assert result["net_profit"].to_list() == [pytest.approx(0.0), pytest.approx(1000.0)]


def test_total_return_from_starting_equity() -> None:
    """total_return = ending / starting - 1 when starting equity is positive."""
    t0 = _open_time(0)
    t1 = t0 + timedelta(days=DAYS_PER_YEAR)
    result = _build(
        SimplePerformanceEngine(),
        backtesting=_backtesting_frame(
            open_times=[t0, t1],
            equities=[10000.0, 10500.0],
            daily_returns=[0.0, 0.05],
        ),
    )
    assert result["total_return"].to_list()[0] == pytest.approx(0.0)
    assert result["total_return"].to_list()[1] == pytest.approx(0.05)


def test_total_return_zero_when_starting_equity_non_positive() -> None:
    """total_return is 0 when starting equity is not positive."""
    result = _build(
        SimplePerformanceEngine(),
        backtesting=_backtesting_frame(equities=[0.0]),
    )
    assert result["total_return"].to_list() == [pytest.approx(0.0)]


def test_cagr_zero_on_single_timestamp() -> None:
    """cagr is 0 when elapsed calendar time is zero."""
    result = _build(SimplePerformanceEngine())
    assert result["cagr"].to_list() == [pytest.approx(0.0)]


def test_cagr_for_one_year_span() -> None:
    """cagr compounds ending/starting equity over calendar years."""
    t0 = _open_time(0)
    t1 = t0 + timedelta(days=DAYS_PER_YEAR)
    result = _build(
        SimplePerformanceEngine(),
        backtesting=_backtesting_frame(
            open_times=[t0, t1],
            equities=[10000.0, 11000.0],
            daily_returns=[0.0, 0.1],
        ),
    )
    assert result["cagr"].to_list()[1] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Volatility and risk ratios
# ---------------------------------------------------------------------------


def test_volatility_zero_with_insufficient_returns() -> None:
    """volatility is 0 when fewer than two daily_return samples exist."""
    result = _build(SimplePerformanceEngine())
    assert result["volatility"].to_list() == [pytest.approx(0.0)]
    assert result["sharpe_ratio"].to_list() == [None]


def test_volatility_and_sharpe_with_return_sample() -> None:
    """volatility annualizes sample std of daily_return[1:]."""
    t0, t1, t2 = _open_time(0), _open_time(1), _open_time(2)
    returns = [0.0, 0.01, -0.02]
    result = _build(
        SimplePerformanceEngine(),
        backtesting=_backtesting_frame(
            open_times=[t0, t1, t2],
            equities=[10000.0, 10100.0, 9898.0],
            daily_returns=returns,
        ),
    )
    sample = returns[1:]
    mean = sum(sample) / len(sample)
    variance = sum((value - mean) ** 2 for value in sample) / (len(sample) - 1)
    sample_std = math.sqrt(variance)
    periods_per_year = _SECONDS_PER_YEAR / 3600.0
    expected_vol = sample_std * math.sqrt(periods_per_year)
    expected_sharpe = (mean / sample_std) * math.sqrt(periods_per_year)
    assert result["volatility"].to_list()[2] == pytest.approx(expected_vol)
    assert result["sharpe_ratio"].to_list()[2] == pytest.approx(expected_sharpe)


def test_max_drawdown_is_running_max_of_drawdown() -> None:
    """max_drawdown tracks the running maximum of input drawdown."""
    t0, t1, t2 = _open_time(0), _open_time(1), _open_time(2)
    result = _build(
        SimplePerformanceEngine(),
        backtesting=_backtesting_frame(
            open_times=[t0, t1, t2],
            equities=[10000.0, 9500.0, 9800.0],
            daily_returns=[0.0, -0.05, 0.031578947],
            drawdowns=[0.0, 0.05, 0.02],
        ),
    )
    assert result["max_drawdown"].to_list() == [
        pytest.approx(0.0),
        pytest.approx(0.05),
        pytest.approx(0.05),
    ]


def test_drawdown_duration_longest_streak() -> None:
    """drawdown_duration is the longest consecutive streak with drawdown > 0."""
    t0, t1, t2, t3 = _open_time(0), _open_time(1), _open_time(2), _open_time(3)
    result = _build(
        SimplePerformanceEngine(),
        backtesting=_backtesting_frame(
            open_times=[t0, t1, t2, t3],
            equities=[10000.0, 9900.0, 9800.0, 10000.0],
            daily_returns=[0.0, -0.01, -0.010101, 0.020408],
            drawdowns=[0.0, 0.01, 0.02, 0.0],
        ),
    )
    assert result["drawdown_duration"].to_list()[-1] == 2


def test_calmar_null_when_no_drawdown() -> None:
    """calmar_ratio is NULL when max_drawdown is zero."""
    result = _build(SimplePerformanceEngine())
    assert result["calmar_ratio"].to_list() == [None]


def test_calmar_ratio_equals_cagr_over_max_drawdown() -> None:
    """calmar_ratio = cagr / max_drawdown when drawdown is positive."""
    t0 = _open_time(0)
    t1 = t0 + timedelta(days=DAYS_PER_YEAR)
    result = _build(
        SimplePerformanceEngine(),
        backtesting=_backtesting_frame(
            open_times=[t0, t1],
            equities=[10000.0, 11000.0],
            daily_returns=[0.0, 0.1],
            drawdowns=[0.0, 0.05],
        ),
    )
    cagr = result["cagr"].to_list()[1]
    assert result["calmar_ratio"].to_list()[1] == pytest.approx(cagr / 0.05)


# ---------------------------------------------------------------------------
# Trade statistics
# ---------------------------------------------------------------------------


def test_completed_trades_inferred_from_trade_count_increase() -> None:
    """Increasing trade_count attributes realized PnL delta across new trades."""
    t0, t1 = _open_time(0), _open_time(1)
    result = _build(
        SimplePerformanceEngine(),
        backtesting=_backtesting_frame(
            open_times=[t0, t1],
            equities=[10000.0, 10150.0],
            daily_returns=[0.0, 0.015],
            realized_pnls=[0.0, 150.0],
            trade_counts=[0, 1],
        ),
    )
    assert result["total_trades"].to_list() == [0, 1]
    assert result["winning_trades"].to_list() == [0, 1]
    assert result["win_rate"].to_list()[1] == pytest.approx(1.0)
    assert result["first_trade_time"].to_list()[1] == t1
    assert result["last_trade_time"].to_list()[1] == t1
    assert result["gross_profit"].to_list()[1] == pytest.approx(150.0)


def test_losing_trade_and_profit_factor() -> None:
    """Losing trades contribute to gross_loss and profit_factor."""
    t0, t1 = _open_time(0), _open_time(1)
    result = _build(
        SimplePerformanceEngine(),
        backtesting=_backtesting_frame(
            open_times=[t0, t1],
            equities=[10000.0, 9900.0],
            daily_returns=[0.0, -0.01],
            realized_pnls=[0.0, -100.0],
            trade_counts=[0, 1],
        ),
    )
    assert result["losing_trades"].to_list() == [0, 1]
    assert result["gross_loss"].to_list()[1] == pytest.approx(100.0)
    assert result["profit_factor"].to_list()[1] == pytest.approx(0.0)
    assert result["average_loss"].to_list()[1] == pytest.approx(-100.0)


def test_profit_factor_null_when_no_losses() -> None:
    """profit_factor is NULL when gross_loss is zero."""
    t0 = _open_time(0)
    result = _build(
        SimplePerformanceEngine(),
        backtesting=_backtesting_frame(
            open_times=[t0],
            realized_pnls=[200.0],
            trade_counts=[1],
        ),
    )
    assert result["profit_factor"].to_list() == [None]
    assert result["average_win"].to_list()[0] == pytest.approx(200.0)


def test_multiple_trades_at_same_timestamp_split_pnl() -> None:
    """Multiple new trades at one timestamp split realized PnL evenly."""
    t0 = _open_time(0)
    result = _build(
        SimplePerformanceEngine(),
        backtesting=_backtesting_frame(
            open_times=[t0],
            realized_pnls=[300.0],
            trade_counts=[2],
        ),
    )
    assert result["total_trades"].to_list() == [2]
    assert result["winning_trades"].to_list() == [2]
    assert result["expectancy"].to_list()[0] == pytest.approx(150.0)


def test_win_rate_zero_when_no_trades() -> None:
    """win_rate is 0 when no completed trades exist."""
    result = _build(SimplePerformanceEngine())
    assert result["win_rate"].to_list() == [pytest.approx(0.0)]
    assert result["total_trades"].to_list() == [0]
    assert result["first_trade_time"].to_list() == [None]
    assert result["last_trade_time"].to_list() == [None]


def test_zero_realized_pnl_trade_not_counted_as_win_or_loss() -> None:
    """Trades with zero realized PnL do not increment winning or losing counts."""
    t0 = _open_time(0)
    result = _build(
        SimplePerformanceEngine(),
        backtesting=_backtesting_frame(
            open_times=[t0],
            realized_pnls=[0.0],
            trade_counts=[1],
        ),
    )
    assert result["total_trades"].to_list() == [1]
    assert result["winning_trades"].to_list() == [0]
    assert result["losing_trades"].to_list() == [0]


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def test_status_active_until_final_row() -> None:
    """Intermediate rows carry ACTIVE status."""
    t0, t1 = _open_time(0), _open_time(1)
    result = _build(
        SimplePerformanceEngine(),
        backtesting=_backtesting_frame(
            open_times=[t0, t1],
            equities=[10000.0, 10100.0],
            daily_returns=[0.0, 0.01],
        ),
    )
    assert result["status"].to_list()[0] == PerformanceStatus.ACTIVE.value


def test_status_finished_on_last_row() -> None:
    """Final evaluation row carries FINISHED status."""
    t0, t1 = _open_time(0), _open_time(1)
    result = _build(
        SimplePerformanceEngine(),
        backtesting=_backtesting_frame(
            open_times=[t0, t1],
            equities=[10000.0, 10100.0],
            daily_returns=[0.0, 0.01],
        ),
    )
    assert result["status"].to_list()[-1] == PerformanceStatus.FINISHED.value


def test_single_row_has_finished_status() -> None:
    """A single-row ledger is FINISHED on its only row."""
    result = _build(SimplePerformanceEngine())
    assert result["status"].to_list() == [PerformanceStatus.FINISHED.value]


# ---------------------------------------------------------------------------
# Output schema, invariants, and immutability
# ---------------------------------------------------------------------------


def test_output_canonical_ordering_and_dtype_schema() -> None:
    """Engine output enforces canonical column order and PERFORMANCE_SCHEMA dtypes."""
    result = _build(SimplePerformanceEngine())
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result.schema == PERFORMANCE_SCHEMA
    assert result.schema["open_time"] == pl.Datetime("us", "UTC")
    assert result.schema["total_trades"] == pl.Int64


def test_manager_is_preserved_on_every_row() -> None:
    """manager column preserves upstream lineage on every row."""
    t0, t1 = _open_time(0), _open_time(1)
    result = _build(
        SimplePerformanceEngine(),
        backtesting=_backtesting_frame(
            manager="custom-manager",
            open_times=[t0, t1],
            equities=[10000.0, 10100.0],
            daily_returns=[0.0, 0.01],
        ),
    )
    assert result["manager"].to_list() == ["custom-manager", "custom-manager"]


def test_inputs_are_immutable() -> None:
    """build must not mutate the caller-supplied backtesting frame."""
    backtesting = _backtesting_frame()
    before = backtesting.clone()
    SimplePerformanceEngine().build(backtesting)
    assert_frame_equal(backtesting, before)


def test_multiple_timestamps_sorted_by_open_time() -> None:
    """Output rows are sorted by open_time ascending."""
    t0, t2, t1 = _open_time(0), _open_time(2), _open_time(1)
    result = _build(
        SimplePerformanceEngine(),
        backtesting=_backtesting_frame(
            open_times=[t2, t0, t1],
            equities=[10200.0, 10000.0, 10100.0],
            daily_returns=[0.0099, 0.0, 0.01],
        ),
    )
    assert result["open_time"].to_list() == [t0, t1, t2]
