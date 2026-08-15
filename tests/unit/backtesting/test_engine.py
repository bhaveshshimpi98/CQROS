"""Unit tests for CQROS ``SimpleBacktestingEngine``."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.backtesting import (
    ACCOUNTING_INPUT_COLUMNS,
    CANONICAL_COLUMN_ORDER,
    EXIT_ENGINE_INPUT_COLUMNS,
    MERGED_BACKTESTING_SCHEMA,
    POSITION_INPUT_COLUMNS,
    BacktestingStatus,
    BacktestingValidationError,
    SimpleBacktestingEngine,
    validate_accounting_frame,
    validate_exit_engine_frame,
    validate_position_frame,
)

_TIMEFRAME = "1h"
_MANAGER = "simple"
_SYMBOL = "BTCUSDT"
_POSITION_ID = "pos-00000001"


def _open_time(index: int = 0) -> datetime:
    """Build a deterministic UTC open_time for row ``index``."""
    return datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=index)


def _accounting_frame(
    *,
    open_times: list[datetime] | None = None,
    cash_values: list[float] | None = None,
    position_values: list[float] | None = None,
    realized_pnls: list[float] | None = None,
    unrealized_pnls: list[float] | None = None,
    position_ids: list[str] | None = None,
    position_statuses: list[str] | None = None,
) -> pl.DataFrame:
    """Build a minimal accounting frame for engine tests."""
    open_times = open_times if open_times is not None else [_open_time(0)]
    row_count = len(open_times)
    cash_values = cash_values if cash_values is not None else [10000.0] * row_count
    position_values = position_values if position_values is not None else [500.0] * row_count
    realized_pnls = realized_pnls if realized_pnls is not None else [0.0] * row_count
    unrealized_pnls = unrealized_pnls if unrealized_pnls is not None else [500.0] * row_count
    position_ids = position_ids if position_ids is not None else [_POSITION_ID] * row_count
    position_statuses = position_statuses if position_statuses is not None else ["OPEN"] * row_count
    return pl.DataFrame(
        {
            "symbol": [_SYMBOL] * row_count,
            "timeframe": [_TIMEFRAME] * row_count,
            "open_time": open_times,
            "cash": cash_values,
            "position_value": position_values,
            "realized_pnl": realized_pnls,
            "unrealized_pnl": unrealized_pnls,
            "position_id": position_ids,
            "position_status": position_statuses,
        }
    )


def _positions_frame(
    *,
    position_ids: list[str] | None = None,
    statuses: list[str] | None = None,
    realized_pnls: list[float] | None = None,
    opened_ats: list[datetime | None] | None = None,
    updated_ats: list[datetime | None] | None = None,
    closed_ats: list[datetime | None] | None = None,
) -> pl.DataFrame:
    """Build a minimal positions frame for engine tests."""
    position_ids = position_ids if position_ids is not None else [_POSITION_ID]
    row_count = len(position_ids)
    statuses = statuses if statuses is not None else ["OPEN"] * row_count
    realized_pnls = realized_pnls if realized_pnls is not None else [0.0] * row_count
    opened_ats = opened_ats if opened_ats is not None else [_open_time(0)] * row_count
    updated_ats = updated_ats if updated_ats is not None else [None] * row_count
    closed_ats = closed_ats if closed_ats is not None else [None] * row_count
    return pl.DataFrame(
        {
            "symbol": [_SYMBOL] * row_count,
            "timeframe": [_TIMEFRAME] * row_count,
            "position_id": position_ids,
            "status": statuses,
            "realized_pnl": realized_pnls,
            "opened_at": opened_ats,
            "updated_at": updated_ats,
            "closed_at": closed_ats,
        }
    )


def _exit_engine_frame(
    *,
    open_times: list[datetime] | None = None,
    position_ids: list[str] | None = None,
    exit_actions: list[str] | None = None,
) -> pl.DataFrame:
    """Build a minimal exit-engine frame for engine tests."""
    open_times = open_times if open_times is not None else [_open_time(0)]
    row_count = len(open_times)
    position_ids = position_ids if position_ids is not None else [_POSITION_ID] * row_count
    exit_actions = exit_actions if exit_actions is not None else ["HOLD"] * row_count
    return pl.DataFrame(
        {
            "symbol": [_SYMBOL] * row_count,
            "timeframe": [_TIMEFRAME] * row_count,
            "open_time": open_times,
            "position_id": position_ids,
            "exit_action": exit_actions,
        }
    )


def _build(
    engine: SimpleBacktestingEngine,
    *,
    accounting: pl.DataFrame | None = None,
    positions: pl.DataFrame | None = None,
    exit_engine: pl.DataFrame | None = None,
    manager: str = _MANAGER,
) -> pl.DataFrame:
    """Build a performance ledger with default companion frames."""
    return engine.build(
        accounting if accounting is not None else _accounting_frame(),
        positions if positions is not None else _positions_frame(),
        exit_engine if exit_engine is not None else _exit_engine_frame(),
        manager=manager,
    )


# ---------------------------------------------------------------------------
# Input column contracts
# ---------------------------------------------------------------------------


def test_input_columns_contract() -> None:
    """Input column constants enumerate every column the engine consumes."""
    assert "cash" in ACCOUNTING_INPUT_COLUMNS
    assert "position_value" in ACCOUNTING_INPUT_COLUMNS
    assert "realized_pnl" in ACCOUNTING_INPUT_COLUMNS
    assert "unrealized_pnl" in ACCOUNTING_INPUT_COLUMNS
    assert "position_id" in ACCOUNTING_INPUT_COLUMNS
    assert "position_status" in ACCOUNTING_INPUT_COLUMNS

    assert "status" in POSITION_INPUT_COLUMNS
    assert "realized_pnl" in POSITION_INPUT_COLUMNS
    assert "closed_at" in POSITION_INPUT_COLUMNS

    assert "exit_action" in EXIT_ENGINE_INPUT_COLUMNS
    assert "position_id" in EXIT_ENGINE_INPUT_COLUMNS


# ---------------------------------------------------------------------------
# Frame validator tests
# ---------------------------------------------------------------------------


def test_validate_frames_reject_non_dataframe() -> None:
    """Frame validators reject non-DataFrame inputs with BT_FRAME_TYPE."""
    for validator in (
        validate_accounting_frame,
        validate_position_frame,
        validate_exit_engine_frame,
    ):
        with pytest.raises(BacktestingValidationError) as exc_info:
            validator("not-a-frame")  # type: ignore[arg-type]
        assert exc_info.value.error_code == "BT_FRAME_TYPE"


def test_validate_frames_reject_empty_dataframe() -> None:
    """Frame validators reject DataFrames with zero rows with BT_FRAME_EMPTY."""
    empty = pl.DataFrame({"symbol": []})
    for validator in (
        validate_accounting_frame,
        validate_position_frame,
        validate_exit_engine_frame,
    ):
        with pytest.raises(BacktestingValidationError) as exc_info:
            validator(empty)
        assert exc_info.value.error_code == "BT_FRAME_EMPTY"


# ---------------------------------------------------------------------------
# Manager validation
# ---------------------------------------------------------------------------


def test_build_rejects_blank_manager() -> None:
    """Blank or whitespace-only managers raise BT_MANAGER_BLANK."""
    engine = SimpleBacktestingEngine()
    for blank in ("", "   ", "\t"):
        with pytest.raises(BacktestingValidationError) as exc_info:
            _build(engine, manager=blank)
        assert exc_info.value.error_code == "BT_MANAGER_BLANK"


# ---------------------------------------------------------------------------
# Missing column validation
# ---------------------------------------------------------------------------


def test_build_rejects_missing_accounting_columns() -> None:
    """Missing required accounting columns raise BT_MISSING_COLUMNS."""
    engine = SimpleBacktestingEngine()
    with pytest.raises(BacktestingValidationError) as exc_info:
        _build(engine, accounting=_accounting_frame().drop("cash"))
    assert exc_info.value.error_code == "BT_MISSING_COLUMNS"


def test_build_rejects_missing_position_columns() -> None:
    """Missing required position columns raise BT_MISSING_COLUMNS."""
    engine = SimpleBacktestingEngine()
    with pytest.raises(BacktestingValidationError) as exc_info:
        _build(engine, positions=_positions_frame().drop("status"))
    assert exc_info.value.error_code == "BT_MISSING_COLUMNS"


def test_build_rejects_missing_exit_engine_columns() -> None:
    """Missing required exit-engine columns raise BT_MISSING_COLUMNS."""
    engine = SimpleBacktestingEngine()
    with pytest.raises(BacktestingValidationError) as exc_info:
        _build(engine, exit_engine=_exit_engine_frame().drop("exit_action"))
    assert exc_info.value.error_code == "BT_MISSING_COLUMNS"


def test_build_rejects_no_shared_timestamps() -> None:
    """No intersection of accounting/exit timestamps raises BT_NO_TIMESTAMPS."""
    engine = SimpleBacktestingEngine()
    with pytest.raises(BacktestingValidationError) as exc_info:
        _build(
            engine,
            accounting=_accounting_frame(open_times=[_open_time(0)]),
            exit_engine=_exit_engine_frame(open_times=[_open_time(1)]),
        )
    assert exc_info.value.error_code == "BT_NO_TIMESTAMPS"


# ---------------------------------------------------------------------------
# Equity and portfolio metrics
# ---------------------------------------------------------------------------


def test_equity_equals_cash_plus_unrealized() -> None:
    """equity = cash + unrealized_pnl at each evaluation timestamp."""
    result = _build(
        SimpleBacktestingEngine(),
        accounting=_accounting_frame(cash_values=[9000.0], unrealized_pnls=[750.0]),
    )
    assert result["equity"].to_list() == [pytest.approx(9750.0)]


def test_equity_uses_first_cash_row_at_timestamp() -> None:
    """Cash is taken from the first accounting row at each timestamp."""
    accounting = pl.DataFrame(
        {
            "symbol": [_SYMBOL, _SYMBOL],
            "timeframe": [_TIMEFRAME, _TIMEFRAME],
            "open_time": [_open_time(0), _open_time(0)],
            "cash": [8000.0, 9000.0],
            "position_value": [100.0, 200.0],
            "realized_pnl": [0.0, 0.0],
            "unrealized_pnl": [300.0, 400.0],
            "position_id": ["pos-a", "pos-b"],
            "position_status": ["OPEN", "OPEN"],
        }
    )
    result = _build(SimpleBacktestingEngine(), accounting=accounting)
    assert result["cash"].to_list() == [pytest.approx(8000.0)]
    assert result["equity"].to_list() == [pytest.approx(8700.0)]


def test_position_value_and_realized_are_summed() -> None:
    """position_value and realized_pnl are summed across positions at timestamp."""
    accounting = pl.DataFrame(
        {
            "symbol": [_SYMBOL, _SYMBOL],
            "timeframe": [_TIMEFRAME, _TIMEFRAME],
            "open_time": [_open_time(0), _open_time(0)],
            "cash": [10000.0, 10000.0],
            "position_value": [100.0, 250.0],
            "realized_pnl": [50.0, 75.0],
            "unrealized_pnl": [200.0, 300.0],
            "position_id": ["pos-a", "pos-b"],
            "position_status": ["OPEN", "OPEN"],
        }
    )
    result = _build(SimpleBacktestingEngine(), accounting=accounting)
    assert result["position_value"].to_list() == [pytest.approx(350.0)]
    assert result["realized_pnl"].to_list() == [pytest.approx(125.0)]
    assert result["unrealized_pnl"].to_list() == [pytest.approx(500.0)]


def test_total_pnl_equals_realized_plus_unrealized() -> None:
    """total_pnl = realized_pnl + unrealized_pnl at each timestamp."""
    result = _build(
        SimpleBacktestingEngine(),
        accounting=_accounting_frame(realized_pnls=[120.0], unrealized_pnls=[380.0]),
    )
    assert result["total_pnl"].to_list() == [pytest.approx(500.0)]


# ---------------------------------------------------------------------------
# Drawdown and peak equity
# ---------------------------------------------------------------------------


def test_peak_equity_running_max() -> None:
    """peak_equity tracks the running maximum equity."""
    t0, t1, t2 = _open_time(0), _open_time(1), _open_time(2)
    result = _build(
        SimpleBacktestingEngine(),
        accounting=_accounting_frame(
            open_times=[t0, t1, t2],
            cash_values=[10000.0, 10000.0, 10000.0],
            unrealized_pnls=[1000.0, 500.0, 800.0],
        ),
        exit_engine=_exit_engine_frame(open_times=[t0, t1, t2]),
    )
    assert result["peak_equity"].to_list() == [
        pytest.approx(11000.0),
        pytest.approx(11000.0),
        pytest.approx(11000.0),
    ]


def test_drawdown_calculation() -> None:
    """drawdown = (peak_equity - equity) / peak_equity when peak is positive."""
    t0, t1 = _open_time(0), _open_time(1)
    result = _build(
        SimpleBacktestingEngine(),
        accounting=_accounting_frame(
            open_times=[t0, t1],
            cash_values=[10000.0, 10000.0],
            unrealized_pnls=[1000.0, 200.0],
        ),
        exit_engine=_exit_engine_frame(open_times=[t0, t1]),
    )
    assert result["drawdown"].to_list()[0] == pytest.approx(0.0)
    assert result["drawdown"].to_list()[1] == pytest.approx(800.0 / 11000.0)


def test_drawdown_zero_when_peak_zero() -> None:
    """drawdown is 0 when peak_equity is not positive."""
    result = _build(
        SimpleBacktestingEngine(),
        accounting=_accounting_frame(cash_values=[0.0], unrealized_pnls=[0.0]),
    )
    assert result["drawdown"].to_list() == [pytest.approx(0.0)]


def test_max_drawdown_running_max() -> None:
    """max_drawdown is the running maximum drawdown."""
    t0, t1, t2 = _open_time(0), _open_time(1), _open_time(2)
    result = _build(
        SimpleBacktestingEngine(),
        accounting=_accounting_frame(
            open_times=[t0, t1, t2],
            cash_values=[10000.0, 10000.0, 10000.0],
            unrealized_pnls=[1000.0, 200.0, 900.0],
        ),
        exit_engine=_exit_engine_frame(open_times=[t0, t1, t2]),
    )
    dd1 = 800.0 / 11000.0
    assert result["max_drawdown"].to_list()[0] == pytest.approx(0.0)
    assert result["max_drawdown"].to_list()[1] == pytest.approx(dd1)
    assert result["max_drawdown"].to_list()[2] == pytest.approx(dd1)


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------


def test_daily_return_zero_on_first_row() -> None:
    """daily_return is 0 on the first evaluation row."""
    result = _build(SimpleBacktestingEngine())
    assert result["daily_return"].to_list()[0] == pytest.approx(0.0)


def test_daily_return_subsequent_rows() -> None:
    """daily_return = equity_t / equity_(t-1) - 1 on subsequent rows."""
    t0, t1 = _open_time(0), _open_time(1)
    result = _build(
        SimpleBacktestingEngine(),
        accounting=_accounting_frame(
            open_times=[t0, t1],
            cash_values=[10000.0, 10000.0],
            unrealized_pnls=[500.0, 200.0],
        ),
        exit_engine=_exit_engine_frame(open_times=[t0, t1]),
    )
    expected = 10200.0 / 10500.0 - 1.0
    assert result["daily_return"].to_list()[1] == pytest.approx(expected)


def test_daily_return_zero_when_previous_equity_zero() -> None:
    """daily_return is 0 when previous equity is zero."""
    t0, t1 = _open_time(0), _open_time(1)
    result = _build(
        SimpleBacktestingEngine(),
        accounting=_accounting_frame(
            open_times=[t0, t1],
            cash_values=[0.0, 10000.0],
            unrealized_pnls=[0.0, 500.0],
        ),
        exit_engine=_exit_engine_frame(open_times=[t0, t1]),
    )
    assert result["daily_return"].to_list()[1] == pytest.approx(0.0)


def test_cumulative_return_from_initial_equity() -> None:
    """cumulative_return = equity / initial_equity - 1."""
    t0, t1 = _open_time(0), _open_time(1)
    result = _build(
        SimpleBacktestingEngine(),
        accounting=_accounting_frame(
            open_times=[t0, t1],
            cash_values=[10000.0, 10000.0],
            unrealized_pnls=[500.0, 1000.0],
        ),
        exit_engine=_exit_engine_frame(open_times=[t0, t1]),
    )
    assert result["cumulative_return"].to_list()[0] == pytest.approx(0.0)
    assert result["cumulative_return"].to_list()[1] == pytest.approx(11000.0 / 10500.0 - 1.0)


def test_cumulative_return_zero_when_initial_zero() -> None:
    """cumulative_return is 0 when initial equity is zero."""
    result = _build(
        SimpleBacktestingEngine(),
        accounting=_accounting_frame(cash_values=[0.0], unrealized_pnls=[0.0]),
    )
    assert result["cumulative_return"].to_list() == [pytest.approx(0.0)]


# ---------------------------------------------------------------------------
# Evaluation timestamps
# ---------------------------------------------------------------------------


def test_evaluation_timestamps_are_intersection() -> None:
    """Output rows exist only for timestamps present in both accounting and exit."""
    t0, t1, t2 = _open_time(0), _open_time(1), _open_time(2)
    result = _build(
        SimpleBacktestingEngine(),
        accounting=_accounting_frame(open_times=[t0, t1, t2]),
        exit_engine=_exit_engine_frame(open_times=[t0, t2]),
    )
    assert result.height == 2
    assert result["open_time"].to_list() == [t0, t2]


# ---------------------------------------------------------------------------
# Trade statistics
# ---------------------------------------------------------------------------


def test_completed_trades_from_closed_positions() -> None:
    """CLOSED positions contribute to trade statistics by closed_at."""
    t0, t1 = _open_time(0), _open_time(1)
    result = _build(
        SimpleBacktestingEngine(),
        accounting=_accounting_frame(open_times=[t0, t1]),
        exit_engine=_exit_engine_frame(open_times=[t0, t1]),
        positions=_positions_frame(
            statuses=["CLOSED"],
            realized_pnls=[150.0],
            closed_ats=[t0],
        ),
    )
    assert result["trade_count"].to_list()[0] == 1
    assert result["winning_trades"].to_list()[0] == 1
    assert result["win_rate"].to_list()[0] == pytest.approx(1.0)


def test_completed_trades_from_full_exit() -> None:
    """FULL_EXIT rows count trades when position is not already CLOSED."""
    t0 = _open_time(0)
    accounting = _accounting_frame(
        open_times=[t0],
        position_ids=[_POSITION_ID],
        realized_pnls=[-50.0],
    )
    result = _build(
        SimpleBacktestingEngine(),
        accounting=accounting,
        exit_engine=_exit_engine_frame(
            open_times=[t0],
            exit_actions=["FULL_EXIT"],
        ),
        positions=_positions_frame(statuses=["OPEN"]),
    )
    assert result["trade_count"].to_list() == [1]
    assert result["losing_trades"].to_list() == [1]
    assert result["winning_trades"].to_list() == [0]


def test_full_exit_not_double_counted_when_closed() -> None:
    """FULL_EXIT does not double-count positions already marked CLOSED."""
    t0 = _open_time(0)
    result = _build(
        SimpleBacktestingEngine(),
        accounting=_accounting_frame(open_times=[t0]),
        exit_engine=_exit_engine_frame(open_times=[t0], exit_actions=["FULL_EXIT"]),
        positions=_positions_frame(
            statuses=["CLOSED"],
            realized_pnls=[100.0],
            closed_ats=[t0],
        ),
    )
    assert result["trade_count"].to_list() == [1]


def test_win_rate_calculation() -> None:
    """win_rate = winning_trades / trade_count when trades exist."""
    t0, t1 = _open_time(0), _open_time(1)
    result = _build(
        SimpleBacktestingEngine(),
        accounting=_accounting_frame(open_times=[t0, t1]),
        exit_engine=_exit_engine_frame(open_times=[t0, t1]),
        positions=_positions_frame(
            position_ids=["pos-win", "pos-loss"],
            statuses=["CLOSED", "CLOSED"],
            realized_pnls=[200.0, -100.0],
            closed_ats=[t0, t1],
        ),
    )
    assert result["trade_count"].to_list()[1] == 2
    assert result["winning_trades"].to_list()[1] == 1
    assert result["losing_trades"].to_list()[1] == 1
    assert result["win_rate"].to_list()[1] == pytest.approx(0.5)


def test_win_rate_zero_when_no_trades() -> None:
    """win_rate is 0 when no completed trades exist."""
    result = _build(SimpleBacktestingEngine())
    assert result["win_rate"].to_list() == [pytest.approx(0.0)]
    assert result["trade_count"].to_list() == [0]


def test_profit_factor_null_when_no_losses() -> None:
    """profit_factor is NULL when there are no losing trades."""
    t0 = _open_time(0)
    result = _build(
        SimpleBacktestingEngine(),
        accounting=_accounting_frame(open_times=[t0]),
        exit_engine=_exit_engine_frame(open_times=[t0]),
        positions=_positions_frame(
            statuses=["CLOSED"],
            realized_pnls=[250.0],
            closed_ats=[t0],
        ),
    )
    assert result["profit_factor"].to_list() == [None]


def test_profit_factor_calculation_with_losses() -> None:
    """profit_factor = gross_profit / gross_loss when losses exist."""
    t0 = _open_time(0)
    result = _build(
        SimpleBacktestingEngine(),
        accounting=_accounting_frame(open_times=[t0]),
        exit_engine=_exit_engine_frame(open_times=[t0]),
        positions=_positions_frame(
            position_ids=["pos-win", "pos-loss"],
            statuses=["CLOSED", "CLOSED"],
            realized_pnls=[300.0, -100.0],
            closed_ats=[t0, t0],
        ),
    )
    assert result["profit_factor"].to_list()[0] == pytest.approx(3.0)


def test_sharpe_stub_always_null() -> None:
    """sharpe_stub is always NULL in v1."""
    t0, t1 = _open_time(0), _open_time(1)
    result = _build(
        SimpleBacktestingEngine(),
        accounting=_accounting_frame(open_times=[t0, t1]),
        exit_engine=_exit_engine_frame(open_times=[t0, t1]),
    )
    assert result["sharpe_stub"].to_list() == [None, None]


def test_sortino_stub_always_null() -> None:
    """sortino_stub is always NULL in v1."""
    result = _build(SimpleBacktestingEngine())
    assert result["sortino_stub"].to_list() == [None]


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def test_status_active_until_final_row() -> None:
    """Intermediate rows carry ACTIVE status."""
    t0, t1 = _open_time(0), _open_time(1)
    result = _build(
        SimpleBacktestingEngine(),
        accounting=_accounting_frame(open_times=[t0, t1]),
        exit_engine=_exit_engine_frame(open_times=[t0, t1]),
    )
    assert result["status"].to_list()[0] == BacktestingStatus.ACTIVE.value


def test_status_finished_on_last_row() -> None:
    """Final evaluation row carries FINISHED status."""
    t0, t1 = _open_time(0), _open_time(1)
    result = _build(
        SimpleBacktestingEngine(),
        accounting=_accounting_frame(open_times=[t0, t1]),
        exit_engine=_exit_engine_frame(open_times=[t0, t1]),
    )
    assert result["status"].to_list()[-1] == BacktestingStatus.FINISHED.value


def test_single_row_has_finished_status() -> None:
    """A single-row ledger is FINISHED on its only row."""
    result = _build(SimpleBacktestingEngine())
    assert result["status"].to_list() == [BacktestingStatus.FINISHED.value]


# ---------------------------------------------------------------------------
# Output schema, invariants, and immutability
# ---------------------------------------------------------------------------


def test_output_canonical_ordering_and_dtype_schema() -> None:
    """Engine output enforces canonical column order and MERGED_BACKTESTING_SCHEMA dtypes."""
    result = _build(SimpleBacktestingEngine())
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result.schema == MERGED_BACKTESTING_SCHEMA
    assert result.schema["open_time"] == pl.Datetime("us", "UTC")
    assert result.schema["trade_count"] == pl.Int64


def test_manager_is_stamped_on_every_row() -> None:
    """manager column contains the injected manager identity on every row."""
    t0, t1 = _open_time(0), _open_time(1)
    result = _build(
        SimpleBacktestingEngine(),
        manager="custom-manager",
        accounting=_accounting_frame(open_times=[t0, t1]),
        exit_engine=_exit_engine_frame(open_times=[t0, t1]),
    )
    assert result["manager"].to_list() == ["custom-manager", "custom-manager"]


def test_inputs_are_immutable() -> None:
    """build must not mutate any caller-supplied input frame."""
    accounting = _accounting_frame()
    positions = _positions_frame()
    exit_engine = _exit_engine_frame()

    accounting_before = accounting.clone()
    positions_before = positions.clone()
    exit_before = exit_engine.clone()

    SimpleBacktestingEngine().build(accounting, positions, exit_engine, manager=_MANAGER)

    assert_frame_equal(accounting, accounting_before)
    assert_frame_equal(positions, positions_before)
    assert_frame_equal(exit_engine, exit_before)


def test_multiple_timestamps_sorted_by_open_time() -> None:
    """Output rows are sorted by open_time ascending."""
    t0, t2, t1 = _open_time(0), _open_time(2), _open_time(1)
    result = _build(
        SimpleBacktestingEngine(),
        accounting=_accounting_frame(open_times=[t2, t0, t1]),
        exit_engine=_exit_engine_frame(open_times=[t1, t0, t2]),
    )
    assert result["open_time"].to_list() == [t0, t1, t2]


def test_trade_count_increases_over_time() -> None:
    """trade_count accumulates as positions close at successive timestamps."""
    t0, t1, t2 = _open_time(0), _open_time(1), _open_time(2)
    result = _build(
        SimpleBacktestingEngine(),
        accounting=_accounting_frame(open_times=[t0, t1, t2]),
        exit_engine=_exit_engine_frame(open_times=[t0, t1, t2]),
        positions=_positions_frame(
            position_ids=["pos-a", "pos-b"],
            statuses=["CLOSED", "CLOSED"],
            realized_pnls=[100.0, 50.0],
            closed_ats=[t0, t2],
        ),
    )
    assert result["trade_count"].to_list() == [1, 1, 2]


def test_closed_position_uses_updated_at_when_closed_at_null() -> None:
    """CLOSED positions fall back to updated_at when closed_at is NULL."""
    t0, t1 = _open_time(0), _open_time(1)
    result = _build(
        SimpleBacktestingEngine(),
        accounting=_accounting_frame(open_times=[t0, t1]),
        exit_engine=_exit_engine_frame(open_times=[t0, t1]),
        positions=_positions_frame(
            statuses=["CLOSED"],
            realized_pnls=[75.0],
            closed_ats=[None],
            updated_ats=[t1],
        ),
    )
    assert result["trade_count"].to_list()[0] == 0
    assert result["trade_count"].to_list()[1] == 1


def test_closed_position_uses_opened_at_when_other_timestamps_null() -> None:
    """CLOSED positions fall back to opened_at when closed_at and updated_at are NULL."""
    t0 = _open_time(0)
    result = _build(
        SimpleBacktestingEngine(),
        accounting=_accounting_frame(open_times=[t0]),
        exit_engine=_exit_engine_frame(open_times=[t0]),
        positions=_positions_frame(
            statuses=["CLOSED"],
            realized_pnls=[40.0],
            opened_ats=[t0],
            updated_ats=[None],
            closed_ats=[None],
        ),
    )
    assert result["trade_count"].to_list() == [1]


def test_zero_realized_pnl_trade_not_counted_as_win_or_loss() -> None:
    """Trades with zero realized PnL do not increment winning or losing counts."""
    t0 = _open_time(0)
    result = _build(
        SimpleBacktestingEngine(),
        accounting=_accounting_frame(open_times=[t0]),
        exit_engine=_exit_engine_frame(open_times=[t0]),
        positions=_positions_frame(
            statuses=["CLOSED"],
            realized_pnls=[0.0],
            closed_ats=[t0],
        ),
    )
    assert result["trade_count"].to_list() == [1]
    assert result["winning_trades"].to_list() == [0]
    assert result["losing_trades"].to_list() == [0]
