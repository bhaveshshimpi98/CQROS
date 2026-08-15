"""Unit tests for CQROS ``SimpleExitEngine``."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.exit_engine import (
    ACCOUNTING_INPUT_COLUMNS,
    CANONICAL_COLUMN_ORDER,
    MERGED_EXIT_ENGINE_SCHEMA,
    PORTFOLIO_RISK_INPUT_COLUMNS,
    POSITION_INPUT_COLUMNS,
    PRIORITY_ALPHA_DECAY,
    PRIORITY_BREAK_EVEN,
    PRIORITY_COOLDOWN,
    PRIORITY_HOLD,
    PRIORITY_PORTFOLIO_SHUTDOWN,
    PRIORITY_TAKE_PROFIT,
    PRIORITY_TRAILING_STOP,
    PYRAMIDING_INPUT_COLUMNS,
    TRADE_MANAGEMENT_INPUT_COLUMNS,
    ExitAction,
    ExitEngineValidationError,
    ExitReason,
    SimpleExitEngine,
    validate_accounting_frame,
    validate_portfolio_risk_frame,
    validate_position_frame,
    validate_pyramiding_frame,
    validate_trade_management_frame,
)

_TIMEFRAME = "1h"
_MANAGER = "simple"
_ENTRY_PRICE = 100.0
_POSITION_ID = "pos-00000001"


def _open_time(index: int = 0) -> datetime:
    """Build a deterministic UTC open_time for row ``index``."""
    return datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=index)


def _positions_frame(
    *,
    symbols: list[str] | None = None,
    position_ids: list[str] | None = None,
    sides: list[str] | None = None,
) -> pl.DataFrame:
    """Build a minimal position frame for engine tests."""
    symbols = symbols if symbols is not None else ["BTCUSDT"]
    row_count = len(symbols)
    position_ids = position_ids if position_ids is not None else [_POSITION_ID] * row_count
    sides = sides if sides is not None else ["LONG"] * row_count
    return pl.DataFrame(
        {
            "symbol": symbols,
            "timeframe": [_TIMEFRAME] * row_count,
            "position_id": position_ids,
            "side": sides,
        }
    )


def _accounting_frame(
    *,
    symbols: list[str] | None = None,
    position_ids: list[str] | None = None,
    open_times: list[datetime] | None = None,
    position_statuses: list[str] | None = None,
    quantities: list[float] | None = None,
    entry_prices: list[float] | None = None,
) -> pl.DataFrame:
    """Build a minimal accounting frame for engine tests."""
    symbols = symbols if symbols is not None else ["BTCUSDT"]
    row_count = len(symbols)
    position_ids = position_ids if position_ids is not None else [_POSITION_ID] * row_count
    open_times = open_times if open_times is not None else [_open_time(i) for i in range(row_count)]
    position_statuses = position_statuses if position_statuses is not None else ["OPEN"] * row_count
    quantities = quantities if quantities is not None else [1.0] * row_count
    entry_prices = entry_prices if entry_prices is not None else [_ENTRY_PRICE] * row_count
    return pl.DataFrame(
        {
            "symbol": symbols,
            "timeframe": [_TIMEFRAME] * row_count,
            "open_time": open_times,
            "position_id": position_ids,
            "position_status": position_statuses,
            "quantity": quantities,
            "average_entry_price": entry_prices,
        }
    )


def _portfolio_risk_frame(
    *,
    symbols: list[str] | None = None,
    position_ids: list[str] | None = None,
    open_times: list[datetime] | None = None,
    risk_states: list[str] | None = None,
    shutdown_reasons: list[str | None] | None = None,
    cooldown_untils: list[object] | None = None,
) -> pl.DataFrame:
    """Build a minimal portfolio-risk frame for engine tests."""
    symbols = symbols if symbols is not None else ["BTCUSDT"]
    row_count = len(symbols)
    position_ids = position_ids if position_ids is not None else [_POSITION_ID] * row_count
    open_times = open_times if open_times is not None else [_open_time(i) for i in range(row_count)]
    risk_states = risk_states if risk_states is not None else ["NORMAL"] * row_count
    shutdown_reasons = shutdown_reasons if shutdown_reasons is not None else [None] * row_count
    cooldown_untils = cooldown_untils if cooldown_untils is not None else [None] * row_count
    return pl.DataFrame(
        {
            "symbol": symbols,
            "timeframe": [_TIMEFRAME] * row_count,
            "open_time": open_times,
            "position_id": position_ids,
            "portfolio_risk_state": risk_states,
            "shutdown_reason": shutdown_reasons,
            "cooldown_until": cooldown_untils,
        }
    )


def _trade_management_frame(
    *,
    symbols: list[str] | None = None,
    position_ids: list[str] | None = None,
    open_times: list[datetime] | None = None,
    current_prices: list[float] | None = None,
    management_actions: list[str] | None = None,
    action_reasons: list[str | None] | None = None,
) -> pl.DataFrame:
    """Build a minimal trade-management frame for engine tests."""
    symbols = symbols if symbols is not None else ["BTCUSDT"]
    row_count = len(symbols)
    position_ids = position_ids if position_ids is not None else [_POSITION_ID] * row_count
    open_times = open_times if open_times is not None else [_open_time(i) for i in range(row_count)]
    current_prices = current_prices if current_prices is not None else [_ENTRY_PRICE] * row_count
    management_actions = (
        management_actions if management_actions is not None else ["NONE"] * row_count
    )
    action_reasons = action_reasons if action_reasons is not None else [None] * row_count
    return pl.DataFrame(
        {
            "symbol": symbols,
            "timeframe": [_TIMEFRAME] * row_count,
            "open_time": open_times,
            "position_id": position_ids,
            "current_price": current_prices,
            "management_action": management_actions,
            "action_reason": action_reasons,
        }
    )


def _pyramiding_frame(
    *,
    symbols: list[str] | None = None,
    position_ids: list[str] | None = None,
    open_times: list[datetime] | None = None,
    reasons: list[str] | None = None,
) -> pl.DataFrame:
    """Build a minimal pyramiding frame for engine tests."""
    symbols = symbols if symbols is not None else ["BTCUSDT"]
    row_count = len(symbols)
    position_ids = position_ids if position_ids is not None else [_POSITION_ID] * row_count
    open_times = open_times if open_times is not None else [_open_time(i) for i in range(row_count)]
    reasons = reasons if reasons is not None else ["INSUFFICIENT_PROFIT"] * row_count
    return pl.DataFrame(
        {
            "symbol": symbols,
            "timeframe": [_TIMEFRAME] * row_count,
            "open_time": open_times,
            "position_id": position_ids,
            "reason": reasons,
        }
    )


def _evaluate(
    engine: SimpleExitEngine,
    *,
    positions: pl.DataFrame | None = None,
    accounting: pl.DataFrame | None = None,
    portfolio_risk: pl.DataFrame | None = None,
    trade_management: pl.DataFrame | None = None,
    pyramiding: pl.DataFrame | None = None,
    manager: str = _MANAGER,
) -> pl.DataFrame:
    """Evaluate with default companion frames for single-row use-cases."""
    return engine.evaluate(
        positions if positions is not None else _positions_frame(),
        accounting if accounting is not None else _accounting_frame(),
        portfolio_risk if portfolio_risk is not None else _portfolio_risk_frame(),
        trade_management if trade_management is not None else _trade_management_frame(),
        pyramiding if pyramiding is not None else _pyramiding_frame(),
        manager=manager,
    )


# ---------------------------------------------------------------------------
# Input column contracts
# ---------------------------------------------------------------------------


def test_input_columns_contract() -> None:
    """Input column constants enumerate every column the engine consumes."""
    assert "symbol" in ACCOUNTING_INPUT_COLUMNS
    assert "position_id" in ACCOUNTING_INPUT_COLUMNS
    assert "position_status" in ACCOUNTING_INPUT_COLUMNS
    assert "quantity" in ACCOUNTING_INPUT_COLUMNS
    assert "average_entry_price" in ACCOUNTING_INPUT_COLUMNS

    assert "symbol" in POSITION_INPUT_COLUMNS
    assert "position_id" in POSITION_INPUT_COLUMNS
    assert "side" in POSITION_INPUT_COLUMNS

    assert "portfolio_risk_state" in PORTFOLIO_RISK_INPUT_COLUMNS
    assert "shutdown_reason" in PORTFOLIO_RISK_INPUT_COLUMNS
    assert "cooldown_until" in PORTFOLIO_RISK_INPUT_COLUMNS

    assert "current_price" in TRADE_MANAGEMENT_INPUT_COLUMNS
    assert "management_action" in TRADE_MANAGEMENT_INPUT_COLUMNS
    assert "action_reason" in TRADE_MANAGEMENT_INPUT_COLUMNS

    assert "position_id" in PYRAMIDING_INPUT_COLUMNS
    assert "reason" in PYRAMIDING_INPUT_COLUMNS


# ---------------------------------------------------------------------------
# Frame validator tests
# ---------------------------------------------------------------------------


def test_validate_frames_reject_non_dataframe() -> None:
    """Frame validators reject non-DataFrame inputs with EXIT_FRAME_TYPE."""
    for validator in (
        validate_accounting_frame,
        validate_position_frame,
        validate_portfolio_risk_frame,
        validate_trade_management_frame,
        validate_pyramiding_frame,
    ):
        with pytest.raises(ExitEngineValidationError) as exc_info:
            validator("not-a-frame")  # type: ignore[arg-type]
        assert exc_info.value.error_code == "EXIT_FRAME_TYPE"


def test_validate_frames_reject_empty_dataframe() -> None:
    """Frame validators reject DataFrames with zero rows with EXIT_FRAME_EMPTY."""
    empty = pl.DataFrame({"symbol": []})
    for validator in (
        validate_accounting_frame,
        validate_position_frame,
        validate_portfolio_risk_frame,
        validate_trade_management_frame,
        validate_pyramiding_frame,
    ):
        with pytest.raises(ExitEngineValidationError) as exc_info:
            validator(empty)
        assert exc_info.value.error_code == "EXIT_FRAME_EMPTY"


# ---------------------------------------------------------------------------
# Manager validation
# ---------------------------------------------------------------------------


def test_evaluate_rejects_blank_manager() -> None:
    """Blank or whitespace-only managers raise ExitEngineValidationError."""
    engine = SimpleExitEngine()
    for blank in ("", "   ", "\t"):
        with pytest.raises(ExitEngineValidationError) as exc_info:
            _evaluate(engine, manager=blank)
        assert exc_info.value.error_code == "EXIT_MANAGER_BLANK"


# ---------------------------------------------------------------------------
# Missing column validation
# ---------------------------------------------------------------------------


def test_evaluate_rejects_missing_accounting_columns() -> None:
    """Missing required accounting columns raise EXIT_MISSING_COLUMNS."""
    engine = SimpleExitEngine()
    with pytest.raises(ExitEngineValidationError) as exc_info:
        _evaluate(engine, accounting=_accounting_frame().drop("average_entry_price"))
    assert exc_info.value.error_code == "EXIT_MISSING_COLUMNS"


def test_evaluate_rejects_missing_position_columns() -> None:
    """Missing required position columns raise EXIT_MISSING_COLUMNS."""
    engine = SimpleExitEngine()
    with pytest.raises(ExitEngineValidationError) as exc_info:
        _evaluate(engine, positions=_positions_frame().drop("side"))
    assert exc_info.value.error_code == "EXIT_MISSING_COLUMNS"


def test_evaluate_rejects_missing_portfolio_risk_columns() -> None:
    """Missing required portfolio-risk columns raise EXIT_MISSING_COLUMNS."""
    engine = SimpleExitEngine()
    with pytest.raises(ExitEngineValidationError) as exc_info:
        _evaluate(
            engine,
            portfolio_risk=_portfolio_risk_frame().drop("shutdown_reason"),
        )
    assert exc_info.value.error_code == "EXIT_MISSING_COLUMNS"


def test_evaluate_rejects_missing_trade_management_columns() -> None:
    """Missing required trade-management columns raise EXIT_MISSING_COLUMNS."""
    engine = SimpleExitEngine()
    with pytest.raises(ExitEngineValidationError) as exc_info:
        _evaluate(
            engine,
            trade_management=_trade_management_frame().drop("action_reason"),
        )
    assert exc_info.value.error_code == "EXIT_MISSING_COLUMNS"


def test_evaluate_rejects_missing_pyramiding_columns() -> None:
    """Missing required pyramiding columns raise EXIT_MISSING_COLUMNS."""
    engine = SimpleExitEngine()
    with pytest.raises(ExitEngineValidationError) as exc_info:
        _evaluate(engine, pyramiding=_pyramiding_frame().drop("reason"))
    assert exc_info.value.error_code == "EXIT_MISSING_COLUMNS"


# ---------------------------------------------------------------------------
# Coverage validation
# ---------------------------------------------------------------------------


def test_missing_position_id_coverage_fails() -> None:
    """Accounting position_ids absent from positions raise EXIT_POSITION_IDS."""
    engine = SimpleExitEngine()
    with pytest.raises(ExitEngineValidationError) as exc_info:
        _evaluate(
            engine,
            accounting=_accounting_frame(position_ids=["pos-missing"]),
            positions=_positions_frame(position_ids=[_POSITION_ID]),
        )
    assert exc_info.value.error_code == "EXIT_POSITION_IDS"


def test_missing_portfolio_risk_coverage_fails() -> None:
    """Missing portfolio-risk rows for open accounting snapshots raise EXIT_RISK_COVERAGE."""
    engine = SimpleExitEngine()
    with pytest.raises(ExitEngineValidationError) as exc_info:
        _evaluate(
            engine,
            portfolio_risk=_portfolio_risk_frame(open_times=[_open_time(99)]),
        )
    assert exc_info.value.error_code == "EXIT_RISK_COVERAGE"


def test_missing_trade_management_coverage_fails() -> None:
    """Missing TM rows for open accounting snapshots raise EXIT_TM_COVERAGE."""
    engine = SimpleExitEngine()
    with pytest.raises(ExitEngineValidationError) as exc_info:
        _evaluate(
            engine,
            trade_management=_trade_management_frame(open_times=[_open_time(99)]),
        )
    assert exc_info.value.error_code == "EXIT_TM_COVERAGE"


def test_missing_pyramiding_coverage_fails() -> None:
    """Missing pyramiding rows for open accounting snapshots raise EXIT_PYRAMID_COVERAGE."""
    engine = SimpleExitEngine()
    with pytest.raises(ExitEngineValidationError) as exc_info:
        _evaluate(
            engine,
            pyramiding=_pyramiding_frame(open_times=[_open_time(99)]),
        )
    assert exc_info.value.error_code == "EXIT_PYRAMID_COVERAGE"


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_engine_rejects_invalid_initial_risk_percent() -> None:
    """initial_risk_percent must be finite in [0, 1); invalid values raise at construction."""
    with pytest.raises(ExitEngineValidationError) as exc_info:
        SimpleExitEngine(initial_risk_percent=-0.01)
    assert exc_info.value.error_code == "EXIT_LIMIT_RANGE"

    with pytest.raises(ExitEngineValidationError) as exc_info:
        SimpleExitEngine(initial_risk_percent=1.0)
    assert exc_info.value.error_code == "EXIT_LIMIT_RANGE"

    with pytest.raises(ExitEngineValidationError) as exc_info:
        SimpleExitEngine(initial_risk_percent=float("nan"))
    assert exc_info.value.error_code == "EXIT_LIMIT_NON_FINITE"


def test_engine_rejects_invalid_take_profit_multiple() -> None:
    """take_profit_multiple must be finite > 0; invalid values raise at construction."""
    with pytest.raises(ExitEngineValidationError) as exc_info:
        SimpleExitEngine(take_profit_multiple=0.0)
    assert exc_info.value.error_code == "EXIT_LIMIT_RANGE"

    with pytest.raises(ExitEngineValidationError) as exc_info:
        SimpleExitEngine(take_profit_multiple=float("inf"))
    assert exc_info.value.error_code == "EXIT_LIMIT_RANGE"

    with pytest.raises(ExitEngineValidationError) as exc_info:
        SimpleExitEngine(take_profit_multiple=True)  # type: ignore[arg-type]
    assert exc_info.value.error_code == "EXIT_LIMIT_NON_FINITE"


def test_engine_rejects_invalid_partial_exit_percent() -> None:
    """partial_exit_percent must be finite in [0, 1]; 1.0 is valid, < 0 is not."""
    with pytest.raises(ExitEngineValidationError) as exc_info:
        SimpleExitEngine(partial_exit_percent=-0.1)
    assert exc_info.value.error_code == "EXIT_LIMIT_RANGE"

    with pytest.raises(ExitEngineValidationError) as exc_info:
        SimpleExitEngine(partial_exit_percent=1.5)
    assert exc_info.value.error_code == "EXIT_LIMIT_RANGE"


# ---------------------------------------------------------------------------
# Rule 1: Portfolio shutdown → FULL_EXIT / PORTFOLIO_SHUTDOWN
# ---------------------------------------------------------------------------


def test_portfolio_shutdown_produces_full_exit() -> None:
    """SHUTDOWN risk state with non-COOLDOWN shutdown_reason triggers FULL_EXIT."""
    result = _evaluate(
        SimpleExitEngine(),
        portfolio_risk=_portfolio_risk_frame(
            risk_states=["SHUTDOWN"],
            shutdown_reasons=["STOPPED"],
        ),
        trade_management=_trade_management_frame(current_prices=[102.0]),
    )
    assert result.height == 1
    assert result["exit_action"].to_list() == [ExitAction.FULL_EXIT.value]
    assert result["exit_reason"].to_list() == [ExitReason.PORTFOLIO_SHUTDOWN.value]
    assert result["recommended_percent"].to_list() == [1.0]
    assert result["recommended_quantity"].to_list() == [1.0]
    assert result["priority"].to_list() == [PRIORITY_PORTFOLIO_SHUTDOWN]


def test_portfolio_shutdown_with_various_non_cooldown_reasons() -> None:
    """Non-COOLDOWN shutdown reasons all trigger FULL_EXIT/PORTFOLIO_SHUTDOWN."""
    engine = SimpleExitEngine()
    for reason in ("STOPPED", "RISK_LIMIT", "MAX_DRAWDOWN", "MANUAL"):
        result = _evaluate(
            engine,
            portfolio_risk=_portfolio_risk_frame(
                risk_states=["SHUTDOWN"],
                shutdown_reasons=[reason],
            ),
            trade_management=_trade_management_frame(current_prices=[102.0]),
        )
        assert result["exit_action"].to_list() == [ExitAction.FULL_EXIT.value], reason
        assert result["exit_reason"].to_list() == [ExitReason.PORTFOLIO_SHUTDOWN.value], reason


# ---------------------------------------------------------------------------
# Rule 2: Cooldown → HOLD / COOLDOWN
# ---------------------------------------------------------------------------


def test_cooldown_shutdown_reason_produces_hold() -> None:
    """shutdown_reason=COOLDOWN emits HOLD/COOLDOWN regardless of risk state."""
    result = _evaluate(
        SimpleExitEngine(),
        portfolio_risk=_portfolio_risk_frame(
            risk_states=["NORMAL"],
            shutdown_reasons=["COOLDOWN"],
        ),
        trade_management=_trade_management_frame(current_prices=[120.0]),
    )
    assert result.height == 1
    assert result["exit_action"].to_list() == [ExitAction.HOLD.value]
    assert result["exit_reason"].to_list() == [ExitReason.COOLDOWN.value]
    assert result["recommended_percent"].to_list() == [0.0]
    assert result["recommended_quantity"].to_list() == [0.0]
    assert result["priority"].to_list() == [PRIORITY_COOLDOWN]


def test_cooldown_overrides_shutdown_state() -> None:
    """COOLDOWN shutdown_reason takes priority even when risk_state=SHUTDOWN."""
    result = _evaluate(
        SimpleExitEngine(),
        portfolio_risk=_portfolio_risk_frame(
            risk_states=["SHUTDOWN"],
            shutdown_reasons=["COOLDOWN"],
        ),
        trade_management=_trade_management_frame(current_prices=[120.0]),
    )
    assert result["exit_action"].to_list() == [ExitAction.HOLD.value]
    assert result["exit_reason"].to_list() == [ExitReason.COOLDOWN.value]
    assert result["priority"].to_list() == [PRIORITY_COOLDOWN]


# ---------------------------------------------------------------------------
# Rule 3: Trailing stop → FULL_EXIT / TRAILING_STOP
# ---------------------------------------------------------------------------


def test_trailing_stop_action_reason_produces_full_exit() -> None:
    """action_reason=TRAILING_STOP emits FULL_EXIT/TRAILING_STOP."""
    result = _evaluate(
        SimpleExitEngine(),
        trade_management=_trade_management_frame(
            current_prices=[102.0],
            management_actions=["HOLD"],
            action_reasons=["TRAILING_STOP"],
        ),
    )
    assert result.height == 1
    assert result["exit_action"].to_list() == [ExitAction.FULL_EXIT.value]
    assert result["exit_reason"].to_list() == [ExitReason.TRAILING_STOP.value]
    assert result["recommended_percent"].to_list() == [1.0]
    assert result["priority"].to_list() == [PRIORITY_TRAILING_STOP]


# ---------------------------------------------------------------------------
# Rule 4: Break-even → FULL_EXIT / BREAK_EVEN
# ---------------------------------------------------------------------------


def test_breakeven_action_reason_produces_full_exit() -> None:
    """action_reason=BREAKEVEN emits FULL_EXIT/BREAK_EVEN."""
    result = _evaluate(
        SimpleExitEngine(),
        trade_management=_trade_management_frame(
            current_prices=[105.0],
            management_actions=["HOLD"],
            action_reasons=["BREAKEVEN"],
        ),
    )
    assert result.height == 1
    assert result["exit_action"].to_list() == [ExitAction.FULL_EXIT.value]
    assert result["exit_reason"].to_list() == [ExitReason.BREAK_EVEN.value]
    assert result["recommended_percent"].to_list() == [1.0]
    assert result["priority"].to_list() == [PRIORITY_BREAK_EVEN]


# ---------------------------------------------------------------------------
# Rule 5: Take profit (risk/reward >= 3.0) → PARTIAL_EXIT / TAKE_PROFIT
# ---------------------------------------------------------------------------


def test_take_profit_triggers_at_rr_threshold() -> None:
    """RR >= 3.0 (current=115, entry=100, risk=5%) triggers PARTIAL_EXIT/TAKE_PROFIT."""
    result = _evaluate(
        SimpleExitEngine(),
        accounting=_accounting_frame(entry_prices=[100.0], quantities=[2.0]),
        trade_management=_trade_management_frame(current_prices=[115.0]),
    )
    assert result.height == 1
    assert result["exit_action"].to_list() == [ExitAction.PARTIAL_EXIT.value]
    assert result["exit_reason"].to_list() == [ExitReason.TAKE_PROFIT.value]
    assert result["recommended_percent"].to_list() == [pytest.approx(0.50)]
    assert result["recommended_quantity"].to_list() == [pytest.approx(1.0)]
    assert result["priority"].to_list() == [PRIORITY_TAKE_PROFIT]


def test_take_profit_below_threshold_produces_hold() -> None:
    """current=114 with entry=100 and risk=5% does not reach RR=3.0 → HOLD."""
    result = _evaluate(
        SimpleExitEngine(),
        accounting=_accounting_frame(entry_prices=[100.0]),
        trade_management=_trade_management_frame(current_prices=[114.0]),
    )
    assert result["exit_action"].to_list() == [ExitAction.HOLD.value]
    assert result["exit_reason"].to_list() == [ExitReason.NONE.value]


def test_take_profit_custom_multiple() -> None:
    """Custom take_profit_multiple=2.0 triggers at current=110 (entry=100, risk=5%)."""
    result = _evaluate(
        SimpleExitEngine(take_profit_multiple=2.0),
        accounting=_accounting_frame(entry_prices=[100.0]),
        trade_management=_trade_management_frame(current_prices=[110.0]),
    )
    assert result["exit_action"].to_list() == [ExitAction.PARTIAL_EXIT.value]
    assert result["exit_reason"].to_list() == [ExitReason.TAKE_PROFIT.value]


# ---------------------------------------------------------------------------
# Rule 6: Alpha decay → FULL_EXIT / ALPHA_DECAY
# ---------------------------------------------------------------------------


def test_alpha_decay_action_reason_produces_full_exit() -> None:
    """action_reason=ALPHA_DECAY emits FULL_EXIT/ALPHA_DECAY."""
    result = _evaluate(
        SimpleExitEngine(),
        trade_management=_trade_management_frame(
            current_prices=[100.0],
            action_reasons=["ALPHA_DECAY"],
        ),
    )
    assert result.height == 1
    assert result["exit_action"].to_list() == [ExitAction.FULL_EXIT.value]
    assert result["exit_reason"].to_list() == [ExitReason.ALPHA_DECAY.value]
    assert result["recommended_percent"].to_list() == [1.0]
    assert result["priority"].to_list() == [PRIORITY_ALPHA_DECAY]


# ---------------------------------------------------------------------------
# Default: HOLD / NONE
# ---------------------------------------------------------------------------


def test_default_hold_when_no_rule_fires() -> None:
    """No exit rule matched → HOLD/NONE with recommended_percent=0 and priority=0."""
    result = _evaluate(
        SimpleExitEngine(),
        trade_management=_trade_management_frame(current_prices=[102.0]),
    )
    assert result.height == 1
    assert result["exit_action"].to_list() == [ExitAction.HOLD.value]
    assert result["exit_reason"].to_list() == [ExitReason.NONE.value]
    assert result["recommended_percent"].to_list() == [0.0]
    assert result["recommended_quantity"].to_list() == [0.0]
    assert result["priority"].to_list() == [PRIORITY_HOLD]


# ---------------------------------------------------------------------------
# No OPEN positions → empty output
# ---------------------------------------------------------------------------


def test_no_open_positions_returns_empty_frame() -> None:
    """All accounting rows with CLOSED status produce an empty output frame."""
    result = _evaluate(
        SimpleExitEngine(),
        accounting=_accounting_frame(position_statuses=["CLOSED"]),
    )
    assert result.height == 0
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result.schema == MERGED_EXIT_ENGINE_SCHEMA


def test_mixed_open_and_closed_rows_filters_to_open_only() -> None:
    """Only OPEN accounting rows produce exit recommendation rows."""
    open_time_0 = _open_time(0)
    open_time_1 = _open_time(1)
    result = SimpleExitEngine().evaluate(
        _positions_frame(
            symbols=["BTCUSDT", "ETHUSDT"],
            position_ids=["pos-00000001", "pos-00000002"],
        ),
        _accounting_frame(
            symbols=["BTCUSDT", "ETHUSDT"],
            position_ids=["pos-00000001", "pos-00000002"],
            open_times=[open_time_0, open_time_1],
            position_statuses=["OPEN", "CLOSED"],
        ),
        _portfolio_risk_frame(
            symbols=["BTCUSDT"],
            position_ids=["pos-00000001"],
            open_times=[open_time_0],
        ),
        _trade_management_frame(
            symbols=["BTCUSDT"],
            position_ids=["pos-00000001"],
            open_times=[open_time_0],
            current_prices=[102.0],
        ),
        _pyramiding_frame(
            symbols=["BTCUSDT"],
            position_ids=["pos-00000001"],
            open_times=[open_time_0],
        ),
        manager=_MANAGER,
    )
    assert result.height == 1
    assert result["position_id"].to_list() == ["pos-00000001"]


# ---------------------------------------------------------------------------
# Output schema, invariants, and metadata
# ---------------------------------------------------------------------------


def test_output_canonical_ordering_and_dtype_schema() -> None:
    """Engine output enforces canonical column order and MERGED_EXIT_ENGINE_SCHEMA dtypes."""
    result = _evaluate(
        SimpleExitEngine(),
        trade_management=_trade_management_frame(current_prices=[102.0]),
    )
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result.schema == MERGED_EXIT_ENGINE_SCHEMA
    assert result.schema["open_time"] == pl.Datetime("us", "UTC")
    assert result.schema["created_at"] == pl.Datetime("us", "UTC")
    assert result.schema["priority"] == pl.Int64
    assert result.schema["recommended_percent"] == pl.Float64


def test_created_at_equals_open_time() -> None:
    """created_at is set to the accounting open_time for deterministic outputs."""
    open_time = _open_time(5)
    result = _evaluate(
        SimpleExitEngine(),
        accounting=_accounting_frame(open_times=[open_time]),
        portfolio_risk=_portfolio_risk_frame(open_times=[open_time]),
        trade_management=_trade_management_frame(open_times=[open_time], current_prices=[102.0]),
        pyramiding=_pyramiding_frame(open_times=[open_time]),
    )
    created_at = result["created_at"].to_list()[0]
    assert created_at == open_time


def test_manager_is_stamped_on_every_row() -> None:
    """manager column contains the injected manager identity on every row."""
    result = _evaluate(
        SimpleExitEngine(),
        manager="custom-manager",
        trade_management=_trade_management_frame(current_prices=[102.0]),
    )
    assert result["manager"].to_list() == ["custom-manager"]


def test_inputs_are_immutable() -> None:
    """evaluate must not mutate any caller-supplied input frame."""
    positions = _positions_frame()
    accounting = _accounting_frame()
    portfolio_risk = _portfolio_risk_frame()
    trade_management = _trade_management_frame(current_prices=[102.0])
    pyramiding = _pyramiding_frame()

    positions_before = positions.clone()
    accounting_before = accounting.clone()
    risk_before = portfolio_risk.clone()
    tm_before = trade_management.clone()
    pyramid_before = pyramiding.clone()

    SimpleExitEngine().evaluate(
        positions,
        accounting,
        portfolio_risk,
        trade_management,
        pyramiding,
        manager=_MANAGER,
    )

    assert_frame_equal(positions, positions_before)
    assert_frame_equal(accounting, accounting_before)
    assert_frame_equal(portfolio_risk, risk_before)
    assert_frame_equal(trade_management, tm_before)
    assert_frame_equal(pyramiding, pyramid_before)


# ---------------------------------------------------------------------------
# Multiple symbols / multiple rows
# ---------------------------------------------------------------------------


def test_multiple_positions_evaluated_independently() -> None:
    """Multiple open positions produce one exit row per open snapshot."""
    open_time = _open_time(0)
    result = SimpleExitEngine().evaluate(
        _positions_frame(
            symbols=["BTCUSDT", "ETHUSDT"],
            position_ids=["pos-00000001", "pos-00000002"],
        ),
        _accounting_frame(
            symbols=["BTCUSDT", "ETHUSDT"],
            position_ids=["pos-00000001", "pos-00000002"],
            open_times=[open_time, open_time],
            entry_prices=[100.0, 100.0],
        ),
        _portfolio_risk_frame(
            symbols=["BTCUSDT", "ETHUSDT"],
            position_ids=["pos-00000001", "pos-00000002"],
            open_times=[open_time, open_time],
        ),
        _trade_management_frame(
            symbols=["BTCUSDT", "ETHUSDT"],
            position_ids=["pos-00000001", "pos-00000002"],
            open_times=[open_time, open_time],
            current_prices=[102.0, 115.0],
        ),
        _pyramiding_frame(
            symbols=["BTCUSDT", "ETHUSDT"],
            position_ids=["pos-00000001", "pos-00000002"],
            open_times=[open_time, open_time],
        ),
        manager=_MANAGER,
    )
    assert result.height == 2
    by_pos = result.sort("position_id")
    actions = by_pos["exit_action"].to_list()
    assert ExitAction.HOLD.value in actions
    assert ExitAction.PARTIAL_EXIT.value in actions


def test_recommended_quantity_equals_quantity_times_percent() -> None:
    """recommended_quantity = quantity * recommended_percent for take-profit."""
    result = _evaluate(
        SimpleExitEngine(),
        accounting=_accounting_frame(entry_prices=[100.0], quantities=[4.0]),
        trade_management=_trade_management_frame(current_prices=[115.0]),
    )
    qty = result["recommended_quantity"].to_list()[0]
    pct = result["recommended_percent"].to_list()[0]
    assert qty == pytest.approx(4.0 * pct)
    assert qty >= 0.0
    assert 0.0 <= pct <= 1.0


def test_recommended_quantity_is_non_negative_for_all_rules() -> None:
    """recommended_quantity is always >= 0 for every rule outcome."""
    engine = SimpleExitEngine()
    scenarios = [
        # HOLD
        _trade_management_frame(current_prices=[102.0]),
        # TRAILING_STOP
        _trade_management_frame(current_prices=[102.0], action_reasons=["TRAILING_STOP"]),
        # ALPHA_DECAY
        _trade_management_frame(current_prices=[102.0], action_reasons=["ALPHA_DECAY"]),
        # TAKE_PROFIT
        _trade_management_frame(current_prices=[115.0]),
    ]
    for tm in scenarios:
        result = _evaluate(engine, trade_management=tm)
        qty = result["recommended_quantity"].to_list()[0]
        pct = result["recommended_percent"].to_list()[0]
        assert qty >= 0.0, f"Negative quantity for tm={tm}"
        assert 0.0 <= pct <= 1.0, f"Percent out of range for tm={tm}"
