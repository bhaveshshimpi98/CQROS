"""Unit tests for CQROS ``SimplePyramidingEngine``."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.pyramiding import (
    ACCOUNTING_INPUT_COLUMNS,
    CANONICAL_COLUMN_ORDER,
    MARKET_PRICE_INPUT_COLUMNS,
    MERGED_PYRAMIDING_SCHEMA,
    PORTFOLIO_RISK_INPUT_COLUMNS,
    POSITION_INPUT_COLUMNS,
    TRADE_MANAGEMENT_INPUT_COLUMNS,
    PyramidingReason,
    PyramidingValidationError,
    SimplePyramidingEngine,
    validate_accounting_frame,
    validate_market_price_frame,
    validate_portfolio_risk_frame,
    validate_position_frame,
    validate_trade_management_frame,
)

_TIMEFRAME = "1h"
_MANAGER = "simple"
_ENTRY_PRICE = 100.0
_POSITION_ID = "pos-00000001"


def _open_time(index: int) -> datetime:
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
    """Build an accounting frame for engine tests."""
    symbols = symbols if symbols is not None else ["BTCUSDT"]
    row_count = len(symbols)
    position_ids = position_ids if position_ids is not None else [_POSITION_ID] * row_count
    open_times = (
        open_times if open_times is not None else [_open_time(index) for index in range(row_count)]
    )
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
    """Build a portfolio-risk frame for engine tests."""
    symbols = symbols if symbols is not None else ["BTCUSDT"]
    row_count = len(symbols)
    position_ids = position_ids if position_ids is not None else [_POSITION_ID] * row_count
    open_times = (
        open_times if open_times is not None else [_open_time(index) for index in range(row_count)]
    )
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
    management_actions: list[str] | None = None,
    action_reasons: list[str | None] | None = None,
) -> pl.DataFrame:
    """Build a trade-management frame for engine tests."""
    symbols = symbols if symbols is not None else ["BTCUSDT"]
    row_count = len(symbols)
    position_ids = position_ids if position_ids is not None else [_POSITION_ID] * row_count
    open_times = (
        open_times if open_times is not None else [_open_time(index) for index in range(row_count)]
    )
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
            "management_action": management_actions,
            "action_reason": action_reasons,
        }
    )


def _market_prices_frame(
    *,
    symbols: list[str] | None = None,
    open_times: list[datetime] | None = None,
    prices: list[float] | None = None,
    highs: list[float] | None = None,
) -> pl.DataFrame:
    """Build a market-price frame (with price and high columns) for engine tests."""
    symbols = symbols if symbols is not None else ["BTCUSDT"]
    row_count = len(symbols)
    open_times = (
        open_times if open_times is not None else [_open_time(index) for index in range(row_count)]
    )
    prices = prices if prices is not None else [_ENTRY_PRICE] * row_count
    highs = highs if highs is not None else prices
    return pl.DataFrame(
        {
            "symbol": symbols,
            "timeframe": [_TIMEFRAME] * row_count,
            "open_time": open_times,
            "price": prices,
            "high": highs,
        }
    )


def _evaluate(
    engine: SimplePyramidingEngine,
    *,
    positions: pl.DataFrame | None = None,
    accounting: pl.DataFrame | None = None,
    portfolio_risk: pl.DataFrame | None = None,
    trade_management: pl.DataFrame | None = None,
    market_prices: pl.DataFrame | None = None,
    manager: str = _MANAGER,
) -> pl.DataFrame:
    """Evaluate with default companion frames for single-row use-cases."""
    return engine.evaluate(
        positions if positions is not None else _positions_frame(),
        accounting if accounting is not None else _accounting_frame(),
        portfolio_risk if portfolio_risk is not None else _portfolio_risk_frame(),
        trade_management if trade_management is not None else _trade_management_frame(),
        market_prices if market_prices is not None else _market_prices_frame(),
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

    assert "management_action" in TRADE_MANAGEMENT_INPUT_COLUMNS
    assert "action_reason" in TRADE_MANAGEMENT_INPUT_COLUMNS

    assert "price" in MARKET_PRICE_INPUT_COLUMNS
    assert "high" in MARKET_PRICE_INPUT_COLUMNS


# ---------------------------------------------------------------------------
# Frame validator tests
# ---------------------------------------------------------------------------


def test_validate_frames_reject_non_dataframe() -> None:
    """Frame validators reject non-DataFrame inputs."""
    for validator in (
        validate_accounting_frame,
        validate_position_frame,
        validate_portfolio_risk_frame,
        validate_trade_management_frame,
        validate_market_price_frame,
    ):
        with pytest.raises(PyramidingValidationError) as exc_info:
            validator("not-a-frame")  # type: ignore[arg-type]
        assert exc_info.value.error_code == "PYR_FRAME_TYPE"


def test_validate_frames_reject_empty_dataframe() -> None:
    """Frame validators reject DataFrames with zero rows."""
    empty = pl.DataFrame({"symbol": []})
    for validator in (
        validate_accounting_frame,
        validate_position_frame,
        validate_portfolio_risk_frame,
        validate_trade_management_frame,
        validate_market_price_frame,
    ):
        with pytest.raises(PyramidingValidationError) as exc_info:
            validator(empty)
        assert exc_info.value.error_code == "PYR_FRAME_EMPTY"


# ---------------------------------------------------------------------------
# Manager and missing-column validation
# ---------------------------------------------------------------------------


def test_evaluate_rejects_blank_manager() -> None:
    """Blank or whitespace-only managers raise PyramidingValidationError."""
    engine = SimplePyramidingEngine()
    for blank in ("", "   ", "\t"):
        with pytest.raises(PyramidingValidationError) as exc_info:
            _evaluate(engine, manager=blank)
        assert exc_info.value.error_code == "PYR_MANAGER_BLANK"


def test_evaluate_rejects_missing_accounting_columns() -> None:
    """Missing accounting columns raise PYR_MISSING_COLUMNS."""
    engine = SimplePyramidingEngine()
    with pytest.raises(PyramidingValidationError) as exc_info:
        _evaluate(engine, accounting=_accounting_frame().drop("average_entry_price"))
    assert exc_info.value.error_code == "PYR_MISSING_COLUMNS"


def test_evaluate_rejects_missing_position_columns() -> None:
    """Missing position columns raise PYR_MISSING_COLUMNS."""
    engine = SimplePyramidingEngine()
    with pytest.raises(PyramidingValidationError) as exc_info:
        _evaluate(engine, positions=_positions_frame().drop("side"))
    assert exc_info.value.error_code == "PYR_MISSING_COLUMNS"


def test_evaluate_rejects_missing_portfolio_risk_columns() -> None:
    """Missing portfolio-risk columns raise PYR_MISSING_COLUMNS."""
    engine = SimplePyramidingEngine()
    with pytest.raises(PyramidingValidationError) as exc_info:
        _evaluate(
            engine,
            portfolio_risk=_portfolio_risk_frame().drop("shutdown_reason"),
        )
    assert exc_info.value.error_code == "PYR_MISSING_COLUMNS"


def test_evaluate_rejects_missing_trade_management_columns() -> None:
    """Missing trade-management columns raise PYR_MISSING_COLUMNS."""
    engine = SimplePyramidingEngine()
    with pytest.raises(PyramidingValidationError) as exc_info:
        _evaluate(
            engine,
            trade_management=_trade_management_frame().drop("action_reason"),
        )
    assert exc_info.value.error_code == "PYR_MISSING_COLUMNS"


def test_evaluate_rejects_missing_market_price_columns() -> None:
    """Missing market-price columns (including high) raise PYR_MISSING_COLUMNS."""
    engine = SimplePyramidingEngine()
    with pytest.raises(PyramidingValidationError) as exc_info:
        _evaluate(engine, market_prices=_market_prices_frame().drop("high"))
    assert exc_info.value.error_code == "PYR_MISSING_COLUMNS"


def test_missing_position_id_coverage_fails() -> None:
    """Accounting position_ids absent from positions raise PYR_POSITION_IDS."""
    engine = SimplePyramidingEngine()
    with pytest.raises(PyramidingValidationError) as exc_info:
        _evaluate(
            engine,
            accounting=_accounting_frame(position_ids=["pos-missing"]),
            positions=_positions_frame(position_ids=[_POSITION_ID]),
        )
    assert exc_info.value.error_code == "PYR_POSITION_IDS"


def test_missing_market_price_coverage_fails() -> None:
    """Missing market prices for accounting snapshots raise PYR_MARKET_COVERAGE."""
    engine = SimplePyramidingEngine()
    with pytest.raises(PyramidingValidationError) as exc_info:
        _evaluate(
            engine,
            market_prices=_market_prices_frame(open_times=[_open_time(99)]),
        )
    assert exc_info.value.error_code == "PYR_MARKET_COVERAGE"


def test_missing_portfolio_risk_coverage_fails() -> None:
    """Missing portfolio-risk rows for accounting snapshots raise PYR_RISK_COVERAGE."""
    engine = SimplePyramidingEngine()
    with pytest.raises(PyramidingValidationError) as exc_info:
        _evaluate(
            engine,
            portfolio_risk=_portfolio_risk_frame(open_times=[_open_time(99)]),
        )
    assert exc_info.value.error_code == "PYR_RISK_COVERAGE"


def test_missing_trade_management_coverage_fails() -> None:
    """Missing trade-management rows for accounting snapshots raise PYR_TM_COVERAGE."""
    engine = SimplePyramidingEngine()
    with pytest.raises(PyramidingValidationError) as exc_info:
        _evaluate(
            engine,
            trade_management=_trade_management_frame(open_times=[_open_time(99)]),
        )
    assert exc_info.value.error_code == "PYR_TM_COVERAGE"


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_engine_rejects_invalid_max_adds() -> None:
    """max_adds must be a positive integer; invalid values raise at construction."""
    with pytest.raises(PyramidingValidationError) as exc_info:
        SimplePyramidingEngine(max_adds=0)
    assert exc_info.value.error_code == "PYR_MAX_ADDS_INVALID"

    with pytest.raises(PyramidingValidationError) as exc_info:
        SimplePyramidingEngine(max_adds=-1)
    assert exc_info.value.error_code == "PYR_MAX_ADDS_INVALID"

    with pytest.raises(PyramidingValidationError) as exc_info:
        SimplePyramidingEngine(max_adds=True)  # type: ignore[arg-type]
    assert exc_info.value.error_code == "PYR_MAX_ADDS_INVALID"


def test_engine_rejects_invalid_add_fraction() -> None:
    """add_fraction must be finite [0, 1]; invalid values raise at construction."""
    with pytest.raises(PyramidingValidationError) as exc_info:
        SimplePyramidingEngine(add_fraction=-0.1)
    assert exc_info.value.error_code == "PYR_LIMIT_RANGE"

    with pytest.raises(PyramidingValidationError) as exc_info:
        SimplePyramidingEngine(add_fraction=1.5)
    assert exc_info.value.error_code == "PYR_LIMIT_RANGE"

    with pytest.raises(PyramidingValidationError) as exc_info:
        SimplePyramidingEngine(add_fraction=float("nan"))
    assert exc_info.value.error_code == "PYR_LIMIT_NON_FINITE"


def test_engine_rejects_invalid_min_profit_percent() -> None:
    """min_profit_percent must be finite [0, 1); 1.0 is rejected."""
    with pytest.raises(PyramidingValidationError) as exc_info:
        SimplePyramidingEngine(min_profit_percent=1.0)
    assert exc_info.value.error_code == "PYR_LIMIT_RANGE"

    with pytest.raises(PyramidingValidationError) as exc_info:
        SimplePyramidingEngine(min_profit_percent=-0.01)
    assert exc_info.value.error_code == "PYR_LIMIT_RANGE"


# ---------------------------------------------------------------------------
# NOT_ELIGIBLE cases
# ---------------------------------------------------------------------------


def test_closed_position_not_eligible() -> None:
    """CLOSED accounting rows produce NOT_ELIGIBLE regardless of side."""
    result = _evaluate(
        SimplePyramidingEngine(),
        accounting=_accounting_frame(position_statuses=["CLOSED"]),
        market_prices=_market_prices_frame(prices=[120.0], highs=[120.0]),
    )
    assert result.height == 1
    assert result["reason"].to_list() == [PyramidingReason.NOT_ELIGIBLE.value]
    assert result["allow_pyramid"].to_list() == [False]


def test_short_position_not_eligible() -> None:
    """SHORT positions produce NOT_ELIGIBLE even when profit is above threshold."""
    result = _evaluate(
        SimplePyramidingEngine(),
        positions=_positions_frame(sides=["SHORT"]),
        market_prices=_market_prices_frame(prices=[120.0], highs=[120.0]),
    )
    assert result.height == 1
    assert result["reason"].to_list() == [PyramidingReason.NOT_ELIGIBLE.value]
    assert result["allow_pyramid"].to_list() == [False]


# ---------------------------------------------------------------------------
# INSUFFICIENT_PROFIT
# ---------------------------------------------------------------------------


def test_profit_below_threshold_insufficient_profit() -> None:
    """Profit below 5% threshold emits INSUFFICIENT_PROFIT."""
    result = _evaluate(
        SimplePyramidingEngine(),
        market_prices=_market_prices_frame(prices=[102.0], highs=[102.0]),
    )
    assert result["reason"].to_list() == [PyramidingReason.INSUFFICIENT_PROFIT.value]
    assert result["allow_pyramid"].to_list() == [False]
    assert result["add_number"].to_list() == [0]
    assert result["additional_size"].to_list() == [0.0]


# ---------------------------------------------------------------------------
# READY_TO_ADD at exact threshold
# ---------------------------------------------------------------------------


def test_exact_threshold_ready_to_add() -> None:
    """Exactly 5% profit (at entry=100) triggers READY_TO_ADD with add #1 = 0.5 * size."""
    result = _evaluate(
        SimplePyramidingEngine(),
        accounting=_accounting_frame(quantities=[1.0], entry_prices=[100.0]),
        market_prices=_market_prices_frame(prices=[105.0], highs=[105.0]),
    )
    assert result.height == 1
    assert result["reason"].to_list() == [PyramidingReason.READY_TO_ADD.value]
    assert result["allow_pyramid"].to_list() == [True]
    assert result["add_number"].to_list() == [1]
    assert result["position_size"].to_list() == [pytest.approx(1.0)]
    assert result["additional_size"].to_list() == [pytest.approx(0.5)]
    assert result["recommended_size"].to_list() == [pytest.approx(1.5)]
    assert result["profit_pct"].to_list() == [pytest.approx(0.05)]


# ---------------------------------------------------------------------------
# Multiple adds sequence (canonical example)
# ---------------------------------------------------------------------------


def test_multiple_adds_sequence() -> None:
    """Sequential rows grow theoretical size across the CQROS pyramiding example."""
    open_times = [_open_time(i) for i in range(4)]
    prices = [105.0, 110.0, 115.0, 120.0]
    result = SimplePyramidingEngine().evaluate(
        _positions_frame(),
        _accounting_frame(
            symbols=["BTCUSDT"] * 4,
            position_ids=[_POSITION_ID] * 4,
            open_times=open_times,
            quantities=[1.0] * 4,
            entry_prices=[100.0] * 4,
        ),
        _portfolio_risk_frame(
            symbols=["BTCUSDT"] * 4,
            position_ids=[_POSITION_ID] * 4,
            open_times=open_times,
        ),
        _trade_management_frame(
            symbols=["BTCUSDT"] * 4,
            position_ids=[_POSITION_ID] * 4,
            open_times=open_times,
        ),
        _market_prices_frame(
            symbols=["BTCUSDT"] * 4,
            open_times=open_times,
            prices=prices,
            highs=prices,
        ),
        manager=_MANAGER,
    )
    assert result.height == 4
    sorted_result = result.sort("open_time")
    reasons = sorted_result["reason"].to_list()
    add_numbers = sorted_result["add_number"].to_list()
    additional_sizes = sorted_result["additional_size"].to_list()
    recommended_sizes = sorted_result["recommended_size"].to_list()
    position_sizes = sorted_result["position_size"].to_list()

    assert reasons == [
        PyramidingReason.READY_TO_ADD.value,
        PyramidingReason.READY_TO_ADD.value,
        PyramidingReason.READY_TO_ADD.value,
        PyramidingReason.MAX_ADDS_REACHED.value,
    ]
    assert add_numbers[0] == 1
    assert add_numbers[1] == 2
    assert add_numbers[2] == 3
    assert add_numbers[3] == 3

    assert position_sizes[0] == pytest.approx(1.0)
    assert additional_sizes[0] == pytest.approx(0.5)
    assert recommended_sizes[0] == pytest.approx(1.5)

    assert position_sizes[1] == pytest.approx(1.5)
    assert additional_sizes[1] == pytest.approx(0.75)
    assert recommended_sizes[1] == pytest.approx(2.25)

    assert position_sizes[2] == pytest.approx(2.25)
    assert additional_sizes[2] == pytest.approx(1.125)
    assert recommended_sizes[2] == pytest.approx(3.375)

    assert position_sizes[3] == pytest.approx(3.375)
    assert additional_sizes[3] == pytest.approx(0.0)


def test_max_adds_reached_after_fills() -> None:
    """MAX_ADDS_REACHED is emitted once add_number equals max_adds."""
    open_times = [_open_time(i) for i in range(4)]
    prices = [106.0, 112.0, 118.0, 125.0]
    result = SimplePyramidingEngine(max_adds=3, add_fraction=0.5, min_profit_percent=0.05).evaluate(
        _positions_frame(),
        _accounting_frame(
            symbols=["BTCUSDT"] * 4,
            position_ids=[_POSITION_ID] * 4,
            open_times=open_times,
            quantities=[1.0] * 4,
            entry_prices=[100.0] * 4,
        ),
        _portfolio_risk_frame(
            symbols=["BTCUSDT"] * 4,
            position_ids=[_POSITION_ID] * 4,
            open_times=open_times,
        ),
        _trade_management_frame(
            symbols=["BTCUSDT"] * 4,
            position_ids=[_POSITION_ID] * 4,
            open_times=open_times,
        ),
        _market_prices_frame(
            symbols=["BTCUSDT"] * 4,
            open_times=open_times,
            prices=prices,
            highs=prices,
        ),
        manager=_MANAGER,
    )
    last = result.sort("open_time").tail(1)
    assert last["reason"].to_list() == [PyramidingReason.MAX_ADDS_REACHED.value]
    assert last["allow_pyramid"].to_list() == [False]


# ---------------------------------------------------------------------------
# Portfolio risk gating
# ---------------------------------------------------------------------------


def test_portfolio_shutdown_blocks_pyramiding() -> None:
    """Portfolio risk SHUTDOWN state emits PORTFOLIO_SHUTDOWN."""
    result = _evaluate(
        SimplePyramidingEngine(),
        portfolio_risk=_portfolio_risk_frame(
            risk_states=["SHUTDOWN"],
            shutdown_reasons=["STOPPED"],
            cooldown_untils=[None],
        ),
        market_prices=_market_prices_frame(prices=[120.0], highs=[120.0]),
    )
    assert result["reason"].to_list() == [PyramidingReason.PORTFOLIO_SHUTDOWN.value]
    assert result["allow_pyramid"].to_list() == [False]


def test_cooldown_active_blocks_pyramiding() -> None:
    """shutdown_reason=COOLDOWN emits COOLDOWN_ACTIVE regardless of risk state."""
    result = _evaluate(
        SimplePyramidingEngine(),
        portfolio_risk=_portfolio_risk_frame(
            risk_states=["NORMAL"],
            shutdown_reasons=["COOLDOWN"],
            cooldown_untils=[None],
        ),
        market_prices=_market_prices_frame(prices=[120.0], highs=[120.0]),
    )
    assert result["reason"].to_list() == [PyramidingReason.COOLDOWN_ACTIVE.value]
    assert result["allow_pyramid"].to_list() == [False]


def test_portfolio_warning_blocks_pyramiding() -> None:
    """Portfolio risk WARNING state emits PORTFOLIO_WARNING."""
    result = _evaluate(
        SimplePyramidingEngine(),
        portfolio_risk=_portfolio_risk_frame(
            risk_states=["WARNING"],
            shutdown_reasons=[None],
            cooldown_untils=[None],
        ),
        market_prices=_market_prices_frame(prices=[120.0], highs=[120.0]),
    )
    assert result["reason"].to_list() == [PyramidingReason.PORTFOLIO_WARNING.value]
    assert result["allow_pyramid"].to_list() == [False]


# ---------------------------------------------------------------------------
# Trade management gating
# ---------------------------------------------------------------------------


def test_breakeven_active_blocks_pyramiding() -> None:
    """action_reason=BREAKEVEN emits BREAKEVEN_ACTIVE."""
    result = _evaluate(
        SimplePyramidingEngine(),
        trade_management=_trade_management_frame(
            management_actions=["HOLD"],
            action_reasons=["BREAKEVEN"],
        ),
        market_prices=_market_prices_frame(prices=[120.0], highs=[120.0]),
    )
    assert result["reason"].to_list() == [PyramidingReason.BREAKEVEN_ACTIVE.value]
    assert result["allow_pyramid"].to_list() == [False]


def test_trailing_stop_active_blocks_pyramiding() -> None:
    """action_reason=TRAILING_STOP emits TRAILING_STOP_ACTIVE."""
    result = _evaluate(
        SimplePyramidingEngine(),
        trade_management=_trade_management_frame(
            management_actions=["HOLD"],
            action_reasons=["TRAILING_STOP"],
        ),
        market_prices=_market_prices_frame(prices=[120.0], highs=[120.0]),
    )
    assert result["reason"].to_list() == [PyramidingReason.TRAILING_STOP_ACTIVE.value]
    assert result["allow_pyramid"].to_list() == [False]


def test_hold_and_none_management_actions_allow_evaluation() -> None:
    """management_action HOLD and NONE both allow pyramiding evaluation to proceed."""
    engine = SimplePyramidingEngine()
    for action in ("HOLD", "NONE"):
        result = _evaluate(
            engine,
            trade_management=_trade_management_frame(
                management_actions=[action],
                action_reasons=[None],
            ),
            market_prices=_market_prices_frame(prices=[102.0], highs=[102.0]),
        )
        assert result["reason"].to_list() == [PyramidingReason.INSUFFICIENT_PROFIT.value]


# ---------------------------------------------------------------------------
# Output schema and invariants
# ---------------------------------------------------------------------------


def test_output_canonical_ordering_and_dtype_schema() -> None:
    """Engine output enforces canonical column order and MERGED_PYRAMIDING_SCHEMA dtypes."""
    result = _evaluate(
        SimplePyramidingEngine(),
        market_prices=_market_prices_frame(prices=[105.0], highs=[105.0]),
    )
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result.schema == MERGED_PYRAMIDING_SCHEMA
    assert result.schema["open_time"] == pl.Datetime("us", "UTC")
    assert result.schema["allow_pyramid"] == pl.Boolean
    assert result.schema["add_number"] == pl.Int64


def test_trade_id_mirrors_position_id() -> None:
    """trade_id always equals position_id in v1 lineage."""
    result = _evaluate(
        SimplePyramidingEngine(),
        market_prices=_market_prices_frame(prices=[105.0], highs=[105.0]),
    )
    assert result["trade_id"].to_list() == result["position_id"].to_list()


def test_manager_is_stamped_on_every_row() -> None:
    """manager column contains the injected manager identity on every row."""
    result = _evaluate(
        SimplePyramidingEngine(),
        manager="my-manager",
        market_prices=_market_prices_frame(prices=[102.0], highs=[102.0]),
    )
    assert result["manager"].to_list() == ["my-manager"]


def test_highest_price_tracking_across_rows() -> None:
    """Running highest_price is the cumulative max of bar_high and current_price."""
    open_times = [_open_time(i) for i in range(3)]
    prices = [110.0, 95.0, 108.0]
    highs = [112.0, 90.0, 108.0]
    result = SimplePyramidingEngine().evaluate(
        _positions_frame(),
        _accounting_frame(
            symbols=["BTCUSDT"] * 3,
            position_ids=[_POSITION_ID] * 3,
            open_times=open_times,
            entry_prices=[100.0] * 3,
        ),
        _portfolio_risk_frame(
            symbols=["BTCUSDT"] * 3,
            position_ids=[_POSITION_ID] * 3,
            open_times=open_times,
        ),
        _trade_management_frame(
            symbols=["BTCUSDT"] * 3,
            position_ids=[_POSITION_ID] * 3,
            open_times=open_times,
        ),
        _market_prices_frame(
            symbols=["BTCUSDT"] * 3,
            open_times=open_times,
            prices=prices,
            highs=highs,
        ),
        manager=_MANAGER,
    )
    sorted_result = result.sort("open_time")
    highest_prices = sorted_result["highest_price"].to_list()
    assert highest_prices[0] == pytest.approx(112.0)
    assert highest_prices[1] == pytest.approx(112.0)
    assert highest_prices[2] == pytest.approx(112.0)


def test_inputs_are_immutable() -> None:
    """evaluate must not mutate any caller-supplied input frame."""
    positions = _positions_frame()
    accounting = _accounting_frame()
    portfolio_risk = _portfolio_risk_frame()
    trade_management = _trade_management_frame()
    market_prices = _market_prices_frame(prices=[105.0], highs=[105.0])

    positions_before = positions.clone()
    accounting_before = accounting.clone()
    risk_before = portfolio_risk.clone()
    tm_before = trade_management.clone()
    prices_before = market_prices.clone()

    SimplePyramidingEngine().evaluate(
        positions,
        accounting,
        portfolio_risk,
        trade_management,
        market_prices,
        manager=_MANAGER,
    )

    assert_frame_equal(positions, positions_before)
    assert_frame_equal(accounting, accounting_before)
    assert_frame_equal(portfolio_risk, risk_before)
    assert_frame_equal(trade_management, tm_before)
    assert_frame_equal(market_prices, prices_before)


def test_multiple_symbols_evaluated_independently() -> None:
    """Multiple position_ids produce one decision row per open snapshot."""
    open_time = _open_time(0)
    result = SimplePyramidingEngine().evaluate(
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
        ),
        _market_prices_frame(
            symbols=["BTCUSDT", "ETHUSDT"],
            open_times=[open_time, open_time],
            prices=[102.0, 106.0],
            highs=[102.0, 106.0],
        ),
        manager=_MANAGER,
    )
    assert result.height == 2
    by_pos = result.sort("position_id")
    assert by_pos["reason"].to_list() == [
        PyramidingReason.INSUFFICIENT_PROFIT.value,
        PyramidingReason.READY_TO_ADD.value,
    ]
