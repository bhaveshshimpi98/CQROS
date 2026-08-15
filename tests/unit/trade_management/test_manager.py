"""Unit tests for CQROS ``SimpleTradeManagementManager``."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.trade_management import (
    ACCOUNTING_INPUT_COLUMNS,
    CANONICAL_COLUMN_ORDER,
    DEFAULT_INITIAL_RISK_PERCENT,
    DEFAULT_TRAIL_PERCENT,
    MARKET_PRICE_INPUT_COLUMNS,
    MERGED_TRADE_MANAGEMENT_SCHEMA,
    PORTFOLIO_RISK_INPUT_COLUMNS,
    POSITION_INPUT_COLUMNS,
    ManagementAction,
    ShutdownReason,
    SimpleTradeManagementManager,
    TradeManagementValidationError,
    validate_accounting_frame,
    validate_market_price_frame,
    validate_portfolio_risk_frame,
    validate_position_frame,
)

_TIMEFRAME = "1h"
_MODEL_NAME = "alpha-lgbm"
_MODEL_VERSION = "1.0.0"
_POLICY = "fixed_risk"
_OPTIMIZER = "equal_weight"
_MANAGER = "simple"
_ENTRY_PRICE = 100.0


def _open_time(index: int) -> datetime:
    """Build a deterministic UTC open_time timestamp for row ``index``."""
    return datetime(2024, 1, 1, hour=index, tzinfo=UTC)


def _position_frame(
    *,
    symbols: list[str] | None = None,
    position_ids: list[str] | None = None,
) -> pl.DataFrame:
    """Build a minimal position identity frame."""
    symbols = symbols if symbols is not None else ["BTCUSDT"]
    row_count = len(symbols)
    position_ids = (
        position_ids
        if position_ids is not None
        else [f"pos-{index + 1:08d}" for index in range(row_count)]
    )
    return pl.DataFrame(
        {
            "symbol": symbols,
            "timeframe": [_TIMEFRAME] * row_count,
            "position_id": position_ids,
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
    unrealized_pnls: list[float] | None = None,
) -> pl.DataFrame:
    """Build an accounting-shaped frame for trade-management manager tests."""
    symbols = symbols if symbols is not None else ["BTCUSDT"]
    row_count = len(symbols)
    position_ids = (
        position_ids
        if position_ids is not None
        else [f"pos-{index + 1:08d}" for index in range(row_count)]
    )
    open_times = (
        open_times if open_times is not None else [_open_time(index) for index in range(row_count)]
    )
    position_statuses = position_statuses if position_statuses is not None else ["OPEN"] * row_count
    quantities = quantities if quantities is not None else [1.0] * row_count
    entry_prices = entry_prices if entry_prices is not None else [_ENTRY_PRICE] * row_count
    unrealized_pnls = unrealized_pnls if unrealized_pnls is not None else [0.0] * row_count
    return pl.DataFrame(
        {
            "symbol": symbols,
            "timeframe": [_TIMEFRAME] * row_count,
            "open_time": open_times,
            "position_id": position_ids,
            "position_status": position_statuses,
            "quantity": quantities,
            "average_entry_price": entry_prices,
            "unrealized_pnl": unrealized_pnls,
            "model_name": [_MODEL_NAME] * row_count,
            "model_version": [_MODEL_VERSION] * row_count,
            "optimizer": [_OPTIMIZER] * row_count,
            "policy": [_POLICY] * row_count,
        }
    )


def _portfolio_risk_frame(
    *,
    symbols: list[str] | None = None,
    position_ids: list[str] | None = None,
    open_times: list[datetime] | None = None,
    risk_states: list[str] | None = None,
    allow_new_entries: list[bool] | None = None,
) -> pl.DataFrame:
    """Build a portfolio-risk-shaped frame for trade-management manager tests."""
    symbols = symbols if symbols is not None else ["BTCUSDT"]
    row_count = len(symbols)
    position_ids = (
        position_ids
        if position_ids is not None
        else [f"pos-{index + 1:08d}" for index in range(row_count)]
    )
    open_times = (
        open_times if open_times is not None else [_open_time(index) for index in range(row_count)]
    )
    risk_states = risk_states if risk_states is not None else ["NORMAL"] * row_count
    allow_new_entries = allow_new_entries if allow_new_entries is not None else [True] * row_count
    return pl.DataFrame(
        {
            "symbol": symbols,
            "timeframe": [_TIMEFRAME] * row_count,
            "open_time": open_times,
            "position_id": position_ids,
            "portfolio_risk_state": risk_states,
            "allow_new_entries": allow_new_entries,
        }
    )


def _market_prices_frame(
    *,
    symbols: list[str] | None = None,
    open_times: list[datetime] | None = None,
    prices: list[float] | None = None,
) -> pl.DataFrame:
    """Build a market-price frame keyed by symbol, timeframe, and open_time."""
    symbols = symbols if symbols is not None else ["BTCUSDT"]
    row_count = len(symbols)
    open_times = (
        open_times if open_times is not None else [_open_time(index) for index in range(row_count)]
    )
    prices = prices if prices is not None else [_ENTRY_PRICE] * row_count
    return pl.DataFrame(
        {
            "symbol": symbols,
            "timeframe": [_TIMEFRAME] * row_count,
            "open_time": open_times,
            "price": prices,
        }
    )


def _evaluate(
    manager: SimpleTradeManagementManager,
    *,
    accounting: pl.DataFrame | None = None,
    positions: pl.DataFrame | None = None,
    portfolio_risk: pl.DataFrame | None = None,
    market_prices: pl.DataFrame | None = None,
    manager_name: str = _MANAGER,
) -> pl.DataFrame:
    """Evaluate trade-management decisions with default companion frames."""
    accounting_frame = accounting if accounting is not None else _accounting_frame()
    positions_frame = positions if positions is not None else _position_frame()
    risk_frame = portfolio_risk if portfolio_risk is not None else _portfolio_risk_frame()
    prices_frame = market_prices if market_prices is not None else _market_prices_frame()
    return manager.evaluate(
        positions_frame,
        accounting_frame,
        risk_frame,
        prices_frame,
        manager=manager_name,
    )


def test_input_columns_contract() -> None:
    """Input column contracts enumerate the columns the manager consumes."""
    for column in ACCOUNTING_INPUT_COLUMNS:
        assert column in (
            "symbol",
            "timeframe",
            "open_time",
            "position_id",
            "position_status",
            "quantity",
            "average_entry_price",
            "unrealized_pnl",
            "model_name",
            "model_version",
            "optimizer",
            "policy",
        )
    for column in ("symbol", "timeframe", "position_id"):
        assert column in POSITION_INPUT_COLUMNS
    for column in (
        "symbol",
        "timeframe",
        "open_time",
        "position_id",
        "portfolio_risk_state",
        "allow_new_entries",
    ):
        assert column in PORTFOLIO_RISK_INPUT_COLUMNS
    for column in ("symbol", "timeframe", "open_time", "price"):
        assert column in MARKET_PRICE_INPUT_COLUMNS


def test_validate_frames_reject_invalid_inputs() -> None:
    """Frame validators reject non-DataFrame and empty frames."""
    with pytest.raises(TradeManagementValidationError) as exc_info:
        validate_accounting_frame("not-a-frame")
    assert exc_info.value.error_code == "TME_FRAME_TYPE"
    with pytest.raises(TradeManagementValidationError) as exc_info:
        validate_accounting_frame(pl.DataFrame({"symbol": []}))
    assert exc_info.value.error_code == "TME_FRAME_EMPTY"

    with pytest.raises(TradeManagementValidationError) as exc_info:
        validate_position_frame("not-a-frame")
    assert exc_info.value.error_code == "TME_FRAME_TYPE"

    with pytest.raises(TradeManagementValidationError) as exc_info:
        validate_portfolio_risk_frame(pl.DataFrame({"symbol": []}))
    assert exc_info.value.error_code == "TME_FRAME_EMPTY"

    with pytest.raises(TradeManagementValidationError) as exc_info:
        validate_market_price_frame("not-a-frame")
    assert exc_info.value.error_code == "TME_FRAME_TYPE"


def test_evaluate_rejects_empty_and_non_dataframe() -> None:
    """The manager rejects empty datasets and non-DataFrame inputs."""
    manager = SimpleTradeManagementManager()
    positions = _position_frame()
    with pytest.raises(TradeManagementValidationError) as exc_info:
        manager.evaluate(
            positions,
            pl.DataFrame({"symbol": []}),
            _portfolio_risk_frame(),
            _market_prices_frame(),
            manager=_MANAGER,
        )
    assert exc_info.value.error_code == "TME_FRAME_EMPTY"
    with pytest.raises(TradeManagementValidationError) as exc_info:
        manager.evaluate(
            pl.DataFrame({"symbol": []}),
            _accounting_frame(),
            _portfolio_risk_frame(),
            _market_prices_frame(),
            manager=_MANAGER,
        )
    assert exc_info.value.error_code == "TME_FRAME_EMPTY"


def test_single_position_none_action() -> None:
    """A modest gain below 1R emits NONE with no stop update."""
    result = _evaluate(
        SimpleTradeManagementManager(),
        market_prices=_market_prices_frame(prices=[104.0]),
    )
    assert result.height == 1
    assert result["management_action"].to_list() == [ManagementAction.NONE.value]
    assert result["action_reason"].to_list() == [ShutdownReason.NONE.value]
    assert result["stop_price"].to_list() == [None]
    assert result["take_profit_price"].to_list() == [None]
    assert result["exit_quantity"].to_list() == [0.0]
    assert result["allow_pyramid"].to_list() == [False]


def test_trailing_stop_triggered() -> None:
    """A pullback below the trailing threshold emits UPDATE_STOP / TRAILING_STOP."""
    open_times = [_open_time(0), _open_time(1)]
    result = _evaluate(
        SimpleTradeManagementManager(),
        accounting=_accounting_frame(
            symbols=["BTCUSDT", "BTCUSDT"],
            position_ids=["pos-00000001", "pos-00000001"],
            open_times=open_times,
        ),
        positions=_position_frame(position_ids=["pos-00000001"]),
        portfolio_risk=_portfolio_risk_frame(
            symbols=["BTCUSDT", "BTCUSDT"],
            position_ids=["pos-00000001", "pos-00000001"],
            open_times=open_times,
        ),
        market_prices=_market_prices_frame(
            symbols=["BTCUSDT", "BTCUSDT"],
            open_times=open_times,
            prices=[120.0, 110.0],
        ),
    )
    trailing_row = result.filter(pl.col("open_time") == _open_time(1))
    assert trailing_row["management_action"].to_list() == [ManagementAction.UPDATE_STOP.value]
    assert trailing_row["action_reason"].to_list() == [ShutdownReason.TRAILING_STOP.value]
    assert trailing_row["stop_price"].to_list() == [114.0]
    assert trailing_row["trail_price"].to_list() == [114.0]


def test_trailing_stop_not_triggered() -> None:
    """Price above the trailing threshold does not emit a trailing stop."""
    result = _evaluate(
        SimpleTradeManagementManager(),
        market_prices=_market_prices_frame(prices=[102.0]),
    )
    assert result["management_action"].to_list() == [ManagementAction.NONE.value]
    assert result["action_reason"].to_list() == [ShutdownReason.NONE.value]


def test_breakeven_triggered() -> None:
    """Reward at or above 1R emits UPDATE_STOP / BREAKEVEN at entry."""
    result = _evaluate(
        SimpleTradeManagementManager(),
        market_prices=_market_prices_frame(prices=[106.0]),
    )
    assert result["management_action"].to_list() == [ManagementAction.UPDATE_STOP.value]
    assert result["action_reason"].to_list() == [ShutdownReason.BREAKEVEN.value]
    assert result["stop_price"].to_list() == [_ENTRY_PRICE]
    assert result["breakeven_price"].to_list() == [_ENTRY_PRICE]


def test_breakeven_not_triggered() -> None:
    """Reward below 1R does not emit a break-even stop."""
    result = _evaluate(
        SimpleTradeManagementManager(),
        market_prices=_market_prices_frame(prices=[104.0]),
    )
    assert result["management_action"].to_list() == [ManagementAction.NONE.value]
    assert result["action_reason"].to_list() == [ShutdownReason.NONE.value]
    assert result["breakeven_price"].to_list() == [None]


def test_trailing_priority_over_breakeven() -> None:
    """Trailing-stop hits take priority when both trailing and break-even apply."""
    open_times = [_open_time(0), _open_time(1)]
    result = _evaluate(
        SimpleTradeManagementManager(),
        accounting=_accounting_frame(
            symbols=["BTCUSDT", "BTCUSDT"],
            position_ids=["pos-00000001", "pos-00000001"],
            open_times=open_times,
        ),
        positions=_position_frame(position_ids=["pos-00000001"]),
        portfolio_risk=_portfolio_risk_frame(
            symbols=["BTCUSDT", "BTCUSDT"],
            position_ids=["pos-00000001", "pos-00000001"],
            open_times=open_times,
        ),
        market_prices=_market_prices_frame(
            symbols=["BTCUSDT", "BTCUSDT"],
            open_times=open_times,
            prices=[112.0, 106.0],
        ),
    )
    row = result.filter(pl.col("open_time") == _open_time(1))
    assert row["management_action"].to_list() == [ManagementAction.UPDATE_STOP.value]
    assert row["action_reason"].to_list() == [ShutdownReason.TRAILING_STOP.value]
    assert row["stop_price"].to_list() == [pytest.approx(106.4)]


def test_multiple_positions_evaluated() -> None:
    """Multiple position_ids produce one decision row per open snapshot."""
    result = _evaluate(
        SimpleTradeManagementManager(),
        accounting=_accounting_frame(
            symbols=["BTCUSDT", "ETHUSDT"],
            position_ids=["pos-00000001", "pos-00000002"],
        ),
        positions=_position_frame(
            symbols=["BTCUSDT", "ETHUSDT"],
            position_ids=["pos-00000001", "pos-00000002"],
        ),
        portfolio_risk=_portfolio_risk_frame(
            symbols=["BTCUSDT", "ETHUSDT"],
            position_ids=["pos-00000001", "pos-00000002"],
        ),
        market_prices=_market_prices_frame(
            symbols=["BTCUSDT", "ETHUSDT"],
            prices=[104.0, 106.0],
        ),
    )
    assert result.height == 2
    by_position = result.sort("position_id")
    assert by_position["position_id"].to_list() == ["pos-00000001", "pos-00000002"]
    assert by_position["management_action"].to_list() == [
        ManagementAction.NONE.value,
        ManagementAction.UPDATE_STOP.value,
    ]


def test_highest_and_lowest_price_tracking() -> None:
    """Running highest and lowest prices are tracked per position_id."""
    open_times = [_open_time(0), _open_time(1), _open_time(2)]
    result = _evaluate(
        SimpleTradeManagementManager(),
        accounting=_accounting_frame(
            symbols=["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            position_ids=["pos-00000001", "pos-00000001", "pos-00000001"],
            open_times=open_times,
        ),
        positions=_position_frame(position_ids=["pos-00000001"]),
        portfolio_risk=_portfolio_risk_frame(
            symbols=["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            position_ids=["pos-00000001", "pos-00000001", "pos-00000001"],
            open_times=open_times,
        ),
        market_prices=_market_prices_frame(
            symbols=["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            open_times=open_times,
            prices=[110.0, 95.0, 108.0],
        ),
    )
    sorted_result = result.sort("open_time")
    assert sorted_result["highest_price"].to_list() == [110.0, 110.0, 110.0]
    assert sorted_result["lowest_price"].to_list() == [110.0, 95.0, 95.0]


def test_allow_pyramid_always_false() -> None:
    """allow_pyramid remains False in v1 even when allow_new_entries is True."""
    result = _evaluate(
        SimpleTradeManagementManager(),
        portfolio_risk=_portfolio_risk_frame(allow_new_entries=[True]),
    )
    assert result["allow_pyramid"].to_list() == [False]


def test_allow_pyramid_false_when_allow_new_entries_false() -> None:
    """allow_pyramid stays False when portfolio risk blocks new entries."""
    result = _evaluate(
        SimpleTradeManagementManager(),
        portfolio_risk=_portfolio_risk_frame(allow_new_entries=[False]),
    )
    assert result["allow_pyramid"].to_list() == [False]


def test_inputs_are_immutable() -> None:
    """evaluate must not mutate caller-supplied input frames."""
    accounting = _accounting_frame()
    positions = _position_frame()
    portfolio_risk = _portfolio_risk_frame()
    market_prices = _market_prices_frame()
    accounting_before = accounting.clone()
    positions_before = positions.clone()
    risk_before = portfolio_risk.clone()
    prices_before = market_prices.clone()
    SimpleTradeManagementManager().evaluate(
        positions,
        accounting,
        portfolio_risk,
        market_prices,
        manager=_MANAGER,
    )
    assert_frame_equal(accounting, accounting_before)
    assert_frame_equal(positions, positions_before)
    assert_frame_equal(portfolio_risk, risk_before)
    assert_frame_equal(market_prices, prices_before)


def test_output_canonical_ordering_and_dtype_schema() -> None:
    """Manager output enforces canonical order and merged-schema dtypes."""
    result = _evaluate(SimpleTradeManagementManager())
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result.schema == MERGED_TRADE_MANAGEMENT_SCHEMA
    assert result.schema["open_time"] == pl.Datetime("us", "UTC")
    assert result.schema["allow_pyramid"] == pl.Boolean


def test_missing_market_price_coverage_fails() -> None:
    """Missing market prices for open snapshots raise validation errors."""
    with pytest.raises(TradeManagementValidationError) as exc_info:
        _evaluate(
            SimpleTradeManagementManager(),
            market_prices=_market_prices_frame(open_times=[_open_time(1)]),
        )
    assert exc_info.value.error_code == "TME_MARKET_COVERAGE"


def test_missing_portfolio_risk_coverage_fails() -> None:
    """Missing portfolio-risk rows for open snapshots raise validation errors."""
    with pytest.raises(TradeManagementValidationError) as exc_info:
        _evaluate(
            SimpleTradeManagementManager(),
            portfolio_risk=_portfolio_risk_frame(open_times=[_open_time(1)]),
        )
    assert exc_info.value.error_code == "TME_RISK_COVERAGE"


def test_no_open_positions_error() -> None:
    """Accounting frames without OPEN rows are rejected."""
    with pytest.raises(TradeManagementValidationError) as exc_info:
        _evaluate(
            SimpleTradeManagementManager(),
            accounting=_accounting_frame(position_statuses=["CLOSED"]),
        )
    assert exc_info.value.error_code == "TME_NO_OPEN_POSITIONS"


def test_closed_positions_do_not_produce_rows() -> None:
    """Only OPEN accounting rows produce trade-management decisions."""
    accounting = pl.concat(
        [
            _accounting_frame(position_statuses=["OPEN"]),
            _accounting_frame(
                symbols=["ETHUSDT"],
                position_ids=["pos-00000002"],
                position_statuses=["CLOSED"],
            ),
        ]
    )
    result = _evaluate(
        SimpleTradeManagementManager(),
        accounting=accounting,
        positions=_position_frame(
            symbols=["BTCUSDT", "ETHUSDT"],
            position_ids=["pos-00000001", "pos-00000002"],
        ),
        portfolio_risk=_portfolio_risk_frame(
            symbols=["BTCUSDT", "ETHUSDT"],
            position_ids=["pos-00000001", "pos-00000002"],
        ),
        market_prices=_market_prices_frame(
            symbols=["BTCUSDT", "ETHUSDT"],
            prices=[104.0, 106.0],
        ),
    )
    assert result.height == 1
    assert result["position_id"].to_list() == ["pos-00000001"]


def test_manager_rejects_blank_manager() -> None:
    """Blank managers raise validation errors."""
    with pytest.raises(TradeManagementValidationError) as exc_info:
        _evaluate(SimpleTradeManagementManager(), manager_name="   ")
    assert exc_info.value.error_code == "TME_MANAGER_BLANK"


def test_manager_rejects_missing_accounting_columns() -> None:
    """Missing required accounting columns raise validation errors."""
    with pytest.raises(TradeManagementValidationError) as exc_info:
        _evaluate(
            SimpleTradeManagementManager(),
            accounting=_accounting_frame().drop("unrealized_pnl"),
        )
    assert exc_info.value.error_code == "TME_MISSING_COLUMNS"


def test_missing_position_id_coverage_fails() -> None:
    """Accounting position_ids absent from positions raise validation errors."""
    with pytest.raises(TradeManagementValidationError) as exc_info:
        _evaluate(
            SimpleTradeManagementManager(),
            accounting=_accounting_frame(position_ids=["pos-missing"]),
            positions=_position_frame(position_ids=["pos-00000001"]),
        )
    assert exc_info.value.error_code == "TME_POSITION_IDS"


def test_manager_rejects_invalid_trail_percent() -> None:
    """Invalid trail_percent values raise validation errors at construction."""
    with pytest.raises(TradeManagementValidationError) as exc_info:
        SimpleTradeManagementManager(trail_percent=1.0)
    assert exc_info.value.error_code == "TME_LIMIT_RANGE"

    with pytest.raises(TradeManagementValidationError) as exc_info:
        SimpleTradeManagementManager(initial_risk_percent=-0.1)
    assert exc_info.value.error_code == "TME_LIMIT_RANGE"


def test_default_fractions_match_schema_constants() -> None:
    """Default manager fractions match schema default constants."""
    manager = SimpleTradeManagementManager()
    result = _evaluate(
        manager,
        market_prices=_market_prices_frame(prices=[106.0]),
    )
    assert DEFAULT_TRAIL_PERCENT == 0.05
    assert DEFAULT_INITIAL_RISK_PERCENT == 0.05
    assert result["trail_price"].to_list() == [106.0 * (1.0 - DEFAULT_TRAIL_PERCENT)]


def test_lineage_is_preserved_and_manager_is_stamped() -> None:
    """Lineage metadata is preserved while manager is stamped from the argument."""
    result = _evaluate(SimpleTradeManagementManager(), manager_name="ledger")
    assert result["manager"].to_list() == ["ledger"]
    assert result["model_name"].to_list() == [_MODEL_NAME]
    assert result["model_version"].to_list() == [_MODEL_VERSION]
    assert result["optimizer"].to_list() == [_OPTIMIZER]
    assert result["policy"].to_list() == [_POLICY]
    assert result["risk_state"].to_list() == ["NORMAL"]
