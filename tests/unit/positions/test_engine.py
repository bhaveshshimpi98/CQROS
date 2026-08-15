"""Unit tests for CQROS ``AverageCostPositionEngine``."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.positions import (
    CANONICAL_COLUMN_ORDER,
    MERGED_POSITION_SCHEMA,
    AverageCostPositionEngine,
    PositionStatus,
    PositionValidationError,
    validate_trade_frame,
)

_TIMEFRAME = "1h"
_MODEL_NAME = "alpha-lgbm"
_MODEL_VERSION = "1.0.0"
_POLICY = "fixed_risk"
_OPTIMIZER = "equal_weight"
_MANAGER = "simple"


def _execution_time(index: int) -> datetime:
    """Build a deterministic UTC execution_time for row ``index``."""
    return datetime(2024, 1, 1, hour=index, tzinfo=UTC)


def _trade_frame(
    *,
    sides: list[str],
    quantities: list[float],
    prices: list[float],
    fees: list[float] | None = None,
    symbols: list[str] | None = None,
    execution_times: list[datetime] | None = None,
) -> pl.DataFrame:
    """Build an executed-trade-shaped frame for engine tests."""
    row_count = len(sides)
    return pl.DataFrame(
        {
            "symbol": symbols if symbols is not None else ["BTCUSDT"] * row_count,
            "timeframe": [_TIMEFRAME] * row_count,
            "open_time": (
                execution_times
                if execution_times is not None
                else [_execution_time(index) for index in range(row_count)]
            ),
            "model_name": [_MODEL_NAME] * row_count,
            "model_version": [_MODEL_VERSION] * row_count,
            "optimizer": [_OPTIMIZER] * row_count,
            "policy": [_POLICY] * row_count,
            "manager": [_MANAGER] * row_count,
            "signal": sides,
            "side": sides,
            "order_type": ["MARKET"] * row_count,
            "requested_quantity": quantities,
            "executed_quantity": quantities,
            "requested_price": prices,
            "executed_price": prices,
            "fees": fees if fees is not None else [0.0] * row_count,
            "slippage": [0.0] * row_count,
            "status": ["FILLED"] * row_count,
            "execution_time": (
                execution_times
                if execution_times is not None
                else [_execution_time(index) for index in range(row_count)]
            ),
        }
    )


def test_validate_trade_frame_rejects_invalid_inputs() -> None:
    """validate_trade_frame rejects non-DataFrame and empty frames."""
    with pytest.raises(PositionValidationError) as exc_info:
        validate_trade_frame("not-a-frame")
    assert exc_info.value.error_code == "POS_FRAME_TYPE"
    with pytest.raises(PositionValidationError) as exc_info:
        validate_trade_frame(pl.DataFrame({"symbol": []}))
    assert exc_info.value.error_code == "POS_FRAME_EMPTY"


def test_long_position_open() -> None:
    """A single BUY opens a long position with average-cost entry."""
    trades = _trade_frame(sides=["BUY"], quantities=[2.0], prices=[100.0], fees=[1.5])
    original = trades.clone()
    positions = AverageCostPositionEngine().build(trades, manager=_MANAGER)
    assert_frame_equal(trades, original)
    assert tuple(positions.columns) == CANONICAL_COLUMN_ORDER
    assert positions.schema == MERGED_POSITION_SCHEMA
    assert positions.height == 1
    assert positions["status"].to_list() == [PositionStatus.OPEN.value]
    assert positions["side"].to_list() == ["LONG"]
    assert positions["quantity"].to_list() == [2.0]
    assert positions["average_entry_price"].to_list() == [100.0]
    assert positions["market_price"].to_list() == [100.0]
    assert positions["realized_pnl"].to_list() == [0.0]
    assert positions["unrealized_pnl"].to_list() == [0.0]
    assert positions["fees_paid"].to_list() == [1.5]
    assert positions["closed_at"].to_list() == [None]
    assert positions["manager"].to_list() == [_MANAGER]
    assert positions["position_id"].to_list() == ["pos-00000001"]


def test_multiple_buys_weighted_average_entry() -> None:
    """Multiple BUYs increase quantity and recompute weighted average entry."""
    trades = _trade_frame(
        sides=["BUY", "BUY"],
        quantities=[1.0, 3.0],
        prices=[100.0, 120.0],
        fees=[0.5, 1.5],
    )
    positions = AverageCostPositionEngine().build(trades, manager=_MANAGER)
    assert positions.height == 1
    assert positions["quantity"].to_list() == [4.0]
    assert positions["average_entry_price"].to_list() == [115.0]
    assert positions["market_price"].to_list() == [120.0]
    assert positions["fees_paid"].to_list() == [2.0]
    assert positions["unrealized_pnl"].to_list() == [20.0]
    assert positions["status"].to_list() == [PositionStatus.OPEN.value]


def test_partial_sell_realizes_pnl() -> None:
    """A partial SELL reduces quantity and realizes average-cost PnL."""
    trades = _trade_frame(
        sides=["BUY", "SELL"],
        quantities=[4.0, 1.0],
        prices=[100.0, 130.0],
        fees=[0.0, 0.25],
    )
    positions = AverageCostPositionEngine().build(trades, manager=_MANAGER)
    assert positions.height == 1
    assert positions["quantity"].to_list() == [3.0]
    assert positions["average_entry_price"].to_list() == [100.0]
    assert positions["realized_pnl"].to_list() == [30.0]
    assert positions["unrealized_pnl"].to_list() == [90.0]
    assert positions["fees_paid"].to_list() == [0.25]
    assert positions["status"].to_list() == [PositionStatus.OPEN.value]
    assert positions["closed_at"].to_list() == [None]


def test_full_close_and_reopen() -> None:
    """A full SELL closes the position; a later BUY opens a new position id."""
    trades = _trade_frame(
        sides=["BUY", "SELL", "BUY"],
        quantities=[2.0, 2.0, 1.0],
        prices=[100.0, 110.0, 90.0],
    )
    positions = AverageCostPositionEngine().build(trades, manager=_MANAGER)
    assert positions.height == 2
    closed = positions.filter(pl.col("status") == PositionStatus.CLOSED.value)
    opened = positions.filter(pl.col("status") == PositionStatus.OPEN.value)
    assert closed.height == 1
    assert opened.height == 1
    assert closed["quantity"].to_list() == [0.0]
    assert closed["realized_pnl"].to_list() == [20.0]
    assert closed["unrealized_pnl"].to_list() == [0.0]
    assert closed["closed_at"].to_list() == [_execution_time(1)]
    assert closed["position_id"].to_list() == ["pos-00000001"]
    assert opened["position_id"].to_list() == ["pos-00000002"]
    assert opened["quantity"].to_list() == [1.0]
    assert opened["average_entry_price"].to_list() == [90.0]


def test_engine_rejects_shorts_and_oversells() -> None:
    """SELL without inventory and oversells are rejected."""
    short = _trade_frame(sides=["SELL"], quantities=[1.0], prices=[100.0])
    with pytest.raises(PositionValidationError) as exc_info:
        AverageCostPositionEngine().build(short, manager=_MANAGER)
    assert exc_info.value.error_code == "POS_NO_SHORTS"

    oversell = _trade_frame(
        sides=["BUY", "SELL"],
        quantities=[1.0, 2.0],
        prices=[100.0, 110.0],
    )
    with pytest.raises(PositionValidationError) as exc_info:
        AverageCostPositionEngine().build(oversell, manager=_MANAGER)
    assert exc_info.value.error_code == "POS_OVERSELL"


def test_engine_rejects_blank_manager_and_invalid_side() -> None:
    """Blank managers and unsupported sides raise validation errors."""
    trades = _trade_frame(sides=["BUY"], quantities=[1.0], prices=[100.0])
    with pytest.raises(PositionValidationError) as exc_info:
        AverageCostPositionEngine().build(trades, manager="  ")
    assert exc_info.value.error_code == "POS_MANAGER_BLANK"

    invalid = _trade_frame(sides=["HOLD"], quantities=[1.0], prices=[100.0])
    with pytest.raises(PositionValidationError) as exc_info:
        AverageCostPositionEngine().build(invalid, manager=_MANAGER)
    assert exc_info.value.error_code == "POS_INVALID_SIDE"
