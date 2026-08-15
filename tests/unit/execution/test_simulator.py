"""Unit tests for CQROS ``SimpleExecutionSimulator``."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.execution import (
    CANONICAL_COLUMN_ORDER,
    MERGED_TRADE_SCHEMA,
    ExecutionStatus,
    ExecutionValidationError,
    SimpleExecutionSimulator,
    validate_order_frame,
)

_TIMEFRAME = "1h"
_MODEL_NAME = "alpha-lgbm"
_MODEL_VERSION = "1.0.0"
_POLICY = "fixed_risk"
_OPTIMIZER = "equal_weight"
_MANAGER = "simple"
_FIXED_NOW = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)


def _open_time(index: int) -> datetime:
    """Build a deterministic UTC open_time for row ``index``."""
    return datetime(2024, 1, 1, tzinfo=UTC).replace(microsecond=index)


def _order_frame(
    *,
    symbols: list[str],
    sides: list[str],
    quantities: list[float],
    order_types: list[str] | None = None,
    limit_prices: list[float | None] | None = None,
    signals: list[str] | None = None,
    open_times: list[datetime] | None = None,
) -> pl.DataFrame:
    """Build an OMS order-shaped frame for simulator tests."""
    row_count = len(symbols)
    ids = [f"order-{index}" for index in range(row_count)]
    data: dict[str, object] = {
        "symbol": symbols,
        "timeframe": [_TIMEFRAME] * row_count,
        "open_time": (
            open_times
            if open_times is not None
            else [_open_time(index) for index in range(row_count)]
        ),
        "order_id": ids,
        "parent_order_id": ids,
        "model_name": [_MODEL_NAME] * row_count,
        "model_version": [_MODEL_VERSION] * row_count,
        "policy": [_POLICY] * row_count,
        "optimizer": [_OPTIMIZER] * row_count,
        "side": sides,
        "order_type": order_types if order_types is not None else ["MARKET"] * row_count,
        "quantity": quantities,
        "limit_price": (limit_prices if limit_prices is not None else [None] * row_count),
        "stop_price": [None] * row_count,
        "filled_quantity": [0.0] * row_count,
        "average_fill_price": [None] * row_count,
        "status": ["PENDING"] * row_count,
        "created_at": [_FIXED_NOW] * row_count,
        "updated_at": [_FIXED_NOW] * row_count,
    }
    if signals is not None:
        data["signal"] = signals
    return pl.DataFrame(data)


def test_validate_order_frame_rejects_invalid_inputs() -> None:
    """validate_order_frame rejects non-DataFrame and empty frames."""
    with pytest.raises(ExecutionValidationError) as exc_info:
        validate_order_frame("not-a-frame")
    assert exc_info.value.error_code == "EXEC_FRAME_TYPE"
    with pytest.raises(ExecutionValidationError) as exc_info:
        validate_order_frame(pl.DataFrame({"symbol": []}))
    assert exc_info.value.error_code == "EXEC_FRAME_EMPTY"


def test_simple_simulator_fills_market_orders_deterministically() -> None:
    """Market orders fill immediately with zero fees and zero slippage."""
    orders = _order_frame(
        symbols=["BTCUSDT", "ETHUSDT"],
        sides=["BUY", "SELL"],
        quantities=[1.5, 2.0],
        limit_prices=[100.0, 200.0],
    )
    original = orders.clone()
    trades = SimpleExecutionSimulator().execute(orders, manager=_MANAGER)
    assert_frame_equal(orders, original)
    assert tuple(trades.columns) == CANONICAL_COLUMN_ORDER
    assert trades.schema == MERGED_TRADE_SCHEMA
    assert trades["status"].to_list() == [ExecutionStatus.FILLED.value] * 2
    assert trades["executed_quantity"].to_list() == [1.5, 2.0]
    assert trades["requested_quantity"].to_list() == [1.5, 2.0]
    assert trades["executed_price"].to_list() == [100.0, 200.0]
    assert trades["requested_price"].to_list() == [100.0, 200.0]
    assert trades["fees"].to_list() == [0.0, 0.0]
    assert trades["slippage"].to_list() == [0.0, 0.0]
    assert trades["manager"].to_list() == [_MANAGER, _MANAGER]
    assert trades["signal"].to_list() == ["BUY", "SELL"]
    assert trades["execution_time"].to_list() == trades["open_time"].to_list()


def test_simple_simulator_preserves_signal_and_fills_null_limit_price() -> None:
    """Explicit signal is preserved; null limit prices fill at 0.0."""
    orders = _order_frame(
        symbols=["BTCUSDT"],
        sides=["BUY"],
        quantities=[1.0],
        signals=["BUY"],
        limit_prices=[None],
    )
    trades = SimpleExecutionSimulator().execute(orders, manager=_MANAGER)
    assert trades["signal"].to_list() == ["BUY"]
    assert trades["requested_price"].to_list() == [0.0]
    assert trades["executed_price"].to_list() == [0.0]


def test_simple_simulator_skips_non_market_orders() -> None:
    """Only MARKET rows are executed; non-market-only frames fail."""
    mixed = _order_frame(
        symbols=["BTCUSDT", "ETHUSDT"],
        sides=["BUY", "SELL"],
        quantities=[1.0, 2.0],
        order_types=["MARKET", "LIMIT"],
        limit_prices=[10.0, 20.0],
    )
    trades = SimpleExecutionSimulator().execute(mixed, manager=_MANAGER)
    assert trades.height == 1
    assert trades["symbol"].to_list() == ["BTCUSDT"]

    limits_only = _order_frame(
        symbols=["BTCUSDT"],
        sides=["BUY"],
        quantities=[1.0],
        order_types=["LIMIT"],
        limit_prices=[10.0],
    )
    with pytest.raises(ExecutionValidationError, match="MARKET") as exc_info:
        SimpleExecutionSimulator().execute(limits_only, manager=_MANAGER)
    assert exc_info.value.error_code == "EXEC_NO_MARKET_ORDERS"


def test_simple_simulator_rejects_blank_manager_and_duplicates() -> None:
    """Blank manager and duplicate primary keys are rejected."""
    orders = _order_frame(
        symbols=["BTCUSDT"],
        sides=["BUY"],
        quantities=[1.0],
        limit_prices=[10.0],
    )
    with pytest.raises(ExecutionValidationError) as exc_info:
        SimpleExecutionSimulator().execute(orders, manager="  ")
    assert exc_info.value.error_code == "EXEC_MANAGER_BLANK"

    duplicates = _order_frame(
        symbols=["BTCUSDT", "BTCUSDT"],
        sides=["BUY", "SELL"],
        quantities=[1.0, 2.0],
        limit_prices=[10.0, 20.0],
        open_times=[_open_time(0), _open_time(0)],
    )
    with pytest.raises(ExecutionValidationError) as exc_info:
        SimpleExecutionSimulator().execute(duplicates, manager=_MANAGER)
    assert exc_info.value.error_code == "EXEC_DUPLICATE_KEYS"
