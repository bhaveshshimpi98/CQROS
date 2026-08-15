"""Unit tests for CQROS ``SimplePortfolioAccountingEngine``."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.accounting import (
    CANONICAL_COLUMN_ORDER,
    MERGED_ACCOUNTING_SCHEMA,
    POSITION_INPUT_COLUMNS,
    AccountingValidationError,
    PositionStatus,
    SimplePortfolioAccountingEngine,
    validate_position_frame,
)

_TIMEFRAME = "1h"
_MODEL_NAME = "alpha-lgbm"
_MODEL_VERSION = "1.0.0"
_POLICY = "fixed_risk"
_OPTIMIZER = "equal_weight"
_MANAGER = "simple"


def _opened_at(index: int) -> datetime:
    """Build a deterministic UTC opened_at timestamp for row ``index``."""
    return datetime(2024, 1, 1, hour=index, tzinfo=UTC)


def _position_frame(
    *,
    symbols: list[str] | None = None,
    position_ids: list[str] | None = None,
    sides: list[str] | None = None,
    statuses: list[str] | None = None,
    quantities: list[float] | None = None,
    average_entry_prices: list[float] | None = None,
    market_prices: list[float] | None = None,
    realized: list[float] | None = None,
    unrealized: list[float] | None = None,
    opened_ats: list[datetime] | None = None,
) -> pl.DataFrame:
    """Build a canonical-position-shaped frame for engine tests."""
    symbols = symbols if symbols is not None else ["BTCUSDT"]
    row_count = len(symbols)
    position_ids = (
        position_ids
        if position_ids is not None
        else [f"pos-{index + 1:08d}" for index in range(row_count)]
    )
    sides = sides if sides is not None else ["LONG"] * row_count
    statuses = statuses if statuses is not None else [PositionStatus.OPEN.value] * row_count
    quantities = quantities if quantities is not None else [1.0] * row_count
    average_entry_prices = (
        average_entry_prices if average_entry_prices is not None else [100.0] * row_count
    )
    market_prices = market_prices if market_prices is not None else [100.0] * row_count
    realized = realized if realized is not None else [0.0] * row_count
    unrealized = unrealized if unrealized is not None else [0.0] * row_count
    opened_ats = (
        opened_ats if opened_ats is not None else [_opened_at(index) for index in range(row_count)]
    )
    return pl.DataFrame(
        {
            "symbol": symbols,
            "timeframe": [_TIMEFRAME] * row_count,
            "position_id": position_ids,
            "side": sides,
            "status": statuses,
            "quantity": quantities,
            "average_entry_price": average_entry_prices,
            "market_price": market_prices,
            "realized_pnl": realized,
            "unrealized_pnl": unrealized,
            "opened_at": opened_ats,
            "model_name": [_MODEL_NAME] * row_count,
            "model_version": [_MODEL_VERSION] * row_count,
            "optimizer": [_OPTIMIZER] * row_count,
            "policy": [_POLICY] * row_count,
            "manager": [_MANAGER] * row_count,
        }
    )


def test_position_input_columns_contract() -> None:
    """POSITION_INPUT_COLUMNS enumerate the position columns the engine consumes."""
    for column in (
        "symbol",
        "timeframe",
        "position_id",
        "side",
        "status",
        "quantity",
        "average_entry_price",
        "market_price",
        "realized_pnl",
        "opened_at",
        "manager",
    ):
        assert column in POSITION_INPUT_COLUMNS


def test_validate_position_frame_rejects_invalid_inputs() -> None:
    """validate_position_frame rejects non-DataFrame and empty frames."""
    with pytest.raises(AccountingValidationError) as exc_info:
        validate_position_frame("not-a-frame")
    assert exc_info.value.error_code == "ACC_FRAME_TYPE"
    with pytest.raises(AccountingValidationError) as exc_info:
        validate_position_frame(pl.DataFrame({"symbol": []}))
    assert exc_info.value.error_code == "ACC_FRAME_EMPTY"


def test_build_rejects_empty_and_non_dataframe() -> None:
    """The engine rejects empty datasets and non-DataFrame inputs."""
    engine = SimplePortfolioAccountingEngine()
    with pytest.raises(AccountingValidationError) as exc_info:
        engine.build(pl.DataFrame({"symbol": []}), manager=_MANAGER)
    assert exc_info.value.error_code == "ACC_FRAME_EMPTY"
    with pytest.raises(AccountingValidationError) as exc_info:
        engine.build("nope", manager=_MANAGER)  # type: ignore[arg-type]
    assert exc_info.value.error_code == "ACC_FRAME_TYPE"


def test_single_row_mark_to_market() -> None:
    """A single long row computes mark-to-market accounting deterministically."""
    positions = _position_frame(
        quantities=[2.0],
        average_entry_prices=[100.0],
        market_prices=[110.0],
        realized=[5.0],
    )
    original = positions.clone()
    accounting = SimplePortfolioAccountingEngine().build(positions, manager=_MANAGER)
    assert_frame_equal(positions, original)
    assert tuple(accounting.columns) == CANONICAL_COLUMN_ORDER
    assert accounting.schema == MERGED_ACCOUNTING_SCHEMA
    assert accounting.height == 1
    assert accounting["mark_price"].to_list() == [110.0]
    assert accounting["market_value"].to_list() == [220.0]
    assert accounting["position_value"].to_list() == [220.0]
    assert accounting["unrealized_pnl"].to_list() == [20.0]
    assert accounting["total_pnl"].to_list() == [25.0]
    assert accounting["equity"].to_list() == [220.0]
    assert accounting["gross_exposure"].to_list() == [220.0]
    assert accounting["net_exposure"].to_list() == [220.0]
    assert accounting["return_pct"].to_list() == [25.0 / 220.0]
    assert accounting["open_time"].to_list() == [_opened_at(0)]
    assert accounting["position_status"].to_list() == [PositionStatus.OPEN.value]


def test_multiple_rows_exposures_broadcast() -> None:
    """Exposures aggregate across rows and broadcast onto every row."""
    positions = _position_frame(
        symbols=["BTCUSDT", "ETHUSDT"],
        quantities=[2.0, 3.0],
        average_entry_prices=[100.0, 50.0],
        market_prices=[110.0, 40.0],
        realized=[0.0, 0.0],
        opened_ats=[_opened_at(0), _opened_at(1)],
    )
    accounting = SimplePortfolioAccountingEngine().build(positions, manager=_MANAGER)
    assert accounting.height == 2
    assert accounting["market_value"].to_list() == [220.0, 120.0]
    assert accounting["unrealized_pnl"].to_list() == [20.0, -30.0]
    # gross = |220| + |120| = 340; net (long-only) = 220 + 120 = 340
    assert accounting["gross_exposure"].to_list() == [340.0, 340.0]
    assert accounting["net_exposure"].to_list() == [340.0, 340.0]
    assert accounting["equity"].to_list() == [220.0, 120.0]


def test_multiple_symbols_preserve_symbol_identity() -> None:
    """Multiple symbols are preserved on distinct accounting rows."""
    positions = _position_frame(
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        quantities=[1.0, 2.0, 4.0],
        average_entry_prices=[100.0, 50.0, 10.0],
        market_prices=[100.0, 50.0, 10.0],
        opened_ats=[_opened_at(0), _opened_at(1), _opened_at(2)],
    )
    accounting = SimplePortfolioAccountingEngine().build(positions, manager=_MANAGER)
    assert accounting["symbol"].to_list() == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert accounting["market_value"].to_list() == [100.0, 100.0, 40.0]
    assert accounting["gross_exposure"].to_list() == [240.0, 240.0, 240.0]


def test_closed_position_zero_quantity() -> None:
    """A closed position with zero quantity yields zero value and safe return."""
    positions = _position_frame(
        statuses=[PositionStatus.CLOSED.value],
        quantities=[0.0],
        average_entry_prices=[100.0],
        market_prices=[110.0],
        realized=[15.0],
    )
    accounting = SimplePortfolioAccountingEngine().build(positions, manager=_MANAGER)
    assert accounting["position_status"].to_list() == [PositionStatus.CLOSED.value]
    assert accounting["market_value"].to_list() == [0.0]
    assert accounting["position_value"].to_list() == [0.0]
    assert accounting["unrealized_pnl"].to_list() == [0.0]
    assert accounting["total_pnl"].to_list() == [15.0]
    # equity == cash + market_value == 0.0 -> return_pct falls back to 0.0
    assert accounting["equity"].to_list() == [0.0]
    assert accounting["return_pct"].to_list() == [0.0]


def test_default_cash_is_zero() -> None:
    """The default cash balance is zero and drives equity from market value."""
    positions = _position_frame(quantities=[1.0], market_prices=[100.0])
    accounting = SimplePortfolioAccountingEngine().build(positions, manager=_MANAGER)
    assert accounting["cash"].to_list() == [0.0]
    assert accounting["equity"].to_list() == [100.0]


def test_custom_cash_balance() -> None:
    """A custom cash balance broadcasts onto every row and lifts equity."""
    positions = _position_frame(quantities=[1.0], market_prices=[100.0])
    accounting = SimplePortfolioAccountingEngine(cash=1000.0).build(positions, manager=_MANAGER)
    assert accounting["cash"].to_list() == [1000.0]
    assert accounting["equity"].to_list() == [1100.0]


def test_engine_is_deterministic() -> None:
    """Repeated builds on identical inputs produce identical accounting frames."""
    positions = _position_frame(
        symbols=["BTCUSDT", "ETHUSDT"],
        quantities=[2.0, 3.0],
        market_prices=[110.0, 40.0],
        opened_ats=[_opened_at(0), _opened_at(1)],
    )
    engine = SimplePortfolioAccountingEngine(cash=500.0)
    first = engine.build(positions, manager=_MANAGER)
    second = engine.build(positions, manager=_MANAGER)
    assert_frame_equal(first, second)


def test_lineage_is_preserved_and_manager_is_stamped() -> None:
    """Lineage metadata is preserved while manager is stamped from the argument."""
    positions = _position_frame()
    accounting = SimplePortfolioAccountingEngine().build(positions, manager="ledger")
    assert accounting["manager"].to_list() == ["ledger"]
    assert accounting["model_name"].to_list() == [_MODEL_NAME]
    assert accounting["model_version"].to_list() == [_MODEL_VERSION]
    assert accounting["optimizer"].to_list() == [_OPTIMIZER]
    assert accounting["policy"].to_list() == [_POLICY]


def test_engine_rejects_non_long_sides() -> None:
    """Non-LONG position sides are rejected."""
    positions = _position_frame(sides=["SHORT"])
    with pytest.raises(AccountingValidationError) as exc_info:
        SimplePortfolioAccountingEngine().build(positions, manager=_MANAGER)
    assert exc_info.value.error_code == "ACC_INVALID_SIDE"


def test_engine_rejects_blank_manager() -> None:
    """Blank managers raise validation errors."""
    positions = _position_frame()
    with pytest.raises(AccountingValidationError) as exc_info:
        SimplePortfolioAccountingEngine().build(positions, manager="   ")
    assert exc_info.value.error_code == "ACC_MANAGER_BLANK"


def test_engine_rejects_missing_columns() -> None:
    """Missing required position columns raise validation errors."""
    positions = _position_frame().drop("market_price")
    with pytest.raises(AccountingValidationError) as exc_info:
        SimplePortfolioAccountingEngine().build(positions, manager=_MANAGER)
    assert exc_info.value.error_code == "ACC_MISSING_COLUMNS"


@pytest.mark.parametrize("cash", [float("nan"), float("inf"), float("-inf")])
def test_engine_rejects_non_finite_cash(cash: float) -> None:
    """Non-finite cash balances raise validation errors."""
    with pytest.raises(AccountingValidationError) as exc_info:
        SimplePortfolioAccountingEngine(cash=cash)
    assert exc_info.value.error_code == "ACC_CASH_NON_FINITE"


def test_output_schema_dtype_enforcement() -> None:
    """Engine output enforces canonical merged-accounting dtypes."""
    positions = _position_frame()
    accounting = SimplePortfolioAccountingEngine().build(positions, manager=_MANAGER)
    assert accounting.schema == MERGED_ACCOUNTING_SCHEMA
    assert accounting.schema["open_time"] == pl.Datetime("us", "UTC")
    assert accounting.schema["equity"] == pl.Float64
