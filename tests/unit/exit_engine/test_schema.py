"""Unit tests for CQROS merged exit-engine recommendation schema."""

from __future__ import annotations

import polars as pl

from cqros.exit_engine import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    DEFAULT_INITIAL_RISK_PERCENT,
    DEFAULT_PARTIAL_EXIT_PERCENT,
    DEFAULT_TAKE_PROFIT_MULTIPLE,
    EXIT_ENGINE_COLUMNS,
    MERGED_EXIT_ENGINE_SCHEMA,
    PRIMARY_KEY_COLUMNS,
    PRIORITY_ALPHA_DECAY,
    PRIORITY_BREAK_EVEN,
    PRIORITY_COOLDOWN,
    PRIORITY_HOLD,
    PRIORITY_PORTFOLIO_SHUTDOWN,
    PRIORITY_REGIME_EXIT,
    PRIORITY_TAKE_PROFIT,
    PRIORITY_TIME_STOP,
    PRIORITY_TRAILING_STOP,
    REQUIRED_COLUMNS,
    ExitAction,
    ExitReason,
    exit_actions,
    exit_reasons,
    values,
)
from cqros.exit_engine.schema import (
    MERGED_EXIT_ENGINE_SCHEMA as MERGED_EXIT_ENGINE_SCHEMA_DIRECT,
)


def test_primary_key_and_required_columns() -> None:
    """Primary key and required columns match the canonical exit-engine contract."""
    assert PRIMARY_KEY_COLUMNS == ("symbol", "timeframe", "open_time", "position_id")
    assert REQUIRED_COLUMNS == CANONICAL_COLUMN_ORDER
    assert CANONICAL_COLUMN_ORDER == EXIT_ENGINE_COLUMNS


def test_exit_engine_columns_contain_required_domain_columns() -> None:
    """EXIT_ENGINE_COLUMNS enumerates identity, pricing, state, and decision fields."""
    for column in (
        "symbol",
        "timeframe",
        "open_time",
        "position_id",
        "manager",
        "entry_price",
        "current_price",
        "quantity",
        "risk_reward_ratio",
        "risk_state",
        "trade_state",
        "pyramid_state",
        "exit_action",
        "exit_reason",
        "recommended_quantity",
        "recommended_percent",
        "priority",
        "created_at",
    ):
        assert column in EXIT_ENGINE_COLUMNS


def test_column_dtypes_and_merged_schema() -> None:
    """Merged schema dtypes match COLUMN_DTYPES in canonical order."""
    assert MERGED_EXIT_ENGINE_SCHEMA is MERGED_EXIT_ENGINE_SCHEMA_DIRECT
    assert MERGED_EXIT_ENGINE_SCHEMA.names() == list(CANONICAL_COLUMN_ORDER)
    for column in CANONICAL_COLUMN_ORDER:
        assert MERGED_EXIT_ENGINE_SCHEMA[column] == COLUMN_DTYPES[column]
    assert COLUMN_DTYPES["open_time"] == pl.Datetime("us", "UTC")
    assert COLUMN_DTYPES["created_at"] == pl.Datetime("us", "UTC")
    assert COLUMN_DTYPES["entry_price"] == pl.Float64
    assert COLUMN_DTYPES["recommended_percent"] == pl.Float64
    assert COLUMN_DTYPES["priority"] == pl.Int64
    assert COLUMN_DTYPES["exit_action"] == pl.Utf8
    assert COLUMN_DTYPES["exit_reason"] == pl.Utf8


def test_canonical_order_starts_with_identity_keys() -> None:
    """Canonical column order begins with symbol/timeframe/open_time/position_id."""
    assert CANONICAL_COLUMN_ORDER[0] == "symbol"
    assert CANONICAL_COLUMN_ORDER[1] == "timeframe"
    assert CANONICAL_COLUMN_ORDER[2] == "open_time"
    assert CANONICAL_COLUMN_ORDER[3] == "position_id"
    assert CANONICAL_COLUMN_ORDER[-1] == "created_at"


def test_exit_action_enum_members() -> None:
    """ExitAction exposes HOLD, PARTIAL_EXIT, and FULL_EXIT members."""
    assert ExitAction.HOLD.value == "HOLD"
    assert ExitAction.PARTIAL_EXIT.value == "PARTIAL_EXIT"
    assert ExitAction.FULL_EXIT.value == "FULL_EXIT"
    assert len(list(ExitAction)) == 3


def test_exit_reason_enum_members() -> None:
    """ExitReason exposes every v1 enumeration member with correct string values."""
    assert ExitReason.NONE.value == "NONE"
    assert ExitReason.TAKE_PROFIT.value == "TAKE_PROFIT"
    assert ExitReason.STOP_LOSS.value == "STOP_LOSS"
    assert ExitReason.TRAILING_STOP.value == "TRAILING_STOP"
    assert ExitReason.BREAK_EVEN.value == "BREAK_EVEN"
    assert ExitReason.ALPHA_DECAY.value == "ALPHA_DECAY"
    assert ExitReason.TIME_STOP.value == "TIME_STOP"
    assert ExitReason.PORTFOLIO_SHUTDOWN.value == "PORTFOLIO_SHUTDOWN"
    assert ExitReason.COOLDOWN.value == "COOLDOWN"
    assert ExitReason.REGIME_EXIT.value == "REGIME_EXIT"
    assert ExitReason.EMERGENCY_EXIT.value == "EMERGENCY_EXIT"


def test_exit_actions_helper() -> None:
    """exit_actions() returns a tuple of all three ExitAction members."""
    actions = exit_actions()
    assert actions == (ExitAction.HOLD, ExitAction.PARTIAL_EXIT, ExitAction.FULL_EXIT)
    assert isinstance(actions, tuple)


def test_exit_reasons_helper() -> None:
    """exit_reasons() returns a tuple of all ExitReason members in declaration order."""
    reasons = exit_reasons()
    assert ExitReason.NONE in reasons
    assert ExitReason.PORTFOLIO_SHUTDOWN in reasons
    assert ExitReason.TAKE_PROFIT in reasons
    assert ExitReason.COOLDOWN in reasons
    assert isinstance(reasons, tuple)
    assert len(reasons) == len(list(ExitReason))


def test_values_helper_extracts_string_values() -> None:
    """values() returns a tuple of string values for an enum class."""
    action_values = values(ExitAction)
    assert action_values == ("HOLD", "PARTIAL_EXIT", "FULL_EXIT")
    reason_values = values(ExitReason)
    assert "TAKE_PROFIT" in reason_values
    assert "PORTFOLIO_SHUTDOWN" in reason_values
    assert isinstance(reason_values, tuple)


def test_default_rule_constants() -> None:
    """Default exit-rule constants match the v1 contract."""
    assert DEFAULT_INITIAL_RISK_PERCENT == 0.05
    assert DEFAULT_TAKE_PROFIT_MULTIPLE == 3.0
    assert DEFAULT_PARTIAL_EXIT_PERCENT == 0.50


def test_priority_ordering() -> None:
    """Priority constants follow the documented urgency order (lower = higher urgency)."""
    assert PRIORITY_PORTFOLIO_SHUTDOWN < PRIORITY_COOLDOWN
    assert PRIORITY_COOLDOWN < PRIORITY_TRAILING_STOP
    assert PRIORITY_TRAILING_STOP < PRIORITY_BREAK_EVEN
    assert PRIORITY_BREAK_EVEN < PRIORITY_TAKE_PROFIT
    assert PRIORITY_TAKE_PROFIT < PRIORITY_ALPHA_DECAY
    assert PRIORITY_ALPHA_DECAY < PRIORITY_TIME_STOP
    assert PRIORITY_TIME_STOP < PRIORITY_REGIME_EXIT
    assert PRIORITY_HOLD == 0
    assert PRIORITY_PORTFOLIO_SHUTDOWN == 1
    assert PRIORITY_COOLDOWN == 2
    assert PRIORITY_TRAILING_STOP == 3
    assert PRIORITY_BREAK_EVEN == 4
    assert PRIORITY_TAKE_PROFIT == 5
    assert PRIORITY_ALPHA_DECAY == 6
    assert PRIORITY_TIME_STOP == 7
    assert PRIORITY_REGIME_EXIT == 8
