"""Unit tests for CQROS merged trade management decision schema."""

from __future__ import annotations

import polars as pl

from cqros.trade_management import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    DEFAULT_INITIAL_RISK_PERCENT,
    DEFAULT_TRAIL_PERCENT,
    MERGED_TRADE_MANAGEMENT_SCHEMA,
    METADATA_COLUMNS,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    TRADE_MANAGEMENT_COLUMNS,
    ManagementAction,
    ShutdownReason,
    management_actions,
    shutdown_reasons,
    values,
)
from cqros.trade_management.schema import (
    MERGED_TRADE_MANAGEMENT_SCHEMA as MERGED_TRADE_MANAGEMENT_SCHEMA_DIRECT,
)


def test_primary_key_and_required_columns() -> None:
    """Primary key and required columns match the canonical trade-management contract."""
    assert PRIMARY_KEY_COLUMNS == ("symbol", "timeframe", "open_time", "position_id")
    assert REQUIRED_COLUMNS == CANONICAL_COLUMN_ORDER
    assert "model_name" in METADATA_COLUMNS
    assert "model_version" in METADATA_COLUMNS
    assert "optimizer" in METADATA_COLUMNS
    assert "policy" in METADATA_COLUMNS


def test_trade_management_columns_contain_required_domain_columns() -> None:
    """TRADE_MANAGEMENT_COLUMNS enumerates identity, pricing, and decision fields."""
    for column in (
        "manager",
        "position_id",
        "position_status",
        "quantity",
        "entry_price",
        "current_price",
        "highest_price",
        "lowest_price",
        "unrealized_pnl",
        "risk_state",
        "management_action",
        "action_reason",
        "stop_price",
        "take_profit_price",
        "trail_price",
        "breakeven_price",
        "allow_pyramid",
        "exit_quantity",
    ):
        assert column in TRADE_MANAGEMENT_COLUMNS


def test_column_dtypes_and_merged_schema() -> None:
    """Merged schema dtypes match COLUMN_DTYPES in canonical order."""
    assert MERGED_TRADE_MANAGEMENT_SCHEMA is MERGED_TRADE_MANAGEMENT_SCHEMA_DIRECT
    assert MERGED_TRADE_MANAGEMENT_SCHEMA.names() == list(CANONICAL_COLUMN_ORDER)
    for column in CANONICAL_COLUMN_ORDER:
        assert MERGED_TRADE_MANAGEMENT_SCHEMA[column] == COLUMN_DTYPES[column]
    assert COLUMN_DTYPES["open_time"] == pl.Datetime("us", "UTC")
    assert COLUMN_DTYPES["entry_price"] == pl.Float64
    assert COLUMN_DTYPES["allow_pyramid"] == pl.Boolean
    assert COLUMN_DTYPES["management_action"] == pl.Utf8
    assert COLUMN_DTYPES["symbol"] == pl.Utf8


def test_canonical_order_ends_with_metadata_columns() -> None:
    """Canonical column order terminates with the lineage metadata columns."""
    assert CANONICAL_COLUMN_ORDER[-len(METADATA_COLUMNS) :] == METADATA_COLUMNS
    assert CANONICAL_COLUMN_ORDER[0] == "symbol"
    assert CANONICAL_COLUMN_ORDER[1] == "timeframe"
    assert CANONICAL_COLUMN_ORDER[2] == "open_time"


def test_management_action_and_shutdown_reason_helpers() -> None:
    """Action and reason helpers expose every v1 enumeration member."""
    assert ManagementAction.NONE.value == "NONE"
    assert ManagementAction.UPDATE_STOP.value == "UPDATE_STOP"
    assert management_actions() == (
        ManagementAction.NONE,
        ManagementAction.UPDATE_STOP,
        ManagementAction.PARTIAL_EXIT,
        ManagementAction.FULL_EXIT,
        ManagementAction.ALLOW_PYRAMID,
    )
    assert values(ManagementAction) == (
        "NONE",
        "UPDATE_STOP",
        "PARTIAL_EXIT",
        "FULL_EXIT",
        "ALLOW_PYRAMID",
    )

    assert ShutdownReason.NONE.value == "NONE"
    assert ShutdownReason.TRAILING_STOP.value == "TRAILING_STOP"
    assert ShutdownReason.BREAKEVEN.value == "BREAKEVEN"
    assert shutdown_reasons() == (
        ShutdownReason.NONE,
        ShutdownReason.TRAILING_STOP,
        ShutdownReason.BREAKEVEN,
        ShutdownReason.PARTIAL_PROFIT,
        ShutdownReason.ALPHA_DECAY,
        ShutdownReason.TIME_EXIT,
        ShutdownReason.PORTFOLIO_RISK,
    )
    assert values(ShutdownReason) == (
        "NONE",
        "TRAILING_STOP",
        "BREAKEVEN",
        "PARTIAL_PROFIT",
        "ALPHA_DECAY",
        "TIME_EXIT",
        "PORTFOLIO_RISK",
    )


def test_default_rule_constants() -> None:
    """Default trade-management rule constants match the v1 contract."""
    assert DEFAULT_TRAIL_PERCENT == 0.05
    assert DEFAULT_INITIAL_RISK_PERCENT == 0.05
