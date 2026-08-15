"""Unit tests for CQROS merged pyramiding recommendation schema."""

from __future__ import annotations

import polars as pl

from cqros.pyramiding import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    DEFAULT_ADD_FRACTION,
    DEFAULT_MAX_ADDS,
    DEFAULT_MIN_PROFIT_PERCENT,
    MERGED_PYRAMIDING_SCHEMA,
    PRIMARY_KEY_COLUMNS,
    PYRAMIDING_COLUMNS,
    REQUIRED_COLUMNS,
    PyramidingReason,
    pyramiding_reasons,
    values,
)
from cqros.pyramiding.schema import (
    MERGED_PYRAMIDING_SCHEMA as MERGED_PYRAMIDING_SCHEMA_DIRECT,
)


def test_primary_key_and_required_columns() -> None:
    """Primary key and required columns match the canonical pyramiding contract."""
    assert PRIMARY_KEY_COLUMNS == ("symbol", "timeframe", "open_time", "position_id")
    assert REQUIRED_COLUMNS == CANONICAL_COLUMN_ORDER
    assert CANONICAL_COLUMN_ORDER == PYRAMIDING_COLUMNS


def test_pyramiding_columns_contain_required_domain_columns() -> None:
    """PYRAMIDING_COLUMNS enumerates identity, pricing, sizing, and decision fields."""
    for column in (
        "manager",
        "symbol",
        "timeframe",
        "open_time",
        "position_id",
        "trade_id",
        "entry_price",
        "current_price",
        "highest_price",
        "position_size",
        "add_number",
        "max_adds",
        "additional_size",
        "recommended_size",
        "profit_pct",
        "allow_pyramid",
        "reason",
    ):
        assert column in PYRAMIDING_COLUMNS


def test_column_dtypes_and_merged_schema() -> None:
    """Merged schema dtypes match COLUMN_DTYPES in canonical order."""
    assert MERGED_PYRAMIDING_SCHEMA is MERGED_PYRAMIDING_SCHEMA_DIRECT
    assert MERGED_PYRAMIDING_SCHEMA.names() == list(CANONICAL_COLUMN_ORDER)
    for column in CANONICAL_COLUMN_ORDER:
        assert MERGED_PYRAMIDING_SCHEMA[column] == COLUMN_DTYPES[column]
    assert COLUMN_DTYPES["open_time"] == pl.Datetime("us", "UTC")
    assert COLUMN_DTYPES["entry_price"] == pl.Float64
    assert COLUMN_DTYPES["allow_pyramid"] == pl.Boolean
    assert COLUMN_DTYPES["add_number"] == pl.Int64
    assert COLUMN_DTYPES["reason"] == pl.Utf8


def test_canonical_order_starts_with_manager() -> None:
    """Canonical column order begins with manager then identity keys."""
    assert CANONICAL_COLUMN_ORDER[0] == "manager"
    assert CANONICAL_COLUMN_ORDER[1] == "symbol"
    assert CANONICAL_COLUMN_ORDER[2] == "timeframe"
    assert CANONICAL_COLUMN_ORDER[3] == "open_time"
    assert CANONICAL_COLUMN_ORDER[-1] == "reason"


def test_pyramiding_reason_helpers() -> None:
    """Reason helpers expose every v1 enumeration member."""
    assert PyramidingReason.READY_TO_ADD.value == "READY_TO_ADD"
    assert PyramidingReason.NOT_ELIGIBLE.value == "NOT_ELIGIBLE"
    assert pyramiding_reasons() == (
        PyramidingReason.NOT_ELIGIBLE,
        PyramidingReason.INSUFFICIENT_PROFIT,
        PyramidingReason.MAX_ADDS_REACHED,
        PyramidingReason.PORTFOLIO_WARNING,
        PyramidingReason.PORTFOLIO_SHUTDOWN,
        PyramidingReason.COOLDOWN_ACTIVE,
        PyramidingReason.TRAILING_STOP_ACTIVE,
        PyramidingReason.BREAKEVEN_ACTIVE,
        PyramidingReason.READY_TO_ADD,
    )
    assert values(PyramidingReason) == (
        "NOT_ELIGIBLE",
        "INSUFFICIENT_PROFIT",
        "MAX_ADDS_REACHED",
        "PORTFOLIO_WARNING",
        "PORTFOLIO_SHUTDOWN",
        "COOLDOWN_ACTIVE",
        "TRAILING_STOP_ACTIVE",
        "BREAKEVEN_ACTIVE",
        "READY_TO_ADD",
    )


def test_default_rule_constants() -> None:
    """Default pyramiding rule constants match the v1 contract."""
    assert DEFAULT_MAX_ADDS == 3
    assert DEFAULT_ADD_FRACTION == 0.50
    assert DEFAULT_MIN_PROFIT_PERCENT == 0.05
