"""Unit tests for CQROS merged portfolio accounting schema."""

from __future__ import annotations

import polars as pl

from cqros.accounting import (
    ACCOUNTING_COLUMNS,
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    MERGED_ACCOUNTING_SCHEMA,
    METADATA_COLUMNS,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    PositionStatus,
    position_statuses,
    values,
)
from cqros.accounting.schema import MERGED_ACCOUNTING_SCHEMA as MERGED_ACCOUNTING_SCHEMA_DIRECT


def test_primary_key_and_required_columns() -> None:
    """Primary key and required columns match the canonical accounting contract."""
    assert PRIMARY_KEY_COLUMNS == ("symbol", "timeframe", "open_time", "position_id")
    assert REQUIRED_COLUMNS == CANONICAL_COLUMN_ORDER
    assert "model_name" in METADATA_COLUMNS
    assert "model_version" in METADATA_COLUMNS
    assert "optimizer" in METADATA_COLUMNS
    assert "policy" in METADATA_COLUMNS


def test_accounting_columns_contain_required_domain_columns() -> None:
    """ACCOUNTING_COLUMNS enumerates identity, mark-to-market, and return fields."""
    for column in (
        "manager",
        "position_id",
        "position_status",
        "quantity",
        "average_entry_price",
        "mark_price",
        "position_value",
        "market_value",
        "cash",
        "realized_pnl",
        "unrealized_pnl",
        "total_pnl",
        "gross_exposure",
        "net_exposure",
        "equity",
        "return_pct",
    ):
        assert column in ACCOUNTING_COLUMNS


def test_column_dtypes_and_merged_schema() -> None:
    """Merged schema dtypes match COLUMN_DTYPES in canonical order."""
    assert MERGED_ACCOUNTING_SCHEMA is MERGED_ACCOUNTING_SCHEMA_DIRECT
    assert MERGED_ACCOUNTING_SCHEMA.names() == list(CANONICAL_COLUMN_ORDER)
    for column in CANONICAL_COLUMN_ORDER:
        assert MERGED_ACCOUNTING_SCHEMA[column] == COLUMN_DTYPES[column]
    assert COLUMN_DTYPES["open_time"] == pl.Datetime("us", "UTC")
    assert COLUMN_DTYPES["quantity"] == pl.Float64
    assert COLUMN_DTYPES["equity"] == pl.Float64
    assert COLUMN_DTYPES["return_pct"] == pl.Float64
    assert COLUMN_DTYPES["symbol"] == pl.Utf8


def test_canonical_order_ends_with_metadata_columns() -> None:
    """Canonical column order terminates with the lineage metadata columns."""
    assert CANONICAL_COLUMN_ORDER[-len(METADATA_COLUMNS) :] == METADATA_COLUMNS
    assert CANONICAL_COLUMN_ORDER[0] == "symbol"
    assert CANONICAL_COLUMN_ORDER[1] == "timeframe"
    assert CANONICAL_COLUMN_ORDER[2] == "open_time"


def test_position_status_helpers() -> None:
    """PositionStatus helpers expose only the v1 lifecycle members."""
    assert PositionStatus.OPEN.value == "OPEN"
    assert PositionStatus.CLOSED.value == "CLOSED"
    assert position_statuses() == (PositionStatus.OPEN, PositionStatus.CLOSED)
    assert values(PositionStatus) == ("OPEN", "CLOSED")
