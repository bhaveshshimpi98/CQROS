"""Unit tests for CQROS position schema."""

from __future__ import annotations

import polars as pl

from cqros.positions import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    MERGED_POSITION_SCHEMA,
    METADATA_COLUMNS,
    POSITION_COLUMNS,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    PositionSide,
    PositionStatus,
    position_sides,
    position_statuses,
    values,
)
from cqros.positions.schema import MERGED_POSITION_SCHEMA as MERGED_POSITION_SCHEMA_DIRECT


def test_primary_key_and_required_columns() -> None:
    """Primary key and required columns match the canonical position contract."""
    assert PRIMARY_KEY_COLUMNS == ("symbol", "timeframe", "position_id")
    assert REQUIRED_COLUMNS == CANONICAL_COLUMN_ORDER
    assert "manager" in METADATA_COLUMNS
    assert "policy" in METADATA_COLUMNS
    assert "quantity" in POSITION_COLUMNS
    assert "average_entry_price" in POSITION_COLUMNS
    assert "realized_pnl" in POSITION_COLUMNS
    assert "unrealized_pnl" in POSITION_COLUMNS
    assert "opened_at" in POSITION_COLUMNS
    assert "closed_at" in POSITION_COLUMNS


def test_column_dtypes_and_merged_schema() -> None:
    """Merged schema dtypes match COLUMN_DTYPES in canonical order."""
    assert MERGED_POSITION_SCHEMA is MERGED_POSITION_SCHEMA_DIRECT
    assert MERGED_POSITION_SCHEMA.names() == list(CANONICAL_COLUMN_ORDER)
    for column in CANONICAL_COLUMN_ORDER:
        assert MERGED_POSITION_SCHEMA[column] == COLUMN_DTYPES[column]
    assert COLUMN_DTYPES["opened_at"] == pl.Datetime("us", "UTC")
    assert COLUMN_DTYPES["closed_at"] == pl.Datetime("us", "UTC")
    assert COLUMN_DTYPES["quantity"] == pl.Float64
    assert COLUMN_DTYPES["realized_pnl"] == pl.Float64


def test_position_enum_helpers() -> None:
    """PositionSide and PositionStatus helpers expose v1 members only."""
    assert PositionSide.LONG.value == "LONG"
    assert PositionStatus.OPEN.value == "OPEN"
    assert PositionStatus.CLOSED.value == "CLOSED"
    assert position_sides() == (PositionSide.LONG,)
    assert position_statuses() == (PositionStatus.OPEN, PositionStatus.CLOSED)
    assert values(PositionSide) == ("LONG",)
    assert values(PositionStatus) == ("OPEN", "CLOSED")
