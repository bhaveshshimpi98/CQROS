"""Unit tests for CQROS executed-trade schema."""

from __future__ import annotations

import polars as pl

from cqros.execution import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    MERGED_TRADE_SCHEMA,
    METADATA_COLUMNS,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    TRADE_COLUMNS,
    ExecutionStatus,
    execution_statuses,
    values,
)
from cqros.execution.schema import MERGED_TRADE_SCHEMA as MERGED_TRADE_SCHEMA_DIRECT


def test_primary_key_and_required_columns() -> None:
    """Primary key and required columns match the canonical trade contract."""
    assert PRIMARY_KEY_COLUMNS == ("symbol", "timeframe", "open_time")
    assert REQUIRED_COLUMNS == CANONICAL_COLUMN_ORDER
    assert "manager" in METADATA_COLUMNS
    assert "signal" in METADATA_COLUMNS
    assert "requested_quantity" in TRADE_COLUMNS
    assert "executed_quantity" in TRADE_COLUMNS
    assert "status" in TRADE_COLUMNS
    assert "execution_time" in TRADE_COLUMNS


def test_column_dtypes_and_merged_schema() -> None:
    """Merged schema dtypes match COLUMN_DTYPES in canonical order."""
    assert MERGED_TRADE_SCHEMA is MERGED_TRADE_SCHEMA_DIRECT
    assert MERGED_TRADE_SCHEMA.names() == list(CANONICAL_COLUMN_ORDER)
    for column in CANONICAL_COLUMN_ORDER:
        assert MERGED_TRADE_SCHEMA[column] == COLUMN_DTYPES[column]
    assert COLUMN_DTYPES["open_time"] == pl.Datetime("us", "UTC")
    assert COLUMN_DTYPES["execution_time"] == pl.Datetime("us", "UTC")
    assert COLUMN_DTYPES["requested_quantity"] == pl.Float64
    assert COLUMN_DTYPES["fees"] == pl.Float64


def test_execution_status_helpers() -> None:
    """ExecutionStatus helpers expose FILLED only for v1."""
    assert ExecutionStatus.FILLED.value == "FILLED"
    assert execution_statuses() == (ExecutionStatus.FILLED,)
    assert values(ExecutionStatus) == ("FILLED",)
