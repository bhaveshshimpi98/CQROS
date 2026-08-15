"""CQROS merged portfolio accounting dataset schema.

Purpose:
    Define the canonical columnar contract for portfolio accounting snapshots
    produced by the CQROS Portfolio Accounting Engine from position datasets.

Responsibilities:
    - Declare the merged-dataset primary key
    - Enumerate model metadata and accounting columns
    - Expose required columns, canonical column order, and expected dtypes
    - Remain free of accounting math, validation, and persistence logic

Dependencies:
    ``polars`` and the Python standard library.

Public API:
    ``PRIMARY_KEY_COLUMNS``, ``METADATA_COLUMNS``, ``ACCOUNTING_COLUMNS``,
    ``REQUIRED_COLUMNS``, ``CANONICAL_COLUMN_ORDER``, ``COLUMN_DTYPES``,
    ``MERGED_ACCOUNTING_SCHEMA``, ``PositionStatus``, ``position_statuses``,
    ``values``

Notes:
    This module describes column presence and dtypes only; it does not
    validate frames, compute mark-to-market accounting, or persist snapshots.
    ``manager``, ``optimizer``, and ``policy`` preserve upstream position /
    execution / OMS lineage on every accounting row. Accounting Engine v1
    supports long-only cash mark-to-market accounting (no leverage, funding,
    borrowing, commissions, slippage, or interest).
"""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType
from typing import Final

import polars as pl

__all__ = [
    "ACCOUNTING_COLUMNS",
    "CANONICAL_COLUMN_ORDER",
    "COLUMN_DTYPES",
    "MERGED_ACCOUNTING_SCHEMA",
    "METADATA_COLUMNS",
    "PRIMARY_KEY_COLUMNS",
    "REQUIRED_COLUMNS",
    "PositionStatus",
    "position_statuses",
    "values",
]

PRIMARY_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
    "position_id",
)

# Model identity and upstream construction lineage preserved onto each row.
METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "model_name",
    "model_version",
    "optimizer",
    "policy",
)

# Position identity, mark-to-market, cash, PnL, exposure, and return columns.
ACCOUNTING_COLUMNS: Final[tuple[str, ...]] = (
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
)

CANONICAL_COLUMN_ORDER: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
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
    *METADATA_COLUMNS,
)

REQUIRED_COLUMNS: Final[tuple[str, ...]] = CANONICAL_COLUMN_ORDER

COLUMN_DTYPES: Final = MappingProxyType(
    {
        "symbol": pl.Utf8,
        "timeframe": pl.Utf8,
        "open_time": pl.Datetime("us", "UTC"),
        "manager": pl.Utf8,
        "position_id": pl.Utf8,
        "position_status": pl.Utf8,
        "quantity": pl.Float64,
        "average_entry_price": pl.Float64,
        "mark_price": pl.Float64,
        "position_value": pl.Float64,
        "market_value": pl.Float64,
        "cash": pl.Float64,
        "realized_pnl": pl.Float64,
        "unrealized_pnl": pl.Float64,
        "total_pnl": pl.Float64,
        "gross_exposure": pl.Float64,
        "net_exposure": pl.Float64,
        "equity": pl.Float64,
        "return_pct": pl.Float64,
        "model_name": pl.Utf8,
        "model_version": pl.Utf8,
        "optimizer": pl.Utf8,
        "policy": pl.Utf8,
    }
)

MERGED_ACCOUNTING_SCHEMA: Final[pl.Schema] = pl.Schema(
    [(column, COLUMN_DTYPES[column]) for column in CANONICAL_COLUMN_ORDER]
)


# Prefer str+Enum over StrEnum so enum values embed directly into Polars.
class PositionStatus(str, Enum):  # noqa: UP042
    """Canonical position lifecycle status mirrored onto accounting rows.

    Attributes:
        OPEN: Position has non-zero quantity.
        CLOSED: Position quantity has been fully reduced to zero.
    """

    OPEN = "OPEN"
    CLOSED = "CLOSED"


def position_statuses() -> tuple[PositionStatus, ...]:
    """Return an immutable copy of every ``PositionStatus`` member.

    Returns:
        All position-status members in declaration order.
    """
    return (PositionStatus.OPEN, PositionStatus.CLOSED)


def values[EnumT: Enum](enum_cls: type[EnumT]) -> tuple[str, ...]:
    """Return an immutable copy of every string value for ``enum_cls``.

    Args:
        enum_cls: Enumeration class whose member values are requested.

    Returns:
        All member string values in declaration order.
    """
    return tuple(member.value for member in enum_cls)
