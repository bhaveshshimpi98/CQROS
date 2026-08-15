"""CQROS merged position dataset schema.

Purpose:
    Define the canonical columnar contract for portfolio positions produced by
    the CQROS Position Engine from executed-trade datasets.

Responsibilities:
    - Declare the merged-dataset primary key
    - Enumerate model metadata and position accounting columns
    - Expose required columns, canonical column order, and expected dtypes
    - Remain free of accounting, validation, and persistence logic

Dependencies:
    ``polars`` and the Python standard library.

Public API:
    ``PRIMARY_KEY_COLUMNS``, ``METADATA_COLUMNS``, ``POSITION_COLUMNS``,
    ``REQUIRED_COLUMNS``, ``CANONICAL_COLUMN_ORDER``, ``COLUMN_DTYPES``,
    ``MERGED_POSITION_SCHEMA``, ``PositionSide``, ``PositionStatus``,
    ``position_sides``, ``position_statuses``, ``values``

Notes:
    This module describes column presence and dtypes only; it does not
    validate frames, compute average-cost accounting, or persist positions.
    ``manager``, ``optimizer``, and ``policy`` preserve upstream execution /
    OMS / risk lineage on every position row. Position Engine v1 supports
    long-only cash positions (no leverage, shorts, funding, or margin).
"""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType
from typing import Final

import polars as pl

__all__ = [
    "CANONICAL_COLUMN_ORDER",
    "COLUMN_DTYPES",
    "MERGED_POSITION_SCHEMA",
    "METADATA_COLUMNS",
    "POSITION_COLUMNS",
    "PRIMARY_KEY_COLUMNS",
    "REQUIRED_COLUMNS",
    "PositionSide",
    "PositionStatus",
    "position_sides",
    "position_statuses",
    "values",
]

PRIMARY_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "position_id",
)

# Model identity and upstream construction lineage preserved onto each position.
METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "model_name",
    "model_version",
    "optimizer",
    "policy",
    "manager",
)

# Side, lifecycle, sizing, pricing, PnL, fees, and timestamp columns.
POSITION_COLUMNS: Final[tuple[str, ...]] = (
    "side",
    "status",
    "quantity",
    "average_entry_price",
    "market_price",
    "realized_pnl",
    "unrealized_pnl",
    "fees_paid",
    "opened_at",
    "updated_at",
    "closed_at",
)

CANONICAL_COLUMN_ORDER: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "position_id",
    "side",
    "status",
    "quantity",
    "average_entry_price",
    "market_price",
    "realized_pnl",
    "unrealized_pnl",
    "fees_paid",
    "opened_at",
    "updated_at",
    "closed_at",
    *METADATA_COLUMNS,
)

REQUIRED_COLUMNS: Final[tuple[str, ...]] = CANONICAL_COLUMN_ORDER

COLUMN_DTYPES: Final = MappingProxyType(
    {
        "symbol": pl.Utf8,
        "timeframe": pl.Utf8,
        "position_id": pl.Utf8,
        "side": pl.Utf8,
        "status": pl.Utf8,
        "quantity": pl.Float64,
        "average_entry_price": pl.Float64,
        "market_price": pl.Float64,
        "realized_pnl": pl.Float64,
        "unrealized_pnl": pl.Float64,
        "fees_paid": pl.Float64,
        "opened_at": pl.Datetime("us", "UTC"),
        "updated_at": pl.Datetime("us", "UTC"),
        "closed_at": pl.Datetime("us", "UTC"),
        "model_name": pl.Utf8,
        "model_version": pl.Utf8,
        "optimizer": pl.Utf8,
        "policy": pl.Utf8,
        "manager": pl.Utf8,
    }
)

MERGED_POSITION_SCHEMA: Final[pl.Schema] = pl.Schema(
    [(column, COLUMN_DTYPES[column]) for column in CANONICAL_COLUMN_ORDER]
)


# Prefer str+Enum over StrEnum so enum values embed directly into Polars.
class PositionSide(str, Enum):  # noqa: UP042
    """Canonical position direction for Position Engine v1.

    Attributes:
        LONG: Net long cash exposure. Shorts are not supported in v1.
    """

    LONG = "LONG"


class PositionStatus(str, Enum):  # noqa: UP042
    """Canonical position lifecycle status.

    Attributes:
        OPEN: Position has non-zero quantity.
        CLOSED: Position quantity has been fully reduced to zero.
    """

    OPEN = "OPEN"
    CLOSED = "CLOSED"


def position_sides() -> tuple[PositionSide, ...]:
    """Return an immutable copy of every ``PositionSide`` member.

    Returns:
        All position-side members in declaration order.
    """
    return (PositionSide.LONG,)


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
