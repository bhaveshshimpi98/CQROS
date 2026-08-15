"""CQROS merged pyramiding recommendation dataset schema.

Purpose:
    Define the canonical columnar contract for pyramiding recommendations
    produced by the CQROS Pyramiding Engine from open position snapshots,
    accounting, portfolio risk, trade management, and market prices.

Responsibilities:
    - Declare the merged-dataset primary key
    - Enumerate pyramiding recommendation columns
    - Expose required columns, canonical column order, and expected dtypes
    - Expose pyramiding reason enumerations
    - Centralize default pyramiding rule constants
    - Remain free of pyramiding math, validation, and persistence logic

Dependencies:
    ``polars`` and the Python standard library.

Public API:
    ``PRIMARY_KEY_COLUMNS``, ``PYRAMIDING_COLUMNS``, ``REQUIRED_COLUMNS``,
    ``CANONICAL_COLUMN_ORDER``, ``COLUMN_DTYPES``,
    ``MERGED_PYRAMIDING_SCHEMA``, ``PyramidingReason``, ``pyramiding_reasons``,
    ``values``, ``DEFAULT_MAX_ADDS``, ``DEFAULT_ADD_FRACTION``,
    ``DEFAULT_MIN_PROFIT_PERCENT``

Notes:
    This module describes column presence and dtypes only; it does not
    evaluate pyramiding rules, validate frames, or persist recommendations.
    ``manager`` preserves upstream order-manager lineage on every row.
"""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType
from typing import Final

import polars as pl

__all__ = [
    "CANONICAL_COLUMN_ORDER",
    "COLUMN_DTYPES",
    "DEFAULT_ADD_FRACTION",
    "DEFAULT_MAX_ADDS",
    "DEFAULT_MIN_PROFIT_PERCENT",
    "MERGED_PYRAMIDING_SCHEMA",
    "PRIMARY_KEY_COLUMNS",
    "PYRAMIDING_COLUMNS",
    "REQUIRED_COLUMNS",
    "PyramidingReason",
    "pyramiding_reasons",
    "values",
]

# Default pyramiding rule limits (v1). No magic numbers elsewhere.
DEFAULT_MAX_ADDS: Final[int] = 3
DEFAULT_ADD_FRACTION: Final[float] = 0.50
DEFAULT_MIN_PROFIT_PERCENT: Final[float] = 0.05

PRIMARY_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
    "position_id",
)

# Position identity, pricing, sizing, and pyramiding-decision columns.
PYRAMIDING_COLUMNS: Final[tuple[str, ...]] = (
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
)

CANONICAL_COLUMN_ORDER: Final[tuple[str, ...]] = PYRAMIDING_COLUMNS

REQUIRED_COLUMNS: Final[tuple[str, ...]] = CANONICAL_COLUMN_ORDER

COLUMN_DTYPES: Final = MappingProxyType(
    {
        "manager": pl.Utf8,
        "symbol": pl.Utf8,
        "timeframe": pl.Utf8,
        "open_time": pl.Datetime("us", "UTC"),
        "position_id": pl.Utf8,
        "trade_id": pl.Utf8,
        "entry_price": pl.Float64,
        "current_price": pl.Float64,
        "highest_price": pl.Float64,
        "position_size": pl.Float64,
        "add_number": pl.Int64,
        "max_adds": pl.Int64,
        "additional_size": pl.Float64,
        "recommended_size": pl.Float64,
        "profit_pct": pl.Float64,
        "allow_pyramid": pl.Boolean,
        "reason": pl.Utf8,
    }
)

MERGED_PYRAMIDING_SCHEMA: Final[pl.Schema] = pl.Schema(
    [(column, COLUMN_DTYPES[column]) for column in CANONICAL_COLUMN_ORDER]
)


# Prefer str+Enum over StrEnum so enum values embed directly into Polars.
class PyramidingReason(str, Enum):  # noqa: UP042
    """Canonical reasons recorded on the ``reason`` column.

    Attributes:
        NOT_ELIGIBLE: Position is closed, non-long, or otherwise ineligible.
        INSUFFICIENT_PROFIT: Unrealized profit has not reached the next add
            threshold.
        MAX_ADDS_REACHED: The configured maximum number of adds is already
            completed.
        PORTFOLIO_WARNING: Portfolio risk is in WARNING state.
        PORTFOLIO_SHUTDOWN: Portfolio risk is in SHUTDOWN state.
        COOLDOWN_ACTIVE: Portfolio-risk cooldown is active.
        TRAILING_STOP_ACTIVE: Trade management reports a trailing-stop action.
        BREAKEVEN_ACTIVE: Trade management reports a break-even action.
        READY_TO_ADD: All eligibility and profit rules allow an add.
    """

    NOT_ELIGIBLE = "NOT_ELIGIBLE"
    INSUFFICIENT_PROFIT = "INSUFFICIENT_PROFIT"
    MAX_ADDS_REACHED = "MAX_ADDS_REACHED"
    PORTFOLIO_WARNING = "PORTFOLIO_WARNING"
    PORTFOLIO_SHUTDOWN = "PORTFOLIO_SHUTDOWN"
    COOLDOWN_ACTIVE = "COOLDOWN_ACTIVE"
    TRAILING_STOP_ACTIVE = "TRAILING_STOP_ACTIVE"
    BREAKEVEN_ACTIVE = "BREAKEVEN_ACTIVE"
    READY_TO_ADD = "READY_TO_ADD"


def pyramiding_reasons() -> tuple[PyramidingReason, ...]:
    """Return an immutable copy of every ``PyramidingReason`` member.

    Returns:
        All pyramiding-reason members in declaration order.
    """
    return (
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


def values[EnumT: Enum](enum_cls: type[EnumT]) -> tuple[str, ...]:
    """Return an immutable copy of every string value for ``enum_cls``.

    Args:
        enum_cls: Enumeration class whose member values are requested.

    Returns:
        All member string values in declaration order.
    """
    return tuple(member.value for member in enum_cls)
