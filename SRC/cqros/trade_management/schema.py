"""CQROS merged trade management decision dataset schema.

Purpose:
    Define the canonical columnar contract for trade management decisions
    produced by the CQROS Trade Management Engine from open position
    snapshots, accounting, portfolio risk, and market prices.

Responsibilities:
    - Declare the merged-dataset primary key
    - Enumerate model metadata and trade-management decision columns
    - Expose required columns, canonical column order, and expected dtypes
    - Expose management action and action-reason enumerations
    - Centralize default trade-management rule constants
    - Remain free of management math, validation, and persistence logic

Dependencies:
    ``polars`` and the Python standard library.

Public API:
    ``PRIMARY_KEY_COLUMNS``, ``METADATA_COLUMNS``, ``TRADE_MANAGEMENT_COLUMNS``,
    ``REQUIRED_COLUMNS``, ``CANONICAL_COLUMN_ORDER``, ``COLUMN_DTYPES``,
    ``MERGED_TRADE_MANAGEMENT_SCHEMA``, ``ManagementAction``,
    ``ShutdownReason``, ``management_actions``, ``shutdown_reasons``,
    ``values``, ``DEFAULT_TRAIL_PERCENT``, ``DEFAULT_INITIAL_RISK_PERCENT``

Notes:
    This module describes column presence and dtypes only; it does not
    evaluate management rules, validate frames, or persist decisions.
    ``manager``, ``optimizer``, and ``policy`` preserve upstream accounting /
    position / portfolio-risk lineage on every trade-management row.
    ``ShutdownReason`` values populate the ``action_reason`` column.
"""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType
from typing import Final

import polars as pl

__all__ = [
    "CANONICAL_COLUMN_ORDER",
    "COLUMN_DTYPES",
    "DEFAULT_INITIAL_RISK_PERCENT",
    "DEFAULT_TRAIL_PERCENT",
    "MERGED_TRADE_MANAGEMENT_SCHEMA",
    "METADATA_COLUMNS",
    "PRIMARY_KEY_COLUMNS",
    "REQUIRED_COLUMNS",
    "TRADE_MANAGEMENT_COLUMNS",
    "ManagementAction",
    "ShutdownReason",
    "management_actions",
    "shutdown_reasons",
    "values",
]

# Default trade-management rule limits (v1). No magic numbers elsewhere.
DEFAULT_TRAIL_PERCENT: Final[float] = 0.05
DEFAULT_INITIAL_RISK_PERCENT: Final[float] = 0.05

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

# Position state, pricing, risk context, and management-decision columns.
TRADE_MANAGEMENT_COLUMNS: Final[tuple[str, ...]] = (
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
)

CANONICAL_COLUMN_ORDER: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
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
        "entry_price": pl.Float64,
        "current_price": pl.Float64,
        "highest_price": pl.Float64,
        "lowest_price": pl.Float64,
        "unrealized_pnl": pl.Float64,
        "risk_state": pl.Utf8,
        "management_action": pl.Utf8,
        "action_reason": pl.Utf8,
        "stop_price": pl.Float64,
        "take_profit_price": pl.Float64,
        "trail_price": pl.Float64,
        "breakeven_price": pl.Float64,
        "allow_pyramid": pl.Boolean,
        "exit_quantity": pl.Float64,
        "model_name": pl.Utf8,
        "model_version": pl.Utf8,
        "optimizer": pl.Utf8,
        "policy": pl.Utf8,
    }
)

MERGED_TRADE_MANAGEMENT_SCHEMA: Final[pl.Schema] = pl.Schema(
    [(column, COLUMN_DTYPES[column]) for column in CANONICAL_COLUMN_ORDER]
)


# Prefer str+Enum over StrEnum so enum values embed directly into Polars.
class ManagementAction(str, Enum):  # noqa: UP042
    """Canonical trade-management actions recommended for an open position.

    Attributes:
        NONE: No stop or exit update is recommended.
        UPDATE_STOP: Update the protective stop price.
        PARTIAL_EXIT: Partial exit (reserved; not implemented in v1).
        FULL_EXIT: Full exit (reserved; not implemented in v1).
        ALLOW_PYRAMID: Allow pyramiding (reserved; not implemented in v1).
    """

    NONE = "NONE"
    UPDATE_STOP = "UPDATE_STOP"
    PARTIAL_EXIT = "PARTIAL_EXIT"
    FULL_EXIT = "FULL_EXIT"
    ALLOW_PYRAMID = "ALLOW_PYRAMID"


class ShutdownReason(str, Enum):  # noqa: UP042
    """Canonical reasons recorded on the ``action_reason`` column.

    Attributes:
        NONE: No management action reason.
        TRAILING_STOP: Trailing stop was hit or updated.
        BREAKEVEN: Break-even stop move was recommended.
        PARTIAL_PROFIT: Partial profit exit (reserved; not implemented in v1).
        ALPHA_DECAY: Alpha-decay exit (reserved; not implemented in v1).
        TIME_EXIT: Time-based exit (reserved; not implemented in v1).
        PORTFOLIO_RISK: Portfolio-risk driven action (reserved; not in v1).
    """

    NONE = "NONE"
    TRAILING_STOP = "TRAILING_STOP"
    BREAKEVEN = "BREAKEVEN"
    PARTIAL_PROFIT = "PARTIAL_PROFIT"
    ALPHA_DECAY = "ALPHA_DECAY"
    TIME_EXIT = "TIME_EXIT"
    PORTFOLIO_RISK = "PORTFOLIO_RISK"


def management_actions() -> tuple[ManagementAction, ...]:
    """Return an immutable copy of every ``ManagementAction`` member.

    Returns:
        All management-action members in declaration order.
    """
    return (
        ManagementAction.NONE,
        ManagementAction.UPDATE_STOP,
        ManagementAction.PARTIAL_EXIT,
        ManagementAction.FULL_EXIT,
        ManagementAction.ALLOW_PYRAMID,
    )


def shutdown_reasons() -> tuple[ShutdownReason, ...]:
    """Return an immutable copy of every ``ShutdownReason`` member.

    Returns:
        All action-reason members in declaration order.
    """
    return (
        ShutdownReason.NONE,
        ShutdownReason.TRAILING_STOP,
        ShutdownReason.BREAKEVEN,
        ShutdownReason.PARTIAL_PROFIT,
        ShutdownReason.ALPHA_DECAY,
        ShutdownReason.TIME_EXIT,
        ShutdownReason.PORTFOLIO_RISK,
    )


def values[EnumT: Enum](enum_cls: type[EnumT]) -> tuple[str, ...]:
    """Return an immutable copy of every string value for ``enum_cls``.

    Args:
        enum_cls: Enumeration class whose member values are requested.

    Returns:
        All member string values in declaration order.
    """
    return tuple(member.value for member in enum_cls)
