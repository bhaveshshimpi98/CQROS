"""CQROS merged exit-engine recommendation dataset schema.

Purpose:
    Define the canonical columnar contract for exit recommendations produced
    by the CQROS Exit Engine from open position snapshots, accounting,
    portfolio risk, trade management, and pyramiding.

Responsibilities:
    - Declare the merged-dataset primary key
    - Enumerate exit recommendation columns
    - Expose required columns, canonical column order, and expected dtypes
    - Expose exit action and exit reason enumerations
    - Centralize default exit-rule constants
    - Remain free of exit math, validation, and persistence logic

Dependencies:
    ``polars`` and the Python standard library.

Public API:
    ``PRIMARY_KEY_COLUMNS``, ``EXIT_ENGINE_COLUMNS``, ``REQUIRED_COLUMNS``,
    ``CANONICAL_COLUMN_ORDER``, ``COLUMN_DTYPES``,
    ``MERGED_EXIT_ENGINE_SCHEMA``, ``ExitAction``, ``ExitReason``,
    ``exit_actions``, ``exit_reasons``, ``values``,
    ``DEFAULT_INITIAL_RISK_PERCENT``, ``DEFAULT_TAKE_PROFIT_MULTIPLE``,
    ``DEFAULT_PARTIAL_EXIT_PERCENT``, ``PRIORITY_HOLD``,
    ``PRIORITY_PORTFOLIO_SHUTDOWN``, ``PRIORITY_COOLDOWN``,
    ``PRIORITY_TRAILING_STOP``, ``PRIORITY_BREAK_EVEN``,
    ``PRIORITY_TAKE_PROFIT``, ``PRIORITY_ALPHA_DECAY``,
    ``PRIORITY_TIME_STOP``, ``PRIORITY_REGIME_EXIT``

Notes:
    This module describes column presence and dtypes only; it does not
    evaluate exit rules, validate frames, or persist recommendations.
    ``manager`` preserves upstream order-manager lineage on every row.
    Exit recommendations are advisory only and never execute orders.
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
    "DEFAULT_PARTIAL_EXIT_PERCENT",
    "DEFAULT_TAKE_PROFIT_MULTIPLE",
    "EXIT_ENGINE_COLUMNS",
    "MERGED_EXIT_ENGINE_SCHEMA",
    "PRIMARY_KEY_COLUMNS",
    "PRIORITY_ALPHA_DECAY",
    "PRIORITY_BREAK_EVEN",
    "PRIORITY_COOLDOWN",
    "PRIORITY_HOLD",
    "PRIORITY_PORTFOLIO_SHUTDOWN",
    "PRIORITY_REGIME_EXIT",
    "PRIORITY_TAKE_PROFIT",
    "PRIORITY_TIME_STOP",
    "PRIORITY_TRAILING_STOP",
    "REQUIRED_COLUMNS",
    "ExitAction",
    "ExitReason",
    "exit_actions",
    "exit_reasons",
    "values",
]

# Default exit-rule limits (v1). No magic numbers elsewhere.
DEFAULT_INITIAL_RISK_PERCENT: Final[float] = 0.05
DEFAULT_TAKE_PROFIT_MULTIPLE: Final[float] = 3.0
DEFAULT_PARTIAL_EXIT_PERCENT: Final[float] = 0.50

# Rule priority constants (lower number = higher urgency).
PRIORITY_HOLD: Final[int] = 0
PRIORITY_PORTFOLIO_SHUTDOWN: Final[int] = 1
PRIORITY_COOLDOWN: Final[int] = 2
PRIORITY_TRAILING_STOP: Final[int] = 3
PRIORITY_BREAK_EVEN: Final[int] = 4
PRIORITY_TAKE_PROFIT: Final[int] = 5
PRIORITY_ALPHA_DECAY: Final[int] = 6
PRIORITY_TIME_STOP: Final[int] = 7
PRIORITY_REGIME_EXIT: Final[int] = 8

PRIMARY_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
    "position_id",
)

# Position identity, pricing, state lineage, and exit-decision columns.
EXIT_ENGINE_COLUMNS: Final[tuple[str, ...]] = (
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
)

CANONICAL_COLUMN_ORDER: Final[tuple[str, ...]] = EXIT_ENGINE_COLUMNS

REQUIRED_COLUMNS: Final[tuple[str, ...]] = CANONICAL_COLUMN_ORDER

COLUMN_DTYPES: Final = MappingProxyType(
    {
        "symbol": pl.Utf8,
        "timeframe": pl.Utf8,
        "open_time": pl.Datetime("us", "UTC"),
        "position_id": pl.Utf8,
        "manager": pl.Utf8,
        "entry_price": pl.Float64,
        "current_price": pl.Float64,
        "quantity": pl.Float64,
        "risk_reward_ratio": pl.Float64,
        "risk_state": pl.Utf8,
        "trade_state": pl.Utf8,
        "pyramid_state": pl.Utf8,
        "exit_action": pl.Utf8,
        "exit_reason": pl.Utf8,
        "recommended_quantity": pl.Float64,
        "recommended_percent": pl.Float64,
        "priority": pl.Int64,
        "created_at": pl.Datetime("us", "UTC"),
    }
)

MERGED_EXIT_ENGINE_SCHEMA: Final[pl.Schema] = pl.Schema(
    [(column, COLUMN_DTYPES[column]) for column in CANONICAL_COLUMN_ORDER]
)


# Prefer str+Enum over StrEnum so enum values embed directly into Polars.
class ExitAction(str, Enum):  # noqa: UP042
    """Canonical exit actions recommended for an open position.

    Attributes:
        HOLD: No exit is recommended.
        PARTIAL_EXIT: Reduce the position by ``recommended_percent``.
        FULL_EXIT: Close the entire position.
    """

    HOLD = "HOLD"
    PARTIAL_EXIT = "PARTIAL_EXIT"
    FULL_EXIT = "FULL_EXIT"


class ExitReason(str, Enum):  # noqa: UP042
    """Canonical reasons recorded on the ``exit_reason`` column.

    Attributes:
        NONE: No exit reason (HOLD without a specific rule).
        TAKE_PROFIT: Risk/reward take-profit threshold reached.
        STOP_LOSS: Stop-loss exit (reserved; not implemented in v1).
        TRAILING_STOP: Trade management trailing-stop request.
        BREAK_EVEN: Trade management break-even exit request.
        ALPHA_DECAY: Trade management alpha-decay exit request.
        TIME_STOP: Time-based exit (stub; always HOLD in v1).
        PORTFOLIO_SHUTDOWN: Portfolio risk shutdown requires immediate exit.
        COOLDOWN: Portfolio-risk cooldown; exits are suppressed.
        REGIME_EXIT: Regime-driven exit (stub; always HOLD in v1).
        EMERGENCY_EXIT: Emergency exit (reserved; not implemented in v1).
    """

    NONE = "NONE"
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    TRAILING_STOP = "TRAILING_STOP"
    BREAK_EVEN = "BREAK_EVEN"
    ALPHA_DECAY = "ALPHA_DECAY"
    TIME_STOP = "TIME_STOP"
    PORTFOLIO_SHUTDOWN = "PORTFOLIO_SHUTDOWN"
    COOLDOWN = "COOLDOWN"
    REGIME_EXIT = "REGIME_EXIT"
    EMERGENCY_EXIT = "EMERGENCY_EXIT"


def exit_actions() -> tuple[ExitAction, ...]:
    """Return an immutable copy of every ``ExitAction`` member.

    Returns:
        All exit-action members in declaration order.
    """
    return (
        ExitAction.HOLD,
        ExitAction.PARTIAL_EXIT,
        ExitAction.FULL_EXIT,
    )


def exit_reasons() -> tuple[ExitReason, ...]:
    """Return an immutable copy of every ``ExitReason`` member.

    Returns:
        All exit-reason members in declaration order.
    """
    return (
        ExitReason.NONE,
        ExitReason.TAKE_PROFIT,
        ExitReason.STOP_LOSS,
        ExitReason.TRAILING_STOP,
        ExitReason.BREAK_EVEN,
        ExitReason.ALPHA_DECAY,
        ExitReason.TIME_STOP,
        ExitReason.PORTFOLIO_SHUTDOWN,
        ExitReason.COOLDOWN,
        ExitReason.REGIME_EXIT,
        ExitReason.EMERGENCY_EXIT,
    )


def values[EnumT: Enum](enum_cls: type[EnumT]) -> tuple[str, ...]:
    """Return an immutable copy of every string value for ``enum_cls``.

    Args:
        enum_cls: Enumeration class whose member values are requested.

    Returns:
        All member string values in declaration order.
    """
    return tuple(member.value for member in enum_cls)
