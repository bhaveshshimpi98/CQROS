"""CQROS merged performance metrics dataset schema.

Purpose:
    Define the canonical columnar contract for performance metric ledgers
    produced by the CQROS Performance Engine from backtesting datasets.

Responsibilities:
    - Declare the merged-dataset primary key
    - Enumerate performance metric columns
    - Expose required columns, canonical column order, and expected dtypes
    - Expose the performance status enumeration
    - Remain free of performance math, validation, and persistence logic

Dependencies:
    ``polars`` and the Python standard library.

Public API:
    ``PRIMARY_KEY_COLUMNS``, ``PERFORMANCE_COLUMNS``, ``REQUIRED_COLUMNS``,
    ``CANONICAL_COLUMN_ORDER``, ``COLUMN_DTYPES``,
    ``PERFORMANCE_SCHEMA``, ``PerformanceStatus``,
    ``performance_statuses``, ``values``

Notes:
    This module describes column presence and dtypes only; it does not
    compute ratios, validate frames, or persist ledgers.
    ``manager`` preserves upstream order-manager lineage on every row.
    Risk-adjusted columns use explicit ``*_ratio`` names
    (``sharpe_ratio``, ``sortino_ratio``, ``calmar_ratio``).
"""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType
from typing import Final

import polars as pl

__all__ = [
    "CANONICAL_COLUMN_ORDER",
    "COLUMN_DTYPES",
    "PERFORMANCE_COLUMNS",
    "PERFORMANCE_SCHEMA",
    "PRIMARY_KEY_COLUMNS",
    "REQUIRED_COLUMNS",
    "PerformanceStatus",
    "performance_statuses",
    "values",
]

PRIMARY_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
)

# Portfolio identity, return/risk metrics, trade statistics, and status.
PERFORMANCE_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
    "manager",
    "total_return",
    "cagr",
    "volatility",
    "downside_volatility",
    "max_drawdown",
    "drawdown_duration",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "total_trades",
    "winning_trades",
    "losing_trades",
    "win_rate",
    "average_win",
    "average_loss",
    "profit_factor",
    "expectancy",
    "starting_equity",
    "ending_equity",
    "net_profit",
    "gross_profit",
    "gross_loss",
    "first_trade_time",
    "last_trade_time",
    "status",
)

CANONICAL_COLUMN_ORDER: Final[tuple[str, ...]] = PERFORMANCE_COLUMNS

REQUIRED_COLUMNS: Final[tuple[str, ...]] = CANONICAL_COLUMN_ORDER

COLUMN_DTYPES: Final = MappingProxyType(
    {
        "symbol": pl.Utf8,
        "timeframe": pl.Utf8,
        "open_time": pl.Datetime("us", "UTC"),
        "manager": pl.Utf8,
        "total_return": pl.Float64,
        "cagr": pl.Float64,
        "volatility": pl.Float64,
        "downside_volatility": pl.Float64,
        "max_drawdown": pl.Float64,
        "drawdown_duration": pl.Int64,
        "sharpe_ratio": pl.Float64,
        "sortino_ratio": pl.Float64,
        "calmar_ratio": pl.Float64,
        "total_trades": pl.Int64,
        "winning_trades": pl.Int64,
        "losing_trades": pl.Int64,
        "win_rate": pl.Float64,
        "average_win": pl.Float64,
        "average_loss": pl.Float64,
        "profit_factor": pl.Float64,
        "expectancy": pl.Float64,
        "starting_equity": pl.Float64,
        "ending_equity": pl.Float64,
        "net_profit": pl.Float64,
        "gross_profit": pl.Float64,
        "gross_loss": pl.Float64,
        "first_trade_time": pl.Datetime("us", "UTC"),
        "last_trade_time": pl.Datetime("us", "UTC"),
        "status": pl.Utf8,
    }
)

PERFORMANCE_SCHEMA: Final[pl.Schema] = pl.Schema(
    [(column, COLUMN_DTYPES[column]) for column in CANONICAL_COLUMN_ORDER]
)


# Prefer str+Enum over StrEnum so enum values embed directly into Polars.
class PerformanceStatus(str, Enum):  # noqa: UP042
    """Canonical lifecycle status for a performance metrics row.

    Attributes:
        ACTIVE: Intermediate evaluation timestamp in an ongoing ledger.
        FINISHED: Final evaluation timestamp of the reconstructed ledger.
    """

    ACTIVE = "ACTIVE"
    FINISHED = "FINISHED"


def performance_statuses() -> tuple[PerformanceStatus, ...]:
    """Return an immutable copy of every ``PerformanceStatus`` member.

    Returns:
        All performance-status members in declaration order.
    """
    return (PerformanceStatus.ACTIVE, PerformanceStatus.FINISHED)


def values[EnumT: Enum](enum_cls: type[EnumT]) -> tuple[str, ...]:
    """Return an immutable copy of every string value for ``enum_cls``.

    Args:
        enum_cls: Enumeration class whose member values are requested.

    Returns:
        All member string values in declaration order.
    """
    return tuple(member.value for member in enum_cls)
