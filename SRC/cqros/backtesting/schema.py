"""CQROS merged backtesting performance dataset schema.

Purpose:
    Define the canonical columnar contract for historical performance
    ledgers produced by the CQROS Backtesting Engine from accounting,
    position, and exit-engine datasets.

Responsibilities:
    - Declare the merged-dataset primary key
    - Enumerate performance ledger columns
    - Expose required columns, canonical column order, and expected dtypes
    - Expose the backtesting status enumeration
    - Remain free of performance math, validation, and persistence logic

Dependencies:
    ``polars`` and the Python standard library.

Public API:
    ``PRIMARY_KEY_COLUMNS``, ``BACKTESTING_COLUMNS``, ``REQUIRED_COLUMNS``,
    ``CANONICAL_COLUMN_ORDER``, ``COLUMN_DTYPES``,
    ``MERGED_BACKTESTING_SCHEMA``, ``BacktestingStatus``,
    ``backtesting_statuses``, ``values``

Notes:
    This module describes column presence and dtypes only; it does not
    compute equity curves, validate frames, or persist ledgers.
    ``manager`` preserves upstream order-manager lineage on every row.
    Backtesting Engine v1 reconstructs performance only and never trades
    or executes orders. ``sharpe_stub`` and ``sortino_stub`` are reserved
    null columns until risk-adjusted ratios are implemented.
"""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType
from typing import Final

import polars as pl

__all__ = [
    "BACKTESTING_COLUMNS",
    "CANONICAL_COLUMN_ORDER",
    "COLUMN_DTYPES",
    "MERGED_BACKTESTING_SCHEMA",
    "PRIMARY_KEY_COLUMNS",
    "REQUIRED_COLUMNS",
    "BacktestingStatus",
    "backtesting_statuses",
    "values",
]

PRIMARY_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
)

# Portfolio identity, equity curve, trade statistics, and status columns.
BACKTESTING_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
    "manager",
    "equity",
    "cash",
    "position_value",
    "realized_pnl",
    "unrealized_pnl",
    "total_pnl",
    "drawdown",
    "peak_equity",
    "daily_return",
    "cumulative_return",
    "trade_count",
    "winning_trades",
    "losing_trades",
    "win_rate",
    "profit_factor",
    "sharpe_stub",
    "sortino_stub",
    "max_drawdown",
    "status",
)

CANONICAL_COLUMN_ORDER: Final[tuple[str, ...]] = BACKTESTING_COLUMNS

REQUIRED_COLUMNS: Final[tuple[str, ...]] = CANONICAL_COLUMN_ORDER

COLUMN_DTYPES: Final = MappingProxyType(
    {
        "symbol": pl.Utf8,
        "timeframe": pl.Utf8,
        "open_time": pl.Datetime("us", "UTC"),
        "manager": pl.Utf8,
        "equity": pl.Float64,
        "cash": pl.Float64,
        "position_value": pl.Float64,
        "realized_pnl": pl.Float64,
        "unrealized_pnl": pl.Float64,
        "total_pnl": pl.Float64,
        "drawdown": pl.Float64,
        "peak_equity": pl.Float64,
        "daily_return": pl.Float64,
        "cumulative_return": pl.Float64,
        "trade_count": pl.Int64,
        "winning_trades": pl.Int64,
        "losing_trades": pl.Int64,
        "win_rate": pl.Float64,
        "profit_factor": pl.Float64,
        "sharpe_stub": pl.Float64,
        "sortino_stub": pl.Float64,
        "max_drawdown": pl.Float64,
        "status": pl.Utf8,
    }
)

MERGED_BACKTESTING_SCHEMA: Final[pl.Schema] = pl.Schema(
    [(column, COLUMN_DTYPES[column]) for column in CANONICAL_COLUMN_ORDER]
)


# Prefer str+Enum over StrEnum so enum values embed directly into Polars.
class BacktestingStatus(str, Enum):  # noqa: UP042
    """Canonical lifecycle status for a backtesting performance row.

    Attributes:
        ACTIVE: Intermediate evaluation timestamp in an ongoing ledger.
        FINISHED: Final evaluation timestamp of the reconstructed ledger.
    """

    ACTIVE = "ACTIVE"
    FINISHED = "FINISHED"


def backtesting_statuses() -> tuple[BacktestingStatus, ...]:
    """Return an immutable copy of every ``BacktestingStatus`` member.

    Returns:
        All backtesting-status members in declaration order.
    """
    return (BacktestingStatus.ACTIVE, BacktestingStatus.FINISHED)


def values[EnumT: Enum](enum_cls: type[EnumT]) -> tuple[str, ...]:
    """Return an immutable copy of every string value for ``enum_cls``.

    Args:
        enum_cls: Enumeration class whose member values are requested.

    Returns:
        All member string values in declaration order.
    """
    return tuple(member.value for member in enum_cls)
