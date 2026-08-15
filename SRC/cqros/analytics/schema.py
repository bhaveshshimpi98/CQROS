"""CQROS merged analytics metrics dataset schema.

Purpose:
    Define the canonical columnar contract for analytics metric ledgers
    produced by the CQROS Analytics Engine from performance datasets.

Responsibilities:
    - Declare the merged-dataset primary key
    - Enumerate analytics metric columns
    - Expose required columns, canonical column order, and expected dtypes
    - Expose the analytics status enumeration
    - Remain free of analytics math, validation, and persistence logic

Dependencies:
    ``polars`` and the Python standard library.

Public API:
    ``PRIMARY_KEY_COLUMNS``, ``ANALYTICS_COLUMNS``, ``REQUIRED_COLUMNS``,
    ``CANONICAL_COLUMN_ORDER``, ``COLUMN_DTYPES``,
    ``ANALYTICS_SCHEMA``, ``AnalyticsStatus``,
    ``analytics_statuses``, ``analytics_status_values``

Notes:
    This module describes column presence and dtypes only; it does not
    compute rolling metrics, validate frames, or persist ledgers.
    ``manager`` preserves upstream order-manager lineage on every row.
    ``open_time`` uses ``Int64``, unlike the datetime-typed performance
    ledger primary key.
"""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType
from typing import Final

import polars as pl

__all__ = [
    "ANALYTICS_COLUMNS",
    "ANALYTICS_SCHEMA",
    "CANONICAL_COLUMN_ORDER",
    "COLUMN_DTYPES",
    "PRIMARY_KEY_COLUMNS",
    "REQUIRED_COLUMNS",
    "AnalyticsStatus",
    "analytics_status_values",
    "analytics_statuses",
]

PRIMARY_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
)

# Portfolio identity, rolling metrics, benchmark analytics, and status.
ANALYTICS_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
    "manager",
    "rolling_return",
    "rolling_volatility",
    "rolling_sharpe",
    "rolling_sortino",
    "rolling_max_drawdown",
    "rolling_win_rate",
    "rolling_profit_factor",
    "rolling_expectancy",
    "rolling_cagr",
    "rolling_calmar",
    "rolling_recovery_factor",
    "benchmark_return",
    "benchmark_alpha",
    "benchmark_beta",
    "benchmark_correlation",
    "benchmark_tracking_error",
    "benchmark_information_ratio",
    "status",
)

CANONICAL_COLUMN_ORDER: Final[tuple[str, ...]] = ANALYTICS_COLUMNS

REQUIRED_COLUMNS: Final[tuple[str, ...]] = CANONICAL_COLUMN_ORDER

COLUMN_DTYPES: Final = MappingProxyType(
    {
        "symbol": pl.Utf8,
        "timeframe": pl.Utf8,
        "open_time": pl.Int64,
        "manager": pl.Utf8,
        "rolling_return": pl.Float64,
        "rolling_volatility": pl.Float64,
        "rolling_sharpe": pl.Float64,
        "rolling_sortino": pl.Float64,
        "rolling_max_drawdown": pl.Float64,
        "rolling_win_rate": pl.Float64,
        "rolling_profit_factor": pl.Float64,
        "rolling_expectancy": pl.Float64,
        "rolling_cagr": pl.Float64,
        "rolling_calmar": pl.Float64,
        "rolling_recovery_factor": pl.Float64,
        "benchmark_return": pl.Float64,
        "benchmark_alpha": pl.Float64,
        "benchmark_beta": pl.Float64,
        "benchmark_correlation": pl.Float64,
        "benchmark_tracking_error": pl.Float64,
        "benchmark_information_ratio": pl.Float64,
        "status": pl.Utf8,
    }
)

ANALYTICS_SCHEMA: Final[pl.Schema] = pl.Schema(
    [(column, COLUMN_DTYPES[column]) for column in CANONICAL_COLUMN_ORDER]
)


# Prefer str+Enum over StrEnum so enum values embed directly into Polars.
class AnalyticsStatus(str, Enum):  # noqa: UP042
    """Canonical lifecycle status for an analytics metrics row.

    Attributes:
        ACTIVE: Intermediate evaluation timestamp in an ongoing ledger.
        FINISHED: Final evaluation timestamp of the reconstructed ledger.
    """

    ACTIVE = "ACTIVE"
    FINISHED = "FINISHED"


def analytics_statuses() -> tuple[AnalyticsStatus, ...]:
    """Return an immutable copy of every ``AnalyticsStatus`` member.

    Returns:
        All analytics-status members in declaration order.
    """
    return (AnalyticsStatus.ACTIVE, AnalyticsStatus.FINISHED)


def analytics_status_values() -> tuple[str, ...]:
    """Return an immutable copy of every ``AnalyticsStatus`` string value.

    Returns:
        All analytics-status string values in declaration order.
    """
    return tuple(member.value for member in analytics_statuses())
