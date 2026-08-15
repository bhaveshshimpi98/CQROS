"""CQROS merged portfolio risk decision dataset schema.

Purpose:
    Define the canonical columnar contract for portfolio risk decisions
    produced by the CQROS Portfolio Risk Manager from accounting snapshots.

Responsibilities:
    - Declare the merged-dataset primary key
    - Enumerate model metadata and portfolio-risk decision columns
    - Expose required columns, canonical column order, and expected dtypes
    - Expose portfolio risk state and shutdown reason enumerations
    - Centralize default portfolio-risk limit constants
    - Remain free of risk math, validation, and persistence logic

Dependencies:
    ``polars`` and the Python standard library.

Public API:
    ``PRIMARY_KEY_COLUMNS``, ``METADATA_COLUMNS``, ``PORTFOLIO_RISK_COLUMNS``,
    ``REQUIRED_COLUMNS``, ``CANONICAL_COLUMN_ORDER``, ``COLUMN_DTYPES``,
    ``MERGED_PORTFOLIO_RISK_SCHEMA``, ``PortfolioRiskState``,
    ``ShutdownReason``, ``portfolio_risk_states``, ``shutdown_reasons``,
    ``values``, ``DEFAULT_DAILY_LOSS_LIMIT``, ``DEFAULT_GROSS_EXPOSURE_LIMIT``,
    ``DEFAULT_COOLDOWN_HOURS``

Notes:
    This module describes column presence and dtypes only; it does not
    evaluate risk rules, validate frames, or persist decisions.
    ``manager``, ``optimizer``, and ``policy`` preserve upstream accounting /
    position lineage on every portfolio-risk row.
"""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType
from typing import Final

import polars as pl

__all__ = [
    "CANONICAL_COLUMN_ORDER",
    "COLUMN_DTYPES",
    "DEFAULT_COOLDOWN_HOURS",
    "DEFAULT_DAILY_LOSS_LIMIT",
    "DEFAULT_GROSS_EXPOSURE_LIMIT",
    "MERGED_PORTFOLIO_RISK_SCHEMA",
    "METADATA_COLUMNS",
    "PORTFOLIO_RISK_COLUMNS",
    "PRIMARY_KEY_COLUMNS",
    "REQUIRED_COLUMNS",
    "PortfolioRiskState",
    "ShutdownReason",
    "portfolio_risk_states",
    "shutdown_reasons",
    "values",
]

# Default portfolio-risk rule limits (v1). No magic numbers elsewhere.
DEFAULT_DAILY_LOSS_LIMIT: Final[float] = 0.02
DEFAULT_GROSS_EXPOSURE_LIMIT: Final[float] = 1.00
DEFAULT_COOLDOWN_HOURS: Final[int] = 24

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

# Portfolio exposure, daily PnL, and risk-decision columns.
PORTFOLIO_RISK_COLUMNS: Final[tuple[str, ...]] = (
    "manager",
    "position_id",
    "equity",
    "gross_exposure",
    "net_exposure",
    "daily_realized_pnl",
    "daily_unrealized_pnl",
    "daily_total_pnl",
    "daily_return_pct",
    "daily_drawdown_pct",
    "portfolio_risk_state",
    "allow_new_entries",
    "shutdown_reason",
    "cooldown_until",
)

CANONICAL_COLUMN_ORDER: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
    "manager",
    "position_id",
    "equity",
    "gross_exposure",
    "net_exposure",
    "daily_realized_pnl",
    "daily_unrealized_pnl",
    "daily_total_pnl",
    "daily_return_pct",
    "daily_drawdown_pct",
    "portfolio_risk_state",
    "allow_new_entries",
    "shutdown_reason",
    "cooldown_until",
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
        "equity": pl.Float64,
        "gross_exposure": pl.Float64,
        "net_exposure": pl.Float64,
        "daily_realized_pnl": pl.Float64,
        "daily_unrealized_pnl": pl.Float64,
        "daily_total_pnl": pl.Float64,
        "daily_return_pct": pl.Float64,
        "daily_drawdown_pct": pl.Float64,
        "portfolio_risk_state": pl.Utf8,
        "allow_new_entries": pl.Boolean,
        "shutdown_reason": pl.Utf8,
        "cooldown_until": pl.Datetime("us", "UTC"),
        "model_name": pl.Utf8,
        "model_version": pl.Utf8,
        "optimizer": pl.Utf8,
        "policy": pl.Utf8,
    }
)

MERGED_PORTFOLIO_RISK_SCHEMA: Final[pl.Schema] = pl.Schema(
    [(column, COLUMN_DTYPES[column]) for column in CANONICAL_COLUMN_ORDER]
)


# Prefer str+Enum over StrEnum so enum values embed directly into Polars.
class PortfolioRiskState(str, Enum):  # noqa: UP042
    """Canonical portfolio-level risk decision states.

    Attributes:
        NORMAL: No portfolio risk limits are breached.
        WARNING: Soft limit breached; new entries are blocked.
        SHUTDOWN: Hard limit or cooldown; new entries are blocked.
    """

    NORMAL = "NORMAL"
    WARNING = "WARNING"
    SHUTDOWN = "SHUTDOWN"


class ShutdownReason(str, Enum):  # noqa: UP042
    """Canonical reasons recorded when new entries are blocked.

    Attributes:
        NONE: No shutdown or warning reason (normal operation).
        DAILY_LOSS_LIMIT: Daily loss exceeded the configured fraction of equity.
        COOLDOWN: Prior daily-loss shutdown cooldown is still active.
        EXPOSURE_LIMIT: Gross exposure exceeded the configured equity fraction.
    """

    NONE = ""
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    COOLDOWN = "COOLDOWN"
    EXPOSURE_LIMIT = "EXPOSURE_LIMIT"


def portfolio_risk_states() -> tuple[PortfolioRiskState, ...]:
    """Return an immutable copy of every ``PortfolioRiskState`` member.

    Returns:
        All portfolio-risk-state members in declaration order.
    """
    return (
        PortfolioRiskState.NORMAL,
        PortfolioRiskState.WARNING,
        PortfolioRiskState.SHUTDOWN,
    )


def shutdown_reasons() -> tuple[ShutdownReason, ...]:
    """Return an immutable copy of every ``ShutdownReason`` member.

    Returns:
        All shutdown-reason members in declaration order.
    """
    return (
        ShutdownReason.NONE,
        ShutdownReason.DAILY_LOSS_LIMIT,
        ShutdownReason.COOLDOWN,
        ShutdownReason.EXPOSURE_LIMIT,
    )


def values[EnumT: Enum](enum_cls: type[EnumT]) -> tuple[str, ...]:
    """Return an immutable copy of every string value for ``enum_cls``.

    Args:
        enum_cls: Enumeration class whose member values are requested.

    Returns:
        All member string values in declaration order.
    """
    return tuple(member.value for member in enum_cls)
