"""CQROS monitoring dataset schema.

Purpose:
    Define the canonical columnar contract for monitoring dataset ledgers
    produced by the CQROS Monitoring layer from reporting and related
    research artifacts.

Responsibilities:
    - Declare the merged-dataset primary key
    - Enumerate monitoring metadata columns
    - Expose required columns, canonical column order, and expected dtypes
    - Expose the monitoring status enumeration
    - Remain free of monitor evaluation, validation, and persistence logic

Dependencies:
    ``polars`` and the Python standard library.

Public API:
    ``PRIMARY_KEY_COLUMNS``, ``MONITORING_COLUMNS``, ``REQUIRED_COLUMNS``,
    ``CANONICAL_COLUMN_ORDER``, ``COLUMN_DTYPES``,
    ``MONITORING_SCHEMA``, ``MonitoringStatus``,
    ``monitoring_statuses``, ``monitoring_status_values``

Notes:
    This module describes column presence and dtypes only; it does not
    evaluate monitors, validate frames, or persist ledgers.
    ``manager`` preserves upstream order-manager lineage on every row.
    ``open_time`` uses ``Int64``, matching the reporting ledger timestamp
    convention.
"""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType
from typing import Final

import polars as pl

__all__ = [
    "CANONICAL_COLUMN_ORDER",
    "COLUMN_DTYPES",
    "MONITORING_COLUMNS",
    "MONITORING_SCHEMA",
    "PRIMARY_KEY_COLUMNS",
    "REQUIRED_COLUMNS",
    "MonitoringStatus",
    "monitoring_status_values",
    "monitoring_statuses",
]

PRIMARY_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
)

# Portfolio identity, monitor metadata, and status.
MONITORING_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
    "manager",
    "monitor_type",
    "monitor_name",
    "severity",
    "metric_name",
    "metric_value",
    "threshold",
    "alert",
    "status",
)

CANONICAL_COLUMN_ORDER: Final[tuple[str, ...]] = MONITORING_COLUMNS

REQUIRED_COLUMNS: Final[tuple[str, ...]] = CANONICAL_COLUMN_ORDER

COLUMN_DTYPES: Final = MappingProxyType(
    {
        "symbol": pl.Utf8,
        "timeframe": pl.Utf8,
        "open_time": pl.Int64,
        "manager": pl.Utf8,
        "monitor_type": pl.Utf8,
        "monitor_name": pl.Utf8,
        "severity": pl.Utf8,
        "metric_name": pl.Utf8,
        "metric_value": pl.Float64,
        "threshold": pl.Float64,
        "alert": pl.Boolean,
        "status": pl.Utf8,
    }
)

MONITORING_SCHEMA: Final[pl.Schema] = pl.Schema(
    [(column, COLUMN_DTYPES[column]) for column in CANONICAL_COLUMN_ORDER]
)


# Prefer str+Enum over StrEnum so enum values embed directly into Polars.
class MonitoringStatus(str, Enum):  # noqa: UP042
    """Canonical evaluation status for a monitoring dataset row.

    Attributes:
        NORMAL: Monitor metric is within acceptable bounds.
        WARNING: Monitor metric breached a warning threshold.
        CRITICAL: Monitor metric breached a critical threshold.
    """

    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


def monitoring_statuses() -> tuple[MonitoringStatus, ...]:
    """Return an immutable copy of every ``MonitoringStatus`` member.

    Returns:
        All monitoring-status members in declaration order.
    """
    return (
        MonitoringStatus.NORMAL,
        MonitoringStatus.WARNING,
        MonitoringStatus.CRITICAL,
    )


def monitoring_status_values() -> tuple[str, ...]:
    """Return an immutable copy of every ``MonitoringStatus`` string value.

    Returns:
        All monitoring-status string values in declaration order.
    """
    return tuple(member.value for member in monitoring_statuses())
