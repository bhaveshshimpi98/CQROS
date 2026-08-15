"""CQROS reporting dataset schema.

Purpose:
    Define the canonical columnar contract for reporting dataset ledgers
    produced by the CQROS Reporting layer from analytics and related
    research artifacts.

Responsibilities:
    - Declare the merged-dataset primary key
    - Enumerate reporting metadata columns
    - Expose required columns, canonical column order, and expected dtypes
    - Expose the reporting status enumeration
    - Remain free of report generation, validation, and persistence logic

Dependencies:
    ``polars`` and the Python standard library.

Public API:
    ``PRIMARY_KEY_COLUMNS``, ``REPORTING_COLUMNS``, ``REQUIRED_COLUMNS``,
    ``CANONICAL_COLUMN_ORDER``, ``COLUMN_DTYPES``,
    ``REPORTING_SCHEMA``, ``ReportingStatus``,
    ``reporting_statuses``, ``reporting_status_values``

Notes:
    This module describes column presence and dtypes only; it does not
    generate reports, validate frames, or persist ledgers.
    ``manager`` preserves upstream order-manager lineage on every row.
    ``open_time`` and ``generated_at`` use ``Int64``, matching the
    analytics ledger timestamp convention.
"""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType
from typing import Final

import polars as pl

__all__ = [
    "CANONICAL_COLUMN_ORDER",
    "COLUMN_DTYPES",
    "PRIMARY_KEY_COLUMNS",
    "REPORTING_COLUMNS",
    "REPORTING_SCHEMA",
    "REQUIRED_COLUMNS",
    "ReportingStatus",
    "reporting_status_values",
    "reporting_statuses",
]

PRIMARY_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
)

# Portfolio identity, report metadata, and status.
REPORTING_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
    "manager",
    "report_name",
    "report_type",
    "report_format",
    "report_version",
    "report_path",
    "generated_at",
    "status",
)

CANONICAL_COLUMN_ORDER: Final[tuple[str, ...]] = REPORTING_COLUMNS

REQUIRED_COLUMNS: Final[tuple[str, ...]] = CANONICAL_COLUMN_ORDER

COLUMN_DTYPES: Final = MappingProxyType(
    {
        "symbol": pl.Utf8,
        "timeframe": pl.Utf8,
        "open_time": pl.Int64,
        "manager": pl.Utf8,
        "report_name": pl.Utf8,
        "report_type": pl.Utf8,
        "report_format": pl.Utf8,
        "report_version": pl.Utf8,
        "report_path": pl.Utf8,
        "generated_at": pl.Int64,
        "status": pl.Utf8,
    }
)

REPORTING_SCHEMA: Final[pl.Schema] = pl.Schema(
    [(column, COLUMN_DTYPES[column]) for column in CANONICAL_COLUMN_ORDER]
)


# Prefer str+Enum over StrEnum so enum values embed directly into Polars.
class ReportingStatus(str, Enum):  # noqa: UP042
    """Canonical lifecycle status for a reporting dataset row.

    Attributes:
        GENERATED: Report artifact was produced successfully.
        FAILED: Report generation failed for this evaluation key.
    """

    GENERATED = "GENERATED"
    FAILED = "FAILED"


def reporting_statuses() -> tuple[ReportingStatus, ...]:
    """Return an immutable copy of every ``ReportingStatus`` member.

    Returns:
        All reporting-status members in declaration order.
    """
    return (ReportingStatus.GENERATED, ReportingStatus.FAILED)


def reporting_status_values() -> tuple[str, ...]:
    """Return an immutable copy of every ``ReportingStatus`` string value.

    Returns:
        All reporting-status string values in declaration order.
    """
    return tuple(member.value for member in reporting_statuses())
