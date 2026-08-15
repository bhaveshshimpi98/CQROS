"""CQROS factor timeframe analysis metrics dataset schema.

Purpose:
    Define the canonical columnar contract for factor timeframe analysis
    ledgers produced by the CQROS Factor Timeframe Analysis layer.

Responsibilities:
    - Declare the factor-timeframe-analysis primary key
    - Enumerate factor metadata and timeframe decision columns
    - Expose required columns, canonical column order, and expected dtypes
    - Expose the timeframe analysis status enumeration
    - Remain free of analysis math, engine, and persistence logic

Dependencies:
    ``polars`` and the Python standard library.

Public API:
    ``PRIMARY_KEY_COLUMNS``, ``FACTOR_TIMEFRAME_ANALYSIS_COLUMNS``,
    ``REQUIRED_COLUMNS``, ``CANONICAL_COLUMN_ORDER``, ``COLUMN_DTYPES``,
    ``TIMEFRAME_ANALYSIS_SCHEMA``, ``TimeframeAnalysisStatus``,
    ``timeframe_analysis_statuses``, ``timeframe_analysis_status_values``

Notes:
    This module describes column presence and dtypes only; it does not
    compute timeframe rankings, validate frames, or persist ledgers.
    ``status`` stores ``TimeframeAnalysisStatus`` enum string values
    (``PASS``, ``FAIL``).
"""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType
from typing import Final

import polars as pl

__all__ = [
    "CANONICAL_COLUMN_ORDER",
    "COLUMN_DTYPES",
    "FACTOR_TIMEFRAME_ANALYSIS_COLUMNS",
    "PRIMARY_KEY_COLUMNS",
    "REQUIRED_COLUMNS",
    "TIMEFRAME_ANALYSIS_SCHEMA",
    "TimeframeAnalysisStatus",
    "timeframe_analysis_status_values",
    "timeframe_analysis_statuses",
]

PRIMARY_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "factor_name",
    "factor_version",
    "analysis_time",
)

# Factor identity, metadata, timeframe decision fields, selection flag,
# Factor Selection lineage, and status.
FACTOR_TIMEFRAME_ANALYSIS_COLUMNS: Final[tuple[str, ...]] = (
    "factor_name",
    "factor_version",
    "factor_category",
    "analysis_time",
    "best_timeframe",
    "best_selection_score",
    "timeframe_rank",
    "timeframe_stability",
    "winner_margin",
    "score_gap",
    "timeframe_confidence",
    "selected",
    "source_selection_version",
    "status",
)

CANONICAL_COLUMN_ORDER: Final[tuple[str, ...]] = FACTOR_TIMEFRAME_ANALYSIS_COLUMNS

REQUIRED_COLUMNS: Final[tuple[str, ...]] = CANONICAL_COLUMN_ORDER

COLUMN_DTYPES: Final = MappingProxyType(
    {
        "factor_name": pl.String,
        "factor_version": pl.String,
        "factor_category": pl.String,
        "analysis_time": pl.Int64,
        "best_timeframe": pl.String,
        "best_selection_score": pl.Float64,
        "timeframe_rank": pl.Int32,
        "timeframe_stability": pl.Float64,
        "winner_margin": pl.Float64,
        "score_gap": pl.Float64,
        "timeframe_confidence": pl.Float64,
        "selected": pl.Boolean,
        "source_selection_version": pl.String,
        "status": pl.String,
    }
)

TIMEFRAME_ANALYSIS_SCHEMA: Final[pl.Schema] = pl.Schema(
    [(column, COLUMN_DTYPES[column]) for column in CANONICAL_COLUMN_ORDER]
)


# Prefer str+Enum over StrEnum so enum values embed directly into Polars.
class TimeframeAnalysisStatus(str, Enum):  # noqa: UP042
    """Canonical evaluation status for a factor timeframe analysis row.

    Attributes:
        PASS: A winning timeframe and non-null best selection score exist.
        FAIL: Best timeframe or best selection score is missing.
    """

    PASS = "PASS"
    FAIL = "FAIL"


def timeframe_analysis_statuses() -> tuple[TimeframeAnalysisStatus, ...]:
    """Return an immutable copy of every ``TimeframeAnalysisStatus`` member.

    Returns:
        All timeframe-analysis-status members in declaration order.
    """
    return (TimeframeAnalysisStatus.PASS, TimeframeAnalysisStatus.FAIL)


def timeframe_analysis_status_values() -> tuple[str, ...]:
    """Return an immutable copy of every ``TimeframeAnalysisStatus`` string value.

    Returns:
        All timeframe-analysis-status string values in declaration order.
    """
    return tuple(member.value for member in timeframe_analysis_statuses())
