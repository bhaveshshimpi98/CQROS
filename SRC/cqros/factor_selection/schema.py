"""CQROS factor selection metrics dataset schema.

Purpose:
    Define the canonical columnar contract for factor selection decision
    ledgers produced by the CQROS Factor Selection layer.

Responsibilities:
    - Declare the factor-selection primary key
    - Enumerate factor metadata and selection decision columns
    - Expose required columns, canonical column order, and expected dtypes
    - Expose the factor selection status enumeration
    - Remain free of selection math, engine, and persistence logic

Dependencies:
    ``polars`` and the Python standard library.

Public API:
    ``PRIMARY_KEY_COLUMNS``, ``FACTOR_SELECTION_COLUMNS``,
    ``REQUIRED_COLUMNS``, ``CANONICAL_COLUMN_ORDER``, ``COLUMN_DTYPES``,
    ``FACTOR_SELECTION_SCHEMA``, ``FactorSelectionStatus``,
    ``factor_selection_statuses``, ``factor_selection_status_values``,
    ``ELIGIBILITY_COLUMNS``, ``ELIGIBILITY_COLUMN_DTYPES``

Notes:
    This module describes column presence and dtypes only; it does not
    compute selection scores, validate frames, or persist ledgers.
    ``status`` stores ``FactorSelectionStatus`` enum string values
    (``SELECTED``, ``REJECTED``).

    Eligibility metadata columns are appended after the canonical selection
    columns. They are written by ``SimpleFactorSelectionEngine`` when a
    ``FactorEligibilityPolicy`` is attached, and are not part of the legacy
    schema. Downstream code that requires eligibility metadata must call
    ``require_eligibility_metadata`` from ``cqros.factor_selection.eligibility``
    rather than silently defaulting to ``ELIGIBLE``.
"""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType
from typing import Final

import polars as pl

__all__ = [
    "CANONICAL_COLUMN_ORDER",
    "COLUMN_DTYPES",
    "ELIGIBILITY_COLUMN_DTYPES",
    "ELIGIBILITY_COLUMNS",
    "FACTOR_SELECTION_COLUMNS",
    "FACTOR_SELECTION_SCHEMA",
    "PRIMARY_KEY_COLUMNS",
    "REQUIRED_COLUMNS",
    "FactorSelectionStatus",
    "factor_selection_status_values",
    "factor_selection_statuses",
]

PRIMARY_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "factor_name",
    "factor_version",
    "timeframe",
    "selection_time",
)

# Factor identity, metadata, selection decision fields, orientation, and status.
FACTOR_SELECTION_COLUMNS: Final[tuple[str, ...]] = (
    "factor_name",
    "factor_version",
    "timeframe",
    "selection_time",
    "factor_category",
    "selected",
    "selection_score",
    "selection_rank",
    "selection_reason",
    "selection_ic",
    "selected_direction",
    "orientation_policy",
    "status",
)

CANONICAL_COLUMN_ORDER: Final[tuple[str, ...]] = FACTOR_SELECTION_COLUMNS

REQUIRED_COLUMNS: Final[tuple[str, ...]] = CANONICAL_COLUMN_ORDER

COLUMN_DTYPES: Final = MappingProxyType(
    {
        "factor_name": pl.String,
        "factor_version": pl.String,
        "timeframe": pl.String,
        "selection_time": pl.Int64,
        "factor_category": pl.String,
        "selected": pl.Boolean,
        "selection_score": pl.Float64,
        "selection_rank": pl.Int32,
        "selection_reason": pl.String,
        "selection_ic": pl.Float64,
        "selected_direction": pl.Int8,
        "orientation_policy": pl.String,
        "status": pl.String,
    }
)

FACTOR_SELECTION_SCHEMA: Final[pl.Schema] = pl.Schema(
    [(column, COLUMN_DTYPES[column]) for column in CANONICAL_COLUMN_ORDER]
)

# Eligibility metadata columns appended by FactorEligibilityPolicy.
# These extend the canonical schema; legacy artifacts will not carry them.
ELIGIBILITY_COLUMNS: Final[tuple[str, ...]] = (
    "eligibility_status",
    "eligibility_reason",
    "eligibility_policy",
    "usable_observations",
    "total_observations",
    "coverage_ratio",
    "null_rate",
    "required_lookback",
    "available_history",
    "warmup_sufficient",
    "companion_dependencies",
    "companion_coverage_status",
)

ELIGIBILITY_COLUMN_DTYPES: Final = MappingProxyType(
    {
        "eligibility_status": pl.String,
        "eligibility_reason": pl.String,
        "eligibility_policy": pl.String,
        "usable_observations": pl.Int64,
        "total_observations": pl.Int64,
        "coverage_ratio": pl.Float64,
        "null_rate": pl.Float64,
        "required_lookback": pl.Int64,
        "available_history": pl.Int64,
        "warmup_sufficient": pl.Boolean,
        "companion_dependencies": pl.String,
        "companion_coverage_status": pl.String,
    }
)


# Prefer str+Enum over StrEnum so enum values embed directly into Polars.
class FactorSelectionStatus(str, Enum):  # noqa: UP042
    """Canonical evaluation status for a factor selection metrics row.

    Attributes:
        SELECTED: Factor passed selection criteria and is retained.
        REJECTED: Factor failed selection criteria and is excluded.
    """

    SELECTED = "SELECTED"
    REJECTED = "REJECTED"


def factor_selection_statuses() -> tuple[FactorSelectionStatus, ...]:
    """Return an immutable copy of every ``FactorSelectionStatus`` member.

    Returns:
        All factor-selection-status members in declaration order.
    """
    return (FactorSelectionStatus.SELECTED, FactorSelectionStatus.REJECTED)


def factor_selection_status_values() -> tuple[str, ...]:
    """Return an immutable copy of every ``FactorSelectionStatus`` string value.

    Returns:
        All factor-selection-status string values in declaration order.
    """
    return tuple(member.value for member in factor_selection_statuses())
