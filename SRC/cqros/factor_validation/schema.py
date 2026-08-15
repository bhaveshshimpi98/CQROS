"""CQROS factor validation metrics dataset schema.

Purpose:
    Define the canonical columnar contract for factor validation metric
    ledgers produced by the CQROS Factor Validation layer.

Responsibilities:
    - Declare the factor-validation primary key
    - Enumerate factor metadata and validation metric columns
    - Expose required columns, canonical column order, and expected dtypes
    - Expose the factor validation status enumeration
    - Remain free of validation math, engine, and persistence logic

Dependencies:
    ``polars`` and the Python standard library.

Public API:
    ``PRIMARY_KEY_COLUMNS``, ``FACTOR_VALIDATION_COLUMNS``,
    ``REQUIRED_COLUMNS``, ``CANONICAL_COLUMN_ORDER``, ``COLUMN_DTYPES``,
    ``FACTOR_VALIDATION_SCHEMA``, ``FactorValidationStatus``,
    ``factor_validation_statuses``, ``factor_validation_status_values``

Notes:
    This module describes column presence and dtypes only; it does not
    compute validation statistics, validate frames, or persist ledgers.
    ``status`` stores ``FactorValidationStatus`` enum string values
    (``PASS``, ``FAIL``, ``SKIPPED``).

    ``observations`` counts valid factor–label observation pairs.
    ``ic_observations`` counts valid cross-sectional IC timestamps used
    for IC-series statistics (ICIR, t-stat, and related measures).

    ``validation_start_time`` / ``validation_end_time`` define the evaluated
    sample window. ``validation_time`` remains part of the primary key as the
    existing ledger identity endpoint and is not redefined here.

    Primary-key expansion to include dataset/label/window identity is deferred
    until repository and lineage review. The current primary key remains
    ``(factor_name, factor_version, timeframe, validation_time)``.
"""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType
from typing import Final

import polars as pl

__all__ = [
    "CANONICAL_COLUMN_ORDER",
    "COLUMN_DTYPES",
    "FACTOR_VALIDATION_COLUMNS",
    "FACTOR_VALIDATION_SCHEMA",
    "PRIMARY_KEY_COLUMNS",
    "REQUIRED_COLUMNS",
    "FactorValidationStatus",
    "factor_validation_status_values",
    "factor_validation_statuses",
]

PRIMARY_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "factor_name",
    "factor_version",
    "timeframe",
    "validation_time",
)

# Factor identity, lineage metadata, validation metrics, and status.
FACTOR_VALIDATION_COLUMNS: Final[tuple[str, ...]] = (
    "factor_name",
    "factor_version",
    "timeframe",
    "validation_time",
    "factor_category",
    "dataset_version",
    "label_version",
    "validation_start_time",
    "validation_end_time",
    "information_coefficient",
    "rank_information_coefficient",
    "ic_information_ratio",
    "ic_std",
    "ic_p_value",
    "ic_t_stat",
    "ic_decay",
    "turnover",
    "monotonicity_score",
    "quantile_spread",
    "observations",
    "ic_observations",
    "status",
)

CANONICAL_COLUMN_ORDER: Final[tuple[str, ...]] = FACTOR_VALIDATION_COLUMNS

REQUIRED_COLUMNS: Final[tuple[str, ...]] = CANONICAL_COLUMN_ORDER

COLUMN_DTYPES: Final = MappingProxyType(
    {
        "factor_name": pl.String,
        "factor_version": pl.String,
        "timeframe": pl.String,
        "validation_time": pl.Int64,
        "factor_category": pl.String,
        "dataset_version": pl.String,
        "label_version": pl.String,
        "validation_start_time": pl.Int64,
        "validation_end_time": pl.Int64,
        "information_coefficient": pl.Float64,
        "rank_information_coefficient": pl.Float64,
        "ic_information_ratio": pl.Float64,
        "ic_std": pl.Float64,
        "ic_p_value": pl.Float64,
        "ic_t_stat": pl.Float64,
        "ic_decay": pl.Float64,
        "turnover": pl.Float64,
        "monotonicity_score": pl.Float64,
        "quantile_spread": pl.Float64,
        "observations": pl.Int64,
        "ic_observations": pl.Int64,
        "status": pl.String,
    }
)

FACTOR_VALIDATION_SCHEMA: Final[pl.Schema] = pl.Schema(
    [(column, COLUMN_DTYPES[column]) for column in CANONICAL_COLUMN_ORDER]
)


# Prefer str+Enum over StrEnum so enum values embed directly into Polars.
class FactorValidationStatus(str, Enum):  # noqa: UP042
    """Canonical evaluation status for a factor validation metrics row.

    Attributes:
        PASS: Validation completed and the row passed governance evaluation.
        FAIL: Validation completed and the row failed governance evaluation.
        SKIPPED: Validation was not evaluated for the row (for example
            insufficient data). Distinct from ``FAIL`` so non-evaluable
            rows are not conflated with policy failures.
    """

    PASS = "PASS"
    FAIL = "FAIL"
    SKIPPED = "SKIPPED"


def factor_validation_statuses() -> tuple[FactorValidationStatus, ...]:
    """Return an immutable copy of every ``FactorValidationStatus`` member.

    Returns:
        All factor-validation-status members in declaration order.
    """
    return (
        FactorValidationStatus.PASS,
        FactorValidationStatus.FAIL,
        FactorValidationStatus.SKIPPED,
    )


def factor_validation_status_values() -> tuple[str, ...]:
    """Return an immutable copy of every ``FactorValidationStatus`` string value.

    Returns:
        All factor-validation-status string values in declaration order.
    """
    return tuple(member.value for member in factor_validation_statuses())
