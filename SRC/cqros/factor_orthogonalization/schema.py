"""CQROS factor orthogonalization metrics dataset schema.

Purpose:
    Define the canonical columnar contract for combination-unit factor
    orthogonalization decision ledgers produced by the CQROS Factor
    Orthogonalization layer.

Responsibilities:
    - Declare the factor-orthogonalization primary key
    - Enumerate combination identity, redundancy, lineage, and decision columns
    - Expose required columns, canonical column order, and expected dtypes
    - Expose the factor orthogonalization status enumeration
    - Remain free of orthogonalization math, engine, and persistence logic

Dependencies:
    ``polars`` and the Python standard library.

Public API:
    ``PRIMARY_KEY_COLUMNS``, ``FACTOR_ORTHOGONALIZATION_COLUMNS``,
    ``REQUIRED_COLUMNS``, ``CANONICAL_COLUMN_ORDER``, ``COLUMN_DTYPES``,
    ``FACTOR_ORTHOGONALIZATION_SCHEMA``, ``FactorOrthogonalizationStatus``

Notes:
    This module describes column presence and dtypes only; it does not
    compute orthogonalization scores, validate frames, or persist ledgers.
    ``status`` stores ``FactorOrthogonalizationStatus`` enum string values
    (``PASS``, ``FAIL``). Orthogonalization operates on Factor Combination
    rows (combination-unit), not individual factors.
"""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType
from typing import Final

import polars as pl

__all__ = [
    "CANONICAL_COLUMN_ORDER",
    "COLUMN_DTYPES",
    "FACTOR_ORTHOGONALIZATION_COLUMNS",
    "FACTOR_ORTHOGONALIZATION_SCHEMA",
    "PRIMARY_KEY_COLUMNS",
    "REQUIRED_COLUMNS",
    "FactorOrthogonalizationStatus",
]

PRIMARY_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "combination_id",
    "timeframe",
    "analysis_time",
)

# Combination identity, redundancy metrics, lineage, decision fields, status.
FACTOR_ORTHOGONALIZATION_COLUMNS: Final[tuple[str, ...]] = (
    "combination_id",
    "factor_names",
    "factor_versions",
    "factor_categories",
    "timeframe",
    "combination_size",
    "combination_method",
    "orthogonalization_method",
    "orthogonalization_version",
    "analysis_time",
    "source_combination_rank",
    "source_combination_score",
    "source_stability_score",
    "source_confidence_score",
    "correlation_score",
    "vif_score",
    "redundancy_score",
    "orthogonality_score",
    "information_retained",
    "correlation_overlap",
    "correlation_threshold",
    "min_overlap_threshold",
    "redundancy_checked",
    "redundancy_rejected",
    "redundancy_reference_combination_id",
    "selected",
    "orthogonalization_rank",
    "orthogonalization_reason",
    "source_combination_version",
    "source_fta_version",
    "source_selection_version",
    "dataset_version",
    "validation_start_time",
    "validation_end_time",
    "status",
)

CANONICAL_COLUMN_ORDER: Final[tuple[str, ...]] = FACTOR_ORTHOGONALIZATION_COLUMNS

REQUIRED_COLUMNS: Final[tuple[str, ...]] = CANONICAL_COLUMN_ORDER

COLUMN_DTYPES: Final = MappingProxyType(
    {
        "combination_id": pl.String,
        "factor_names": pl.List(pl.String),
        "factor_versions": pl.List(pl.String),
        "factor_categories": pl.List(pl.String),
        "timeframe": pl.String,
        "combination_size": pl.Int32,
        "combination_method": pl.String,
        "orthogonalization_method": pl.String,
        "orthogonalization_version": pl.String,
        "analysis_time": pl.Int64,
        "source_combination_rank": pl.Int32,
        "source_combination_score": pl.Float64,
        "source_stability_score": pl.Float64,
        "source_confidence_score": pl.Float64,
        "correlation_score": pl.Float64,
        "vif_score": pl.Float64,
        "redundancy_score": pl.Float64,
        "orthogonality_score": pl.Float64,
        "information_retained": pl.Float64,
        "correlation_overlap": pl.Int64,
        "correlation_threshold": pl.Float64,
        "min_overlap_threshold": pl.Int64,
        "redundancy_checked": pl.Boolean,
        "redundancy_rejected": pl.Boolean,
        "redundancy_reference_combination_id": pl.String,
        "selected": pl.Boolean,
        "orthogonalization_rank": pl.Int32,
        "orthogonalization_reason": pl.String,
        "source_combination_version": pl.String,
        "source_fta_version": pl.String,
        "source_selection_version": pl.String,
        "dataset_version": pl.String,
        "validation_start_time": pl.Int64,
        "validation_end_time": pl.Int64,
        "status": pl.String,
    }
)

FACTOR_ORTHOGONALIZATION_SCHEMA: Final[pl.Schema] = pl.Schema(
    [(column, COLUMN_DTYPES[column]) for column in CANONICAL_COLUMN_ORDER]
)


# Prefer str+Enum over StrEnum so enum values embed directly into Polars.
class FactorOrthogonalizationStatus(str, Enum):  # noqa: UP042
    """Canonical evaluation status for a factor orthogonalization metrics row.

    Attributes:
        PASS: Combination survives orthogonalization redundancy filtering.
        FAIL: Combination fails orthogonalization redundancy filtering.
    """

    PASS = "PASS"
    FAIL = "FAIL"
