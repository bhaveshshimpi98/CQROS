"""CQROS factor combination metrics dataset schema.

Purpose:
    Define the canonical columnar contract for factor combination
    evaluation ledgers produced by the CQROS Factor Combination layer.

Responsibilities:
    - Declare the factor-combination primary key
    - Enumerate combination identity, metric, and decision columns
    - Expose required columns, canonical column order, and expected dtypes
    - Expose the factor combination status enumeration
    - Remain free of combination math, engine, and persistence logic

Dependencies:
    ``polars`` and the Python standard library.

Public API:
    ``PRIMARY_KEY_COLUMNS``, ``FACTOR_COMBINATION_COLUMNS``,
    ``REQUIRED_COLUMNS``, ``CANONICAL_COLUMN_ORDER``, ``COLUMN_DTYPES``,
    ``FACTOR_COMBINATION_SCHEMA``, ``FactorCombinationStatus``

Notes:
    This module describes column presence and dtypes only; it does not
    compute combination statistics, validate frames, or persist ledgers.
    ``status`` stores ``FactorCombinationStatus`` enum string values
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
    "FACTOR_COMBINATION_COLUMNS",
    "FACTOR_COMBINATION_SCHEMA",
    "PRIMARY_KEY_COLUMNS",
    "REQUIRED_COLUMNS",
    "FactorCombinationStatus",
]

PRIMARY_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "combination_id",
    "timeframe",
    "analysis_time",
)

# Combination identity, member factors, metrics, scores, and status.
FACTOR_COMBINATION_COLUMNS: Final[tuple[str, ...]] = (
    "combination_id",
    "factor_names",
    "factor_versions",
    "factor_categories",
    "timeframe",
    "combination_size",
    "combination_method",
    "analysis_time",
    "information_coefficient",
    "rank_information_coefficient",
    "ic_information_ratio",
    "quantile_spread",
    "hit_rate",
    "turnover",
    "correlation_penalty",
    "diversification_score",
    "stability_score",
    "confidence_score",
    "combination_score",
    "combination_rank",
    "status",
)

CANONICAL_COLUMN_ORDER: Final[tuple[str, ...]] = FACTOR_COMBINATION_COLUMNS

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
        "analysis_time": pl.Int64,
        "information_coefficient": pl.Float64,
        "rank_information_coefficient": pl.Float64,
        "ic_information_ratio": pl.Float64,
        "quantile_spread": pl.Float64,
        "hit_rate": pl.Float64,
        "turnover": pl.Float64,
        "correlation_penalty": pl.Float64,
        "diversification_score": pl.Float64,
        "stability_score": pl.Float64,
        "confidence_score": pl.Float64,
        "combination_score": pl.Float64,
        "combination_rank": pl.Int32,
        "status": pl.String,
    }
)

FACTOR_COMBINATION_SCHEMA: Final[pl.Schema] = pl.Schema(
    [(column, COLUMN_DTYPES[column]) for column in CANONICAL_COLUMN_ORDER]
)


# Prefer str+Enum over StrEnum so enum values embed directly into Polars.
class FactorCombinationStatus(str, Enum):  # noqa: UP042
    """Canonical evaluation status for a factor combination metrics row.

    Attributes:
        PASS: Combination evaluation metrics satisfy configured thresholds.
        FAIL: Combination evaluation metrics fail configured thresholds.
    """

    PASS = "PASS"
    FAIL = "FAIL"
