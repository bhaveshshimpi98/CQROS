"""CQROS alpha dataset schema.

Purpose:
    Define the canonical columnar contract for alpha prediction datasets
    produced by the CQROS Alpha layer.

Responsibilities:
    - Declare the alpha primary key
    - Enumerate alpha identity, prediction, and decision columns
    - Expose required columns, canonical column order, and expected dtypes
    - Expose the alpha status enumeration
    - Remain free of alpha math, engine, and persistence logic

Dependencies:
    ``polars`` and the Python standard library.

Public API:
    ``PRIMARY_KEY_COLUMNS``, ``ALPHA_COLUMNS``,
    ``REQUIRED_COLUMNS``, ``CANONICAL_COLUMN_ORDER``, ``COLUMN_DTYPES``,
    ``ALPHA_SCHEMA``, ``AlphaStatus``

Notes:
    This module describes column presence and dtypes only; it does not
    compute alpha scores, validate frames, or persist ledgers.
    ``status`` stores ``AlphaStatus`` enum string values
    (``PASS``, ``FAIL``).
"""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType
from typing import Final

import polars as pl

__all__ = [
    "ALPHA_COLUMNS",
    "ALPHA_SCHEMA",
    "CANONICAL_COLUMN_ORDER",
    "COLUMN_DTYPES",
    "PRIMARY_KEY_COLUMNS",
    "REQUIRED_COLUMNS",
    "AlphaStatus",
]

PRIMARY_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "factor_set_id",
    "alpha_model",
    "alpha_version",
    "symbol",
    "timeframe",
    "prediction_time",
)

# Alpha identity, prediction outputs, confidence fields, and status.
ALPHA_COLUMNS: Final[tuple[str, ...]] = (
    "factor_set_id",
    "alpha_model",
    "alpha_version",
    "symbol",
    "timeframe",
    "prediction_time",
    "expected_return",
    "alpha_score",
    "confidence",
    "uncertainty",
    "prediction_horizon",
    "status",
)

CANONICAL_COLUMN_ORDER: Final[tuple[str, ...]] = ALPHA_COLUMNS

REQUIRED_COLUMNS: Final[tuple[str, ...]] = CANONICAL_COLUMN_ORDER

COLUMN_DTYPES: Final = MappingProxyType(
    {
        "factor_set_id": pl.String,
        "alpha_model": pl.String,
        "alpha_version": pl.String,
        "symbol": pl.String,
        "timeframe": pl.String,
        "prediction_time": pl.Int64,
        "expected_return": pl.Float64,
        "alpha_score": pl.Float64,
        "confidence": pl.Float64,
        "uncertainty": pl.Float64,
        "prediction_horizon": pl.Int32,
        "status": pl.String,
    }
)

ALPHA_SCHEMA: Final[pl.Schema] = pl.Schema(
    [(column, COLUMN_DTYPES[column]) for column in CANONICAL_COLUMN_ORDER]
)


# Prefer str+Enum over StrEnum so enum values embed directly into Polars.
class AlphaStatus(str, Enum):  # noqa: UP042
    """Canonical evaluation status for an alpha prediction row.

    Attributes:
        PASS: Alpha prediction satisfies configured thresholds.
        FAIL: Alpha prediction fails configured thresholds.
    """

    PASS = "PASS"
    FAIL = "FAIL"
