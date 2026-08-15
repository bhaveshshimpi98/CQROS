"""CQROS regime dataset schema.

Purpose:
    Define the canonical columnar contract for regime classification
    datasets produced by the CQROS Regime layer.

Responsibilities:
    - Declare the regime primary key
    - Enumerate regime identity, classification, and decision columns
    - Expose required columns, canonical column order, and expected dtypes
    - Expose the regime status enumeration
    - Remain free of regime math, engine, and persistence logic

Dependencies:
    ``polars`` and the Python standard library.

Public API:
    ``PRIMARY_KEY_COLUMNS``, ``REGIME_COLUMNS``,
    ``REQUIRED_COLUMNS``, ``CANONICAL_COLUMN_ORDER``, ``COLUMN_DTYPES``,
    ``REGIME_SCHEMA``, ``RegimeStatus``

Notes:
    This module describes column presence and dtypes only; it does not
    compute regime scores, validate frames, or persist ledgers.
    ``status`` stores ``RegimeStatus`` enum string values
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
    "PRIMARY_KEY_COLUMNS",
    "REGIME_COLUMNS",
    "REGIME_SCHEMA",
    "REQUIRED_COLUMNS",
    "RegimeStatus",
]

PRIMARY_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "regime_id",
    "symbol",
    "timeframe",
    "regime_time",
)

# Regime identity, classification outputs, scores, and status.
REGIME_COLUMNS: Final[tuple[str, ...]] = (
    "regime_id",
    "factor_set_id",
    "alpha_id",
    "symbol",
    "timeframe",
    "regime_time",
    "regime_type",
    "regime_probability",
    "regime_score",
    "regime_version",
    "status",
    "metadata",
)

CANONICAL_COLUMN_ORDER: Final[tuple[str, ...]] = REGIME_COLUMNS

REQUIRED_COLUMNS: Final[tuple[str, ...]] = CANONICAL_COLUMN_ORDER

COLUMN_DTYPES: Final = MappingProxyType(
    {
        "regime_id": pl.String,
        "factor_set_id": pl.String,
        "alpha_id": pl.String,
        "symbol": pl.String,
        "timeframe": pl.String,
        "regime_time": pl.Datetime("ms"),
        "regime_type": pl.String,
        "regime_probability": pl.Float64,
        "regime_score": pl.Float64,
        "regime_version": pl.String,
        "status": pl.String,
        "metadata": pl.List(pl.String),
    }
)

REGIME_SCHEMA: Final[pl.Schema] = pl.Schema(
    [(column, COLUMN_DTYPES[column]) for column in CANONICAL_COLUMN_ORDER]
)


# Prefer str+Enum over StrEnum so enum values embed directly into Polars.
class RegimeStatus(str, Enum):  # noqa: UP042
    """Canonical evaluation status for a regime classification row.

    Attributes:
        PASS: Regime classification satisfies configured thresholds.
        FAIL: Regime classification fails configured thresholds.
    """

    PASS = "PASS"
    FAIL = "FAIL"
