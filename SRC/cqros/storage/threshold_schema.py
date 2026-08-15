"""CQROS regression threshold dataset schema.

Purpose:
    Define the canonical columnar contract for production-approved regression
    signal thresholds persisted independently of research calibration.

Responsibilities:
    - Declare the threshold-dataset primary key
    - Enumerate threshold value and metadata columns
    - Expose required columns, canonical column order, and expected dtypes
    - Remain free of threshold estimation, signal generation, and persistence

Dependencies:
    ``polars`` and the Python standard library.

Public API:
    ``PRIMARY_KEY_COLUMNS``, ``THRESHOLD_COLUMNS``, ``METADATA_COLUMNS``,
    ``REQUIRED_COLUMNS``, ``CANONICAL_COLUMN_ORDER``, ``COLUMN_DTYPES``,
    ``THRESHOLD_SCHEMA``
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

import polars as pl

__all__ = [
    "CANONICAL_COLUMN_ORDER",
    "COLUMN_DTYPES",
    "METADATA_COLUMNS",
    "PRIMARY_KEY_COLUMNS",
    "REQUIRED_COLUMNS",
    "THRESHOLD_COLUMNS",
    "THRESHOLD_SCHEMA",
]

PRIMARY_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "model_name",
    "model_version",
    "profile",
)

THRESHOLD_COLUMNS: Final[tuple[str, ...]] = (
    "buy_threshold",
    "sell_threshold",
)

METADATA_COLUMNS: Final[tuple[str, ...]] = ("created_at",)

CANONICAL_COLUMN_ORDER: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "model_name",
    "model_version",
    "buy_threshold",
    "sell_threshold",
    "profile",
    "created_at",
)

REQUIRED_COLUMNS: Final[tuple[str, ...]] = CANONICAL_COLUMN_ORDER

COLUMN_DTYPES: Final = MappingProxyType(
    {
        "symbol": pl.String,
        "timeframe": pl.String,
        "model_name": pl.String,
        "model_version": pl.String,
        "buy_threshold": pl.Float64,
        "sell_threshold": pl.Float64,
        "profile": pl.String,
        "created_at": pl.Datetime("us", "UTC"),
    }
)

THRESHOLD_SCHEMA: Final[pl.Schema] = pl.Schema(
    [(column, COLUMN_DTYPES[column]) for column in CANONICAL_COLUMN_ORDER]
)
