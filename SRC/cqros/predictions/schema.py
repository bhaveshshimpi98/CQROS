"""CQROS merged prediction dataset schema.

Purpose:
    Define the canonical columnar contract for model prediction outputs
    persisted prior to signal generation.

Responsibilities:
    - Declare the merged-dataset primary key
    - Enumerate model metadata and prediction output columns
    - Expose required columns, canonical column order, and expected dtypes
    - Remain free of inference, validation, signal generation, and
      persistence logic

Dependencies:
    ``polars`` and the Python standard library.

Public API:
    ``PRIMARY_KEY_COLUMNS``, ``METADATA_COLUMNS``, ``PREDICTION_COLUMNS``,
    ``REQUIRED_COLUMNS``, ``CANONICAL_COLUMN_ORDER``, ``COLUMN_DTYPES``,
    ``MERGED_PREDICTION_SCHEMA``

Notes:
    This module describes column presence and dtypes only; it does not
    validate frames or compute predictions.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

import polars as pl

__all__ = [
    "CANONICAL_COLUMN_ORDER",
    "COLUMN_DTYPES",
    "MERGED_PREDICTION_SCHEMA",
    "METADATA_COLUMNS",
    "PREDICTION_COLUMNS",
    "PRIMARY_KEY_COLUMNS",
    "REQUIRED_COLUMNS",
]

PRIMARY_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
)

# Model identity columns persisted alongside each prediction row.
METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "model_name",
    "model_version",
)

# Continuous model prediction column.
PREDICTION_COLUMNS: Final[tuple[str, ...]] = ("prediction",)

CANONICAL_COLUMN_ORDER: Final[tuple[str, ...]] = (
    *PRIMARY_KEY_COLUMNS,
    *METADATA_COLUMNS,
    *PREDICTION_COLUMNS,
)

REQUIRED_COLUMNS: Final[tuple[str, ...]] = CANONICAL_COLUMN_ORDER

COLUMN_DTYPES: Final = MappingProxyType(
    {
        "symbol": pl.String,
        "timeframe": pl.String,
        "open_time": pl.Int64,
        "model_name": pl.String,
        "model_version": pl.String,
        "prediction": pl.Float64,
    }
)

MERGED_PREDICTION_SCHEMA: Final[pl.Schema] = pl.Schema(
    [(column, COLUMN_DTYPES[column]) for column in CANONICAL_COLUMN_ORDER]
)
