"""CQROS merged signal dataset schema.

Purpose:
    Define the canonical columnar contract for trading signals produced by
    CQROS signal policies from prediction datasets.

Responsibilities:
    - Declare the merged-dataset primary key
    - Enumerate model metadata and discrete signal output columns
    - Expose required columns, canonical column order, and expected dtypes
    - Remain free of signal generation, validation, and persistence logic

Dependencies:
    ``polars`` and the Python standard library.

Public API:
    ``PRIMARY_KEY_COLUMNS``, ``METADATA_COLUMNS``, ``SIGNAL_COLUMNS``,
    ``REQUIRED_COLUMNS``, ``CANONICAL_COLUMN_ORDER``, ``COLUMN_DTYPES``,
    ``MERGED_SIGNAL_SCHEMA``

Notes:
    This module describes column presence and dtypes only; it does not
    validate frames or compute signals. The ``signal`` column stores
    ``Signal`` enum string values (``BUY``, ``SELL``, ``HOLD``).
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

import polars as pl

__all__ = [
    "CANONICAL_COLUMN_ORDER",
    "COLUMN_DTYPES",
    "MERGED_SIGNAL_SCHEMA",
    "METADATA_COLUMNS",
    "PRIMARY_KEY_COLUMNS",
    "REQUIRED_COLUMNS",
    "SIGNAL_COLUMNS",
]

PRIMARY_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
)

# Model identity columns preserved from the prediction dataset.
METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "model_name",
    "model_version",
)

# Discrete trading signal column.
SIGNAL_COLUMNS: Final[tuple[str, ...]] = ("signal",)

CANONICAL_COLUMN_ORDER: Final[tuple[str, ...]] = (
    *PRIMARY_KEY_COLUMNS,
    *METADATA_COLUMNS,
    *SIGNAL_COLUMNS,
)

REQUIRED_COLUMNS: Final[tuple[str, ...]] = CANONICAL_COLUMN_ORDER

COLUMN_DTYPES: Final = MappingProxyType(
    {
        "symbol": pl.String,
        "timeframe": pl.String,
        "open_time": pl.Int64,
        "model_name": pl.String,
        "model_version": pl.String,
        "signal": pl.String,
    }
)

MERGED_SIGNAL_SCHEMA: Final[pl.Schema] = pl.Schema(
    [(column, COLUMN_DTYPES[column]) for column in CANONICAL_COLUMN_ORDER]
)
