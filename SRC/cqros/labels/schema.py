"""CQROS merged label dataset schema.

Purpose:
    Define the canonical columnar contract for the merged label matrix
    persisted by ``LabelRepository``.

Responsibilities:
    - Declare the merged-dataset primary key
    - Enumerate every currently defined label output column
    - Expose required columns, canonical column order, and expected dtypes
    - Remain free of label computation, validation, and persistence logic

Dependencies:
    ``polars`` and the Python standard library.

Public API:
    ``PRIMARY_KEY_COLUMNS``, ``REGRESSION_LABEL_COLUMNS``,
    ``CLASSIFICATION_LABEL_COLUMNS``, ``LABEL_COLUMNS``,
    ``REQUIRED_COLUMNS``, ``CANONICAL_COLUMN_ORDER``, ``COLUMN_DTYPES``,
    ``MERGED_LABEL_SCHEMA``

Notes:
    Label values may be null at the trailing edge of a series where forward
    horizons are incomplete. This module describes column presence and
    dtypes only; it does not validate frames.
"""

from __future__ import annotations

from typing import Final

import polars as pl

__all__ = [
    "CANONICAL_COLUMN_ORDER",
    "CLASSIFICATION_LABEL_COLUMNS",
    "COLUMN_DTYPES",
    "LABEL_COLUMNS",
    "MERGED_LABEL_SCHEMA",
    "PRIMARY_KEY_COLUMNS",
    "REGRESSION_LABEL_COLUMNS",
    "REQUIRED_COLUMNS",
]

PRIMARY_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
)

# Canonical regression label columns ordered by horizon.
REGRESSION_LABEL_COLUMNS: Final[tuple[str, ...]] = (
    "future_return_1",
    "future_return_5",
    "future_return_10",
    "future_return_20",
)

# Canonical classification label columns ordered by horizon.
CLASSIFICATION_LABEL_COLUMNS: Final[tuple[str, ...]] = (
    "direction_1",
    "direction_5",
    "direction_10",
    "direction_20",
)

# One-to-one with the current Label Engine catalog.
LABEL_COLUMNS: Final[tuple[str, ...]] = (
    *REGRESSION_LABEL_COLUMNS,
    *CLASSIFICATION_LABEL_COLUMNS,
)

CANONICAL_COLUMN_ORDER: Final[tuple[str, ...]] = (
    *PRIMARY_KEY_COLUMNS,
    *LABEL_COLUMNS,
)

REQUIRED_COLUMNS: Final[tuple[str, ...]] = CANONICAL_COLUMN_ORDER

COLUMN_DTYPES: Final[dict[str, pl.DataType]] = {
    "symbol": pl.String,
    "timeframe": pl.String,
    "open_time": pl.Int64,
    "future_return_1": pl.Float64,
    "future_return_5": pl.Float64,
    "future_return_10": pl.Float64,
    "future_return_20": pl.Float64,
    "direction_1": pl.Int8,
    "direction_5": pl.Int8,
    "direction_10": pl.Int8,
    "direction_20": pl.Int8,
}

MERGED_LABEL_SCHEMA: Final[pl.Schema] = pl.Schema(
    [(column, COLUMN_DTYPES[column]) for column in CANONICAL_COLUMN_ORDER]
)
