"""CQROS merged portfolio dataset schema.

Purpose:
    Define the canonical columnar contract for target portfolio allocations
    produced from CQROS signal datasets.

Responsibilities:
    - Declare the merged-dataset primary key
    - Enumerate model metadata and portfolio allocation columns
    - Expose required columns, canonical column order, and expected dtypes
    - Remain free of optimization, validation, and persistence logic

Dependencies:
    ``polars`` and the Python standard library.

Public API:
    ``PRIMARY_KEY_COLUMNS``, ``METADATA_COLUMNS``, ``PORTFOLIO_COLUMNS``,
    ``REQUIRED_COLUMNS``, ``CANONICAL_COLUMN_ORDER``, ``COLUMN_DTYPES``,
    ``MERGED_PORTFOLIO_SCHEMA``

Notes:
    This module describes column presence and dtypes only; it does not
    validate frames or compute allocations. The ``signal`` column stores
    discrete signal string values; ``target_weight`` stores the portfolio
    allocation weight for each primary-key row. ``optimizer`` preserves
    portfolio-layer provenance for downstream Risk and OMS lineage.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

import polars as pl

__all__ = [
    "CANONICAL_COLUMN_ORDER",
    "COLUMN_DTYPES",
    "MERGED_PORTFOLIO_SCHEMA",
    "METADATA_COLUMNS",
    "PORTFOLIO_COLUMNS",
    "PRIMARY_KEY_COLUMNS",
    "REQUIRED_COLUMNS",
]

PRIMARY_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
)

# Model identity preserved from signals plus portfolio optimizer lineage.
METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "model_name",
    "model_version",
    "optimizer",
)

# Discrete signal and continuous target allocation weight.
PORTFOLIO_COLUMNS: Final[tuple[str, ...]] = (
    "signal",
    "target_weight",
)

CANONICAL_COLUMN_ORDER: Final[tuple[str, ...]] = (
    *PRIMARY_KEY_COLUMNS,
    *METADATA_COLUMNS,
    *PORTFOLIO_COLUMNS,
)

REQUIRED_COLUMNS: Final[tuple[str, ...]] = CANONICAL_COLUMN_ORDER

COLUMN_DTYPES: Final = MappingProxyType(
    {
        "symbol": pl.Utf8,
        "timeframe": pl.Utf8,
        "open_time": pl.Datetime("us", "UTC"),
        "model_name": pl.Utf8,
        "model_version": pl.Utf8,
        "optimizer": pl.Utf8,
        "signal": pl.Utf8,
        "target_weight": pl.Float64,
    }
)

MERGED_PORTFOLIO_SCHEMA: Final[pl.Schema] = pl.Schema(
    [(column, COLUMN_DTYPES[column]) for column in CANONICAL_COLUMN_ORDER]
)
