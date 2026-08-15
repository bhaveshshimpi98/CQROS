"""CQROS merged feature dataset schema.

Purpose:
    Define the canonical columnar contract for the merged feature matrix
    persisted by ``FeatureRepository``.

Responsibilities:
    - Declare the merged-dataset primary key
    - Enumerate every currently implemented feature output column
    - Expose required columns, canonical column order, and expected dtypes
    - Remain free of feature computation, validation, and persistence logic

Dependencies:
    ``polars`` and the Python standard library.

Public API:
    ``PRIMARY_KEY_COLUMNS``, ``FEATURE_COLUMNS``, ``FEATURE_NAMES``,
    ``REQUIRED_COLUMNS``, ``CANONICAL_COLUMN_ORDER``, ``COLUMN_DTYPES``,
    ``MERGED_FEATURE_SCHEMA``

Notes:
    Feature values may be null during warm-up windows. This module describes
    column presence and dtypes only; it does not validate frames.
"""

from __future__ import annotations

from typing import Final

import polars as pl

__all__ = [
    "CANONICAL_COLUMN_ORDER",
    "COLUMN_DTYPES",
    "FEATURE_COLUMNS",
    "FEATURE_NAMES",
    "MERGED_FEATURE_SCHEMA",
    "PRIMARY_KEY_COLUMNS",
    "REQUIRED_COLUMNS",
]

PRIMARY_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
)

# Canonical feature output columns in package order:
# price → funding → open_interest → taker → long_short.
FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    # price
    "returns",
    "log_returns",
    "rolling_mean",
    "rolling_std",
    "rolling_max",
    "rolling_min",
    "atr",
    "dollar_volume",
    # funding
    "funding_change",
    "funding_rolling_mean",
    "funding_zscore",
    "funding_momentum",
    # open_interest
    "oi_change",
    "oi_percent_change",
    "oi_rolling_mean",
    "oi_zscore",
    "oi_momentum",
    # taker
    "buy_pressure",
    "sell_pressure",
    "buy_sell_ratio",
    "delta_volume",
    "flow_imbalance",
    # long_short
    "ratio_change",
    "ratio_momentum",
    "ratio_zscore",
    "crowding_score",
)

# One-to-one with FEATURE_COLUMNS for the current Feature Engine catalog.
FEATURE_NAMES: Final[tuple[str, ...]] = FEATURE_COLUMNS

CANONICAL_COLUMN_ORDER: Final[tuple[str, ...]] = (
    *PRIMARY_KEY_COLUMNS,
    *FEATURE_COLUMNS,
)

REQUIRED_COLUMNS: Final[tuple[str, ...]] = CANONICAL_COLUMN_ORDER

COLUMN_DTYPES: Final[dict[str, pl.DataType]] = {
    "symbol": pl.String,
    "timeframe": pl.String,
    "open_time": pl.Int64,
    "returns": pl.Float64,
    "log_returns": pl.Float64,
    "rolling_mean": pl.Float64,
    "rolling_std": pl.Float64,
    "rolling_max": pl.Float64,
    "rolling_min": pl.Float64,
    "atr": pl.Float64,
    "dollar_volume": pl.Float64,
    "funding_change": pl.Float64,
    "funding_rolling_mean": pl.Float64,
    "funding_zscore": pl.Float64,
    "funding_momentum": pl.Float64,
    "oi_change": pl.Float64,
    "oi_percent_change": pl.Float64,
    "oi_rolling_mean": pl.Float64,
    "oi_zscore": pl.Float64,
    "oi_momentum": pl.Float64,
    "buy_pressure": pl.Float64,
    "sell_pressure": pl.Float64,
    "buy_sell_ratio": pl.Float64,
    "delta_volume": pl.Float64,
    "flow_imbalance": pl.Float64,
    "ratio_change": pl.Float64,
    "ratio_momentum": pl.Float64,
    "ratio_zscore": pl.Float64,
    "crowding_score": pl.Float64,
}

MERGED_FEATURE_SCHEMA: Final[pl.Schema] = pl.Schema(
    [(column, COLUMN_DTYPES[column]) for column in CANONICAL_COLUMN_ORDER]
)
