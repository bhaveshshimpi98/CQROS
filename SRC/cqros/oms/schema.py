"""CQROS merged order dataset schema.

Purpose:
    Define the canonical columnar contract for Order Management System (OMS)
    order datasets produced from CQROS risk-approved portfolio allocations.

Responsibilities:
    - Declare the merged-dataset primary key
    - Enumerate model metadata and order columns
    - Expose required columns, canonical column order, and expected dtypes
    - Remain free of order generation, execution, validation, and persistence
      logic

Dependencies:
    ``polars`` and the Python standard library.

Public API:
    ``PRIMARY_KEY_COLUMNS``, ``METADATA_COLUMNS``, ``ORDER_COLUMNS``,
    ``REQUIRED_COLUMNS``, ``CANONICAL_COLUMN_ORDER``, ``COLUMN_DTYPES``,
    ``MERGED_ORDER_SCHEMA``

Notes:
    This module describes column presence and dtypes only; it does not
    validate frames, transition order state, or persist orders. Canonical
    column order places ``parent_order_id`` immediately after the primary
    key and before metadata columns.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

import polars as pl

__all__ = [
    "CANONICAL_COLUMN_ORDER",
    "COLUMN_DTYPES",
    "MERGED_ORDER_SCHEMA",
    "METADATA_COLUMNS",
    "ORDER_COLUMNS",
    "PRIMARY_KEY_COLUMNS",
    "REQUIRED_COLUMNS",
]

PRIMARY_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
    "order_id",
)

# Model and construction identity preserved alongside each order row.
METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "model_name",
    "model_version",
    "policy",
    "optimizer",
)

# Order identity, sizing, pricing, fill, and lifecycle columns.
ORDER_COLUMNS: Final[tuple[str, ...]] = (
    "parent_order_id",
    "side",
    "order_type",
    "quantity",
    "limit_price",
    "stop_price",
    "filled_quantity",
    "average_fill_price",
    "status",
    "created_at",
    "updated_at",
)

# parent_order_id follows the primary key and precedes metadata.
CANONICAL_COLUMN_ORDER: Final[tuple[str, ...]] = (
    *PRIMARY_KEY_COLUMNS,
    "parent_order_id",
    *METADATA_COLUMNS,
    "side",
    "order_type",
    "quantity",
    "limit_price",
    "stop_price",
    "filled_quantity",
    "average_fill_price",
    "status",
    "created_at",
    "updated_at",
)

REQUIRED_COLUMNS: Final[tuple[str, ...]] = CANONICAL_COLUMN_ORDER

COLUMN_DTYPES: Final = MappingProxyType(
    {
        "symbol": pl.Utf8,
        "timeframe": pl.Utf8,
        "open_time": pl.Datetime("us", "UTC"),
        "order_id": pl.Utf8,
        "parent_order_id": pl.Utf8,
        "model_name": pl.Utf8,
        "model_version": pl.Utf8,
        "policy": pl.Utf8,
        "optimizer": pl.Utf8,
        "side": pl.Utf8,
        "order_type": pl.Utf8,
        "quantity": pl.Float64,
        "limit_price": pl.Float64,
        "stop_price": pl.Float64,
        "filled_quantity": pl.Float64,
        "average_fill_price": pl.Float64,
        "status": pl.Utf8,
        "created_at": pl.Datetime("us", "UTC"),
        "updated_at": pl.Datetime("us", "UTC"),
    }
)

MERGED_ORDER_SCHEMA: Final[pl.Schema] = pl.Schema(
    [(column, COLUMN_DTYPES[column]) for column in CANONICAL_COLUMN_ORDER]
)
