"""CQROS merged risk decision dataset schema.

Purpose:
    Define the canonical columnar contract for risk decisions produced by
    the CQROS Risk Management subsystem from portfolio allocation datasets.

Responsibilities:
    - Declare the merged-dataset primary key
    - Enumerate model metadata and risk decision columns
    - Expose required columns, canonical column order, and expected dtypes
    - Remain free of policy, calculation, validation, and persistence logic

Dependencies:
    ``polars`` and the Python standard library.

Public API:
    ``PRIMARY_KEY_COLUMNS``, ``METADATA_COLUMNS``, ``RISK_COLUMNS``,
    ``REQUIRED_COLUMNS``, ``CANONICAL_COLUMN_ORDER``, ``COLUMN_DTYPES``,
    ``MERGED_RISK_SCHEMA``

Notes:
    This module describes column presence and dtypes only; it does not
    validate frames or compute risk decisions. The ``signal`` column stores
    discrete signal string values; ``target_weight`` stores the proposed
    portfolio allocation; ``approved_weight`` stores the risk-adjusted
    allocation; ``decision`` and ``reason`` store the risk outcome and
    explanatory text for each primary-key row. ``optimizer`` and ``policy``
    preserve upstream execution lineage for OMS order generation.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

import polars as pl

__all__ = [
    "CANONICAL_COLUMN_ORDER",
    "COLUMN_DTYPES",
    "MERGED_RISK_SCHEMA",
    "METADATA_COLUMNS",
    "PRIMARY_KEY_COLUMNS",
    "REQUIRED_COLUMNS",
    "RISK_COLUMNS",
]

PRIMARY_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
)

# Model identity and execution lineage preserved from upstream datasets.
METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "model_name",
    "model_version",
    "optimizer",
    "policy",
)

# Discrete signal, proposed and approved weights, and risk outcome.
RISK_COLUMNS: Final[tuple[str, ...]] = (
    "signal",
    "target_weight",
    "approved_weight",
    "decision",
    "reason",
)

CANONICAL_COLUMN_ORDER: Final[tuple[str, ...]] = (
    *PRIMARY_KEY_COLUMNS,
    *METADATA_COLUMNS,
    *RISK_COLUMNS,
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
        "policy": pl.Utf8,
        "signal": pl.Utf8,
        "target_weight": pl.Float64,
        "approved_weight": pl.Float64,
        "decision": pl.Utf8,
        "reason": pl.Utf8,
    }
)

MERGED_RISK_SCHEMA: Final[pl.Schema] = pl.Schema(
    [(column, COLUMN_DTYPES[column]) for column in CANONICAL_COLUMN_ORDER]
)
