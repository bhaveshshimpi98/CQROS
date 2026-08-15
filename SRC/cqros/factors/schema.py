"""CQROS factor dataset schema.

Purpose:
    Define the canonical columnar contract for factor datasets produced by
    the CQROS Factor Research Engine.

Responsibilities:
    - Declare the factor-dataset primary key
    - Enumerate factor metadata and factor value columns
    - Expose required columns, canonical column order, and expected dtypes
    - Expose the factor status enumeration
    - Remain free of factor computation, validation, and persistence logic

Dependencies:
    ``polars`` and the Python standard library.

Public API:
    ``PRIMARY_KEY_COLUMNS``, ``METADATA_COLUMNS``, ``FACTOR_COLUMNS``,
    ``REQUIRED_COLUMNS``, ``CANONICAL_COLUMN_ORDER``, ``COLUMN_DTYPES``,
    ``FACTOR_SCHEMA``, ``FactorStatus``, ``factor_statuses``,
    ``factor_status_values``

Notes:
    This module describes column presence and dtypes only; it does not
    compute factors, validate frames, or persist datasets.
    ``status`` stores ``FactorStatus`` enum string values
    (``ACTIVE``, ``DEPRECATED``).
"""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType
from typing import Final

import polars as pl

__all__ = [
    "CANONICAL_COLUMN_ORDER",
    "COLUMN_DTYPES",
    "FACTOR_COLUMNS",
    "FACTOR_SCHEMA",
    "METADATA_COLUMNS",
    "PRIMARY_KEY_COLUMNS",
    "REQUIRED_COLUMNS",
    "FactorStatus",
    "factor_status_values",
    "factor_statuses",
]

PRIMARY_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
)

# Factor identity and classification columns.
METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "factor_name",
    "factor_version",
    "factor_category",
    "factor_group",
)

# Factor value, window, enablement, and lifecycle columns.
FACTOR_COLUMNS: Final[tuple[str, ...]] = (
    "factor_value",
    "lookback",
    "prediction_horizon",
    "enabled",
    "status",
)

CANONICAL_COLUMN_ORDER: Final[tuple[str, ...]] = (
    *PRIMARY_KEY_COLUMNS,
    *METADATA_COLUMNS,
    *FACTOR_COLUMNS,
)

REQUIRED_COLUMNS: Final[tuple[str, ...]] = CANONICAL_COLUMN_ORDER

COLUMN_DTYPES: Final = MappingProxyType(
    {
        "symbol": pl.String,
        "timeframe": pl.String,
        "open_time": pl.Int64,
        "factor_name": pl.String,
        "factor_version": pl.String,
        "factor_category": pl.String,
        "factor_group": pl.String,
        "factor_value": pl.Float64,
        "lookback": pl.Int32,
        "prediction_horizon": pl.Int32,
        "enabled": pl.Boolean,
        "status": pl.String,
    }
)

FACTOR_SCHEMA: Final[pl.Schema] = pl.Schema(
    [(column, COLUMN_DTYPES[column]) for column in CANONICAL_COLUMN_ORDER]
)


# Prefer str+Enum over StrEnum so enum values embed directly into Polars.
class FactorStatus(str, Enum):  # noqa: UP042
    """Canonical lifecycle status for a factor dataset row.

    Attributes:
        ACTIVE: Factor is eligible for research and downstream use.
        DEPRECATED: Factor remains readable but should not be promoted.
    """

    ACTIVE = "ACTIVE"
    DEPRECATED = "DEPRECATED"


def factor_statuses() -> tuple[FactorStatus, ...]:
    """Return an immutable copy of every ``FactorStatus`` member.

    Returns:
        All factor-status members in declaration order.
    """
    return (FactorStatus.ACTIVE, FactorStatus.DEPRECATED)


def factor_status_values() -> tuple[str, ...]:
    """Return an immutable copy of every ``FactorStatus`` string value.

    Returns:
        All factor-status string values in declaration order.
    """
    return tuple(member.value for member in factor_statuses())
