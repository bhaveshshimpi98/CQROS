"""CQROS merged executed-trade dataset schema.

Purpose:
    Define the canonical columnar contract for executed trades produced by the
    CQROS Execution Engine from OMS order datasets.

Responsibilities:
    - Declare the merged-dataset primary key
    - Enumerate model metadata and trade execution columns
    - Expose required columns, canonical column order, and expected dtypes
    - Remain free of simulation, validation, and persistence logic

Dependencies:
    ``polars`` and the Python standard library.

Public API:
    ``PRIMARY_KEY_COLUMNS``, ``METADATA_COLUMNS``, ``TRADE_COLUMNS``,
    ``REQUIRED_COLUMNS``, ``CANONICAL_COLUMN_ORDER``, ``COLUMN_DTYPES``,
    ``MERGED_TRADE_SCHEMA``, ``ExecutionStatus``, ``execution_statuses``,
    ``values``

Notes:
    This module describes column presence and dtypes only; it does not
    validate frames, simulate fills, or persist trades. ``manager`` and
    ``signal`` preserve upstream OMS / risk lineage on every trade row.
"""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType
from typing import Final

import polars as pl

__all__ = [
    "CANONICAL_COLUMN_ORDER",
    "COLUMN_DTYPES",
    "ExecutionStatus",
    "MERGED_TRADE_SCHEMA",
    "METADATA_COLUMNS",
    "PRIMARY_KEY_COLUMNS",
    "REQUIRED_COLUMNS",
    "TRADE_COLUMNS",
    "execution_statuses",
    "values",
]

PRIMARY_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
)

# Model identity and upstream construction lineage preserved onto each trade.
METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "model_name",
    "model_version",
    "optimizer",
    "policy",
    "manager",
    "signal",
)

# Side, sizing, pricing, cost, status, and execution timestamp columns.
TRADE_COLUMNS: Final[tuple[str, ...]] = (
    "side",
    "order_type",
    "requested_quantity",
    "executed_quantity",
    "requested_price",
    "executed_price",
    "fees",
    "slippage",
    "status",
    "execution_time",
)

CANONICAL_COLUMN_ORDER: Final[tuple[str, ...]] = (
    *PRIMARY_KEY_COLUMNS,
    *METADATA_COLUMNS,
    *TRADE_COLUMNS,
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
        "manager": pl.Utf8,
        "signal": pl.Utf8,
        "side": pl.Utf8,
        "order_type": pl.Utf8,
        "requested_quantity": pl.Float64,
        "executed_quantity": pl.Float64,
        "requested_price": pl.Float64,
        "executed_price": pl.Float64,
        "fees": pl.Float64,
        "slippage": pl.Float64,
        "status": pl.Utf8,
        "execution_time": pl.Datetime("us", "UTC"),
    }
)

MERGED_TRADE_SCHEMA: Final[pl.Schema] = pl.Schema(
    [(column, COLUMN_DTYPES[column]) for column in CANONICAL_COLUMN_ORDER]
)


# Prefer str+Enum over StrEnum so enum values embed directly into Polars.
class ExecutionStatus(str, Enum):  # noqa: UP042
    """Canonical executed-trade lifecycle status.

    Attributes:
        FILLED: Order completely filled by the execution simulator.
    """

    FILLED = "FILLED"


def execution_statuses() -> tuple[ExecutionStatus, ...]:
    """Return an immutable copy of every ``ExecutionStatus`` member.

    Returns:
        All execution-status members in declaration order.
    """
    return (ExecutionStatus.FILLED,)


def values[EnumT: Enum](enum_cls: type[EnumT]) -> tuple[str, ...]:
    """Return an immutable copy of every string value for ``enum_cls``.

    Args:
        enum_cls: Enumeration class whose member values are requested.

    Returns:
        All member string values in declaration order.
    """
    return tuple(member.value for member in enum_cls)
