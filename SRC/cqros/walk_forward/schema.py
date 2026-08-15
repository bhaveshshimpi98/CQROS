"""CQROS walk-forward evaluation dataset schema.

Purpose:
    Define the canonical columnar contract for walk-forward evaluation
    ledgers produced by the CQROS Walk-Forward layer.

Responsibilities:
    - Declare the walk-forward primary key
    - Enumerate fold metadata and evaluation metric columns
    - Expose required columns, canonical column order, and expected dtypes
    - Expose the walk-forward status enumeration
    - Remain free of walk-forward math, engine, and persistence logic

Dependencies:
    ``polars`` and the Python standard library.

Public API:
    ``PRIMARY_KEY_COLUMNS``, ``WALK_FORWARD_COLUMNS``,
    ``REQUIRED_COLUMNS``, ``CANONICAL_COLUMN_ORDER``, ``COLUMN_DTYPES``,
    ``WALK_FORWARD_SCHEMA``, ``WalkForwardStatus``,
    ``walk_forward_statuses``, ``walk_forward_status_values``

Notes:
    This module describes column presence and dtypes only; it does not
    compute fold windows, validate frames, or persist ledgers.
    ``status`` stores ``WalkForwardStatus`` enum string values
    (``PASS``, ``FAIL``).
"""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType
from typing import Final

import polars as pl

__all__ = [
    "CANONICAL_COLUMN_ORDER",
    "COLUMN_DTYPES",
    "PRIMARY_KEY_COLUMNS",
    "REQUIRED_COLUMNS",
    "WALK_FORWARD_COLUMNS",
    "WALK_FORWARD_SCHEMA",
    "WalkForwardStatus",
    "walk_forward_status_values",
    "walk_forward_statuses",
]

PRIMARY_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "strategy_name",
    "strategy_version",
    "timeframe",
    "fold_id",
)

# Strategy identity, fold metadata, evaluation metrics, and status.
WALK_FORWARD_COLUMNS: Final[tuple[str, ...]] = (
    "strategy_name",
    "strategy_version",
    "timeframe",
    "fold_id",
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    "train_rows",
    "test_rows",
    "selected_factors",
    "model_version",
    "train_score",
    "test_score",
    "overfit_gap",
    "status",
)

CANONICAL_COLUMN_ORDER: Final[tuple[str, ...]] = WALK_FORWARD_COLUMNS

REQUIRED_COLUMNS: Final[tuple[str, ...]] = CANONICAL_COLUMN_ORDER

COLUMN_DTYPES: Final = MappingProxyType(
    {
        "strategy_name": pl.String,
        "strategy_version": pl.String,
        "timeframe": pl.String,
        "fold_id": pl.Int32,
        "train_start": pl.Int64,
        "train_end": pl.Int64,
        "test_start": pl.Int64,
        "test_end": pl.Int64,
        "train_rows": pl.Int64,
        "test_rows": pl.Int64,
        "selected_factors": pl.Int32,
        "model_version": pl.String,
        "train_score": pl.Float64,
        "test_score": pl.Float64,
        "overfit_gap": pl.Float64,
        "status": pl.String,
    }
)

WALK_FORWARD_SCHEMA: Final[pl.Schema] = pl.Schema(
    [(column, COLUMN_DTYPES[column]) for column in CANONICAL_COLUMN_ORDER]
)


# Prefer str+Enum over StrEnum so enum values embed directly into Polars.
class WalkForwardStatus(str, Enum):  # noqa: UP042
    """Canonical evaluation status for a walk-forward metrics row.

    Attributes:
        PASS: Walk-forward fold evaluation satisfies configured thresholds.
        FAIL: Walk-forward fold evaluation fails configured thresholds.
    """

    PASS = "PASS"
    FAIL = "FAIL"


def walk_forward_statuses() -> tuple[WalkForwardStatus, ...]:
    """Return an immutable copy of every ``WalkForwardStatus`` member.

    Returns:
        All walk-forward-status members in declaration order.
    """
    return (WalkForwardStatus.PASS, WalkForwardStatus.FAIL)


def walk_forward_status_values() -> tuple[str, ...]:
    """Return an immutable copy of every ``WalkForwardStatus`` string value.

    Returns:
        All walk-forward-status string values in declaration order.
    """
    return tuple(member.value for member in walk_forward_statuses())
