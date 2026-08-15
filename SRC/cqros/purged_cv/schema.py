"""CQROS purged cross-validation dataset schema.

Purpose:
    Define the canonical columnar contract for purged cross-validation
    ledgers produced by the CQROS Purged Cross Validation layer.

Responsibilities:
    - Declare the purged-CV primary key
    - Enumerate fold metadata and evaluation metric columns
    - Expose required columns, canonical column order, and expected dtypes
    - Expose the purged-CV status enumeration
    - Remain free of fold generation, leakage detection, and persistence logic

Dependencies:
    ``polars`` and the Python standard library.

Public API:
    ``PRIMARY_KEY_COLUMNS``, ``PURGED_CV_COLUMNS``,
    ``REQUIRED_COLUMNS``, ``CANONICAL_COLUMN_ORDER``, ``COLUMN_DTYPES``,
    ``PURGED_CV_SCHEMA``, ``PurgedCVStatus``,
    ``purged_cv_statuses``, ``purged_cv_status_values``

Notes:
    This module describes column presence and dtypes only; it does not
    generate folds, detect leakage, validate frames, or persist ledgers.
    ``status`` stores ``PurgedCVStatus`` enum string values
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
    "PURGED_CV_COLUMNS",
    "PURGED_CV_SCHEMA",
    "REQUIRED_COLUMNS",
    "PurgedCVStatus",
    "purged_cv_status_values",
    "purged_cv_statuses",
]

PRIMARY_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "strategy_name",
    "strategy_version",
    "timeframe",
    "fold_id",
)

# Strategy identity, fold metadata, evaluation metrics, and status.
PURGED_CV_COLUMNS: Final[tuple[str, ...]] = (
    "strategy_name",
    "strategy_version",
    "timeframe",
    "fold_id",
    "train_start_time",
    "train_end_time",
    "test_start_time",
    "test_end_time",
    "purge_size",
    "embargo_size",
    "train_rows",
    "test_rows",
    "train_score",
    "test_score",
    "overfit_gap",
    "status",
)

CANONICAL_COLUMN_ORDER: Final[tuple[str, ...]] = PURGED_CV_COLUMNS

REQUIRED_COLUMNS: Final[tuple[str, ...]] = CANONICAL_COLUMN_ORDER

COLUMN_DTYPES: Final = MappingProxyType(
    {
        "strategy_name": pl.String,
        "strategy_version": pl.String,
        "timeframe": pl.String,
        "fold_id": pl.Int32,
        "train_start_time": pl.Int64,
        "train_end_time": pl.Int64,
        "test_start_time": pl.Int64,
        "test_end_time": pl.Int64,
        "purge_size": pl.Int64,
        "embargo_size": pl.Int64,
        "train_rows": pl.Int64,
        "test_rows": pl.Int64,
        "train_score": pl.Float64,
        "test_score": pl.Float64,
        "overfit_gap": pl.Float64,
        "status": pl.String,
    }
)

PURGED_CV_SCHEMA: Final[pl.Schema] = pl.Schema(
    [(column, COLUMN_DTYPES[column]) for column in CANONICAL_COLUMN_ORDER]
)


# Prefer str+Enum over StrEnum so enum values embed directly into Polars.
class PurgedCVStatus(str, Enum):  # noqa: UP042
    """Canonical evaluation status for a purged cross-validation metrics row.

    Attributes:
        PASS: Purged-CV fold evaluation satisfies configured thresholds.
        FAIL: Purged-CV fold evaluation fails configured thresholds.
    """

    PASS = "PASS"
    FAIL = "FAIL"


def purged_cv_statuses() -> tuple[PurgedCVStatus, ...]:
    """Return an immutable copy of every ``PurgedCVStatus`` member.

    Returns:
        All purged-CV-status members in declaration order.
    """
    return (PurgedCVStatus.PASS, PurgedCVStatus.FAIL)


def purged_cv_status_values() -> tuple[str, ...]:
    """Return an immutable copy of every ``PurgedCVStatus`` string value.

    Returns:
        All purged-CV-status string values in declaration order.
    """
    return tuple(member.value for member in purged_cv_statuses())
