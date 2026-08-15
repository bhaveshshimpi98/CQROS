"""CQROS Research Model Ledger schema.

Purpose:
    Define the canonical columnar contract for Research Model Ledger
    datasets produced by the CQROS Models layer from Regime rows.

Responsibilities:
    - Declare the models primary key
    - Enumerate model identity, conditioning, and decision columns
    - Expose required columns, canonical column order, and expected dtypes
    - Expose the model status enumeration
    - Remain free of model math, engine, and persistence logic

Dependencies:
    ``polars`` and the Python standard library.

Public API:
    ``PRIMARY_KEY_COLUMNS``, ``MODELS_COLUMNS``,
    ``REQUIRED_COLUMNS``, ``CANONICAL_COLUMN_ORDER``, ``COLUMN_DTYPES``,
    ``MODELS_SCHEMA``, ``ModelStatus``

Notes:
    This module describes column presence and dtypes only; it does not
    train models, validate frames, or persist artifacts.
    ``validation_score`` is the ledger model score field and carries
    ``regime_score`` from the Regime → Models handoff.
    ``status`` stores ``ModelStatus`` enum string values
    (``PASS``, ``FAIL``).
    This schema is not the supervised ``cqros.ml`` trained-model artifact
    contract.
"""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType
from typing import Final

import polars as pl

__all__ = [
    "CANONICAL_COLUMN_ORDER",
    "COLUMN_DTYPES",
    "MODELS_COLUMNS",
    "MODELS_SCHEMA",
    "PRIMARY_KEY_COLUMNS",
    "REQUIRED_COLUMNS",
    "ModelStatus",
]

PRIMARY_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "model_id",
    "regime_id",
    "symbol",
    "timeframe",
    "training_time",
)

# Model identity, training outputs, feature linkage, and status.
MODELS_COLUMNS: Final[tuple[str, ...]] = (
    "model_id",
    "regime_id",
    "symbol",
    "timeframe",
    "training_time",
    "model_type",
    "model_version",
    "prediction_horizon",
    "validation_score",
    "feature_set_id",
    "model_metadata",
    "status",
)

CANONICAL_COLUMN_ORDER: Final[tuple[str, ...]] = MODELS_COLUMNS

REQUIRED_COLUMNS: Final[tuple[str, ...]] = CANONICAL_COLUMN_ORDER

COLUMN_DTYPES: Final = MappingProxyType(
    {
        "model_id": pl.String,
        "regime_id": pl.String,
        "symbol": pl.String,
        "timeframe": pl.String,
        "training_time": pl.Int64,
        "model_type": pl.String,
        "model_version": pl.String,
        "prediction_horizon": pl.Int32,
        "validation_score": pl.Float64,
        "feature_set_id": pl.String,
        "model_metadata": pl.List(pl.String),
        "status": pl.String,
    }
)

MODELS_SCHEMA: Final[pl.Schema] = pl.Schema(
    [(column, COLUMN_DTYPES[column]) for column in CANONICAL_COLUMN_ORDER]
)


# Prefer str+Enum over StrEnum so enum values embed directly into Polars.
class ModelStatus(str, Enum):  # noqa: UP042
    """Canonical evaluation status for a model artifact row.

    Attributes:
        PASS: Model artifact satisfies configured thresholds.
        FAIL: Model artifact fails configured thresholds.
    """

    PASS = "PASS"
    FAIL = "FAIL"
