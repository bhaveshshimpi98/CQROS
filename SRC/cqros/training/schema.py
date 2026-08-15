"""CQROS merged training dataset schema.

Purpose:
    Define the canonical columnar contract for the merged training matrix
    persisted by ``TrainingRepository``.

Responsibilities:
    - Declare the merged-dataset primary key shared with features and labels
    - Compose feature and label columns from their canonical schema modules
    - Expose required columns, canonical column order, and expected dtypes
    - Remain free of joins, validation, persistence, and training logic

Dependencies:
    ``polars``, ``cqros.features.schema``, and ``cqros.labels.schema``.

Public API:
    ``PRIMARY_KEY_COLUMNS``, ``FEATURE_COLUMNS``, ``LABEL_COLUMNS``,
    ``REQUIRED_COLUMNS``, ``CANONICAL_COLUMN_ORDER``, ``COLUMN_DTYPES``,
    ``MERGED_TRAINING_SCHEMA``

Notes:
    Feature values may be null during warm-up windows. Label values may be
    null at the trailing edge where forward horizons are incomplete. This
    module describes column presence and dtypes only; it does not validate
    frames.
"""

from __future__ import annotations

from typing import Final

import polars as pl

from cqros.features.schema import (
    COLUMN_DTYPES as FEATURE_COLUMN_DTYPES,
)
from cqros.features.schema import FEATURE_COLUMNS, PRIMARY_KEY_COLUMNS
from cqros.labels.schema import (
    COLUMN_DTYPES as LABEL_COLUMN_DTYPES,
)
from cqros.labels.schema import LABEL_COLUMNS
from cqros.labels.schema import (
    PRIMARY_KEY_COLUMNS as LABEL_PRIMARY_KEY_COLUMNS,
)

__all__ = [
    "CANONICAL_COLUMN_ORDER",
    "COLUMN_DTYPES",
    "FEATURE_COLUMNS",
    "LABEL_COLUMNS",
    "MERGED_TRAINING_SCHEMA",
    "PRIMARY_KEY_COLUMNS",
    "REQUIRED_COLUMNS",
]

if PRIMARY_KEY_COLUMNS != LABEL_PRIMARY_KEY_COLUMNS:
    raise RuntimeError("Feature and label PRIMARY_KEY_COLUMNS must be identical for training.")

CANONICAL_COLUMN_ORDER: Final[tuple[str, ...]] = (
    *PRIMARY_KEY_COLUMNS,
    *FEATURE_COLUMNS,
    *LABEL_COLUMNS,
)

REQUIRED_COLUMNS: Final[tuple[str, ...]] = CANONICAL_COLUMN_ORDER

COLUMN_DTYPES: Final[dict[str, pl.DataType]] = {
    **{column: FEATURE_COLUMN_DTYPES[column] for column in PRIMARY_KEY_COLUMNS},
    **{column: FEATURE_COLUMN_DTYPES[column] for column in FEATURE_COLUMNS},
    **{column: LABEL_COLUMN_DTYPES[column] for column in LABEL_COLUMNS},
}

MERGED_TRAINING_SCHEMA: Final[pl.Schema] = pl.Schema(
    [(column, COLUMN_DTYPES[column]) for column in CANONICAL_COLUMN_ORDER]
)
