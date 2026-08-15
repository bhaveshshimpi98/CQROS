"""CQROS factor-validation input dataset schema.

Purpose:
    Compose the columnar contract for the in-memory validation dataset
    assembled by ``ValidationDatasetBuilder`` from canonical Factors and
    Labels partitions.

Responsibilities:
    - Reuse Factors and Labels schema exports without duplication
    - Declare join keys shared by Factors and Labels
    - Declare the label columns appended for validation engine input
    - Expose required columns, canonical column order, and expected dtypes
    - Remain free of joins, persistence, and validation math

Dependencies:
    ``polars``, ``cqros.factors.schema``, and ``cqros.labels.schema``.

Public API:
    ``PRIMARY_KEY_COLUMNS``, ``FACTOR_COLUMNS``, ``VALIDATION_LABEL_COLUMNS``,
    ``REQUIRED_COLUMNS``, ``CANONICAL_COLUMN_ORDER``, ``COLUMN_DTYPES``,
    ``VALIDATION_DATASET_SCHEMA``, ``FACTOR_OBSERVATION_KEY_COLUMNS``

Notes:
    This schema describes the assembled engine-input frame only. It is never
    persisted as a lake tier. Factor and label parquet contracts remain
    unchanged.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

import polars as pl

from cqros.factors.schema import (
    CANONICAL_COLUMN_ORDER as FACTOR_CANONICAL_COLUMN_ORDER,
)
from cqros.factors.schema import (
    COLUMN_DTYPES as FACTOR_COLUMN_DTYPES,
)
from cqros.factors.schema import PRIMARY_KEY_COLUMNS
from cqros.labels.schema import (
    COLUMN_DTYPES as LABEL_COLUMN_DTYPES,
)
from cqros.labels.schema import (
    PRIMARY_KEY_COLUMNS as LABEL_PRIMARY_KEY_COLUMNS,
)
from cqros.labels.schema import REGRESSION_LABEL_COLUMNS

__all__ = [
    "CANONICAL_COLUMN_ORDER",
    "COLUMN_DTYPES",
    "FACTOR_COLUMNS",
    "FACTOR_OBSERVATION_KEY_COLUMNS",
    "PRIMARY_KEY_COLUMNS",
    "REQUIRED_COLUMNS",
    "VALIDATION_DATASET_SCHEMA",
    "VALIDATION_LABEL_COLUMNS",
]

if PRIMARY_KEY_COLUMNS != LABEL_PRIMARY_KEY_COLUMNS:
    raise RuntimeError(
        "Factor and label PRIMARY_KEY_COLUMNS must be identical for validation assembly."
    )

# Canonical Factors columns preserved through the join.
FACTOR_COLUMNS: Final[tuple[str, ...]] = FACTOR_CANONICAL_COLUMN_ORDER

# Label columns required by SimpleFactorValidationEngine input
# (``FACTOR_INPUT_COLUMNS``). Must remain a subset of regression labels.
# IC Decay additionally uses any present ``future_return_{h}`` horizons; only
# ``future_return_1`` is required for Phase-1 cross-sectional metrics.
_REQUIRED_FORWARD_RETURN: Final[str] = "future_return_1"
if _REQUIRED_FORWARD_RETURN not in REGRESSION_LABEL_COLUMNS:
    raise RuntimeError(f"{_REQUIRED_FORWARD_RETURN!r} must exist in REGRESSION_LABEL_COLUMNS.")

VALIDATION_LABEL_COLUMNS: Final[tuple[str, ...]] = (_REQUIRED_FORWARD_RETURN,)

# Long-format Factors uniqueness: bar identity plus factor identity.
FACTOR_OBSERVATION_KEY_COLUMNS: Final[tuple[str, ...]] = (
    *PRIMARY_KEY_COLUMNS,
    "factor_name",
)

CANONICAL_COLUMN_ORDER: Final[tuple[str, ...]] = (
    *FACTOR_COLUMNS,
    *VALIDATION_LABEL_COLUMNS,
)

REQUIRED_COLUMNS: Final[tuple[str, ...]] = CANONICAL_COLUMN_ORDER

COLUMN_DTYPES: Final = MappingProxyType(
    {
        **{column: FACTOR_COLUMN_DTYPES[column] for column in FACTOR_COLUMNS},
        **{column: LABEL_COLUMN_DTYPES[column] for column in VALIDATION_LABEL_COLUMNS},
    }
)

VALIDATION_DATASET_SCHEMA: Final[pl.Schema] = pl.Schema(
    [(column, COLUMN_DTYPES[column]) for column in CANONICAL_COLUMN_ORDER]
)
