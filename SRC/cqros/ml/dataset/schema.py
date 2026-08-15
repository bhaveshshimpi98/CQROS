"""CQROS ML dataset schema.

Purpose:
    Expose the canonical columnar contract for every future ML component by
    reusing the merged Training schema as the single source of truth.

Responsibilities:
    - Re-export Training schema constants without redefining columns or dtypes
    - Surface regression and classification label partitions from the Label
      schema that compose Training ``LABEL_COLUMNS``
    - Provide helper accessors that return immutable copies
    - Remain free of loading, splitting, scaling, and training logic

Dependencies:
    ``cqros.training.schema`` and ``cqros.labels.schema``.

Public API:
    ``PRIMARY_KEY_COLUMNS``, ``FEATURE_COLUMNS``, ``LABEL_COLUMNS``,
    ``REGRESSION_LABEL_COLUMNS``, ``CLASSIFICATION_LABEL_COLUMNS``,
    ``REQUIRED_COLUMNS``, ``CANONICAL_COLUMN_ORDER``, ``COLUMN_DTYPES``,
    ``MERGED_TRAINING_SCHEMA``, and the helper functions listed in ``__all__``.

Notes:
    This module is a schema abstraction only. It describes column presence,
    ordering, and dtypes; it does not validate frames or access repositories.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import polars as pl

from cqros.labels.schema import CLASSIFICATION_LABEL_COLUMNS, REGRESSION_LABEL_COLUMNS
from cqros.training.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    MERGED_TRAINING_SCHEMA,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
)

__all__ = [
    "CANONICAL_COLUMN_ORDER",
    "CLASSIFICATION_LABEL_COLUMNS",
    "COLUMN_DTYPES",
    "FEATURE_COLUMNS",
    "LABEL_COLUMNS",
    "MERGED_TRAINING_SCHEMA",
    "PRIMARY_KEY_COLUMNS",
    "REGRESSION_LABEL_COLUMNS",
    "REQUIRED_COLUMNS",
    "canonical_column_order",
    "classification_label_columns",
    "column_dtypes",
    "feature_columns",
    "label_columns",
    "primary_key_columns",
    "regression_label_columns",
    "required_columns",
]

if LABEL_COLUMNS != (*REGRESSION_LABEL_COLUMNS, *CLASSIFICATION_LABEL_COLUMNS):
    raise RuntimeError(
        "Training LABEL_COLUMNS must equal regression then classification "
        "label columns from the Label schema."
    )


def primary_key_columns() -> tuple[str, ...]:
    """Return an immutable copy of the primary-key column names.

    Returns:
        Primary-key columns in canonical order.
    """
    return (*PRIMARY_KEY_COLUMNS,)


def feature_columns() -> tuple[str, ...]:
    """Return an immutable copy of the feature column names.

    Returns:
        Feature columns in canonical order.
    """
    return (*FEATURE_COLUMNS,)


def label_columns() -> tuple[str, ...]:
    """Return an immutable copy of the label column names.

    Returns:
        Label columns in canonical order.
    """
    return (*LABEL_COLUMNS,)


def regression_label_columns() -> tuple[str, ...]:
    """Return an immutable copy of the regression label column names.

    Returns:
        Regression label columns in canonical order.
    """
    return (*REGRESSION_LABEL_COLUMNS,)


def classification_label_columns() -> tuple[str, ...]:
    """Return an immutable copy of the classification label column names.

    Returns:
        Classification label columns in canonical order.
    """
    return (*CLASSIFICATION_LABEL_COLUMNS,)


def required_columns() -> tuple[str, ...]:
    """Return an immutable copy of the required column names.

    Returns:
        Required columns in canonical order.
    """
    return (*REQUIRED_COLUMNS,)


def canonical_column_order() -> tuple[str, ...]:
    """Return an immutable copy of the canonical column order.

    Returns:
        All merged-dataset columns in canonical order.
    """
    return (*CANONICAL_COLUMN_ORDER,)


def column_dtypes() -> Mapping[str, pl.DataType]:
    """Return an immutable mapping of column name to expected dtype.

    Returns:
        Read-only mapping covering every canonical column.
    """
    return MappingProxyType(dict(COLUMN_DTYPES))
