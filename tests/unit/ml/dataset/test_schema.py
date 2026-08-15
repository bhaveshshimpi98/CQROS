"""Unit tests for the CQROS ML Dataset schema."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import polars as pl
import pytest

from cqros.core.exceptions import DatasetError
from cqros.labels.schema import (
    CLASSIFICATION_LABEL_COLUMNS as LABEL_CLASSIFICATION_LABEL_COLUMNS,
)
from cqros.labels.schema import (
    REGRESSION_LABEL_COLUMNS as LABEL_REGRESSION_LABEL_COLUMNS,
)
from cqros.ml.dataset import (
    CANONICAL_COLUMN_ORDER,
    CLASSIFICATION_LABEL_COLUMNS,
    COLUMN_DTYPES,
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    MERGED_TRAINING_SCHEMA,
    PRIMARY_KEY_COLUMNS,
    REGRESSION_LABEL_COLUMNS,
    REQUIRED_COLUMNS,
    DatasetSchemaError,
    canonical_column_order,
    classification_label_columns,
    column_dtypes,
    feature_columns,
    label_columns,
    primary_key_columns,
    regression_label_columns,
    required_columns,
)
from cqros.ml.dataset.schema import (
    CANONICAL_COLUMN_ORDER as CANONICAL_COLUMN_ORDER_DIRECT,
)
from cqros.training.schema import (
    CANONICAL_COLUMN_ORDER as TRAINING_CANONICAL_COLUMN_ORDER,
)
from cqros.training.schema import COLUMN_DTYPES as TRAINING_COLUMN_DTYPES
from cqros.training.schema import FEATURE_COLUMNS as TRAINING_FEATURE_COLUMNS
from cqros.training.schema import LABEL_COLUMNS as TRAINING_LABEL_COLUMNS
from cqros.training.schema import (
    MERGED_TRAINING_SCHEMA as TRAINING_MERGED_TRAINING_SCHEMA,
)
from cqros.training.schema import PRIMARY_KEY_COLUMNS as TRAINING_PRIMARY_KEY_COLUMNS
from cqros.training.schema import REQUIRED_COLUMNS as TRAINING_REQUIRED_COLUMNS


def test_schema_is_exported_from_package() -> None:
    """Package export matches the schema module constant."""
    assert CANONICAL_COLUMN_ORDER is CANONICAL_COLUMN_ORDER_DIRECT


def test_exported_constants_match_training_schema() -> None:
    """ML Dataset constants are the Training schema objects, not copies."""
    assert PRIMARY_KEY_COLUMNS is TRAINING_PRIMARY_KEY_COLUMNS
    assert FEATURE_COLUMNS is TRAINING_FEATURE_COLUMNS
    assert LABEL_COLUMNS is TRAINING_LABEL_COLUMNS
    assert REQUIRED_COLUMNS is TRAINING_REQUIRED_COLUMNS
    assert CANONICAL_COLUMN_ORDER is TRAINING_CANONICAL_COLUMN_ORDER
    assert COLUMN_DTYPES is TRAINING_COLUMN_DTYPES
    assert MERGED_TRAINING_SCHEMA is TRAINING_MERGED_TRAINING_SCHEMA


def test_label_partition_constants_match_label_schema() -> None:
    """Regression and classification partitions reuse the Label schema."""
    assert REGRESSION_LABEL_COLUMNS is LABEL_REGRESSION_LABEL_COLUMNS
    assert CLASSIFICATION_LABEL_COLUMNS is LABEL_CLASSIFICATION_LABEL_COLUMNS


def test_merged_training_schema_identity() -> None:
    """MERGED_TRAINING_SCHEMA is the Training schema instance."""
    assert MERGED_TRAINING_SCHEMA is TRAINING_MERGED_TRAINING_SCHEMA


def test_primary_key_columns() -> None:
    """Primary key is symbol, timeframe, open_time."""
    assert PRIMARY_KEY_COLUMNS == ("symbol", "timeframe", "open_time")
    assert primary_key_columns() == PRIMARY_KEY_COLUMNS


def test_feature_columns_helper() -> None:
    """feature_columns returns the Training feature column names."""
    assert feature_columns() == FEATURE_COLUMNS
    assert feature_columns() == TRAINING_FEATURE_COLUMNS


def test_label_columns_helper() -> None:
    """label_columns returns the Training label column names."""
    assert label_columns() == LABEL_COLUMNS
    assert label_columns() == (
        *REGRESSION_LABEL_COLUMNS,
        *CLASSIFICATION_LABEL_COLUMNS,
    )


def test_regression_and_classification_label_helpers() -> None:
    """Regression and classification helpers preserve Label schema order."""
    assert regression_label_columns() == REGRESSION_LABEL_COLUMNS
    assert classification_label_columns() == CLASSIFICATION_LABEL_COLUMNS


def test_canonical_column_order() -> None:
    """Canonical order is primary key, then features, then labels."""
    feature_end = len(PRIMARY_KEY_COLUMNS) + len(FEATURE_COLUMNS)
    assert CANONICAL_COLUMN_ORDER[: len(PRIMARY_KEY_COLUMNS)] == PRIMARY_KEY_COLUMNS
    assert CANONICAL_COLUMN_ORDER[len(PRIMARY_KEY_COLUMNS) : feature_end] == FEATURE_COLUMNS
    assert CANONICAL_COLUMN_ORDER[feature_end:] == LABEL_COLUMNS
    assert canonical_column_order() == (
        *PRIMARY_KEY_COLUMNS,
        *FEATURE_COLUMNS,
        *LABEL_COLUMNS,
    )


def test_required_columns_match_canonical_order() -> None:
    """Required columns expose the full merged schema contract."""
    assert REQUIRED_COLUMNS == CANONICAL_COLUMN_ORDER
    assert required_columns() == canonical_column_order()


def test_column_names_are_unique() -> None:
    """Canonical column names contain no duplicates."""
    assert len(CANONICAL_COLUMN_ORDER) == len(set(CANONICAL_COLUMN_ORDER))
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))
    assert len(LABEL_COLUMNS) == len(set(LABEL_COLUMNS))
    assert len(REGRESSION_LABEL_COLUMNS) == len(set(REGRESSION_LABEL_COLUMNS))
    assert len(CLASSIFICATION_LABEL_COLUMNS) == len(set(CLASSIFICATION_LABEL_COLUMNS))
    assert set(FEATURE_COLUMNS).isdisjoint(set(LABEL_COLUMNS))
    assert set(PRIMARY_KEY_COLUMNS).isdisjoint(set(FEATURE_COLUMNS))
    assert set(PRIMARY_KEY_COLUMNS).isdisjoint(set(LABEL_COLUMNS))
    assert set(REGRESSION_LABEL_COLUMNS).isdisjoint(set(CLASSIFICATION_LABEL_COLUMNS))


def test_dtype_mapping_matches_training_schema() -> None:
    """Dtype mapping matches the Training schema for every column."""
    assert COLUMN_DTYPES is TRAINING_COLUMN_DTYPES
    assert tuple(COLUMN_DTYPES) == CANONICAL_COLUMN_ORDER
    for column in CANONICAL_COLUMN_ORDER:
        assert COLUMN_DTYPES[column] == TRAINING_COLUMN_DTYPES[column]
        assert column_dtypes()[column] == TRAINING_COLUMN_DTYPES[column]


def test_merged_training_schema_matches_canonical_order_and_dtypes() -> None:
    """Polars schema preserves canonical order and expected dtypes."""
    assert MERGED_TRAINING_SCHEMA.names() == list(CANONICAL_COLUMN_ORDER)
    for column in CANONICAL_COLUMN_ORDER:
        assert MERGED_TRAINING_SCHEMA[column] == COLUMN_DTYPES[column]


def test_helper_functions_return_expected_values() -> None:
    """Helpers return the same values as the exported constants."""
    assert primary_key_columns() == PRIMARY_KEY_COLUMNS
    assert feature_columns() == FEATURE_COLUMNS
    assert label_columns() == LABEL_COLUMNS
    assert regression_label_columns() == REGRESSION_LABEL_COLUMNS
    assert classification_label_columns() == CLASSIFICATION_LABEL_COLUMNS
    assert required_columns() == REQUIRED_COLUMNS
    assert canonical_column_order() == CANONICAL_COLUMN_ORDER
    assert dict(column_dtypes()) == dict(COLUMN_DTYPES)


def test_helper_functions_return_immutable_copies() -> None:
    """Helpers return independent immutable copies, not internal objects."""
    assert primary_key_columns() is not PRIMARY_KEY_COLUMNS
    assert feature_columns() is not FEATURE_COLUMNS
    assert label_columns() is not LABEL_COLUMNS
    assert regression_label_columns() is not REGRESSION_LABEL_COLUMNS
    assert classification_label_columns() is not CLASSIFICATION_LABEL_COLUMNS
    assert required_columns() is not REQUIRED_COLUMNS
    assert canonical_column_order() is not CANONICAL_COLUMN_ORDER

    dtypes = column_dtypes()
    assert dtypes is not COLUMN_DTYPES
    assert isinstance(dtypes, MappingProxyType)
    with pytest.raises(TypeError):
        dtypes["symbol"] = pl.Int64  # type: ignore[index]


def test_helper_return_types() -> None:
    """Helpers return tuples for column lists and a Mapping for dtypes."""
    assert isinstance(primary_key_columns(), tuple)
    assert isinstance(feature_columns(), tuple)
    assert isinstance(label_columns(), tuple)
    assert isinstance(regression_label_columns(), tuple)
    assert isinstance(classification_label_columns(), tuple)
    assert isinstance(required_columns(), tuple)
    assert isinstance(canonical_column_order(), tuple)
    assert isinstance(column_dtypes(), Mapping)
    assert all(isinstance(name, str) for name in canonical_column_order())
    for column, dtype in column_dtypes().items():
        assert isinstance(column, str)
        assert dtype == COLUMN_DTYPES[column]


def test_dataset_schema_error_inherits_dataset_error() -> None:
    """DatasetSchemaError is part of the DatasetError hierarchy."""
    assert issubclass(DatasetSchemaError, DatasetError)
    error = DatasetSchemaError("schema contract failed")
    assert isinstance(error, DatasetError)
    assert str(error) == "schema contract failed"
