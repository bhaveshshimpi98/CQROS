"""Unit tests for the CQROS merged training dataset schema."""

from __future__ import annotations

import polars as pl

from cqros.features.schema import (
    COLUMN_DTYPES as FEATURE_COLUMN_DTYPES,
)
from cqros.features.schema import (
    FEATURE_COLUMNS as FEATURE_SCHEMA_FEATURE_COLUMNS,
)
from cqros.features.schema import (
    PRIMARY_KEY_COLUMNS as FEATURE_PRIMARY_KEY_COLUMNS,
)
from cqros.labels.schema import (
    COLUMN_DTYPES as LABEL_COLUMN_DTYPES,
)
from cqros.labels.schema import (
    LABEL_COLUMNS as LABEL_SCHEMA_LABEL_COLUMNS,
)
from cqros.labels.schema import (
    PRIMARY_KEY_COLUMNS as LABEL_PRIMARY_KEY_COLUMNS,
)
from cqros.training import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    MERGED_TRAINING_SCHEMA,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
)
from cqros.training.schema import (
    CANONICAL_COLUMN_ORDER as CANONICAL_COLUMN_ORDER_DIRECT,
)


def test_merged_training_schema_is_exported_from_package() -> None:
    """Package export matches the schema module constant."""
    assert CANONICAL_COLUMN_ORDER is CANONICAL_COLUMN_ORDER_DIRECT


def test_primary_key_columns() -> None:
    """Primary key is symbol, timeframe, open_time."""
    assert PRIMARY_KEY_COLUMNS == ("symbol", "timeframe", "open_time")


def test_primary_key_matches_feature_and_label_schemas() -> None:
    """Training primary key is identical to feature and label primary keys."""
    assert PRIMARY_KEY_COLUMNS == FEATURE_PRIMARY_KEY_COLUMNS
    assert PRIMARY_KEY_COLUMNS == LABEL_PRIMARY_KEY_COLUMNS
    assert FEATURE_PRIMARY_KEY_COLUMNS == LABEL_PRIMARY_KEY_COLUMNS


def test_feature_columns_imported_from_feature_schema() -> None:
    """FEATURE_COLUMNS is the feature schema constant, not a local copy."""
    assert FEATURE_COLUMNS is FEATURE_SCHEMA_FEATURE_COLUMNS


def test_label_columns_imported_from_label_schema() -> None:
    """LABEL_COLUMNS is the label schema constant, not a local copy."""
    assert LABEL_COLUMNS is LABEL_SCHEMA_LABEL_COLUMNS


def test_canonical_column_order() -> None:
    """Canonical order is primary key, then features, then labels."""
    feature_end = len(PRIMARY_KEY_COLUMNS) + len(FEATURE_COLUMNS)
    assert CANONICAL_COLUMN_ORDER[: len(PRIMARY_KEY_COLUMNS)] == PRIMARY_KEY_COLUMNS
    assert CANONICAL_COLUMN_ORDER[len(PRIMARY_KEY_COLUMNS) : feature_end] == FEATURE_COLUMNS
    assert CANONICAL_COLUMN_ORDER[feature_end:] == LABEL_COLUMNS
    assert CANONICAL_COLUMN_ORDER == (
        *PRIMARY_KEY_COLUMNS,
        *FEATURE_COLUMNS,
        *LABEL_COLUMNS,
    )


def test_required_columns_match_canonical_order() -> None:
    """Required columns expose the full merged schema contract."""
    assert REQUIRED_COLUMNS == CANONICAL_COLUMN_ORDER


def test_required_columns_are_complete() -> None:
    """Required columns include the primary key and every feature/label column."""
    assert set(PRIMARY_KEY_COLUMNS).issubset(set(REQUIRED_COLUMNS))
    assert set(FEATURE_COLUMNS).issubset(set(REQUIRED_COLUMNS))
    assert set(LABEL_COLUMNS).issubset(set(REQUIRED_COLUMNS))
    assert len(REQUIRED_COLUMNS) == (
        len(PRIMARY_KEY_COLUMNS) + len(FEATURE_COLUMNS) + len(LABEL_COLUMNS)
    )


def test_column_names_are_unique() -> None:
    """Canonical column names contain no duplicates."""
    assert len(CANONICAL_COLUMN_ORDER) == len(set(CANONICAL_COLUMN_ORDER))
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))
    assert len(LABEL_COLUMNS) == len(set(LABEL_COLUMNS))
    assert set(FEATURE_COLUMNS).isdisjoint(set(LABEL_COLUMNS))
    assert set(PRIMARY_KEY_COLUMNS).isdisjoint(set(FEATURE_COLUMNS))
    assert set(PRIMARY_KEY_COLUMNS).isdisjoint(set(LABEL_COLUMNS))


def test_column_dtypes_cover_canonical_order() -> None:
    """Expected dtypes are defined for every canonical column."""
    assert tuple(COLUMN_DTYPES) == CANONICAL_COLUMN_ORDER
    assert COLUMN_DTYPES["symbol"] == pl.String
    assert COLUMN_DTYPES["timeframe"] == pl.String
    assert COLUMN_DTYPES["open_time"] == pl.Int64


def test_primary_key_dtypes_match_source_schemas() -> None:
    """Primary-key dtypes remain identical to feature and label schemas."""
    for column in PRIMARY_KEY_COLUMNS:
        assert COLUMN_DTYPES[column] == FEATURE_COLUMN_DTYPES[column]
        assert COLUMN_DTYPES[column] == LABEL_COLUMN_DTYPES[column]


def test_feature_dtypes_reused_from_feature_schema() -> None:
    """Feature column dtypes are taken from the feature schema."""
    for column in FEATURE_COLUMNS:
        assert COLUMN_DTYPES[column] == FEATURE_COLUMN_DTYPES[column]


def test_label_dtypes_reused_from_label_schema() -> None:
    """Label column dtypes are taken from the label schema."""
    for column in LABEL_COLUMNS:
        assert COLUMN_DTYPES[column] == LABEL_COLUMN_DTYPES[column]


def test_no_dtype_conflicts_between_feature_and_label_schemas() -> None:
    """Shared primary-key columns have matching dtypes across source schemas."""
    for column in PRIMARY_KEY_COLUMNS:
        assert FEATURE_COLUMN_DTYPES[column] == LABEL_COLUMN_DTYPES[column]


def test_merged_training_schema_matches_canonical_order_and_dtypes() -> None:
    """Polars schema preserves canonical order and expected dtypes."""
    assert MERGED_TRAINING_SCHEMA.names() == list(CANONICAL_COLUMN_ORDER)
    for column in CANONICAL_COLUMN_ORDER:
        assert MERGED_TRAINING_SCHEMA[column] == COLUMN_DTYPES[column]


def test_primary_key_precedes_feature_and_label_columns() -> None:
    """Primary key columns occupy the leading positions of the schema."""
    assert MERGED_TRAINING_SCHEMA.names()[:3] == list(PRIMARY_KEY_COLUMNS)
    assert CANONICAL_COLUMN_ORDER.index("symbol") == 0
    assert CANONICAL_COLUMN_ORDER.index("timeframe") == 1
    assert CANONICAL_COLUMN_ORDER.index("open_time") == 2


def test_feature_and_label_columns_are_included() -> None:
    """Every feature and label column appears in the merged schema."""
    assert set(FEATURE_COLUMNS).issubset(set(CANONICAL_COLUMN_ORDER))
    assert set(LABEL_COLUMNS).issubset(set(CANONICAL_COLUMN_ORDER))
    for column in FEATURE_COLUMNS:
        assert column in MERGED_TRAINING_SCHEMA
    for column in LABEL_COLUMNS:
        assert column in MERGED_TRAINING_SCHEMA
