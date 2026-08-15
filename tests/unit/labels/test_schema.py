"""Unit tests for the CQROS merged label dataset schema."""

from __future__ import annotations

import polars as pl

from cqros.labels import (
    CANONICAL_COLUMN_ORDER,
    CLASSIFICATION_LABEL_COLUMNS,
    COLUMN_DTYPES,
    LABEL_COLUMNS,
    MERGED_LABEL_SCHEMA,
    PRIMARY_KEY_COLUMNS,
    REGRESSION_LABEL_COLUMNS,
    REQUIRED_COLUMNS,
)
from cqros.labels.schema import (
    CANONICAL_COLUMN_ORDER as CANONICAL_COLUMN_ORDER_DIRECT,
)


def test_merged_label_schema_is_exported_from_package() -> None:
    """Package export matches the schema module constant."""
    assert CANONICAL_COLUMN_ORDER is CANONICAL_COLUMN_ORDER_DIRECT


def test_primary_key_columns() -> None:
    """Primary key is symbol, timeframe, open_time."""
    assert PRIMARY_KEY_COLUMNS == ("symbol", "timeframe", "open_time")


def test_canonical_column_order() -> None:
    """Canonical order is primary key, then regression, then classification."""
    assert CANONICAL_COLUMN_ORDER[:3] == PRIMARY_KEY_COLUMNS
    assert CANONICAL_COLUMN_ORDER[3:7] == REGRESSION_LABEL_COLUMNS
    assert CANONICAL_COLUMN_ORDER[7:] == CLASSIFICATION_LABEL_COLUMNS
    assert CANONICAL_COLUMN_ORDER == (
        "symbol",
        "timeframe",
        "open_time",
        "future_return_1",
        "future_return_5",
        "future_return_10",
        "future_return_20",
        "direction_1",
        "direction_5",
        "direction_10",
        "direction_20",
    )


def test_label_columns_match_regression_and_classification() -> None:
    """LABEL_COLUMNS concatenates regression then classification columns."""
    assert LABEL_COLUMNS == (
        *REGRESSION_LABEL_COLUMNS,
        *CLASSIFICATION_LABEL_COLUMNS,
    )
    assert CANONICAL_COLUMN_ORDER[3:] == LABEL_COLUMNS


def test_required_columns_match_canonical_order() -> None:
    """Required columns expose the full merged schema contract."""
    assert REQUIRED_COLUMNS == CANONICAL_COLUMN_ORDER


def test_column_names_are_unique() -> None:
    """Canonical column names contain no duplicates."""
    assert len(CANONICAL_COLUMN_ORDER) == len(set(CANONICAL_COLUMN_ORDER))
    assert len(LABEL_COLUMNS) == len(set(LABEL_COLUMNS))
    assert len(REGRESSION_LABEL_COLUMNS) == len(set(REGRESSION_LABEL_COLUMNS))
    assert len(CLASSIFICATION_LABEL_COLUMNS) == len(set(CLASSIFICATION_LABEL_COLUMNS))


def test_required_columns_are_complete() -> None:
    """Required columns include the primary key and every label column."""
    assert set(PRIMARY_KEY_COLUMNS).issubset(set(REQUIRED_COLUMNS))
    assert set(REGRESSION_LABEL_COLUMNS).issubset(set(REQUIRED_COLUMNS))
    assert set(CLASSIFICATION_LABEL_COLUMNS).issubset(set(REQUIRED_COLUMNS))
    assert len(REQUIRED_COLUMNS) == (
        len(PRIMARY_KEY_COLUMNS) + len(REGRESSION_LABEL_COLUMNS) + len(CLASSIFICATION_LABEL_COLUMNS)
    )


def test_column_dtypes_cover_canonical_order() -> None:
    """Expected dtypes are defined for every canonical column."""
    assert tuple(COLUMN_DTYPES) == CANONICAL_COLUMN_ORDER
    assert COLUMN_DTYPES["symbol"] == pl.String
    assert COLUMN_DTYPES["timeframe"] == pl.String
    assert COLUMN_DTYPES["open_time"] == pl.Int64


def test_regression_label_dtypes_are_float64() -> None:
    """Regression label columns use Float64."""
    for column in REGRESSION_LABEL_COLUMNS:
        assert COLUMN_DTYPES[column] == pl.Float64


def test_classification_label_dtypes_are_int8() -> None:
    """Classification (direction) label columns use Int8."""
    for column in CLASSIFICATION_LABEL_COLUMNS:
        assert COLUMN_DTYPES[column] == pl.Int8


def test_merged_label_schema_matches_canonical_order_and_dtypes() -> None:
    """Polars schema preserves canonical order and expected dtypes."""
    assert MERGED_LABEL_SCHEMA.names() == list(CANONICAL_COLUMN_ORDER)
    for column in CANONICAL_COLUMN_ORDER:
        assert MERGED_LABEL_SCHEMA[column] == COLUMN_DTYPES[column]


def test_primary_key_precedes_label_columns() -> None:
    """Primary key columns occupy the leading positions of the schema."""
    assert MERGED_LABEL_SCHEMA.names()[:3] == list(PRIMARY_KEY_COLUMNS)
    assert CANONICAL_COLUMN_ORDER.index("symbol") == 0
    assert CANONICAL_COLUMN_ORDER.index("timeframe") == 1
    assert CANONICAL_COLUMN_ORDER.index("open_time") == 2
