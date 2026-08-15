"""Unit tests for the CQROS merged order dataset schema."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import polars as pl
import pytest

from cqros.oms import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    MERGED_ORDER_SCHEMA,
    METADATA_COLUMNS,
    ORDER_COLUMNS,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
)
from cqros.oms.schema import (
    CANONICAL_COLUMN_ORDER as CANONICAL_COLUMN_ORDER_DIRECT,
)
from cqros.oms.schema import COLUMN_DTYPES as COLUMN_DTYPES_DIRECT
from cqros.oms.schema import MERGED_ORDER_SCHEMA as MERGED_ORDER_SCHEMA_DIRECT
from cqros.oms.schema import METADATA_COLUMNS as METADATA_COLUMNS_DIRECT
from cqros.oms.schema import ORDER_COLUMNS as ORDER_COLUMNS_DIRECT
from cqros.oms.schema import PRIMARY_KEY_COLUMNS as PRIMARY_KEY_COLUMNS_DIRECT
from cqros.oms.schema import REQUIRED_COLUMNS as REQUIRED_COLUMNS_DIRECT


def test_merged_order_schema_is_exported_from_package() -> None:
    """Package exports match the schema module constants by identity."""
    assert CANONICAL_COLUMN_ORDER is CANONICAL_COLUMN_ORDER_DIRECT
    assert COLUMN_DTYPES is COLUMN_DTYPES_DIRECT
    assert MERGED_ORDER_SCHEMA is MERGED_ORDER_SCHEMA_DIRECT
    assert METADATA_COLUMNS is METADATA_COLUMNS_DIRECT
    assert ORDER_COLUMNS is ORDER_COLUMNS_DIRECT
    assert PRIMARY_KEY_COLUMNS is PRIMARY_KEY_COLUMNS_DIRECT
    assert REQUIRED_COLUMNS is REQUIRED_COLUMNS_DIRECT


def test_primary_key_columns() -> None:
    """Primary key is symbol, timeframe, open_time, order_id."""
    assert PRIMARY_KEY_COLUMNS == (
        "symbol",
        "timeframe",
        "open_time",
        "order_id",
    )


def test_metadata_and_order_columns() -> None:
    """Metadata and order partitions expose the expected columns."""
    assert METADATA_COLUMNS == (
        "model_name",
        "model_version",
        "policy",
        "optimizer",
    )
    assert ORDER_COLUMNS == (
        "parent_order_id",
        "side",
        "order_type",
        "quantity",
        "limit_price",
        "stop_price",
        "filled_quantity",
        "average_fill_price",
        "status",
        "created_at",
        "updated_at",
    )


def test_canonical_column_order() -> None:
    """Canonical order places parent_order_id before metadata columns."""
    assert CANONICAL_COLUMN_ORDER[:4] == PRIMARY_KEY_COLUMNS
    assert CANONICAL_COLUMN_ORDER[4] == "parent_order_id"
    assert CANONICAL_COLUMN_ORDER[5:9] == METADATA_COLUMNS
    assert CANONICAL_COLUMN_ORDER[9:] == (
        "side",
        "order_type",
        "quantity",
        "limit_price",
        "stop_price",
        "filled_quantity",
        "average_fill_price",
        "status",
        "created_at",
        "updated_at",
    )
    assert CANONICAL_COLUMN_ORDER == (
        "symbol",
        "timeframe",
        "open_time",
        "order_id",
        "parent_order_id",
        "model_name",
        "model_version",
        "policy",
        "optimizer",
        "side",
        "order_type",
        "quantity",
        "limit_price",
        "stop_price",
        "filled_quantity",
        "average_fill_price",
        "status",
        "created_at",
        "updated_at",
    )


def test_required_columns_match_canonical_order() -> None:
    """Required columns expose the full merged schema contract."""
    assert REQUIRED_COLUMNS == CANONICAL_COLUMN_ORDER
    assert REQUIRED_COLUMNS is CANONICAL_COLUMN_ORDER


def test_required_columns_are_complete() -> None:
    """Required columns include the primary key and every order column."""
    assert set(PRIMARY_KEY_COLUMNS).issubset(set(REQUIRED_COLUMNS))
    assert set(METADATA_COLUMNS).issubset(set(REQUIRED_COLUMNS))
    assert set(ORDER_COLUMNS).issubset(set(REQUIRED_COLUMNS))
    assert len(REQUIRED_COLUMNS) == (
        len(PRIMARY_KEY_COLUMNS) + len(METADATA_COLUMNS) + len(ORDER_COLUMNS)
    )


def test_column_names_are_unique() -> None:
    """Canonical column names contain no duplicates."""
    assert len(CANONICAL_COLUMN_ORDER) == len(set(CANONICAL_COLUMN_ORDER))
    assert len(METADATA_COLUMNS) == len(set(METADATA_COLUMNS))
    assert len(ORDER_COLUMNS) == len(set(ORDER_COLUMNS))
    assert set(PRIMARY_KEY_COLUMNS).isdisjoint(set(METADATA_COLUMNS))
    assert set(PRIMARY_KEY_COLUMNS).isdisjoint(set(ORDER_COLUMNS))
    assert set(METADATA_COLUMNS).isdisjoint(set(ORDER_COLUMNS))


def test_column_dtypes_cover_canonical_order() -> None:
    """Expected dtypes are defined for every canonical column."""
    assert tuple(COLUMN_DTYPES) == CANONICAL_COLUMN_ORDER
    assert COLUMN_DTYPES["symbol"] == pl.Utf8
    assert COLUMN_DTYPES["timeframe"] == pl.Utf8
    assert COLUMN_DTYPES["open_time"] == pl.Datetime("us", "UTC")
    assert COLUMN_DTYPES["order_id"] == pl.Utf8
    assert COLUMN_DTYPES["parent_order_id"] == pl.Utf8
    assert COLUMN_DTYPES["model_name"] == pl.Utf8
    assert COLUMN_DTYPES["model_version"] == pl.Utf8
    assert COLUMN_DTYPES["policy"] == pl.Utf8
    assert COLUMN_DTYPES["optimizer"] == pl.Utf8
    assert COLUMN_DTYPES["side"] == pl.Utf8
    assert COLUMN_DTYPES["order_type"] == pl.Utf8
    assert COLUMN_DTYPES["quantity"] == pl.Float64
    assert COLUMN_DTYPES["limit_price"] == pl.Float64
    assert COLUMN_DTYPES["stop_price"] == pl.Float64
    assert COLUMN_DTYPES["filled_quantity"] == pl.Float64
    assert COLUMN_DTYPES["average_fill_price"] == pl.Float64
    assert COLUMN_DTYPES["status"] == pl.Utf8
    assert COLUMN_DTYPES["created_at"] == pl.Datetime("us", "UTC")
    assert COLUMN_DTYPES["updated_at"] == pl.Datetime("us", "UTC")


def test_merged_order_schema_matches_canonical_order_and_dtypes() -> None:
    """Polars schema preserves canonical order and expected dtypes."""
    assert MERGED_ORDER_SCHEMA.names() == list(CANONICAL_COLUMN_ORDER)
    for column in CANONICAL_COLUMN_ORDER:
        assert MERGED_ORDER_SCHEMA[column] == COLUMN_DTYPES[column]


def test_schema_identity() -> None:
    """Merged schema identity is stable across package and module imports."""
    assert MERGED_ORDER_SCHEMA is MERGED_ORDER_SCHEMA_DIRECT
    assert MERGED_ORDER_SCHEMA.names() == [
        "symbol",
        "timeframe",
        "open_time",
        "order_id",
        "parent_order_id",
        "model_name",
        "model_version",
        "policy",
        "optimizer",
        "side",
        "order_type",
        "quantity",
        "limit_price",
        "stop_price",
        "filled_quantity",
        "average_fill_price",
        "status",
        "created_at",
        "updated_at",
    ]


def test_primary_key_precedes_order_columns() -> None:
    """Primary key columns occupy the leading positions of the schema."""
    assert MERGED_ORDER_SCHEMA.names()[:4] == list(PRIMARY_KEY_COLUMNS)
    assert CANONICAL_COLUMN_ORDER.index("symbol") == 0
    assert CANONICAL_COLUMN_ORDER.index("timeframe") == 1
    assert CANONICAL_COLUMN_ORDER.index("open_time") == 2
    assert CANONICAL_COLUMN_ORDER.index("order_id") == 3


def test_exported_collections_are_immutable() -> None:
    """Column tuples and dtype mapping are immutable exports."""
    assert isinstance(PRIMARY_KEY_COLUMNS, tuple)
    assert isinstance(METADATA_COLUMNS, tuple)
    assert isinstance(ORDER_COLUMNS, tuple)
    assert isinstance(REQUIRED_COLUMNS, tuple)
    assert isinstance(CANONICAL_COLUMN_ORDER, tuple)
    assert isinstance(COLUMN_DTYPES, Mapping)
    assert isinstance(COLUMN_DTYPES, MappingProxyType)

    with pytest.raises(TypeError):
        COLUMN_DTYPES["symbol"] = pl.Int64  # type: ignore[index]

    with pytest.raises(TypeError):
        PRIMARY_KEY_COLUMNS[0] = "asset"  # type: ignore[index]

    with pytest.raises(TypeError):
        CANONICAL_COLUMN_ORDER[0] = "asset"  # type: ignore[index]
