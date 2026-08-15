"""Unit tests for the CQROS factor dataset schema."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import polars as pl
import pytest

from cqros.factors import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    FACTOR_COLUMNS,
    FACTOR_SCHEMA,
    METADATA_COLUMNS,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    FactorStatus,
    factor_status_values,
    factor_statuses,
)
from cqros.factors.schema import (
    CANONICAL_COLUMN_ORDER as CANONICAL_COLUMN_ORDER_DIRECT,
)
from cqros.factors.schema import COLUMN_DTYPES as COLUMN_DTYPES_DIRECT
from cqros.factors.schema import FACTOR_COLUMNS as FACTOR_COLUMNS_DIRECT
from cqros.factors.schema import FACTOR_SCHEMA as FACTOR_SCHEMA_DIRECT
from cqros.factors.schema import METADATA_COLUMNS as METADATA_COLUMNS_DIRECT
from cqros.factors.schema import (
    PRIMARY_KEY_COLUMNS as PRIMARY_KEY_COLUMNS_DIRECT,
)
from cqros.factors.schema import REQUIRED_COLUMNS as REQUIRED_COLUMNS_DIRECT


def test_factor_schema_is_exported_from_package() -> None:
    """Package exports match the schema module constants by identity."""
    assert CANONICAL_COLUMN_ORDER is CANONICAL_COLUMN_ORDER_DIRECT
    assert COLUMN_DTYPES is COLUMN_DTYPES_DIRECT
    assert FACTOR_COLUMNS is FACTOR_COLUMNS_DIRECT
    assert FACTOR_SCHEMA is FACTOR_SCHEMA_DIRECT
    assert METADATA_COLUMNS is METADATA_COLUMNS_DIRECT
    assert PRIMARY_KEY_COLUMNS is PRIMARY_KEY_COLUMNS_DIRECT
    assert REQUIRED_COLUMNS is REQUIRED_COLUMNS_DIRECT


def test_primary_key_columns() -> None:
    """Primary key is symbol, timeframe, open_time."""
    assert PRIMARY_KEY_COLUMNS == ("symbol", "timeframe", "open_time")


def test_metadata_and_factor_columns() -> None:
    """Metadata and factor partitions expose the expected columns."""
    assert METADATA_COLUMNS == (
        "factor_name",
        "factor_version",
        "factor_category",
        "factor_group",
    )
    assert FACTOR_COLUMNS == (
        "factor_value",
        "lookback",
        "prediction_horizon",
        "enabled",
        "status",
    )


def test_canonical_column_order() -> None:
    """Canonical order is primary key, metadata, then factor fields."""
    assert CANONICAL_COLUMN_ORDER[:3] == PRIMARY_KEY_COLUMNS
    assert CANONICAL_COLUMN_ORDER[3:7] == METADATA_COLUMNS
    assert CANONICAL_COLUMN_ORDER[7:] == FACTOR_COLUMNS
    assert CANONICAL_COLUMN_ORDER == (
        "symbol",
        "timeframe",
        "open_time",
        "factor_name",
        "factor_version",
        "factor_category",
        "factor_group",
        "factor_value",
        "lookback",
        "prediction_horizon",
        "enabled",
        "status",
    )


def test_required_columns_match_canonical_order() -> None:
    """Required columns expose the full factor schema contract."""
    assert REQUIRED_COLUMNS == CANONICAL_COLUMN_ORDER
    assert REQUIRED_COLUMNS is CANONICAL_COLUMN_ORDER


def test_required_columns_are_complete() -> None:
    """Required columns include the primary key and every factor column."""
    assert set(PRIMARY_KEY_COLUMNS).issubset(set(REQUIRED_COLUMNS))
    assert set(METADATA_COLUMNS).issubset(set(REQUIRED_COLUMNS))
    assert set(FACTOR_COLUMNS).issubset(set(REQUIRED_COLUMNS))
    assert len(REQUIRED_COLUMNS) == (
        len(PRIMARY_KEY_COLUMNS) + len(METADATA_COLUMNS) + len(FACTOR_COLUMNS)
    )


def test_column_names_are_unique() -> None:
    """Canonical column names contain no duplicates."""
    assert len(CANONICAL_COLUMN_ORDER) == len(set(CANONICAL_COLUMN_ORDER))
    assert len(METADATA_COLUMNS) == len(set(METADATA_COLUMNS))
    assert len(FACTOR_COLUMNS) == len(set(FACTOR_COLUMNS))
    assert set(PRIMARY_KEY_COLUMNS).isdisjoint(set(METADATA_COLUMNS))
    assert set(PRIMARY_KEY_COLUMNS).isdisjoint(set(FACTOR_COLUMNS))
    assert set(METADATA_COLUMNS).isdisjoint(set(FACTOR_COLUMNS))


def test_column_dtypes_cover_canonical_order() -> None:
    """Expected dtypes are defined for every canonical column."""
    assert tuple(COLUMN_DTYPES) == CANONICAL_COLUMN_ORDER
    assert COLUMN_DTYPES["symbol"] == pl.String
    assert COLUMN_DTYPES["timeframe"] == pl.String
    assert COLUMN_DTYPES["open_time"] == pl.Int64
    assert COLUMN_DTYPES["factor_name"] == pl.String
    assert COLUMN_DTYPES["factor_version"] == pl.String
    assert COLUMN_DTYPES["factor_category"] == pl.String
    assert COLUMN_DTYPES["factor_group"] == pl.String
    assert COLUMN_DTYPES["factor_value"] == pl.Float64
    assert COLUMN_DTYPES["lookback"] == pl.Int32
    assert COLUMN_DTYPES["prediction_horizon"] == pl.Int32
    assert COLUMN_DTYPES["enabled"] == pl.Boolean
    assert COLUMN_DTYPES["status"] == pl.String


def test_factor_schema_matches_canonical_order_and_dtypes() -> None:
    """Polars schema preserves canonical order and expected dtypes."""
    assert FACTOR_SCHEMA.names() == list(CANONICAL_COLUMN_ORDER)
    for column in CANONICAL_COLUMN_ORDER:
        assert FACTOR_SCHEMA[column] == COLUMN_DTYPES[column]


def test_schema_consistency() -> None:
    """Factor schema identity is stable across package and module imports."""
    assert FACTOR_SCHEMA is FACTOR_SCHEMA_DIRECT
    assert FACTOR_SCHEMA.names() == [
        "symbol",
        "timeframe",
        "open_time",
        "factor_name",
        "factor_version",
        "factor_category",
        "factor_group",
        "factor_value",
        "lookback",
        "prediction_horizon",
        "enabled",
        "status",
    ]
    assert len(CANONICAL_COLUMN_ORDER) == 12
    assert len(FACTOR_SCHEMA) == 12


def test_primary_key_precedes_factor_columns() -> None:
    """Primary key columns occupy the leading positions of the schema."""
    assert FACTOR_SCHEMA.names()[:3] == list(PRIMARY_KEY_COLUMNS)
    assert CANONICAL_COLUMN_ORDER.index("symbol") == 0
    assert CANONICAL_COLUMN_ORDER.index("timeframe") == 1
    assert CANONICAL_COLUMN_ORDER.index("open_time") == 2


def test_factor_status_enum_members() -> None:
    """FactorStatus exposes ACTIVE and DEPRECATED members."""
    assert FactorStatus.ACTIVE.value == "ACTIVE"
    assert FactorStatus.DEPRECATED.value == "DEPRECATED"
    assert len(list(FactorStatus)) == 2


def test_factor_statuses_helper() -> None:
    """factor_statuses() returns a tuple of all FactorStatus members."""
    statuses = factor_statuses()
    assert statuses == (FactorStatus.ACTIVE, FactorStatus.DEPRECATED)
    assert isinstance(statuses, tuple)


def test_factor_status_values_helper() -> None:
    """factor_status_values() returns valid ACTIVE and DEPRECATED strings."""
    status_values = factor_status_values()
    assert status_values == ("ACTIVE", "DEPRECATED")
    assert isinstance(status_values, tuple)
    assert set(status_values) == {member.value for member in FactorStatus}


def test_exported_collections_are_immutable() -> None:
    """Column tuples and dtype mapping are immutable exports."""
    assert isinstance(PRIMARY_KEY_COLUMNS, tuple)
    assert isinstance(METADATA_COLUMNS, tuple)
    assert isinstance(FACTOR_COLUMNS, tuple)
    assert isinstance(REQUIRED_COLUMNS, tuple)
    assert isinstance(CANONICAL_COLUMN_ORDER, tuple)
    assert isinstance(COLUMN_DTYPES, Mapping)
    assert isinstance(COLUMN_DTYPES, MappingProxyType)

    with pytest.raises(TypeError):
        COLUMN_DTYPES["symbol"] = pl.Int64  # type: ignore[index]

    with pytest.raises(TypeError):
        CANONICAL_COLUMN_ORDER[0] = "changed"  # type: ignore[index]
