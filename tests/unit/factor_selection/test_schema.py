"""Unit tests for CQROS factor selection metrics schema."""

from __future__ import annotations

import polars as pl

from cqros.factor_selection import (
    FACTOR_SELECTION_COLUMNS,
    FACTOR_SELECTION_SCHEMA,
    FactorSelectionStatus,
)
from cqros.factor_selection.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    factor_selection_status_values,
    factor_selection_statuses,
)
from cqros.factor_selection.schema import (
    FACTOR_SELECTION_SCHEMA as FACTOR_SELECTION_SCHEMA_DIRECT,
)

_SELECTION_DECISION_COLUMNS: tuple[str, ...] = (
    "selected",
    "selection_score",
    "selection_rank",
    "selection_reason",
    "selection_ic",
    "selected_direction",
    "orientation_policy",
)


def test_primary_key_and_required_columns() -> None:
    """Primary key and required columns match the canonical selection contract."""
    assert PRIMARY_KEY_COLUMNS == (
        "factor_name",
        "factor_version",
        "timeframe",
        "selection_time",
    )
    assert REQUIRED_COLUMNS == CANONICAL_COLUMN_ORDER
    assert CANONICAL_COLUMN_ORDER == FACTOR_SELECTION_COLUMNS


def test_required_columns_are_complete() -> None:
    """REQUIRED_COLUMNS covers every canonical selection column exactly once."""
    assert set(REQUIRED_COLUMNS) == set(FACTOR_SELECTION_COLUMNS)
    assert len(REQUIRED_COLUMNS) == len(FACTOR_SELECTION_COLUMNS)


def test_factor_selection_columns_contain_required_domain_columns() -> None:
    """FACTOR_SELECTION_COLUMNS enumerates identity, selection, and status fields."""
    for column in (
        "factor_name",
        "factor_version",
        "timeframe",
        "selection_time",
        "factor_category",
        *_SELECTION_DECISION_COLUMNS,
        "status",
    ):
        assert column in FACTOR_SELECTION_COLUMNS


def test_canonical_column_order_has_no_duplicates() -> None:
    """Canonical factor selection column order contains no duplicate names."""
    assert len(CANONICAL_COLUMN_ORDER) == len(set(CANONICAL_COLUMN_ORDER))


def test_column_dtypes_and_factor_selection_schema() -> None:
    """Factor selection schema dtypes match COLUMN_DTYPES in canonical order."""
    assert FACTOR_SELECTION_SCHEMA is FACTOR_SELECTION_SCHEMA_DIRECT
    assert FACTOR_SELECTION_SCHEMA.names() == list(CANONICAL_COLUMN_ORDER)
    for column in CANONICAL_COLUMN_ORDER:
        assert FACTOR_SELECTION_SCHEMA[column] == COLUMN_DTYPES[column]
    assert COLUMN_DTYPES["factor_name"] == pl.String
    assert COLUMN_DTYPES["factor_version"] == pl.String
    assert COLUMN_DTYPES["timeframe"] == pl.String
    assert COLUMN_DTYPES["selection_time"] == pl.Int64
    assert COLUMN_DTYPES["factor_category"] == pl.String
    assert COLUMN_DTYPES["selected"] == pl.Boolean
    assert COLUMN_DTYPES["selection_score"] == pl.Float64
    assert COLUMN_DTYPES["selection_rank"] == pl.Int32
    assert COLUMN_DTYPES["selection_reason"] == pl.String
    assert COLUMN_DTYPES["selection_ic"] == pl.Float64
    assert COLUMN_DTYPES["selected_direction"] == pl.Int8
    assert COLUMN_DTYPES["orientation_policy"] == pl.String
    assert COLUMN_DTYPES["status"] == pl.String


def test_column_dtypes_cover_every_canonical_column() -> None:
    """COLUMN_DTYPES defines an entry for every canonical selection column."""
    assert set(COLUMN_DTYPES.keys()) == set(CANONICAL_COLUMN_ORDER)


def test_canonical_order_starts_with_identity_keys() -> None:
    """Canonical column order begins with primary key, then metadata, then status."""
    assert CANONICAL_COLUMN_ORDER[0] == "factor_name"
    assert CANONICAL_COLUMN_ORDER[1] == "factor_version"
    assert CANONICAL_COLUMN_ORDER[2] == "timeframe"
    assert CANONICAL_COLUMN_ORDER[3] == "selection_time"
    assert CANONICAL_COLUMN_ORDER[4] == "factor_category"
    assert CANONICAL_COLUMN_ORDER[-1] == "status"


def test_canonical_order_places_selection_fields_before_status() -> None:
    """Selection decision columns appear in the declared order before status."""
    selection_slice = CANONICAL_COLUMN_ORDER[5:-1]
    assert selection_slice == _SELECTION_DECISION_COLUMNS


def test_primary_key_precedes_selection_columns() -> None:
    """Primary key columns occupy the leading positions of the schema."""
    assert FACTOR_SELECTION_SCHEMA.names()[:4] == list(PRIMARY_KEY_COLUMNS)
    assert CANONICAL_COLUMN_ORDER.index("factor_name") == 0
    assert CANONICAL_COLUMN_ORDER.index("factor_version") == 1
    assert CANONICAL_COLUMN_ORDER.index("timeframe") == 2
    assert CANONICAL_COLUMN_ORDER.index("selection_time") == 3


def test_factor_selection_status_enum_members() -> None:
    """FactorSelectionStatus exposes SELECTED and REJECTED members."""
    assert FactorSelectionStatus.SELECTED.value == "SELECTED"
    assert FactorSelectionStatus.REJECTED.value == "REJECTED"
    assert len(list(FactorSelectionStatus)) == 2


def test_factor_selection_statuses_helper() -> None:
    """factor_selection_statuses() returns a tuple of all status members."""
    statuses = factor_selection_statuses()
    assert statuses == (FactorSelectionStatus.SELECTED, FactorSelectionStatus.REJECTED)
    assert isinstance(statuses, tuple)


def test_factor_selection_status_values_helper() -> None:
    """factor_selection_status_values() returns valid SELECTED and REJECTED strings."""
    status_values = factor_selection_status_values()
    assert status_values == ("SELECTED", "REJECTED")
    assert isinstance(status_values, tuple)
    assert set(status_values) == {member.value for member in FactorSelectionStatus}


def test_factor_selection_schema_has_thirteen_columns() -> None:
    """Factor selection schema defines exactly 13 canonical columns."""
    assert len(CANONICAL_COLUMN_ORDER) == 13
    assert len(FACTOR_SELECTION_SCHEMA) == 13


def test_selection_decision_column_dtypes() -> None:
    """Selection decision columns use the declared Boolean/Float64/Int32/String dtypes."""
    assert COLUMN_DTYPES["selected"] == pl.Boolean
    assert COLUMN_DTYPES["selection_score"] == pl.Float64
    assert COLUMN_DTYPES["selection_rank"] == pl.Int32
    assert COLUMN_DTYPES["selection_reason"] == pl.String
    assert COLUMN_DTYPES["selection_ic"] == pl.Float64
    assert COLUMN_DTYPES["selected_direction"] == pl.Int8
    assert COLUMN_DTYPES["orientation_policy"] == pl.String
