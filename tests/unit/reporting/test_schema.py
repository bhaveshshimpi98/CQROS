"""Unit tests for CQROS reporting dataset schema."""

from __future__ import annotations

import polars as pl

from cqros.reporting import REPORTING_COLUMNS, REPORTING_SCHEMA, ReportingStatus
from cqros.reporting.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    reporting_status_values,
    reporting_statuses,
)
from cqros.reporting.schema import (
    REPORTING_SCHEMA as REPORTING_SCHEMA_DIRECT,
)

_REPORT_COLUMNS: tuple[str, ...] = (
    "report_name",
    "report_type",
    "report_format",
    "report_version",
    "report_path",
    "generated_at",
)


def test_primary_key_and_required_columns() -> None:
    """Primary key and required columns match the canonical reporting contract."""
    assert PRIMARY_KEY_COLUMNS == ("symbol", "timeframe", "open_time")
    assert REQUIRED_COLUMNS == CANONICAL_COLUMN_ORDER
    assert CANONICAL_COLUMN_ORDER == REPORTING_COLUMNS


def test_required_columns_are_complete() -> None:
    """REQUIRED_COLUMNS covers every canonical reporting column exactly once."""
    assert set(REQUIRED_COLUMNS) == set(REPORTING_COLUMNS)
    assert len(REQUIRED_COLUMNS) == len(REPORTING_COLUMNS)


def test_reporting_columns_contain_required_domain_columns() -> None:
    """REPORTING_COLUMNS enumerates identity, report metadata, and status fields."""
    for column in (
        "symbol",
        "timeframe",
        "open_time",
        "manager",
        *_REPORT_COLUMNS,
        "status",
    ):
        assert column in REPORTING_COLUMNS


def test_canonical_column_order_has_no_duplicates() -> None:
    """Canonical reporting column order contains no duplicate names."""
    assert len(CANONICAL_COLUMN_ORDER) == len(set(CANONICAL_COLUMN_ORDER))


def test_column_dtypes_and_reporting_schema() -> None:
    """Reporting schema dtypes match COLUMN_DTYPES in canonical order."""
    assert REPORTING_SCHEMA is REPORTING_SCHEMA_DIRECT
    assert REPORTING_SCHEMA.names() == list(CANONICAL_COLUMN_ORDER)
    for column in CANONICAL_COLUMN_ORDER:
        assert REPORTING_SCHEMA[column] == COLUMN_DTYPES[column]
    assert COLUMN_DTYPES["symbol"] == pl.Utf8
    assert COLUMN_DTYPES["timeframe"] == pl.Utf8
    assert COLUMN_DTYPES["open_time"] == pl.Int64
    assert COLUMN_DTYPES["manager"] == pl.Utf8
    assert COLUMN_DTYPES["generated_at"] == pl.Int64
    assert COLUMN_DTYPES["status"] == pl.Utf8


def test_column_dtypes_cover_every_canonical_column() -> None:
    """COLUMN_DTYPES defines an entry for every canonical reporting column."""
    assert set(COLUMN_DTYPES.keys()) == set(CANONICAL_COLUMN_ORDER)


def test_canonical_order_starts_with_identity_keys() -> None:
    """Canonical column order begins with symbol/timeframe/open_time/manager."""
    assert CANONICAL_COLUMN_ORDER[0] == "symbol"
    assert CANONICAL_COLUMN_ORDER[1] == "timeframe"
    assert CANONICAL_COLUMN_ORDER[2] == "open_time"
    assert CANONICAL_COLUMN_ORDER[3] == "manager"
    assert CANONICAL_COLUMN_ORDER[-1] == "status"


def test_canonical_order_places_report_fields_before_status() -> None:
    """Report metadata fields appear in the declared order before status."""
    report_slice = CANONICAL_COLUMN_ORDER[4:-1]
    assert report_slice == _REPORT_COLUMNS


def test_reporting_status_enum_members() -> None:
    """ReportingStatus exposes GENERATED and FAILED members."""
    assert ReportingStatus.GENERATED.value == "GENERATED"
    assert ReportingStatus.FAILED.value == "FAILED"
    assert len(list(ReportingStatus)) == 2


def test_reporting_statuses_helper() -> None:
    """reporting_statuses() returns a tuple of all ReportingStatus members."""
    statuses = reporting_statuses()
    assert statuses == (ReportingStatus.GENERATED, ReportingStatus.FAILED)
    assert isinstance(statuses, tuple)


def test_reporting_status_values_helper() -> None:
    """reporting_status_values() returns valid GENERATED and FAILED strings."""
    status_values = reporting_status_values()
    assert status_values == ("GENERATED", "FAILED")
    assert isinstance(status_values, tuple)
    assert set(status_values) == {member.value for member in ReportingStatus}


def test_reporting_schema_has_eleven_columns() -> None:
    """Reporting schema defines exactly 11 canonical columns."""
    assert len(CANONICAL_COLUMN_ORDER) == 11
    assert len(REPORTING_SCHEMA) == 11


def test_report_string_columns_are_utf8() -> None:
    """Report metadata string columns use Utf8 dtype."""
    for column in (
        "report_name",
        "report_type",
        "report_format",
        "report_version",
        "report_path",
    ):
        assert COLUMN_DTYPES[column] == pl.Utf8
