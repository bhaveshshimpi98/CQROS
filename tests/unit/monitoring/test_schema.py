"""Unit tests for CQROS monitoring dataset schema."""

from __future__ import annotations

import polars as pl

from cqros.monitoring import MONITORING_COLUMNS, MONITORING_SCHEMA, MonitoringStatus
from cqros.monitoring.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    monitoring_status_values,
    monitoring_statuses,
)
from cqros.monitoring.schema import (
    MONITORING_SCHEMA as MONITORING_SCHEMA_DIRECT,
)

_MONITOR_COLUMNS: tuple[str, ...] = (
    "monitor_type",
    "monitor_name",
    "severity",
    "metric_name",
    "metric_value",
    "threshold",
    "alert",
)


def test_primary_key_and_required_columns() -> None:
    """Primary key and required columns match the canonical monitoring contract."""
    assert PRIMARY_KEY_COLUMNS == ("symbol", "timeframe", "open_time")
    assert REQUIRED_COLUMNS == CANONICAL_COLUMN_ORDER
    assert CANONICAL_COLUMN_ORDER == MONITORING_COLUMNS


def test_required_columns_are_complete() -> None:
    """REQUIRED_COLUMNS covers every canonical monitoring column exactly once."""
    assert set(REQUIRED_COLUMNS) == set(MONITORING_COLUMNS)
    assert len(REQUIRED_COLUMNS) == len(MONITORING_COLUMNS)


def test_monitoring_columns_contain_required_domain_columns() -> None:
    """MONITORING_COLUMNS enumerates identity, monitor metadata, and status fields."""
    for column in (
        "symbol",
        "timeframe",
        "open_time",
        "manager",
        *_MONITOR_COLUMNS,
        "status",
    ):
        assert column in MONITORING_COLUMNS


def test_canonical_column_order_has_no_duplicates() -> None:
    """Canonical monitoring column order contains no duplicate names."""
    assert len(CANONICAL_COLUMN_ORDER) == len(set(CANONICAL_COLUMN_ORDER))


def test_column_dtypes_and_monitoring_schema() -> None:
    """Monitoring schema dtypes match COLUMN_DTYPES in canonical order."""
    assert MONITORING_SCHEMA is MONITORING_SCHEMA_DIRECT
    assert MONITORING_SCHEMA.names() == list(CANONICAL_COLUMN_ORDER)
    for column in CANONICAL_COLUMN_ORDER:
        assert MONITORING_SCHEMA[column] == COLUMN_DTYPES[column]
    assert COLUMN_DTYPES["symbol"] == pl.Utf8
    assert COLUMN_DTYPES["timeframe"] == pl.Utf8
    assert COLUMN_DTYPES["open_time"] == pl.Int64
    assert COLUMN_DTYPES["manager"] == pl.Utf8
    assert COLUMN_DTYPES["metric_value"] == pl.Float64
    assert COLUMN_DTYPES["threshold"] == pl.Float64
    assert COLUMN_DTYPES["alert"] == pl.Boolean
    assert COLUMN_DTYPES["status"] == pl.Utf8


def test_column_dtypes_cover_every_canonical_column() -> None:
    """COLUMN_DTYPES defines an entry for every canonical monitoring column."""
    assert set(COLUMN_DTYPES.keys()) == set(CANONICAL_COLUMN_ORDER)


def test_canonical_order_starts_with_identity_keys() -> None:
    """Canonical column order begins with symbol/timeframe/open_time/manager."""
    assert CANONICAL_COLUMN_ORDER[0] == "symbol"
    assert CANONICAL_COLUMN_ORDER[1] == "timeframe"
    assert CANONICAL_COLUMN_ORDER[2] == "open_time"
    assert CANONICAL_COLUMN_ORDER[3] == "manager"
    assert CANONICAL_COLUMN_ORDER[-1] == "status"


def test_canonical_order_places_monitor_fields_before_status() -> None:
    """Monitor metadata fields appear in the declared order before status."""
    monitor_slice = CANONICAL_COLUMN_ORDER[4:-1]
    assert monitor_slice == _MONITOR_COLUMNS


def test_monitoring_status_enum_members() -> None:
    """MonitoringStatus exposes NORMAL, WARNING, and CRITICAL members."""
    assert MonitoringStatus.NORMAL.value == "NORMAL"
    assert MonitoringStatus.WARNING.value == "WARNING"
    assert MonitoringStatus.CRITICAL.value == "CRITICAL"
    assert len(list(MonitoringStatus)) == 3


def test_monitoring_statuses_helper() -> None:
    """monitoring_statuses() returns a tuple of all MonitoringStatus members."""
    statuses = monitoring_statuses()
    assert statuses == (
        MonitoringStatus.NORMAL,
        MonitoringStatus.WARNING,
        MonitoringStatus.CRITICAL,
    )
    assert isinstance(statuses, tuple)


def test_monitoring_status_values_helper() -> None:
    """monitoring_status_values() returns valid NORMAL/WARNING/CRITICAL strings."""
    status_values = monitoring_status_values()
    assert status_values == ("NORMAL", "WARNING", "CRITICAL")
    assert isinstance(status_values, tuple)
    assert set(status_values) == {member.value for member in MonitoringStatus}


def test_monitoring_schema_has_twelve_columns() -> None:
    """Monitoring schema defines exactly 12 canonical columns."""
    assert len(CANONICAL_COLUMN_ORDER) == 12
    assert len(MONITORING_SCHEMA) == 12


def test_monitor_string_columns_are_utf8() -> None:
    """Monitor metadata string columns use Utf8 dtype."""
    for column in (
        "monitor_type",
        "monitor_name",
        "severity",
        "metric_name",
    ):
        assert COLUMN_DTYPES[column] == pl.Utf8
