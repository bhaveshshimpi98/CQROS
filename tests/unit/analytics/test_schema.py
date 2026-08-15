"""Unit tests for CQROS merged analytics metrics schema."""

from __future__ import annotations

import polars as pl

from cqros.analytics import ANALYTICS_COLUMNS, ANALYTICS_SCHEMA, AnalyticsStatus
from cqros.analytics.schema import (
    ANALYTICS_SCHEMA as ANALYTICS_SCHEMA_DIRECT,
)
from cqros.analytics.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    analytics_status_values,
    analytics_statuses,
)

_METRIC_COLUMNS: tuple[str, ...] = (
    "rolling_return",
    "rolling_volatility",
    "rolling_sharpe",
    "rolling_sortino",
    "rolling_max_drawdown",
    "rolling_win_rate",
    "rolling_profit_factor",
    "rolling_expectancy",
    "rolling_cagr",
    "rolling_calmar",
    "rolling_recovery_factor",
    "benchmark_return",
    "benchmark_alpha",
    "benchmark_beta",
    "benchmark_correlation",
    "benchmark_tracking_error",
    "benchmark_information_ratio",
)


def test_primary_key_and_required_columns() -> None:
    """Primary key and required columns match the canonical analytics contract."""
    assert PRIMARY_KEY_COLUMNS == ("symbol", "timeframe", "open_time")
    assert REQUIRED_COLUMNS == CANONICAL_COLUMN_ORDER
    assert CANONICAL_COLUMN_ORDER == ANALYTICS_COLUMNS


def test_required_columns_are_complete() -> None:
    """REQUIRED_COLUMNS covers every canonical analytics column exactly once."""
    assert set(REQUIRED_COLUMNS) == set(ANALYTICS_COLUMNS)
    assert len(REQUIRED_COLUMNS) == len(ANALYTICS_COLUMNS)


def test_analytics_columns_contain_required_domain_columns() -> None:
    """ANALYTICS_COLUMNS enumerates identity, rolling, benchmark, and status fields."""
    for column in (
        "symbol",
        "timeframe",
        "open_time",
        "manager",
        *_METRIC_COLUMNS,
        "status",
    ):
        assert column in ANALYTICS_COLUMNS


def test_canonical_column_order_has_no_duplicates() -> None:
    """Canonical analytics column order contains no duplicate names."""
    assert len(CANONICAL_COLUMN_ORDER) == len(set(CANONICAL_COLUMN_ORDER))


def test_column_dtypes_and_analytics_schema() -> None:
    """Analytics schema dtypes match COLUMN_DTYPES in canonical order."""
    assert ANALYTICS_SCHEMA is ANALYTICS_SCHEMA_DIRECT
    assert ANALYTICS_SCHEMA.names() == list(CANONICAL_COLUMN_ORDER)
    for column in CANONICAL_COLUMN_ORDER:
        assert ANALYTICS_SCHEMA[column] == COLUMN_DTYPES[column]
    assert COLUMN_DTYPES["symbol"] == pl.Utf8
    assert COLUMN_DTYPES["timeframe"] == pl.Utf8
    assert COLUMN_DTYPES["open_time"] == pl.Int64
    assert COLUMN_DTYPES["manager"] == pl.Utf8
    assert COLUMN_DTYPES["status"] == pl.Utf8


def test_column_dtypes_cover_every_canonical_column() -> None:
    """COLUMN_DTYPES defines an entry for every canonical analytics column."""
    assert set(COLUMN_DTYPES.keys()) == set(CANONICAL_COLUMN_ORDER)


def test_canonical_order_starts_with_identity_keys() -> None:
    """Canonical column order begins with symbol/timeframe/open_time/manager."""
    assert CANONICAL_COLUMN_ORDER[0] == "symbol"
    assert CANONICAL_COLUMN_ORDER[1] == "timeframe"
    assert CANONICAL_COLUMN_ORDER[2] == "open_time"
    assert CANONICAL_COLUMN_ORDER[3] == "manager"
    assert CANONICAL_COLUMN_ORDER[-1] == "status"


def test_canonical_order_places_metrics_before_status() -> None:
    """Rolling and benchmark metrics appear in the declared order before status."""
    metric_slice = CANONICAL_COLUMN_ORDER[4:-1]
    assert metric_slice == _METRIC_COLUMNS


def test_analytics_status_enum_members() -> None:
    """AnalyticsStatus exposes ACTIVE and FINISHED members."""
    assert AnalyticsStatus.ACTIVE.value == "ACTIVE"
    assert AnalyticsStatus.FINISHED.value == "FINISHED"
    assert len(list(AnalyticsStatus)) == 2


def test_analytics_statuses_helper() -> None:
    """analytics_statuses() returns a tuple of all AnalyticsStatus members."""
    statuses = analytics_statuses()
    assert statuses == (AnalyticsStatus.ACTIVE, AnalyticsStatus.FINISHED)
    assert isinstance(statuses, tuple)


def test_analytics_status_values_helper() -> None:
    """analytics_status_values() returns valid ACTIVE and FINISHED strings."""
    status_values = analytics_status_values()
    assert status_values == ("ACTIVE", "FINISHED")
    assert isinstance(status_values, tuple)
    assert set(status_values) == {member.value for member in AnalyticsStatus}


def test_analytics_schema_has_twenty_two_columns() -> None:
    """Analytics schema defines exactly 22 canonical columns."""
    assert len(CANONICAL_COLUMN_ORDER) == 22
    assert len(ANALYTICS_SCHEMA) == 22


def test_metric_columns_are_float64() -> None:
    """Rolling and benchmark metric columns use Float64 dtype."""
    for column in _METRIC_COLUMNS:
        assert COLUMN_DTYPES[column] == pl.Float64
