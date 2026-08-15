"""Unit tests for CQROS merged performance metrics schema."""

from __future__ import annotations

import polars as pl

from cqros.performance import PERFORMANCE_COLUMNS, PERFORMANCE_SCHEMA, PerformanceStatus
from cqros.performance.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    performance_statuses,
    values,
)
from cqros.performance.schema import (
    PERFORMANCE_SCHEMA as PERFORMANCE_SCHEMA_DIRECT,
)


def test_primary_key_and_required_columns() -> None:
    """Primary key and required columns match the canonical performance contract."""
    assert PRIMARY_KEY_COLUMNS == ("symbol", "timeframe", "open_time")
    assert REQUIRED_COLUMNS == CANONICAL_COLUMN_ORDER
    assert CANONICAL_COLUMN_ORDER == PERFORMANCE_COLUMNS


def test_performance_columns_contain_required_domain_columns() -> None:
    """PERFORMANCE_COLUMNS enumerates identity, return/risk, trade, and status fields."""
    for column in (
        "symbol",
        "timeframe",
        "open_time",
        "manager",
        "total_return",
        "cagr",
        "volatility",
        "downside_volatility",
        "max_drawdown",
        "drawdown_duration",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "total_trades",
        "winning_trades",
        "losing_trades",
        "win_rate",
        "average_win",
        "average_loss",
        "profit_factor",
        "expectancy",
        "starting_equity",
        "ending_equity",
        "net_profit",
        "gross_profit",
        "gross_loss",
        "first_trade_time",
        "last_trade_time",
        "status",
    ):
        assert column in PERFORMANCE_COLUMNS


def test_column_dtypes_and_performance_schema() -> None:
    """Performance schema dtypes match COLUMN_DTYPES in canonical order."""
    assert PERFORMANCE_SCHEMA is PERFORMANCE_SCHEMA_DIRECT
    assert PERFORMANCE_SCHEMA.names() == list(CANONICAL_COLUMN_ORDER)
    for column in CANONICAL_COLUMN_ORDER:
        assert PERFORMANCE_SCHEMA[column] == COLUMN_DTYPES[column]
    assert COLUMN_DTYPES["open_time"] == pl.Datetime("us", "UTC")
    assert COLUMN_DTYPES["total_return"] == pl.Float64
    assert COLUMN_DTYPES["total_trades"] == pl.Int64
    assert COLUMN_DTYPES["status"] == pl.Utf8


def test_canonical_order_starts_with_identity_keys() -> None:
    """Canonical column order begins with symbol/timeframe/open_time/manager."""
    assert CANONICAL_COLUMN_ORDER[0] == "symbol"
    assert CANONICAL_COLUMN_ORDER[1] == "timeframe"
    assert CANONICAL_COLUMN_ORDER[2] == "open_time"
    assert CANONICAL_COLUMN_ORDER[3] == "manager"
    assert CANONICAL_COLUMN_ORDER[-1] == "status"


def test_performance_status_enum_members() -> None:
    """PerformanceStatus exposes ACTIVE and FINISHED members."""
    assert PerformanceStatus.ACTIVE.value == "ACTIVE"
    assert PerformanceStatus.FINISHED.value == "FINISHED"
    assert len(list(PerformanceStatus)) == 2


def test_performance_statuses_helper() -> None:
    """performance_statuses() returns a tuple of all PerformanceStatus members."""
    statuses = performance_statuses()
    assert statuses == (PerformanceStatus.ACTIVE, PerformanceStatus.FINISHED)
    assert isinstance(statuses, tuple)


def test_values_helper_extracts_string_values() -> None:
    """values() returns a tuple of string values for an enum class."""
    status_values = values(PerformanceStatus)
    assert status_values == ("ACTIVE", "FINISHED")
    assert isinstance(status_values, tuple)


def test_performance_schema_has_twenty_nine_columns() -> None:
    """Performance schema defines exactly 29 canonical columns."""
    assert len(CANONICAL_COLUMN_ORDER) == 29
    assert len(PERFORMANCE_SCHEMA) == 29


def test_risk_adjusted_ratio_columns_are_float64() -> None:
    """Risk-adjusted ratio columns use Float64 dtype."""
    assert COLUMN_DTYPES["sharpe_ratio"] == pl.Float64
    assert COLUMN_DTYPES["sortino_ratio"] == pl.Float64
    assert COLUMN_DTYPES["calmar_ratio"] == pl.Float64
    assert COLUMN_DTYPES["profit_factor"] == pl.Float64


def test_trade_count_columns_are_int64() -> None:
    """Trade statistic count columns use Int64 dtype."""
    assert COLUMN_DTYPES["total_trades"] == pl.Int64
    assert COLUMN_DTYPES["winning_trades"] == pl.Int64
    assert COLUMN_DTYPES["losing_trades"] == pl.Int64
    assert COLUMN_DTYPES["drawdown_duration"] == pl.Int64


def test_return_and_equity_columns_are_float64() -> None:
    """Return and equity metric columns use Float64 dtype."""
    assert COLUMN_DTYPES["total_return"] == pl.Float64
    assert COLUMN_DTYPES["cagr"] == pl.Float64
    assert COLUMN_DTYPES["volatility"] == pl.Float64
    assert COLUMN_DTYPES["downside_volatility"] == pl.Float64
    assert COLUMN_DTYPES["max_drawdown"] == pl.Float64
    assert COLUMN_DTYPES["starting_equity"] == pl.Float64
    assert COLUMN_DTYPES["ending_equity"] == pl.Float64
    assert COLUMN_DTYPES["net_profit"] == pl.Float64


def test_trade_time_columns_are_utc_datetime() -> None:
    """first_trade_time and last_trade_time use UTC microsecond datetime dtype."""
    assert COLUMN_DTYPES["first_trade_time"] == pl.Datetime("us", "UTC")
    assert COLUMN_DTYPES["last_trade_time"] == pl.Datetime("us", "UTC")
