"""Unit tests for CQROS merged backtesting performance schema."""

from __future__ import annotations

import polars as pl

from cqros.backtesting import (
    BACKTESTING_COLUMNS,
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    MERGED_BACKTESTING_SCHEMA,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    BacktestingStatus,
    backtesting_statuses,
    values,
)
from cqros.backtesting.schema import (
    MERGED_BACKTESTING_SCHEMA as MERGED_BACKTESTING_SCHEMA_DIRECT,
)


def test_primary_key_and_required_columns() -> None:
    """Primary key and required columns match the canonical backtesting contract."""
    assert PRIMARY_KEY_COLUMNS == ("symbol", "timeframe", "open_time")
    assert REQUIRED_COLUMNS == CANONICAL_COLUMN_ORDER
    assert CANONICAL_COLUMN_ORDER == BACKTESTING_COLUMNS


def test_backtesting_columns_contain_required_domain_columns() -> None:
    """BACKTESTING_COLUMNS enumerates identity, equity, trade stats, and status fields."""
    for column in (
        "symbol",
        "timeframe",
        "open_time",
        "manager",
        "equity",
        "cash",
        "position_value",
        "realized_pnl",
        "unrealized_pnl",
        "total_pnl",
        "drawdown",
        "peak_equity",
        "daily_return",
        "cumulative_return",
        "trade_count",
        "winning_trades",
        "losing_trades",
        "win_rate",
        "profit_factor",
        "sharpe_stub",
        "sortino_stub",
        "max_drawdown",
        "status",
    ):
        assert column in BACKTESTING_COLUMNS


def test_column_dtypes_and_merged_schema() -> None:
    """Merged schema dtypes match COLUMN_DTYPES in canonical order."""
    assert MERGED_BACKTESTING_SCHEMA is MERGED_BACKTESTING_SCHEMA_DIRECT
    assert MERGED_BACKTESTING_SCHEMA.names() == list(CANONICAL_COLUMN_ORDER)
    for column in CANONICAL_COLUMN_ORDER:
        assert MERGED_BACKTESTING_SCHEMA[column] == COLUMN_DTYPES[column]
    assert COLUMN_DTYPES["open_time"] == pl.Datetime("us", "UTC")
    assert COLUMN_DTYPES["equity"] == pl.Float64
    assert COLUMN_DTYPES["trade_count"] == pl.Int64
    assert COLUMN_DTYPES["status"] == pl.Utf8


def test_canonical_order_starts_with_identity_keys() -> None:
    """Canonical column order begins with symbol/timeframe/open_time/manager."""
    assert CANONICAL_COLUMN_ORDER[0] == "symbol"
    assert CANONICAL_COLUMN_ORDER[1] == "timeframe"
    assert CANONICAL_COLUMN_ORDER[2] == "open_time"
    assert CANONICAL_COLUMN_ORDER[3] == "manager"
    assert CANONICAL_COLUMN_ORDER[-1] == "status"


def test_backtesting_status_enum_members() -> None:
    """BacktestingStatus exposes ACTIVE and FINISHED members."""
    assert BacktestingStatus.ACTIVE.value == "ACTIVE"
    assert BacktestingStatus.FINISHED.value == "FINISHED"
    assert len(list(BacktestingStatus)) == 2


def test_backtesting_statuses_helper() -> None:
    """backtesting_statuses() returns a tuple of all BacktestingStatus members."""
    statuses = backtesting_statuses()
    assert statuses == (BacktestingStatus.ACTIVE, BacktestingStatus.FINISHED)
    assert isinstance(statuses, tuple)


def test_values_helper_extracts_string_values() -> None:
    """values() returns a tuple of string values for an enum class."""
    status_values = values(BacktestingStatus)
    assert status_values == ("ACTIVE", "FINISHED")
    assert isinstance(status_values, tuple)


def test_merged_schema_has_twenty_three_columns() -> None:
    """Merged backtesting schema defines exactly 23 canonical columns."""
    assert len(CANONICAL_COLUMN_ORDER) == 23
    assert len(MERGED_BACKTESTING_SCHEMA) == 23


def test_stub_and_profit_factor_columns_are_float64() -> None:
    """Reserved stub columns and profit_factor use Float64 dtype."""
    assert COLUMN_DTYPES["profit_factor"] == pl.Float64
    assert COLUMN_DTYPES["sharpe_stub"] == pl.Float64
    assert COLUMN_DTYPES["sortino_stub"] == pl.Float64


def test_trade_count_columns_are_int64() -> None:
    """Trade statistic count columns use Int64 dtype."""
    assert COLUMN_DTYPES["trade_count"] == pl.Int64
    assert COLUMN_DTYPES["winning_trades"] == pl.Int64
    assert COLUMN_DTYPES["losing_trades"] == pl.Int64


def test_drawdown_and_return_columns_are_float64() -> None:
    """Drawdown and return metric columns use Float64 dtype."""
    assert COLUMN_DTYPES["drawdown"] == pl.Float64
    assert COLUMN_DTYPES["peak_equity"] == pl.Float64
    assert COLUMN_DTYPES["daily_return"] == pl.Float64
    assert COLUMN_DTYPES["cumulative_return"] == pl.Float64
    assert COLUMN_DTYPES["max_drawdown"] == pl.Float64
