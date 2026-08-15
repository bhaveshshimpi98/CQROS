"""Unit tests for CQROS ``TradeManagementVerifier``."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from cqros.trade_management import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    ERROR_REQUIRED_COLUMNS,
    ERROR_SCHEMA_MISMATCH,
    MERGED_TRADE_MANAGEMENT_SCHEMA,
    ManagementAction,
    ShutdownReason,
    TradeManagementValidationError,
    TradeManagementVerifier,
)

_TIMEFRAME = "1h"
_MANAGER = "simple"


def _open_time(index: int) -> datetime:
    """Build a deterministic UTC open_time for row ``index``."""
    return datetime(2024, 1, 1, hour=index, tzinfo=UTC)


def _trade_management_frame(
    *,
    symbols: list[str] | None = None,
    position_ids: list[str] | None = None,
    open_times: list[datetime] | None = None,
    actions: list[str] | None = None,
    reasons: list[str] | None = None,
    allow_pyramid: list[bool] | None = None,
    managers: list[str] | None = None,
    current_prices: list[float] | None = None,
    highest_prices: list[float] | None = None,
    lowest_prices: list[float] | None = None,
    entry_prices: list[float] | None = None,
    stop_prices: list[float | None] | None = None,
    breakeven_prices: list[float | None] | None = None,
) -> pl.DataFrame:
    """Build a canonical trade-management frame for verifier tests."""
    symbols = symbols if symbols is not None else ["BTCUSDT", "ETHUSDT"]
    row_count = len(symbols)
    open_times = (
        open_times if open_times is not None else [_open_time(index) for index in range(row_count)]
    )
    position_ids = (
        position_ids
        if position_ids is not None
        else [f"pos-{index + 1:08d}" for index in range(row_count)]
    )
    actions = actions if actions is not None else [ManagementAction.NONE.value] * row_count
    reasons = reasons if reasons is not None else [ShutdownReason.NONE.value] * row_count
    allow_pyramid = allow_pyramid if allow_pyramid is not None else [False] * row_count
    managers = managers if managers is not None else [_MANAGER] * row_count
    entry_prices = entry_prices if entry_prices is not None else [100.0] * row_count
    current_prices = current_prices if current_prices is not None else [104.0] * row_count
    highest_prices = highest_prices if highest_prices is not None else current_prices
    lowest_prices = lowest_prices if lowest_prices is not None else current_prices
    stop_prices = stop_prices if stop_prices is not None else [None] * row_count
    breakeven_prices = breakeven_prices if breakeven_prices is not None else [None] * row_count
    return pl.DataFrame(
        {
            "symbol": symbols,
            "timeframe": [_TIMEFRAME] * row_count,
            "open_time": open_times,
            "manager": managers,
            "position_id": position_ids,
            "position_status": ["OPEN"] * row_count,
            "quantity": [1.0] * row_count,
            "entry_price": entry_prices,
            "current_price": current_prices,
            "highest_price": highest_prices,
            "lowest_price": lowest_prices,
            "unrealized_pnl": [0.0] * row_count,
            "risk_state": ["NORMAL"] * row_count,
            "management_action": actions,
            "action_reason": reasons,
            "stop_price": stop_prices,
            "take_profit_price": [None] * row_count,
            "trail_price": [98.8] * row_count,
            "breakeven_price": breakeven_prices,
            "allow_pyramid": allow_pyramid,
            "exit_quantity": [0.0] * row_count,
            "model_name": ["alpha-lgbm"] * row_count,
            "model_version": ["1.0.0"] * row_count,
            "optimizer": ["equal_weight"] * row_count,
            "policy": ["fixed_risk"] * row_count,
        },
        schema=dict(COLUMN_DTYPES),
    ).select(list(CANONICAL_COLUMN_ORDER))


def test_verifier_passes_canonical_frame() -> None:
    """A clean canonical trade-management frame passes verification."""
    report = TradeManagementVerifier().verify(_trade_management_frame())
    assert report.passed is True
    assert report.rows_checked == 2
    assert report.duplicate_timestamp_rows == 0
    assert report.null_rows == 0
    assert report.warnings == ()


def test_verifier_requires_merged_schema_identity() -> None:
    """Canonical frames cast to MERGED_TRADE_MANAGEMENT_SCHEMA remain verifiable."""
    frame = _trade_management_frame().cast(MERGED_TRADE_MANAGEMENT_SCHEMA)
    report = TradeManagementVerifier().verify(frame)
    assert report.passed is True


def test_verifier_rejects_missing_columns() -> None:
    """Missing required columns raise TME-VERIFICATION-001."""
    frame = _trade_management_frame().drop("entry_price")
    with pytest.raises(TradeManagementValidationError) as exc_info:
        TradeManagementVerifier().verify(frame)
    assert exc_info.value.error_code == ERROR_REQUIRED_COLUMNS


def test_verifier_rejects_dtype_mismatch() -> None:
    """Column dtype mismatches raise TME-VERIFICATION-002."""
    bad_dtype = _trade_management_frame().with_columns(pl.col("entry_price").cast(pl.Int64))
    with pytest.raises(TradeManagementValidationError) as exc_info:
        TradeManagementVerifier().verify(bad_dtype)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH


def test_verifier_reports_duplicate_primary_keys() -> None:
    """Duplicate primary keys fail verification with a warning."""
    duplicates = _trade_management_frame(
        symbols=["BTCUSDT", "BTCUSDT"],
        position_ids=["pos-00000001", "pos-00000001"],
        open_times=[_open_time(0), _open_time(0)],
    )
    report = TradeManagementVerifier().verify(duplicates)
    assert report.passed is False
    assert report.duplicate_timestamp_rows == 1
    assert "Duplicate trade-management primary keys detected." in report.warnings


def test_verifier_reports_null_rows() -> None:
    """Null values in required columns fail verification with a warning."""
    nulls = _trade_management_frame().with_columns(
        pl.when(pl.arange(0, pl.len()) == 1).then(None).otherwise(pl.col("symbol")).alias("symbol")
    )
    report = TradeManagementVerifier().verify(nulls)
    assert report.passed is False
    assert report.null_rows > 0
    assert "Rows containing NULL values." in report.warnings


def test_verifier_reports_invalid_enums() -> None:
    """Invalid management_action and action_reason values fail verification."""
    invalid_action = _trade_management_frame(
        actions=[ManagementAction.NONE.value, "PANIC"],
    )
    report = TradeManagementVerifier().verify(invalid_action)
    assert report.passed is False
    assert "Invalid ManagementAction values detected." in report.warnings

    invalid_reason = _trade_management_frame(
        reasons=[ShutdownReason.NONE.value, "UNKNOWN_REASON"],
    )
    report = TradeManagementVerifier().verify(invalid_reason)
    assert report.passed is False
    assert "Invalid ShutdownReason values detected." in report.warnings


def test_verifier_reports_boolean_nulls() -> None:
    """Null allow_pyramid values fail verification with a warning."""
    null_bool = _trade_management_frame().with_columns(
        pl.when(pl.arange(0, pl.len()) == 1)
        .then(None)
        .otherwise(pl.col("allow_pyramid"))
        .alias("allow_pyramid")
    )
    report = TradeManagementVerifier().verify(null_bool)
    assert report.passed is False
    assert "Invalid allow_pyramid boolean values detected." in report.warnings


def test_verifier_reports_price_inconsistency() -> None:
    """Inconsistent price relationships fail verification with a warning."""
    inconsistent = _trade_management_frame(
        symbols=["BTCUSDT"],
        position_ids=["pos-00000001"],
        open_times=[_open_time(0)],
        actions=[ManagementAction.UPDATE_STOP.value],
        reasons=[ShutdownReason.BREAKEVEN.value],
        stop_prices=[100.0],
        breakeven_prices=[99.0],
    )
    report = TradeManagementVerifier().verify(inconsistent)
    assert report.passed is False
    assert "Inconsistent price relationships detected." in report.warnings


def test_verifier_reports_incomplete_lineage() -> None:
    """Blank lineage metadata fails verification with a warning."""
    incomplete = _trade_management_frame(managers=[_MANAGER, ""])
    report = TradeManagementVerifier().verify(incomplete)
    assert report.passed is False
    assert "Incomplete lineage metadata detected." in report.warnings


def test_verifier_reports_unsorted_open_time() -> None:
    """A frame not sorted by open_time fails verification with a warning."""
    unsorted = _trade_management_frame(open_times=[_open_time(3), _open_time(1)])
    report = TradeManagementVerifier().verify(unsorted)
    assert report.passed is False
    assert "Frame is not sorted by open_time." in report.warnings


def test_verifier_reports_non_canonical_order() -> None:
    """A non-canonical column order fails verification with a warning."""
    reordered = _trade_management_frame().select(list(reversed(CANONICAL_COLUMN_ORDER)))
    report = TradeManagementVerifier().verify(reordered)
    assert report.passed is False
    assert "Frame column order does not match canonical order." in report.warnings
