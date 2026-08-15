"""Unit tests for CQROS ``ExecutionVerifier``."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from cqros.execution import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    ERROR_REQUIRED_COLUMNS,
    ERROR_SCHEMA_MISMATCH,
    MERGED_TRADE_SCHEMA,
    ExecutionStatus,
    ExecutionValidationError,
    ExecutionVerifier,
)

_TIMEFRAME = "1h"
_MANAGER = "simple"


def _open_time(index: int) -> datetime:
    """Build a deterministic UTC open_time for row ``index``."""
    return datetime(2024, 1, 1, hour=index, tzinfo=UTC)


def _trade_frame(
    *,
    symbols: list[str] | None = None,
    statuses: list[str] | None = None,
    open_times: list[datetime] | None = None,
) -> pl.DataFrame:
    """Build a canonical trade frame for verifier tests."""
    symbols = symbols if symbols is not None else ["BTCUSDT", "ETHUSDT"]
    row_count = len(symbols)
    times = (
        open_times if open_times is not None else [_open_time(index) for index in range(row_count)]
    )
    statuses = statuses if statuses is not None else [ExecutionStatus.FILLED.value] * row_count
    return pl.DataFrame(
        {
            "symbol": symbols,
            "timeframe": [_TIMEFRAME] * row_count,
            "open_time": times,
            "model_name": ["alpha-lgbm"] * row_count,
            "model_version": ["1.0.0"] * row_count,
            "optimizer": ["equal_weight"] * row_count,
            "policy": ["fixed_risk"] * row_count,
            "manager": [_MANAGER] * row_count,
            "signal": ["BUY"] * row_count,
            "side": ["BUY"] * row_count,
            "order_type": ["MARKET"] * row_count,
            "requested_quantity": [1.0] * row_count,
            "executed_quantity": [1.0] * row_count,
            "requested_price": [100.0] * row_count,
            "executed_price": [100.0] * row_count,
            "fees": [0.0] * row_count,
            "slippage": [0.0] * row_count,
            "status": statuses,
            "execution_time": times,
        },
        schema=dict(COLUMN_DTYPES),
    ).select(list(CANONICAL_COLUMN_ORDER))


def test_verifier_passes_canonical_frame() -> None:
    """A clean canonical trade frame passes verification."""
    report = ExecutionVerifier().verify(_trade_frame())
    assert report.passed is True
    assert report.rows_checked == 2
    assert report.duplicate_timestamp_rows == 0
    assert report.null_rows == 0
    assert report.warnings == ()


def test_verifier_rejects_missing_columns_and_dtype_mismatch() -> None:
    """Missing columns and dtype mismatches raise verification errors."""
    frame = _trade_frame().drop("status")
    with pytest.raises(ExecutionValidationError) as exc_info:
        ExecutionVerifier().verify(frame)
    assert exc_info.value.error_code == ERROR_REQUIRED_COLUMNS

    bad_dtype = _trade_frame().with_columns(pl.col("fees").cast(pl.Int64))
    with pytest.raises(ExecutionValidationError) as exc_info:
        ExecutionVerifier().verify(bad_dtype)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH


def test_verifier_reports_duplicates_and_invalid_status() -> None:
    """Duplicate keys and invalid status values fail verification."""
    duplicates = _trade_frame(
        symbols=["BTCUSDT", "BTCUSDT"],
        open_times=[_open_time(0), _open_time(0)],
    )
    report = ExecutionVerifier().verify(duplicates)
    assert report.passed is False
    assert report.duplicate_timestamp_rows == 1
    assert "Duplicate timestamps detected." in report.warnings

    invalid_status = _trade_frame(statuses=["FILLED", "PENDING"])
    report = ExecutionVerifier().verify(invalid_status)
    assert report.passed is False
    assert report.invalid_numeric_rows == 1
    assert "Invalid ExecutionStatus values detected." in report.warnings


def test_verifier_requires_merged_schema_identity() -> None:
    """Canonical frames cast to MERGED_TRADE_SCHEMA remain verifiable."""
    frame = _trade_frame().cast(MERGED_TRADE_SCHEMA)
    report = ExecutionVerifier().verify(frame)
    assert report.passed is True
