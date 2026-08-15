"""Unit tests for CQROS ``BacktestingVerifier``."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from cqros.backtesting import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    ERROR_REQUIRED_COLUMNS,
    ERROR_SCHEMA_MISMATCH,
    BacktestingStatus,
    BacktestingValidationError,
    BacktestingVerifier,
)

_MANAGER = "simple"
_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1h"
_OPEN_TIME = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)


def _canonical_frame(
    *,
    symbol: str = _SYMBOL,
    timeframe: str = _TIMEFRAME,
    open_time: datetime = _OPEN_TIME,
    manager: str = _MANAGER,
    equity: float = 10500.0,
    drawdown: float = 0.0,
    max_drawdown: float = 0.0,
    win_rate: float = 0.0,
    profit_factor: float | None = None,
    status: str = BacktestingStatus.FINISHED.value,
) -> pl.DataFrame:
    """Build a canonical backtesting frame that should pass verification."""
    return pl.DataFrame(
        {
            "symbol": [symbol],
            "timeframe": [timeframe],
            "open_time": [open_time],
            "manager": [manager],
            "equity": [equity],
            "cash": [10000.0],
            "position_value": [500.0],
            "realized_pnl": [0.0],
            "unrealized_pnl": [500.0],
            "total_pnl": [500.0],
            "drawdown": [drawdown],
            "peak_equity": [equity],
            "daily_return": [0.0],
            "cumulative_return": [0.0],
            "trade_count": [0],
            "winning_trades": [0],
            "losing_trades": [0],
            "win_rate": [win_rate],
            "profit_factor": [profit_factor],
            "sharpe_stub": [None],
            "sortino_stub": [None],
            "max_drawdown": [max_drawdown],
            "status": [status],
        },
        schema=dict(COLUMN_DTYPES),
    ).select(list(CANONICAL_COLUMN_ORDER))


# ---------------------------------------------------------------------------
# Clean frame passes
# ---------------------------------------------------------------------------


def test_verifier_passes_on_canonical_frame() -> None:
    """A correctly formed performance frame passes all verifier checks."""
    verifier = BacktestingVerifier()
    report = verifier.verify(_canonical_frame())
    assert report.passed is True
    assert report.rows_checked == 1
    assert report.null_rows == 0
    assert report.nan_rows == 0
    assert report.duplicate_timestamp_rows == 0
    assert report.invalid_timestamp_rows == 0
    assert report.invalid_numeric_rows == 0
    assert report.warnings == ()


def test_verifier_passes_on_active_status_frame() -> None:
    """An ACTIVE status frame passes all verifier checks."""
    verifier = BacktestingVerifier()
    frame = _canonical_frame(status=BacktestingStatus.ACTIVE.value)
    report = verifier.verify(frame)
    assert report.passed is True
    assert report.rows_checked == 1


def test_verifier_passes_with_null_profit_factor() -> None:
    """NULL profit_factor is allowed and does not fail verification."""
    verifier = BacktestingVerifier()
    frame = _canonical_frame(profit_factor=None)
    report = verifier.verify(frame)
    assert report.passed is True


# ---------------------------------------------------------------------------
# Missing / mismatched columns
# ---------------------------------------------------------------------------


def test_verifier_raises_on_missing_required_column() -> None:
    """Missing required columns raise BacktestingValidationError with ERROR_REQUIRED_COLUMNS."""
    verifier = BacktestingVerifier()
    frame = _canonical_frame().drop("equity")
    with pytest.raises(BacktestingValidationError) as exc_info:
        verifier.verify(frame)
    assert exc_info.value.error_code == ERROR_REQUIRED_COLUMNS


def test_verifier_raises_on_dtype_mismatch() -> None:
    """Wrong column dtype raises BacktestingValidationError with ERROR_SCHEMA_MISMATCH."""
    verifier = BacktestingVerifier()
    frame = _canonical_frame().with_columns(pl.col("trade_count").cast(pl.Float64))
    with pytest.raises(BacktestingValidationError) as exc_info:
        verifier.verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH


# ---------------------------------------------------------------------------
# Duplicate primary keys
# ---------------------------------------------------------------------------


def test_verifier_warns_on_duplicate_primary_keys() -> None:
    """Duplicate (symbol, timeframe, open_time) emits a warning."""
    verifier = BacktestingVerifier()
    row = _canonical_frame()
    duplicate = pl.concat([row, row])
    report = verifier.verify(duplicate)
    assert report.passed is False
    assert report.duplicate_timestamp_rows == 1
    assert any("Duplicate" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Null rows
# ---------------------------------------------------------------------------


def test_verifier_counts_null_rows_in_equity() -> None:
    """Rows with NULL equity are counted in null_rows."""
    verifier = BacktestingVerifier()
    frame = _canonical_frame().with_columns(pl.Series("equity", [None], dtype=pl.Float64))
    report = verifier.verify(frame)
    assert report.null_rows == 1
    assert report.passed is False


def test_verifier_counts_null_rows_in_status() -> None:
    """Rows with NULL status are counted in null_rows."""
    verifier = BacktestingVerifier()
    frame = _canonical_frame().with_columns(pl.Series("status", [None], dtype=pl.Utf8))
    report = verifier.verify(frame)
    assert report.null_rows == 1
    assert report.passed is False


# ---------------------------------------------------------------------------
# Invalid enum values
# ---------------------------------------------------------------------------


def test_verifier_warns_on_invalid_status() -> None:
    """Non-canonical status values emit an invalid-status warning."""
    verifier = BacktestingVerifier()
    frame = _canonical_frame().with_columns(pl.lit("INVALID_STATUS").alias("status"))
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("BacktestingStatus" in w or "status" in w.lower() for w in report.warnings)


def test_verifier_warns_on_empty_status() -> None:
    """Empty string in status emits an empty-status warning."""
    verifier = BacktestingVerifier()
    frame = _canonical_frame().with_columns(pl.lit("").alias("status"))
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("status" in w.lower() or "Empty" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Drawdown and win-rate range checks
# ---------------------------------------------------------------------------


def test_verifier_warns_on_drawdown_out_of_range() -> None:
    """Drawdown values outside [0, 1] emit a drawdown-range warning."""
    verifier = BacktestingVerifier()
    frame = _canonical_frame(drawdown=1.5)
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("Drawdown" in w for w in report.warnings)


def test_verifier_warns_on_max_drawdown_less_than_drawdown() -> None:
    """max_drawdown < drawdown emits a max-drawdown warning."""
    verifier = BacktestingVerifier()
    frame = _canonical_frame(drawdown=0.25, max_drawdown=0.10)
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("max_drawdown" in w for w in report.warnings)


def test_verifier_warns_on_win_rate_out_of_range() -> None:
    """Win rate values outside [0, 1] emit a win-rate warning."""
    verifier = BacktestingVerifier()
    for bad_rate in (-0.1, 1.1):
        frame = _canonical_frame().with_columns(pl.lit(bad_rate).alias("win_rate"))
        report = verifier.verify(frame)
        assert report.passed is False, f"Expected failure for win_rate={bad_rate}"
        assert any("Win rate" in w for w in report.warnings)


def test_verifier_warns_on_negative_profit_factor() -> None:
    """Negative profit_factor values emit a profit-factor warning."""
    verifier = BacktestingVerifier()
    frame = _canonical_frame(profit_factor=-0.5)
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("profit_factor" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Non-finite numeric values
# ---------------------------------------------------------------------------


def test_verifier_warns_on_nan_equity() -> None:
    """NaN equity triggers a non-finite value warning."""
    verifier = BacktestingVerifier()
    frame = _canonical_frame().with_columns(pl.lit(float("nan")).alias("equity"))
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.nan_rows > 0 or report.invalid_numeric_rows > 0


def test_verifier_warns_on_inf_cumulative_return() -> None:
    """Infinite cumulative_return triggers a non-finite value warning."""
    verifier = BacktestingVerifier()
    frame = _canonical_frame().with_columns(pl.lit(float("inf")).alias("cumulative_return"))
    report = verifier.verify(frame)
    assert report.passed is False


# ---------------------------------------------------------------------------
# Invalid timestamps
# ---------------------------------------------------------------------------


def test_verifier_counts_invalid_timestamp_rows() -> None:
    """Rows with NULL open_time values are counted as invalid timestamps."""
    verifier = BacktestingVerifier()
    frame = _canonical_frame().with_columns(
        pl.lit(None).cast(pl.Datetime("us", "UTC")).alias("open_time")
    )
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.invalid_timestamp_rows > 0 or report.null_rows > 0


# ---------------------------------------------------------------------------
# Column order and sorting
# ---------------------------------------------------------------------------


def test_verifier_warns_on_non_canonical_column_order() -> None:
    """Frame with columns out of canonical order emits a column-order warning."""
    verifier = BacktestingVerifier()
    columns = list(CANONICAL_COLUMN_ORDER)
    columns.reverse()
    frame = _canonical_frame().select(columns)
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("column order" in w.lower() or "Column" in w for w in report.warnings)


def test_verifier_warns_on_unsorted_open_time() -> None:
    """Frame not sorted by open_time emits an unsorted warning."""
    verifier = BacktestingVerifier()
    t0 = _OPEN_TIME
    t1 = _OPEN_TIME + timedelta(hours=1)
    row_a = _canonical_frame(open_time=t1, symbol="BTCUSDT")
    row_b = _canonical_frame(open_time=t0, symbol="ETHUSDT")
    frame = pl.concat([row_a, row_b])
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("sorted" in w.lower() for w in report.warnings)


# ---------------------------------------------------------------------------
# Multiple rows and lineage
# ---------------------------------------------------------------------------


def test_verifier_aggregates_multiple_rows() -> None:
    """Verifier correctly aggregates checks across multiple rows."""
    verifier = BacktestingVerifier()
    t0 = _OPEN_TIME
    t1 = _OPEN_TIME + timedelta(hours=1)
    row_a = _canonical_frame(open_time=t0, status=BacktestingStatus.ACTIVE.value)
    row_b = _canonical_frame(open_time=t1, symbol="ETHUSDT")
    frame = pl.concat([row_a, row_b])
    report = verifier.verify(frame)
    assert report.rows_checked == 2
    assert report.passed is True


def test_verifier_incomplete_lineage_emits_warning() -> None:
    """Blank manager (lineage column) emits an incomplete-lineage warning."""
    verifier = BacktestingVerifier()
    frame = _canonical_frame().with_columns(pl.lit("").alias("manager"))
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("lineage" in w.lower() or "Lineage" in w for w in report.warnings)


def test_verifier_passes_win_rate_boundary_values() -> None:
    """win_rate = 0.0 and 1.0 are valid boundary values."""
    verifier = BacktestingVerifier()
    for boundary in (0.0, 1.0):
        frame = _canonical_frame().with_columns(pl.lit(boundary).alias("win_rate"))
        report = verifier.verify(frame)
        assert report.passed is True, f"Expected pass for win_rate={boundary}"


def test_verifier_passes_drawdown_boundary_values() -> None:
    """drawdown = 0.0 and 1.0 are valid boundary values."""
    verifier = BacktestingVerifier()
    for boundary in (0.0, 1.0):
        frame = _canonical_frame(drawdown=boundary, max_drawdown=boundary)
        report = verifier.verify(frame)
        assert report.passed is True, f"Expected pass for drawdown={boundary}"
