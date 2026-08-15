"""Unit tests for CQROS ``ExitEngineVerifier``."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from cqros.exit_engine import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    ERROR_REQUIRED_COLUMNS,
    ERROR_SCHEMA_MISMATCH,
    ExitAction,
    ExitEngineValidationError,
    ExitEngineVerifier,
    ExitReason,
)

_MANAGER = "simple"
_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1h"
_OPEN_TIME = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
_POSITION_ID = "pos-00000001"


def _canonical_frame(
    *,
    symbol: str = _SYMBOL,
    timeframe: str = _TIMEFRAME,
    open_time: datetime = _OPEN_TIME,
    position_id: str = _POSITION_ID,
    manager: str = _MANAGER,
    entry_price: float = 100.0,
    current_price: float = 102.0,
    quantity: float = 1.0,
    risk_reward_ratio: float = 0.4,
    risk_state: str = "NORMAL",
    trade_state: str = "NONE",
    pyramid_state: str = "INSUFFICIENT_PROFIT",
    exit_action: str = ExitAction.HOLD.value,
    exit_reason: str = ExitReason.NONE.value,
    recommended_quantity: float = 0.0,
    recommended_percent: float = 0.0,
    priority: int = 0,
) -> pl.DataFrame:
    """Build a canonical exit-engine frame that should pass verification."""
    return pl.DataFrame(
        {
            "symbol": [symbol],
            "timeframe": [timeframe],
            "open_time": [open_time],
            "position_id": [position_id],
            "manager": [manager],
            "entry_price": [entry_price],
            "current_price": [current_price],
            "quantity": [quantity],
            "risk_reward_ratio": [risk_reward_ratio],
            "risk_state": [risk_state],
            "trade_state": [trade_state],
            "pyramid_state": [pyramid_state],
            "exit_action": [exit_action],
            "exit_reason": [exit_reason],
            "recommended_quantity": [recommended_quantity],
            "recommended_percent": [recommended_percent],
            "priority": [priority],
            "created_at": [open_time],
        },
        schema=dict(COLUMN_DTYPES),
    ).select(list(CANONICAL_COLUMN_ORDER))


# ---------------------------------------------------------------------------
# Clean frame passes
# ---------------------------------------------------------------------------


def test_verifier_passes_on_canonical_hold_frame() -> None:
    """A correctly formed HOLD frame passes all verifier checks."""
    verifier = ExitEngineVerifier()
    report = verifier.verify(_canonical_frame())
    assert report.passed is True
    assert report.rows_checked == 1
    assert report.null_rows == 0
    assert report.nan_rows == 0
    assert report.duplicate_timestamp_rows == 0
    assert report.invalid_timestamp_rows == 0
    assert report.invalid_numeric_rows == 0
    assert report.warnings == ()


def test_verifier_passes_on_full_exit_frame() -> None:
    """A FULL_EXIT/PORTFOLIO_SHUTDOWN frame passes all verifier checks."""
    verifier = ExitEngineVerifier()
    frame = _canonical_frame(
        exit_action=ExitAction.FULL_EXIT.value,
        exit_reason=ExitReason.PORTFOLIO_SHUTDOWN.value,
        recommended_quantity=1.0,
        recommended_percent=1.0,
        priority=1,
    )
    report = verifier.verify(frame)
    assert report.passed is True
    assert report.rows_checked == 1


def test_verifier_passes_on_partial_exit_frame() -> None:
    """A PARTIAL_EXIT/TAKE_PROFIT frame passes all verifier checks."""
    verifier = ExitEngineVerifier()
    frame = _canonical_frame(
        exit_action=ExitAction.PARTIAL_EXIT.value,
        exit_reason=ExitReason.TAKE_PROFIT.value,
        recommended_quantity=0.5,
        recommended_percent=0.5,
        priority=5,
    )
    report = verifier.verify(frame)
    assert report.passed is True


# ---------------------------------------------------------------------------
# Missing / mismatched columns
# ---------------------------------------------------------------------------


def test_verifier_raises_on_missing_required_column() -> None:
    """Missing required columns raise ExitEngineValidationError with ERROR_REQUIRED_COLUMNS."""
    verifier = ExitEngineVerifier()
    frame = _canonical_frame().drop("exit_action")
    with pytest.raises(ExitEngineValidationError) as exc_info:
        verifier.verify(frame)
    assert exc_info.value.error_code == ERROR_REQUIRED_COLUMNS


def test_verifier_raises_on_dtype_mismatch() -> None:
    """Wrong column dtype raises ExitEngineValidationError with ERROR_SCHEMA_MISMATCH."""
    verifier = ExitEngineVerifier()
    frame = _canonical_frame().with_columns(pl.col("priority").cast(pl.Float64))
    with pytest.raises(ExitEngineValidationError) as exc_info:
        verifier.verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH


# ---------------------------------------------------------------------------
# Duplicate primary keys
# ---------------------------------------------------------------------------


def test_verifier_warns_on_duplicate_primary_keys() -> None:
    """Duplicate (symbol, timeframe, open_time, position_id) emits a warning."""
    verifier = ExitEngineVerifier()
    row = _canonical_frame()
    duplicate = pl.concat([row, row])
    report = verifier.verify(duplicate)
    assert report.passed is False
    assert report.duplicate_timestamp_rows == 1
    assert any("Duplicate" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Null rows
# ---------------------------------------------------------------------------


def test_verifier_warns_on_null_values() -> None:
    """NULL values in required columns produce a null-row warning."""
    verifier = ExitEngineVerifier()
    frame = _canonical_frame().with_columns(pl.lit(None).alias("manager"))
    with pytest.raises(ExitEngineValidationError):
        verifier.verify(frame)


def test_verifier_counts_null_rows() -> None:
    """Rows with NULL values are counted in null_rows."""
    verifier = ExitEngineVerifier()
    frame = _canonical_frame().with_columns(pl.Series("risk_state", [None], dtype=pl.Utf8))
    report = verifier.verify(frame)
    assert report.null_rows == 1
    assert report.passed is False


# ---------------------------------------------------------------------------
# Invalid enum values
# ---------------------------------------------------------------------------


def test_verifier_warns_on_invalid_exit_action() -> None:
    """Non-canonical exit_action values emit an invalid-action warning."""
    verifier = ExitEngineVerifier()
    frame = _canonical_frame().with_columns(pl.lit("INVALID_ACTION").alias("exit_action"))
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("ExitAction" in w or "exit_action" in w.lower() for w in report.warnings)


def test_verifier_warns_on_invalid_exit_reason() -> None:
    """Non-canonical exit_reason values emit an invalid-reason warning."""
    verifier = ExitEngineVerifier()
    frame = _canonical_frame().with_columns(pl.lit("INVALID_REASON").alias("exit_reason"))
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("ExitReason" in w or "exit_reason" in w.lower() for w in report.warnings)


def test_verifier_warns_on_empty_exit_action() -> None:
    """Empty string in exit_action emits an empty-action warning."""
    verifier = ExitEngineVerifier()
    frame = _canonical_frame().with_columns(pl.lit("").alias("exit_action"))
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("exit_action" in w.lower() or "Empty" in w for w in report.warnings)


def test_verifier_warns_on_empty_exit_reason() -> None:
    """Empty string in exit_reason emits an empty-reason warning."""
    verifier = ExitEngineVerifier()
    frame = _canonical_frame().with_columns(pl.lit("").alias("exit_reason"))
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("exit_reason" in w.lower() or "Empty" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Non-negative quantity checks
# ---------------------------------------------------------------------------


def test_verifier_warns_on_negative_quantity() -> None:
    """Negative quantity triggers a negative-quantity warning."""
    verifier = ExitEngineVerifier()
    frame = _canonical_frame().with_columns(pl.lit(-1.0).alias("quantity"))
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("egative" in w for w in report.warnings)


def test_verifier_warns_on_negative_recommended_quantity() -> None:
    """Negative recommended_quantity triggers a negative-quantity warning."""
    verifier = ExitEngineVerifier()
    frame = _canonical_frame().with_columns(pl.lit(-0.5).alias("recommended_quantity"))
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("egative" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# recommended_percent range [0, 1]
# ---------------------------------------------------------------------------


def test_verifier_warns_on_percent_out_of_range() -> None:
    """recommended_percent outside [0, 1] emits a percent-range warning."""
    verifier = ExitEngineVerifier()
    for bad_pct in (-0.1, 1.1, 2.0):
        frame = _canonical_frame().with_columns(pl.lit(bad_pct).alias("recommended_percent"))
        report = verifier.verify(frame)
        assert report.passed is False, f"Expected failure for pct={bad_pct}"
        assert any("percent" in w.lower() or "recommended_percent" in w for w in report.warnings)


def test_verifier_passes_percent_boundary_values() -> None:
    """recommended_percent = 0.0 and 1.0 are valid boundary values."""
    verifier = ExitEngineVerifier()
    for boundary in (0.0, 1.0):
        frame = _canonical_frame().with_columns(pl.lit(boundary).alias("recommended_percent"))
        report = verifier.verify(frame)
        assert report.passed is True, f"Expected pass for pct={boundary}"


# ---------------------------------------------------------------------------
# Non-finite numeric values
# ---------------------------------------------------------------------------


def test_verifier_warns_on_nan_entry_price() -> None:
    """NaN entry_price triggers a non-finite value warning."""
    verifier = ExitEngineVerifier()
    frame = _canonical_frame().with_columns(pl.lit(float("nan")).alias("entry_price"))
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.nan_rows > 0 or report.invalid_numeric_rows > 0


def test_verifier_warns_on_inf_current_price() -> None:
    """Infinite current_price triggers a non-finite value warning."""
    verifier = ExitEngineVerifier()
    frame = _canonical_frame().with_columns(pl.lit(float("inf")).alias("current_price"))
    report = verifier.verify(frame)
    assert report.passed is False


# ---------------------------------------------------------------------------
# Invalid timestamps
# ---------------------------------------------------------------------------


def test_verifier_counts_invalid_timestamp_rows() -> None:
    """Rows with NULL open_time values are counted as invalid timestamps."""
    verifier = ExitEngineVerifier()
    frame = _canonical_frame().with_columns(
        pl.lit(None).cast(pl.Datetime("us", "UTC")).alias("open_time")
    )
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.invalid_timestamp_rows > 0 or report.null_rows > 0


# ---------------------------------------------------------------------------
# Column order check
# ---------------------------------------------------------------------------


def test_verifier_warns_on_non_canonical_column_order() -> None:
    """Frame with columns out of canonical order emits a column-order warning."""
    verifier = ExitEngineVerifier()
    columns = list(CANONICAL_COLUMN_ORDER)
    columns.reverse()
    frame = _canonical_frame().select(columns)
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("column order" in w.lower() or "Column" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Multiple rows
# ---------------------------------------------------------------------------


def test_verifier_aggregates_multiple_rows() -> None:
    """Verifier correctly aggregates checks across multiple rows."""
    verifier = ExitEngineVerifier()
    row_a = _canonical_frame(position_id="pos-00000001")
    row_b = _canonical_frame(position_id="pos-00000002")
    frame = pl.concat([row_a, row_b])
    report = verifier.verify(frame)
    assert report.rows_checked == 2
    assert report.passed is True


def test_verifier_incomplete_lineage_emits_warning() -> None:
    """Blank manager (lineage column) emits an incomplete-lineage warning."""
    verifier = ExitEngineVerifier()
    frame = _canonical_frame().with_columns(pl.lit("").alias("manager"))
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("lineage" in w.lower() or "Lineage" in w for w in report.warnings)
