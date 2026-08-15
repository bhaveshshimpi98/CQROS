"""Unit tests for CQROS ``MonitoringVerifier``."""

from __future__ import annotations

import math

import polars as pl
import pytest

from cqros.monitoring import MonitoringStatus, MonitoringValidationError, MonitoringVerifier
from cqros.monitoring.schema import CANONICAL_COLUMN_ORDER, COLUMN_DTYPES
from cqros.monitoring.verifier import ERROR_REQUIRED_COLUMNS, ERROR_SCHEMA_MISMATCH

_MANAGER = "simple"
_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1h"
_OPEN_TIME = 1_718_452_800_000


def _canonical_frame(
    *,
    symbol: str = _SYMBOL,
    timeframe: str = _TIMEFRAME,
    open_time: int = _OPEN_TIME,
    manager: str = _MANAGER,
    monitor_type: str = "system",
    monitor_name: str = "report_monitor",
    severity: str = "NORMAL",
    metric_name: str = "report_generation",
    metric_value: float = 1.0,
    threshold: float = 1.0,
    alert: bool = False,
    status: str = MonitoringStatus.NORMAL.value,
) -> pl.DataFrame:
    """Build a canonical monitoring frame that should pass verification."""
    return pl.DataFrame(
        {
            "symbol": [symbol],
            "timeframe": [timeframe],
            "open_time": [open_time],
            "manager": [manager],
            "monitor_type": [monitor_type],
            "monitor_name": [monitor_name],
            "severity": [severity],
            "metric_name": [metric_name],
            "metric_value": [metric_value],
            "threshold": [threshold],
            "alert": [alert],
            "status": [status],
        },
        schema=dict(COLUMN_DTYPES),
    ).select(list(CANONICAL_COLUMN_ORDER))


# ---------------------------------------------------------------------------
# Clean frame passes
# ---------------------------------------------------------------------------


def test_verifier_passes_on_canonical_frame() -> None:
    """A correctly formed monitoring frame passes all verifier checks."""
    verifier = MonitoringVerifier()
    report = verifier.verify(_canonical_frame())
    assert report.passed is True
    assert report.rows_checked == 1
    assert report.null_rows == 0
    assert report.nan_rows == 0
    assert report.duplicate_timestamp_rows == 0
    assert report.invalid_timestamp_rows == 0
    assert report.invalid_numeric_rows == 0
    assert report.warnings == ()


def test_verifier_passes_on_warning_status_frame() -> None:
    """A WARNING status frame passes all verifier checks."""
    verifier = MonitoringVerifier()
    frame = _canonical_frame(status=MonitoringStatus.WARNING.value)
    report = verifier.verify(frame)
    assert report.passed is True
    assert report.rows_checked == 1


def test_verifier_passes_on_critical_status_frame() -> None:
    """A CRITICAL status frame passes all verifier checks."""
    verifier = MonitoringVerifier()
    frame = _canonical_frame(status=MonitoringStatus.CRITICAL.value)
    report = verifier.verify(frame)
    assert report.passed is True
    assert report.rows_checked == 1


def test_verifier_pass_summary() -> None:
    """PASS summary reports zero defect counters and empty warnings."""
    verifier = MonitoringVerifier()
    report = verifier.verify(_canonical_frame())
    assert report.passed is True
    assert report.rows_checked == 1
    assert report.duplicate_timestamp_rows == 0
    assert report.null_rows == 0
    assert report.nan_rows == 0
    assert report.invalid_timestamp_rows == 0
    assert report.invalid_numeric_rows == 0
    assert report.warnings == ()


# ---------------------------------------------------------------------------
# Missing / mismatched columns
# ---------------------------------------------------------------------------


def test_verifier_raises_on_missing_required_column() -> None:
    """Missing required columns raise MonitoringValidationError."""
    verifier = MonitoringVerifier()
    frame = _canonical_frame().drop("monitor_name")
    with pytest.raises(MonitoringValidationError) as exc_info:
        verifier.verify(frame)
    assert exc_info.value.error_code == ERROR_REQUIRED_COLUMNS


def test_verifier_raises_on_dtype_mismatch() -> None:
    """Wrong column dtype raises MonitoringValidationError with ERROR_SCHEMA_MISMATCH."""
    verifier = MonitoringVerifier()
    frame = _canonical_frame().with_columns(pl.col("open_time").cast(pl.Float64))
    with pytest.raises(MonitoringValidationError) as exc_info:
        verifier.verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH


def test_verifier_raises_on_invalid_schema() -> None:
    """Schema mismatches raise MonitoringValidationError for invalid dtypes."""
    verifier = MonitoringVerifier()
    frame = _canonical_frame().with_columns(pl.col("metric_value").cast(pl.Utf8))
    with pytest.raises(MonitoringValidationError) as exc_info:
        verifier.verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH


def test_verifier_raises_on_invalid_alert_type() -> None:
    """Non-boolean alert dtype raises MonitoringValidationError."""
    verifier = MonitoringVerifier()
    frame = _canonical_frame().with_columns(pl.col("alert").cast(pl.Utf8))
    with pytest.raises(MonitoringValidationError) as exc_info:
        verifier.verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH


# ---------------------------------------------------------------------------
# Duplicate primary keys
# ---------------------------------------------------------------------------


def test_verifier_warns_on_duplicate_primary_keys() -> None:
    """Duplicate (symbol, timeframe, open_time) emits a warning."""
    verifier = MonitoringVerifier()
    row = _canonical_frame()
    duplicate = pl.concat([row, row])
    report = verifier.verify(duplicate)
    assert report.passed is False
    assert report.duplicate_timestamp_rows == 1
    assert any("Duplicate" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Null rows
# ---------------------------------------------------------------------------


def test_verifier_counts_null_rows_in_alert() -> None:
    """Rows with NULL alert are counted in null_rows."""
    verifier = MonitoringVerifier()
    frame = _canonical_frame().with_columns(pl.Series("alert", [None], dtype=pl.Boolean))
    report = verifier.verify(frame)
    assert report.null_rows == 1
    assert report.passed is False


def test_verifier_counts_null_rows_in_status() -> None:
    """Rows with NULL status are counted in null_rows."""
    verifier = MonitoringVerifier()
    frame = _canonical_frame().with_columns(pl.Series("status", [None], dtype=pl.Utf8))
    report = verifier.verify(frame)
    assert report.null_rows == 1
    assert report.passed is False


# ---------------------------------------------------------------------------
# Invalid enum values
# ---------------------------------------------------------------------------


def test_verifier_warns_on_invalid_status() -> None:
    """Non-canonical status values emit an invalid-status warning."""
    verifier = MonitoringVerifier()
    frame = _canonical_frame().with_columns(pl.lit("INVALID_STATUS").alias("status"))
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("MonitoringStatus" in w or "status" in w.lower() for w in report.warnings)


def test_verifier_warns_on_empty_status() -> None:
    """Empty string in status emits an empty-status warning."""
    verifier = MonitoringVerifier()
    frame = _canonical_frame().with_columns(pl.lit("").alias("status"))
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("status" in w.lower() or "Empty" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Empty required monitor metadata fields
# ---------------------------------------------------------------------------


def test_verifier_warns_on_empty_monitor_type() -> None:
    """Empty monitor_type emits a monitor-metadata warning."""
    verifier = MonitoringVerifier()
    frame = _canonical_frame(monitor_type="")
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("monitor metadata" in w.lower() or "Empty" in w for w in report.warnings)


def test_verifier_warns_on_empty_monitor_name() -> None:
    """Empty monitor_name emits a monitor-metadata warning."""
    verifier = MonitoringVerifier()
    frame = _canonical_frame(monitor_name="")
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("monitor metadata" in w.lower() or "Empty" in w for w in report.warnings)


def test_verifier_warns_on_empty_severity() -> None:
    """Empty severity emits a monitor-metadata warning."""
    verifier = MonitoringVerifier()
    frame = _canonical_frame(severity="")
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("monitor metadata" in w.lower() or "Empty" in w for w in report.warnings)


def test_verifier_warns_on_empty_metric_name() -> None:
    """Empty metric_name emits a monitor-metadata warning."""
    verifier = MonitoringVerifier()
    frame = _canonical_frame(metric_name="")
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("monitor metadata" in w.lower() or "Empty" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Non-finite numeric values
# ---------------------------------------------------------------------------


def test_verifier_warns_on_non_finite_metric_value() -> None:
    """Non-finite metric_value emits a numeric warning."""
    verifier = MonitoringVerifier()
    frame = _canonical_frame(metric_value=math.inf)
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.invalid_numeric_rows == 1
    assert any("finite" in w.lower() or "NaN" in w for w in report.warnings)


def test_verifier_warns_on_nan_threshold() -> None:
    """NaN threshold emits a numeric or NaN warning."""
    verifier = MonitoringVerifier()
    frame = _canonical_frame(threshold=math.nan)
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.invalid_numeric_rows > 0 or report.nan_rows > 0
    assert report.warnings != ()


def test_verifier_passes_when_metric_values_are_finite() -> None:
    """Finite metric_value and threshold values are valid."""
    verifier = MonitoringVerifier()
    frame = _canonical_frame(metric_value=0.5, threshold=1.0)
    report = verifier.verify(frame)
    assert report.passed is True


# ---------------------------------------------------------------------------
# Invalid timestamps
# ---------------------------------------------------------------------------


def test_verifier_counts_invalid_timestamp_rows() -> None:
    """Rows with NULL open_time values are counted as invalid timestamps."""
    verifier = MonitoringVerifier()
    frame = _canonical_frame().with_columns(pl.lit(None).cast(pl.Int64).alias("open_time"))
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.invalid_timestamp_rows > 0 or report.null_rows > 0


# ---------------------------------------------------------------------------
# Column order and sorting
# ---------------------------------------------------------------------------


def test_verifier_warns_on_non_canonical_column_order() -> None:
    """Frame with columns out of canonical order emits a column-order warning."""
    verifier = MonitoringVerifier()
    columns = list(CANONICAL_COLUMN_ORDER)
    columns.reverse()
    frame = _canonical_frame().select(columns)
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("column order" in w.lower() or "Column" in w for w in report.warnings)


def test_verifier_warns_on_unsorted_open_time() -> None:
    """Frame not sorted by open_time emits an unsorted warning."""
    verifier = MonitoringVerifier()
    t0 = _OPEN_TIME
    t1 = _OPEN_TIME + 3_600_000
    row_a = _canonical_frame(open_time=t1, symbol="BTCUSDT")
    row_b = _canonical_frame(open_time=t0, symbol="ETHUSDT")
    frame = pl.concat([row_a, row_b])
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("sorted" in w.lower() for w in report.warnings)


# ---------------------------------------------------------------------------
# Multiple rows, lineage, and FAIL summary
# ---------------------------------------------------------------------------


def test_verifier_aggregates_multiple_rows() -> None:
    """Verifier correctly aggregates checks across multiple rows."""
    verifier = MonitoringVerifier()
    t0 = _OPEN_TIME
    t1 = _OPEN_TIME + 3_600_000
    row_a = _canonical_frame(open_time=t0, status=MonitoringStatus.NORMAL.value)
    row_b = _canonical_frame(open_time=t1, symbol="ETHUSDT")
    frame = pl.concat([row_a, row_b])
    report = verifier.verify(frame)
    assert report.rows_checked == 2
    assert report.passed is True


def test_verifier_incomplete_lineage_emits_warning() -> None:
    """Blank manager (lineage column) emits an incomplete-lineage warning."""
    verifier = MonitoringVerifier()
    frame = _canonical_frame().with_columns(pl.lit("").alias("manager"))
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("lineage" in w.lower() or "Lineage" in w for w in report.warnings)


def test_verifier_fail_summary() -> None:
    """FAIL summary reports defect counters and non-empty warnings."""
    verifier = MonitoringVerifier()
    frame = _canonical_frame(metric_value=math.inf)
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.rows_checked == 1
    assert report.invalid_numeric_rows > 0
    assert report.warnings != ()
