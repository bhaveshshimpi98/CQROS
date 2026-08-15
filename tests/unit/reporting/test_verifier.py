"""Unit tests for CQROS ``ReportingVerifier``."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.reporting import ReportingStatus, ReportingValidationError, ReportingVerifier
from cqros.reporting.schema import CANONICAL_COLUMN_ORDER, COLUMN_DTYPES
from cqros.reporting.verifier import ERROR_REQUIRED_COLUMNS, ERROR_SCHEMA_MISMATCH

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
    report_name: str = "performance_report",
    report_type: str = "analytics",
    report_format: str = "parquet",
    report_version: str = "v1",
    report_path: str = "",
    generated_at: int | None = None,
    status: str = ReportingStatus.GENERATED.value,
) -> pl.DataFrame:
    """Build a canonical reporting frame that should pass verification."""
    resolved_generated_at = open_time if generated_at is None else generated_at
    return pl.DataFrame(
        {
            "symbol": [symbol],
            "timeframe": [timeframe],
            "open_time": [open_time],
            "manager": [manager],
            "report_name": [report_name],
            "report_type": [report_type],
            "report_format": [report_format],
            "report_version": [report_version],
            "report_path": [report_path],
            "generated_at": [resolved_generated_at],
            "status": [status],
        },
        schema=dict(COLUMN_DTYPES),
    ).select(list(CANONICAL_COLUMN_ORDER))


# ---------------------------------------------------------------------------
# Clean frame passes
# ---------------------------------------------------------------------------


def test_verifier_passes_on_canonical_frame() -> None:
    """A correctly formed reporting frame passes all verifier checks."""
    verifier = ReportingVerifier()
    report = verifier.verify(_canonical_frame())
    assert report.passed is True
    assert report.rows_checked == 1
    assert report.null_rows == 0
    assert report.nan_rows == 0
    assert report.duplicate_timestamp_rows == 0
    assert report.invalid_timestamp_rows == 0
    assert report.invalid_numeric_rows == 0
    assert report.warnings == ()


def test_verifier_passes_on_failed_status_frame() -> None:
    """A FAILED status frame passes all verifier checks."""
    verifier = ReportingVerifier()
    frame = _canonical_frame(status=ReportingStatus.FAILED.value)
    report = verifier.verify(frame)
    assert report.passed is True
    assert report.rows_checked == 1


def test_verifier_passes_with_empty_report_path() -> None:
    """Empty report_path strings are allowed and do not fail verification."""
    verifier = ReportingVerifier()
    frame = _canonical_frame(report_path="")
    report = verifier.verify(frame)
    assert report.passed is True


def test_verifier_pass_summary() -> None:
    """PASS summary reports zero defect counters and empty warnings."""
    verifier = ReportingVerifier()
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
    """Missing required columns raise ReportingValidationError."""
    verifier = ReportingVerifier()
    frame = _canonical_frame().drop("report_name")
    with pytest.raises(ReportingValidationError) as exc_info:
        verifier.verify(frame)
    assert exc_info.value.error_code == ERROR_REQUIRED_COLUMNS


def test_verifier_raises_on_dtype_mismatch() -> None:
    """Wrong column dtype raises ReportingValidationError with ERROR_SCHEMA_MISMATCH."""
    verifier = ReportingVerifier()
    frame = _canonical_frame().with_columns(pl.col("open_time").cast(pl.Float64))
    with pytest.raises(ReportingValidationError) as exc_info:
        verifier.verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH


def test_verifier_raises_on_invalid_schema() -> None:
    """Schema mismatches raise ReportingValidationError for invalid dtypes."""
    verifier = ReportingVerifier()
    frame = _canonical_frame().with_columns(pl.col("generated_at").cast(pl.Float64))
    with pytest.raises(ReportingValidationError) as exc_info:
        verifier.verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH


# ---------------------------------------------------------------------------
# Duplicate primary keys
# ---------------------------------------------------------------------------


def test_verifier_warns_on_duplicate_primary_keys() -> None:
    """Duplicate (symbol, timeframe, open_time) emits a warning."""
    verifier = ReportingVerifier()
    row = _canonical_frame()
    duplicate = pl.concat([row, row])
    report = verifier.verify(duplicate)
    assert report.passed is False
    assert report.duplicate_timestamp_rows == 1
    assert any("Duplicate" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Null rows
# ---------------------------------------------------------------------------


def test_verifier_counts_null_rows_in_report_path() -> None:
    """Rows with NULL report_path are counted in null_rows."""
    verifier = ReportingVerifier()
    frame = _canonical_frame().with_columns(pl.Series("report_path", [None], dtype=pl.Utf8))
    report = verifier.verify(frame)
    assert report.null_rows == 1
    assert report.passed is False


def test_verifier_counts_null_rows_in_status() -> None:
    """Rows with NULL status are counted in null_rows."""
    verifier = ReportingVerifier()
    frame = _canonical_frame().with_columns(pl.Series("status", [None], dtype=pl.Utf8))
    report = verifier.verify(frame)
    assert report.null_rows == 1
    assert report.passed is False


# ---------------------------------------------------------------------------
# Invalid enum values
# ---------------------------------------------------------------------------


def test_verifier_warns_on_invalid_status() -> None:
    """Non-canonical status values emit an invalid-status warning."""
    verifier = ReportingVerifier()
    frame = _canonical_frame().with_columns(pl.lit("INVALID_STATUS").alias("status"))
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("ReportingStatus" in w or "status" in w.lower() for w in report.warnings)


def test_verifier_warns_on_empty_status() -> None:
    """Empty string in status emits an empty-status warning."""
    verifier = ReportingVerifier()
    frame = _canonical_frame().with_columns(pl.lit("").alias("status"))
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("status" in w.lower() or "Empty" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Empty required report metadata fields
# ---------------------------------------------------------------------------


def test_verifier_warns_on_empty_report_name() -> None:
    """Empty report_name emits a report-metadata warning."""
    verifier = ReportingVerifier()
    frame = _canonical_frame(report_name="")
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("report metadata" in w.lower() or "Empty" in w for w in report.warnings)


def test_verifier_warns_on_empty_report_type() -> None:
    """Empty report_type emits a report-metadata warning."""
    verifier = ReportingVerifier()
    frame = _canonical_frame(report_type="")
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("report metadata" in w.lower() or "Empty" in w for w in report.warnings)


def test_verifier_warns_on_empty_report_format() -> None:
    """Empty report_format emits a report-metadata warning."""
    verifier = ReportingVerifier()
    frame = _canonical_frame(report_format="")
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("report metadata" in w.lower() or "Empty" in w for w in report.warnings)


def test_verifier_warns_on_empty_report_version() -> None:
    """Empty report_version emits a report-metadata warning."""
    verifier = ReportingVerifier()
    frame = _canonical_frame(report_version="")
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("report metadata" in w.lower() or "Empty" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# generated_at ordering
# ---------------------------------------------------------------------------


def test_verifier_warns_on_generated_at_before_open_time() -> None:
    """generated_at values before open_time emit an ordering warning."""
    verifier = ReportingVerifier()
    frame = _canonical_frame(generated_at=_OPEN_TIME - 1)
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.invalid_numeric_rows == 1
    assert any("generated_at" in w for w in report.warnings)


def test_verifier_passes_when_generated_at_equals_open_time() -> None:
    """generated_at equal to open_time is valid."""
    verifier = ReportingVerifier()
    frame = _canonical_frame(generated_at=_OPEN_TIME)
    report = verifier.verify(frame)
    assert report.passed is True


def test_verifier_passes_when_generated_at_after_open_time() -> None:
    """generated_at after open_time is valid."""
    verifier = ReportingVerifier()
    frame = _canonical_frame(generated_at=_OPEN_TIME + 1)
    report = verifier.verify(frame)
    assert report.passed is True


# ---------------------------------------------------------------------------
# Invalid timestamps
# ---------------------------------------------------------------------------


def test_verifier_counts_invalid_timestamp_rows() -> None:
    """Rows with NULL open_time values are counted as invalid timestamps."""
    verifier = ReportingVerifier()
    frame = _canonical_frame().with_columns(pl.lit(None).cast(pl.Int64).alias("open_time"))
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.invalid_timestamp_rows > 0 or report.null_rows > 0


# ---------------------------------------------------------------------------
# Column order and sorting
# ---------------------------------------------------------------------------


def test_verifier_warns_on_non_canonical_column_order() -> None:
    """Frame with columns out of canonical order emits a column-order warning."""
    verifier = ReportingVerifier()
    columns = list(CANONICAL_COLUMN_ORDER)
    columns.reverse()
    frame = _canonical_frame().select(columns)
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("column order" in w.lower() or "Column" in w for w in report.warnings)


def test_verifier_warns_on_unsorted_open_time() -> None:
    """Frame not sorted by open_time emits an unsorted warning."""
    verifier = ReportingVerifier()
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
    verifier = ReportingVerifier()
    t0 = _OPEN_TIME
    t1 = _OPEN_TIME + 3_600_000
    row_a = _canonical_frame(open_time=t0, status=ReportingStatus.GENERATED.value)
    row_b = _canonical_frame(open_time=t1, symbol="ETHUSDT")
    frame = pl.concat([row_a, row_b])
    report = verifier.verify(frame)
    assert report.rows_checked == 2
    assert report.passed is True


def test_verifier_incomplete_lineage_emits_warning() -> None:
    """Blank manager (lineage column) emits an incomplete-lineage warning."""
    verifier = ReportingVerifier()
    frame = _canonical_frame().with_columns(pl.lit("").alias("manager"))
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("lineage" in w.lower() or "Lineage" in w for w in report.warnings)


def test_verifier_fail_summary() -> None:
    """FAIL summary reports defect counters and non-empty warnings."""
    verifier = ReportingVerifier()
    frame = _canonical_frame(generated_at=_OPEN_TIME - 100)
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.rows_checked == 1
    assert report.invalid_numeric_rows > 0
    assert report.warnings != ()
