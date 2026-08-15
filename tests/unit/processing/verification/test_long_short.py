"""Unit tests for CQROS processing ``LongShortVerifier``."""

from __future__ import annotations

import math

import polars as pl
import pytest

from cqros.processing.verification import LongShortVerifier, VerificationReport
from cqros.processing.verification.exceptions import (
    ERROR_REQUIRED_COLUMNS,
    ProcessingValidationError,
)
from cqros.processing.verification.interfaces import DataVerifier
from cqros.processing.verification.long_short import (
    LongShortVerifier as LongShortVerifierFromModule,
)

_START = 1_700_000_000_000
_INTERVAL = 60_000


def _long_short_frame(
    *,
    timestamps: list[int] | None = None,
    long_accounts: list[float | None] | None = None,
    short_accounts: list[float | None] | None = None,
    long_short_ratios: list[float | None] | None = None,
) -> pl.DataFrame:
    """Build a canonical long/short verification frame."""
    if timestamps is None:
        timestamps = [_START, _START + _INTERVAL, _START + 2 * _INTERVAL]
    row_count = len(timestamps)
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "long_account": (long_accounts if long_accounts is not None else [0.55] * row_count),
            "short_account": (short_accounts if short_accounts is not None else [0.45] * row_count),
            "long_short_ratio": (
                long_short_ratios if long_short_ratios is not None else [1.22] * row_count
            ),
        },
        schema={
            "timestamp": pl.Int64,
            "long_account": pl.Float64,
            "short_account": pl.Float64,
            "long_short_ratio": pl.Float64,
        },
    )


def _verifier() -> LongShortVerifier:
    """Build a LongShortVerifier instance."""
    return LongShortVerifier()


def _assert_clean_pass(report: VerificationReport, *, rows: int) -> None:
    """Assert a fully passing report for ``rows`` checked."""
    assert report == VerificationReport(
        rows_checked=rows,
        duplicate_timestamp_rows=0,
        null_rows=0,
        nan_rows=0,
        invalid_timestamp_rows=0,
        invalid_numeric_rows=0,
        warnings=(),
        passed=True,
    )


def test_package_exports_long_short_verifier() -> None:
    """Package re-export matches the long_short module symbol."""
    assert LongShortVerifier is LongShortVerifierFromModule


def test_long_short_verifier_satisfies_data_verifier_protocol() -> None:
    """LongShortVerifier structurally satisfies DataVerifier."""
    assert isinstance(_verifier(), DataVerifier)


def test_valid_frame_passes() -> None:
    """A clean sorted long/short frame passes verification."""
    report = _verifier().verify(_long_short_frame())
    _assert_clean_pass(report, rows=3)


def test_duplicate_timestamps() -> None:
    """Duplicate timestamps are counted and fail verification."""
    frame = _long_short_frame(timestamps=[_START, _START, _START + _INTERVAL])
    report = _verifier().verify(frame)
    assert report.duplicate_timestamp_rows == 1
    assert report.passed is False
    assert "Duplicate timestamps detected." in report.warnings


def test_null_values() -> None:
    """NULL long_account values are counted."""
    frame = _long_short_frame(long_accounts=[0.55, None, 0.55])
    report = _verifier().verify(frame)
    assert report.null_rows == 1
    assert report.passed is False
    assert "Rows containing NULL values." in report.warnings


def test_nan_values() -> None:
    """NaN short_account values are counted."""
    frame = _long_short_frame(short_accounts=[0.45, math.nan, 0.45])
    report = _verifier().verify(frame)
    assert report.nan_rows == 1
    assert report.passed is False
    assert "Rows containing NaN values." in report.warnings


def test_negative_long_account() -> None:
    """Negative long_account fails numeric verification."""
    frame = _long_short_frame(long_accounts=[0.55, -0.1, 0.55])
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 1
    assert report.passed is False
    assert "Invalid long/short numeric values." in report.warnings


def test_negative_short_account() -> None:
    """Negative short_account fails numeric verification."""
    frame = _long_short_frame(short_accounts=[0.45, -0.2, 0.45])
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 1
    assert report.passed is False


def test_negative_long_short_ratio() -> None:
    """Negative long_short_ratio fails numeric verification."""
    frame = _long_short_frame(long_short_ratios=[1.22, -1.0, 1.22])
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 1
    assert report.passed is False


def test_zero_values_allowed() -> None:
    """Zero account and ratio values are valid."""
    frame = _long_short_frame(
        long_accounts=[0.0, 0.55, 0.0],
        short_accounts=[0.0, 0.45, 0.0],
        long_short_ratios=[0.0, 1.22, 0.0],
    )
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=3)
    assert report.invalid_numeric_rows == 0


def test_invalid_timestamps() -> None:
    """Non-positive timestamps are counted as invalid."""
    frame = _long_short_frame(timestamps=[_START, 0, _START + 2 * _INTERVAL])
    report = _verifier().verify(frame)
    assert report.invalid_timestamp_rows == 1
    assert report.passed is False
    assert "Invalid timestamps detected." in report.warnings


def test_unsorted_frame() -> None:
    """Unsorted timestamps fail without incrementing counters."""
    frame = _long_short_frame(timestamps=[_START, _START + 2 * _INTERVAL, _START + _INTERVAL])
    report = _verifier().verify(frame)
    assert report.duplicate_timestamp_rows == 0
    assert report.null_rows == 0
    assert report.nan_rows == 0
    assert report.invalid_timestamp_rows == 0
    assert report.invalid_numeric_rows == 0
    assert report.passed is False
    assert report.warnings == ("Frame is not sorted by timestamp.",)


def test_missing_columns() -> None:
    """Missing required columns raise ProcessingValidationError."""
    frame = pl.DataFrame({"timestamp": [1], "long_account": [0.5]})
    with pytest.raises(ProcessingValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_REQUIRED_COLUMNS
    missing = dict(exc_info.value.details)["missing_columns"]
    assert "short_account" in missing
    assert "long_short_ratio" in missing


def test_empty_frame_passes() -> None:
    """An empty schema-valid frame passes verification."""
    frame = pl.DataFrame(
        schema={
            "timestamp": pl.Int64,
            "long_account": pl.Float64,
            "short_account": pl.Float64,
            "long_short_ratio": pl.Float64,
        }
    )
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=0)


def test_immutability() -> None:
    """verify does not mutate the input DataFrame."""
    frame = _long_short_frame(
        timestamps=[_START, _START, 0],
        long_accounts=[0.55, None, -0.1],
        short_accounts=[0.45, 0.45, 0.45],
        long_short_ratios=[1.22, 1.22, -1.0],
    )
    original = frame.clone()
    report = _verifier().verify(frame)
    assert frame.equals(original)
    assert report.passed is False


def test_report_correctness_combined() -> None:
    """Combined failures populate report fields correctly."""
    frame = pl.DataFrame(
        {
            "timestamp": [_START + _INTERVAL, _START, _START],
            "long_account": [0.55, None, -0.1],
            "short_account": [0.45, 0.45, 0.45],
            "long_short_ratio": [1.22, 1.22, 1.22],
        },
        schema={
            "timestamp": pl.Int64,
            "long_account": pl.Float64,
            "short_account": pl.Float64,
            "long_short_ratio": pl.Float64,
        },
    )
    report = _verifier().verify(frame)
    assert report.rows_checked == 3
    assert report.duplicate_timestamp_rows == 1
    assert report.null_rows == 1
    assert report.invalid_numeric_rows == 1
    assert report.passed is False
    assert "Duplicate timestamps detected." in report.warnings
    assert "Rows containing NULL values." in report.warnings
    assert "Invalid long/short numeric values." in report.warnings
    assert "Frame is not sorted by timestamp." in report.warnings
    assert isinstance(report.warnings, tuple)


def test_multiple_numeric_failures_counted_once() -> None:
    """A row violating multiple numeric rules is counted once."""
    frame = _long_short_frame(
        timestamps=[_START],
        long_accounts=[-1.0],
        short_accounts=[-2.0],
        long_short_ratios=[-3.0],
    )
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 1
    assert report.passed is False
