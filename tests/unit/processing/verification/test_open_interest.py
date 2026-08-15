"""Unit tests for CQROS processing ``OpenInterestVerifier``."""

from __future__ import annotations

import math

import polars as pl
import pytest

from cqros.processing.verification import OpenInterestVerifier, VerificationReport
from cqros.processing.verification.exceptions import (
    ERROR_REQUIRED_COLUMNS,
    ProcessingValidationError,
)
from cqros.processing.verification.interfaces import DataVerifier
from cqros.processing.verification.open_interest import (
    OpenInterestVerifier as OpenInterestVerifierFromModule,
)

_START = 1_700_000_000_000
_INTERVAL = 60_000


def _open_interest_frame(
    *,
    timestamps: list[int] | None = None,
    open_interests: list[float] | None = None,
) -> pl.DataFrame:
    """Build a canonical open-interest verification frame."""
    if timestamps is None:
        timestamps = [_START, _START + _INTERVAL, _START + 2 * _INTERVAL]
    row_count = len(timestamps)
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open_interest": (
                open_interests if open_interests is not None else [1000.0] * row_count
            ),
        },
        schema={
            "timestamp": pl.Int64,
            "open_interest": pl.Float64,
        },
    )


def _verifier() -> OpenInterestVerifier:
    """Build an OpenInterestVerifier instance."""
    return OpenInterestVerifier()


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


def test_package_exports_open_interest_verifier() -> None:
    """Package re-export matches the open_interest module symbol."""
    assert OpenInterestVerifier is OpenInterestVerifierFromModule


def test_open_interest_verifier_satisfies_data_verifier_protocol() -> None:
    """OpenInterestVerifier structurally satisfies DataVerifier."""
    assert isinstance(_verifier(), DataVerifier)


def test_valid_frame_passes() -> None:
    """A clean sorted open-interest frame passes verification."""
    report = _verifier().verify(_open_interest_frame())
    _assert_clean_pass(report, rows=3)


def test_duplicate_timestamps() -> None:
    """Duplicate timestamps are counted and fail verification."""
    frame = _open_interest_frame(timestamps=[_START, _START, _START + _INTERVAL])
    report = _verifier().verify(frame)
    assert report.duplicate_timestamp_rows == 1
    assert report.passed is False
    assert "Duplicate timestamps detected." in report.warnings


def test_null_open_interest() -> None:
    """NULL open_interest values are counted."""
    frame = _open_interest_frame(open_interests=[1000.0, None, 1000.0])  # type: ignore[list-item]
    report = _verifier().verify(frame)
    assert report.null_rows == 1
    assert report.passed is False
    assert "Rows containing NULL values." in report.warnings


def test_nan_open_interest() -> None:
    """NaN open_interest values are counted."""
    frame = _open_interest_frame(open_interests=[1000.0, math.nan, 1000.0])
    report = _verifier().verify(frame)
    assert report.nan_rows == 1
    assert report.passed is False
    assert "Rows containing NaN values." in report.warnings


def test_negative_open_interest() -> None:
    """Negative open_interest fails numeric verification."""
    frame = _open_interest_frame(open_interests=[1000.0, -1.0, 1000.0])
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 1
    assert report.passed is False
    assert "Invalid open_interest values." in report.warnings


def test_zero_open_interest_allowed() -> None:
    """Zero open_interest is valid and does not fail verification."""
    frame = _open_interest_frame(open_interests=[1000.0, 0.0, 1000.0])
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=3)
    assert report.invalid_numeric_rows == 0


def test_invalid_timestamps() -> None:
    """Non-positive timestamps are counted as invalid."""
    frame = _open_interest_frame(timestamps=[_START, 0, _START + 2 * _INTERVAL])
    report = _verifier().verify(frame)
    assert report.invalid_timestamp_rows == 1
    assert report.passed is False
    assert "Invalid timestamps detected." in report.warnings


def test_unsorted_frame() -> None:
    """Unsorted timestamps fail without incrementing counters."""
    frame = _open_interest_frame(timestamps=[_START, _START + 2 * _INTERVAL, _START + _INTERVAL])
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
    frame = pl.DataFrame({"timestamp": [1]})
    with pytest.raises(ProcessingValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_REQUIRED_COLUMNS
    assert "open_interest" in dict(exc_info.value.details)["missing_columns"]


def test_empty_frame_passes() -> None:
    """An empty schema-valid frame passes verification."""
    frame = pl.DataFrame(
        schema={
            "timestamp": pl.Int64,
            "open_interest": pl.Float64,
        }
    )
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=0)


def test_immutability() -> None:
    """verify does not mutate the input DataFrame."""
    frame = _open_interest_frame(
        timestamps=[_START, _START, 0],
        open_interests=[1000.0, None, -1.0],  # type: ignore[list-item]
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
            "open_interest": [1000.0, None, -5.0],
        },
        schema={"timestamp": pl.Int64, "open_interest": pl.Float64},
    )
    report = _verifier().verify(frame)
    assert report.rows_checked == 3
    assert report.duplicate_timestamp_rows == 1
    assert report.null_rows == 1
    assert report.invalid_numeric_rows == 1
    assert report.passed is False
    assert "Duplicate timestamps detected." in report.warnings
    assert "Rows containing NULL values." in report.warnings
    assert "Invalid open_interest values." in report.warnings
    assert "Frame is not sorted by timestamp." in report.warnings
    assert isinstance(report.warnings, tuple)
