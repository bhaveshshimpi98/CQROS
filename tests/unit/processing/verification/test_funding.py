"""Unit tests for CQROS processing ``FundingVerifier``."""

from __future__ import annotations

import math
from typing import cast

import polars as pl
import pytest

from cqros.processing.verification import FundingVerifier, VerificationReport
from cqros.processing.verification.exceptions import (
    ERROR_REQUIRED_COLUMNS,
    ProcessingValidationError,
)
from cqros.processing.verification.funding import FundingVerifier as FundingVerifierFromModule
from cqros.processing.verification.interfaces import DataVerifier

_START = 1_700_000_000_000
_INTERVAL = 8 * 60 * 60 * 1000


def _funding_frame(
    *,
    funding_times: list[int] | None = None,
    funding_rates: list[float] | None = None,
    mark_prices: list[float] | None = None,
) -> pl.DataFrame:
    """Build a canonical funding verification frame."""
    if funding_times is None:
        funding_times = [_START, _START + _INTERVAL, _START + 2 * _INTERVAL]
    row_count = len(funding_times)
    return pl.DataFrame(
        {
            "funding_time": funding_times,
            "funding_rate": (funding_rates if funding_rates is not None else [0.0001] * row_count),
            "mark_price": mark_prices if mark_prices is not None else [42000.0] * row_count,
        },
        schema={
            "funding_time": pl.Int64,
            "funding_rate": pl.Float64,
            "mark_price": pl.Float64,
        },
    )


def _verifier() -> FundingVerifier:
    """Build a FundingVerifier instance."""
    return FundingVerifier()


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


def test_package_exports_funding_verifier() -> None:
    """Package re-export matches the funding module symbol."""
    assert FundingVerifier is FundingVerifierFromModule


def test_funding_verifier_satisfies_data_verifier_protocol() -> None:
    """FundingVerifier structurally satisfies DataVerifier."""
    assert isinstance(_verifier(), DataVerifier)


def test_valid_frame_passes() -> None:
    """A clean sorted funding frame passes verification."""
    report = _verifier().verify(_funding_frame())
    _assert_clean_pass(report, rows=3)


def test_duplicate_timestamps() -> None:
    """Duplicate funding_time values are counted and fail verification."""
    frame = _funding_frame(funding_times=[_START, _START, _START + _INTERVAL])
    report = _verifier().verify(frame)
    assert report.duplicate_timestamp_rows == 1
    assert report.passed is False
    assert "Duplicate timestamps detected." in report.warnings


def test_null_funding_rate() -> None:
    """NULL funding_rate values are counted."""
    frame = _funding_frame(funding_rates=[0.0001, None, 0.0001])  # type: ignore[list-item]
    report = _verifier().verify(frame)
    assert report.null_rows == 1
    assert report.passed is False
    assert "Rows containing NULL values." in report.warnings


def test_null_mark_price_allowed() -> None:
    """NULL mark_price is valid and does not increment null_rows."""
    frame = _funding_frame(mark_prices=[42000.0, None, 42000.0])  # type: ignore[list-item]
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=3)
    assert report.null_rows == 0
    assert report.invalid_numeric_rows == 0


def test_all_null_mark_price_allowed() -> None:
    """A frame where every mark_price is null still passes when otherwise valid."""
    frame = _funding_frame(mark_prices=[None, None, None])  # type: ignore[list-item]
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=3)


def test_null_mark_price_with_negative_non_null_counts_once() -> None:
    """Null mark_price is ignored; negative non-null mark_price fails numeric checks."""
    frame = _funding_frame(mark_prices=[None, -1.0, 42000.0])  # type: ignore[list-item]
    report = _verifier().verify(frame)
    assert report.null_rows == 0
    assert report.invalid_numeric_rows == 1
    assert report.passed is False
    assert "Invalid mark_price values." in report.warnings
    assert "Rows containing NULL values." not in report.warnings


def test_nan_funding_rate() -> None:
    """NaN funding_rate values are counted."""
    frame = _funding_frame(funding_rates=[0.0001, math.nan, 0.0001])
    report = _verifier().verify(frame)
    assert report.nan_rows == 1
    assert report.passed is False
    assert "Rows containing NaN values." in report.warnings


def test_negative_funding_rate_allowed() -> None:
    """Negative funding_rate values are valid and do not fail verification."""
    frame = _funding_frame(funding_rates=[0.0001, -0.0005, 0.0002])
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=3)
    assert report.invalid_numeric_rows == 0


def test_negative_mark_price() -> None:
    """Negative mark_price fails numeric verification."""
    frame = _funding_frame(mark_prices=[42000.0, -1.0, 42000.0])
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 1
    assert report.passed is False
    assert "Invalid mark_price values." in report.warnings


def test_nan_mark_price() -> None:
    """NaN mark_price values are counted as NaN rows."""
    frame = _funding_frame(mark_prices=[42000.0, math.nan, 42000.0])
    report = _verifier().verify(frame)
    assert report.nan_rows == 1
    assert report.invalid_numeric_rows == 0
    assert report.passed is False
    assert "Rows containing NaN values." in report.warnings


def test_invalid_timestamps() -> None:
    """Non-positive funding_time values are counted as invalid."""
    frame = _funding_frame(funding_times=[_START, 0, _START + 2 * _INTERVAL])
    report = _verifier().verify(frame)
    assert report.invalid_timestamp_rows == 1
    assert report.passed is False
    assert "Invalid timestamps detected." in report.warnings


def test_unsorted_frame() -> None:
    """Unsorted funding_time fails without incrementing counters."""
    frame = _funding_frame(funding_times=[_START, _START + 2 * _INTERVAL, _START + _INTERVAL])
    report = _verifier().verify(frame)
    assert report.duplicate_timestamp_rows == 0
    assert report.null_rows == 0
    assert report.nan_rows == 0
    assert report.invalid_timestamp_rows == 0
    assert report.invalid_numeric_rows == 0
    assert report.passed is False
    assert report.warnings == ("Frame is not sorted by funding_time.",)


def test_missing_columns() -> None:
    """Missing required columns raise ProcessingValidationError."""
    frame = pl.DataFrame({"funding_time": [1], "funding_rate": [0.1]})
    with pytest.raises(ProcessingValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_REQUIRED_COLUMNS
    missing = cast(tuple[str, ...], dict(exc_info.value.details)["missing_columns"])
    assert "mark_price" in missing


def test_empty_frame_passes() -> None:
    """An empty schema-valid frame passes verification."""
    frame = pl.DataFrame(
        schema={
            "funding_time": pl.Int64,
            "funding_rate": pl.Float64,
            "mark_price": pl.Float64,
        }
    )
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=0)


def test_immutability() -> None:
    """verify does not mutate the input DataFrame."""
    frame = _funding_frame(
        funding_times=[_START, _START, 0],
        funding_rates=[0.0001, None, math.nan],  # type: ignore[list-item]
        mark_prices=[42000.0, -1.0, 42000.0],
    )
    original = frame.clone()
    report = _verifier().verify(frame)
    assert frame.equals(original)
    assert report.passed is False


def test_report_values_combined() -> None:
    """Combined failures populate report fields correctly."""
    frame = _funding_frame(
        funding_times=[_START + _INTERVAL, _START, _START],
        funding_rates=[0.0001, None, math.nan],  # type: ignore[list-item]
        mark_prices=[42000.0, -1.0, 42000.0],
    )
    report = _verifier().verify(frame)
    assert report.rows_checked == 3
    assert report.duplicate_timestamp_rows == 1
    assert report.null_rows == 1
    assert report.nan_rows == 1
    assert report.invalid_numeric_rows == 1
    assert report.passed is False
    assert "Duplicate timestamps detected." in report.warnings
    assert "Rows containing NULL values." in report.warnings
    assert "Rows containing NaN values." in report.warnings
    assert "Invalid mark_price values." in report.warnings
    assert "Frame is not sorted by funding_time." in report.warnings
    assert isinstance(report.warnings, tuple)
