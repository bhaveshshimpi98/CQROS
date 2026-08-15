"""Unit tests for CQROS open-interest cleaning module."""

from __future__ import annotations

from copy import deepcopy
from math import nan

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.core.constants import MILLISECONDS_PER_MINUTE
from cqros.processing import CleaningReport, OpenInterestCleaner
from cqros.processing.cleaning import (
    CleaningReport as CleaningReportFromPackage,
)
from cqros.processing.cleaning import (
    OpenInterestCleaner as OpenInterestCleanerFromPackage,
)
from cqros.processing.cleaning import open_interest as open_interest_module
from cqros.processing.exceptions import ProcessingValidationError

_SYMBOL = "BTCUSDT"
_START = 1_699_999_980_000
_INTERVAL = 5 * MILLISECONDS_PER_MINUTE


def _open_interest_frame(
    *,
    timestamps: list[int] | None = None,
    open_interests: list[float | None] | None = None,
    symbols: list[str] | None = None,
) -> pl.DataFrame:
    """Build a canonical open-interest frame for cleaning tests."""
    if timestamps is None:
        timestamps = [
            _START,
            _START + _INTERVAL,
            _START + 2 * _INTERVAL,
        ]
    row_count = len(timestamps)
    return pl.DataFrame(
        {
            "symbol": symbols if symbols is not None else [_SYMBOL] * row_count,
            "timestamp": list(timestamps),
            "open_interest": (
                list(open_interests) if open_interests is not None else [100.0] * row_count
            ),
        },
        schema={
            "symbol": pl.String,
            "timestamp": pl.Int64,
            "open_interest": pl.Float64,
        },
    )


def _zero_report(*, rows_before: int, rows_after: int) -> CleaningReport:
    """Build a CleaningReport with all removal counters at zero."""
    return CleaningReport(
        rows_before=rows_before,
        rows_after=rows_after,
        duplicates_removed=0,
        null_rows_removed=0,
        invalid_price_rows_removed=0,
        invalid_volume_rows_removed=0,
        invalid_trade_count_rows_removed=0,
        invalid_timestamp_rows_removed=0,
        warnings=(),
    )


# --- Package API ---


def test_package_exports_match_processing_reexports() -> None:
    """cleaning package and processing package export the same OpenInterestCleaner."""
    assert CleaningReport is CleaningReportFromPackage
    assert OpenInterestCleaner is OpenInterestCleanerFromPackage
    assert open_interest_module.__all__ == ["OpenInterestCleaner"]


# --- Happy path ---


def test_clean_valid_frame_unchanged_values_and_zero_removals() -> None:
    """A valid frame keeps all values and reports zero removals."""
    frame = _open_interest_frame()
    cleaned, report = OpenInterestCleaner().clean(frame)
    assert_frame_equal(cleaned, frame)
    assert report == _zero_report(rows_before=3, rows_after=3)


def test_clean_already_clean_frame() -> None:
    """An already-clean sorted frame is unchanged."""
    frame = _open_interest_frame(
        timestamps=[_START, _START + _INTERVAL],
        open_interests=[0.0, 150.5],
    )
    cleaned, report = OpenInterestCleaner().clean(frame)
    assert_frame_equal(cleaned, frame)
    assert report == _zero_report(rows_before=2, rows_after=2)


def test_clean_empty_frame() -> None:
    """An empty schema-valid frame cleans to empty with zero removals."""
    frame = _open_interest_frame(timestamps=[])
    cleaned, report = OpenInterestCleaner().clean(frame)
    assert cleaned.height == 0
    assert report == _zero_report(rows_before=0, rows_after=0)


def test_clean_does_not_mutate_input() -> None:
    """clean leaves the caller frame unchanged."""
    frame = _open_interest_frame(
        timestamps=[_START + _INTERVAL, _START],
        open_interests=[110.0, 100.0],
    )
    before = deepcopy(frame.to_dicts())
    cleaned, _report = OpenInterestCleaner().clean(frame)
    assert frame.to_dicts() == before
    assert cleaned is not frame


def test_clean_is_deterministic() -> None:
    """Repeated cleaning of the same input yields identical outputs."""
    frame = _open_interest_frame(
        timestamps=[_START + _INTERVAL, _START, _START + _INTERVAL],
        open_interests=[110.0, 100.0, -5.0],
    )
    cleaner = OpenInterestCleaner()
    first_frame, first_report = cleaner.clean(frame)
    second_frame, second_report = cleaner.clean(frame)
    assert_frame_equal(first_frame, second_frame)
    assert first_report == second_report


# --- Duplicate timestamps ---


def test_removes_duplicate_timestamps_keeps_first() -> None:
    """Duplicate timestamp rows are removed; first occurrence is kept."""
    frame = _open_interest_frame(
        timestamps=[_START, _START, _START + _INTERVAL],
        open_interests=[100.0, 999.0, 110.0],
    )
    cleaned, report = OpenInterestCleaner().clean(frame)
    assert cleaned.height == 2
    assert report.rows_before == 3
    assert report.rows_after == 2
    assert report.duplicates_removed == 1
    assert report.warnings == ("Removed 1 duplicate timestamp row(s).",)
    assert cleaned.get_column("open_interest").to_list() == [100.0, 110.0]


# --- NULL / NaN open_interest ---


def test_removes_null_open_interest() -> None:
    """Rows with null open_interest are removed and counted."""
    frame = _open_interest_frame(
        timestamps=[_START, _START + _INTERVAL, _START + 2 * _INTERVAL],
        open_interests=[100.0, None, 120.0],
    )
    cleaned, report = OpenInterestCleaner().clean(frame)
    assert cleaned.height == 2
    assert report.null_rows_removed == 1
    assert report.warnings == ("Removed 1 row(s) with null or NaN open_interest.",)
    assert cleaned.get_column("timestamp").to_list() == [
        _START,
        _START + 2 * _INTERVAL,
    ]


def test_removes_nan_open_interest() -> None:
    """Rows with NaN open_interest are removed and counted."""
    frame = _open_interest_frame(
        timestamps=[_START, _START + _INTERVAL],
        open_interests=[100.0, nan],
    )
    cleaned, report = OpenInterestCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.null_rows_removed == 1
    assert cleaned.get_column("open_interest").to_list() == [100.0]


def test_retains_zero_open_interest() -> None:
    """Zero open_interest is valid and retained."""
    frame = _open_interest_frame(timestamps=[_START], open_interests=[0.0])
    cleaned, report = OpenInterestCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.null_rows_removed == 0
    assert report.invalid_volume_rows_removed == 0
    assert cleaned.get_column("open_interest").to_list() == [0.0]


# --- Negative open_interest ---


def test_removes_negative_open_interest() -> None:
    """Negative open_interest rows are removed and counted via volume counter."""
    frame = _open_interest_frame(
        timestamps=[_START, _START + _INTERVAL, _START + 2 * _INTERVAL],
        open_interests=[100.0, -1.0, 120.0],
    )
    cleaned, report = OpenInterestCleaner().clean(frame)
    assert cleaned.height == 2
    assert report.invalid_volume_rows_removed == 1
    assert report.warnings == ("Removed 1 row(s) with negative open_interest.",)
    assert cleaned.get_column("open_interest").to_list() == [100.0, 120.0]


# --- Timestamp validation ---


def test_removes_non_positive_timestamps() -> None:
    """Zero and negative timestamp values are removed."""
    frame = _open_interest_frame(
        timestamps=[0, -1, _START],
        open_interests=[100.0, 110.0, 120.0],
    )
    cleaned, report = OpenInterestCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.invalid_timestamp_rows_removed == 2
    assert cleaned.get_column("timestamp").to_list() == [_START]
    assert report.warnings == ("Removed 2 row(s) with invalid timestamps.",)


def test_removes_null_timestamps() -> None:
    """Null timestamp values are removed as invalid timestamps."""
    frame = pl.DataFrame(
        {
            "symbol": [_SYMBOL, _SYMBOL],
            "timestamp": [None, _START],
            "open_interest": [100.0, 110.0],
        },
        schema={
            "symbol": pl.String,
            "timestamp": pl.Int64,
            "open_interest": pl.Float64,
        },
    )
    cleaned, report = OpenInterestCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.invalid_timestamp_rows_removed == 1
    assert cleaned.get_column("timestamp").to_list() == [_START]


def test_removes_fractional_timestamps() -> None:
    """Non-integer timestamp values are removed."""
    frame = pl.DataFrame(
        {
            "symbol": [_SYMBOL, _SYMBOL],
            "timestamp": [float(_START) + 0.5, float(_START + _INTERVAL)],
            "open_interest": [100.0, 110.0],
        },
        schema={
            "symbol": pl.String,
            "timestamp": pl.Float64,
            "open_interest": pl.Float64,
        },
    )
    cleaned, report = OpenInterestCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.invalid_timestamp_rows_removed == 1
    assert cleaned.get_column("timestamp").to_list() == [float(_START + _INTERVAL)]


# --- Sort / stability ---


def test_sorts_by_timestamp_with_stable_order() -> None:
    """Output is sorted ascending by timestamp with stable relative order."""
    frame = _open_interest_frame(
        timestamps=[_START + _INTERVAL, _START, _START + 2 * _INTERVAL],
        open_interests=[110.0, 100.0, 120.0],
    )
    cleaned, report = OpenInterestCleaner().clean(frame)
    assert cleaned.get_column("timestamp").to_list() == [
        _START,
        _START + _INTERVAL,
        _START + 2 * _INTERVAL,
    ]
    assert cleaned.get_column("open_interest").to_list() == [100.0, 110.0, 120.0]
    assert report.duplicates_removed == 0


def test_does_not_modify_timestamps_of_kept_rows() -> None:
    """Kept rows retain original timestamp values."""
    frame = _open_interest_frame(
        timestamps=[_START + _INTERVAL, _START],
        open_interests=[110.0, 100.0],
    )
    cleaned, _report = OpenInterestCleaner().clean(frame)
    assert cleaned.get_column("timestamp").to_list() == [
        _START,
        _START + _INTERVAL,
    ]


def test_does_not_fill_interpolate_smooth_or_clip() -> None:
    """Cleaning never invents observations; gap rows stay absent and values intact."""
    frame = _open_interest_frame(
        timestamps=[_START, _START + 2 * _INTERVAL],
        open_interests=[100.0, 200.0],
    )
    cleaned, _report = OpenInterestCleaner().clean(frame)
    assert cleaned.height == 2
    assert cleaned.get_column("open_interest").to_list() == [100.0, 200.0]
    assert cleaned.get_column("timestamp").to_list() == [
        _START,
        _START + 2 * _INTERVAL,
    ]


# --- Report counters / attribution ---


def test_report_counters_and_combined_warnings() -> None:
    """Removal counters and warnings follow processing order attribution."""
    frame = pl.DataFrame(
        {
            "symbol": [_SYMBOL] * 5,
            "timestamp": [
                _START,
                _START,
                _START + _INTERVAL,
                0,
                _START + 2 * _INTERVAL,
            ],
            "open_interest": [100.0, 999.0, None, 110.0, -5.0],
        },
        schema={
            "symbol": pl.String,
            "timestamp": pl.Int64,
            "open_interest": pl.Float64,
        },
    )
    cleaned, report = OpenInterestCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.rows_before == 5
    assert report.rows_after == 1
    assert report.duplicates_removed == 1
    assert report.null_rows_removed == 1
    assert report.invalid_timestamp_rows_removed == 1
    assert report.invalid_volume_rows_removed == 1
    assert report.invalid_price_rows_removed == 0
    assert report.invalid_trade_count_rows_removed == 0
    assert report.warnings == (
        "Removed 1 duplicate timestamp row(s).",
        "Removed 1 row(s) with null or NaN open_interest.",
        "Removed 1 row(s) with invalid timestamps.",
        "Removed 1 row(s) with negative open_interest.",
    )
    assert cleaned.get_column("timestamp").to_list() == [_START]
    assert cleaned.get_column("open_interest").to_list() == [100.0]


def test_row_failing_multiple_rules_counted_once_by_first_rule() -> None:
    """A row matching earlier and later rules is attributed to the earlier one."""
    frame = _open_interest_frame(
        timestamps=[_START, _START],
        open_interests=[100.0, None],
    )
    cleaned, report = OpenInterestCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.duplicates_removed == 1
    assert report.null_rows_removed == 0
    assert report.invalid_volume_rows_removed == 0
    assert report.rows_before == 2
    assert report.rows_after == 1


# --- Schema validation ---


def test_missing_mandatory_column_raises() -> None:
    """Missing mandatory columns raise ProcessingValidationError."""
    frame = _open_interest_frame().drop("open_interest")
    with pytest.raises(
        ProcessingValidationError,
        match="missing required open interest columns",
    ) as exc:
        OpenInterestCleaner().clean(frame)
    assert exc.value.error_code == "PROCESSING-CLEANING-OPEN-INTEREST-001"
    assert "open_interest" in exc.value.details["missing_columns"]  # type: ignore[operator]


def test_preserves_extra_columns() -> None:
    """Non-mandatory extra columns are retained on kept rows."""
    frame = _open_interest_frame().with_columns(pl.lit("x").alias("extra"))
    cleaned, report = OpenInterestCleaner().clean(frame)
    assert "extra" in cleaned.columns
    assert cleaned.get_column("extra").to_list() == ["x", "x", "x"]
    assert report.rows_after == 3
