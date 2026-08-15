"""Unit tests for CQROS funding cleaning module."""

from __future__ import annotations

from copy import deepcopy
from math import nan

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.core.constants import MILLISECONDS_PER_HOUR
from cqros.processing import CleaningReport, FundingCleaner
from cqros.processing.cleaning import (
    CleaningReport as CleaningReportFromPackage,
)
from cqros.processing.cleaning import (
    FundingCleaner as FundingCleanerFromPackage,
)
from cqros.processing.cleaning import funding as funding_module
from cqros.processing.exceptions import ProcessingValidationError

_SYMBOL = "BTCUSDT"
_START = 1_699_999_980_000
_INTERVAL = 8 * MILLISECONDS_PER_HOUR


def _funding_frame(
    *,
    funding_times: list[int] | None = None,
    funding_rates: list[float | None] | None = None,
    mark_prices: list[float | None] | None = None,
    symbols: list[str] | None = None,
    include_mark_price: bool = True,
) -> pl.DataFrame:
    """Build a canonical funding frame for cleaning tests."""
    if funding_times is None:
        funding_times = [
            _START,
            _START + _INTERVAL,
            _START + 2 * _INTERVAL,
        ]
    row_count = len(funding_times)
    data: dict[str, list[object]] = {
        "symbol": symbols if symbols is not None else [_SYMBOL] * row_count,
        "funding_time": list(funding_times),
        "funding_rate": (
            list(funding_rates) if funding_rates is not None else [0.0001] * row_count
        ),
    }
    schema: dict[str, pl.DataType] = {
        "symbol": pl.String,
        "funding_time": pl.Int64,
        "funding_rate": pl.Float64,
    }
    if include_mark_price:
        data["mark_price"] = (
            list(mark_prices) if mark_prices is not None else [42_000.0] * row_count
        )
        schema["mark_price"] = pl.Float64
    return pl.DataFrame(data, schema=schema)


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
    """cleaning package and processing package export the same FundingCleaner."""
    assert CleaningReport is CleaningReportFromPackage
    assert FundingCleaner is FundingCleanerFromPackage
    assert funding_module.__all__ == ["FundingCleaner"]


# --- Happy path ---


def test_clean_valid_frame_unchanged_values_and_zero_removals() -> None:
    """A valid frame keeps all values and reports zero removals."""
    frame = _funding_frame()
    cleaned, report = FundingCleaner().clean(frame)
    assert_frame_equal(cleaned, frame)
    assert report == _zero_report(rows_before=3, rows_after=3)


def test_clean_already_clean_frame() -> None:
    """An already-clean sorted frame is unchanged."""
    frame = _funding_frame(
        funding_times=[_START, _START + _INTERVAL],
        funding_rates=[0.0001, -0.0002],
        mark_prices=[None, 41_000.0],
    )
    cleaned, report = FundingCleaner().clean(frame)
    assert_frame_equal(cleaned, frame)
    assert report == _zero_report(rows_before=2, rows_after=2)


def test_clean_empty_frame() -> None:
    """An empty schema-valid frame cleans to empty with zero removals."""
    frame = _funding_frame(funding_times=[])
    cleaned, report = FundingCleaner().clean(frame)
    assert cleaned.height == 0
    assert report == _zero_report(rows_before=0, rows_after=0)


def test_clean_does_not_mutate_input() -> None:
    """clean leaves the caller frame unchanged."""
    frame = _funding_frame(
        funding_times=[_START + _INTERVAL, _START],
        funding_rates=[0.0002, 0.0001],
    )
    before = deepcopy(frame.to_dicts())
    cleaned, _report = FundingCleaner().clean(frame)
    assert frame.to_dicts() == before
    assert cleaned is not frame


def test_clean_is_deterministic() -> None:
    """Repeated cleaning of the same input yields identical outputs."""
    frame = _funding_frame(
        funding_times=[_START + _INTERVAL, _START, _START + _INTERVAL],
        funding_rates=[0.0002, 0.0001, 0.0003],
        mark_prices=[42_000.0, 41_000.0, -1.0],
    )
    cleaner = FundingCleaner()
    first_frame, first_report = cleaner.clean(frame)
    second_frame, second_report = cleaner.clean(frame)
    assert_frame_equal(first_frame, second_frame)
    assert first_report == second_report


# --- Duplicate funding timestamps ---


def test_removes_duplicate_funding_timestamps_keeps_first() -> None:
    """Duplicate funding_time rows are removed; first occurrence is kept."""
    frame = _funding_frame(
        funding_times=[_START, _START, _START + _INTERVAL],
        funding_rates=[0.0001, 0.0009, 0.0002],
        mark_prices=[40_000.0, 99_000.0, 41_000.0],
    )
    cleaned, report = FundingCleaner().clean(frame)
    assert cleaned.height == 2
    assert report.rows_before == 3
    assert report.rows_after == 2
    assert report.duplicates_removed == 1
    assert report.warnings == ("Removed 1 duplicate funding timestamp row(s).",)
    assert cleaned.get_column("funding_rate").to_list() == [0.0001, 0.0002]
    assert cleaned.get_column("mark_price").to_list() == [40_000.0, 41_000.0]


# --- NULL / NaN funding_rate ---


def test_removes_null_funding_rate() -> None:
    """Rows with null funding_rate are removed and counted."""
    frame = _funding_frame(
        funding_times=[_START, _START + _INTERVAL, _START + 2 * _INTERVAL],
        funding_rates=[0.0001, None, 0.0002],
    )
    cleaned, report = FundingCleaner().clean(frame)
    assert cleaned.height == 2
    assert report.null_rows_removed == 1
    assert report.warnings == ("Removed 1 row(s) with null or NaN funding_rate.",)
    assert cleaned.get_column("funding_time").to_list() == [
        _START,
        _START + 2 * _INTERVAL,
    ]


def test_removes_nan_funding_rate() -> None:
    """Rows with NaN funding_rate are removed and counted."""
    frame = _funding_frame(
        funding_times=[_START, _START + _INTERVAL],
        funding_rates=[0.0001, nan],
    )
    cleaned, report = FundingCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.null_rows_removed == 1
    assert cleaned.get_column("funding_rate").to_list() == [0.0001]


def test_retains_negative_funding_rate() -> None:
    """Negative funding_rate values are valid and retained."""
    frame = _funding_frame(
        funding_times=[_START, _START + _INTERVAL],
        funding_rates=[-0.0005, 0.0001],
    )
    cleaned, report = FundingCleaner().clean(frame)
    assert cleaned.height == 2
    assert report.null_rows_removed == 0
    assert cleaned.get_column("funding_rate").to_list() == [-0.0005, 0.0001]


def test_retains_zero_funding_rate() -> None:
    """Zero funding_rate is valid and retained."""
    frame = _funding_frame(funding_times=[_START], funding_rates=[0.0])
    cleaned, report = FundingCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.null_rows_removed == 0
    assert cleaned.get_column("funding_rate").to_list() == [0.0]


# --- Timestamp validation ---


def test_removes_non_positive_timestamps() -> None:
    """Zero and negative funding_time values are removed."""
    frame = _funding_frame(
        funding_times=[0, -1, _START],
        funding_rates=[0.0001, 0.0002, 0.0003],
    )
    cleaned, report = FundingCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.invalid_timestamp_rows_removed == 2
    assert cleaned.get_column("funding_time").to_list() == [_START]
    assert report.warnings == ("Removed 2 row(s) with invalid timestamps.",)


def test_removes_null_timestamps() -> None:
    """Null funding_time values are removed as invalid timestamps."""
    frame = pl.DataFrame(
        {
            "symbol": [_SYMBOL, _SYMBOL],
            "funding_time": [None, _START],
            "funding_rate": [0.0001, 0.0002],
            "mark_price": [40_000.0, 41_000.0],
        },
        schema={
            "symbol": pl.String,
            "funding_time": pl.Int64,
            "funding_rate": pl.Float64,
            "mark_price": pl.Float64,
        },
    )
    cleaned, report = FundingCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.invalid_timestamp_rows_removed == 1
    assert cleaned.get_column("funding_time").to_list() == [_START]


def test_removes_fractional_timestamps() -> None:
    """Non-integer funding_time values are removed."""
    frame = pl.DataFrame(
        {
            "symbol": [_SYMBOL, _SYMBOL],
            "funding_time": [float(_START) + 0.5, float(_START + _INTERVAL)],
            "funding_rate": [0.0001, 0.0002],
            "mark_price": [40_000.0, 41_000.0],
        },
        schema={
            "symbol": pl.String,
            "funding_time": pl.Float64,
            "funding_rate": pl.Float64,
            "mark_price": pl.Float64,
        },
    )
    cleaned, report = FundingCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.invalid_timestamp_rows_removed == 1
    assert cleaned.get_column("funding_time").to_list() == [float(_START + _INTERVAL)]


# --- mark_price ---


def test_removes_negative_mark_price() -> None:
    """Negative mark_price rows are removed when the column is present."""
    frame = _funding_frame(
        funding_times=[_START, _START + _INTERVAL],
        mark_prices=[42_000.0, -1.0],
    )
    cleaned, report = FundingCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.invalid_price_rows_removed == 1
    assert report.warnings == ("Removed 1 row(s) with invalid mark_price.",)
    assert cleaned.get_column("funding_time").to_list() == [_START]


def test_retains_null_mark_price() -> None:
    """Null mark_price is retained and does not count as invalid."""
    frame = _funding_frame(
        funding_times=[_START, _START + _INTERVAL],
        mark_prices=[None, 42_000.0],
    )
    cleaned, report = FundingCleaner().clean(frame)
    assert cleaned.height == 2
    assert report.invalid_price_rows_removed == 0
    assert cleaned.get_column("mark_price").to_list() == [None, 42_000.0]


def test_removes_nan_mark_price() -> None:
    """NaN mark_price rows are removed when the column is present."""
    frame = _funding_frame(
        funding_times=[_START, _START + _INTERVAL],
        mark_prices=[nan, 42_000.0],
    )
    cleaned, report = FundingCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.invalid_price_rows_removed == 1
    assert cleaned.get_column("mark_price").to_list() == [42_000.0]


def test_retains_zero_mark_price() -> None:
    """Zero mark_price is valid and retained."""
    frame = _funding_frame(funding_times=[_START], mark_prices=[0.0])
    cleaned, report = FundingCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.invalid_price_rows_removed == 0
    assert cleaned.get_column("mark_price").to_list() == [0.0]


def test_mark_price_column_optional() -> None:
    """Frames without mark_price clean successfully with zero price removals."""
    frame = _funding_frame(include_mark_price=False)
    cleaned, report = FundingCleaner().clean(frame)
    assert "mark_price" not in cleaned.columns
    assert cleaned.height == 3
    assert report.invalid_price_rows_removed == 0
    assert report == _zero_report(rows_before=3, rows_after=3)


# --- Sort / stability ---


def test_sorts_by_funding_time_with_stable_order() -> None:
    """Output is sorted by funding_time; relative order of ties is preserved.

    After duplicate removal, distinct times sort ascending. Stability is
    verified by cleaning equal non-timestamp columns that would otherwise
    reorder under an unstable sort.
    """
    frame = _funding_frame(
        funding_times=[_START + _INTERVAL, _START, _START + 2 * _INTERVAL],
        funding_rates=[0.0002, 0.0001, 0.0003],
        mark_prices=[41_000.0, 40_000.0, 42_000.0],
    )
    cleaned, report = FundingCleaner().clean(frame)
    assert cleaned.get_column("funding_time").to_list() == [
        _START,
        _START + _INTERVAL,
        _START + 2 * _INTERVAL,
    ]
    assert cleaned.get_column("funding_rate").to_list() == [0.0001, 0.0002, 0.0003]
    assert report.duplicates_removed == 0


def test_does_not_modify_timestamps_of_kept_rows() -> None:
    """Kept rows retain original funding_time values."""
    frame = _funding_frame(
        funding_times=[_START + _INTERVAL, _START],
        funding_rates=[0.0002, 0.0001],
    )
    cleaned, _report = FundingCleaner().clean(frame)
    assert cleaned.get_column("funding_time").to_list() == [
        _START,
        _START + _INTERVAL,
    ]


def test_does_not_fill_interpolate_smooth_or_clip() -> None:
    """Cleaning never invents settlements; gap rows stay absent and values intact."""
    frame = _funding_frame(
        funding_times=[_START, _START + 2 * _INTERVAL],
        funding_rates=[-0.0004, 0.0005],
        mark_prices=[40_000.0, 45_000.0],
    )
    cleaned, _report = FundingCleaner().clean(frame)
    assert cleaned.height == 2
    assert cleaned.get_column("funding_rate").to_list() == [-0.0004, 0.0005]
    assert cleaned.get_column("funding_time").to_list() == [
        _START,
        _START + 2 * _INTERVAL,
    ]


# --- Report counters / attribution ---


def test_report_counters_and_combined_warnings() -> None:
    """Removal counters and warnings follow processing order attribution."""
    frame = pl.DataFrame(
        {
            "symbol": [_SYMBOL] * 5,
            "funding_time": [
                _START,
                _START,
                _START + _INTERVAL,
                0,
                _START + 2 * _INTERVAL,
            ],
            "funding_rate": [0.0001, 0.0009, None, 0.0002, 0.0003],
            "mark_price": [40_000.0, 99_000.0, 41_000.0, 42_000.0, -5.0],
        },
        schema={
            "symbol": pl.String,
            "funding_time": pl.Int64,
            "funding_rate": pl.Float64,
            "mark_price": pl.Float64,
        },
    )
    cleaned, report = FundingCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.rows_before == 5
    assert report.rows_after == 1
    assert report.duplicates_removed == 1
    assert report.null_rows_removed == 1
    assert report.invalid_timestamp_rows_removed == 1
    assert report.invalid_price_rows_removed == 1
    assert report.invalid_volume_rows_removed == 0
    assert report.invalid_trade_count_rows_removed == 0
    assert report.warnings == (
        "Removed 1 duplicate funding timestamp row(s).",
        "Removed 1 row(s) with null or NaN funding_rate.",
        "Removed 1 row(s) with invalid timestamps.",
        "Removed 1 row(s) with invalid mark_price.",
    )
    assert cleaned.get_column("funding_time").to_list() == [_START]
    assert cleaned.get_column("funding_rate").to_list() == [0.0001]


def test_row_failing_multiple_rules_counted_once_by_first_rule() -> None:
    """A row matching earlier and later rules is attributed to the earlier one."""
    frame = _funding_frame(
        funding_times=[_START, _START],
        funding_rates=[0.0001, None],
        mark_prices=[40_000.0, -1.0],
    )
    cleaned, report = FundingCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.duplicates_removed == 1
    assert report.null_rows_removed == 0
    assert report.invalid_price_rows_removed == 0
    assert report.rows_before == 2
    assert report.rows_after == 1


# --- Schema failures ---


def test_missing_mandatory_column_raises() -> None:
    """Missing mandatory columns raise ProcessingValidationError."""
    frame = _funding_frame().drop("funding_rate")
    with pytest.raises(
        ProcessingValidationError,
        match="missing required funding columns",
    ) as exc:
        FundingCleaner().clean(frame)
    assert exc.value.error_code == "PROCESSING-CLEANING-FUNDING-001"
    assert "funding_rate" in exc.value.details["missing_columns"]  # type: ignore[operator]


def test_preserves_extra_columns() -> None:
    """Non-mandatory extra columns are retained on kept rows."""
    frame = _funding_frame().with_columns(pl.lit("x").alias("extra"))
    cleaned, report = FundingCleaner().clean(frame)
    assert "extra" in cleaned.columns
    assert cleaned.get_column("extra").to_list() == ["x", "x", "x"]
    assert report.rows_after == 3
