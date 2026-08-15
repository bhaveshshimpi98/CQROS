"""Unit tests for CQROS long/short ratio cleaning module."""

from __future__ import annotations

from copy import deepcopy
from math import nan

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.core.constants import MILLISECONDS_PER_MINUTE
from cqros.processing import CleaningReport, LongShortCleaner
from cqros.processing.cleaning import (
    CleaningReport as CleaningReportFromPackage,
)
from cqros.processing.cleaning import (
    LongShortCleaner as LongShortCleanerFromPackage,
)
from cqros.processing.cleaning import long_short as long_short_module
from cqros.processing.exceptions import ProcessingValidationError

_SYMBOL = "BTCUSDT"
_START = 1_699_999_980_000
_INTERVAL = 5 * MILLISECONDS_PER_MINUTE


def _long_short_frame(
    *,
    timestamps: list[int] | None = None,
    long_accounts: list[float | None] | None = None,
    short_accounts: list[float | None] | None = None,
    long_short_ratios: list[float | None] | None = None,
    symbols: list[str] | None = None,
) -> pl.DataFrame:
    """Build a canonical long/short frame for cleaning tests."""
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
            "long_account": (
                list(long_accounts) if long_accounts is not None else [0.55] * row_count
            ),
            "short_account": (
                list(short_accounts) if short_accounts is not None else [0.45] * row_count
            ),
            "long_short_ratio": (
                list(long_short_ratios) if long_short_ratios is not None else [1.2222] * row_count
            ),
        },
        schema={
            "symbol": pl.String,
            "timestamp": pl.Int64,
            "long_account": pl.Float64,
            "short_account": pl.Float64,
            "long_short_ratio": pl.Float64,
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
    """cleaning package and processing package export the same LongShortCleaner."""
    assert CleaningReport is CleaningReportFromPackage
    assert LongShortCleaner is LongShortCleanerFromPackage
    assert long_short_module.__all__ == ["LongShortCleaner"]


# --- Happy path ---


def test_clean_valid_frame_unchanged_values_and_zero_removals() -> None:
    """A valid frame keeps all values and reports zero removals."""
    frame = _long_short_frame()
    cleaned, report = LongShortCleaner().clean(frame)
    assert_frame_equal(cleaned, frame)
    assert report == _zero_report(rows_before=3, rows_after=3)


def test_clean_already_clean_frame() -> None:
    """An already-clean sorted frame is unchanged."""
    frame = _long_short_frame(
        timestamps=[_START, _START + _INTERVAL],
        long_accounts=[0.0, 0.6],
        short_accounts=[0.0, 0.4],
        long_short_ratios=[0.0, 1.5],
    )
    cleaned, report = LongShortCleaner().clean(frame)
    assert_frame_equal(cleaned, frame)
    assert report == _zero_report(rows_before=2, rows_after=2)


def test_clean_empty_frame() -> None:
    """An empty schema-valid frame cleans to empty with zero removals."""
    frame = _long_short_frame(timestamps=[])
    cleaned, report = LongShortCleaner().clean(frame)
    assert cleaned.height == 0
    assert report == _zero_report(rows_before=0, rows_after=0)


def test_clean_does_not_mutate_input() -> None:
    """clean leaves the caller frame unchanged."""
    frame = _long_short_frame(
        timestamps=[_START + _INTERVAL, _START],
        long_accounts=[0.6, 0.55],
        short_accounts=[0.4, 0.45],
        long_short_ratios=[1.5, 1.2222],
    )
    before = deepcopy(frame.to_dicts())
    cleaned, _report = LongShortCleaner().clean(frame)
    assert frame.to_dicts() == before
    assert cleaned is not frame


def test_clean_is_deterministic() -> None:
    """Repeated cleaning of the same input yields identical outputs."""
    frame = _long_short_frame(
        timestamps=[_START + _INTERVAL, _START, _START + _INTERVAL],
        long_accounts=[0.6, 0.55, -0.1],
        short_accounts=[0.4, 0.45, 0.4],
        long_short_ratios=[1.5, 1.2222, 1.0],
    )
    cleaner = LongShortCleaner()
    first_frame, first_report = cleaner.clean(frame)
    second_frame, second_report = cleaner.clean(frame)
    assert_frame_equal(first_frame, second_frame)
    assert first_report == second_report


def test_does_not_recompute_long_short_ratio() -> None:
    """Stored long_short_ratio values are retained without recomputation."""
    frame = _long_short_frame(
        timestamps=[_START],
        long_accounts=[0.8],
        short_accounts=[0.2],
        long_short_ratios=[9.99],
    )
    cleaned, _report = LongShortCleaner().clean(frame)
    assert cleaned.get_column("long_short_ratio").to_list() == [9.99]


# --- Duplicate timestamps ---


def test_removes_duplicate_timestamps_keeps_first() -> None:
    """Duplicate timestamp rows are removed; first occurrence is kept."""
    frame = _long_short_frame(
        timestamps=[_START, _START, _START + _INTERVAL],
        long_accounts=[0.55, 0.99, 0.6],
        short_accounts=[0.45, 0.01, 0.4],
        long_short_ratios=[1.2222, 99.0, 1.5],
    )
    cleaned, report = LongShortCleaner().clean(frame)
    assert cleaned.height == 2
    assert report.duplicates_removed == 1
    assert report.warnings == ("Removed 1 duplicate timestamp row(s).",)
    assert cleaned.get_column("long_account").to_list() == [0.55, 0.6]


# --- NULL / NaN fields ---


@pytest.mark.parametrize(
    ("long_accounts", "short_accounts", "long_short_ratios"),
    [
        ([None, 0.6], [0.45, 0.4], [1.2222, 1.5]),
        ([0.55, 0.6], [None, 0.4], [1.2222, 1.5]),
        ([0.55, 0.6], [0.45, 0.4], [None, 1.5]),
    ],
)
def test_removes_null_ratio_fields(
    long_accounts: list[float | None],
    short_accounts: list[float | None],
    long_short_ratios: list[float | None],
) -> None:
    """Rows with null long/short fields are removed and counted."""
    frame = _long_short_frame(
        timestamps=[_START, _START + _INTERVAL],
        long_accounts=long_accounts,
        short_accounts=short_accounts,
        long_short_ratios=long_short_ratios,
    )
    cleaned, report = LongShortCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.null_rows_removed == 1
    assert report.warnings == ("Removed 1 row(s) with null or NaN long/short ratio fields.",)
    assert cleaned.get_column("timestamp").to_list() == [_START + _INTERVAL]


@pytest.mark.parametrize(
    ("long_accounts", "short_accounts", "long_short_ratios"),
    [
        ([nan, 0.6], [0.45, 0.4], [1.2222, 1.5]),
        ([0.55, 0.6], [nan, 0.4], [1.2222, 1.5]),
        ([0.55, 0.6], [0.45, 0.4], [nan, 1.5]),
    ],
)
def test_removes_nan_ratio_fields(
    long_accounts: list[float | None],
    short_accounts: list[float | None],
    long_short_ratios: list[float | None],
) -> None:
    """Rows with NaN long/short fields are removed and counted."""
    frame = _long_short_frame(
        timestamps=[_START, _START + _INTERVAL],
        long_accounts=long_accounts,
        short_accounts=short_accounts,
        long_short_ratios=long_short_ratios,
    )
    cleaned, report = LongShortCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.null_rows_removed == 1
    assert cleaned.get_column("timestamp").to_list() == [_START + _INTERVAL]


# --- Negative ratios ---


def test_removes_negative_long_account() -> None:
    """Negative long_account rows are removed and counted."""
    frame = _long_short_frame(
        timestamps=[_START, _START + _INTERVAL],
        long_accounts=[-0.1, 0.6],
        short_accounts=[0.45, 0.4],
        long_short_ratios=[1.0, 1.5],
    )
    cleaned, report = LongShortCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.invalid_volume_rows_removed == 1
    assert report.warnings == ("Removed 1 row(s) with invalid long/short ratios.",)
    assert cleaned.get_column("long_account").to_list() == [0.6]


def test_removes_negative_short_account() -> None:
    """Negative short_account rows are removed and counted."""
    frame = _long_short_frame(
        timestamps=[_START, _START + _INTERVAL],
        long_accounts=[0.55, 0.6],
        short_accounts=[-0.2, 0.4],
        long_short_ratios=[1.0, 1.5],
    )
    cleaned, report = LongShortCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.invalid_volume_rows_removed == 1
    assert cleaned.get_column("short_account").to_list() == [0.4]


def test_removes_negative_long_short_ratio() -> None:
    """Negative long_short_ratio rows are removed and counted."""
    frame = _long_short_frame(
        timestamps=[_START, _START + _INTERVAL],
        long_accounts=[0.55, 0.6],
        short_accounts=[0.45, 0.4],
        long_short_ratios=[-1.0, 1.5],
    )
    cleaned, report = LongShortCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.invalid_volume_rows_removed == 1
    assert cleaned.get_column("long_short_ratio").to_list() == [1.5]


def test_retains_zero_ratio_fields() -> None:
    """Zero long_account, short_account, and long_short_ratio are retained."""
    frame = _long_short_frame(
        timestamps=[_START],
        long_accounts=[0.0],
        short_accounts=[0.0],
        long_short_ratios=[0.0],
    )
    cleaned, report = LongShortCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.invalid_volume_rows_removed == 0
    assert cleaned.get_column("long_account").to_list() == [0.0]
    assert cleaned.get_column("short_account").to_list() == [0.0]
    assert cleaned.get_column("long_short_ratio").to_list() == [0.0]


# --- Timestamp validation ---


def test_removes_non_positive_timestamps() -> None:
    """Zero and negative timestamp values are removed."""
    frame = _long_short_frame(
        timestamps=[0, -1, _START],
        long_accounts=[0.5, 0.55, 0.6],
        short_accounts=[0.5, 0.45, 0.4],
        long_short_ratios=[1.0, 1.2222, 1.5],
    )
    cleaned, report = LongShortCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.invalid_timestamp_rows_removed == 2
    assert cleaned.get_column("timestamp").to_list() == [_START]


def test_removes_null_timestamps() -> None:
    """Null timestamp values are removed as invalid timestamps."""
    frame = pl.DataFrame(
        {
            "symbol": [_SYMBOL, _SYMBOL],
            "timestamp": [None, _START],
            "long_account": [0.55, 0.6],
            "short_account": [0.45, 0.4],
            "long_short_ratio": [1.2222, 1.5],
        },
        schema={
            "symbol": pl.String,
            "timestamp": pl.Int64,
            "long_account": pl.Float64,
            "short_account": pl.Float64,
            "long_short_ratio": pl.Float64,
        },
    )
    cleaned, report = LongShortCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.invalid_timestamp_rows_removed == 1
    assert cleaned.get_column("timestamp").to_list() == [_START]


def test_removes_fractional_timestamps() -> None:
    """Non-integer timestamp values are removed."""
    frame = pl.DataFrame(
        {
            "symbol": [_SYMBOL, _SYMBOL],
            "timestamp": [float(_START) + 0.5, float(_START + _INTERVAL)],
            "long_account": [0.55, 0.6],
            "short_account": [0.45, 0.4],
            "long_short_ratio": [1.2222, 1.5],
        },
        schema={
            "symbol": pl.String,
            "timestamp": pl.Float64,
            "long_account": pl.Float64,
            "short_account": pl.Float64,
            "long_short_ratio": pl.Float64,
        },
    )
    cleaned, report = LongShortCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.invalid_timestamp_rows_removed == 1
    assert cleaned.get_column("timestamp").to_list() == [float(_START + _INTERVAL)]


# --- Sort / stability ---


def test_sorts_by_timestamp_with_stable_order() -> None:
    """Output is sorted ascending by timestamp with stable relative order."""
    frame = _long_short_frame(
        timestamps=[_START + _INTERVAL, _START, _START + 2 * _INTERVAL],
        long_accounts=[0.6, 0.55, 0.7],
        short_accounts=[0.4, 0.45, 0.3],
        long_short_ratios=[1.5, 1.2222, 2.3333],
    )
    cleaned, report = LongShortCleaner().clean(frame)
    assert cleaned.get_column("timestamp").to_list() == [
        _START,
        _START + _INTERVAL,
        _START + 2 * _INTERVAL,
    ]
    assert cleaned.get_column("long_account").to_list() == [0.55, 0.6, 0.7]
    assert report.duplicates_removed == 0


def test_does_not_modify_timestamps_of_kept_rows() -> None:
    """Kept rows retain original timestamp values."""
    frame = _long_short_frame(
        timestamps=[_START + _INTERVAL, _START],
        long_accounts=[0.6, 0.55],
        short_accounts=[0.4, 0.45],
        long_short_ratios=[1.5, 1.2222],
    )
    cleaned, _report = LongShortCleaner().clean(frame)
    assert cleaned.get_column("timestamp").to_list() == [
        _START,
        _START + _INTERVAL,
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
            "long_account": [0.55, 0.99, None, 0.6, -0.1],
            "short_account": [0.45, 0.01, 0.4, 0.4, 0.4],
            "long_short_ratio": [1.2222, 99.0, 1.5, 1.5, 1.0],
        },
        schema={
            "symbol": pl.String,
            "timestamp": pl.Int64,
            "long_account": pl.Float64,
            "short_account": pl.Float64,
            "long_short_ratio": pl.Float64,
        },
    )
    cleaned, report = LongShortCleaner().clean(frame)
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
        "Removed 1 row(s) with null or NaN long/short ratio fields.",
        "Removed 1 row(s) with invalid timestamps.",
        "Removed 1 row(s) with invalid long/short ratios.",
    )
    assert cleaned.get_column("timestamp").to_list() == [_START]
    assert cleaned.get_column("long_account").to_list() == [0.55]


def test_row_failing_multiple_rules_counted_once_by_first_rule() -> None:
    """A row matching earlier and later rules is attributed to the earlier one."""
    frame = _long_short_frame(
        timestamps=[_START, _START],
        long_accounts=[0.55, None],
        short_accounts=[0.45, -0.1],
        long_short_ratios=[1.2222, 1.0],
    )
    cleaned, report = LongShortCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.duplicates_removed == 1
    assert report.null_rows_removed == 0
    assert report.invalid_volume_rows_removed == 0
    assert report.rows_before == 2
    assert report.rows_after == 1


# --- Schema validation ---


def test_missing_mandatory_column_raises() -> None:
    """Missing mandatory columns raise ProcessingValidationError."""
    frame = _long_short_frame().drop("long_short_ratio")
    with pytest.raises(
        ProcessingValidationError,
        match="missing required long/short columns",
    ) as exc:
        LongShortCleaner().clean(frame)
    assert exc.value.error_code == "PROCESSING-CLEANING-LONG-SHORT-001"
    assert "long_short_ratio" in exc.value.details["missing_columns"]  # type: ignore[operator]


def test_preserves_extra_columns() -> None:
    """Non-mandatory extra columns are retained on kept rows."""
    frame = _long_short_frame().with_columns(pl.lit("x").alias("extra"))
    cleaned, report = LongShortCleaner().clean(frame)
    assert "extra" in cleaned.columns
    assert cleaned.get_column("extra").to_list() == ["x", "x", "x"]
    assert report.rows_after == 3
