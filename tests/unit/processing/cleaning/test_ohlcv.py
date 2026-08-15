"""Unit tests for CQROS OHLCV cleaning module."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, is_dataclass

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.core.constants import MILLISECONDS_PER_MINUTE
from cqros.processing import CleaningReport, OHLCVCleaner
from cqros.processing.cleaning import (
    CleaningReport as CleaningReportFromPackage,
)
from cqros.processing.cleaning import (
    OHLCVCleaner as OHLCVCleanerFromPackage,
)
from cqros.processing.cleaning import ohlcv as cleaning_module
from cqros.processing.exceptions import ProcessingValidationError

_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1m"
_START = 1_699_999_980_000
_INTERVAL = MILLISECONDS_PER_MINUTE


def _ohlcv_frame(
    *,
    open_times: list[int] | None = None,
    opens: list[float] | None = None,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    closes: list[float] | None = None,
    volumes: list[float] | None = None,
    quote_volumes: list[float] | None = None,
    trade_counts: list[int] | None = None,
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    close_times: list[int] | None = None,
) -> pl.DataFrame:
    """Build a canonical OHLCV frame for cleaning tests."""
    if open_times is None:
        open_times = [
            _START,
            _START + _INTERVAL,
            _START + 2 * _INTERVAL,
        ]
    row_count = len(open_times)
    return pl.DataFrame(
        {
            "symbol": symbols if symbols is not None else [_SYMBOL] * row_count,
            "timeframe": timeframes if timeframes is not None else [_TIMEFRAME] * row_count,
            "open_time": open_times,
            "close_time": (
                close_times
                if close_times is not None
                else [value + _INTERVAL - 1 for value in open_times]
            ),
            "open": opens if opens is not None else [100.0] * row_count,
            "high": highs if highs is not None else [101.0] * row_count,
            "low": lows if lows is not None else [99.0] * row_count,
            "close": closes if closes is not None else [100.5] * row_count,
            "volume": volumes if volumes is not None else [10.0] * row_count,
            "quote_volume": (quote_volumes if quote_volumes is not None else [1000.0] * row_count),
            "trade_count": trade_counts if trade_counts is not None else [42] * row_count,
        },
        schema={
            "symbol": pl.String,
            "timeframe": pl.String,
            "open_time": pl.Int64,
            "close_time": pl.Int64,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
            "quote_volume": pl.Float64,
            "trade_count": pl.Int64,
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


# --- CleaningReport ---


def test_cleaning_report_is_immutable_slotted_dataclass() -> None:
    """CleaningReport is a frozen slotted dataclass."""
    report = _zero_report(rows_before=1, rows_after=1)
    assert is_dataclass(report)
    assert CleaningReport.__slots__ == (
        "rows_before",
        "rows_after",
        "duplicates_removed",
        "null_rows_removed",
        "invalid_price_rows_removed",
        "invalid_volume_rows_removed",
        "invalid_trade_count_rows_removed",
        "invalid_timestamp_rows_removed",
        "warnings",
    )
    with pytest.raises(FrozenInstanceError):
        report.rows_after = 0  # type: ignore[misc]


def test_package_exports_match_processing_reexports() -> None:
    """cleaning package and processing package export the same types."""
    assert CleaningReport is CleaningReportFromPackage
    assert OHLCVCleaner is OHLCVCleanerFromPackage
    assert cleaning_module.__all__ == ["CleaningReport", "OHLCVCleaner"]


# --- Happy path ---


def test_clean_valid_frame_unchanged_values_and_zero_removals() -> None:
    """A valid frame keeps all values and reports zero removals."""
    frame = _ohlcv_frame()
    cleaned, report = OHLCVCleaner().clean(frame)
    assert_frame_equal(cleaned, frame)
    assert report == _zero_report(rows_before=3, rows_after=3)


def test_clean_empty_frame() -> None:
    """An empty schema-valid frame cleans to empty with zero removals."""
    frame = _ohlcv_frame(open_times=[])
    cleaned, report = OHLCVCleaner().clean(frame)
    assert cleaned.height == 0
    assert report == _zero_report(rows_before=0, rows_after=0)


def test_clean_does_not_mutate_input() -> None:
    """clean leaves the caller frame unchanged."""
    frame = _ohlcv_frame(open_times=[_START + _INTERVAL, _START], opens=[2.0, 1.0])
    before = deepcopy(frame.to_dicts())
    cleaned, _report = OHLCVCleaner().clean(frame)
    assert frame.to_dicts() == before
    assert cleaned is not frame


def test_clean_is_deterministic() -> None:
    """Repeated cleaning of the same input yields identical outputs."""
    frame = _ohlcv_frame(
        open_times=[_START + _INTERVAL, _START, _START + _INTERVAL],
        opens=[2.0, 1.0, 2.0],
        volumes=[10.0, 10.0, -1.0],
    )
    cleaner = OHLCVCleaner()
    first_frame, first_report = cleaner.clean(frame)
    second_frame, second_report = cleaner.clean(frame)
    assert_frame_equal(first_frame, second_frame)
    assert first_report == second_report


# --- Exact duplicates ---


def test_removes_exact_duplicate_rows() -> None:
    """Exact duplicate rows are removed; first occurrence is kept."""
    frame = pl.concat([_ohlcv_frame(), _ohlcv_frame().head(1)])
    cleaned, report = OHLCVCleaner().clean(frame)
    assert cleaned.height == 3
    assert report.rows_before == 4
    assert report.rows_after == 3
    assert report.duplicates_removed == 1
    assert report.warnings == ("Removed 1 exact duplicate row(s).",)
    assert_frame_equal(cleaned, _ohlcv_frame())


def test_does_not_remove_non_exact_duplicates() -> None:
    """Rows sharing open_time but differing elsewhere are retained."""
    frame = _ohlcv_frame(
        open_times=[_START, _START],
        opens=[100.0, 101.0],
        highs=[101.0, 102.0],
        lows=[99.0, 100.0],
        closes=[100.5, 101.5],
    )
    cleaned, report = OHLCVCleaner().clean(frame)
    assert cleaned.height == 2
    assert report.duplicates_removed == 0


# --- Null mandatory columns ---


def test_removes_rows_with_null_mandatory_values() -> None:
    """Rows containing nulls in mandatory columns are removed."""
    frame = _ohlcv_frame(opens=[100.0, None, 100.0])  # type: ignore[list-item]
    cleaned, report = OHLCVCleaner().clean(frame)
    assert cleaned.height == 2
    assert report.null_rows_removed == 1
    assert report.warnings == ("Removed 1 row(s) with null or NaN mandatory values.",)
    assert cleaned.get_column("open_time").to_list() == [
        _START,
        _START + 2 * _INTERVAL,
    ]


@pytest.mark.parametrize(
    "column",
    [
        "symbol",
        "timeframe",
        "open_time",
        "close_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trade_count",
    ],
)
def test_null_removal_covers_every_mandatory_column(column: str) -> None:
    """Nulls in each mandatory column cause row removal."""
    frame = _ohlcv_frame()
    values = frame.get_column(column).to_list()
    values[0] = None
    frame = frame.with_columns(pl.Series(column, values))

    cleaned, report = OHLCVCleaner().clean(frame)
    assert cleaned.height == 2
    assert report.null_rows_removed == 1


@pytest.mark.parametrize(
    "column",
    [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trade_count",
    ],
)
def test_nan_removal_covers_every_mandatory_numeric_column(column: str) -> None:
    """NaN in each mandatory numeric column causes row removal."""
    frame = _ohlcv_frame()
    values = [float(value) for value in frame.get_column(column).to_list()]
    values[0] = float("nan")
    frame = frame.with_columns(pl.Series(column, values, dtype=pl.Float64))

    cleaned, report = OHLCVCleaner().clean(frame)
    assert cleaned.height == 2
    assert report.null_rows_removed == 1
    assert report.invalid_price_rows_removed == 0
    assert report.invalid_volume_rows_removed == 0
    assert report.invalid_trade_count_rows_removed == 0
    assert report.warnings == ("Removed 1 row(s) with null or NaN mandatory values.",)


def test_removes_mixture_of_null_and_nan_rows() -> None:
    """NULL and NaN mandatory numeric defects are both counted as null removals."""
    frame = _ohlcv_frame(
        opens=[100.0, None, 100.0, 100.0],  # type: ignore[list-item]
        open_times=[
            _START,
            _START + _INTERVAL,
            _START + 2 * _INTERVAL,
            _START + 3 * _INTERVAL,
        ],
    )
    highs = frame.get_column("high").to_list()
    highs[2] = float("nan")
    frame = frame.with_columns(pl.Series("high", highs, dtype=pl.Float64))
    before = frame.clone()

    cleaned, report = OHLCVCleaner().clean(frame)

    assert_frame_equal(frame, before)
    assert cleaned.height == 2
    assert report.rows_before == 4
    assert report.rows_after == 2
    assert report.null_rows_removed == 2
    assert report.duplicates_removed == 0
    assert report.invalid_price_rows_removed == 0
    assert report.invalid_volume_rows_removed == 0
    assert report.invalid_trade_count_rows_removed == 0
    assert report.invalid_timestamp_rows_removed == 0
    assert report.warnings == ("Removed 2 row(s) with null or NaN mandatory values.",)
    assert cleaned.get_column("open_time").to_list() == [
        _START,
        _START + 3 * _INTERVAL,
    ]


def test_nan_removal_does_not_mutate_input() -> None:
    """NaN cleaning leaves the caller frame unchanged."""
    frame = _ohlcv_frame()
    values = frame.get_column("close").to_list()
    values[1] = float("nan")
    frame = frame.with_columns(pl.Series("close", values, dtype=pl.Float64))
    before = frame.clone()

    cleaned, report = OHLCVCleaner().clean(frame)

    assert_frame_equal(frame, before)
    assert cleaned is not frame
    assert cleaned.height == 2
    assert report.null_rows_removed == 1


# --- Invalid prices ---


def test_removes_non_positive_prices() -> None:
    """Rows with open/high/low/close <= 0 are removed."""
    frame = _ohlcv_frame(
        opens=[100.0, 0.0, -1.0],
        highs=[101.0, 101.0, 101.0],
        lows=[99.0, 99.0, 99.0],
        closes=[100.5, 100.5, 100.5],
    )
    cleaned, report = OHLCVCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.invalid_price_rows_removed == 2
    assert cleaned.get_column("open").to_list() == [100.0]


def test_removes_high_below_max_of_open_close_low() -> None:
    """Rows where high < max(open, close, low) are removed."""
    frame = _ohlcv_frame(
        open_times=[_START, _START + _INTERVAL],
        opens=[100.0, 100.0],
        highs=[101.0, 99.5],
        lows=[99.0, 99.0],
        closes=[100.5, 100.5],
    )
    cleaned, report = OHLCVCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.invalid_price_rows_removed == 1
    assert cleaned.get_column("high").to_list() == [101.0]


def test_removes_low_above_min_of_open_close_high() -> None:
    """Rows where low > min(open, close, high) are removed."""
    frame = _ohlcv_frame(
        open_times=[_START, _START + _INTERVAL],
        opens=[100.0, 100.0],
        highs=[101.0, 101.0],
        lows=[99.0, 100.5],
        closes=[100.5, 100.0],
    )
    cleaned, report = OHLCVCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.invalid_price_rows_removed == 1
    assert cleaned.get_column("low").to_list() == [99.0]


def test_keeps_boundary_valid_ohlc() -> None:
    """high == max(open, close, low) and low == min(...) are kept."""
    frame = _ohlcv_frame(
        open_times=[_START],
        opens=[100.0],
        highs=[100.5],
        lows=[100.0],
        closes=[100.5],
    )
    cleaned, report = OHLCVCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.invalid_price_rows_removed == 0


# --- Invalid volume / trade count ---


def test_removes_negative_volume_and_quote_volume() -> None:
    """Negative volume or quote_volume rows are removed and counted."""
    frame = _ohlcv_frame(
        volumes=[10.0, -0.1, 10.0],
        quote_volumes=[1000.0, 1000.0, -1.0],
    )
    cleaned, report = OHLCVCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.invalid_volume_rows_removed == 2
    assert report.warnings == ("Removed 2 row(s) with invalid volume fields.",)


def test_keeps_zero_volume_fields() -> None:
    """Zero volume, quote_volume, and trade_count are valid."""
    frame = _ohlcv_frame(
        open_times=[_START],
        volumes=[0.0],
        quote_volumes=[0.0],
        trade_counts=[0],
    )
    cleaned, report = OHLCVCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.invalid_volume_rows_removed == 0
    assert report.invalid_trade_count_rows_removed == 0


def test_removes_negative_trade_count() -> None:
    """Negative trade_count rows are removed and counted separately."""
    frame = _ohlcv_frame(trade_counts=[42, -1, 7])
    cleaned, report = OHLCVCleaner().clean(frame)
    assert cleaned.height == 2
    assert report.invalid_trade_count_rows_removed == 1
    assert report.warnings == ("Removed 1 row(s) with invalid trade_count.",)


# --- Invalid timestamps ---


def test_removes_close_time_not_after_open_time() -> None:
    """Rows with close_time <= open_time are removed."""
    frame = _ohlcv_frame(
        open_times=[_START, _START + _INTERVAL, _START + 2 * _INTERVAL],
        close_times=[
            _START + _INTERVAL - 1,
            _START + _INTERVAL,
            _START + 2 * _INTERVAL - 5,
        ],
    )
    cleaned, report = OHLCVCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.invalid_timestamp_rows_removed == 2
    assert cleaned.get_column("open_time").to_list() == [_START]


# --- Sort / stability / no mutation of values ---


def test_sorts_by_open_time_with_stable_order() -> None:
    """Output is sorted by open_time preserving relative order of ties."""
    frame = _ohlcv_frame(
        open_times=[_START + _INTERVAL, _START, _START + _INTERVAL],
        opens=[102.0, 100.0, 103.0],
        highs=[104.0, 101.0, 105.0],
        lows=[101.0, 99.0, 102.0],
        closes=[103.0, 100.5, 104.0],
    )
    cleaned, report = OHLCVCleaner().clean(frame)
    assert cleaned.get_column("open_time").to_list() == [
        _START,
        _START + _INTERVAL,
        _START + _INTERVAL,
    ]
    assert cleaned.get_column("open").to_list() == [100.0, 102.0, 103.0]
    assert report.duplicates_removed == 0


def test_does_not_fill_interpolate_smooth_or_clip() -> None:
    """Cleaning never invents prices; gap rows stay absent and values intact."""
    frame = _ohlcv_frame(
        open_times=[_START, _START + 2 * _INTERVAL],
        opens=[100.0, 200.0],
        highs=[101.0, 201.0],
        lows=[99.0, 199.0],
        closes=[100.5, 200.5],
    )
    cleaned, _report = OHLCVCleaner().clean(frame)
    assert cleaned.height == 2
    assert cleaned.get_column("open").to_list() == [100.0, 200.0]
    assert cleaned.get_column("open_time").to_list() == [
        _START,
        _START + 2 * _INTERVAL,
    ]


def test_does_not_modify_timestamps_of_kept_rows() -> None:
    """Kept rows retain original open_time and close_time values."""
    frame = _ohlcv_frame(
        open_times=[_START + _INTERVAL, _START],
        close_times=[_START + 2 * _INTERVAL - 1, _START + _INTERVAL - 1],
    )
    cleaned, _report = OHLCVCleaner().clean(frame)
    assert cleaned.get_column("open_time").to_list() == [_START, _START + _INTERVAL]
    assert cleaned.get_column("close_time").to_list() == [
        _START + _INTERVAL - 1,
        _START + 2 * _INTERVAL - 1,
    ]


# --- Attribution / sequential counting ---


def test_row_failing_multiple_rules_counted_once_by_first_rule() -> None:
    """A row matching earlier and later rules is attributed to the earlier one."""
    base = _ohlcv_frame(open_times=[_START], volumes=[10.0])
    exact_duplicate = base
    negative_volume = _ohlcv_frame(open_times=[_START], volumes=[-5.0])
    frame = pl.concat([base, exact_duplicate, negative_volume])
    cleaned, report = OHLCVCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.duplicates_removed == 1
    assert report.invalid_volume_rows_removed == 1
    assert report.rows_before == 3
    assert report.rows_after == 1


def test_combined_warnings_are_ordered_and_complete() -> None:
    """Warnings cover every non-zero removal category in stable order."""
    rows = [
        # valid
        {
            "symbol": _SYMBOL,
            "timeframe": _TIMEFRAME,
            "open_time": _START,
            "close_time": _START + _INTERVAL - 1,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10.0,
            "quote_volume": 1000.0,
            "trade_count": 42,
        },
        # exact duplicate of valid
        {
            "symbol": _SYMBOL,
            "timeframe": _TIMEFRAME,
            "open_time": _START,
            "close_time": _START + _INTERVAL - 1,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10.0,
            "quote_volume": 1000.0,
            "trade_count": 42,
        },
        # null open
        {
            "symbol": _SYMBOL,
            "timeframe": _TIMEFRAME,
            "open_time": _START + _INTERVAL,
            "close_time": _START + 2 * _INTERVAL - 1,
            "open": None,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10.0,
            "quote_volume": 1000.0,
            "trade_count": 42,
        },
        # invalid price (high too low)
        {
            "symbol": _SYMBOL,
            "timeframe": _TIMEFRAME,
            "open_time": _START + 2 * _INTERVAL,
            "close_time": _START + 3 * _INTERVAL - 1,
            "open": 100.0,
            "high": 99.0,
            "low": 98.0,
            "close": 100.5,
            "volume": 10.0,
            "quote_volume": 1000.0,
            "trade_count": 42,
        },
        # invalid volume
        {
            "symbol": _SYMBOL,
            "timeframe": _TIMEFRAME,
            "open_time": _START + 3 * _INTERVAL,
            "close_time": _START + 4 * _INTERVAL - 1,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": -1.0,
            "quote_volume": 1000.0,
            "trade_count": 42,
        },
        # invalid trade_count
        {
            "symbol": _SYMBOL,
            "timeframe": _TIMEFRAME,
            "open_time": _START + 4 * _INTERVAL,
            "close_time": _START + 5 * _INTERVAL - 1,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10.0,
            "quote_volume": 1000.0,
            "trade_count": -3,
        },
        # invalid timestamp
        {
            "symbol": _SYMBOL,
            "timeframe": _TIMEFRAME,
            "open_time": _START + 5 * _INTERVAL,
            "close_time": _START + 5 * _INTERVAL,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10.0,
            "quote_volume": 1000.0,
            "trade_count": 42,
        },
    ]
    frame = pl.DataFrame(rows)
    cleaned, report = OHLCVCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.duplicates_removed == 1
    assert report.null_rows_removed == 1
    assert report.invalid_price_rows_removed == 1
    assert report.invalid_volume_rows_removed == 1
    assert report.invalid_trade_count_rows_removed == 1
    assert report.invalid_timestamp_rows_removed == 1
    assert report.warnings == (
        "Removed 1 exact duplicate row(s).",
        "Removed 1 row(s) with null or NaN mandatory values.",
        "Removed 1 row(s) with invalid OHLC prices.",
        "Removed 1 row(s) with invalid volume fields.",
        "Removed 1 row(s) with invalid trade_count.",
        "Removed 1 row(s) with invalid timestamps.",
    )


# --- Schema failures ---


def test_missing_mandatory_column_raises() -> None:
    """Missing mandatory columns raise ProcessingValidationError."""
    frame = _ohlcv_frame().drop("volume")
    with pytest.raises(ProcessingValidationError, match="missing required OHLCV columns") as exc:
        OHLCVCleaner().clean(frame)
    assert exc.value.error_code == "PROCESSING-CLEANING-OHLCV-001"
    assert "volume" in exc.value.details["missing_columns"]  # type: ignore[operator]


def test_preserves_extra_columns() -> None:
    """Non-mandatory extra columns are retained on kept rows."""
    frame = _ohlcv_frame().with_columns(pl.lit("x").alias("extra"))
    cleaned, report = OHLCVCleaner().clean(frame)
    assert "extra" in cleaned.columns
    assert cleaned.get_column("extra").to_list() == ["x", "x", "x"]
    assert report.rows_after == 3
