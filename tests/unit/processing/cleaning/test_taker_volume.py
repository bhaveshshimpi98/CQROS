"""Unit tests for CQROS taker-volume cleaning module."""

from __future__ import annotations

from copy import deepcopy
from math import nan

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.core.constants import MILLISECONDS_PER_MINUTE
from cqros.processing import CleaningReport, TakerVolumeCleaner
from cqros.processing.cleaning import (
    CleaningReport as CleaningReportFromPackage,
)
from cqros.processing.cleaning import (
    TakerVolumeCleaner as TakerVolumeCleanerFromPackage,
)
from cqros.processing.cleaning import taker_volume as taker_volume_module
from cqros.processing.exceptions import ProcessingValidationError

_SYMBOL = "BTCUSDT"
_START = 1_699_999_980_000
_INTERVAL = 5 * MILLISECONDS_PER_MINUTE


def _taker_volume_frame(
    *,
    timestamps: list[int] | None = None,
    buy_volumes: list[float | None] | None = None,
    sell_volumes: list[float | None] | None = None,
    buy_sell_ratios: list[float | None] | None = None,
    symbols: list[str] | None = None,
    include_buy_sell_ratio: bool = True,
) -> pl.DataFrame:
    """Build a canonical taker-volume frame for cleaning tests."""
    if timestamps is None:
        timestamps = [
            _START,
            _START + _INTERVAL,
            _START + 2 * _INTERVAL,
        ]
    row_count = len(timestamps)
    buys = list(buy_volumes) if buy_volumes is not None else [60.0] * row_count
    sells = list(sell_volumes) if sell_volumes is not None else [40.0] * row_count
    data: dict[str, list[object]] = {
        "symbol": symbols if symbols is not None else [_SYMBOL] * row_count,
        "timestamp": list(timestamps),
        "buy_volume": buys,
        "sell_volume": sells,
    }
    schema: dict[str, pl.DataType] = {
        "symbol": pl.String,
        "timestamp": pl.Int64,
        "buy_volume": pl.Float64,
        "sell_volume": pl.Float64,
    }
    if include_buy_sell_ratio:
        if buy_sell_ratios is not None:
            ratios: list[float | None] = list(buy_sell_ratios)
        else:
            ratios = [
                (
                    (None if sell == 0 else buy / sell)
                    if buy is not None and sell is not None
                    else None
                )
                for buy, sell in zip(buys, sells, strict=True)
            ]
        data["buy_sell_ratio"] = ratios
        schema["buy_sell_ratio"] = pl.Float64
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
    """cleaning package and processing package export the same TakerVolumeCleaner."""
    assert CleaningReport is CleaningReportFromPackage
    assert TakerVolumeCleaner is TakerVolumeCleanerFromPackage
    assert taker_volume_module.__all__ == ["TakerVolumeCleaner"]


# --- Happy path ---


def test_clean_valid_frame_unchanged_values_and_zero_removals() -> None:
    """A valid frame keeps volumes and reports zero removals."""
    frame = _taker_volume_frame()
    cleaned, report = TakerVolumeCleaner().clean(frame)
    assert_frame_equal(cleaned, frame)
    assert report == _zero_report(rows_before=3, rows_after=3)


def test_clean_already_clean_frame() -> None:
    """An already-clean sorted frame is unchanged aside from ratio recompute."""
    frame = _taker_volume_frame(
        timestamps=[_START, _START + _INTERVAL],
        buy_volumes=[0.0, 80.0],
        sell_volumes=[0.0, 40.0],
        buy_sell_ratios=[None, 2.0],
    )
    cleaned, report = TakerVolumeCleaner().clean(frame)
    assert_frame_equal(cleaned, frame)
    assert report == _zero_report(rows_before=2, rows_after=2)


def test_clean_empty_frame() -> None:
    """An empty schema-valid frame cleans to empty with zero removals."""
    frame = _taker_volume_frame(timestamps=[])
    cleaned, report = TakerVolumeCleaner().clean(frame)
    assert cleaned.height == 0
    assert report == _zero_report(rows_before=0, rows_after=0)
    assert "buy_sell_ratio" in cleaned.columns


def test_clean_does_not_mutate_input() -> None:
    """clean leaves the caller frame unchanged."""
    frame = _taker_volume_frame(
        timestamps=[_START + _INTERVAL, _START],
        buy_volumes=[80.0, 60.0],
        sell_volumes=[40.0, 40.0],
        buy_sell_ratios=[2.0, 1.5],
    )
    before = deepcopy(frame.to_dicts())
    cleaned, _report = TakerVolumeCleaner().clean(frame)
    assert frame.to_dicts() == before
    assert cleaned is not frame


def test_clean_is_deterministic() -> None:
    """Repeated cleaning of the same input yields identical outputs."""
    frame = _taker_volume_frame(
        timestamps=[_START + _INTERVAL, _START, _START + _INTERVAL],
        buy_volumes=[80.0, 60.0, -1.0],
        sell_volumes=[40.0, 40.0, 40.0],
    )
    cleaner = TakerVolumeCleaner()
    first_frame, first_report = cleaner.clean(frame)
    second_frame, second_report = cleaner.clean(frame)
    assert_frame_equal(first_frame, second_frame)
    assert first_report == second_report


# --- Duplicate timestamps ---


def test_removes_duplicate_timestamps_keeps_first() -> None:
    """Duplicate timestamp rows are removed; first occurrence is kept."""
    frame = _taker_volume_frame(
        timestamps=[_START, _START, _START + _INTERVAL],
        buy_volumes=[60.0, 999.0, 80.0],
        sell_volumes=[40.0, 1.0, 40.0],
    )
    cleaned, report = TakerVolumeCleaner().clean(frame)
    assert cleaned.height == 2
    assert report.duplicates_removed == 1
    assert report.warnings == ("Removed 1 duplicate timestamp row(s).",)
    assert cleaned.get_column("buy_volume").to_list() == [60.0, 80.0]


# --- NULL / NaN volumes ---


def test_removes_null_buy_volume() -> None:
    """Rows with null buy_volume are removed and counted."""
    frame = _taker_volume_frame(
        timestamps=[_START, _START + _INTERVAL],
        buy_volumes=[None, 80.0],
        sell_volumes=[40.0, 40.0],
    )
    cleaned, report = TakerVolumeCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.null_rows_removed == 1
    assert cleaned.get_column("buy_volume").to_list() == [80.0]


def test_removes_null_sell_volume() -> None:
    """Rows with null sell_volume are removed and counted."""
    frame = _taker_volume_frame(
        timestamps=[_START, _START + _INTERVAL],
        buy_volumes=[60.0, 80.0],
        sell_volumes=[None, 40.0],
    )
    cleaned, report = TakerVolumeCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.null_rows_removed == 1
    assert cleaned.get_column("sell_volume").to_list() == [40.0]


def test_removes_nan_buy_volume() -> None:
    """Rows with NaN buy_volume are removed and counted."""
    frame = _taker_volume_frame(
        timestamps=[_START, _START + _INTERVAL],
        buy_volumes=[nan, 80.0],
        sell_volumes=[40.0, 40.0],
    )
    cleaned, report = TakerVolumeCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.null_rows_removed == 1
    assert cleaned.get_column("buy_volume").to_list() == [80.0]


def test_removes_nan_sell_volume() -> None:
    """Rows with NaN sell_volume are removed and counted."""
    frame = _taker_volume_frame(
        timestamps=[_START, _START + _INTERVAL],
        buy_volumes=[60.0, 80.0],
        sell_volumes=[nan, 40.0],
    )
    cleaned, report = TakerVolumeCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.null_rows_removed == 1
    assert cleaned.get_column("sell_volume").to_list() == [40.0]


# --- Negative volumes ---


def test_removes_negative_buy_volume() -> None:
    """Negative buy_volume rows are removed and counted."""
    frame = _taker_volume_frame(
        timestamps=[_START, _START + _INTERVAL],
        buy_volumes=[-1.0, 80.0],
        sell_volumes=[40.0, 40.0],
    )
    cleaned, report = TakerVolumeCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.invalid_volume_rows_removed == 1
    assert report.warnings == ("Removed 1 row(s) with negative buy_volume or sell_volume.",)
    assert cleaned.get_column("buy_volume").to_list() == [80.0]


def test_removes_negative_sell_volume() -> None:
    """Negative sell_volume rows are removed and counted."""
    frame = _taker_volume_frame(
        timestamps=[_START, _START + _INTERVAL],
        buy_volumes=[60.0, 80.0],
        sell_volumes=[-5.0, 40.0],
    )
    cleaned, report = TakerVolumeCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.invalid_volume_rows_removed == 1
    assert cleaned.get_column("sell_volume").to_list() == [40.0]


def test_retains_zero_sell_volume() -> None:
    """Zero sell_volume is retained with a null buy_sell_ratio."""
    frame = _taker_volume_frame(
        timestamps=[_START],
        buy_volumes=[60.0],
        sell_volumes=[0.0],
        buy_sell_ratios=[999.0],
    )
    cleaned, report = TakerVolumeCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.invalid_volume_rows_removed == 0
    assert report.null_rows_removed == 0
    assert cleaned.get_column("sell_volume").to_list() == [0.0]
    assert cleaned.get_column("buy_sell_ratio").to_list() == [None]


# --- buy_sell_ratio recomputation ---


def test_recomputes_buy_sell_ratio() -> None:
    """buy_sell_ratio is always recomputed from cleaned volumes."""
    frame = _taker_volume_frame(
        timestamps=[_START, _START + _INTERVAL],
        buy_volumes=[60.0, 80.0],
        sell_volumes=[40.0, 20.0],
        buy_sell_ratios=[0.0, 0.0],
    )
    cleaned, report = TakerVolumeCleaner().clean(frame)
    assert report.rows_after == 2
    assert cleaned.get_column("buy_sell_ratio").to_list() == [1.5, 4.0]


def test_ratio_null_when_sell_volume_zero() -> None:
    """buy_sell_ratio is null when sell_volume is zero."""
    frame = _taker_volume_frame(
        timestamps=[_START, _START + _INTERVAL],
        buy_volumes=[0.0, 50.0],
        sell_volumes=[0.0, 25.0],
        buy_sell_ratios=[1.0, 1.0],
    )
    cleaned, _report = TakerVolumeCleaner().clean(frame)
    assert cleaned.get_column("buy_sell_ratio").to_list() == [None, 2.0]


def test_adds_buy_sell_ratio_when_missing() -> None:
    """Missing buy_sell_ratio column is created by recomputation."""
    frame = _taker_volume_frame(
        timestamps=[_START],
        buy_volumes=[60.0],
        sell_volumes=[40.0],
        include_buy_sell_ratio=False,
    )
    cleaned, report = TakerVolumeCleaner().clean(frame)
    assert report.rows_after == 1
    assert cleaned.get_column("buy_sell_ratio").to_list() == [1.5]


# --- Timestamp validation ---


def test_removes_non_positive_timestamps() -> None:
    """Zero and negative timestamp values are removed."""
    frame = _taker_volume_frame(
        timestamps=[0, -1, _START],
        buy_volumes=[60.0, 70.0, 80.0],
        sell_volumes=[40.0, 40.0, 40.0],
    )
    cleaned, report = TakerVolumeCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.invalid_timestamp_rows_removed == 2
    assert cleaned.get_column("timestamp").to_list() == [_START]


def test_removes_null_timestamps() -> None:
    """Null timestamp values are removed as invalid timestamps."""
    frame = pl.DataFrame(
        {
            "symbol": [_SYMBOL, _SYMBOL],
            "timestamp": [None, _START],
            "buy_volume": [60.0, 80.0],
            "sell_volume": [40.0, 40.0],
            "buy_sell_ratio": [1.5, 2.0],
        },
        schema={
            "symbol": pl.String,
            "timestamp": pl.Int64,
            "buy_volume": pl.Float64,
            "sell_volume": pl.Float64,
            "buy_sell_ratio": pl.Float64,
        },
    )
    cleaned, report = TakerVolumeCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.invalid_timestamp_rows_removed == 1
    assert cleaned.get_column("timestamp").to_list() == [_START]


def test_removes_fractional_timestamps() -> None:
    """Non-integer timestamp values are removed."""
    frame = pl.DataFrame(
        {
            "symbol": [_SYMBOL, _SYMBOL],
            "timestamp": [float(_START) + 0.5, float(_START + _INTERVAL)],
            "buy_volume": [60.0, 80.0],
            "sell_volume": [40.0, 40.0],
            "buy_sell_ratio": [1.5, 2.0],
        },
        schema={
            "symbol": pl.String,
            "timestamp": pl.Float64,
            "buy_volume": pl.Float64,
            "sell_volume": pl.Float64,
            "buy_sell_ratio": pl.Float64,
        },
    )
    cleaned, report = TakerVolumeCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.invalid_timestamp_rows_removed == 1
    assert cleaned.get_column("timestamp").to_list() == [float(_START + _INTERVAL)]


# --- Sort / stability ---


def test_sorts_by_timestamp_with_stable_order() -> None:
    """Output is sorted ascending by timestamp with stable relative order."""
    frame = _taker_volume_frame(
        timestamps=[_START + _INTERVAL, _START, _START + 2 * _INTERVAL],
        buy_volumes=[80.0, 60.0, 90.0],
        sell_volumes=[40.0, 40.0, 45.0],
    )
    cleaned, report = TakerVolumeCleaner().clean(frame)
    assert cleaned.get_column("timestamp").to_list() == [
        _START,
        _START + _INTERVAL,
        _START + 2 * _INTERVAL,
    ]
    assert cleaned.get_column("buy_volume").to_list() == [60.0, 80.0, 90.0]
    assert report.duplicates_removed == 0


def test_does_not_modify_timestamps_of_kept_rows() -> None:
    """Kept rows retain original timestamp values."""
    frame = _taker_volume_frame(
        timestamps=[_START + _INTERVAL, _START],
        buy_volumes=[80.0, 60.0],
        sell_volumes=[40.0, 40.0],
    )
    cleaned, _report = TakerVolumeCleaner().clean(frame)
    assert cleaned.get_column("timestamp").to_list() == [
        _START,
        _START + _INTERVAL,
    ]


# --- Report counters / attribution ---


def test_report_counters_and_combined_warnings() -> None:
    """Removal counters and warnings follow processing order attribution."""
    frame = pl.DataFrame(
        {
            "symbol": [_SYMBOL] * 6,
            "timestamp": [
                _START,
                _START,
                _START + _INTERVAL,
                _START + 2 * _INTERVAL,
                0,
                _START + 3 * _INTERVAL,
            ],
            "buy_volume": [60.0, 999.0, None, 80.0, 70.0, -1.0],
            "sell_volume": [40.0, 1.0, 40.0, None, 40.0, 40.0],
            "buy_sell_ratio": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        schema={
            "symbol": pl.String,
            "timestamp": pl.Int64,
            "buy_volume": pl.Float64,
            "sell_volume": pl.Float64,
            "buy_sell_ratio": pl.Float64,
        },
    )
    cleaned, report = TakerVolumeCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.rows_before == 6
    assert report.rows_after == 1
    assert report.duplicates_removed == 1
    assert report.null_rows_removed == 2
    assert report.invalid_timestamp_rows_removed == 1
    assert report.invalid_volume_rows_removed == 1
    assert report.invalid_price_rows_removed == 0
    assert report.invalid_trade_count_rows_removed == 0
    assert report.warnings == (
        "Removed 1 duplicate timestamp row(s).",
        "Removed 2 row(s) with null or NaN buy_volume or sell_volume.",
        "Removed 1 row(s) with invalid timestamps.",
        "Removed 1 row(s) with negative buy_volume or sell_volume.",
    )
    assert cleaned.get_column("timestamp").to_list() == [_START]
    assert cleaned.get_column("buy_sell_ratio").to_list() == [1.5]


def test_row_failing_multiple_rules_counted_once_by_first_rule() -> None:
    """A row matching earlier and later rules is attributed to the earlier one."""
    frame = _taker_volume_frame(
        timestamps=[_START, _START],
        buy_volumes=[60.0, None],
        sell_volumes=[40.0, -1.0],
    )
    cleaned, report = TakerVolumeCleaner().clean(frame)
    assert cleaned.height == 1
    assert report.duplicates_removed == 1
    assert report.null_rows_removed == 0
    assert report.invalid_volume_rows_removed == 0
    assert report.rows_before == 2
    assert report.rows_after == 1


# --- Schema validation ---


def test_missing_mandatory_column_raises() -> None:
    """Missing mandatory columns raise ProcessingValidationError."""
    frame = _taker_volume_frame().drop("buy_volume")
    with pytest.raises(
        ProcessingValidationError,
        match="missing required taker volume columns",
    ) as exc:
        TakerVolumeCleaner().clean(frame)
    assert exc.value.error_code == "PROCESSING-CLEANING-TAKER-VOLUME-001"
    assert "buy_volume" in exc.value.details["missing_columns"]  # type: ignore[operator]


def test_preserves_extra_columns() -> None:
    """Non-mandatory extra columns are retained on kept rows."""
    frame = _taker_volume_frame().with_columns(pl.lit("x").alias("extra"))
    cleaned, report = TakerVolumeCleaner().clean(frame)
    assert "extra" in cleaned.columns
    assert cleaned.get_column("extra").to_list() == ["x", "x", "x"]
    assert report.rows_after == 3
