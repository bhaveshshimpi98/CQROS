"""Unit tests for CQROS OHLCV processing module."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, is_dataclass

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.core.constants import MILLISECONDS_PER_MINUTE
from cqros.core.exceptions import ValidationError
from cqros.processing.exceptions import ProcessingValidationError
from cqros.processing.interfaces import ProcessingStep
from cqros.processing.ohlcv import (
    DetectGapProcessor,
    GapDetectionReport,
    OHLCVProcessingPipeline,
    RemoveDuplicateTimestampProcessor,
    SortByTimestampProcessor,
    ValidateOHLCProcessor,
    ValidateSchemaProcessor,
    ValidateTimestampProcessor,
    ValidateVolumeProcessor,
)

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
    timeframe: str = _TIMEFRAME,
) -> pl.DataFrame:
    """Build a canonical OHLCV frame for processing tests."""
    if open_times is None:
        open_times = [
            _START,
            _START + _INTERVAL,
            _START + 2 * _INTERVAL,
        ]
    row_count = len(open_times)
    return pl.DataFrame(
        {
            "symbol": [_SYMBOL] * row_count,
            "timeframe": [timeframe] * row_count,
            "open_time": open_times,
            "close_time": [value + _INTERVAL - 1 for value in open_times],
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


# --- SortByTimestampProcessor ---


def test_sort_by_timestamp_ascending_and_stable() -> None:
    """Rows are sorted ascending by open_time with stable relative order."""
    frame = _ohlcv_frame(
        open_times=[_START + _INTERVAL, _START, _START + _INTERVAL],
        opens=[2.0, 1.0, 3.0],
    )
    result = SortByTimestampProcessor().process(frame)
    assert result.get_column("open_time").to_list() == [
        _START,
        _START + _INTERVAL,
        _START + _INTERVAL,
    ]
    assert result.get_column("open").to_list() == [1.0, 2.0, 3.0]


def test_sort_does_not_mutate_input() -> None:
    """Sort returns a new frame and leaves the input unchanged."""
    frame = _ohlcv_frame(open_times=[_START + _INTERVAL, _START])
    before = frame.clone()
    result = SortByTimestampProcessor().process(frame)
    assert_frame_equal(frame, before)
    assert result is not frame


def test_sort_satisfies_processing_step_protocol() -> None:
    """SortByTimestampProcessor satisfies ProcessingStep."""
    assert isinstance(SortByTimestampProcessor(), ProcessingStep)


# --- RemoveDuplicateTimestampProcessor ---


def test_remove_duplicates_keeps_first_and_reports_count() -> None:
    """Duplicate open_time rows are removed; first occurrence is kept."""
    frame = _ohlcv_frame(
        open_times=[_START, _START, _START + _INTERVAL],
        opens=[1.0, 2.0, 3.0],
    )
    processor = RemoveDuplicateTimestampProcessor()
    result = processor.process(frame)
    assert result.height == 2
    assert result.get_column("open").to_list() == [1.0, 3.0]
    assert processor.removed_count == 1


def test_remove_duplicates_zero_when_unique() -> None:
    """removed_count is zero when timestamps are unique."""
    processor = RemoveDuplicateTimestampProcessor()
    result = processor.process(_ohlcv_frame())
    assert result.height == 3
    assert processor.removed_count == 0


def test_remove_duplicates_does_not_mutate_input() -> None:
    """Dedupe leaves the caller frame unchanged."""
    frame = _ohlcv_frame(open_times=[_START, _START])
    before = deepcopy(frame.to_dicts())
    _ = RemoveDuplicateTimestampProcessor().process(frame)
    assert frame.to_dicts() == before


# --- ValidateSchemaProcessor ---


def test_validate_schema_accepts_complete_frame() -> None:
    """A complete OHLCV schema passes validation."""
    frame = _ohlcv_frame()
    result = ValidateSchemaProcessor().process(frame)
    assert_frame_equal(result, frame)
    assert result is not frame


def test_validate_schema_rejects_missing_columns() -> None:
    """Missing required columns raise ProcessingValidationError."""
    frame = _ohlcv_frame().drop("volume")
    with pytest.raises(ProcessingValidationError, match="missing required OHLCV columns") as exc:
        ValidateSchemaProcessor().process(frame)
    assert exc.value.error_code == "PROCESSING-OHLCV-001"
    assert "volume" in exc.value.details["missing_columns"]  # type: ignore[operator]


# --- ValidateTimestampProcessor ---


def test_validate_timestamps_accepts_valid_frame() -> None:
    """Strictly increasing integer timestamps with valid close times pass."""
    frame = _ohlcv_frame()
    result = ValidateTimestampProcessor().process(frame)
    assert_frame_equal(result, frame)


def test_validate_timestamps_rejects_non_increasing_open_time() -> None:
    """Non-increasing open_time raises ProcessingValidationError."""
    frame = _ohlcv_frame(open_times=[_START, _START + _INTERVAL, _START + _INTERVAL])
    with pytest.raises(ProcessingValidationError, match="strictly increasing") as exc:
        ValidateTimestampProcessor().process(frame)
    assert exc.value.error_code == "PROCESSING-OHLCV-004"


def test_validate_timestamps_rejects_close_not_after_open() -> None:
    """close_time <= open_time raises ProcessingValidationError."""
    frame = _ohlcv_frame()
    frame = frame.with_columns(pl.col("open_time").alias("close_time"))
    with pytest.raises(ProcessingValidationError, match="close_time must be greater") as exc:
        ValidateTimestampProcessor().process(frame)
    assert exc.value.error_code == "PROCESSING-OHLCV-005"


def test_validate_timestamps_rejects_non_integer_dtype() -> None:
    """Float timestamp columns are rejected."""
    frame = _ohlcv_frame().with_columns(pl.col("open_time").cast(pl.Float64))
    with pytest.raises(ProcessingValidationError, match="integer dtype") as exc:
        ValidateTimestampProcessor().process(frame)
    assert exc.value.error_code == "PROCESSING-OHLCV-002"


def test_validate_timestamps_rejects_future_when_enabled() -> None:
    """Future timestamps are rejected when reject_future is enabled."""
    frame = _ohlcv_frame()
    reference = _START
    with pytest.raises(ProcessingValidationError, match="not be in the future") as exc:
        ValidateTimestampProcessor(
            reject_future=True,
            reference_time_ms=reference,
        ).process(frame)
    assert exc.value.error_code == "PROCESSING-OHLCV-006"


def test_validate_timestamps_allows_future_when_disabled() -> None:
    """Future timestamps are allowed when reject_future is False."""
    frame = _ohlcv_frame()
    result = ValidateTimestampProcessor(
        reject_future=False,
        reference_time_ms=_START,
    ).process(frame)
    assert result.height == frame.height


# --- ValidateOHLCProcessor ---


def test_validate_ohlc_accepts_consistent_rows() -> None:
    """Consistent OHLC rows pass validation."""
    frame = _ohlcv_frame()
    result = ValidateOHLCProcessor().process(frame)
    assert_frame_equal(result, frame)


@pytest.mark.parametrize(
    ("highs", "lows", "opens", "closes", "match"),
    [
        ([99.0], [98.0], [100.0], [100.5], "high must be greater than or equal to open"),
        ([100.0], [98.0], [100.0], [101.0], "high must be greater than or equal to close"),
        ([101.0], [102.0], [100.0], [100.0], "high must be greater than or equal to low"),
        ([102.0], [100.5], [100.0], [101.0], "low must be less than or equal to open"),
        ([102.0], [100.5], [101.0], [100.0], "low must be less than or equal to close"),
    ],
)
def test_validate_ohlc_rejects_inconsistent_rows(
    highs: list[float],
    lows: list[float],
    opens: list[float],
    closes: list[float],
    match: str,
) -> None:
    """OHLC consistency violations raise ProcessingValidationError."""
    frame = _ohlcv_frame(
        open_times=[_START],
        highs=highs,
        lows=lows,
        opens=opens,
        closes=closes,
    )
    with pytest.raises(ProcessingValidationError, match=match) as exc:
        ValidateOHLCProcessor().process(frame)
    assert exc.value.error_code == "PROCESSING-OHLCV-007"


# --- ValidateVolumeProcessor ---


def test_validate_volume_accepts_non_negative() -> None:
    """Non-negative volume fields pass validation."""
    frame = _ohlcv_frame(
        open_times=[_START],
        volumes=[0.0],
        quote_volumes=[0.0],
        trade_counts=[0],
    )
    result = ValidateVolumeProcessor().process(frame)
    assert result.height == 1


@pytest.mark.parametrize(
    ("volumes", "quote_volumes", "trade_counts", "column"),
    [
        ([-1.0], [1.0], [1], "volume"),
        ([1.0], [-1.0], [1], "quote_volume"),
        ([1.0], [1.0], [-1], "trade_count"),
    ],
)
def test_validate_volume_rejects_negatives(
    volumes: list[float],
    quote_volumes: list[float],
    trade_counts: list[int],
    column: str,
) -> None:
    """Negative volume fields raise ProcessingValidationError."""
    frame = _ohlcv_frame(
        open_times=[_START],
        volumes=volumes,
        quote_volumes=quote_volumes,
        trade_counts=trade_counts,
    )
    with pytest.raises(ProcessingValidationError, match=column) as exc:
        ValidateVolumeProcessor().process(frame)
    assert exc.value.error_code == "PROCESSING-OHLCV-008"


# --- DetectGapProcessor ---


def test_detect_gaps_reports_missing_intervals_without_mutation() -> None:
    """Gaps are reported as metadata; rows are not filled."""
    frame = _ohlcv_frame(
        open_times=[
            _START,
            _START + 3 * _INTERVAL,
            _START + 4 * _INTERVAL,
        ]
    )
    before = frame.clone()
    processor = DetectGapProcessor(interval_ms=_INTERVAL)
    result = processor.process(frame)
    assert_frame_equal(result, before)
    assert result is not frame
    report = processor.last_report
    assert report is not None
    assert report.missing_intervals == (_START + _INTERVAL, _START + 2 * _INTERVAL)
    assert report.gap_count == 1
    assert report.largest_gap == 2


def test_detect_gaps_multiple_regions() -> None:
    """Multiple gap regions increase gap_count and largest_gap correctly."""
    frame = _ohlcv_frame(
        open_times=[
            _START,
            _START + 2 * _INTERVAL,
            _START + 5 * _INTERVAL,
        ]
    )
    processor = DetectGapProcessor(interval_ms=_INTERVAL)
    _ = processor.process(frame)
    report = processor.last_report
    assert report is not None
    assert report.gap_count == 2
    assert report.largest_gap == 2
    assert report.missing_intervals == (
        _START + _INTERVAL,
        _START + 3 * _INTERVAL,
        _START + 4 * _INTERVAL,
    )


def test_detect_gaps_no_gaps() -> None:
    """Contiguous frames produce an empty gap report."""
    processor = DetectGapProcessor(interval_ms=_INTERVAL)
    _ = processor.process(_ohlcv_frame())
    report = processor.last_report
    assert report == GapDetectionReport(missing_intervals=(), gap_count=0, largest_gap=0)


def test_detect_gaps_rejects_invalid_interval() -> None:
    """Non-positive interval_ms raises ValidationError at construction."""
    with pytest.raises(ValidationError, match="interval_ms"):
        DetectGapProcessor(interval_ms=0)


def test_gap_detection_report_is_frozen() -> None:
    """GapDetectionReport is an immutable slotted dataclass."""
    report = GapDetectionReport(missing_intervals=(1,), gap_count=1, largest_gap=1)
    assert is_dataclass(report)
    with pytest.raises(FrozenInstanceError):
        report.gap_count = 2  # type: ignore[misc]


# --- OHLCVProcessingPipeline ---


def test_ohlcv_pipeline_processes_raw_frame() -> None:
    """Composed pipeline sorts, dedupes, validates, and detects gaps."""
    frame = _ohlcv_frame(
        open_times=[
            _START + 2 * _INTERVAL,
            _START,
            _START,
            _START + 4 * _INTERVAL,
        ],
        opens=[104.0, 100.0, 101.0, 105.0],
        highs=[105.0, 102.0, 103.0, 106.0],
        lows=[103.0, 99.0, 100.0, 104.0],
        closes=[104.5, 101.0, 102.0, 105.5],
    )
    pipeline = OHLCVProcessingPipeline(interval_ms=_INTERVAL)
    result = pipeline.run(frame)
    assert result.get_column("open_time").to_list() == [
        _START,
        _START + 2 * _INTERVAL,
        _START + 4 * _INTERVAL,
    ]
    assert result.get_column("open").to_list() == [100.0, 104.0, 105.0]
    assert pipeline.removed_duplicate_count == 1
    report = pipeline.gap_report
    assert report is not None
    assert report.gap_count == 2
    assert report.missing_intervals == (
        _START + _INTERVAL,
        _START + 3 * _INTERVAL,
    )


def test_ohlcv_pipeline_does_not_mutate_input() -> None:
    """Pipeline execution leaves the caller frame unchanged."""
    frame = _ohlcv_frame(open_times=[_START + _INTERVAL, _START])
    snapshot = deepcopy(frame.to_dicts())
    _ = OHLCVProcessingPipeline(interval_ms=_INTERVAL).run(frame)
    assert frame.to_dicts() == snapshot


def test_ohlcv_pipeline_propagates_validation_errors() -> None:
    """Schema validation failures propagate as ProcessingValidationError."""
    frame = _ohlcv_frame().drop("high")
    with pytest.raises(ProcessingValidationError, match="missing required"):
        OHLCVProcessingPipeline(interval_ms=_INTERVAL).run(frame)


def test_ohlcv_pipeline_step_order() -> None:
    """Pipeline exposes processors in the documented execution order."""
    pipeline = OHLCVProcessingPipeline(interval_ms=_INTERVAL)
    names = tuple(step.name for step in pipeline.steps)
    assert names == (
        "validate_ohlcv_schema",
        "sort_by_timestamp",
        "remove_duplicate_timestamps",
        "validate_ohlcv_timestamps",
        "validate_ohlc",
        "validate_volume",
        "detect_ohlcv_gaps",
    )


def test_package_exports_ohlcv_symbols() -> None:
    """OHLCV processors and pipeline are exported from the package."""
    import cqros.processing as processing_package

    for name in (
        "GapDetectionReport",
        "SortByTimestampProcessor",
        "RemoveDuplicateTimestampProcessor",
        "ValidateSchemaProcessor",
        "ValidateTimestampProcessor",
        "ValidateOHLCProcessor",
        "ValidateVolumeProcessor",
        "DetectGapProcessor",
        "OHLCVProcessingPipeline",
    ):
        assert name in processing_package.__all__
        assert getattr(processing_package, name).__name__ == name
