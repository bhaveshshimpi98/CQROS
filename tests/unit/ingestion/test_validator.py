"""Unit tests for CQROS market-data validation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, is_dataclass

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.core.constants import MILLISECONDS_PER_MINUTE
from cqros.core.exceptions import ValidationError
from cqros.ingestion import (
    MarketDataValidator,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)
from cqros.ingestion.validator import (
    MarketDataValidator as MarketDataValidatorDirect,
)
from cqros.ingestion.validator import (
    ValidationIssue as ValidationIssueDirect,
)
from cqros.ingestion.validator import (
    ValidationReport as ValidationReportDirect,
)
from cqros.ingestion.validator import (
    ValidationSeverity as ValidationSeverityDirect,
)

_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1m"
# Aligned to the 1-minute grid (Unix ms divisible by 60_000).
_START = 1_699_999_980_000


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
    """Build a canonical OHLCV frame for validation tests."""
    if open_times is None:
        open_times = [
            _START,
            _START + MILLISECONDS_PER_MINUTE,
            _START + 2 * MILLISECONDS_PER_MINUTE,
        ]
    row_count = len(open_times)
    return pl.DataFrame(
        {
            "symbol": [_SYMBOL] * row_count,
            "timeframe": [timeframe] * row_count,
            "open_time": open_times,
            "close_time": [value + MILLISECONDS_PER_MINUTE - 1 for value in open_times],
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


def test_exports_match_module_symbols() -> None:
    """Package exports match the validator module symbols."""
    assert MarketDataValidator is MarketDataValidatorDirect
    assert ValidationIssue is ValidationIssueDirect
    assert ValidationReport is ValidationReportDirect
    assert ValidationSeverity is ValidationSeverityDirect


def test_validation_issue_and_report_are_immutable() -> None:
    """ValidationIssue and ValidationReport are frozen slotted dataclasses."""
    issue = ValidationIssue(
        severity=ValidationSeverity.ERROR,
        check="schema",
        message="missing column",
    )
    report = ValidationReport(timeframe=_TIMEFRAME, row_count=0, issues=(issue,))
    assert is_dataclass(issue)
    assert is_dataclass(report)
    with pytest.raises(FrozenInstanceError):
        issue.message = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        report.row_count = 1  # type: ignore[misc]


def test_validate_accepts_clean_ohlcv_frame() -> None:
    """A clean aligned OHLCV frame produces a valid report."""
    frame = _ohlcv_frame()
    report = MarketDataValidator().validate(frame, _TIMEFRAME)
    assert report.is_valid is True
    assert report.has_errors() is False
    assert report.row_count == 3
    assert report.timeframe == _TIMEFRAME
    assert report.errors() == ()


def test_validate_does_not_mutate_input_frame() -> None:
    """Validation leaves the caller frame unchanged."""
    frame = _ohlcv_frame()
    before = frame.clone()
    snapshot = deepcopy(frame.to_dicts())
    _ = MarketDataValidator().validate(frame, _TIMEFRAME)
    assert_frame_equal(frame, before)
    assert frame.to_dicts() == snapshot


def test_validate_rejects_unsupported_timeframe() -> None:
    """Unsupported timeframes raise a contract ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        MarketDataValidator().validate(_ohlcv_frame(), "2m")
    assert exc_info.value.error_code == "INGESTION-VALIDATOR-001"


def test_schema_missing_column() -> None:
    """Missing required columns are reported as schema errors."""
    frame = _ohlcv_frame().drop("volume")
    report = MarketDataValidator().validate(frame, _TIMEFRAME)
    assert report.is_valid is False
    assert any(issue.check == "schema" for issue in report.errors())


def test_schema_wrong_dtype() -> None:
    """Incorrect column dtypes are reported as schema errors."""
    frame = _ohlcv_frame().with_columns(pl.col("open").cast(pl.Float32))
    report = MarketDataValidator().validate(frame, _TIMEFRAME)
    assert report.is_valid is False
    schema_errors = [issue for issue in report.errors() if issue.check == "schema"]
    assert schema_errors
    assert schema_errors[0].column == "open"


def test_schema_unexpected_column_is_warning() -> None:
    """Unexpected columns produce warnings without invalidating the report."""
    frame = _ohlcv_frame().with_columns(pl.lit("extra").alias("extra_col"))
    report = MarketDataValidator().validate(frame, _TIMEFRAME)
    assert report.is_valid is True
    assert report.has_warnings() is True
    assert any(issue.check == "schema" for issue in report.warnings())


def test_null_detection() -> None:
    """Null values in required columns are reported."""
    frame = _ohlcv_frame().with_columns(
        pl.when(pl.arange(0, pl.len()) == 1).then(None).otherwise(pl.col("close")).alias("close")
    )
    report = MarketDataValidator().validate(frame, _TIMEFRAME)
    assert report.is_valid is False
    null_issues = [issue for issue in report.errors() if issue.check == "nulls"]
    assert len(null_issues) == 1
    assert null_issues[0].column == "close"
    assert null_issues[0].row_index == 1
    assert null_issues[0].count == 1


def test_nan_detection() -> None:
    """NaN values in float columns are reported as null-check failures."""
    frame = _ohlcv_frame(opens=[100.0, float("nan"), 100.0])
    report = MarketDataValidator().validate(frame, _TIMEFRAME)
    assert report.is_valid is False
    assert any(issue.check == "nulls" and issue.column == "open" for issue in report.errors())


def test_ohlc_consistency_high_below_low() -> None:
    """High below low fails OHLC consistency."""
    frame = _ohlcv_frame(highs=[101.0, 98.0, 101.0], lows=[99.0, 99.0, 99.0])
    report = MarketDataValidator().validate(frame, _TIMEFRAME)
    assert report.is_valid is False
    assert any(issue.check == "ohlc_consistency" for issue in report.errors())


def test_ohlc_consistency_open_outside_range() -> None:
    """Open above high fails OHLC consistency."""
    frame = _ohlcv_frame(opens=[100.0, 102.0, 100.0], highs=[101.0, 101.0, 101.0])
    report = MarketDataValidator().validate(frame, _TIMEFRAME)
    assert report.is_valid is False
    assert any(issue.check == "ohlc_consistency" for issue in report.errors())


def test_non_negative_volume() -> None:
    """Negative volume fails the non-negative volume check."""
    frame = _ohlcv_frame(volumes=[10.0, -1.0, 10.0])
    report = MarketDataValidator().validate(frame, _TIMEFRAME)
    assert report.is_valid is False
    volume_issues = [issue for issue in report.errors() if issue.check == "non_negative_volume"]
    assert volume_issues
    assert volume_issues[0].column == "volume"
    assert volume_issues[0].row_index == 1


def test_duplicate_timestamps() -> None:
    """Duplicate open_time values are reported."""
    frame = _ohlcv_frame(
        open_times=[
            _START,
            _START,
            _START + MILLISECONDS_PER_MINUTE,
        ]
    )
    report = MarketDataValidator().validate(frame, _TIMEFRAME)
    assert report.is_valid is False
    assert any(issue.check == "duplicate_timestamps" for issue in report.errors())
    assert any(issue.check == "timestamp_monotonicity" for issue in report.errors())


def test_timestamp_monotonicity() -> None:
    """Out-of-order open_time values fail monotonicity."""
    frame = _ohlcv_frame(
        open_times=[
            _START,
            _START + 2 * MILLISECONDS_PER_MINUTE,
            _START + MILLISECONDS_PER_MINUTE,
        ]
    )
    report = MarketDataValidator().validate(frame, _TIMEFRAME)
    assert report.is_valid is False
    assert any(issue.check == "timestamp_monotonicity" for issue in report.errors())


def test_missing_timestamps() -> None:
    """Gaps larger than one interval are reported as missing timestamps."""
    frame = _ohlcv_frame(
        open_times=[
            _START,
            _START + MILLISECONDS_PER_MINUTE,
            _START + 4 * MILLISECONDS_PER_MINUTE,
        ]
    )
    report = MarketDataValidator().validate(frame, _TIMEFRAME)
    assert report.is_valid is False
    missing = [issue for issue in report.errors() if issue.check == "missing_timestamps"]
    assert len(missing) == 1
    assert missing[0].count == 2


def test_timeframe_alignment() -> None:
    """Misaligned open_time values fail timeframe alignment."""
    frame = _ohlcv_frame(
        open_times=[
            _START + 1,
            _START + MILLISECONDS_PER_MINUTE + 1,
            _START + 2 * MILLISECONDS_PER_MINUTE + 1,
        ]
    )
    report = MarketDataValidator().validate(frame, _TIMEFRAME)
    assert report.is_valid is False
    assert any(issue.check == "timeframe_alignment" for issue in report.errors())


def test_weekly_timeframe_alignment() -> None:
    """Weekly bars must open on Monday 00:00 UTC."""
    monday = 1_704_067_200_000  # 2024-01-01 Monday 00:00:00 UTC
    tuesday = monday + 86_400_000
    frame = _ohlcv_frame(
        open_times=[monday, tuesday],
        timeframe="1w",
    )
    report = MarketDataValidator().validate(frame, "1w")
    assert report.is_valid is False
    assert any(issue.check == "timeframe_alignment" for issue in report.errors())


def test_empty_frame_is_warning() -> None:
    """An empty but schema-valid frame is valid with an empty-dataset warning."""
    frame = _ohlcv_frame(open_times=[])
    report = MarketDataValidator().validate(frame, _TIMEFRAME)
    assert report.is_valid is True
    assert report.has_warnings() is True
    assert any(issue.check == "empty_dataset" for issue in report.warnings())


def test_report_helpers() -> None:
    """ValidationReport helpers expose errors and warnings separately."""
    frame = _ohlcv_frame(volumes=[10.0, -1.0, 10.0]).with_columns(pl.lit("x").alias("extra"))
    report = MarketDataValidator().validate(frame, _TIMEFRAME)
    assert report.has_errors() is True
    assert report.has_warnings() is True
    assert all(issue.severity is ValidationSeverity.ERROR for issue in report.errors())
    assert all(issue.severity is ValidationSeverity.WARNING for issue in report.warnings())
