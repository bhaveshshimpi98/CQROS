"""Unit tests for CQROS processing ``OHLCVVerifier``."""

from __future__ import annotations

import math
from typing import cast

import polars as pl
import pytest

from cqros.processing.verification import OHLCVVerifier, VerificationReport
from cqros.processing.verification.exceptions import (
    ERROR_REQUIRED_COLUMNS,
    ProcessingValidationError,
)
from cqros.processing.verification.interfaces import DataVerifier
from cqros.processing.verification.ohlcv import OHLCVVerifier as OHLCVVerifierFromModule

_START = 1_700_000_000_000
_INTERVAL = 60_000

_CANONICAL_SCHEMA = {
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
}


def _ohlcv_frame(
    *,
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    open_times: list[int] | None = None,
    close_times: list[int] | None = None,
    opens: list[float] | None = None,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    closes: list[float] | None = None,
    volumes: list[float] | None = None,
    quote_volumes: list[float] | None = None,
    trade_counts: list[int] | None = None,
) -> pl.DataFrame:
    """Build a canonical processed OHLCV verification frame."""
    if open_times is None:
        open_times = [_START, _START + _INTERVAL, _START + 2 * _INTERVAL]
    row_count = len(open_times)
    if close_times is None:
        close_times = [value + _INTERVAL - 1 for value in open_times]
    return pl.DataFrame(
        {
            "symbol": symbols if symbols is not None else ["BTCUSDT"] * row_count,
            "timeframe": timeframes if timeframes is not None else ["1h"] * row_count,
            "open_time": open_times,
            "close_time": close_times,
            "open": opens if opens is not None else [100.0] * row_count,
            "high": highs if highs is not None else [101.0] * row_count,
            "low": lows if lows is not None else [99.0] * row_count,
            "close": closes if closes is not None else [100.5] * row_count,
            "volume": volumes if volumes is not None else [10.0] * row_count,
            "quote_volume": (quote_volumes if quote_volumes is not None else [1000.0] * row_count),
            "trade_count": trade_counts if trade_counts is not None else [42] * row_count,
        },
        schema=_CANONICAL_SCHEMA,
    )


def _verifier() -> OHLCVVerifier:
    """Build an OHLCVVerifier instance."""
    return OHLCVVerifier()


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


def test_package_exports_ohlcv_verifier() -> None:
    """Package re-export matches the ohlcv module symbol."""
    assert OHLCVVerifier is OHLCVVerifierFromModule


def test_ohlcv_verifier_satisfies_data_verifier_protocol() -> None:
    """OHLCVVerifier structurally satisfies DataVerifier."""
    assert isinstance(_verifier(), DataVerifier)


def test_canonical_schema_passes() -> None:
    """A frame with the canonical processed OHLCV schema passes."""
    frame = _ohlcv_frame()
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=3)


def test_valid_processed_ohlcv_passes() -> None:
    """Valid processed OHLCV with open_time and close_time passes."""
    frame = _ohlcv_frame(
        symbols=["BTCUSDT", "ETHUSDT", "BTCUSDT"],
        timeframes=["1h", "1h", "4h"],
        open_times=[_START, _START, _START + _INTERVAL],
        close_times=[
            _START + _INTERVAL - 1,
            _START + _INTERVAL - 1,
            _START + 2 * _INTERVAL - 1,
        ],
    )
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=3)


def test_valid_frame_passes() -> None:
    """A clean sorted OHLCV frame passes verification."""
    frame = _ohlcv_frame()
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=3)


def test_duplicate_symbol_timeframe_open_time() -> None:
    """Duplicate (symbol, timeframe, open_time) keys are counted."""
    frame = _ohlcv_frame(open_times=[_START, _START, _START + _INTERVAL])
    report = _verifier().verify(frame)
    assert report.duplicate_timestamp_rows == 1
    assert report.passed is False
    assert "Duplicate timestamps detected." in report.warnings


def test_same_open_time_different_symbol_not_duplicate() -> None:
    """Identical open_time values across symbols are not duplicates."""
    frame = _ohlcv_frame(
        symbols=["BTCUSDT", "ETHUSDT", "BTCUSDT"],
        open_times=[_START, _START, _START + _INTERVAL],
    )
    report = _verifier().verify(frame)
    assert report.duplicate_timestamp_rows == 0
    assert report.passed is True


def test_null_values() -> None:
    """NULL values in required columns are counted."""
    frame = _ohlcv_frame(opens=[100.0, None, 100.0])  # type: ignore[list-item]
    report = _verifier().verify(frame)
    assert report.null_rows == 1
    assert report.passed is False
    assert "Rows containing NULL values." in report.warnings


def test_nan_values() -> None:
    """NaN values in floating columns are counted."""
    frame = _ohlcv_frame(volumes=[10.0, math.nan, 10.0])
    report = _verifier().verify(frame)
    assert report.nan_rows == 1
    assert report.passed is False
    assert "Rows containing NaN values." in report.warnings


def test_invalid_open_time() -> None:
    """Non-positive open_time values are counted as invalid timestamps."""
    frame = _ohlcv_frame(open_times=[_START, 0, _START + 2 * _INTERVAL])
    report = _verifier().verify(frame)
    assert report.invalid_timestamp_rows == 1
    assert report.passed is False
    assert "Invalid timestamps detected." in report.warnings


def test_negative_open_time_fails() -> None:
    """Negative open_time fails timestamp verification."""
    frame = _ohlcv_frame(open_times=[_START, -1, _START + 2 * _INTERVAL])
    report = _verifier().verify(frame)
    assert report.invalid_timestamp_rows == 1
    assert report.passed is False
    assert "Invalid timestamps detected." in report.warnings


def test_negative_close_time_fails() -> None:
    """Negative close_time fails numeric verification."""
    frame = _ohlcv_frame(
        open_times=[_START, _START + _INTERVAL, _START + 2 * _INTERVAL],
        close_times=[_START + _INTERVAL - 1, -1, _START + 3 * _INTERVAL - 1],
    )
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 1
    assert report.passed is False
    assert "Invalid OHLCV numeric relationships." in report.warnings


def test_close_time_not_after_open_time_fails() -> None:
    """close_time <= open_time fails numeric verification."""
    frame = _ohlcv_frame(
        open_times=[_START, _START + _INTERVAL, _START + 2 * _INTERVAL],
        close_times=[
            _START + _INTERVAL - 1,
            _START + _INTERVAL,
            _START + 3 * _INTERVAL - 1,
        ],
    )
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 1
    assert report.passed is False
    assert "Invalid OHLCV numeric relationships." in report.warnings


def test_negative_volume() -> None:
    """Negative volume fails numeric verification."""
    frame = _ohlcv_frame(volumes=[10.0, -1.0, 10.0])
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 1
    assert "Invalid OHLCV numeric relationships." in report.warnings
    assert report.passed is False


def test_negative_quote_volume() -> None:
    """Negative quote_volume fails numeric verification."""
    frame = _ohlcv_frame(quote_volumes=[1000.0, -0.5, 1000.0])
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 1
    assert report.passed is False


def test_negative_trade_count() -> None:
    """Negative trade_count fails numeric verification."""
    frame = _ohlcv_frame(trade_counts=[42, -1, 42])
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 1
    assert report.passed is False


def test_zero_prices() -> None:
    """Zero OHLC prices are invalid."""
    frame = _ohlcv_frame(opens=[100.0, 0.0, 100.0])
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 1
    assert report.passed is False


@pytest.mark.parametrize(
    ("highs", "lows", "opens", "closes"),
    (
        ([101.0, 98.0, 101.0], [99.0, 99.0, 99.0], None, None),
        ([101.0, 99.5, 101.0], None, [100.0, 100.0, 100.0], None),
        ([101.0, 100.0, 101.0], None, None, [100.5, 100.5, 100.5]),
        (None, [99.0, 100.5, 99.0], [100.0, 100.0, 100.0], None),
        (None, [99.0, 100.6, 99.0], None, [100.5, 100.5, 100.5]),
    ),
)
def test_ohlc_relationship_failures(
    highs: list[float] | None,
    lows: list[float] | None,
    opens: list[float] | None,
    closes: list[float] | None,
) -> None:
    """Each OHLC relationship violation is detected."""
    frame = _ohlcv_frame(highs=highs, lows=lows, opens=opens, closes=closes)
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 1
    assert report.passed is False
    assert "Invalid OHLCV numeric relationships." in report.warnings


def test_multiple_failures_in_same_row_counted_once() -> None:
    """A single row violating multiple numeric rules counts once."""
    frame = _ohlcv_frame(
        opens=[100.0, 0.0],
        highs=[101.0, -1.0],
        lows=[99.0, 50.0],
        closes=[100.5, -2.0],
        volumes=[10.0, -5.0],
        quote_volumes=[1000.0, -1.0],
        trade_counts=[42, -3],
        open_times=[_START, _START + _INTERVAL],
        close_times=[_START + _INTERVAL - 1, _START + _INTERVAL],
    )
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 1
    assert report.rows_checked == 2
    assert report.passed is False


def test_unsorted_open_time_warning_only() -> None:
    """Unsorted open_time fails without incrementing other counters."""
    frame = _ohlcv_frame(open_times=[_START, _START + 2 * _INTERVAL, _START + _INTERVAL])
    report = _verifier().verify(frame)
    assert report.duplicate_timestamp_rows == 0
    assert report.null_rows == 0
    assert report.nan_rows == 0
    assert report.invalid_timestamp_rows == 0
    assert report.invalid_numeric_rows == 0
    assert report.passed is False
    assert report.warnings == ("Frame is not sorted by open_time.",)


def test_empty_frame_passes() -> None:
    """An empty schema-valid frame passes verification."""
    frame = pl.DataFrame(schema=_CANONICAL_SCHEMA)
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=0)


def test_missing_open_time_fails() -> None:
    """Missing open_time raises ProcessingValidationError."""
    frame = _ohlcv_frame().drop("open_time")
    with pytest.raises(ProcessingValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_REQUIRED_COLUMNS
    missing = cast(tuple[str, ...], dict(exc_info.value.details)["missing_columns"])
    assert "open_time" in missing


def test_missing_close_time_fails() -> None:
    """Missing close_time raises ProcessingValidationError."""
    frame = _ohlcv_frame().drop("close_time")
    with pytest.raises(ProcessingValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_REQUIRED_COLUMNS
    missing = cast(tuple[str, ...], dict(exc_info.value.details)["missing_columns"])
    assert "close_time" in missing


def test_missing_required_columns() -> None:
    """Missing required columns raise ProcessingValidationError."""
    frame = pl.DataFrame({"open_time": [1], "open": [1.0]})
    with pytest.raises(ProcessingValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_REQUIRED_COLUMNS
    missing = cast(tuple[str, ...], dict(exc_info.value.details)["missing_columns"])
    assert "symbol" in missing
    assert "close_time" in missing
    assert "high" in missing
    assert "volume" in missing


def test_immutability() -> None:
    """verify does not mutate the input DataFrame."""
    frame = _ohlcv_frame(
        open_times=[_START, _START, 0],
        volumes=[10.0, -1.0, math.nan],
        opens=[100.0, None, 100.0],  # type: ignore[list-item]
    )
    original = frame.clone()
    report = _verifier().verify(frame)
    assert frame.equals(original)
    assert report.passed is False


def test_verification_report_correctness_combined() -> None:
    """Combined failures populate every report field correctly."""
    frame = _ohlcv_frame(
        open_times=[_START + _INTERVAL, _START, _START],
        opens=[100.0, None, 0.0],  # type: ignore[list-item]
        volumes=[10.0, math.nan, -1.0],
    )
    report = _verifier().verify(frame)
    assert report.rows_checked == 3
    assert report.duplicate_timestamp_rows == 1
    assert report.null_rows == 1
    assert report.nan_rows == 1
    assert report.invalid_timestamp_rows == 0
    assert report.invalid_numeric_rows >= 1
    assert report.passed is False
    assert "Duplicate timestamps detected." in report.warnings
    assert "Rows containing NULL values." in report.warnings
    assert "Rows containing NaN values." in report.warnings
    assert "Invalid OHLCV numeric relationships." in report.warnings
    assert "Frame is not sorted by open_time." in report.warnings
    assert isinstance(report.warnings, tuple)
