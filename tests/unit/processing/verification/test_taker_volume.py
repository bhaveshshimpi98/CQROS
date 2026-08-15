"""Unit tests for CQROS processing ``TakerVolumeVerifier``."""

from __future__ import annotations

import math

import polars as pl
import pytest

from cqros.processing.verification import TakerVolumeVerifier, VerificationReport
from cqros.processing.verification.exceptions import (
    ERROR_REQUIRED_COLUMNS,
    ProcessingValidationError,
)
from cqros.processing.verification.interfaces import DataVerifier
from cqros.processing.verification.taker_volume import (
    TakerVolumeVerifier as TakerVolumeVerifierFromModule,
)

_START = 1_700_000_000_000
_INTERVAL = 60_000


def _taker_frame(
    *,
    timestamps: list[int] | None = None,
    buy_volumes: list[float | None] | None = None,
    sell_volumes: list[float | None] | None = None,
    buy_sell_ratios: list[float | None] | None = None,
) -> pl.DataFrame:
    """Build a canonical taker-volume verification frame."""
    if timestamps is None:
        timestamps = [_START, _START + _INTERVAL, _START + 2 * _INTERVAL]
    row_count = len(timestamps)
    if buy_volumes is None:
        buy_volumes = [10.0] * row_count
    if sell_volumes is None:
        sell_volumes = [5.0] * row_count
    if buy_sell_ratios is None:
        buy_sell_ratios = []
        for buy, sell in zip(buy_volumes, sell_volumes, strict=True):
            if buy is None or sell is None or sell == 0.0:
                buy_sell_ratios.append(None)
            else:
                buy_sell_ratios.append(buy / sell)
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "buy_volume": buy_volumes,
            "sell_volume": sell_volumes,
            "buy_sell_ratio": buy_sell_ratios,
        },
        schema={
            "timestamp": pl.Int64,
            "buy_volume": pl.Float64,
            "sell_volume": pl.Float64,
            "buy_sell_ratio": pl.Float64,
        },
    )


def _verifier() -> TakerVolumeVerifier:
    """Build a TakerVolumeVerifier instance."""
    return TakerVolumeVerifier()


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


def test_package_exports_taker_volume_verifier() -> None:
    """Package re-export matches the taker_volume module symbol."""
    assert TakerVolumeVerifier is TakerVolumeVerifierFromModule


def test_taker_volume_verifier_satisfies_data_verifier_protocol() -> None:
    """TakerVolumeVerifier structurally satisfies DataVerifier."""
    assert isinstance(_verifier(), DataVerifier)


def test_valid_frame_passes() -> None:
    """A clean sorted taker-volume frame passes verification."""
    report = _verifier().verify(_taker_frame())
    _assert_clean_pass(report, rows=3)


def test_duplicate_timestamps() -> None:
    """Duplicate timestamps are counted and fail verification."""
    frame = _taker_frame(timestamps=[_START, _START, _START + _INTERVAL])
    report = _verifier().verify(frame)
    assert report.duplicate_timestamp_rows == 1
    assert report.passed is False
    assert "Duplicate timestamps detected." in report.warnings


def test_null_values() -> None:
    """NULL buy_volume values are counted."""
    frame = _taker_frame(buy_volumes=[10.0, None, 10.0])
    report = _verifier().verify(frame)
    assert report.null_rows == 1
    assert report.passed is False
    assert "Rows containing NULL values." in report.warnings


def test_nan_values() -> None:
    """NaN sell_volume values are counted."""
    frame = _taker_frame(sell_volumes=[5.0, math.nan, 5.0])
    report = _verifier().verify(frame)
    assert report.nan_rows == 1
    assert report.passed is False
    assert "Rows containing NaN values." in report.warnings


def test_negative_buy_volume() -> None:
    """Negative buy_volume fails numeric verification."""
    frame = _taker_frame(buy_volumes=[10.0, -1.0, 10.0], sell_volumes=[5.0, 5.0, 5.0])
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 1
    assert report.passed is False
    assert "Invalid taker volume numeric values." in report.warnings


def test_negative_sell_volume() -> None:
    """Negative sell_volume fails numeric verification."""
    frame = _taker_frame(buy_volumes=[10.0, 10.0, 10.0], sell_volumes=[5.0, -1.0, 5.0])
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 1
    assert report.passed is False


def test_correct_ratio() -> None:
    """Consistent buy_sell_ratio values pass verification."""
    frame = _taker_frame(
        buy_volumes=[10.0, 3.0, 7.0],
        sell_volumes=[5.0, 3.0, 2.0],
        buy_sell_ratios=[2.0, 1.0, 3.5],
    )
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=3)


def test_incorrect_ratio() -> None:
    """Inconsistent buy_sell_ratio fails numeric verification."""
    frame = _taker_frame(
        buy_volumes=[10.0, 10.0],
        sell_volumes=[5.0, 5.0],
        buy_sell_ratios=[2.0, 3.0],
        timestamps=[_START, _START + _INTERVAL],
    )
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 1
    assert report.passed is False


def test_sell_volume_zero_with_null_ratio() -> None:
    """sell_volume == 0 with NULL buy_sell_ratio is valid."""
    frame = _taker_frame(
        buy_volumes=[10.0, 4.0],
        sell_volumes=[5.0, 0.0],
        buy_sell_ratios=[2.0, None],
        timestamps=[_START, _START + _INTERVAL],
    )
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=2)


def test_sell_volume_zero_with_non_null_ratio() -> None:
    """sell_volume == 0 with a non-NULL ratio is invalid."""
    frame = _taker_frame(
        buy_volumes=[10.0, 4.0],
        sell_volumes=[5.0, 0.0],
        buy_sell_ratios=[2.0, 1.0],
        timestamps=[_START, _START + _INTERVAL],
    )
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 1
    assert report.passed is False


def test_invalid_timestamps() -> None:
    """Non-positive timestamps are counted as invalid."""
    frame = _taker_frame(timestamps=[_START, 0, _START + 2 * _INTERVAL])
    report = _verifier().verify(frame)
    assert report.invalid_timestamp_rows == 1
    assert report.passed is False
    assert "Invalid timestamps detected." in report.warnings


def test_unsorted_frame() -> None:
    """Unsorted timestamps fail without incrementing counters."""
    frame = _taker_frame(timestamps=[_START, _START + 2 * _INTERVAL, _START + _INTERVAL])
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
    frame = pl.DataFrame({"timestamp": [1], "buy_volume": [1.0]})
    with pytest.raises(ProcessingValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_REQUIRED_COLUMNS
    missing = dict(exc_info.value.details)["missing_columns"]
    assert "sell_volume" in missing
    assert "buy_sell_ratio" in missing


def test_empty_frame_passes() -> None:
    """An empty schema-valid frame passes verification."""
    frame = pl.DataFrame(
        schema={
            "timestamp": pl.Int64,
            "buy_volume": pl.Float64,
            "sell_volume": pl.Float64,
            "buy_sell_ratio": pl.Float64,
        }
    )
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=0)


def test_immutability() -> None:
    """verify does not mutate the input DataFrame."""
    frame = _taker_frame(
        timestamps=[_START, _START, 0],
        buy_volumes=[10.0, None, -1.0],
        sell_volumes=[5.0, 5.0, 5.0],
        buy_sell_ratios=[2.0, 1.0, 3.0],
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
            "buy_volume": [10.0, None, -1.0],
            "sell_volume": [5.0, 5.0, 5.0],
            "buy_sell_ratio": [2.0, 1.0, 3.0],
        },
        schema={
            "timestamp": pl.Int64,
            "buy_volume": pl.Float64,
            "sell_volume": pl.Float64,
            "buy_sell_ratio": pl.Float64,
        },
    )
    report = _verifier().verify(frame)
    assert report.rows_checked == 3
    assert report.duplicate_timestamp_rows == 1
    assert report.null_rows == 1
    assert report.invalid_numeric_rows == 1
    assert report.passed is False
    assert "Duplicate timestamps detected." in report.warnings
    assert "Rows containing NULL values." in report.warnings
    assert "Invalid taker volume numeric values." in report.warnings
    assert "Frame is not sorted by timestamp." in report.warnings
    assert isinstance(report.warnings, tuple)


def test_nan_ratio_with_nonzero_sell_is_invalid() -> None:
    """NaN buy_sell_ratio with non-zero sell_volume is invalid."""
    frame = _taker_frame(
        buy_volumes=[10.0],
        sell_volumes=[5.0],
        buy_sell_ratios=[math.nan],
        timestamps=[_START],
    )
    report = _verifier().verify(frame)
    assert report.nan_rows == 1
    assert report.invalid_numeric_rows == 1
    assert report.passed is False


def test_numeric_helpers_handle_edge_values() -> None:
    """Private numeric helpers reject non-finite and non-numeric inputs."""
    from cqros.processing.verification.taker_volume import (
        _as_finite_float,
        _is_invalid_numeric_row,
    )

    assert _as_finite_float(None) is None
    assert _as_finite_float(True) is None
    assert _as_finite_float("x") is None
    assert _as_finite_float(math.nan) is None
    assert _as_finite_float(3) == 3.0
    assert _as_finite_float(1.5) == 1.5
    assert _is_invalid_numeric_row(None, 5.0, 1.0) is False
    assert _is_invalid_numeric_row(10.0, None, 1.0) is False
