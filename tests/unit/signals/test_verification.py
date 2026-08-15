"""Unit tests for CQROS signals ``SignalVerifier``."""

from __future__ import annotations

from typing import cast

import polars as pl
import pytest

from cqros.processing.verification.interfaces import DataVerifier
from cqros.signals import Signal, SignalVerifier
from cqros.signals.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    MERGED_SIGNAL_SCHEMA,
    REQUIRED_COLUMNS,
)
from cqros.signals.verification import (
    ERROR_REQUIRED_COLUMNS,
    ERROR_SCHEMA_MISMATCH,
    SignalValidationError,
    VerificationReport,
)
from cqros.signals.verification import (
    SignalVerifier as SignalVerifierFromPackage,
)
from cqros.signals.verification.verifier import (
    SignalVerifier as SignalVerifierFromModule,
)

_START = 1_700_000_000_000
_INTERVAL = 3_600_000
_MODEL_NAME = "alpha-lgbm"
_MODEL_VERSION = "1.0.0"


def _signal_frame(
    *,
    symbols: list[str | None] | None = None,
    timeframes: list[str | None] | None = None,
    open_times: list[int] | None = None,
    model_names: list[str | None] | None = None,
    model_versions: list[str | None] | None = None,
    signals: list[str | None] | None = None,
    column_order: tuple[str, ...] | None = None,
) -> pl.DataFrame:
    """Build a canonical merged signal verification frame."""
    if open_times is None:
        open_times = [_START, _START + _INTERVAL, _START + 2 * _INTERVAL]
    row_count = len(open_times)
    default_signals = [
        Signal.BUY.value,
        Signal.SELL.value,
        Signal.HOLD.value,
    ]
    while len(default_signals) < row_count:
        default_signals.append(Signal.HOLD.value)
    data: dict[str, object] = {
        "symbol": symbols if symbols is not None else ["BTCUSDT"] * row_count,
        "timeframe": timeframes if timeframes is not None else ["1h"] * row_count,
        "open_time": open_times,
        "model_name": (model_names if model_names is not None else [_MODEL_NAME] * row_count),
        "model_version": (
            model_versions if model_versions is not None else [_MODEL_VERSION] * row_count
        ),
        "signal": signals if signals is not None else default_signals[:row_count],
    }
    order = column_order if column_order is not None else CANONICAL_COLUMN_ORDER
    frame = pl.DataFrame(data, schema=dict(COLUMN_DTYPES))
    return frame.select(list(order))


def _verifier() -> SignalVerifier:
    """Build a SignalVerifier instance."""
    return SignalVerifier()


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


def test_package_exports_signal_verifier() -> None:
    """Package re-exports match the verification module symbol."""
    assert SignalVerifier is SignalVerifierFromModule
    assert SignalVerifierFromPackage is SignalVerifierFromModule


def test_signal_verifier_satisfies_data_verifier_protocol() -> None:
    """SignalVerifier structurally satisfies DataVerifier."""
    assert isinstance(_verifier(), DataVerifier)


def test_successful_verification() -> None:
    """A clean sorted merged signal frame passes verification."""
    report = _verifier().verify(_signal_frame())
    _assert_clean_pass(report, rows=3)


def test_empty_frame_passes() -> None:
    """An empty schema-valid frame passes verification."""
    frame = pl.DataFrame(schema=MERGED_SIGNAL_SCHEMA).select(list(CANONICAL_COLUMN_ORDER))
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=0)


def test_canonical_schema_columns() -> None:
    """Successful verification inspects the canonical Signal column set."""
    frame = _signal_frame()
    assert frame.columns == list(CANONICAL_COLUMN_ORDER)
    assert frame.columns == [
        "symbol",
        "timeframe",
        "open_time",
        "model_name",
        "model_version",
        "signal",
    ]
    assert "prediction" not in frame.columns
    assert "confidence" not in frame.columns
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=3)


def test_pass_report() -> None:
    """PASS report has zero defect counters and empty warnings."""
    report = _verifier().verify(_signal_frame())
    assert report.passed is True
    assert report.warnings == ()
    assert report.duplicate_timestamp_rows == 0
    assert report.null_rows == 0
    assert report.nan_rows == 0
    assert report.invalid_timestamp_rows == 0
    assert report.invalid_numeric_rows == 0


def test_duplicate_primary_keys() -> None:
    """Duplicate (symbol, timeframe, open_time) keys fail verification."""
    frame = _signal_frame(open_times=[_START, _START, _START + _INTERVAL])
    report = _verifier().verify(frame)
    assert report.duplicate_timestamp_rows == 1
    assert report.passed is False
    assert "Duplicate timestamps detected." in report.warnings


def test_duplicate_keys_allow_same_open_time_across_symbols() -> None:
    """Identical open_time values for distinct symbols do not count as duplicates."""
    frame = _signal_frame(
        symbols=["BTCUSDT", "ETHUSDT", "BTCUSDT"],
        open_times=[_START, _START, _START + _INTERVAL],
    )
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=3)


def test_missing_columns() -> None:
    """Missing required columns raise SignalValidationError."""
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timeframe": ["1h"],
            "open_time": [_START],
        }
    )
    with pytest.raises(SignalValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_REQUIRED_COLUMNS
    missing = cast(tuple[str, ...], dict(exc_info.value.details)["missing_columns"])
    assert "model_name" in missing
    assert "model_version" in missing
    assert "signal" in missing
    assert set(REQUIRED_COLUMNS) - set(frame.columns) == set(missing)


def test_null_values() -> None:
    """NULL values in required columns are counted and fail verification."""
    frame = _signal_frame(signals=[Signal.BUY.value, None, Signal.HOLD.value])
    report = _verifier().verify(frame)
    assert report.null_rows == 1
    assert report.passed is False
    assert "Rows containing NULL values." in report.warnings


def test_null_primary_key_rows() -> None:
    """NULL primary-key values are counted as null rows."""
    frame = _signal_frame(symbols=["BTCUSDT", None, "BTCUSDT"])
    report = _verifier().verify(frame)
    assert report.null_rows == 1
    assert report.passed is False
    assert "Rows containing NULL values." in report.warnings


def test_null_metadata_rows() -> None:
    """NULL model metadata values are counted as null rows."""
    frame = _signal_frame(model_names=[_MODEL_NAME, None, _MODEL_NAME])
    report = _verifier().verify(frame)
    assert report.null_rows == 1
    assert report.passed is False
    assert "Rows containing NULL values." in report.warnings


def test_nan_values_not_applicable_for_string_schema() -> None:
    """Canonical signal frames have no floating columns, so nan_rows stays 0."""
    report = _verifier().verify(_signal_frame())
    assert report.nan_rows == 0
    assert report.passed is True


def test_invalid_timestamps() -> None:
    """Non-positive open_time values are counted as invalid."""
    frame = _signal_frame(open_times=[_START, 0, _START + 2 * _INTERVAL])
    report = _verifier().verify(frame)
    assert report.invalid_timestamp_rows == 1
    assert report.passed is False
    assert "Invalid timestamps detected." in report.warnings


def test_negative_timestamps() -> None:
    """Negative open_time values are counted as invalid."""
    frame = _signal_frame(open_times=[_START, -1, _START + 2 * _INTERVAL])
    report = _verifier().verify(frame)
    assert report.invalid_timestamp_rows == 1
    assert report.passed is False


def test_unsorted_timestamps() -> None:
    """Unsorted open_time fails without incrementing counters."""
    frame = _signal_frame(
        open_times=[_START, _START + 2 * _INTERVAL, _START + _INTERVAL],
    )
    report = _verifier().verify(frame)
    assert report.duplicate_timestamp_rows == 0
    assert report.null_rows == 0
    assert report.nan_rows == 0
    assert report.invalid_timestamp_rows == 0
    assert report.invalid_numeric_rows == 0
    assert report.passed is False
    assert report.warnings == ("Frame is not sorted by open_time.",)


def test_invalid_signal_values() -> None:
    """Signal values outside BUY/SELL/HOLD fail verification."""
    frame = _signal_frame(
        signals=[Signal.BUY.value, "LONG", Signal.HOLD.value],
    )
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 1
    assert report.passed is False
    assert "Invalid signal values detected." in report.warnings


def test_invalid_signal_values_use_signal_enum() -> None:
    """Only Signal enum values are accepted in the signal column."""
    frame = _signal_frame(
        signals=[Signal.BUY.value, Signal.SELL.value, Signal.HOLD.value],
    )
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=3)

    frame = _signal_frame(signals=["buy", "sell", "hold"])
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 3
    assert report.passed is False
    assert "Invalid signal values detected." in report.warnings


def test_invalid_string_values() -> None:
    """Empty string values in string columns fail verification."""
    frame = _signal_frame(model_names=[_MODEL_NAME, "", _MODEL_NAME])
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 1
    assert report.passed is False
    assert "Invalid string values detected." in report.warnings


def test_incorrect_column_order() -> None:
    """Wrong column order fails verification with a deterministic warning."""
    reordered = (
        *CANONICAL_COLUMN_ORDER[1:],
        CANONICAL_COLUMN_ORDER[0],
    )
    frame = _signal_frame(column_order=reordered)
    report = _verifier().verify(frame)
    assert report.passed is False
    assert report.warnings == ("Frame column order does not match canonical order.",)


def test_dtype_mismatch_open_time() -> None:
    """Wrong open_time dtype raises SignalValidationError schema mismatch."""
    frame = _signal_frame().with_columns(pl.col("open_time").cast(pl.Float64))
    with pytest.raises(SignalValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH
    mismatched = cast(
        tuple[str, ...],
        dict(exc_info.value.details)["mismatched_columns"],
    )
    assert mismatched == ("open_time",)


def test_dtype_mismatch_signal_column() -> None:
    """Non-String signal columns raise SignalValidationError."""
    frame = _signal_frame().with_columns(pl.col("signal").cast(pl.Categorical))
    with pytest.raises(SignalValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH
    mismatched = cast(
        tuple[str, ...],
        dict(exc_info.value.details)["mismatched_columns"],
    )
    assert "signal" in mismatched


def test_dtype_mismatch_model_name_column() -> None:
    """Non-String model_name columns raise SignalValidationError."""
    frame = _signal_frame().with_columns(pl.col("model_name").cast(pl.Categorical))
    with pytest.raises(SignalValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH
    mismatched = cast(
        tuple[str, ...],
        dict(exc_info.value.details)["mismatched_columns"],
    )
    assert "model_name" in mismatched


def test_immutability() -> None:
    """verify does not mutate the input DataFrame."""
    frame = _signal_frame(
        open_times=[_START + _INTERVAL, _START, _START],
        signals=[Signal.BUY.value, None, "LONG"],
    )
    original = frame.clone()
    report = _verifier().verify(frame)
    assert frame.equals(original)
    assert report.passed is False


def test_report_values_combined() -> None:
    """Combined failures populate report fields correctly."""
    reordered = (
        *CANONICAL_COLUMN_ORDER[1:],
        CANONICAL_COLUMN_ORDER[0],
    )
    frame = _signal_frame(
        open_times=[_START + _INTERVAL, _START, _START],
        signals=[Signal.BUY.value, None, "LONG"],
        model_names=[_MODEL_NAME, "", _MODEL_NAME],
        column_order=reordered,
    )
    report = _verifier().verify(frame)
    assert report.rows_checked == 3
    assert report.duplicate_timestamp_rows == 1
    assert report.null_rows == 1
    assert report.invalid_numeric_rows >= 1
    assert report.passed is False
    assert "Frame column order does not match canonical order." in report.warnings
    assert "Duplicate timestamps detected." in report.warnings
    assert "Rows containing NULL values." in report.warnings
    assert "Invalid signal values detected." in report.warnings
    assert "Invalid string values detected." in report.warnings
    assert "Frame is not sorted by open_time." in report.warnings
    assert isinstance(report.warnings, tuple)
