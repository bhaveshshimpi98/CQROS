"""Unit tests for CQROS predictions ``PredictionVerifier``."""

from __future__ import annotations

import math
from typing import cast

import polars as pl
import pytest

from cqros.predictions import PredictionVerifier
from cqros.predictions.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    MERGED_PREDICTION_SCHEMA,
    REQUIRED_COLUMNS,
)
from cqros.predictions.verification import (
    ERROR_REQUIRED_COLUMNS,
    ERROR_SCHEMA_MISMATCH,
    PredictionValidationError,
    VerificationReport,
)
from cqros.predictions.verification import (
    PredictionVerifier as PredictionVerifierFromPackage,
)
from cqros.predictions.verification.verifier import (
    PredictionVerifier as PredictionVerifierFromModule,
)
from cqros.processing.verification.interfaces import DataVerifier

_START = 1_700_000_000_000
_INTERVAL = 3_600_000
_MODEL_NAME = "alpha-lgbm"
_MODEL_VERSION = "1.0.0"


def _prediction_frame(
    *,
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    open_times: list[int] | None = None,
    model_names: list[str | None] | None = None,
    model_versions: list[str | None] | None = None,
    predictions: list[float | None] | None = None,
    column_order: tuple[str, ...] | None = None,
) -> pl.DataFrame:
    """Build a canonical merged prediction verification frame."""
    if open_times is None:
        open_times = [_START, _START + _INTERVAL, _START + 2 * _INTERVAL]
    row_count = len(open_times)
    data: dict[str, object] = {
        "symbol": symbols if symbols is not None else ["BTCUSDT"] * row_count,
        "timeframe": timeframes if timeframes is not None else ["1h"] * row_count,
        "open_time": open_times,
        "model_name": (model_names if model_names is not None else [_MODEL_NAME] * row_count),
        "model_version": (
            model_versions if model_versions is not None else [_MODEL_VERSION] * row_count
        ),
        "prediction": (
            predictions
            if predictions is not None
            else [0.1 * (index + 1) for index in range(row_count)]
        ),
    }
    order = column_order if column_order is not None else CANONICAL_COLUMN_ORDER
    frame = pl.DataFrame(data, schema=dict(COLUMN_DTYPES))
    return frame.select(list(order))


def _verifier() -> PredictionVerifier:
    """Build a PredictionVerifier instance."""
    return PredictionVerifier()


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


def test_package_exports_prediction_verifier() -> None:
    """Package re-exports match the verification module symbol."""
    assert PredictionVerifier is PredictionVerifierFromModule
    assert PredictionVerifierFromPackage is PredictionVerifierFromModule


def test_prediction_verifier_satisfies_data_verifier_protocol() -> None:
    """PredictionVerifier structurally satisfies DataVerifier."""
    assert isinstance(_verifier(), DataVerifier)


def test_successful_verification() -> None:
    """A clean sorted merged prediction frame passes verification."""
    report = _verifier().verify(_prediction_frame())
    _assert_clean_pass(report, rows=3)


def test_empty_frame_passes() -> None:
    """An empty schema-valid frame passes verification."""
    frame = pl.DataFrame(schema=MERGED_PREDICTION_SCHEMA).select(list(CANONICAL_COLUMN_ORDER))
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=0)


def test_pass_report() -> None:
    """PASS report has zero defect counters and empty warnings."""
    report = _verifier().verify(_prediction_frame())
    assert report.passed is True
    assert report.warnings == ()
    assert report.duplicate_timestamp_rows == 0
    assert report.null_rows == 0
    assert report.nan_rows == 0
    assert report.invalid_timestamp_rows == 0
    assert report.invalid_numeric_rows == 0


def test_duplicate_primary_keys() -> None:
    """Duplicate (symbol, timeframe, open_time) keys fail verification."""
    frame = _prediction_frame(open_times=[_START, _START, _START + _INTERVAL])
    report = _verifier().verify(frame)
    assert report.duplicate_timestamp_rows == 1
    assert report.passed is False
    assert "Duplicate timestamps detected." in report.warnings


def test_duplicate_keys_allow_same_open_time_across_symbols() -> None:
    """Identical open_time values for distinct symbols do not count as duplicates."""
    frame = _prediction_frame(
        symbols=["BTCUSDT", "ETHUSDT", "BTCUSDT"],
        open_times=[_START, _START, _START + _INTERVAL],
    )
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=3)


def test_missing_columns() -> None:
    """Missing required columns raise PredictionValidationError."""
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timeframe": ["1h"],
            "open_time": [_START],
        }
    )
    with pytest.raises(PredictionValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_REQUIRED_COLUMNS
    missing = cast(tuple[str, ...], dict(exc_info.value.details)["missing_columns"])
    assert "model_name" in missing
    assert "model_version" in missing
    assert "prediction" in missing
    assert set(REQUIRED_COLUMNS) - set(frame.columns) == set(missing)


def test_null_values() -> None:
    """NULL values in required columns are counted and fail verification."""
    frame = _prediction_frame(predictions=[0.1, None, 0.05])
    report = _verifier().verify(frame)
    assert report.null_rows == 1
    assert report.passed is False
    assert "Rows containing NULL values." in report.warnings


def test_null_primary_key_rows() -> None:
    """NULL primary-key values are counted as null rows."""
    frame = _prediction_frame(symbols=["BTCUSDT", None, "BTCUSDT"])  # type: ignore[list-item]
    report = _verifier().verify(frame)
    assert report.null_rows == 1
    assert report.passed is False
    assert "Rows containing NULL values." in report.warnings


def test_null_metadata_rows() -> None:
    """NULL model metadata values are counted as null rows."""
    frame = _prediction_frame(model_names=[_MODEL_NAME, None, _MODEL_NAME])
    report = _verifier().verify(frame)
    assert report.null_rows == 1
    assert report.passed is False
    assert "Rows containing NULL values." in report.warnings


def test_nan_values() -> None:
    """NaN prediction values are counted and fail verification."""
    frame = _prediction_frame(predictions=[0.1, math.nan, 0.05])
    report = _verifier().verify(frame)
    assert report.nan_rows == 1
    assert report.invalid_numeric_rows == 0
    assert report.passed is False
    assert "Rows containing NaN values." in report.warnings


def test_infinite_values() -> None:
    """Infinite prediction values are counted as invalid numeric rows."""
    frame = _prediction_frame(predictions=[0.1, math.inf, -math.inf])
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 2
    assert report.nan_rows == 0
    assert report.passed is False
    assert "Infinite prediction values detected." in report.warnings


def test_invalid_timestamps() -> None:
    """Non-positive open_time values are counted as invalid."""
    frame = _prediction_frame(open_times=[_START, 0, _START + 2 * _INTERVAL])
    report = _verifier().verify(frame)
    assert report.invalid_timestamp_rows == 1
    assert report.passed is False
    assert "Invalid timestamps detected." in report.warnings


def test_negative_timestamps() -> None:
    """Negative open_time values are counted as invalid."""
    frame = _prediction_frame(open_times=[_START, -1, _START + 2 * _INTERVAL])
    report = _verifier().verify(frame)
    assert report.invalid_timestamp_rows == 1
    assert report.passed is False


def test_unsorted_timestamps() -> None:
    """Unsorted open_time fails without incrementing counters."""
    frame = _prediction_frame(
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


def test_incorrect_column_order() -> None:
    """Wrong column order fails verification with a deterministic warning."""
    reordered = (
        *CANONICAL_COLUMN_ORDER[1:],
        CANONICAL_COLUMN_ORDER[0],
    )
    frame = _prediction_frame(column_order=reordered)
    report = _verifier().verify(frame)
    assert report.passed is False
    assert report.warnings == ("Frame column order does not match canonical order.",)


def test_dtype_mismatch_open_time() -> None:
    """Wrong open_time dtype raises PredictionValidationError schema mismatch."""
    frame = _prediction_frame().with_columns(pl.col("open_time").cast(pl.Float64))
    with pytest.raises(PredictionValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH
    mismatched = cast(
        tuple[str, ...],
        dict(exc_info.value.details)["mismatched_columns"],
    )
    assert mismatched == ("open_time",)


def test_dtype_mismatch_prediction_column() -> None:
    """Non-Float64 prediction columns raise PredictionValidationError."""
    frame = _prediction_frame().with_columns(pl.col("prediction").cast(pl.Float32))
    with pytest.raises(PredictionValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH
    mismatched = cast(
        tuple[str, ...],
        dict(exc_info.value.details)["mismatched_columns"],
    )
    assert "prediction" in mismatched


def test_dtype_mismatch_model_name_column() -> None:
    """Non-String model_name columns raise PredictionValidationError."""
    frame = _prediction_frame().with_columns(pl.col("model_name").cast(pl.Categorical))
    with pytest.raises(PredictionValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH
    mismatched = cast(
        tuple[str, ...],
        dict(exc_info.value.details)["mismatched_columns"],
    )
    assert "model_name" in mismatched


def test_immutability() -> None:
    """verify does not mutate the input DataFrame."""
    frame = _prediction_frame(
        open_times=[_START + _INTERVAL, _START, _START],
        predictions=[0.1, None, math.nan],
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
    frame = pl.DataFrame(
        {
            "timeframe": ["1h", "1h", "1h"],
            "open_time": [_START + _INTERVAL, _START, _START],
            "model_name": [_MODEL_NAME, _MODEL_NAME, _MODEL_NAME],
            "model_version": [_MODEL_VERSION, _MODEL_VERSION, _MODEL_VERSION],
            "prediction": [math.inf, None, math.nan],
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
        },
        schema=dict(COLUMN_DTYPES),
    ).select(list(reordered))
    report = _verifier().verify(frame)
    assert report.rows_checked == 3
    assert report.duplicate_timestamp_rows == 1
    assert report.null_rows == 1
    assert report.nan_rows == 1
    assert report.invalid_numeric_rows == 1
    assert report.passed is False
    assert "Frame column order does not match canonical order." in report.warnings
    assert "Duplicate timestamps detected." in report.warnings
    assert "Rows containing NULL values." in report.warnings
    assert "Rows containing NaN values." in report.warnings
    assert "Infinite prediction values detected." in report.warnings
    assert "Frame is not sorted by open_time." in report.warnings
    assert isinstance(report.warnings, tuple)
