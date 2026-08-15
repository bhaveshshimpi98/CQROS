"""Unit tests for CQROS training ``TrainingVerifier``."""

from __future__ import annotations

import math
from typing import cast

import polars as pl
import pytest

from cqros.processing.verification.interfaces import DataVerifier
from cqros.training import TrainingVerifier
from cqros.training.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    REQUIRED_COLUMNS,
)
from cqros.training.verification import (
    ERROR_REQUIRED_COLUMNS,
    ERROR_SCHEMA_MISMATCH,
    TrainingValidationError,
    VerificationReport,
)
from cqros.training.verification import (
    TrainingVerifier as TrainingVerifierFromPackage,
)
from cqros.training.verification.verifier import (
    TrainingVerifier as TrainingVerifierFromModule,
)

_START = 1_700_000_000_000
_INTERVAL = 3_600_000


def _feature_values(row_count: int, *, value: float = 0.01) -> dict[str, list[float]]:
    """Build default float values for every feature column."""
    return {column: [value] * row_count for column in FEATURE_COLUMNS}


def _label_values(row_count: int) -> dict[str, list[float] | list[int]]:
    """Build default values for every label column."""
    values: dict[str, list[float] | list[int]] = {}
    for column in LABEL_COLUMNS:
        if column.startswith("direction_"):
            values[column] = [1] * row_count
        else:
            values[column] = [0.01] * row_count
    return values


def _training_frame(
    *,
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    open_times: list[int] | None = None,
    feature_overrides: dict[str, list[float | None]] | None = None,
    label_overrides: dict[str, list[float | int | None]] | None = None,
    column_order: tuple[str, ...] | None = None,
) -> pl.DataFrame:
    """Build a canonical merged training verification frame."""
    if open_times is None:
        open_times = [_START, _START + _INTERVAL, _START + 2 * _INTERVAL]
    row_count = len(open_times)
    data: dict[str, object] = {
        "symbol": symbols if symbols is not None else ["BTCUSDT"] * row_count,
        "timeframe": timeframes if timeframes is not None else ["1h"] * row_count,
        "open_time": open_times,
    }
    data.update(_feature_values(row_count))
    data.update(_label_values(row_count))
    if feature_overrides is not None:
        data.update(feature_overrides)
    if label_overrides is not None:
        data.update(label_overrides)
    order = column_order if column_order is not None else CANONICAL_COLUMN_ORDER
    frame = pl.DataFrame(data, schema=COLUMN_DTYPES)
    return frame.select(list(order))


def _verifier() -> TrainingVerifier:
    """Build a TrainingVerifier instance."""
    return TrainingVerifier()


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


def test_package_exports_training_verifier() -> None:
    """Package re-exports match the verification module symbol."""
    assert TrainingVerifier is TrainingVerifierFromModule
    assert TrainingVerifierFromPackage is TrainingVerifierFromModule


def test_training_verifier_satisfies_data_verifier_protocol() -> None:
    """TrainingVerifier structurally satisfies DataVerifier."""
    assert isinstance(_verifier(), DataVerifier)


def test_successful_verification() -> None:
    """A clean sorted merged training frame passes verification."""
    report = _verifier().verify(_training_frame())
    _assert_clean_pass(report, rows=3)


def test_empty_frame_passes() -> None:
    """An empty schema-valid frame passes verification."""
    frame = pl.DataFrame(schema=COLUMN_DTYPES).select(list(CANONICAL_COLUMN_ORDER))
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=0)


def test_pass_report() -> None:
    """PASS report has zero defect counters and empty warnings."""
    report = _verifier().verify(_training_frame())
    assert report.passed is True
    assert report.warnings == ()
    assert report.duplicate_timestamp_rows == 0
    assert report.null_rows == 0
    assert report.nan_rows == 0
    assert report.invalid_timestamp_rows == 0
    assert report.invalid_numeric_rows == 0


def test_duplicate_primary_keys() -> None:
    """Duplicate (symbol, timeframe, open_time) keys fail verification."""
    frame = _training_frame(open_times=[_START, _START, _START + _INTERVAL])
    report = _verifier().verify(frame)
    assert report.duplicate_timestamp_rows == 1
    assert report.passed is False
    assert "Duplicate timestamps detected." in report.warnings


def test_duplicate_keys_allow_same_open_time_across_symbols() -> None:
    """Identical open_time values for distinct symbols do not count as duplicates."""
    frame = _training_frame(
        symbols=["BTCUSDT", "ETHUSDT", "BTCUSDT"],
        open_times=[_START, _START, _START + _INTERVAL],
    )
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=3)


def test_missing_columns() -> None:
    """Missing required columns raise TrainingValidationError."""
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timeframe": ["1h"],
            "open_time": [_START],
        }
    )
    with pytest.raises(TrainingValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_REQUIRED_COLUMNS
    missing = cast(tuple[str, ...], dict(exc_info.value.details)["missing_columns"])
    assert FEATURE_COLUMNS[0] in missing
    assert LABEL_COLUMNS[0] in missing
    assert set(REQUIRED_COLUMNS) - set(frame.columns) == set(missing)


def test_null_values() -> None:
    """NULL values in required columns are counted and fail verification."""
    frame = _training_frame(
        feature_overrides={"returns": [0.01, None, 0.01]},  # type: ignore[dict-item]
    )
    report = _verifier().verify(frame)
    assert report.null_rows == 1
    assert report.passed is False
    assert "Rows containing NULL values." in report.warnings


def test_null_primary_key_rows() -> None:
    """NULL primary-key values are counted as null rows."""
    frame = _training_frame(symbols=["BTCUSDT", None, "BTCUSDT"])  # type: ignore[list-item]
    report = _verifier().verify(frame)
    assert report.null_rows == 1
    assert report.passed is False
    assert "Rows containing NULL values." in report.warnings


def test_nan_values() -> None:
    """NaN feature values are counted and fail verification."""
    frame = _training_frame(feature_overrides={"rolling_mean": [0.01, math.nan, 0.01]})
    report = _verifier().verify(frame)
    assert report.nan_rows == 1
    assert report.invalid_numeric_rows == 0
    assert report.passed is False
    assert "Rows containing NaN values." in report.warnings


def test_nan_label_values() -> None:
    """NaN regression label values are counted and fail verification."""
    frame = _training_frame(label_overrides={"future_return_5": [0.01, math.nan, 0.01]})
    report = _verifier().verify(frame)
    assert report.nan_rows == 1
    assert report.passed is False
    assert "Rows containing NaN values." in report.warnings


def test_infinite_values() -> None:
    """Infinite feature values are counted as invalid numeric rows."""
    frame = _training_frame(
        feature_overrides={"atr": [0.01, math.inf, -math.inf]},
    )
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 2
    assert report.nan_rows == 0
    assert report.passed is False
    assert "Infinite training values detected." in report.warnings


def test_infinite_label_values() -> None:
    """Infinite regression label values are counted as invalid numeric rows."""
    frame = _training_frame(
        label_overrides={"future_return_10": [0.01, math.inf, -math.inf]},
    )
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 2
    assert report.passed is False
    assert "Infinite training values detected." in report.warnings


def test_invalid_timestamps() -> None:
    """Non-positive open_time values are counted as invalid."""
    frame = _training_frame(open_times=[_START, 0, _START + 2 * _INTERVAL])
    report = _verifier().verify(frame)
    assert report.invalid_timestamp_rows == 1
    assert report.passed is False
    assert "Invalid timestamps detected." in report.warnings


def test_negative_timestamps() -> None:
    """Negative open_time values are counted as invalid."""
    frame = _training_frame(open_times=[_START, -1, _START + 2 * _INTERVAL])
    report = _verifier().verify(frame)
    assert report.invalid_timestamp_rows == 1
    assert report.passed is False


def test_unsorted_timestamps() -> None:
    """Unsorted open_time fails without incrementing counters."""
    frame = _training_frame(
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
    frame = _training_frame(column_order=reordered)
    report = _verifier().verify(frame)
    assert report.passed is False
    assert report.warnings == ("Frame column order does not match canonical order.",)


def test_dtype_mismatch_open_time() -> None:
    """Wrong open_time dtype raises TrainingValidationError schema mismatch."""
    frame = _training_frame().with_columns(pl.col("open_time").cast(pl.Float64))
    with pytest.raises(TrainingValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH
    mismatched = cast(
        tuple[str, ...],
        dict(exc_info.value.details)["mismatched_columns"],
    )
    assert mismatched == ("open_time",)


def test_dtype_mismatch_feature_column() -> None:
    """Non-Float64 feature columns raise TrainingValidationError."""
    frame = _training_frame().with_columns(pl.col("returns").cast(pl.Float32))
    with pytest.raises(TrainingValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH
    mismatched = cast(
        tuple[str, ...],
        dict(exc_info.value.details)["mismatched_columns"],
    )
    assert "returns" in mismatched


def test_dtype_mismatch_direction_label() -> None:
    """Non-Int8 direction labels raise TrainingValidationError."""
    frame = _training_frame().with_columns(pl.col("direction_1").cast(pl.Int64))
    with pytest.raises(TrainingValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH
    mismatched = cast(
        tuple[str, ...],
        dict(exc_info.value.details)["mismatched_columns"],
    )
    assert "direction_1" in mismatched


def test_immutability() -> None:
    """verify does not mutate the input DataFrame."""
    frame = _training_frame(
        open_times=[_START + _INTERVAL, _START, _START],
        feature_overrides={
            "returns": [0.01, None, math.nan],  # type: ignore[dict-item]
            "atr": [0.01, math.inf, 0.01],
        },
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
    frame = _training_frame(
        open_times=[_START + _INTERVAL, _START, _START],
        feature_overrides={
            "returns": [0.01, None, math.nan],  # type: ignore[dict-item]
            "atr": [0.01, math.inf, 0.01],
        },
        column_order=reordered,
    )
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
    assert "Infinite training values detected." in report.warnings
    assert "Frame is not sorted by open_time." in report.warnings
    assert isinstance(report.warnings, tuple)
