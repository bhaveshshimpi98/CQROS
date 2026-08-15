"""Unit tests for CQROS labels ``LabelVerifier``."""

from __future__ import annotations

import math
from typing import cast

import polars as pl
import pytest

from cqros.labels import LabelVerifier
from cqros.labels.schema import (
    CANONICAL_COLUMN_ORDER,
    CLASSIFICATION_LABEL_COLUMNS,
    COLUMN_DTYPES,
    REGRESSION_LABEL_COLUMNS,
    REQUIRED_COLUMNS,
)
from cqros.labels.verification import (
    ERROR_REQUIRED_COLUMNS,
    ERROR_SCHEMA_MISMATCH,
    LabelValidationError,
    VerificationReport,
)
from cqros.labels.verification import (
    LabelVerifier as LabelVerifierFromPackage,
)
from cqros.labels.verification.verifier import LabelVerifier as LabelVerifierFromModule
from cqros.processing.verification.interfaces import DataVerifier

_START = 1_700_000_000_000
_INTERVAL = 3_600_000


def _regression_values(row_count: int, *, value: float = 0.01) -> dict[str, list[float]]:
    """Build default float values for every regression label column."""
    return {column: [value] * row_count for column in REGRESSION_LABEL_COLUMNS}


def _direction_values(row_count: int, *, value: int = 1) -> dict[str, list[int]]:
    """Build default Int8-compatible values for every direction label column."""
    return {column: [value] * row_count for column in CLASSIFICATION_LABEL_COLUMNS}


def _label_frame(
    *,
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    open_times: list[int] | None = None,
    regression_overrides: dict[str, list[float | None]] | None = None,
    direction_overrides: dict[str, list[int | None]] | None = None,
    column_order: tuple[str, ...] | None = None,
) -> pl.DataFrame:
    """Build a canonical merged label verification frame."""
    if open_times is None:
        open_times = [_START, _START + _INTERVAL, _START + 2 * _INTERVAL]
    row_count = len(open_times)
    data: dict[str, object] = {
        "symbol": symbols if symbols is not None else ["BTCUSDT"] * row_count,
        "timeframe": timeframes if timeframes is not None else ["1h"] * row_count,
        "open_time": open_times,
    }
    data.update(_regression_values(row_count))
    data.update(_direction_values(row_count))
    if regression_overrides is not None:
        data.update(regression_overrides)
    if direction_overrides is not None:
        data.update(direction_overrides)
    order = column_order if column_order is not None else CANONICAL_COLUMN_ORDER
    frame = pl.DataFrame(data, schema=COLUMN_DTYPES)
    return frame.select(list(order))


def _verifier() -> LabelVerifier:
    """Build a LabelVerifier instance."""
    return LabelVerifier()


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


def test_package_exports_label_verifier() -> None:
    """Package re-exports match the verification module symbol."""
    assert LabelVerifier is LabelVerifierFromModule
    assert LabelVerifierFromPackage is LabelVerifierFromModule


def test_label_verifier_satisfies_data_verifier_protocol() -> None:
    """LabelVerifier structurally satisfies DataVerifier."""
    assert isinstance(_verifier(), DataVerifier)


def test_successful_verification() -> None:
    """A clean sorted merged label frame passes verification."""
    report = _verifier().verify(_label_frame())
    _assert_clean_pass(report, rows=3)


def test_empty_frame_passes() -> None:
    """An empty schema-valid frame passes verification."""
    frame = pl.DataFrame(schema=COLUMN_DTYPES).select(list(CANONICAL_COLUMN_ORDER))
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=0)


def test_duplicate_primary_keys() -> None:
    """Duplicate (symbol, timeframe, open_time) keys fail verification."""
    frame = _label_frame(open_times=[_START, _START, _START + _INTERVAL])
    report = _verifier().verify(frame)
    assert report.duplicate_timestamp_rows == 1
    assert report.passed is False
    assert "Duplicate timestamps detected." in report.warnings


def test_duplicate_keys_allow_same_open_time_across_symbols() -> None:
    """Identical open_time values for distinct symbols do not count as duplicates."""
    frame = _label_frame(
        symbols=["BTCUSDT", "ETHUSDT", "BTCUSDT"],
        open_times=[_START, _START, _START + _INTERVAL],
    )
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=3)


def test_missing_columns() -> None:
    """Missing required columns raise LabelValidationError."""
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timeframe": ["1h"],
            "open_time": [_START],
        }
    )
    with pytest.raises(LabelValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_REQUIRED_COLUMNS
    missing = cast(tuple[str, ...], dict(exc_info.value.details)["missing_columns"])
    assert "future_return_1" in missing
    assert set(REQUIRED_COLUMNS) - set(frame.columns) == set(missing)


def test_null_values() -> None:
    """NULL values in required columns are counted and fail verification."""
    frame = _label_frame(
        regression_overrides={"future_return_1": [0.01, None, 0.01]},  # type: ignore[dict-item]
    )
    report = _verifier().verify(frame)
    assert report.null_rows == 1
    assert report.passed is False
    assert "Rows containing NULL values." in report.warnings


def test_null_primary_key_rows() -> None:
    """NULL primary-key values are counted as null rows."""
    frame = _label_frame(symbols=["BTCUSDT", None, "BTCUSDT"])  # type: ignore[list-item]
    report = _verifier().verify(frame)
    assert report.null_rows == 1
    assert report.passed is False
    assert "Rows containing NULL values." in report.warnings


def test_nan_values() -> None:
    """NaN regression label values are counted and fail verification."""
    frame = _label_frame(regression_overrides={"future_return_5": [0.01, math.nan, 0.01]})
    report = _verifier().verify(frame)
    assert report.nan_rows == 1
    assert report.invalid_numeric_rows == 0
    assert report.passed is False
    assert "Rows containing NaN values." in report.warnings


def test_infinite_values() -> None:
    """Infinite regression label values are counted as invalid numeric rows."""
    frame = _label_frame(
        regression_overrides={"future_return_10": [0.01, math.inf, -math.inf]},
    )
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 2
    assert report.nan_rows == 0
    assert report.passed is False
    assert "Infinite label values detected." in report.warnings


def test_invalid_timestamps() -> None:
    """Non-positive open_time values are counted as invalid."""
    frame = _label_frame(open_times=[_START, 0, _START + 2 * _INTERVAL])
    report = _verifier().verify(frame)
    assert report.invalid_timestamp_rows == 1
    assert report.passed is False
    assert "Invalid timestamps detected." in report.warnings


def test_negative_timestamps() -> None:
    """Negative open_time values are counted as invalid."""
    frame = _label_frame(open_times=[_START, -1, _START + 2 * _INTERVAL])
    report = _verifier().verify(frame)
    assert report.invalid_timestamp_rows == 1
    assert report.passed is False


def test_unsorted_timestamps() -> None:
    """Unsorted open_time fails without incrementing counters."""
    frame = _label_frame(
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
    frame = _label_frame(column_order=reordered)
    report = _verifier().verify(frame)
    assert report.passed is False
    assert report.warnings == ("Frame column order does not match canonical order.",)


def test_dtype_mismatch_open_time() -> None:
    """Wrong open_time dtype raises LabelValidationError schema mismatch."""
    frame = _label_frame().with_columns(pl.col("open_time").cast(pl.Float64))
    with pytest.raises(LabelValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH
    mismatched = cast(
        tuple[str, ...],
        dict(exc_info.value.details)["mismatched_columns"],
    )
    assert mismatched == ("open_time",)


def test_dtype_mismatch_regression_label() -> None:
    """Non-Float64 regression labels raise LabelValidationError."""
    frame = _label_frame().with_columns(pl.col("future_return_1").cast(pl.Float32))
    with pytest.raises(LabelValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH
    mismatched = cast(
        tuple[str, ...],
        dict(exc_info.value.details)["mismatched_columns"],
    )
    assert "future_return_1" in mismatched


def test_dtype_mismatch_direction_label() -> None:
    """Non-Int8 direction labels raise LabelValidationError."""
    frame = _label_frame().with_columns(pl.col("direction_1").cast(pl.Int64))
    with pytest.raises(LabelValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH
    mismatched = cast(
        tuple[str, ...],
        dict(exc_info.value.details)["mismatched_columns"],
    )
    assert "direction_1" in mismatched


def test_immutability() -> None:
    """verify does not mutate the input DataFrame."""
    frame = _label_frame(
        open_times=[_START + _INTERVAL, _START, _START],
        regression_overrides={
            "future_return_1": [0.01, None, math.nan],  # type: ignore[dict-item]
            "future_return_5": [0.01, math.inf, 0.01],
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
    frame = _label_frame(
        open_times=[_START + _INTERVAL, _START, _START],
        regression_overrides={
            "future_return_1": [0.01, None, math.nan],  # type: ignore[dict-item]
            "future_return_20": [0.01, math.inf, 0.01],
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
    assert "Infinite label values detected." in report.warnings
    assert "Frame is not sorted by open_time." in report.warnings
    assert isinstance(report.warnings, tuple)
