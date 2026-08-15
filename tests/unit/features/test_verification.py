"""Unit tests for CQROS features ``FeatureVerifier``."""

from __future__ import annotations

import math
from typing import cast

import polars as pl
import pytest

from cqros.features import FeatureVerifier
from cqros.features.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    FEATURE_COLUMNS,
    REQUIRED_COLUMNS,
)
from cqros.features.verification import (
    ERROR_SCHEMA_MISMATCH,
    FeatureValidationError,
    VerificationReport,
)
from cqros.features.verification import (
    FeatureVerifier as FeatureVerifierFromPackage,
)
from cqros.features.verification.verifier import FeatureVerifier as FeatureVerifierFromModule
from cqros.processing.verification.exceptions import (
    ERROR_REQUIRED_COLUMNS,
    ProcessingValidationError,
)
from cqros.processing.verification.interfaces import DataVerifier

_START = 1_700_000_000_000
_INTERVAL = 3_600_000


def _feature_values(row_count: int, *, value: float = 1.0) -> dict[str, list[float]]:
    """Build default float values for every feature column."""
    return {column: [value] * row_count for column in FEATURE_COLUMNS}


def _feature_frame(
    *,
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    open_times: list[int] | None = None,
    feature_overrides: dict[str, list[float | None]] | None = None,
    column_order: tuple[str, ...] | None = None,
) -> pl.DataFrame:
    """Build a canonical merged feature verification frame."""
    if open_times is None:
        open_times = [_START, _START + _INTERVAL, _START + 2 * _INTERVAL]
    row_count = len(open_times)
    data: dict[str, object] = {
        "symbol": symbols if symbols is not None else ["BTCUSDT"] * row_count,
        "timeframe": timeframes if timeframes is not None else ["1h"] * row_count,
        "open_time": open_times,
    }
    data.update(_feature_values(row_count))
    if feature_overrides is not None:
        data.update(feature_overrides)
    order = column_order if column_order is not None else CANONICAL_COLUMN_ORDER
    frame = pl.DataFrame(data, schema=COLUMN_DTYPES)
    return frame.select(list(order))


def _verifier() -> FeatureVerifier:
    """Build a FeatureVerifier instance."""
    return FeatureVerifier()


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


def test_package_exports_feature_verifier() -> None:
    """Package re-exports match the verification module symbol."""
    assert FeatureVerifier is FeatureVerifierFromModule
    assert FeatureVerifierFromPackage is FeatureVerifierFromModule


def test_feature_verifier_satisfies_data_verifier_protocol() -> None:
    """FeatureVerifier structurally satisfies DataVerifier."""
    assert isinstance(_verifier(), DataVerifier)


def test_successful_verification() -> None:
    """A clean sorted merged feature frame passes verification."""
    report = _verifier().verify(_feature_frame())
    _assert_clean_pass(report, rows=3)


def test_empty_frame_passes() -> None:
    """An empty schema-valid frame passes verification."""
    frame = pl.DataFrame(schema=COLUMN_DTYPES).select(list(CANONICAL_COLUMN_ORDER))
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=0)


def test_duplicate_keys() -> None:
    """Duplicate (symbol, timeframe, open_time) keys fail verification."""
    frame = _feature_frame(open_times=[_START, _START, _START + _INTERVAL])
    report = _verifier().verify(frame)
    assert report.duplicate_timestamp_rows == 1
    assert report.passed is False
    assert "Duplicate timestamps detected." in report.warnings


def test_duplicate_keys_allow_same_open_time_across_symbols() -> None:
    """Identical open_time values for distinct symbols do not count as duplicates."""
    frame = _feature_frame(
        symbols=["BTCUSDT", "ETHUSDT", "BTCUSDT"],
        open_times=[_START, _START, _START + _INTERVAL],
    )
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=3)


def test_missing_columns() -> None:
    """Missing required columns raise ProcessingValidationError."""
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timeframe": ["1h"],
            "open_time": [_START],
        }
    )
    with pytest.raises(ProcessingValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_REQUIRED_COLUMNS
    missing = cast(tuple[str, ...], dict(exc_info.value.details)["missing_columns"])
    assert "returns" in missing
    assert set(REQUIRED_COLUMNS) - set(frame.columns) == set(missing)


def test_null_rows() -> None:
    """NULL values in required columns are counted and fail verification."""
    frame = _feature_frame(
        feature_overrides={"returns": [1.0, None, 1.0]},  # type: ignore[dict-item]
    )
    report = _verifier().verify(frame)
    assert report.null_rows == 1
    assert report.passed is False
    assert "Rows containing NULL values." in report.warnings


def test_null_primary_key_rows() -> None:
    """NULL primary-key values are counted as null rows."""
    frame = _feature_frame(symbols=["BTCUSDT", None, "BTCUSDT"])  # type: ignore[list-item]
    report = _verifier().verify(frame)
    assert report.null_rows == 1
    assert report.passed is False
    assert "Rows containing NULL values." in report.warnings


def test_nan_rows() -> None:
    """NaN feature values are counted and fail verification."""
    frame = _feature_frame(feature_overrides={"atr": [1.0, math.nan, 1.0]})
    report = _verifier().verify(frame)
    assert report.nan_rows == 1
    assert report.invalid_numeric_rows == 0
    assert report.passed is False
    assert "Rows containing NaN values." in report.warnings


def test_infinite_values() -> None:
    """Infinite feature values are counted as invalid numeric rows."""
    frame = _feature_frame(
        feature_overrides={"oi_zscore": [1.0, math.inf, -math.inf]},
    )
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 2
    assert report.nan_rows == 0
    assert report.passed is False
    assert "Infinite feature values detected." in report.warnings


def test_unsorted_data() -> None:
    """Unsorted open_time fails without incrementing counters."""
    frame = _feature_frame(
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


def test_canonical_column_order_mismatch() -> None:
    """Wrong column order fails verification with a deterministic warning."""
    reordered = (
        *CANONICAL_COLUMN_ORDER[1:],
        CANONICAL_COLUMN_ORDER[0],
    )
    frame = _feature_frame(column_order=reordered)
    report = _verifier().verify(frame)
    assert report.passed is False
    assert report.warnings == ("Frame column order does not match canonical order.",)


def test_schema_mismatch_wrong_dtype() -> None:
    """Wrong column dtypes raise FeatureValidationError schema mismatch."""
    frame = _feature_frame().with_columns(pl.col("open_time").cast(pl.Float64))
    with pytest.raises(FeatureValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH
    mismatched = cast(
        tuple[str, ...],
        dict(exc_info.value.details)["mismatched_columns"],
    )
    assert mismatched == ("open_time",)


def test_schema_mismatch_feature_dtype() -> None:
    """Non-Float64 feature columns raise FeatureValidationError."""
    frame = _feature_frame().with_columns(pl.col("returns").cast(pl.Float32))
    with pytest.raises(FeatureValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH
    mismatched = cast(
        tuple[str, ...],
        dict(exc_info.value.details)["mismatched_columns"],
    )
    assert "returns" in mismatched


def test_invalid_timestamps() -> None:
    """Non-positive open_time values are counted as invalid."""
    frame = _feature_frame(open_times=[_START, 0, _START + 2 * _INTERVAL])
    report = _verifier().verify(frame)
    assert report.invalid_timestamp_rows == 1
    assert report.passed is False
    assert "Invalid timestamps detected." in report.warnings


def test_negative_timestamps() -> None:
    """Negative open_time values are counted as invalid."""
    frame = _feature_frame(open_times=[_START, -1, _START + 2 * _INTERVAL])
    report = _verifier().verify(frame)
    assert report.invalid_timestamp_rows == 1
    assert report.passed is False


def test_immutability() -> None:
    """verify does not mutate the input DataFrame."""
    frame = _feature_frame(
        open_times=[_START + _INTERVAL, _START, _START],
        feature_overrides={
            "returns": [1.0, None, math.nan],  # type: ignore[dict-item]
            "atr": [1.0, math.inf, 1.0],
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
    frame = _feature_frame(
        open_times=[_START + _INTERVAL, _START, _START],
        feature_overrides={
            "returns": [1.0, None, math.nan],  # type: ignore[dict-item]
            "log_returns": [1.0, math.inf, 1.0],
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
    assert "Infinite feature values detected." in report.warnings
    assert "Frame is not sorted by open_time." in report.warnings
    assert isinstance(report.warnings, tuple)
