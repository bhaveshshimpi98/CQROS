"""Unit tests for CQROS portfolio ``PortfolioVerifier``."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from math import inf, nan
from typing import cast

import polars as pl
import pytest

from cqros.portfolio import PortfolioVerifier
from cqros.portfolio.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    MERGED_PORTFOLIO_SCHEMA,
    REQUIRED_COLUMNS,
)
from cqros.portfolio.verification import (
    ERROR_REQUIRED_COLUMNS,
    ERROR_SCHEMA_MISMATCH,
    PortfolioValidationError,
    VerificationReport,
)
from cqros.portfolio.verification import (
    PortfolioVerifier as PortfolioVerifierFromPackage,
)
from cqros.portfolio.verification.verifier import (
    PortfolioVerifier as PortfolioVerifierFromModule,
)
from cqros.processing.verification.interfaces import DataVerifier
from cqros.signals import Signal

_START = datetime(2024, 1, 1, tzinfo=UTC)
_INTERVAL = timedelta(hours=1)
_MODEL_NAME = "alpha-lgbm"
_MODEL_VERSION = "1.0.0"
_OPTIMIZER = "equal_weight"


def _open_times(count: int = 3) -> list[datetime]:
    """Build monotonically increasing UTC open_time values."""
    return [_START + (index * _INTERVAL) for index in range(count)]


def _portfolio_frame(
    *,
    symbols: list[str | None] | None = None,
    timeframes: list[str | None] | None = None,
    open_times: Sequence[datetime | None] | None = None,
    model_names: list[str | None] | None = None,
    model_versions: list[str | None] | None = None,
    optimizers: list[str | None] | None = None,
    signals: list[str | None] | None = None,
    target_weights: list[float | None] | None = None,
    column_order: tuple[str, ...] | None = None,
) -> pl.DataFrame:
    """Build a canonical merged portfolio verification frame."""
    times = open_times if open_times is not None else _open_times()
    row_count = len(times)
    default_signals = [Signal.BUY.value, Signal.SELL.value, Signal.HOLD.value]
    while len(default_signals) < row_count:
        default_signals.append(Signal.HOLD.value)
    default_weights = [0.5, -0.5, 0.0]
    while len(default_weights) < row_count:
        default_weights.append(0.0)
    data: dict[str, object] = {
        "symbol": symbols if symbols is not None else ["BTCUSDT"] * row_count,
        "timeframe": timeframes if timeframes is not None else ["1h"] * row_count,
        "open_time": times,
        "model_name": (model_names if model_names is not None else [_MODEL_NAME] * row_count),
        "model_version": (
            model_versions if model_versions is not None else [_MODEL_VERSION] * row_count
        ),
        "optimizer": (optimizers if optimizers is not None else [_OPTIMIZER] * row_count),
        "signal": signals if signals is not None else default_signals[:row_count],
        "target_weight": (
            target_weights if target_weights is not None else default_weights[:row_count]
        ),
    }
    order = column_order if column_order is not None else CANONICAL_COLUMN_ORDER
    frame = pl.DataFrame(data, schema=dict(COLUMN_DTYPES))
    return frame.select(list(order))


def _verifier() -> PortfolioVerifier:
    """Build a PortfolioVerifier instance."""
    return PortfolioVerifier()


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


def test_package_exports_portfolio_verifier() -> None:
    """Package re-exports match the verification module symbol."""
    assert PortfolioVerifier is PortfolioVerifierFromModule
    assert PortfolioVerifierFromPackage is PortfolioVerifierFromModule


def test_portfolio_verifier_satisfies_data_verifier_protocol() -> None:
    """PortfolioVerifier structurally satisfies DataVerifier."""
    assert isinstance(_verifier(), DataVerifier)


def test_successful_verification() -> None:
    """A clean sorted merged portfolio frame passes verification."""
    report = _verifier().verify(_portfolio_frame())
    _assert_clean_pass(report, rows=3)


def test_empty_frame_passes() -> None:
    """An empty schema-valid frame passes verification."""
    frame = pl.DataFrame(schema=MERGED_PORTFOLIO_SCHEMA).select(list(CANONICAL_COLUMN_ORDER))
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=0)


def test_canonical_schema_columns() -> None:
    """Successful verification inspects the canonical Portfolio column set."""
    frame = _portfolio_frame()
    assert frame.columns == list(CANONICAL_COLUMN_ORDER)
    assert frame.columns == [
        "symbol",
        "timeframe",
        "open_time",
        "model_name",
        "model_version",
        "optimizer",
        "signal",
        "target_weight",
    ]
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=3)


def test_pass_report() -> None:
    """PASS report has zero defect counters and empty warnings."""
    report = _verifier().verify(_portfolio_frame())
    assert report.passed is True
    assert report.warnings == ()
    assert report.duplicate_timestamp_rows == 0
    assert report.null_rows == 0
    assert report.nan_rows == 0
    assert report.invalid_timestamp_rows == 0
    assert report.invalid_numeric_rows == 0


def test_duplicate_primary_keys() -> None:
    """Duplicate (symbol, timeframe, open_time) keys fail verification."""
    times = [_START, _START, _START + _INTERVAL]
    frame = _portfolio_frame(open_times=times)
    report = _verifier().verify(frame)
    assert report.duplicate_timestamp_rows == 1
    assert report.passed is False
    assert "Duplicate timestamps detected." in report.warnings


def test_duplicate_keys_allow_same_open_time_across_symbols() -> None:
    """Identical open_time values for distinct symbols do not count as duplicates."""
    frame = _portfolio_frame(
        symbols=["BTCUSDT", "ETHUSDT", "BTCUSDT"],
        open_times=[_START, _START, _START + _INTERVAL],
    )
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=3)


def test_missing_columns() -> None:
    """Missing required columns raise PortfolioValidationError."""
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timeframe": ["1h"],
            "open_time": [_START],
        }
    )
    with pytest.raises(PortfolioValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_REQUIRED_COLUMNS
    missing = cast(tuple[str, ...], dict(exc_info.value.details)["missing_columns"])
    assert "model_name" in missing
    assert "model_version" in missing
    assert "optimizer" in missing
    assert "signal" in missing
    assert "target_weight" in missing
    assert set(REQUIRED_COLUMNS) - set(frame.columns) == set(missing)


def test_missing_optimizer_column_raises() -> None:
    """Missing optimizer produces the same hard schema failure as any required column."""
    frame = _portfolio_frame().drop("optimizer")
    with pytest.raises(PortfolioValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_REQUIRED_COLUMNS
    missing = cast(tuple[str, ...], dict(exc_info.value.details)["missing_columns"])
    assert missing == ("optimizer",)


def test_null_values() -> None:
    """NULL values in required columns are counted and fail verification."""
    frame = _portfolio_frame(signals=[Signal.BUY.value, None, Signal.HOLD.value])
    report = _verifier().verify(frame)
    assert report.null_rows == 1
    assert report.passed is False
    assert "Rows containing NULL values." in report.warnings


def test_null_primary_key_rows() -> None:
    """NULL primary-key values are counted as null rows."""
    frame = _portfolio_frame(symbols=["BTCUSDT", None, "BTCUSDT"])
    report = _verifier().verify(frame)
    assert report.null_rows == 1
    assert report.passed is False
    assert "Rows containing NULL values." in report.warnings


def test_null_timestamp_rows() -> None:
    """NULL open_time values count as both null and invalid timestamp rows."""
    frame = _portfolio_frame(open_times=[_START, None, _START + 2 * _INTERVAL])
    report = _verifier().verify(frame)
    assert report.null_rows == 1
    assert report.invalid_timestamp_rows == 1
    assert report.passed is False
    assert "Rows containing NULL values." in report.warnings
    assert "Invalid timestamps detected." in report.warnings


def test_nan_target_weight_rows() -> None:
    """NaN target_weight values are counted and fail verification."""
    frame = _portfolio_frame(target_weights=[0.5, nan, 0.0])
    report = _verifier().verify(frame)
    assert report.nan_rows == 1
    assert report.passed is False
    assert "Rows containing NaN values." in report.warnings


def test_infinite_target_weight_rows() -> None:
    """Infinite target_weight values are counted and fail verification."""
    frame = _portfolio_frame(target_weights=[0.5, inf, -inf])
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 2
    assert report.passed is False
    assert "Invalid target weight values detected." in report.warnings


def test_unsorted_timestamps() -> None:
    """Unsorted open_time fails without incrementing counters."""
    frame = _portfolio_frame(
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
    frame = _portfolio_frame(
        signals=[Signal.BUY.value, "LONG", Signal.HOLD.value],
    )
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 1
    assert report.passed is False
    assert "Invalid signal values detected." in report.warnings


def test_invalid_signal_values_use_signal_enum() -> None:
    """Only Signal enum values are accepted in the signal column."""
    frame = _portfolio_frame(
        signals=[Signal.BUY.value, Signal.SELL.value, Signal.HOLD.value],
    )
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=3)

    frame = _portfolio_frame(signals=["buy", "sell", "hold"])
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 3
    assert report.passed is False
    assert "Invalid signal values detected." in report.warnings


def test_invalid_string_values() -> None:
    """Empty string values in string columns fail verification."""
    frame = _portfolio_frame(model_names=[_MODEL_NAME, "", _MODEL_NAME])
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 1
    assert report.passed is False
    assert "Invalid string values detected." in report.warnings


def test_empty_optimizer_values() -> None:
    """Empty optimizer values are reported as invalid strings, not hard failures."""
    frame = _portfolio_frame(optimizers=[_OPTIMIZER, "", _OPTIMIZER])
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
    frame = _portfolio_frame(column_order=reordered)
    report = _verifier().verify(frame)
    assert report.passed is False
    assert report.warnings == ("Frame column order does not match canonical order.",)


def test_dtype_mismatch_open_time() -> None:
    """Wrong open_time dtype raises PortfolioValidationError schema mismatch."""
    frame = _portfolio_frame().with_columns(pl.col("open_time").cast(pl.Int64))
    with pytest.raises(PortfolioValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH
    mismatched = cast(
        tuple[str, ...],
        dict(exc_info.value.details)["mismatched_columns"],
    )
    assert mismatched == ("open_time",)


def test_dtype_mismatch_target_weight() -> None:
    """Non-Float64 target_weight columns raise PortfolioValidationError."""
    frame = _portfolio_frame().with_columns(pl.col("target_weight").cast(pl.Float32))
    with pytest.raises(PortfolioValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH
    mismatched = cast(
        tuple[str, ...],
        dict(exc_info.value.details)["mismatched_columns"],
    )
    assert "target_weight" in mismatched


def test_dtype_mismatch_optimizer() -> None:
    """Non-Utf8 optimizer columns raise PortfolioValidationError."""
    frame = _portfolio_frame().with_columns(pl.col("optimizer").cast(pl.Categorical))
    with pytest.raises(PortfolioValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH
    mismatched = cast(
        tuple[str, ...],
        dict(exc_info.value.details)["mismatched_columns"],
    )
    assert "optimizer" in mismatched


def test_dtype_mismatch_signal_column() -> None:
    """Non-Utf8 signal columns raise PortfolioValidationError."""
    frame = _portfolio_frame().with_columns(pl.col("signal").cast(pl.Categorical))
    with pytest.raises(PortfolioValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH
    mismatched = cast(
        tuple[str, ...],
        dict(exc_info.value.details)["mismatched_columns"],
    )
    assert "signal" in mismatched


def test_does_not_enforce_weight_sum_rules() -> None:
    """Verifier does not require BUY/SELL weights to sum to +/-1.0."""
    frame = _portfolio_frame(
        signals=[Signal.BUY.value, Signal.BUY.value, Signal.SELL.value],
        target_weights=[0.1, 0.1, -0.1],
    )
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=3)


def test_immutability() -> None:
    """verify does not mutate the input DataFrame."""
    frame = _portfolio_frame(
        open_times=[_START + _INTERVAL, _START, _START],
        signals=[Signal.BUY.value, None, "LONG"],
        target_weights=[0.5, nan, inf],
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
    frame = _portfolio_frame(
        open_times=[_START + _INTERVAL, _START, _START],
        signals=[Signal.BUY.value, None, "LONG"],
        model_names=[_MODEL_NAME, "", _MODEL_NAME],
        target_weights=[0.5, nan, inf],
        column_order=reordered,
    )
    report = _verifier().verify(frame)
    assert report.rows_checked == 3
    assert report.duplicate_timestamp_rows == 1
    assert report.null_rows == 1
    assert report.nan_rows == 1
    assert report.invalid_numeric_rows >= 1
    assert report.passed is False
    assert "Frame column order does not match canonical order." in report.warnings
    assert "Duplicate timestamps detected." in report.warnings
    assert "Rows containing NULL values." in report.warnings
    assert "Rows containing NaN values." in report.warnings
    assert "Invalid signal values detected." in report.warnings
    assert "Invalid string values detected." in report.warnings
    assert "Invalid target weight values detected." in report.warnings
    assert "Frame is not sorted by open_time." in report.warnings
    assert isinstance(report.warnings, tuple)


def test_portfolio_verifier_exported_from_root_package() -> None:
    """Portfolio package re-exports PortfolioVerifier."""
    from cqros import portfolio as portfolio_package

    assert "PortfolioVerifier" in portfolio_package.__all__
    assert portfolio_package.PortfolioVerifier is PortfolioVerifierFromModule
