"""Unit tests for CQROS ``RegressionSignalPolicy``."""

from __future__ import annotations

import math
from typing import cast

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.signals import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    MERGED_SIGNAL_SCHEMA,
    RegressionSignalPolicy,
    Signal,
    SignalPolicy,
    SignalValidationError,
)
from cqros.signals.policies import (
    RegressionSignalPolicy as RegressionSignalPolicyDirect,
)

_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1h"
_MODEL_NAME = "alpha-lgbm"
_MODEL_VERSION = "1.0.0"
_BUY_THRESHOLD = 0.5
_SELL_THRESHOLD = -0.5


def _prediction_frame(
    *,
    predictions: list[float | None],
    open_times: list[int] | None = None,
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    model_names: list[str] | None = None,
    model_versions: list[str] | None = None,
) -> pl.DataFrame:
    """Build a canonical prediction DataFrame for regression policy tests."""
    row_count = len(predictions)
    return pl.DataFrame(
        {
            "symbol": symbols if symbols is not None else [_SYMBOL] * row_count,
            "timeframe": (timeframes if timeframes is not None else [_TIMEFRAME] * row_count),
            "open_time": (open_times if open_times is not None else list(range(row_count))),
            "model_name": (model_names if model_names is not None else [_MODEL_NAME] * row_count),
            "model_version": (
                model_versions if model_versions is not None else [_MODEL_VERSION] * row_count
            ),
            "prediction": predictions,
        },
        schema={
            "symbol": pl.String,
            "timeframe": pl.String,
            "open_time": pl.Int64,
            "model_name": pl.String,
            "model_version": pl.String,
            "prediction": pl.Float64,
        },
    )


def _policy() -> RegressionSignalPolicy:
    """Return a regression policy with the shared test thresholds."""
    return RegressionSignalPolicy(
        buy_threshold=_BUY_THRESHOLD,
        sell_threshold=_SELL_THRESHOLD,
    )


def test_exported_from_package() -> None:
    """Package export matches the policies module by identity."""
    assert RegressionSignalPolicy is RegressionSignalPolicyDirect


def test_satisfies_signal_policy_protocol() -> None:
    """RegressionSignalPolicy structurally satisfies SignalPolicy."""
    assert isinstance(_policy(), SignalPolicy)


def test_buy_generation() -> None:
    """Predictions strictly above buy_threshold map to BUY."""
    result = _policy().generate(_prediction_frame(predictions=[0.51, 1.2]))
    assert result.get_column("signal").to_list() == [
        Signal.BUY.value,
        Signal.BUY.value,
    ]


def test_sell_generation() -> None:
    """Predictions strictly below sell_threshold map to SELL."""
    result = _policy().generate(_prediction_frame(predictions=[-0.51, -1.2]))
    assert result.get_column("signal").to_list() == [
        Signal.SELL.value,
        Signal.SELL.value,
    ]


def test_hold_generation() -> None:
    """Predictions strictly between thresholds map to HOLD."""
    result = _policy().generate(_prediction_frame(predictions=[0.0, 0.49, -0.49]))
    assert result.get_column("signal").to_list() == [
        Signal.HOLD.value,
        Signal.HOLD.value,
        Signal.HOLD.value,
    ]


def test_buy_threshold_equality() -> None:
    """prediction equal to buy_threshold maps to BUY."""
    result = _policy().generate(_prediction_frame(predictions=[_BUY_THRESHOLD]))
    assert result.get_column("signal").to_list() == [Signal.BUY.value]


def test_sell_threshold_equality() -> None:
    """prediction equal to sell_threshold maps to SELL."""
    result = _policy().generate(_prediction_frame(predictions=[_SELL_THRESHOLD]))
    assert result.get_column("signal").to_list() == [Signal.SELL.value]


def test_threshold_boundary_conditions() -> None:
    """Inclusive buy/sell boundaries and open HOLD interval are respected."""
    result = _policy().generate(
        _prediction_frame(
            predictions=[
                _BUY_THRESHOLD,
                _BUY_THRESHOLD + 0.01,
                (_BUY_THRESHOLD + _SELL_THRESHOLD) / 2.0,
                _SELL_THRESHOLD - 0.01,
                _SELL_THRESHOLD,
            ]
        )
    )
    assert result.get_column("signal").to_list() == [
        Signal.BUY.value,
        Signal.BUY.value,
        Signal.HOLD.value,
        Signal.SELL.value,
        Signal.SELL.value,
    ]


def test_mixed_predictions() -> None:
    """Mixed continuous predictions map to BUY, SELL, and HOLD."""
    result = _policy().generate(
        _prediction_frame(predictions=[0.8, -0.9, 0.1, _BUY_THRESHOLD, _SELL_THRESHOLD])
    )
    assert result.get_column("signal").to_list() == [
        Signal.BUY.value,
        Signal.SELL.value,
        Signal.HOLD.value,
        Signal.BUY.value,
        Signal.SELL.value,
    ]


def test_constructor_non_numeric_threshold() -> None:
    """Non-numeric thresholds raise SignalValidationError."""
    with pytest.raises(SignalValidationError) as exc_info:
        RegressionSignalPolicy(buy_threshold="0.5", sell_threshold=-0.5)  # type: ignore[arg-type]
    assert exc_info.value.error_code == "SIGNAL-POL-001"


def test_constructor_non_finite_threshold() -> None:
    """NaN and infinite thresholds raise SignalValidationError."""
    with pytest.raises(SignalValidationError) as nan_threshold:
        RegressionSignalPolicy(buy_threshold=math.nan, sell_threshold=-0.5)
    assert nan_threshold.value.error_code == "SIGNAL-POL-002"

    with pytest.raises(SignalValidationError) as infinite_threshold:
        RegressionSignalPolicy(buy_threshold=math.inf, sell_threshold=-0.5)
    assert infinite_threshold.value.error_code == "SIGNAL-POL-002"


def test_constructor_unordered_thresholds() -> None:
    """buy_threshold must be strictly greater than sell_threshold."""
    with pytest.raises(SignalValidationError) as unordered:
        RegressionSignalPolicy(buy_threshold=-0.5, sell_threshold=0.5)
    assert unordered.value.error_code == "SIGNAL-POL-003"

    with pytest.raises(SignalValidationError) as equal_thresholds:
        RegressionSignalPolicy(buy_threshold=0.0, sell_threshold=0.0)
    assert equal_thresholds.value.error_code == "SIGNAL-POL-003"


def test_null_prediction_rejected() -> None:
    """Null prediction values raise SignalValidationError."""
    frame = _prediction_frame(predictions=[0.8, None])
    with pytest.raises(SignalValidationError) as exc_info:
        _policy().generate(frame)
    assert exc_info.value.error_code == "SIGNAL-POL-010"
    assert dict(exc_info.value.details)["invalid_prediction_rows"] == 1


def test_nan_prediction_rejected() -> None:
    """NaN prediction values raise SignalValidationError."""
    frame = _prediction_frame(predictions=[0.8, math.nan])
    with pytest.raises(SignalValidationError) as exc_info:
        _policy().generate(frame)
    assert exc_info.value.error_code == "SIGNAL-POL-010"
    assert dict(exc_info.value.details)["invalid_prediction_rows"] == 1


def test_non_finite_prediction_rejected() -> None:
    """Infinite prediction values raise SignalValidationError."""
    frame = _prediction_frame(predictions=[math.inf, -math.inf])
    with pytest.raises(SignalValidationError) as exc_info:
        _policy().generate(frame)
    assert exc_info.value.error_code == "SIGNAL-POL-010"
    assert dict(exc_info.value.details)["invalid_prediction_rows"] == 2


def test_empty_frame() -> None:
    """Empty prediction frames raise SignalValidationError."""
    empty = _prediction_frame(predictions=[]).clear()
    with pytest.raises(SignalValidationError) as exc_info:
        _policy().generate(empty)
    assert exc_info.value.error_code == "SIGNAL-POL-007"


def test_single_row() -> None:
    """A single eligible prediction row produces one signal row."""
    result = _policy().generate(_prediction_frame(predictions=[0.8]))
    assert result.height == 1
    assert result.get_column("signal").to_list() == [Signal.BUY.value]
    assert result.schema == MERGED_SIGNAL_SCHEMA


def test_multiple_rows() -> None:
    """Multiple prediction rows produce one signal per input row."""
    result = _policy().generate(_prediction_frame(predictions=[0.8, -0.9, 0.0]))
    assert result.height == 3
    assert result.get_column("signal").to_list() == [
        Signal.BUY.value,
        Signal.SELL.value,
        Signal.HOLD.value,
    ]


def test_schema() -> None:
    """Output columns and schema match MERGED_SIGNAL_SCHEMA."""
    result = _policy().generate(_prediction_frame(predictions=[0.8, -0.9]))
    assert result.schema == MERGED_SIGNAL_SCHEMA
    assert result.columns == list(CANONICAL_COLUMN_ORDER)
    assert "prediction" not in result.columns


def test_ordering() -> None:
    """Output columns follow CANONICAL_COLUMN_ORDER and preserve row order."""
    frame = _prediction_frame(
        predictions=[0.8, -0.9, 0.0],
        open_times=[40, 10, 30],
    )
    result = _policy().generate(frame)
    assert result.columns == list(CANONICAL_COLUMN_ORDER)
    assert result.get_column("open_time").to_list() == [40, 10, 30]


def test_dtype_casting() -> None:
    """Output dtypes match COLUMN_DTYPES / MERGED_SIGNAL_SCHEMA."""
    result = _policy().generate(_prediction_frame(predictions=[0.8, -0.9]))
    assert result.schema == MERGED_SIGNAL_SCHEMA
    for column in CANONICAL_COLUMN_ORDER:
        assert result.schema[column] == COLUMN_DTYPES[column]


def test_metadata_preservation() -> None:
    """Primary keys and model metadata are preserved unchanged."""
    frame = _prediction_frame(
        predictions=[0.8, -0.9],
        open_times=[100, 200],
        symbols=["BTCUSDT", "ETHUSDT"],
        timeframes=["1h", "4h"],
        model_names=["alpha", "beta"],
        model_versions=["1.0.0", "2.0.0"],
    )
    result = _policy().generate(frame)
    assert result.get_column("symbol").to_list() == ["BTCUSDT", "ETHUSDT"]
    assert result.get_column("timeframe").to_list() == ["1h", "4h"]
    assert result.get_column("open_time").to_list() == [100, 200]
    assert result.get_column("model_name").to_list() == ["alpha", "beta"]
    assert result.get_column("model_version").to_list() == ["1.0.0", "2.0.0"]


def test_input_immutability() -> None:
    """generate returns a new frame and does not mutate the input."""
    frame = _prediction_frame(predictions=[0.8, -0.9, 0.0])
    original = frame.clone()
    result = _policy().generate(frame)
    assert_frame_equal(frame, original)
    assert result is not frame
    assert "signal" not in frame.columns


def test_missing_required_columns() -> None:
    """Missing prediction-schema columns raise SignalValidationError."""
    frame = _prediction_frame(predictions=[0.8]).drop("prediction")
    with pytest.raises(SignalValidationError) as exc_info:
        _policy().generate(frame)
    assert exc_info.value.error_code == "SIGNAL-POL-008"
    missing = cast(tuple[str, ...], dict(exc_info.value.details)["missing_columns"])
    assert "prediction" in missing


def test_dtype_mismatch() -> None:
    """Incorrect prediction-schema dtypes raise SignalValidationError."""
    frame = _prediction_frame(predictions=[0.8]).with_columns(pl.col("open_time").cast(pl.Int32))
    with pytest.raises(SignalValidationError) as exc_info:
        _policy().generate(frame)
    assert exc_info.value.error_code == "SIGNAL-POL-009"
    mismatched = cast(
        tuple[str, ...],
        dict(exc_info.value.details)["mismatched_columns"],
    )
    assert "open_time" in mismatched
