"""Unit tests for CQROS production signal policies."""

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
    ClassificationSignalPolicy,
    RegressionSignalPolicy,
    Signal,
    SignalPolicy,
    SignalValidationError,
)
from cqros.signals.policies import (
    ClassificationSignalPolicy as ClassificationSignalPolicyDirect,
)
from cqros.signals.policies import (
    RegressionSignalPolicy as RegressionSignalPolicyDirect,
)

_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1h"
_MODEL_NAME = "alpha-lgbm"
_MODEL_VERSION = "1.0.0"


def _prediction_frame(
    *,
    predictions: list[float],
    open_times: list[int] | None = None,
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    model_names: list[str] | None = None,
    model_versions: list[str] | None = None,
) -> pl.DataFrame:
    """Build a canonical prediction DataFrame for policy tests."""
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


def test_policies_are_exported_from_package() -> None:
    """Package exports match the policies module by identity."""
    assert RegressionSignalPolicy is RegressionSignalPolicyDirect
    assert ClassificationSignalPolicy is ClassificationSignalPolicyDirect


def test_policies_satisfy_signal_policy_protocol() -> None:
    """Production policies structurally satisfy SignalPolicy."""
    assert isinstance(RegressionSignalPolicy(0.5, -0.5), SignalPolicy)
    assert isinstance(ClassificationSignalPolicy(), SignalPolicy)


def test_regression_buy() -> None:
    """Predictions at or above buy_threshold map to BUY."""
    policy = RegressionSignalPolicy(buy_threshold=0.5, sell_threshold=-0.5)
    result = policy.generate(_prediction_frame(predictions=[0.5, 1.2]))
    assert result.get_column("signal").to_list() == [Signal.BUY.value, Signal.BUY.value]


def test_regression_sell() -> None:
    """Predictions at or below sell_threshold map to SELL."""
    policy = RegressionSignalPolicy(buy_threshold=0.5, sell_threshold=-0.5)
    result = policy.generate(_prediction_frame(predictions=[-0.5, -1.2]))
    assert result.get_column("signal").to_list() == [
        Signal.SELL.value,
        Signal.SELL.value,
    ]


def test_regression_hold() -> None:
    """Predictions strictly between thresholds map to HOLD."""
    policy = RegressionSignalPolicy(buy_threshold=0.5, sell_threshold=-0.5)
    result = policy.generate(_prediction_frame(predictions=[0.0, 0.49, -0.49]))
    assert result.get_column("signal").to_list() == [
        Signal.HOLD.value,
        Signal.HOLD.value,
        Signal.HOLD.value,
    ]


def test_mixed_regression_predictions() -> None:
    """Mixed continuous predictions map to BUY, SELL, and HOLD."""
    policy = RegressionSignalPolicy(buy_threshold=0.5, sell_threshold=-0.5)
    result = policy.generate(_prediction_frame(predictions=[0.8, -0.9, 0.1, 0.5, -0.5]))
    assert result.get_column("signal").to_list() == [
        Signal.BUY.value,
        Signal.SELL.value,
        Signal.HOLD.value,
        Signal.BUY.value,
        Signal.SELL.value,
    ]


def test_classification_buy() -> None:
    """Positive-class predictions map to BUY."""
    policy = ClassificationSignalPolicy(positive_class=1, negative_class=0)
    result = policy.generate(_prediction_frame(predictions=[1.0, 1.0]))
    assert result.get_column("signal").to_list() == [Signal.BUY.value, Signal.BUY.value]


def test_classification_sell() -> None:
    """Negative-class predictions map to SELL."""
    policy = ClassificationSignalPolicy(positive_class=1, negative_class=0)
    result = policy.generate(_prediction_frame(predictions=[0.0, 0.0]))
    assert result.get_column("signal").to_list() == [
        Signal.SELL.value,
        Signal.SELL.value,
    ]


def test_classification_hold() -> None:
    """Unrecognized class predictions map to HOLD."""
    policy = ClassificationSignalPolicy(positive_class=1, negative_class=0)
    result = policy.generate(_prediction_frame(predictions=[2.0, -1.0, 0.5]))
    assert result.get_column("signal").to_list() == [
        Signal.HOLD.value,
        Signal.HOLD.value,
        Signal.HOLD.value,
    ]


def test_invalid_thresholds() -> None:
    """Non-numeric, non-finite, and unordered thresholds raise."""
    with pytest.raises(SignalValidationError) as non_numeric:
        RegressionSignalPolicy(buy_threshold="0.5", sell_threshold=-0.5)  # type: ignore[arg-type]
    assert non_numeric.value.error_code == "SIGNAL-POL-001"

    with pytest.raises(SignalValidationError) as nan_threshold:
        RegressionSignalPolicy(buy_threshold=math.nan, sell_threshold=-0.5)
    assert nan_threshold.value.error_code == "SIGNAL-POL-002"

    with pytest.raises(SignalValidationError) as infinite_threshold:
        RegressionSignalPolicy(buy_threshold=math.inf, sell_threshold=-0.5)
    assert infinite_threshold.value.error_code == "SIGNAL-POL-002"

    with pytest.raises(SignalValidationError) as unordered:
        RegressionSignalPolicy(buy_threshold=-0.5, sell_threshold=0.5)
    assert unordered.value.error_code == "SIGNAL-POL-003"

    with pytest.raises(SignalValidationError) as equal_thresholds:
        RegressionSignalPolicy(buy_threshold=0.0, sell_threshold=0.0)
    assert equal_thresholds.value.error_code == "SIGNAL-POL-003"


def test_identical_class_labels() -> None:
    """Identical positive and negative class labels raise."""
    with pytest.raises(SignalValidationError) as exc_info:
        ClassificationSignalPolicy(positive_class=1, negative_class=1)
    assert exc_info.value.error_code == "SIGNAL-POL-005"


def test_invalid_class_label_types() -> None:
    """Non-integer class labels raise SignalValidationError."""
    with pytest.raises(SignalValidationError) as float_label:
        ClassificationSignalPolicy(positive_class=1.0, negative_class=0)  # type: ignore[arg-type]
    assert float_label.value.error_code == "SIGNAL-POL-004"

    with pytest.raises(SignalValidationError) as bool_label:
        ClassificationSignalPolicy(positive_class=True, negative_class=0)  # type: ignore[arg-type]
    assert bool_label.value.error_code == "SIGNAL-POL-004"

    with pytest.raises(SignalValidationError) as string_label:
        ClassificationSignalPolicy(positive_class="1", negative_class=0)  # type: ignore[arg-type]
    assert string_label.value.error_code == "SIGNAL-POL-004"


def test_empty_dataframe() -> None:
    """Empty prediction frames raise SignalValidationError."""
    policy = RegressionSignalPolicy(buy_threshold=0.5, sell_threshold=-0.5)
    empty = _prediction_frame(predictions=[]).clear()
    with pytest.raises(SignalValidationError) as exc_info:
        policy.generate(empty)
    assert exc_info.value.error_code == "SIGNAL-POL-007"


def test_missing_required_columns() -> None:
    """Missing prediction-schema columns raise SignalValidationError."""
    policy = RegressionSignalPolicy(buy_threshold=0.5, sell_threshold=-0.5)
    frame = _prediction_frame(predictions=[0.8]).drop("prediction")
    with pytest.raises(SignalValidationError) as exc_info:
        policy.generate(frame)
    assert exc_info.value.error_code == "SIGNAL-POL-008"
    missing = cast(tuple[str, ...], dict(exc_info.value.details)["missing_columns"])
    assert "prediction" in missing


def test_dtype_mismatch() -> None:
    """Incorrect prediction-schema dtypes raise SignalValidationError."""
    policy = ClassificationSignalPolicy()
    frame = _prediction_frame(predictions=[1.0]).with_columns(pl.col("open_time").cast(pl.Int32))
    with pytest.raises(SignalValidationError) as exc_info:
        policy.generate(frame)
    assert exc_info.value.error_code == "SIGNAL-POL-009"
    mismatched = cast(
        tuple[str, ...],
        dict(exc_info.value.details)["mismatched_columns"],
    )
    assert "open_time" in mismatched


def test_canonical_column_order() -> None:
    """Policy output columns follow the merged signal canonical order."""
    policy = RegressionSignalPolicy(buy_threshold=0.5, sell_threshold=-0.5)
    result = policy.generate(_prediction_frame(predictions=[0.8, -0.9]))
    assert result.columns == list(CANONICAL_COLUMN_ORDER)
    assert "prediction" not in result.columns
    assert "confidence" not in result.columns


def test_canonical_dtypes() -> None:
    """Policy output dtypes match the merged signal schema."""
    policy = ClassificationSignalPolicy()
    result = policy.generate(_prediction_frame(predictions=[1.0, 0.0]))
    assert result.schema == MERGED_SIGNAL_SCHEMA
    for column in CANONICAL_COLUMN_ORDER:
        assert result.schema[column] == COLUMN_DTYPES[column]


def test_metadata_preservation() -> None:
    """Primary keys and model metadata are preserved from the input."""
    policy = RegressionSignalPolicy(buy_threshold=0.5, sell_threshold=-0.5)
    frame = _prediction_frame(
        predictions=[0.8, -0.9],
        open_times=[100, 200],
        symbols=["BTCUSDT", "ETHUSDT"],
        timeframes=["1h", "4h"],
        model_names=["alpha", "beta"],
        model_versions=["1.0.0", "2.0.0"],
    )
    result = policy.generate(frame)
    assert result.get_column("symbol").to_list() == ["BTCUSDT", "ETHUSDT"]
    assert result.get_column("timeframe").to_list() == ["1h", "4h"]
    assert result.get_column("open_time").to_list() == [100, 200]
    assert result.get_column("model_name").to_list() == ["alpha", "beta"]
    assert result.get_column("model_version").to_list() == ["1.0.0", "2.0.0"]


def test_row_order_preservation() -> None:
    """Policy output preserves input row order."""
    policy = ClassificationSignalPolicy()
    frame = _prediction_frame(
        predictions=[1.0, 0.0, 2.0, 1.0],
        open_times=[40, 10, 30, 20],
    )
    result = policy.generate(frame)
    assert result.get_column("open_time").to_list() == [40, 10, 30, 20]
    assert result.get_column("signal").to_list() == [
        Signal.BUY.value,
        Signal.SELL.value,
        Signal.HOLD.value,
        Signal.BUY.value,
    ]


def test_input_immutability() -> None:
    """Policies return a new frame and do not mutate the input."""
    policy = RegressionSignalPolicy(buy_threshold=0.5, sell_threshold=-0.5)
    frame = _prediction_frame(predictions=[0.8, -0.9, 0.0])
    original = frame.clone()
    result = policy.generate(frame)
    assert_frame_equal(frame, original)
    assert result is not frame
    assert "signal" not in frame.columns
