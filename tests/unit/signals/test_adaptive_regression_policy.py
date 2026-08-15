"""Unit tests for CQROS ``AdaptiveRegressionSignalPolicy``."""

from __future__ import annotations

import math
from typing import cast

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.signals import (
    AdaptiveRegressionSignalPolicy,
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    InMemoryThresholdProvider,
    MERGED_SIGNAL_SCHEMA,
    RegressionThresholds,
    Signal,
    SignalPipeline,
    SignalPolicy,
    SignalPolicyRegistry,
    SignalValidationError,
    ThresholdProvider,
)
from cqros.signals.adaptive_regression_policy import (
    AdaptiveRegressionSignalPolicy as AdaptiveRegressionSignalPolicyDirect,
)
from cqros.storage import SignalPartitionRef

_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1h"
_MODEL_NAME = "alpha-lgbm"
_MODEL_VERSION = "1.0.0"
_BUY_THRESHOLD = 0.5
_SELL_THRESHOLD = -0.5
_GLOBAL = RegressionThresholds(
    buy_threshold=_BUY_THRESHOLD,
    sell_threshold=_SELL_THRESHOLD,
)


class _MissingThresholdProvider:
    """Provider that reports missing thresholds for every partition."""

    def get_thresholds(
        self,
        symbol: str,
        timeframe: str,
        model_name: str,
        model_version: str,
    ) -> RegressionThresholds:
        raise SignalValidationError(
            "thresholds not found",
            error_code="SIGNAL-THR-MISSING",
            details={
                "symbol": symbol,
                "timeframe": timeframe,
                "model_name": model_name,
                "model_version": model_version,
            },
        )


class _NullThresholdProvider:
    """Provider that returns ``None`` instead of thresholds."""

    def get_thresholds(
        self,
        symbol: str,
        timeframe: str,
        model_name: str,
        model_version: str,
    ) -> RegressionThresholds:
        return None  # type: ignore[return-value]


class _InvalidOrderProvider:
    """Provider that returns unordered thresholds."""

    def get_thresholds(
        self,
        symbol: str,
        timeframe: str,
        model_name: str,
        model_version: str,
    ) -> RegressionThresholds:
        return RegressionThresholds(buy_threshold=-0.5, sell_threshold=0.5)


class _NonFiniteProvider:
    """Provider that returns non-finite thresholds."""

    def get_thresholds(
        self,
        symbol: str,
        timeframe: str,
        model_name: str,
        model_version: str,
    ) -> RegressionThresholds:
        return RegressionThresholds(buy_threshold=math.nan, sell_threshold=-0.5)


class _RecordingRepository:
    """Minimal signal repository stub that records save calls."""

    def __init__(self) -> None:
        self.saved: list[pl.DataFrame] = []

    def save(
        self,
        dataframe: pl.DataFrame,
        *,
        exchange: str,
        market: str,
        symbol: str,
        timeframe: str,
        year: int,
    ) -> None:
        self.saved.append(dataframe)


def _prediction_frame(
    *,
    predictions: list[float | None],
    open_times: list[int] | None = None,
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    model_names: list[str] | None = None,
    model_versions: list[str] | None = None,
) -> pl.DataFrame:
    """Build a canonical prediction DataFrame for adaptive policy tests."""
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


def _policy(
    provider: ThresholdProvider | None = None,
) -> AdaptiveRegressionSignalPolicy:
    """Return an adaptive policy with shared global thresholds."""
    if provider is None:
        provider = InMemoryThresholdProvider(global_thresholds=_GLOBAL)
    return AdaptiveRegressionSignalPolicy(provider)


def test_exported_from_package() -> None:
    """Package export matches the adaptive policy module by identity."""
    assert AdaptiveRegressionSignalPolicy is AdaptiveRegressionSignalPolicyDirect


def test_satisfies_signal_policy_protocol() -> None:
    """AdaptiveRegressionSignalPolicy structurally satisfies SignalPolicy."""
    assert isinstance(_policy(), SignalPolicy)


def test_buy_generation() -> None:
    """Predictions at or above buy_threshold map to BUY."""
    result = _policy().generate(_prediction_frame(predictions=[0.51, 1.2, _BUY_THRESHOLD]))
    assert result.get_column("signal").to_list() == [
        Signal.BUY.value,
        Signal.BUY.value,
        Signal.BUY.value,
    ]


def test_sell_generation() -> None:
    """Predictions at or below sell_threshold map to SELL."""
    result = _policy().generate(_prediction_frame(predictions=[-0.51, -1.2, _SELL_THRESHOLD]))
    assert result.get_column("signal").to_list() == [
        Signal.SELL.value,
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


def test_symbol_override_changes_signals() -> None:
    """Per-symbol thresholds from the provider change BUY/SELL boundaries."""
    provider = InMemoryThresholdProvider(
        global_thresholds=_GLOBAL,
        symbol_overrides={
            "ETHUSDT": RegressionThresholds(buy_threshold=0.1, sell_threshold=-0.1),
        },
    )
    frame = _prediction_frame(
        predictions=[0.2, 0.2],
        symbols=["BTCUSDT", "ETHUSDT"],
        open_times=[1, 2],
    )
    result = AdaptiveRegressionSignalPolicy(provider).generate(frame)
    assert result.get_column("signal").to_list() == [
        Signal.HOLD.value,
        Signal.BUY.value,
    ]


def test_timeframe_override_changes_signals() -> None:
    """Symbol+timeframe overrides are applied per partition."""
    provider = InMemoryThresholdProvider(
        global_thresholds=_GLOBAL,
        symbol_timeframe_overrides={
            ("BTCUSDT", "4h"): RegressionThresholds(buy_threshold=0.1, sell_threshold=-0.1),
        },
    )
    frame = _prediction_frame(
        predictions=[0.2, 0.2],
        timeframes=["1h", "4h"],
        open_times=[1, 2],
    )
    result = AdaptiveRegressionSignalPolicy(provider).generate(frame)
    assert result.get_column("signal").to_list() == [
        Signal.HOLD.value,
        Signal.BUY.value,
    ]


def test_model_override_changes_signals() -> None:
    """Model+version overrides are applied per partition."""
    provider = InMemoryThresholdProvider(
        global_thresholds=_GLOBAL,
        model_overrides={
            ("beta", "2.0.0"): RegressionThresholds(buy_threshold=0.1, sell_threshold=-0.1),
        },
    )
    frame = _prediction_frame(
        predictions=[0.2, 0.2],
        model_names=["alpha-lgbm", "beta"],
        model_versions=["1.0.0", "2.0.0"],
        open_times=[1, 2],
    )
    result = AdaptiveRegressionSignalPolicy(provider).generate(frame)
    assert result.get_column("signal").to_list() == [
        Signal.HOLD.value,
        Signal.BUY.value,
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


def test_multi_partition_preserves_row_order() -> None:
    """Mixed partitions preserve input row order after per-partition mapping."""
    provider = InMemoryThresholdProvider(
        global_thresholds=_GLOBAL,
        symbol_overrides={
            "ETHUSDT": RegressionThresholds(buy_threshold=0.1, sell_threshold=-0.1),
        },
    )
    frame = _prediction_frame(
        predictions=[0.2, 0.2, -0.2],
        symbols=["ETHUSDT", "BTCUSDT", "ETHUSDT"],
        open_times=[10, 20, 30],
    )
    result = AdaptiveRegressionSignalPolicy(provider).generate(frame)
    assert result.get_column("open_time").to_list() == [10, 20, 30]
    assert result.get_column("signal").to_list() == [
        Signal.BUY.value,
        Signal.HOLD.value,
        Signal.SELL.value,
    ]


def test_dtype_preservation() -> None:
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


def test_invalid_threshold_order() -> None:
    """buy_threshold <= sell_threshold raises SignalValidationError."""
    with pytest.raises(SignalValidationError) as exc_info:
        AdaptiveRegressionSignalPolicy(_InvalidOrderProvider()).generate(
            _prediction_frame(predictions=[0.0])
        )
    assert exc_info.value.error_code == "SIGNAL-ADAPT-005"


def test_non_finite_thresholds() -> None:
    """Non-finite provider thresholds raise SignalValidationError."""
    with pytest.raises(SignalValidationError) as exc_info:
        AdaptiveRegressionSignalPolicy(_NonFiniteProvider()).generate(
            _prediction_frame(predictions=[0.0])
        )
    assert exc_info.value.error_code == "SIGNAL-ADAPT-004"


def test_missing_thresholds() -> None:
    """Provider failures for missing thresholds propagate as SignalValidationError."""
    with pytest.raises(SignalValidationError) as exc_info:
        AdaptiveRegressionSignalPolicy(_MissingThresholdProvider()).generate(
            _prediction_frame(predictions=[0.0])
        )
    assert exc_info.value.error_code == "SIGNAL-THR-MISSING"


def test_null_thresholds() -> None:
    """Null provider results raise SignalValidationError."""
    with pytest.raises(SignalValidationError) as exc_info:
        AdaptiveRegressionSignalPolicy(_NullThresholdProvider()).generate(
            _prediction_frame(predictions=[0.0])
        )
    assert exc_info.value.error_code == "SIGNAL-ADAPT-003"


def test_constructor_rejects_non_provider() -> None:
    """Non-ThresholdProvider construction arguments are rejected."""
    with pytest.raises(SignalValidationError) as exc_info:
        AdaptiveRegressionSignalPolicy(object())  # type: ignore[arg-type]
    assert exc_info.value.error_code == "SIGNAL-ADAPT-001"


def test_registry_registration() -> None:
    """Adaptive policy can be registered and resolved by name."""
    registry = SignalPolicyRegistry()
    policy = _policy()
    registry.register("adaptive_regression", policy)
    assert registry.exists("adaptive_regression")
    assert registry.get("adaptive_regression") is policy
    assert registry.list() == ("adaptive_regression",)


def test_pipeline_execution() -> None:
    """Adaptive policy executes through SignalPipeline end-to-end."""
    registry = SignalPolicyRegistry()
    registry.register("adaptive_regression", _policy())
    repository = _RecordingRepository()
    pipeline = SignalPipeline(repository, registry)  # type: ignore[arg-type]
    partition_ref = SignalPartitionRef(
        exchange="binance",
        market="usdt_perpetual",
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=2024,
    )
    result = pipeline.run(
        "adaptive_regression",
        _prediction_frame(predictions=[0.8, -0.9, 0.0]),
        partition_ref,
    )
    assert result.schema == MERGED_SIGNAL_SCHEMA
    assert result.get_column("signal").to_list() == [
        Signal.BUY.value,
        Signal.SELL.value,
        Signal.HOLD.value,
    ]
    assert len(repository.saved) == 1
    assert_frame_equal(repository.saved[0], result)


def test_empty_frame() -> None:
    """Empty prediction frames raise SignalValidationError."""
    empty = _prediction_frame(predictions=[]).clear()
    with pytest.raises(SignalValidationError) as exc_info:
        _policy().generate(empty)
    assert exc_info.value.error_code == "SIGNAL-POL-007"


def test_null_prediction_rejected() -> None:
    """Null prediction values raise SignalValidationError."""
    frame = _prediction_frame(predictions=[0.8, None])
    with pytest.raises(SignalValidationError) as exc_info:
        _policy().generate(frame)
    assert exc_info.value.error_code == "SIGNAL-POL-010"
    assert dict(exc_info.value.details)["invalid_prediction_rows"] == 1


def test_missing_required_columns() -> None:
    """Missing prediction-schema columns raise SignalValidationError."""
    frame = _prediction_frame(predictions=[0.8]).drop("prediction")
    with pytest.raises(SignalValidationError) as exc_info:
        _policy().generate(frame)
    assert exc_info.value.error_code == "SIGNAL-POL-008"
    missing = cast(tuple[str, ...], dict(exc_info.value.details)["missing_columns"])
    assert "prediction" in missing
