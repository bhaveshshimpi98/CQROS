"""Unit tests for CQROS regression signal threshold calibration."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass
from math import isfinite

import polars as pl
import pytest

from cqros.core.exceptions import ResearchError
from cqros.research.signal_threshold_calibrator import (
    PredictionDistributionStatistics,
    SignalThresholdCalibrator,
    SymbolTimeframeCalibration,
    ThresholdCalibrationResult,
    ThresholdRecommendation,
)


def _calibrator() -> SignalThresholdCalibrator:
    """Build a calibrator instance."""
    return SignalThresholdCalibrator()


def _prediction_frame(values: list[float]) -> pl.DataFrame:
    """Build a minimal prediction frame for the given values."""
    count = len(values)
    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * count,
            "timeframe": ["1h"] * count,
            "open_time": list(range(count)),
            "model_name": ["alpha"] * count,
            "model_version": ["1.0.0"] * count,
            "prediction": values,
        }
    )


def _uniform_frame(count: int = 100) -> pl.DataFrame:
    """Build evenly spaced predictions in ``[-1, 1]``."""
    if count == 1:
        values = [0.0]
    else:
        values = [-1.0 + (2.0 * index / (count - 1)) for index in range(count)]
    return _prediction_frame(values)


# --- construction / immutability ---


def test_result_types_are_frozen() -> None:
    """Calibration result dataclasses are immutable."""
    result = _calibrator().calibrate(
        {
            ("BTCUSDT", "1h"): _uniform_frame(100),
        }
    )
    assert is_dataclass(result)
    assert isinstance(result, ThresholdCalibrationResult)
    assert isinstance(result.global_statistics, PredictionDistributionStatistics)
    assert isinstance(result.recommendations[0], ThresholdRecommendation)
    assert isinstance(result.symbol_timeframe_results[0], SymbolTimeframeCalibration)
    with pytest.raises(FrozenInstanceError):
        result.rows_analyzed = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.global_statistics.count = 0  # type: ignore[misc]


def test_analyze_does_not_mutate_input() -> None:
    """analyze never mutates the input DataFrame."""
    frame = _uniform_frame(20)
    before = frame.clone()
    _calibrator().analyze(frame)
    assert frame.equals(before)


# --- statistics ---


def test_analyze_computes_basic_statistics() -> None:
    """analyze reports count, range, mean, and median for known values."""
    frame = _prediction_frame([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = _calibrator().analyze(frame)
    assert stats.count == 5
    assert stats.minimum == pytest.approx(1.0)
    assert stats.maximum == pytest.approx(5.0)
    assert stats.mean == pytest.approx(3.0)
    assert stats.median == pytest.approx(3.0)
    assert isfinite(stats.std)
    assert stats.positive_ratio == pytest.approx(1.0)
    assert stats.negative_ratio == pytest.approx(0.0)


def test_analyze_computes_percentiles_and_sign_ratios() -> None:
    """analyze reports configured percentiles and sign ratios."""
    values = [float(index) for index in range(-50, 50)]
    stats = _calibrator().analyze(_prediction_frame(values))
    assert stats.count == 100
    assert stats.percentile_01 <= stats.percentile_025
    assert stats.percentile_025 <= stats.percentile_05
    assert stats.percentile_05 <= stats.percentile_10
    assert stats.percentile_10 <= stats.median
    assert stats.median <= stats.percentile_90
    assert stats.percentile_90 <= stats.percentile_95
    assert stats.percentile_95 <= stats.percentile_975
    assert stats.percentile_975 <= stats.percentile_99
    assert stats.positive_ratio == pytest.approx(0.49)
    assert stats.negative_ratio == pytest.approx(0.50)


def test_analyze_drops_null_predictions() -> None:
    """Null prediction rows are excluded from statistics."""
    frame = pl.DataFrame({"prediction": [1.0, None, 3.0, None, 5.0]})
    stats = _calibrator().analyze(frame)
    assert stats.count == 3
    assert stats.mean == pytest.approx(3.0)


# --- recommendations ---


def test_recommend_profiles_use_configured_percentiles() -> None:
    """Conservative/Balanced/Active map to 99/1, 95/5, and 90/10 percentiles."""
    frame = _uniform_frame(100)
    stats = _calibrator().analyze(frame)
    recommendations = _calibrator().recommend(frame, statistics=stats)
    assert tuple(item.profile for item in recommendations) == (
        "Conservative",
        "Balanced",
        "Active",
    )
    by_profile = {item.profile: item for item in recommendations}
    assert by_profile["Conservative"].buy_threshold == pytest.approx(stats.percentile_99)
    assert by_profile["Conservative"].sell_threshold == pytest.approx(stats.percentile_01)
    assert by_profile["Balanced"].buy_threshold == pytest.approx(stats.percentile_95)
    assert by_profile["Balanced"].sell_threshold == pytest.approx(stats.percentile_05)
    assert by_profile["Active"].buy_threshold == pytest.approx(stats.percentile_90)
    assert by_profile["Active"].sell_threshold == pytest.approx(stats.percentile_10)
    for item in recommendations:
        assert item.buy_threshold > item.sell_threshold
        assert item.expected_buy_ratio >= 0.0
        assert item.expected_sell_ratio >= 0.0
        assert item.expected_hold_ratio >= 0.0
        assert (
            item.expected_buy_ratio + item.expected_sell_ratio + item.expected_hold_ratio
            == pytest.approx(1.0)
        )


def test_expected_ratios_follow_inclusive_threshold_rules() -> None:
    """Expected BUY/SELL/HOLD match RegressionSignalPolicy inclusive rules."""
    values = [float(index) for index in range(100)]
    frame = _prediction_frame(values)
    recommendations = _calibrator().recommend(frame)
    balanced = next(item for item in recommendations if item.profile == "Balanced")
    series = frame.get_column("prediction")
    buy = int((series >= balanced.buy_threshold).sum())
    sell = int((series <= balanced.sell_threshold).sum())
    hold = series.len() - buy - sell
    assert balanced.expected_buy_ratio == pytest.approx(buy / series.len())
    assert balanced.expected_sell_ratio == pytest.approx(sell / series.len())
    assert balanced.expected_hold_ratio == pytest.approx(hold / series.len())


# --- group / aggregate calibration ---


def test_calibrate_group_labels_symbol_and_timeframe() -> None:
    """calibrate_group attaches symbol/timeframe metadata."""
    result = _calibrator().calibrate_group(
        _uniform_frame(50),
        symbol="ETHUSDT",
        timeframe="4h",
    )
    assert result.symbol == "ETHUSDT"
    assert result.timeframe == "4h"
    assert result.statistics.count == 50
    assert len(result.recommendations) == 3


def test_calibrate_aggregates_across_groups() -> None:
    """calibrate pools groups for global statistics and recommendations."""
    left = _prediction_frame([float(index) for index in range(0, 50)])
    right = _prediction_frame([float(index) for index in range(50, 100)])
    result = _calibrator().calibrate(
        {
            ("BTCUSDT", "1h"): left,
            ("ETHUSDT", "4h"): right,
        }
    )
    assert result.symbols_analyzed == ("BTCUSDT", "ETHUSDT")
    assert result.datasets_analyzed == 2
    assert result.rows_analyzed == 100
    assert result.global_statistics.count == 100
    assert result.global_statistics.minimum == pytest.approx(0.0)
    assert result.global_statistics.maximum == pytest.approx(99.0)
    assert len(result.symbol_timeframe_results) == 2
    assert result.symbol_timeframe_results[0].symbol == "BTCUSDT"
    assert result.symbol_timeframe_results[1].symbol == "ETHUSDT"


def test_calibrate_accepts_sequence_of_triples() -> None:
    """calibrate accepts an ordered sequence of group triples."""
    result = _calibrator().calibrate(
        (
            ("ETHUSDT", "1h", _uniform_frame(40)),
            ("BTCUSDT", "1h", _uniform_frame(40)),
        )
    )
    assert result.symbols_analyzed == ("BTCUSDT", "ETHUSDT")
    assert result.symbol_timeframe_results[0].symbol == "BTCUSDT"


# --- validation failures ---


def test_missing_prediction_column_raises() -> None:
    """Frames without a prediction column are rejected."""
    with pytest.raises(ResearchError, match="required column missing") as exc_info:
        _calibrator().analyze(pl.DataFrame({"value": [1.0, 2.0]}))
    assert exc_info.value.error_code == "RESEARCH-STC-002"


def test_empty_after_null_drop_raises() -> None:
    """All-null prediction columns are rejected."""
    with pytest.raises(ResearchError, match="no non-null observations") as exc_info:
        _calibrator().analyze(pl.DataFrame({"prediction": [None, None]}))
    assert exc_info.value.error_code == "RESEARCH-STC-003"


def test_non_finite_predictions_raise() -> None:
    """Infinite prediction values are rejected."""
    with pytest.raises(ResearchError, match="non-finite") as exc_info:
        _calibrator().analyze(pl.DataFrame({"prediction": [1.0, float("inf"), 2.0]}))
    assert exc_info.value.error_code == "RESEARCH-STC-004"


def test_constant_distribution_raises_for_threshold_order() -> None:
    """Constant predictions cannot produce distinct buy/sell thresholds."""
    frame = _prediction_frame([0.5] * 20)
    with pytest.raises(ResearchError, match="distinct buy/sell") as exc_info:
        _calibrator().recommend(frame)
    assert exc_info.value.error_code == "RESEARCH-STC-005"


def test_calibrate_with_no_groups_raises() -> None:
    """Empty group collections are rejected."""
    with pytest.raises(ResearchError, match="no prediction groups") as exc_info:
        _calibrator().calibrate({})
    assert exc_info.value.error_code == "RESEARCH-STC-006"


def test_non_dataframe_raises() -> None:
    """Non-DataFrame inputs are rejected."""
    with pytest.raises(ResearchError, match="polars DataFrame") as exc_info:
        _calibrator().analyze([1.0, 2.0])  # type: ignore[arg-type]
    assert exc_info.value.error_code == "RESEARCH-STC-001"
