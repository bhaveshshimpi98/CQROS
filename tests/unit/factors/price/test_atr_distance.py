"""Unit tests for CQROS ``ATRDistanceFactor``."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.core.exceptions import ValidationError
from cqros.factors.exceptions import FactorError
from cqros.factors.price import ATRDistanceFactor
from cqros.factors.price.atr_distance import ATRDistanceFactor as ATRDistanceFactorDirect
from tests.unit.factors.price._helpers import (
    assert_determinism,
    assert_empty_and_single_row,
    assert_lookback_below_two_raises,
    assert_lookback_negative_raises,
    assert_output_float64_nullable,
    assert_preserves_columns,
    assert_protocol_and_immutability,
)


def _factor(*, lookback: int = 20) -> ATRDistanceFactor:
    """Build an ATR distance factor with an optional lookback override."""
    return ATRDistanceFactor(lookback=lookback)


def _ohlc_frame() -> pl.DataFrame:
    """Return a small OHLC fixture for ATR tests."""
    return pl.DataFrame(
        {
            "high": [12.0, 15.0, 14.0, 16.0, 17.0],
            "low": [10.0, 11.0, 12.0, 13.0, 14.0],
            "close": [11.0, 14.0, 13.0, 15.0, 16.0],
            "volume": [1, 2, 3, 4, 5],
        }
    )


def _true_ranges(highs: list[float], lows: list[float], closes: list[float]) -> list[float]:
    """Return true range series matching CQROS ATRFeature semantics."""
    values: list[float] = []
    for index, (high, low, close) in enumerate(zip(highs, lows, closes, strict=True)):
        candidates = [high - low]
        if index > 0:
            prev_close = closes[index - 1]
            candidates.append(abs(high - prev_close))
            candidates.append(abs(low - prev_close))
        values.append(max(candidates))
        _ = close
    return values


def test_atr_distance_metadata() -> None:
    """ATRDistanceFactor exposes the fixed production metadata contract."""
    factor = _factor()
    assert factor.name == "atr_distance"
    assert factor.version == "1.0.0"
    assert factor.category == "price"
    assert factor.required_features == ("high", "low", "close")
    assert factor.produced_columns == ("atr_distance",)
    assert factor.lookback == 20
    assert factor.metadata.required_features == ("high", "low", "close")


def test_atr_distance_calculation_correctness() -> None:
    """ATR distance matches close minus rolling mean of true range."""
    frame = _ohlc_frame()
    highs = frame.get_column("high").to_list()
    lows = frame.get_column("low").to_list()
    closes = frame.get_column("close").to_list()
    true_ranges = _true_ranges(highs, lows, closes)
    values = _factor(lookback=2).compute(frame).get_column("atr_distance").to_list()
    assert values[0] is None
    assert values[1] == pytest.approx(closes[1] - (true_ranges[0] + true_ranges[1]) / 2.0)
    assert values[2] == pytest.approx(closes[2] - (true_ranges[1] + true_ranges[2]) / 2.0)
    assert values[3] == pytest.approx(closes[3] - (true_ranges[2] + true_ranges[3]) / 2.0)


def test_constant_prices_distance_equals_close() -> None:
    """Constant OHLC makes ATR zero and distance equal to close."""
    frame = pl.DataFrame(
        {
            "high": [10.0, 10.0, 10.0],
            "low": [10.0, 10.0, 10.0],
            "close": [10.0, 10.0, 10.0],
        }
    )
    values = _factor(lookback=2).compute(frame).get_column("atr_distance").to_list()
    assert values[1] == pytest.approx(10.0)
    assert values[2] == pytest.approx(10.0)


def test_increasing_and_decreasing_prices() -> None:
    """Rising and falling OHLC produce finite ATR distance values."""
    rising = pl.DataFrame(
        {
            "high": [2.0, 3.0, 4.0, 5.0],
            "low": [1.0, 2.0, 3.0, 4.0],
            "close": [1.5, 2.5, 3.5, 4.5],
        }
    )
    falling = pl.DataFrame(
        {
            "high": [5.0, 4.0, 3.0, 2.0],
            "low": [4.0, 3.0, 2.0, 1.0],
            "close": [4.5, 3.5, 2.5, 1.5],
        }
    )
    assert _factor(lookback=2).compute(rising).get_column("atr_distance").to_list()[3] is not None
    assert _factor(lookback=2).compute(falling).get_column("atr_distance").to_list()[3] is not None


def test_missing_required_columns_raise() -> None:
    """Missing high, low, or close raises FactorError."""
    with pytest.raises(FactorError, match="required column missing: high") as exc_info:
        _factor(lookback=2).compute(pl.DataFrame({"low": [1.0], "close": [1.0]}))
    assert exc_info.value.error_code == "FACTOR-ATR-DISTANCE-002"

    with pytest.raises(FactorError, match="required column missing: low") as exc_info:
        _factor(lookback=2).compute(pl.DataFrame({"high": [1.0], "close": [1.0]}))
    assert exc_info.value.error_code == "FACTOR-ATR-DISTANCE-002"

    with pytest.raises(FactorError, match="required column missing: close") as exc_info:
        _factor(lookback=2).compute(pl.DataFrame({"high": [1.0], "low": [1.0]}))
    assert exc_info.value.error_code == "FACTOR-ATR-DISTANCE-002"
    assert exc_info.value.details["factor"] == "atr_distance"


def test_validation_immutability_schema_and_exports() -> None:
    """Validation, schema, determinism, and package export contracts."""
    assert_lookback_below_two_raises(
        lambda value: _factor(lookback=value),
        error_code="FACTOR-ATR-DISTANCE-001",
    )
    assert_lookback_negative_raises(lambda value: _factor(lookback=value))
    with pytest.raises(ValidationError):
        ATRDistanceFactor(lookback=0)

    factor = _factor(lookback=2)
    frame = _ohlc_frame()
    assert_protocol_and_immutability(factor, output_column="atr_distance", frame=frame)
    assert_preserves_columns(
        factor,
        output_column="atr_distance",
        frame=frame.select(["high", "low", "close", "volume"]),
    )
    assert_output_float64_nullable(factor, output_column="atr_distance")
    assert_empty_and_single_row(
        lambda: _factor(lookback=2),
        output_column="atr_distance",
        columns={"high": [], "low": [], "close": []},
    )
    assert_determinism(
        lambda: _factor(lookback=2),
        output_column="atr_distance",
        frame=frame,
    )
    large = _factor(lookback=50).compute(frame)
    assert all(value is None for value in large.get_column("atr_distance").to_list())
    assert ATRDistanceFactor is ATRDistanceFactorDirect
