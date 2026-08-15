"""Unit tests for CQROS ``TrendAngleFactor``."""

from __future__ import annotations

import math

import polars as pl
import pytest

from cqros.factors.price import TrendAngleFactor
from cqros.factors.price.trend_angle import TrendAngleFactor as TrendAngleFactorDirect
from tests.unit.factors.price._helpers import (
    assert_determinism,
    assert_empty_and_single_row,
    assert_lookback_below_two_raises,
    assert_lookback_negative_raises,
    assert_missing_close_raises,
    assert_output_float64_nullable,
    assert_preserves_columns,
    assert_protocol_and_immutability,
)


def _factor(*, lookback: int = 20) -> TrendAngleFactor:
    """Build a trend angle factor with an optional lookback override."""
    return TrendAngleFactor(lookback=lookback)


def _ols_angle_degrees(closes: list[float]) -> float:
    """Return atan(OLS slope) in degrees for log(close)."""
    ys = [math.log(value) for value in closes]
    n = len(ys)
    xs = list(range(n))
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xy = sum(x * y for x, y in zip(xs, ys, strict=True))
    sum_x2 = sum(x * x for x in xs)
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    return math.degrees(math.atan(slope))


def test_trend_angle_metadata() -> None:
    """TrendAngleFactor exposes the fixed production metadata contract."""
    factor = _factor()
    assert factor.name == "trend_angle"
    assert factor.version == "1.0.0"
    assert factor.category == "price"
    assert factor.required_features == ("close",)
    assert factor.produced_columns == ("trend_angle",)
    assert factor.lookback == 20
    assert factor.metadata.name == "trend_angle"


def test_trend_angle_calculation_correctness() -> None:
    """Trend angle matches atan(slope) in degrees."""
    closes = [100.0, 102.0, 101.0, 105.0, 110.0]
    values = (
        _factor(lookback=3)
        .compute(pl.DataFrame({"close": closes}))
        .get_column("trend_angle")
        .to_list()
    )
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx(_ols_angle_degrees(closes[0:3]))
    assert values[3] == pytest.approx(_ols_angle_degrees(closes[1:4]))
    assert values[4] == pytest.approx(_ols_angle_degrees(closes[2:5]))


def test_increasing_prices_positive_angle() -> None:
    """Increasing prices yield a positive angle after warm-up."""
    frame = pl.DataFrame({"close": [100.0, 110.0, 121.0, 133.0]})
    values = _factor(lookback=3).compute(frame).get_column("trend_angle").to_list()
    assert values[3] is not None
    assert values[3] > 0.0


def test_decreasing_prices_negative_angle() -> None:
    """Decreasing prices yield a negative angle after warm-up."""
    frame = pl.DataFrame({"close": [133.0, 121.0, 110.0, 100.0]})
    values = _factor(lookback=3).compute(frame).get_column("trend_angle").to_list()
    assert values[3] is not None
    assert values[3] < 0.0


def test_constant_prices_zero_angle() -> None:
    """Constant prices yield a zero angle after warm-up."""
    frame = pl.DataFrame({"close": [10.0, 10.0, 10.0, 10.0]})
    values = _factor(lookback=3).compute(frame).get_column("trend_angle").to_list()
    assert values[3] == pytest.approx(0.0)


def test_validation_immutability_schema_and_exports() -> None:
    """Validation, schema, determinism, and package export contracts."""
    assert_lookback_below_two_raises(
        lambda value: _factor(lookback=value),
        error_code="FACTOR-TREND-ANGLE-001",
    )
    assert_lookback_negative_raises(lambda value: _factor(lookback=value))
    assert_missing_close_raises(
        lambda: _factor(lookback=2),
        error_code="FACTOR-TREND-ANGLE-002",
        factor_name="trend_angle",
    )
    factor = _factor(lookback=2)
    assert_protocol_and_immutability(factor, output_column="trend_angle")
    assert_preserves_columns(factor, output_column="trend_angle")
    assert_output_float64_nullable(factor, output_column="trend_angle")
    assert_empty_and_single_row(lambda: _factor(lookback=2), output_column="trend_angle")
    assert_determinism(
        lambda: _factor(lookback=3),
        output_column="trend_angle",
        frame=pl.DataFrame({"close": [100.0, 102.0, 101.0, 105.0]}),
    )
    assert TrendAngleFactor is TrendAngleFactorDirect
