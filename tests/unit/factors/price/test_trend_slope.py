"""Unit tests for CQROS ``TrendSlopeFactor``."""

from __future__ import annotations

import math

import polars as pl
import pytest

from cqros.factors.price import TrendSlopeFactor
from cqros.factors.price.trend_slope import TrendSlopeFactor as TrendSlopeFactorDirect
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


def _factor(*, lookback: int = 20) -> TrendSlopeFactor:
    """Build a trend slope factor with an optional lookback override."""
    return TrendSlopeFactor(lookback=lookback)


def _ols_slope(closes: list[float]) -> float:
    """Return OLS slope of log(close) on relative index 0..n-1."""
    ys = [math.log(value) for value in closes]
    n = len(ys)
    xs = list(range(n))
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xy = sum(x * y for x, y in zip(xs, ys, strict=True))
    sum_x2 = sum(x * x for x in xs)
    return (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)


def test_trend_slope_metadata() -> None:
    """TrendSlopeFactor exposes the fixed production metadata contract."""
    factor = _factor()
    assert factor.name == "trend_slope"
    assert factor.version == "1.0.0"
    assert factor.category == "price"
    assert factor.required_features == ("close",)
    assert factor.produced_columns == ("trend_slope",)
    assert factor.lookback == 20
    assert factor.metadata.name == "trend_slope"
    assert factor.metadata.lookback == 20


def test_trend_slope_calculation_correctness() -> None:
    """Trend slope matches rolling OLS slope of log(close)."""
    closes = [100.0, 102.0, 101.0, 105.0, 110.0]
    values = (
        _factor(lookback=3)
        .compute(pl.DataFrame({"close": closes}))
        .get_column("trend_slope")
        .to_list()
    )
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx(_ols_slope(closes[0:3]))
    assert values[3] == pytest.approx(_ols_slope(closes[1:4]))
    assert values[4] == pytest.approx(_ols_slope(closes[2:5]))


def test_increasing_prices_positive_slope() -> None:
    """Strictly increasing prices yield a positive slope after warm-up."""
    frame = pl.DataFrame({"close": [100.0, 110.0, 121.0, 133.0]})
    values = _factor(lookback=3).compute(frame).get_column("trend_slope").to_list()
    assert values[3] is not None
    assert values[3] > 0.0


def test_decreasing_prices_negative_slope() -> None:
    """Strictly decreasing prices yield a negative slope after warm-up."""
    frame = pl.DataFrame({"close": [133.0, 121.0, 110.0, 100.0]})
    values = _factor(lookback=3).compute(frame).get_column("trend_slope").to_list()
    assert values[3] is not None
    assert values[3] < 0.0


def test_constant_prices_zero_slope() -> None:
    """Constant prices yield a near-zero slope after warm-up."""
    frame = pl.DataFrame({"close": [10.0, 10.0, 10.0, 10.0]})
    values = _factor(lookback=3).compute(frame).get_column("trend_slope").to_list()
    assert values[3] == pytest.approx(0.0)


def test_null_close_propagates() -> None:
    """Null close values make windows containing them null."""
    frame = pl.DataFrame({"close": [10.0, None, 12.0, 13.0]})
    values = _factor(lookback=2).compute(frame).get_column("trend_slope").to_list()
    assert values[0] is None
    assert values[1] is None
    assert values[2] is None
    assert values[3] == pytest.approx(_ols_slope([12.0, 13.0]))


def test_large_lookback_all_null_when_insufficient_rows() -> None:
    """Lookback larger than the frame yields all-null output."""
    frame = pl.DataFrame({"close": [1.0, 2.0, 3.0]})
    values = _factor(lookback=10).compute(frame).get_column("trend_slope").to_list()
    assert values == [None, None, None]


def test_validation_immutability_schema_and_exports() -> None:
    """Validation, schema, determinism, and package export contracts."""
    assert_lookback_below_two_raises(
        lambda value: _factor(lookback=value),
        error_code="FACTOR-TREND-SLOPE-001",
        value=0,
    )
    assert_lookback_below_two_raises(
        lambda value: _factor(lookback=value),
        error_code="FACTOR-TREND-SLOPE-001",
        value=1,
    )
    assert_lookback_negative_raises(lambda value: _factor(lookback=value))
    assert_missing_close_raises(
        lambda: _factor(lookback=2),
        error_code="FACTOR-TREND-SLOPE-002",
        factor_name="trend_slope",
    )
    factor = _factor(lookback=2)
    assert_protocol_and_immutability(factor, output_column="trend_slope")
    assert_preserves_columns(factor, output_column="trend_slope")
    assert_output_float64_nullable(factor, output_column="trend_slope")
    assert_empty_and_single_row(lambda: _factor(lookback=2), output_column="trend_slope")
    assert_determinism(
        lambda: _factor(lookback=3),
        output_column="trend_slope",
        frame=pl.DataFrame({"close": [100.0, 102.0, 101.0, 105.0]}),
    )
    assert TrendSlopeFactor is TrendSlopeFactorDirect
