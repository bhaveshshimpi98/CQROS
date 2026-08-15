"""Unit tests for CQROS ``LinearRegressionR2Factor``."""

from __future__ import annotations

import math

import polars as pl
import pytest

from cqros.factors.price import LinearRegressionR2Factor
from cqros.factors.price.linear_regression_r2 import (
    LinearRegressionR2Factor as LinearRegressionR2FactorDirect,
)
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


def _factor(*, lookback: int = 20) -> LinearRegressionR2Factor:
    """Build a linear regression R² factor with an optional lookback override."""
    return LinearRegressionR2Factor(lookback=lookback)


def _ols_r2(closes: list[float]) -> float | None:
    """Return OLS R² of log(close) on relative index 0..n-1."""
    ys = [math.log(value) for value in closes]
    n = len(ys)
    xs = list(range(n))
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xy = sum(x * y for x, y in zip(xs, ys, strict=True))
    sum_x2 = sum(x * x for x in xs)
    sum_y2 = sum(y * y for y in ys)
    denom = n * sum_x2 - sum_x * sum_x
    slope = (n * sum_xy - sum_x * sum_y) / denom
    ss_xx = denom / n
    ss_yy = sum_y2 - (sum_y * sum_y) / n
    if ss_yy <= 0.0:
        return None
    return (slope * slope) * ss_xx / ss_yy


def test_linear_regression_r2_metadata() -> None:
    """LinearRegressionR2Factor exposes the fixed production metadata contract."""
    factor = _factor()
    assert factor.name == "linear_regression_r2"
    assert factor.version == "1.0.0"
    assert factor.category == "price"
    assert factor.required_features == ("close",)
    assert factor.produced_columns == ("linear_regression_r2",)
    assert factor.lookback == 20
    assert factor.metadata.produced_columns == ("linear_regression_r2",)


def test_linear_regression_r2_calculation_correctness() -> None:
    """R² matches rolling OLS coefficient of determination on log(close)."""
    closes = [100.0, 102.0, 101.0, 105.0, 110.0]
    values = (
        _factor(lookback=3)
        .compute(pl.DataFrame({"close": closes}))
        .get_column("linear_regression_r2")
        .to_list()
    )
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx(_ols_r2(closes[0:3]))
    assert values[3] == pytest.approx(_ols_r2(closes[1:4]))
    assert values[4] == pytest.approx(_ols_r2(closes[2:5]))


def test_perfect_log_linear_trend_is_one() -> None:
    """Exact exponential growth yields R² near 1."""
    closes = [math.exp(0.0), math.exp(0.1), math.exp(0.2), math.exp(0.3)]
    values = (
        _factor(lookback=4)
        .compute(pl.DataFrame({"close": closes}))
        .get_column("linear_regression_r2")
        .to_list()
    )
    assert values[3] == pytest.approx(1.0)


def test_constant_prices_r2_is_null() -> None:
    """Constant prices make total sum of squares zero and R² null."""
    frame = pl.DataFrame({"close": [10.0, 10.0, 10.0, 10.0]})
    values = _factor(lookback=3).compute(frame).get_column("linear_regression_r2").to_list()
    assert values[3] is None


def test_validation_immutability_schema_and_exports() -> None:
    """Validation, schema, determinism, and package export contracts."""
    assert_lookback_below_two_raises(
        lambda value: _factor(lookback=value),
        error_code="FACTOR-LINEAR-REGRESSION-R2-001",
    )
    assert_lookback_negative_raises(lambda value: _factor(lookback=value))
    assert_missing_close_raises(
        lambda: _factor(lookback=2),
        error_code="FACTOR-LINEAR-REGRESSION-R2-002",
        factor_name="linear_regression_r2",
    )
    factor = _factor(lookback=2)
    assert_protocol_and_immutability(factor, output_column="linear_regression_r2")
    assert_preserves_columns(factor, output_column="linear_regression_r2")
    assert_output_float64_nullable(factor, output_column="linear_regression_r2")
    assert_empty_and_single_row(
        lambda: _factor(lookback=2),
        output_column="linear_regression_r2",
    )
    assert_determinism(
        lambda: _factor(lookback=3),
        output_column="linear_regression_r2",
        frame=pl.DataFrame({"close": [100.0, 102.0, 101.0, 105.0]}),
    )
    assert LinearRegressionR2Factor is LinearRegressionR2FactorDirect
