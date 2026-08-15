"""Unit tests for CQROS ``RegressionResidualZScoreFactor``."""

from __future__ import annotations

import math

import polars as pl
import pytest

from cqros.factors.price import RegressionResidualZScoreFactor
from cqros.factors.price.regression_residual_zscore import (
    RegressionResidualZScoreFactor as RegressionResidualZScoreFactorDirect,
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


def _factor(*, lookback: int = 20) -> RegressionResidualZScoreFactor:
    """Build a regression residual z-score factor with optional lookback."""
    return RegressionResidualZScoreFactor(lookback=lookback)


def _ols_residual_zscore(closes: list[float]) -> float | None:
    """Return end-of-window residual z-score for log(close) OLS."""
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
    mean_y = sum_y / n
    residual = ys[-1] - mean_y - slope * ((n - 1) / 2.0)
    ss_xx = denom / n
    ss_yy = sum_y2 - (sum_y * sum_y) / n
    ss_res = ss_yy - (slope * slope) * ss_xx
    if ss_res <= 0.0:
        return 0.0
    sigma = math.sqrt(ss_res / (n - 1))
    return residual / sigma


def test_regression_residual_zscore_metadata() -> None:
    """RegressionResidualZScoreFactor exposes the fixed metadata contract."""
    factor = _factor()
    assert factor.name == "regression_residual_zscore"
    assert factor.version == "1.0.0"
    assert factor.category == "price"
    assert factor.required_features == ("close",)
    assert factor.produced_columns == ("regression_residual_zscore",)
    assert factor.lookback == 20
    assert factor.metadata.name == "regression_residual_zscore"


def test_regression_residual_zscore_calculation_correctness() -> None:
    """Z-score matches residual divided by in-window residual std."""
    closes = [100.0, 102.0, 101.0, 105.0, 110.0]
    values = (
        _factor(lookback=3)
        .compute(pl.DataFrame({"close": closes}))
        .get_column("regression_residual_zscore")
        .to_list()
    )
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx(_ols_residual_zscore(closes[0:3]))
    assert values[3] == pytest.approx(_ols_residual_zscore(closes[1:4]))
    assert values[4] == pytest.approx(_ols_residual_zscore(closes[2:5]))


def test_perfect_fit_zscore_is_zero() -> None:
    """Perfect fit has zero residual variance and z-score 0.0."""
    closes = [math.exp(0.0), math.exp(0.1), math.exp(0.2), math.exp(0.3)]
    values = (
        _factor(lookback=4)
        .compute(pl.DataFrame({"close": closes}))
        .get_column("regression_residual_zscore")
        .to_list()
    )
    assert values[3] == pytest.approx(0.0)


def test_constant_prices_zscore_is_zero() -> None:
    """Constant prices yield residual z-score 0.0 after warmup."""
    frame = pl.DataFrame({"close": [10.0, 10.0, 10.0, 10.0]})
    values = _factor(lookback=3).compute(frame).get_column("regression_residual_zscore").to_list()
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx(0.0)
    assert values[3] == pytest.approx(0.0)


def test_validation_immutability_schema_and_exports() -> None:
    """Validation, schema, determinism, and package export contracts."""
    assert_lookback_below_two_raises(
        lambda value: _factor(lookback=value),
        error_code="FACTOR-REGRESSION-RESIDUAL-ZSCORE-001",
    )
    assert_lookback_negative_raises(lambda value: _factor(lookback=value))
    assert_missing_close_raises(
        lambda: _factor(lookback=2),
        error_code="FACTOR-REGRESSION-RESIDUAL-ZSCORE-002",
        factor_name="regression_residual_zscore",
    )
    factor = _factor(lookback=2)
    assert_protocol_and_immutability(factor, output_column="regression_residual_zscore")
    assert_preserves_columns(factor, output_column="regression_residual_zscore")
    assert_output_float64_nullable(factor, output_column="regression_residual_zscore")
    assert_empty_and_single_row(
        lambda: _factor(lookback=2),
        output_column="regression_residual_zscore",
    )
    assert_determinism(
        lambda: _factor(lookback=3),
        output_column="regression_residual_zscore",
        frame=pl.DataFrame({"close": [100.0, 102.0, 101.0, 105.0]}),
    )
    assert RegressionResidualZScoreFactor is RegressionResidualZScoreFactorDirect
