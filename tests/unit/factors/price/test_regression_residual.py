"""Unit tests for CQROS ``RegressionResidualFactor``."""

from __future__ import annotations

import math

import polars as pl
import pytest

from cqros.factors.price import RegressionResidualFactor
from cqros.factors.price.regression_residual import (
    RegressionResidualFactor as RegressionResidualFactorDirect,
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


def _factor(*, lookback: int = 20) -> RegressionResidualFactor:
    """Build a regression residual factor with an optional lookback override."""
    return RegressionResidualFactor(lookback=lookback)


def _ols_residual(closes: list[float]) -> float:
    """Return end-of-window OLS residual of log(close)."""
    ys = [math.log(value) for value in closes]
    n = len(ys)
    xs = list(range(n))
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xy = sum(x * y for x, y in zip(xs, ys, strict=True))
    sum_x2 = sum(x * x for x in xs)
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    mean_y = sum_y / n
    return ys[-1] - mean_y - slope * ((n - 1) / 2.0)


def test_regression_residual_metadata() -> None:
    """RegressionResidualFactor exposes the fixed production metadata contract."""
    factor = _factor()
    assert factor.name == "regression_residual"
    assert factor.version == "1.0.0"
    assert factor.category == "price"
    assert factor.required_features == ("close",)
    assert factor.produced_columns == ("regression_residual",)
    assert factor.lookback == 20
    assert factor.metadata.name == "regression_residual"


def test_regression_residual_calculation_correctness() -> None:
    """Residual matches current log(close) minus rolling fitted value."""
    closes = [100.0, 102.0, 101.0, 105.0, 110.0]
    values = (
        _factor(lookback=3)
        .compute(pl.DataFrame({"close": closes}))
        .get_column("regression_residual")
        .to_list()
    )
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx(_ols_residual(closes[0:3]))
    assert values[3] == pytest.approx(_ols_residual(closes[1:4]))
    assert values[4] == pytest.approx(_ols_residual(closes[2:5]))


def test_perfect_fit_residual_near_zero() -> None:
    """Exact log-linear series has near-zero end residual."""
    closes = [math.exp(0.0), math.exp(0.1), math.exp(0.2), math.exp(0.3)]
    values = (
        _factor(lookback=4)
        .compute(pl.DataFrame({"close": closes}))
        .get_column("regression_residual")
        .to_list()
    )
    assert values[3] == pytest.approx(0.0, abs=1e-12)


def test_constant_prices_residual_zero() -> None:
    """Constant prices yield a zero residual after warm-up."""
    frame = pl.DataFrame({"close": [10.0, 10.0, 10.0, 10.0]})
    values = _factor(lookback=3).compute(frame).get_column("regression_residual").to_list()
    assert values[3] == pytest.approx(0.0)


def test_validation_immutability_schema_and_exports() -> None:
    """Validation, schema, determinism, and package export contracts."""
    assert_lookback_below_two_raises(
        lambda value: _factor(lookback=value),
        error_code="FACTOR-REGRESSION-RESIDUAL-001",
    )
    assert_lookback_negative_raises(lambda value: _factor(lookback=value))
    assert_missing_close_raises(
        lambda: _factor(lookback=2),
        error_code="FACTOR-REGRESSION-RESIDUAL-002",
        factor_name="regression_residual",
    )
    factor = _factor(lookback=2)
    assert_protocol_and_immutability(factor, output_column="regression_residual")
    assert_preserves_columns(factor, output_column="regression_residual")
    assert_output_float64_nullable(factor, output_column="regression_residual")
    assert_empty_and_single_row(
        lambda: _factor(lookback=2),
        output_column="regression_residual",
    )
    assert_determinism(
        lambda: _factor(lookback=3),
        output_column="regression_residual",
        frame=pl.DataFrame({"close": [100.0, 102.0, 101.0, 105.0]}),
    )
    assert RegressionResidualFactor is RegressionResidualFactorDirect
