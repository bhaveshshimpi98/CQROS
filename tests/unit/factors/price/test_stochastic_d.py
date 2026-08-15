"""Unit tests for CQROS ``StochasticDFactor``."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.core.exceptions import ValidationError
from cqros.factors.exceptions import FactorError
from cqros.factors.price import StochasticDFactor
from cqros.factors.price.stochastic_d import StochasticDFactor as StochasticDFactorDirect
from tests.unit.factors.price._helpers import (
    assert_determinism,
    assert_empty_and_single_row,
    assert_lookback_below_two_raises,
    assert_lookback_negative_raises,
    assert_output_float64_nullable,
    assert_preserves_columns,
    assert_protocol_and_immutability,
)


def _factor(*, lookback: int = 14, smooth: int = 3) -> StochasticDFactor:
    """Build a Stochastic %D factor with optional parameter overrides."""
    return StochasticDFactor(lookback=lookback, smooth=smooth)


def _ohlc_frame() -> pl.DataFrame:
    """Return a small OHLC fixture for Stochastic %D tests."""
    return pl.DataFrame(
        {
            "high": [11.0, 13.0, 12.0, 15.0, 14.0, 16.0],
            "low": [9.0, 10.0, 8.0, 11.0, 10.0, 12.0],
            "close": [10.0, 12.0, 9.0, 14.0, 13.0, 15.0],
            "volume": [1, 2, 3, 4, 5, 6],
        }
    )


def _stochastic_k(highs: list[float], lows: list[float], closes: list[float]) -> float | None:
    """Return Fast %K for a fully observed window."""
    highest = max(highs)
    lowest = min(lows)
    if highest == lowest:
        return None
    return 100.0 * (closes[-1] - lowest) / (highest - lowest)


def test_stochastic_d_metadata() -> None:
    """StochasticDFactor exposes the fixed production metadata contract."""
    factor = _factor()
    assert factor.name == "stochastic_d"
    assert factor.version == "1.0.0"
    assert factor.category == "price"
    assert factor.required_features == ("high", "low", "close")
    assert factor.produced_columns == ("stochastic_d",)
    assert factor.lookback == 14
    assert factor.smooth == 3
    assert factor.metadata.name == "stochastic_d"


def test_stochastic_d_calculation_correctness() -> None:
    """Stochastic %D matches the smooth-period mean of Fast %K."""
    frame = _ohlc_frame()
    values = _factor(lookback=3, smooth=2).compute(frame).get_column("stochastic_d").to_list()
    k_values = [
        _stochastic_k(
            frame["high"].to_list()[index - 2 : index + 1],
            frame["low"].to_list()[index - 2 : index + 1],
            frame["close"].to_list()[index - 2 : index + 1],
        )
        for index in range(2, frame.height)
    ]
    assert values[0] is None
    assert values[1] is None
    assert values[2] is None
    assert values[3] == pytest.approx((k_values[0] + k_values[1]) / 2.0)
    assert values[4] == pytest.approx((k_values[1] + k_values[2]) / 2.0)


def test_constant_prices_are_null() -> None:
    """Constant OHLC leaves %K undefined and %D null."""
    frame = pl.DataFrame(
        {
            "high": [10.0, 10.0, 10.0, 10.0],
            "low": [10.0, 10.0, 10.0, 10.0],
            "close": [10.0, 10.0, 10.0, 10.0],
        }
    )
    values = _factor(lookback=2, smooth=2).compute(frame).get_column("stochastic_d").to_list()
    assert all(value is None for value in values)


def test_increasing_and_decreasing_prices() -> None:
    """Rising and falling markets produce finite Stochastic %D values."""
    rising = pl.DataFrame(
        {
            "high": [2.0, 3.0, 4.0, 5.0, 6.0],
            "low": [1.0, 2.0, 3.0, 4.0, 5.0],
            "close": [1.5, 2.5, 3.5, 4.5, 5.5],
        }
    )
    falling = pl.DataFrame(
        {
            "high": [6.0, 5.0, 4.0, 3.0, 2.0],
            "low": [5.0, 4.0, 3.0, 2.0, 1.0],
            "close": [5.5, 4.5, 3.5, 2.5, 1.5],
        }
    )
    rising_values = (
        _factor(lookback=2, smooth=2).compute(rising).get_column("stochastic_d").to_list()
    )
    falling_values = (
        _factor(lookback=2, smooth=2).compute(falling).get_column("stochastic_d").to_list()
    )
    assert rising_values[3] is not None
    assert falling_values[3] is not None


def test_invalid_smooth_raises() -> None:
    """Non-positive smooth windows are rejected."""
    with pytest.raises(
        ValidationError,
        match="smooth must be an integer greater than or equal to 1",
    ) as exc_info:
        StochasticDFactor(lookback=14, smooth=0)
    assert exc_info.value.error_code == "FACTOR-STOCHASTIC-D-002"


def test_missing_required_columns_raise() -> None:
    """Missing high, low, or close raises FactorError."""
    with pytest.raises(FactorError, match="required column missing: high") as exc_info:
        _factor(lookback=2).compute(pl.DataFrame({"low": [1.0], "close": [1.0]}))
    assert exc_info.value.error_code == "FACTOR-STOCHASTIC-D-003"

    with pytest.raises(FactorError, match="required column missing: close") as exc_info:
        _factor(lookback=2).compute(pl.DataFrame({"high": [1.0], "low": [1.0]}))
    assert exc_info.value.error_code == "FACTOR-STOCHASTIC-D-003"


def test_validation_immutability_schema_and_exports() -> None:
    """Validation, schema, determinism, and package export contracts."""
    assert_lookback_below_two_raises(
        lambda value: _factor(lookback=value),
        error_code="FACTOR-STOCHASTIC-D-001",
    )
    assert_lookback_negative_raises(lambda value: _factor(lookback=value))
    factor = _factor(lookback=2, smooth=2)
    frame = _ohlc_frame()
    assert_protocol_and_immutability(factor, output_column="stochastic_d", frame=frame)
    assert_preserves_columns(
        factor,
        output_column="stochastic_d",
        frame=frame.select(["high", "low", "close", "volume"]),
    )
    assert_output_float64_nullable(factor, output_column="stochastic_d")
    assert_empty_and_single_row(
        lambda: _factor(lookback=2, smooth=2),
        output_column="stochastic_d",
        columns={"high": [], "low": [], "close": []},
    )
    assert_determinism(
        lambda: _factor(lookback=3, smooth=2),
        output_column="stochastic_d",
        frame=frame,
    )
    large = _factor(lookback=50, smooth=3).compute(
        pl.DataFrame(
            {
                "high": [float(i + 1) for i in range(10)],
                "low": [float(i) for i in range(10)],
                "close": [float(i) + 0.5 for i in range(10)],
            }
        )
    )
    assert all(value is None for value in large.get_column("stochastic_d").to_list())
    assert StochasticDFactor is StochasticDFactorDirect
