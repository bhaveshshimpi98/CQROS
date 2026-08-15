"""Unit tests for CQROS ``StochasticKFactor``."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.core.exceptions import ValidationError
from cqros.factors.exceptions import FactorError
from cqros.factors.price import StochasticKFactor
from cqros.factors.price.stochastic_k import StochasticKFactor as StochasticKFactorDirect
from tests.unit.factors.price._helpers import (
    assert_determinism,
    assert_empty_and_single_row,
    assert_lookback_below_two_raises,
    assert_lookback_negative_raises,
    assert_output_float64_nullable,
    assert_preserves_columns,
    assert_protocol_and_immutability,
)


def _factor(*, lookback: int = 14) -> StochasticKFactor:
    """Build a Stochastic %K factor with an optional lookback override."""
    return StochasticKFactor(lookback=lookback)


def _ohlc_frame() -> pl.DataFrame:
    """Return a small OHLC fixture for Stochastic %K tests."""
    return pl.DataFrame(
        {
            "high": [11.0, 13.0, 12.0, 15.0, 14.0],
            "low": [9.0, 10.0, 8.0, 11.0, 10.0],
            "close": [10.0, 12.0, 9.0, 14.0, 13.0],
            "volume": [1, 2, 3, 4, 5],
        }
    )


def test_stochastic_k_metadata() -> None:
    """StochasticKFactor exposes the fixed production metadata contract."""
    factor = _factor()
    assert factor.name == "stochastic_k"
    assert factor.version == "1.0.0"
    assert factor.category == "price"
    assert factor.required_features == ("high", "low", "close")
    assert factor.produced_columns == ("stochastic_k",)
    assert factor.lookback == 14
    assert factor.metadata.name == "stochastic_k"


def test_stochastic_k_calculation_correctness() -> None:
    """Stochastic %K matches 100 * (close - low_min) / (high_max - low_min)."""
    frame = _ohlc_frame()
    values = _factor(lookback=3).compute(frame).get_column("stochastic_k").to_list()
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx(100.0 * (9.0 - 8.0) / (13.0 - 8.0))
    assert values[3] == pytest.approx(100.0 * (14.0 - 8.0) / (15.0 - 8.0))
    assert values[4] == pytest.approx(100.0 * (13.0 - 8.0) / (15.0 - 8.0))


def test_zero_range_is_null() -> None:
    """Zero high-low range returns null Stochastic %K."""
    frame = pl.DataFrame(
        {
            "high": [10.0, 10.0, 10.0],
            "low": [10.0, 10.0, 10.0],
            "close": [10.0, 10.0, 10.0],
        }
    )
    values = _factor(lookback=3).compute(frame).get_column("stochastic_k").to_list()
    assert values[2] is None


def test_increasing_and_decreasing_extremes() -> None:
    """Close at high yields 100 and close at low yields 0."""
    high_frame = pl.DataFrame(
        {
            "high": [12.0, 14.0, 16.0],
            "low": [10.0, 11.0, 12.0],
            "close": [10.0, 12.0, 16.0],
        }
    )
    assert _factor(lookback=3).compute(high_frame).get_column("stochastic_k").to_list()[
        2
    ] == pytest.approx(100.0)

    low_frame = pl.DataFrame(
        {
            "high": [12.0, 14.0, 16.0],
            "low": [10.0, 11.0, 9.0],
            "close": [11.0, 12.0, 9.0],
        }
    )
    assert _factor(lookback=3).compute(low_frame).get_column("stochastic_k").to_list()[
        2
    ] == pytest.approx(0.0)


def test_missing_required_columns_raise() -> None:
    """Missing high, low, or close raises FactorError."""
    with pytest.raises(FactorError, match="required column missing: high") as exc_info:
        _factor(lookback=2).compute(pl.DataFrame({"low": [1.0], "close": [1.0]}))
    assert exc_info.value.error_code == "FACTOR-STOCHASTIC-K-002"

    with pytest.raises(FactorError, match="required column missing: low") as exc_info:
        _factor(lookback=2).compute(pl.DataFrame({"high": [1.0], "close": [1.0]}))
    assert exc_info.value.error_code == "FACTOR-STOCHASTIC-K-002"

    with pytest.raises(FactorError, match="required column missing: close") as exc_info:
        _factor(lookback=2).compute(pl.DataFrame({"high": [1.0], "low": [1.0]}))
    assert exc_info.value.error_code == "FACTOR-STOCHASTIC-K-002"


def test_validation_immutability_schema_and_exports() -> None:
    """Validation, schema, determinism, and package export contracts."""
    assert_lookback_below_two_raises(
        lambda value: _factor(lookback=value),
        error_code="FACTOR-STOCHASTIC-K-001",
    )
    assert_lookback_negative_raises(lambda value: _factor(lookback=value))
    with pytest.raises(ValidationError):
        StochasticKFactor(lookback=0)

    factor = _factor(lookback=2)
    frame = _ohlc_frame()
    assert_protocol_and_immutability(factor, output_column="stochastic_k", frame=frame)
    assert_preserves_columns(
        factor,
        output_column="stochastic_k",
        frame=frame.select(["high", "low", "close", "volume"]),
    )
    assert_output_float64_nullable(factor, output_column="stochastic_k")
    assert_empty_and_single_row(
        lambda: _factor(lookback=2),
        output_column="stochastic_k",
        columns={"high": [], "low": [], "close": []},
    )
    assert_determinism(
        lambda: _factor(lookback=3),
        output_column="stochastic_k",
        frame=frame,
    )
    large = _factor(lookback=50).compute(
        pl.DataFrame(
            {
                "high": [float(i + 1) for i in range(10)],
                "low": [float(i) for i in range(10)],
                "close": [float(i) + 0.5 for i in range(10)],
            }
        )
    )
    assert all(value is None for value in large.get_column("stochastic_k").to_list())
    assert StochasticKFactor is StochasticKFactorDirect
