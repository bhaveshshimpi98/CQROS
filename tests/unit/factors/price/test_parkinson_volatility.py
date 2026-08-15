"""Unit tests for CQROS ``ParkinsonVolatilityFactor``."""

from __future__ import annotations

import math

import polars as pl
import pytest

from cqros.core.exceptions import ValidationError
from cqros.factors.exceptions import FactorError
from cqros.factors.price import ParkinsonVolatilityFactor
from cqros.factors.price.parkinson_volatility import (
    ParkinsonVolatilityFactor as ParkinsonVolatilityFactorDirect,
)
from tests.unit.factors.price._helpers import (
    assert_determinism,
    assert_empty_and_single_row,
    assert_lookback_below_two_raises,
    assert_lookback_negative_raises,
    assert_output_float64_nullable,
    assert_preserves_columns,
    assert_protocol_and_immutability,
)


def _factor(*, lookback: int = 20) -> ParkinsonVolatilityFactor:
    """Build a Parkinson volatility factor with an optional lookback override."""
    return ParkinsonVolatilityFactor(lookback=lookback)


def _hl_frame() -> pl.DataFrame:
    """Return a small high-low fixture for Parkinson tests."""
    return pl.DataFrame(
        {
            "high": [11.0, 13.0, 12.0, 15.0, 14.0],
            "low": [9.0, 10.0, 8.0, 11.0, 10.0],
            "volume": [1, 2, 3, 4, 5],
        }
    )


def _parkinson(highs: list[float], lows: list[float], *, lookback: int, index: int) -> float | None:
    """Return Parkinson volatility at ``index`` for a fully observed window."""
    if index < lookback - 1:
        return None
    window_high = highs[index - lookback + 1 : index + 1]
    window_low = lows[index - lookback + 1 : index + 1]
    mean_log_hl_sq = (
        sum(math.log(high / low) ** 2 for high, low in zip(window_high, window_low, strict=True))
        / lookback
    )
    return math.sqrt(mean_log_hl_sq / (4.0 * math.log(2.0)))


def test_parkinson_volatility_metadata() -> None:
    """ParkinsonVolatilityFactor exposes the fixed production metadata contract."""
    factor = _factor()
    assert factor.name == "parkinson_volatility"
    assert factor.version == "1.0.0"
    assert factor.category == "price"
    assert factor.required_features == ("high", "low")
    assert factor.produced_columns == ("parkinson_volatility",)
    assert factor.lookback == 20
    assert factor.metadata.required_features == ("high", "low")


def test_parkinson_volatility_calculation_correctness() -> None:
    """Parkinson volatility matches the high-low estimator."""
    frame = _hl_frame()
    highs = frame.get_column("high").to_list()
    lows = frame.get_column("low").to_list()
    values = _factor(lookback=3).compute(frame).get_column("parkinson_volatility").to_list()
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx(_parkinson(highs, lows, lookback=3, index=2))
    assert values[3] == pytest.approx(_parkinson(highs, lows, lookback=3, index=3))
    assert values[4] == pytest.approx(_parkinson(highs, lows, lookback=3, index=4))


def test_constant_prices_zero_volatility() -> None:
    """Equal high and low produce zero Parkinson volatility."""
    frame = pl.DataFrame({"high": [10.0, 10.0, 10.0], "low": [10.0, 10.0, 10.0]})
    values = _factor(lookback=2).compute(frame).get_column("parkinson_volatility").to_list()
    assert values[1] == pytest.approx(0.0)
    assert values[2] == pytest.approx(0.0)


def test_increasing_and_decreasing_ranges() -> None:
    """Positive high-low ranges produce positive Parkinson volatility."""
    rising = pl.DataFrame({"high": [2.0, 3.0, 4.0, 5.0], "low": [1.0, 2.0, 3.0, 4.0]})
    falling = pl.DataFrame({"high": [5.0, 4.0, 3.0, 2.0], "low": [4.0, 3.0, 2.0, 1.0]})
    assert _factor(lookback=2).compute(rising).get_column("parkinson_volatility").to_list()[3] > 0.0
    assert (
        _factor(lookback=2).compute(falling).get_column("parkinson_volatility").to_list()[3] > 0.0
    )


def test_zero_low_propagates_null() -> None:
    """Zero low makes the log ratio undefined and output null."""
    frame = pl.DataFrame({"high": [1.0, 2.0, 3.0], "low": [0.5, 0.0, 1.0]})
    values = _factor(lookback=2).compute(frame).get_column("parkinson_volatility").to_list()
    assert values[1] is None
    assert values[2] is None


def test_missing_required_columns_raise() -> None:
    """Missing high or low raises FactorError."""
    with pytest.raises(FactorError, match="required column missing: high") as exc_info:
        _factor(lookback=2).compute(pl.DataFrame({"low": [1.0]}))
    assert exc_info.value.error_code == "FACTOR-PARKINSON-VOLATILITY-002"

    with pytest.raises(FactorError, match="required column missing: low") as exc_info:
        _factor(lookback=2).compute(pl.DataFrame({"high": [1.0]}))
    assert exc_info.value.error_code == "FACTOR-PARKINSON-VOLATILITY-002"
    assert exc_info.value.details["factor"] == "parkinson_volatility"


def test_validation_immutability_schema_and_exports() -> None:
    """Validation, schema, determinism, and package export contracts."""
    assert_lookback_below_two_raises(
        lambda value: _factor(lookback=value),
        error_code="FACTOR-PARKINSON-VOLATILITY-001",
    )
    assert_lookback_negative_raises(lambda value: _factor(lookback=value))
    with pytest.raises(ValidationError):
        ParkinsonVolatilityFactor(lookback=0)

    factor = _factor(lookback=2)
    frame = _hl_frame()
    assert_protocol_and_immutability(factor, output_column="parkinson_volatility", frame=frame)
    assert_preserves_columns(
        factor,
        output_column="parkinson_volatility",
        frame=frame,
    )
    assert_output_float64_nullable(factor, output_column="parkinson_volatility", frame=frame)
    assert_empty_and_single_row(
        lambda: _factor(lookback=2),
        output_column="parkinson_volatility",
        columns={"high": [], "low": []},
    )
    assert_determinism(
        lambda: _factor(lookback=3),
        output_column="parkinson_volatility",
        frame=frame,
    )
    large = _factor(lookback=50).compute(frame)
    assert all(value is None for value in large.get_column("parkinson_volatility").to_list())
    assert ParkinsonVolatilityFactor is ParkinsonVolatilityFactorDirect
