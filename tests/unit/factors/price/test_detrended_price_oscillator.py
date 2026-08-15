"""Unit tests for CQROS ``DetrendedPriceOscillatorFactor``."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.factors.price import DetrendedPriceOscillatorFactor
from cqros.factors.price.detrended_price_oscillator import (
    DetrendedPriceOscillatorFactor as DetrendedPriceOscillatorFactorDirect,
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


def _factor(*, lookback: int = 20) -> DetrendedPriceOscillatorFactor:
    """Build a DPO factor with an optional lookback override."""
    return DetrendedPriceOscillatorFactor(lookback=lookback)


def _expected_dpo(closes: list[float], *, lookback: int) -> list[float | None]:
    """Return DPO values for a close series."""
    displacement = lookback // 2 + 1
    output: list[float | None] = []
    for index in range(len(closes)):
        if index < lookback - 1 or index < displacement:
            output.append(None)
            continue
        window = closes[index - lookback + 1 : index + 1]
        sma = sum(window) / lookback
        output.append(closes[index - displacement] - sma)
    return output


def test_detrended_price_oscillator_metadata() -> None:
    """DetrendedPriceOscillatorFactor exposes the fixed production metadata contract."""
    factor = _factor()
    assert factor.name == "detrended_price_oscillator"
    assert factor.version == "1.0.0"
    assert factor.category == "price"
    assert factor.required_features == ("close",)
    assert factor.produced_columns == ("detrended_price_oscillator",)
    assert factor.lookback == 20
    assert factor.metadata.name == "detrended_price_oscillator"


def test_detrended_price_oscillator_calculation_correctness() -> None:
    """DPO matches displaced close minus the trailing SMA."""
    closes = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]
    values = (
        _factor(lookback=4)
        .compute(pl.DataFrame({"close": closes}))
        .get_column("detrended_price_oscillator")
        .to_list()
    )
    expected = _expected_dpo(closes, lookback=4)
    for value, expected_value in zip(values, expected, strict=True):
        if expected_value is None:
            assert value is None
        else:
            assert value == pytest.approx(expected_value)


def test_constant_prices_zero_dpo() -> None:
    """Constant prices yield zero DPO once both SMA and displacement are ready."""
    frame = pl.DataFrame({"close": [10.0] * 10})
    values = _factor(lookback=4).compute(frame).get_column("detrended_price_oscillator").to_list()
    displacement = 4 // 2 + 1
    first_valid = max(3, displacement)
    assert values[first_valid] == pytest.approx(0.0)


def test_increasing_and_decreasing_prices() -> None:
    """Rising and falling markets produce finite DPO values after warm-up."""
    rising = _factor(lookback=4).compute(pl.DataFrame({"close": [float(i) for i in range(1, 11)]}))
    falling = _factor(lookback=4).compute(
        pl.DataFrame({"close": [float(i) for i in range(10, 0, -1)]})
    )
    assert rising.get_column("detrended_price_oscillator").to_list()[8] is not None
    assert falling.get_column("detrended_price_oscillator").to_list()[8] is not None


def test_null_close_propagates() -> None:
    """Null close values leave incomplete DPO windows null."""
    frame = pl.DataFrame({"close": [10.0, None, 12.0, 13.0, 14.0, 15.0]})
    values = _factor(lookback=3).compute(frame).get_column("detrended_price_oscillator").to_list()
    assert values[0] is None
    assert values[1] is None
    assert values[2] is None


def test_validation_immutability_schema_and_exports() -> None:
    """Validation, schema, determinism, and package export contracts."""
    assert_lookback_below_two_raises(
        lambda value: _factor(lookback=value),
        error_code="FACTOR-DETRENDED-PRICE-OSCILLATOR-001",
    )
    assert_lookback_negative_raises(lambda value: _factor(lookback=value))
    assert_missing_close_raises(
        lambda: _factor(lookback=2),
        error_code="FACTOR-DETRENDED-PRICE-OSCILLATOR-002",
        factor_name="detrended_price_oscillator",
    )
    factor = _factor(lookback=2)
    assert_protocol_and_immutability(factor, output_column="detrended_price_oscillator")
    assert_preserves_columns(factor, output_column="detrended_price_oscillator")
    assert_output_float64_nullable(factor, output_column="detrended_price_oscillator")
    assert_empty_and_single_row(
        lambda: _factor(lookback=2),
        output_column="detrended_price_oscillator",
    )
    assert_determinism(
        lambda: _factor(lookback=4),
        output_column="detrended_price_oscillator",
        frame=pl.DataFrame({"close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]}),
    )
    large = _factor(lookback=50).compute(pl.DataFrame({"close": [float(i) for i in range(1, 11)]}))
    assert all(value is None for value in large.get_column("detrended_price_oscillator").to_list())
    assert DetrendedPriceOscillatorFactor is DetrendedPriceOscillatorFactorDirect
