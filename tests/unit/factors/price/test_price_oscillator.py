"""Unit tests for CQROS ``PriceOscillatorFactor``."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.core.exceptions import ValidationError
from cqros.factors.price import PriceOscillatorFactor
from cqros.factors.price.price_oscillator import (
    PriceOscillatorFactor as PriceOscillatorFactorDirect,
)
from tests.unit.factors.price._helpers import (
    assert_determinism,
    assert_empty_and_single_row,
    assert_missing_close_raises,
    assert_output_float64_nullable,
    assert_preserves_columns,
    assert_protocol_and_immutability,
)


def _factor(*, fast_span: int = 12, slow_span: int = 26) -> PriceOscillatorFactor:
    """Build a PPO factor with optional span overrides."""
    return PriceOscillatorFactor(fast_span=fast_span, slow_span=slow_span)


def _expected_ppo(closes: list[float], *, fast_span: int, slow_span: int) -> list[float | None]:
    """Return PPO values matching the factor EMA settings."""
    frame = pl.DataFrame({"close": closes})
    ema_fast = pl.col("close").ewm_mean(span=fast_span, adjust=False, min_samples=fast_span)
    ema_slow = pl.col("close").ewm_mean(span=slow_span, adjust=False, min_samples=slow_span)
    ppo = pl.when(ema_slow != 0).then(100.0 * (ema_fast - ema_slow) / ema_slow).otherwise(None)
    return frame.select(ppo.alias("ppo")).get_column("ppo").to_list()


def test_price_oscillator_metadata() -> None:
    """PriceOscillatorFactor exposes the fixed production metadata contract."""
    factor = _factor()
    assert factor.name == "price_oscillator"
    assert factor.version == "1.0.0"
    assert factor.category == "price"
    assert factor.required_features == ("close",)
    assert factor.produced_columns == ("price_oscillator",)
    assert factor.fast_span == 12
    assert factor.slow_span == 26
    assert factor.lookback == 26
    assert factor.metadata.lookback == 26


def test_price_oscillator_calculation_correctness() -> None:
    """PPO matches 100 * (EMA_fast - EMA_slow) / EMA_slow."""
    closes = [float(i) for i in range(1, 16)]
    values = (
        _factor(fast_span=3, slow_span=5)
        .compute(pl.DataFrame({"close": closes}))
        .get_column("price_oscillator")
        .to_list()
    )
    expected = _expected_ppo(closes, fast_span=3, slow_span=5)
    for value, expected_value in zip(values, expected, strict=True):
        if expected_value is None:
            assert value is None
        else:
            assert value == pytest.approx(expected_value)


def test_warmup_nulls_match_slow_span() -> None:
    """The first slow_span - 1 PPO values are null."""
    frame = pl.DataFrame({"close": [float(i) for i in range(1, 11)]})
    values = (
        _factor(fast_span=2, slow_span=4).compute(frame).get_column("price_oscillator").to_list()
    )
    assert values[0] is None
    assert values[1] is None
    assert values[2] is None
    assert values[3] is not None


def test_constant_prices_zero_oscillator() -> None:
    """Constant prices yield a zero PPO after warm-up."""
    frame = pl.DataFrame({"close": [10.0] * 8})
    values = (
        _factor(fast_span=2, slow_span=4).compute(frame).get_column("price_oscillator").to_list()
    )
    assert values[3] == pytest.approx(0.0)
    assert values[7] == pytest.approx(0.0)


def test_increasing_and_decreasing_prices() -> None:
    """Rising markets produce positive PPO and falling markets negative PPO."""
    rising = _factor(fast_span=2, slow_span=4).compute(
        pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    )
    falling = _factor(fast_span=2, slow_span=4).compute(
        pl.DataFrame({"close": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]})
    )
    assert rising.get_column("price_oscillator").to_list()[5] > 0.0
    assert falling.get_column("price_oscillator").to_list()[5] < 0.0


def test_invalid_spans_raise() -> None:
    """Invalid fast/slow spans are rejected."""
    with pytest.raises(
        ValidationError, match="fast_span must be an integer greater than or equal to 2"
    ):
        PriceOscillatorFactor(fast_span=1, slow_span=26)
    with pytest.raises(
        ValidationError, match="slow_span must be an integer greater than or equal to 2"
    ):
        PriceOscillatorFactor(fast_span=12, slow_span=1)
    with pytest.raises(
        ValidationError, match="slow_span must be greater than fast_span"
    ) as exc_info:
        PriceOscillatorFactor(fast_span=26, slow_span=12)
    assert exc_info.value.error_code == "FACTOR-PRICE-OSCILLATOR-003"


def test_validation_immutability_schema_and_exports() -> None:
    """Validation, schema, determinism, and package export contracts."""
    assert_missing_close_raises(
        lambda: _factor(fast_span=2, slow_span=3),
        error_code="FACTOR-PRICE-OSCILLATOR-004",
        factor_name="price_oscillator",
    )
    factor = _factor(fast_span=2, slow_span=3)
    assert_protocol_and_immutability(factor, output_column="price_oscillator")
    assert_preserves_columns(factor, output_column="price_oscillator")
    assert_output_float64_nullable(factor, output_column="price_oscillator")
    assert_empty_and_single_row(
        lambda: _factor(fast_span=2, slow_span=3),
        output_column="price_oscillator",
    )
    assert_determinism(
        lambda: _factor(fast_span=2, slow_span=4),
        output_column="price_oscillator",
        frame=pl.DataFrame({"close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]}),
    )
    large = _factor(fast_span=2, slow_span=50).compute(
        pl.DataFrame({"close": [float(i) for i in range(1, 11)]})
    )
    assert all(value is None for value in large.get_column("price_oscillator").to_list())
    assert PriceOscillatorFactor is PriceOscillatorFactorDirect
