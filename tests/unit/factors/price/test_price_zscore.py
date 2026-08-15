"""Unit tests for CQROS ``PriceZScoreFactor``."""

from __future__ import annotations

import math

import polars as pl
import pytest

from cqros.factors.price import PriceZScoreFactor
from cqros.factors.price.price_zscore import PriceZScoreFactor as PriceZScoreFactorDirect
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


def _factor(*, lookback: int = 20) -> PriceZScoreFactor:
    """Build a price z-score factor with an optional lookback override."""
    return PriceZScoreFactor(lookback=lookback)


def _price_zscore(closes: list[float]) -> float | None:
    """Return population z-score for a fully observed window."""
    mean = sum(closes) / len(closes)
    variance = sum((value - mean) ** 2 for value in closes) / len(closes)
    std = math.sqrt(variance)
    if std == 0.0:
        return 0.0
    return (closes[-1] - mean) / std


def test_price_zscore_metadata() -> None:
    """PriceZScoreFactor exposes the fixed production metadata contract."""
    factor = _factor()
    assert factor.name == "price_zscore"
    assert factor.version == "1.0.0"
    assert factor.category == "price"
    assert factor.required_features == ("close",)
    assert factor.produced_columns == ("price_zscore",)
    assert factor.lookback == 20
    assert factor.metadata.name == "price_zscore"


def test_price_zscore_calculation_correctness() -> None:
    """Price z-score matches (close - mean) / population std."""
    closes = [10.0, 12.0, 11.0, 15.0, 14.0]
    values = (
        _factor(lookback=3)
        .compute(pl.DataFrame({"close": closes}))
        .get_column("price_zscore")
        .to_list()
    )
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx(_price_zscore(closes[0:3]))
    assert values[3] == pytest.approx(_price_zscore(closes[1:4]))
    assert values[4] == pytest.approx(_price_zscore(closes[2:5]))


def test_constant_prices_are_zero() -> None:
    """Constant prices make standard deviation zero and z-score 0.0."""
    frame = pl.DataFrame({"close": [10.0, 10.0, 10.0, 10.0]})
    values = _factor(lookback=3).compute(frame).get_column("price_zscore").to_list()
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx(0.0)
    assert values[3] == pytest.approx(0.0)


def test_increasing_and_decreasing_prices() -> None:
    """Rising windows yield positive z-scores and falling windows negative."""
    rising = _factor(lookback=3).compute(pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0]}))
    falling = _factor(lookback=3).compute(pl.DataFrame({"close": [4.0, 3.0, 2.0, 1.0]}))
    assert rising.get_column("price_zscore").to_list()[3] > 0.0
    assert falling.get_column("price_zscore").to_list()[3] < 0.0


def test_null_close_propagates() -> None:
    """Null close values leave incomplete z-score windows null."""
    frame = pl.DataFrame({"close": [10.0, None, 12.0, 13.0, 14.0]})
    values = _factor(lookback=3).compute(frame).get_column("price_zscore").to_list()
    assert values[0] is None
    assert values[1] is None
    assert values[2] is None
    assert values[3] is None
    assert values[4] == pytest.approx(_price_zscore([12.0, 13.0, 14.0]))


def test_validation_immutability_schema_and_exports() -> None:
    """Validation, schema, determinism, and package export contracts."""
    assert_lookback_below_two_raises(
        lambda value: _factor(lookback=value),
        error_code="FACTOR-PRICE-ZSCORE-001",
    )
    assert_lookback_negative_raises(lambda value: _factor(lookback=value))
    assert_missing_close_raises(
        lambda: _factor(lookback=2),
        error_code="FACTOR-PRICE-ZSCORE-002",
        factor_name="price_zscore",
    )
    factor = _factor(lookback=2)
    assert_protocol_and_immutability(factor, output_column="price_zscore")
    assert_preserves_columns(factor, output_column="price_zscore")
    assert_output_float64_nullable(factor, output_column="price_zscore")
    assert_empty_and_single_row(lambda: _factor(lookback=2), output_column="price_zscore")
    assert_determinism(
        lambda: _factor(lookback=3),
        output_column="price_zscore",
        frame=pl.DataFrame({"close": [10.0, 12.0, 11.0, 15.0]}),
    )
    large = _factor(lookback=50).compute(pl.DataFrame({"close": [float(i) for i in range(1, 11)]}))
    assert all(value is None for value in large.get_column("price_zscore").to_list())
    assert PriceZScoreFactor is PriceZScoreFactorDirect
