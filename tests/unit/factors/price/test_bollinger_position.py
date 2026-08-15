"""Unit tests for CQROS ``BollingerPositionFactor``."""

from __future__ import annotations

import math

import polars as pl
import pytest

from cqros.factors.price import BollingerPositionFactor
from cqros.factors.price.bollinger_position import (
    BollingerPositionFactor as BollingerPositionFactorDirect,
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


def _factor(*, lookback: int = 20) -> BollingerPositionFactor:
    """Build a Bollinger position factor with an optional lookback override."""
    return BollingerPositionFactor(lookback=lookback)


def _bollinger_position(closes: list[float]) -> float | None:
    """Return Bollinger position for a fully observed window."""
    n = len(closes)
    mean = sum(closes) / n
    variance = sum((value - mean) ** 2 for value in closes) / n
    std = math.sqrt(variance)
    width = 4.0 * std
    if width == 0.0:
        return None
    lower = mean - 2.0 * std
    return (closes[-1] - lower) / width


def test_bollinger_position_metadata() -> None:
    """BollingerPositionFactor exposes the fixed production metadata contract."""
    factor = _factor()
    assert factor.name == "bollinger_position"
    assert factor.version == "1.0.0"
    assert factor.category == "price"
    assert factor.required_features == ("close",)
    assert factor.produced_columns == ("bollinger_position",)
    assert factor.lookback == 20
    assert factor.metadata.name == "bollinger_position"


def test_bollinger_position_calculation_correctness() -> None:
    """Bollinger position matches (close - lower) / (upper - lower)."""
    closes = [10.0, 12.0, 11.0, 15.0, 14.0]
    values = (
        _factor(lookback=3)
        .compute(pl.DataFrame({"close": closes}))
        .get_column("bollinger_position")
        .to_list()
    )
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx(_bollinger_position(closes[0:3]))
    assert values[3] == pytest.approx(_bollinger_position(closes[1:4]))
    assert values[4] == pytest.approx(_bollinger_position(closes[2:5]))


def test_zero_band_width_is_null() -> None:
    """Constant prices produce zero band width and null position."""
    frame = pl.DataFrame({"close": [10.0, 10.0, 10.0, 10.0]})
    values = _factor(lookback=3).compute(frame).get_column("bollinger_position").to_list()
    assert values[3] is None


def test_mid_band_is_half() -> None:
    """Close equal to the middle band yields position 0.5."""
    frame = pl.DataFrame({"close": [8.0, 12.0, 10.0]})
    values = _factor(lookback=3).compute(frame).get_column("bollinger_position").to_list()
    assert values[2] == pytest.approx(0.5)


def test_increasing_and_decreasing_prices() -> None:
    """Rising and falling windows remain inside a finite [0, 1]-ish range."""
    rising = _factor(lookback=3).compute(pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0]}))
    falling = _factor(lookback=3).compute(pl.DataFrame({"close": [4.0, 3.0, 2.0, 1.0]}))
    assert rising.get_column("bollinger_position").to_list()[3] is not None
    assert falling.get_column("bollinger_position").to_list()[3] is not None


def test_validation_immutability_schema_and_exports() -> None:
    """Validation, schema, determinism, and package export contracts."""
    assert_lookback_below_two_raises(
        lambda value: _factor(lookback=value),
        error_code="FACTOR-BOLLINGER-POSITION-001",
    )
    assert_lookback_negative_raises(lambda value: _factor(lookback=value))
    assert_missing_close_raises(
        lambda: _factor(lookback=2),
        error_code="FACTOR-BOLLINGER-POSITION-002",
        factor_name="bollinger_position",
    )
    factor = _factor(lookback=2)
    assert_protocol_and_immutability(factor, output_column="bollinger_position")
    assert_preserves_columns(factor, output_column="bollinger_position")
    assert_output_float64_nullable(factor, output_column="bollinger_position")
    assert_empty_and_single_row(
        lambda: _factor(lookback=2),
        output_column="bollinger_position",
    )
    assert_determinism(
        lambda: _factor(lookback=3),
        output_column="bollinger_position",
        frame=pl.DataFrame({"close": [10.0, 12.0, 11.0, 15.0]}),
    )
    assert BollingerPositionFactor is BollingerPositionFactorDirect
