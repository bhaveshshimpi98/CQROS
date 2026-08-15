"""Unit tests for CQROS ``BollingerBandwidthFactor``."""

from __future__ import annotations

import math

import polars as pl
import pytest

from cqros.factors.price import BollingerBandwidthFactor
from cqros.factors.price.bollinger_bandwidth import (
    BollingerBandwidthFactor as BollingerBandwidthFactorDirect,
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


def _factor(*, lookback: int = 20) -> BollingerBandwidthFactor:
    """Build a Bollinger bandwidth factor with an optional lookback override."""
    return BollingerBandwidthFactor(lookback=lookback)


def _bollinger_bandwidth(closes: list[float]) -> float | None:
    """Return Bollinger bandwidth for a fully observed window."""
    n = len(closes)
    mean = sum(closes) / n
    if mean == 0.0:
        return None
    variance = sum((value - mean) ** 2 for value in closes) / n
    return (4.0 * math.sqrt(variance)) / mean


def test_bollinger_bandwidth_metadata() -> None:
    """BollingerBandwidthFactor exposes the fixed production metadata contract."""
    factor = _factor()
    assert factor.name == "bollinger_bandwidth"
    assert factor.version == "1.0.0"
    assert factor.category == "price"
    assert factor.required_features == ("close",)
    assert factor.produced_columns == ("bollinger_bandwidth",)
    assert factor.lookback == 20
    assert factor.metadata.name == "bollinger_bandwidth"


def test_bollinger_bandwidth_calculation_correctness() -> None:
    """Bollinger bandwidth matches (upper - lower) / SMA."""
    closes = [10.0, 12.0, 11.0, 15.0, 14.0]
    values = (
        _factor(lookback=3)
        .compute(pl.DataFrame({"close": closes}))
        .get_column("bollinger_bandwidth")
        .to_list()
    )
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx(_bollinger_bandwidth(closes[0:3]))
    assert values[3] == pytest.approx(_bollinger_bandwidth(closes[1:4]))
    assert values[4] == pytest.approx(_bollinger_bandwidth(closes[2:5]))


def test_constant_prices_zero_bandwidth() -> None:
    """Constant prices produce zero Bollinger bandwidth."""
    frame = pl.DataFrame({"close": [10.0, 10.0, 10.0, 10.0]})
    values = _factor(lookback=3).compute(frame).get_column("bollinger_bandwidth").to_list()
    assert values[3] == pytest.approx(0.0)


def test_zero_sma_is_null() -> None:
    """Zero SMA returns null bandwidth."""
    frame = pl.DataFrame({"close": [1.0, -1.0, 0.0]})
    values = _factor(lookback=3).compute(frame).get_column("bollinger_bandwidth").to_list()
    assert values[2] is None


def test_increasing_and_decreasing_prices() -> None:
    """Rising and falling windows yield positive finite bandwidth."""
    rising = _factor(lookback=3).compute(pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0]}))
    falling = _factor(lookback=3).compute(pl.DataFrame({"close": [4.0, 3.0, 2.0, 1.0]}))
    assert rising.get_column("bollinger_bandwidth").to_list()[3] > 0.0
    assert falling.get_column("bollinger_bandwidth").to_list()[3] > 0.0


def test_validation_immutability_schema_and_exports() -> None:
    """Validation, schema, determinism, and package export contracts."""
    assert_lookback_below_two_raises(
        lambda value: _factor(lookback=value),
        error_code="FACTOR-BOLLINGER-BANDWIDTH-001",
    )
    assert_lookback_negative_raises(lambda value: _factor(lookback=value))
    assert_missing_close_raises(
        lambda: _factor(lookback=2),
        error_code="FACTOR-BOLLINGER-BANDWIDTH-002",
        factor_name="bollinger_bandwidth",
    )
    factor = _factor(lookback=2)
    assert_protocol_and_immutability(factor, output_column="bollinger_bandwidth")
    assert_preserves_columns(factor, output_column="bollinger_bandwidth")
    assert_output_float64_nullable(factor, output_column="bollinger_bandwidth")
    assert_empty_and_single_row(
        lambda: _factor(lookback=2),
        output_column="bollinger_bandwidth",
    )
    assert_determinism(
        lambda: _factor(lookback=3),
        output_column="bollinger_bandwidth",
        frame=pl.DataFrame({"close": [10.0, 12.0, 11.0, 15.0]}),
    )
    large = _factor(lookback=50).compute(pl.DataFrame({"close": [float(i) for i in range(1, 11)]}))
    assert all(value is None for value in large.get_column("bollinger_bandwidth").to_list())
    assert BollingerBandwidthFactor is BollingerBandwidthFactorDirect
