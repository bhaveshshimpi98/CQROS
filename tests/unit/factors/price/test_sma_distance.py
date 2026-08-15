"""Unit tests for CQROS ``SMADistanceFactor``."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.factors.price import SMADistanceFactor
from cqros.factors.price.sma_distance import SMADistanceFactor as SMADistanceFactorDirect
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


def _factor(*, lookback: int = 20) -> SMADistanceFactor:
    """Build an SMA distance factor with an optional lookback override."""
    return SMADistanceFactor(lookback=lookback)


def test_sma_distance_metadata() -> None:
    """SMADistanceFactor exposes the fixed production metadata contract."""
    factor = _factor()
    assert factor.name == "sma_distance"
    assert factor.version == "1.0.0"
    assert factor.category == "price"
    assert factor.required_features == ("close",)
    assert factor.produced_columns == ("sma_distance",)
    assert factor.lookback == 20
    assert factor.metadata.name == "sma_distance"


def test_sma_distance_calculation_correctness() -> None:
    """SMA distance matches (close - SMA) / SMA."""
    frame = pl.DataFrame({"close": [10.0, 12.0, 14.0, 8.0]})
    values = _factor(lookback=3).compute(frame).get_column("sma_distance").to_list()
    assert values[0] is None
    assert values[1] is None
    sma2 = (10.0 + 12.0 + 14.0) / 3.0
    sma3 = (12.0 + 14.0 + 8.0) / 3.0
    assert values[2] == pytest.approx((14.0 - sma2) / sma2)
    assert values[3] == pytest.approx((8.0 - sma3) / sma3)


def test_constant_prices_zero_distance() -> None:
    """Constant prices yield zero SMA distance after warm-up."""
    frame = pl.DataFrame({"close": [10.0, 10.0, 10.0, 10.0]})
    values = _factor(lookback=3).compute(frame).get_column("sma_distance").to_list()
    assert values[3] == pytest.approx(0.0)


def test_increasing_prices_non_negative_at_window_end() -> None:
    """Rising series ends at or above SMA, so distance is non-negative."""
    frame = pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0]})
    values = _factor(lookback=3).compute(frame).get_column("sma_distance").to_list()
    assert values[3] is not None
    assert values[3] >= 0.0


def test_null_close_propagates() -> None:
    """Null close values make incomplete windows null."""
    frame = pl.DataFrame({"close": [10.0, None, 12.0, 13.0]})
    values = _factor(lookback=2).compute(frame).get_column("sma_distance").to_list()
    assert values[0] is None
    assert values[1] is None
    assert values[2] is None
    assert values[3] == pytest.approx((13.0 - 12.5) / 12.5)


def test_validation_immutability_schema_and_exports() -> None:
    """Validation, schema, determinism, and package export contracts."""
    assert_lookback_below_two_raises(
        lambda value: _factor(lookback=value),
        error_code="FACTOR-SMA-DISTANCE-001",
    )
    assert_lookback_negative_raises(lambda value: _factor(lookback=value))
    assert_missing_close_raises(
        lambda: _factor(lookback=2),
        error_code="FACTOR-SMA-DISTANCE-002",
        factor_name="sma_distance",
    )
    factor = _factor(lookback=2)
    assert_protocol_and_immutability(factor, output_column="sma_distance")
    assert_preserves_columns(factor, output_column="sma_distance")
    assert_output_float64_nullable(factor, output_column="sma_distance")
    assert_empty_and_single_row(lambda: _factor(lookback=2), output_column="sma_distance")
    assert_determinism(
        lambda: _factor(lookback=3),
        output_column="sma_distance",
        frame=pl.DataFrame({"close": [10.0, 12.0, 14.0, 8.0]}),
    )
    large = _factor(lookback=50).compute(pl.DataFrame({"close": [float(i) for i in range(1, 11)]}))
    assert all(value is None for value in large.get_column("sma_distance").to_list())
    assert SMADistanceFactor is SMADistanceFactorDirect
