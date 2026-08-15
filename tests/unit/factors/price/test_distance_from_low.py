"""Unit tests for CQROS ``DistanceFromLowFactor``."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.factors.price import DistanceFromLowFactor
from cqros.factors.price.distance_from_low import (
    DistanceFromLowFactor as DistanceFromLowFactorDirect,
)
from tests.unit.factors.price._helpers import (
    assert_lookback_negative_raises,
    assert_lookback_zero_raises,
    assert_missing_close_raises,
    assert_preserves_columns,
    assert_protocol_and_immutability,
)


def _factor(*, lookback: int = 20) -> DistanceFromLowFactor:
    """Build a distance-from-low factor with an optional lookback override."""
    return DistanceFromLowFactor(lookback=lookback)


def test_distance_from_low_metadata() -> None:
    """DistanceFromLowFactor exposes the fixed production metadata contract."""
    factor = _factor()
    assert factor.name == "distance_from_low"
    assert factor.version == "1.0.0"
    assert factor.category == "price"
    assert factor.required_features == ("close",)
    assert factor.produced_columns == ("distance_from_low",)
    assert factor.lookback == 20
    assert factor.metadata.name == "distance_from_low"


def test_distance_from_low_calculation_correctness() -> None:
    """Distance from low matches (close - rolling_low) / rolling_low."""
    frame = pl.DataFrame({"close": [10.0, 12.0, 9.0, 11.0]})
    result = _factor(lookback=3).compute(frame)
    values = result.get_column("distance_from_low").to_list()
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx((9.0 - 9.0) / 9.0)
    assert values[3] == pytest.approx((11.0 - 9.0) / 9.0)


def test_at_low_is_zero() -> None:
    """Distance is zero when close equals the rolling low."""
    frame = pl.DataFrame({"close": [3.0, 2.0, 1.0]})
    values = _factor(lookback=3).compute(frame).get_column("distance_from_low").to_list()
    assert values[2] == pytest.approx(0.0)


def test_lookback_validation_missing_close_and_immutability() -> None:
    """Validation, missing close, immutability, and exports."""
    assert_lookback_zero_raises(
        lambda value: _factor(lookback=value),
        error_code="FACTOR-DISTANCE-FROM-LOW-001",
    )
    assert_lookback_negative_raises(lambda value: _factor(lookback=value))
    assert_missing_close_raises(
        lambda: _factor(lookback=1),
        error_code="FACTOR-DISTANCE-FROM-LOW-002",
        factor_name="distance_from_low",
    )
    factor = _factor(lookback=1)
    assert_protocol_and_immutability(factor, output_column="distance_from_low")
    assert_preserves_columns(factor, output_column="distance_from_low")
    assert DistanceFromLowFactor is DistanceFromLowFactorDirect
