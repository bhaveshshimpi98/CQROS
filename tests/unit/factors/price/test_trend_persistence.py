"""Unit tests for CQROS ``TrendPersistenceFactor``."""

from __future__ import annotations

import math

import polars as pl
import pytest

from cqros.factors.price import TrendPersistenceFactor
from cqros.factors.price.trend_persistence import (
    TrendPersistenceFactor as TrendPersistenceFactorDirect,
)
from tests.unit.factors.price._helpers import (
    assert_lookback_negative_raises,
    assert_lookback_zero_raises,
    assert_missing_close_raises,
    assert_preserves_columns,
    assert_protocol_and_immutability,
)


def _factor(*, lookback: int = 20) -> TrendPersistenceFactor:
    """Build a trend persistence factor with an optional lookback override."""
    return TrendPersistenceFactor(lookback=lookback)


def test_trend_persistence_metadata() -> None:
    """TrendPersistenceFactor exposes the fixed production metadata contract."""
    factor = _factor()
    assert factor.name == "trend_persistence"
    assert factor.version == "1.0.0"
    assert factor.category == "price"
    assert factor.required_features == ("close",)
    assert factor.produced_columns == ("trend_persistence",)
    assert factor.lookback == 20
    assert factor.metadata.name == "trend_persistence"


def test_trend_persistence_calculation_correctness() -> None:
    """Trend persistence matches rolling mean of signed one-period returns."""
    frame = pl.DataFrame({"close": [10.0, 11.0, 12.0, 11.0, 13.0]})
    result = _factor(lookback=3).compute(frame)
    values = result.get_column("trend_persistence").to_list()
    assert values[0] is None
    assert values[1] is None
    assert values[2] is None
    # signs for returns at indices 1..3: +1, +1, -1; mean at index 3 over window 3
    assert values[3] == pytest.approx((1.0 + 1.0 + -1.0) / 3.0)
    # signs at indices 2..4: +1, -1, +1
    assert values[4] == pytest.approx((1.0 + -1.0 + 1.0) / 3.0)


def test_all_up_trend_is_one() -> None:
    """Strictly rising prices yield trend persistence of 1 after warm-up."""
    frame = pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0]})
    values = _factor(lookback=3).compute(frame).get_column("trend_persistence").to_list()
    assert values[3] == pytest.approx(1.0)


def test_zero_return_sign_is_zero() -> None:
    """Zero returns contribute a zero sign to persistence."""
    frame = pl.DataFrame({"close": [10.0, 10.0, 10.0, 10.0]})
    values = _factor(lookback=3).compute(frame).get_column("trend_persistence").to_list()
    assert values[3] == pytest.approx(0.0)
    assert not math.isnan(values[3])


def test_lookback_validation_missing_close_and_immutability() -> None:
    """Validation, missing close, immutability, and exports."""
    assert_lookback_zero_raises(
        lambda value: _factor(lookback=value),
        error_code="FACTOR-TREND-PERSISTENCE-001",
    )
    assert_lookback_negative_raises(lambda value: _factor(lookback=value))
    assert_missing_close_raises(
        lambda: _factor(lookback=1),
        error_code="FACTOR-TREND-PERSISTENCE-002",
        factor_name="trend_persistence",
    )
    factor = _factor(lookback=1)
    assert_protocol_and_immutability(factor, output_column="trend_persistence")
    assert_preserves_columns(factor, output_column="trend_persistence")
    assert TrendPersistenceFactor is TrendPersistenceFactorDirect
