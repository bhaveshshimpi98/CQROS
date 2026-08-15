"""Unit tests for CQROS ``BreakoutStrengthFactor``."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.factors.price import BreakoutStrengthFactor
from cqros.factors.price.breakout_strength import (
    BreakoutStrengthFactor as BreakoutStrengthFactorDirect,
)
from tests.unit.factors.price._helpers import (
    assert_lookback_negative_raises,
    assert_lookback_zero_raises,
    assert_missing_close_raises,
    assert_preserves_columns,
    assert_protocol_and_immutability,
)


def _factor(*, lookback: int = 20) -> BreakoutStrengthFactor:
    """Build a breakout strength factor with an optional lookback override."""
    return BreakoutStrengthFactor(lookback=lookback)


def test_breakout_strength_metadata() -> None:
    """BreakoutStrengthFactor exposes the fixed production metadata contract."""
    factor = _factor()
    assert factor.name == "breakout_strength"
    assert factor.version == "1.0.0"
    assert factor.category == "price"
    assert factor.required_features == ("close",)
    assert factor.produced_columns == ("breakout_strength",)
    assert factor.lookback == 20
    assert factor.metadata.name == "breakout_strength"


def test_breakout_strength_calculation_correctness() -> None:
    """Breakout strength matches return versus prior rolling high."""
    frame = pl.DataFrame({"close": [10.0, 12.0, 11.0, 15.0]})
    result = _factor(lookback=2).compute(frame)
    values = result.get_column("breakout_strength").to_list()
    assert values[0] is None
    assert values[1] is None
    # prior high at index 2 is rolling_max([10, 12]) = 12
    assert values[2] == pytest.approx((11.0 / 12.0) - 1.0)
    # prior high at index 3 is rolling_max([12, 11]) = 12
    assert values[3] == pytest.approx((15.0 / 12.0) - 1.0)


def test_insufficient_history_is_null() -> None:
    """Rows before a full prior high window remain null."""
    frame = pl.DataFrame({"close": [1.0, 2.0, 3.0]})
    values = _factor(lookback=3).compute(frame).get_column("breakout_strength").to_list()
    assert values == [None, None, None]


def test_lookback_validation_missing_close_and_immutability() -> None:
    """Validation, missing close, immutability, and exports."""
    assert_lookback_zero_raises(
        lambda value: _factor(lookback=value),
        error_code="FACTOR-BREAKOUT-STRENGTH-001",
    )
    assert_lookback_negative_raises(lambda value: _factor(lookback=value))
    assert_missing_close_raises(
        lambda: _factor(lookback=1),
        error_code="FACTOR-BREAKOUT-STRENGTH-002",
        factor_name="breakout_strength",
    )
    factor = _factor(lookback=1)
    assert_protocol_and_immutability(factor, output_column="breakout_strength")
    assert_preserves_columns(factor, output_column="breakout_strength")
    assert BreakoutStrengthFactor is BreakoutStrengthFactorDirect
