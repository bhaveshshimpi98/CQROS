"""Unit tests for CQROS ``MaximumDrawdownFactor``."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.factors.price import MaximumDrawdownFactor
from cqros.factors.price.maximum_drawdown import (
    MaximumDrawdownFactor as MaximumDrawdownFactorDirect,
)
from tests.unit.factors.price._helpers import (
    assert_lookback_negative_raises,
    assert_lookback_zero_raises,
    assert_missing_close_raises,
    assert_preserves_columns,
    assert_protocol_and_immutability,
)


def _factor(*, lookback: int = 20) -> MaximumDrawdownFactor:
    """Build a maximum drawdown factor with an optional lookback override."""
    return MaximumDrawdownFactor(lookback=lookback)


def test_maximum_drawdown_metadata() -> None:
    """MaximumDrawdownFactor exposes the fixed production metadata contract."""
    factor = _factor()
    assert factor.name == "maximum_drawdown"
    assert factor.version == "1.0.0"
    assert factor.category == "price"
    assert factor.required_features == ("close",)
    assert factor.produced_columns == ("maximum_drawdown",)
    assert factor.lookback == 20
    assert factor.metadata.produced_columns == ("maximum_drawdown",)


def test_maximum_drawdown_calculation_correctness() -> None:
    """Maximum drawdown matches (rolling_min / rolling_max) - 1."""
    frame = pl.DataFrame({"close": [10.0, 12.0, 8.0, 9.0]})
    result = _factor(lookback=3).compute(frame)
    values = result.get_column("maximum_drawdown").to_list()
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx((8.0 / 12.0) - 1.0)
    assert values[3] == pytest.approx((8.0 / 12.0) - 1.0)


def test_flat_window_drawdown_is_zero() -> None:
    """Constant prices produce zero drawdown after warm-up."""
    frame = pl.DataFrame({"close": [7.0, 7.0, 7.0]})
    values = _factor(lookback=3).compute(frame).get_column("maximum_drawdown").to_list()
    assert values[2] == pytest.approx(0.0)


def test_lookback_validation_missing_close_and_immutability() -> None:
    """Validation, missing close, immutability, and exports."""
    assert_lookback_zero_raises(
        lambda value: _factor(lookback=value),
        error_code="FACTOR-MAXIMUM-DRAWDOWN-001",
    )
    assert_lookback_negative_raises(lambda value: _factor(lookback=value))
    assert_missing_close_raises(
        lambda: _factor(lookback=1),
        error_code="FACTOR-MAXIMUM-DRAWDOWN-002",
        factor_name="maximum_drawdown",
    )
    factor = _factor(lookback=1)
    assert_protocol_and_immutability(factor, output_column="maximum_drawdown")
    assert_preserves_columns(factor, output_column="maximum_drawdown")
    assert MaximumDrawdownFactor is MaximumDrawdownFactorDirect
