"""Unit tests for CQROS ``RateOfChangeFactor``."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.core.exceptions import ValidationError
from cqros.factors.price import RateOfChangeFactor
from cqros.factors.price.rate_of_change import RateOfChangeFactor as RateOfChangeFactorDirect
from tests.unit.factors.price._helpers import (
    assert_determinism,
    assert_empty_and_single_row,
    assert_lookback_negative_raises,
    assert_lookback_zero_raises,
    assert_missing_close_raises,
    assert_output_float64_nullable,
    assert_preserves_columns,
    assert_protocol_and_immutability,
)


def _factor(*, lookback: int = 12) -> RateOfChangeFactor:
    """Build a rate of change factor with an optional lookback override."""
    return RateOfChangeFactor(lookback=lookback)


def test_rate_of_change_metadata() -> None:
    """RateOfChangeFactor exposes the fixed production metadata contract."""
    factor = _factor()
    assert factor.name == "rate_of_change"
    assert factor.version == "1.0.0"
    assert factor.category == "price"
    assert factor.required_features == ("close",)
    assert factor.produced_columns == ("rate_of_change",)
    assert factor.lookback == 12
    assert factor.metadata.name == "rate_of_change"


def test_rate_of_change_calculation_correctness() -> None:
    """Rate of change matches (close / close.shift(lookback)) - 1."""
    frame = pl.DataFrame({"close": [100.0, 110.0, 121.0, 133.1, 146.41]})
    values = _factor(lookback=2).compute(frame).get_column("rate_of_change").to_list()
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx((121.0 / 100.0) - 1.0)
    assert values[3] == pytest.approx((133.1 / 110.0) - 1.0)
    assert values[4] == pytest.approx((146.41 / 121.0) - 1.0)


def test_increasing_and_decreasing_prices() -> None:
    """Rising prices yield positive ROC and falling prices yield negative ROC."""
    rising = _factor(lookback=2).compute(pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0]}))
    falling = _factor(lookback=2).compute(pl.DataFrame({"close": [4.0, 3.0, 2.0, 1.0]}))
    assert rising.get_column("rate_of_change").to_list()[3] == pytest.approx(1.0)
    assert falling.get_column("rate_of_change").to_list()[3] == pytest.approx((1.0 / 3.0) - 1.0)


def test_constant_prices_zero_roc() -> None:
    """Constant prices yield zero rate of change after warm-up."""
    frame = pl.DataFrame({"close": [10.0, 10.0, 10.0, 10.0]})
    values = _factor(lookback=2).compute(frame).get_column("rate_of_change").to_list()
    assert values[2] == pytest.approx(0.0)
    assert values[3] == pytest.approx(0.0)


def test_null_close_propagates() -> None:
    """Null close values leave dependent ROC rows null."""
    frame = pl.DataFrame({"close": [10.0, None, 12.0, 13.0]})
    values = _factor(lookback=2).compute(frame).get_column("rate_of_change").to_list()
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx((12.0 / 10.0) - 1.0)
    assert values[3] is None


def test_validation_immutability_schema_and_exports() -> None:
    """Validation, schema, determinism, and package export contracts."""
    assert_lookback_zero_raises(
        lambda value: _factor(lookback=value),
        error_code="FACTOR-RATE-OF-CHANGE-001",
    )
    assert_lookback_negative_raises(lambda value: _factor(lookback=value))
    with pytest.raises(ValidationError, match="lookback must be an integer greater than 0"):
        RateOfChangeFactor(lookback=0)
    assert_missing_close_raises(
        lambda: _factor(lookback=1),
        error_code="FACTOR-RATE-OF-CHANGE-002",
        factor_name="rate_of_change",
    )
    factor = _factor(lookback=2)
    assert_protocol_and_immutability(factor, output_column="rate_of_change")
    assert_preserves_columns(factor, output_column="rate_of_change")
    assert_output_float64_nullable(factor, output_column="rate_of_change")
    assert_empty_and_single_row(lambda: _factor(lookback=2), output_column="rate_of_change")
    assert_determinism(
        lambda: _factor(lookback=2),
        output_column="rate_of_change",
        frame=pl.DataFrame({"close": [10.0, 11.0, 12.0, 13.0]}),
    )
    large = _factor(lookback=50).compute(pl.DataFrame({"close": [float(i) for i in range(1, 11)]}))
    assert all(value is None for value in large.get_column("rate_of_change").to_list())
    assert RateOfChangeFactor is RateOfChangeFactorDirect
