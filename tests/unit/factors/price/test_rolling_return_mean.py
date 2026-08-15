"""Unit tests for CQROS ``RollingReturnMeanFactor``."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.factors.price import RollingReturnMeanFactor
from cqros.factors.price.rolling_return_mean import (
    RollingReturnMeanFactor as RollingReturnMeanFactorDirect,
)
from tests.unit.factors.price._helpers import (
    assert_lookback_negative_raises,
    assert_lookback_zero_raises,
    assert_missing_close_raises,
    assert_preserves_columns,
    assert_protocol_and_immutability,
)


def _factor(*, lookback: int = 20) -> RollingReturnMeanFactor:
    """Build a rolling return mean factor with an optional lookback override."""
    return RollingReturnMeanFactor(lookback=lookback)


def test_rolling_return_mean_metadata() -> None:
    """RollingReturnMeanFactor exposes the fixed production metadata contract."""
    factor = _factor()
    assert factor.name == "rolling_return_mean"
    assert factor.version == "1.0.0"
    assert factor.category == "price"
    assert factor.required_features == ("close",)
    assert factor.produced_columns == ("rolling_return_mean",)
    assert factor.lookback == 20
    assert factor.metadata.name == "rolling_return_mean"


def test_rolling_return_mean_calculation_correctness() -> None:
    """Rolling return mean matches mean of one-period simple returns."""
    frame = pl.DataFrame({"close": [100.0, 110.0, 121.0, 110.0]})
    result = _factor(lookback=2).compute(frame)
    values = result.get_column("rolling_return_mean").to_list()
    assert values[0] is None
    assert values[1] is None
    r1 = (110.0 / 100.0) - 1.0
    r2 = (121.0 / 110.0) - 1.0
    r3 = (110.0 / 121.0) - 1.0
    assert values[2] == pytest.approx((r1 + r2) / 2.0)
    assert values[3] == pytest.approx((r2 + r3) / 2.0)


def test_insufficient_history_is_null() -> None:
    """Warm-up rows remain null when history is shorter than lookback."""
    frame = pl.DataFrame({"close": [1.0, 2.0]})
    values = _factor(lookback=3).compute(frame).get_column("rolling_return_mean").to_list()
    assert values == [None, None]


def test_lookback_validation_missing_close_and_immutability() -> None:
    """Validation, missing close, immutability, and exports."""
    assert_lookback_zero_raises(
        lambda value: _factor(lookback=value),
        error_code="FACTOR-ROLLING-RETURN-MEAN-001",
    )
    assert_lookback_negative_raises(lambda value: _factor(lookback=value))
    assert_missing_close_raises(
        lambda: _factor(lookback=1),
        error_code="FACTOR-ROLLING-RETURN-MEAN-002",
        factor_name="rolling_return_mean",
    )
    factor = _factor(lookback=1)
    assert_protocol_and_immutability(factor, output_column="rolling_return_mean")
    assert_preserves_columns(factor, output_column="rolling_return_mean")
    assert RollingReturnMeanFactor is RollingReturnMeanFactorDirect
