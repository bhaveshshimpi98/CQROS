"""Unit tests for CQROS ``RollingReturnMedianFactor``."""

from __future__ import annotations

import statistics

import polars as pl
import pytest

from cqros.factors.price import RollingReturnMedianFactor
from cqros.factors.price.rolling_return_median import (
    RollingReturnMedianFactor as RollingReturnMedianFactorDirect,
)
from tests.unit.factors.price._helpers import (
    assert_lookback_negative_raises,
    assert_lookback_zero_raises,
    assert_missing_close_raises,
    assert_preserves_columns,
    assert_protocol_and_immutability,
)


def _factor(*, lookback: int = 20) -> RollingReturnMedianFactor:
    """Build a rolling return median factor with an optional lookback override."""
    return RollingReturnMedianFactor(lookback=lookback)


def test_rolling_return_median_metadata() -> None:
    """RollingReturnMedianFactor exposes the fixed production metadata contract."""
    factor = _factor()
    assert factor.name == "rolling_return_median"
    assert factor.version == "1.0.0"
    assert factor.category == "price"
    assert factor.required_features == ("close",)
    assert factor.produced_columns == ("rolling_return_median",)
    assert factor.lookback == 20
    assert factor.metadata.produced_columns == ("rolling_return_median",)


def test_rolling_return_median_calculation_correctness() -> None:
    """Rolling return median matches median of one-period simple returns."""
    frame = pl.DataFrame({"close": [100.0, 110.0, 99.0, 120.0]})
    result = _factor(lookback=3).compute(frame)
    values = result.get_column("rolling_return_median").to_list()
    assert values[0] is None
    assert values[1] is None
    assert values[2] is None
    returns = [
        (110.0 / 100.0) - 1.0,
        (99.0 / 110.0) - 1.0,
        (120.0 / 99.0) - 1.0,
    ]
    assert values[3] == pytest.approx(statistics.median(returns))


def test_insufficient_history_is_null() -> None:
    """Warm-up rows remain null when history is shorter than lookback."""
    frame = pl.DataFrame({"close": [1.0, 2.0, 3.0]})
    values = _factor(lookback=5).compute(frame).get_column("rolling_return_median").to_list()
    assert values == [None, None, None]


def test_lookback_validation_missing_close_and_immutability() -> None:
    """Validation, missing close, immutability, and exports."""
    assert_lookback_zero_raises(
        lambda value: _factor(lookback=value),
        error_code="FACTOR-ROLLING-RETURN-MEDIAN-001",
    )
    assert_lookback_negative_raises(lambda value: _factor(lookback=value))
    assert_missing_close_raises(
        lambda: _factor(lookback=1),
        error_code="FACTOR-ROLLING-RETURN-MEDIAN-002",
        factor_name="rolling_return_median",
    )
    factor = _factor(lookback=1)
    assert_protocol_and_immutability(factor, output_column="rolling_return_median")
    assert_preserves_columns(factor, output_column="rolling_return_median")
    assert RollingReturnMedianFactor is RollingReturnMedianFactorDirect
