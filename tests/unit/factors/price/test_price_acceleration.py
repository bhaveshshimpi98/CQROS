"""Unit tests for CQROS ``PriceAccelerationFactor``."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.factors.price import PriceAccelerationFactor
from cqros.factors.price.price_acceleration import (
    PriceAccelerationFactor as PriceAccelerationFactorDirect,
)
from tests.unit.factors.price._helpers import (
    assert_lookback_negative_raises,
    assert_lookback_zero_raises,
    assert_missing_close_raises,
    assert_preserves_columns,
    assert_protocol_and_immutability,
)


def _factor(*, lookback: int = 20) -> PriceAccelerationFactor:
    """Build a price acceleration factor with an optional lookback override."""
    return PriceAccelerationFactor(lookback=lookback)


def test_price_acceleration_metadata() -> None:
    """PriceAccelerationFactor exposes the fixed production metadata contract."""
    factor = _factor()
    assert factor.name == "price_acceleration"
    assert factor.version == "1.0.0"
    assert factor.category == "price"
    assert factor.required_features == ("close",)
    assert factor.produced_columns == ("price_acceleration",)
    assert factor.lookback == 20
    meta = factor.metadata
    assert meta.name == "price_acceleration"
    assert meta.produced_columns == ("price_acceleration",)
    assert meta.lookback == 20


def test_price_acceleration_calculation_correctness() -> None:
    """Acceleration matches consecutive non-overlapping momentum difference."""
    frame = pl.DataFrame({"close": [100.0, 110.0, 121.0, 133.1, 146.41]})
    result = _factor(lookback=2).compute(frame)
    values = result.get_column("price_acceleration").to_list()
    assert values[0] is None
    assert values[1] is None
    assert values[2] is None
    assert values[3] is None
    current = (146.41 / 121.0) - 1.0
    prior = (121.0 / 100.0) - 1.0
    assert values[4] == pytest.approx(current - prior)


def test_null_head_rows_match_two_times_lookback() -> None:
    """The first 2 * lookback acceleration values are null."""
    frame = pl.DataFrame({"close": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})
    result = _factor(lookback=2).compute(frame)
    values = result.get_column("price_acceleration").to_list()
    assert values[:4] == [None, None, None, None]
    assert values[4] is not None


def test_internal_null_propagates() -> None:
    """Null close values propagate into acceleration without filling."""
    frame = pl.DataFrame({"close": [100.0, 110.0, None, 133.1, 146.41]})
    result = _factor(lookback=2).compute(frame)
    values = result.get_column("price_acceleration").to_list()
    assert values[4] is None


def test_lookback_validation_and_missing_close() -> None:
    """Lookback validation and missing close fail fast."""
    assert_lookback_zero_raises(
        lambda value: _factor(lookback=value), error_code="FACTOR-PRICE-ACCELERATION-001"
    )
    assert_lookback_negative_raises(lambda value: _factor(lookback=value))
    assert_missing_close_raises(
        lambda: _factor(lookback=1),
        error_code="FACTOR-PRICE-ACCELERATION-002",
        factor_name="price_acceleration",
    )


def test_immutability_protocol_and_exports() -> None:
    """Protocol conformance, immutability, columns, and package exports."""
    factor = _factor(lookback=1)
    assert_protocol_and_immutability(factor, output_column="price_acceleration")
    assert_preserves_columns(factor, output_column="price_acceleration")
    assert PriceAccelerationFactor is PriceAccelerationFactorDirect
    import cqros.factors as factors_package
    import cqros.factors.price as price_package

    assert "PriceAccelerationFactor" in price_package.__all__
    assert "PriceAccelerationFactor" in factors_package.__all__
