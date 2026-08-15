"""Unit tests for CQROS ``EfficiencyRatioFactor``."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.factors.price import EfficiencyRatioFactor
from cqros.factors.price.efficiency_ratio import (
    EfficiencyRatioFactor as EfficiencyRatioFactorDirect,
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


def _factor(*, lookback: int = 20) -> EfficiencyRatioFactor:
    """Build an efficiency ratio factor with an optional lookback override."""
    return EfficiencyRatioFactor(lookback=lookback)


def _efficiency_ratio(closes: list[float], *, lookback: int, index: int) -> float | None:
    """Return Kaufman efficiency ratio at ``index`` for a lookback window."""
    if index < lookback:
        return None
    net = abs(closes[index] - closes[index - lookback])
    path = 0.0
    for offset in range(index - lookback + 1, index + 1):
        path += abs(closes[offset] - closes[offset - 1])
    if path == 0.0:
        return None
    return net / path


def test_efficiency_ratio_metadata() -> None:
    """EfficiencyRatioFactor exposes the fixed production metadata contract."""
    factor = _factor()
    assert factor.name == "efficiency_ratio"
    assert factor.version == "1.0.0"
    assert factor.category == "price"
    assert factor.required_features == ("close",)
    assert factor.produced_columns == ("efficiency_ratio",)
    assert factor.lookback == 20
    assert factor.metadata.name == "efficiency_ratio"


def test_efficiency_ratio_calculation_correctness() -> None:
    """Efficiency ratio matches absolute net movement over path length."""
    closes = [10.0, 12.0, 11.0, 15.0, 14.0]
    values = (
        _factor(lookback=3)
        .compute(pl.DataFrame({"close": closes}))
        .get_column("efficiency_ratio")
        .to_list()
    )
    assert values[0] is None
    assert values[1] is None
    assert values[2] is None
    assert values[3] == pytest.approx(_efficiency_ratio(closes, lookback=3, index=3))
    assert values[4] == pytest.approx(_efficiency_ratio(closes, lookback=3, index=4))


def test_monotonic_prices_efficiency_is_one() -> None:
    """Strictly increasing prices yield efficiency ratio of 1."""
    frame = pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0]})
    values = _factor(lookback=3).compute(frame).get_column("efficiency_ratio").to_list()
    assert values[3] == pytest.approx(1.0)
    assert values[4] == pytest.approx(1.0)


def test_constant_prices_efficiency_is_null() -> None:
    """Constant prices make path length zero and efficiency null."""
    frame = pl.DataFrame({"close": [10.0, 10.0, 10.0, 10.0, 10.0]})
    values = _factor(lookback=3).compute(frame).get_column("efficiency_ratio").to_list()
    assert values[3] is None
    assert values[4] is None


def test_decreasing_prices_efficiency_is_one() -> None:
    """Strictly decreasing prices also yield efficiency ratio of 1."""
    frame = pl.DataFrame({"close": [5.0, 4.0, 3.0, 2.0, 1.0]})
    values = _factor(lookback=3).compute(frame).get_column("efficiency_ratio").to_list()
    assert values[3] == pytest.approx(1.0)


def test_null_close_propagates() -> None:
    """Null close values make incomplete efficiency windows null."""
    frame = pl.DataFrame({"close": [10.0, None, 12.0, 13.0, 14.0]})
    values = _factor(lookback=2).compute(frame).get_column("efficiency_ratio").to_list()
    assert values[0] is None
    assert values[1] is None
    assert values[2] is None
    assert values[3] is None
    assert values[4] == pytest.approx(abs(14.0 - 12.0) / (abs(13.0 - 12.0) + abs(14.0 - 13.0)))


def test_validation_immutability_schema_and_exports() -> None:
    """Validation, schema, determinism, and package export contracts."""
    assert_lookback_below_two_raises(
        lambda value: _factor(lookback=value),
        error_code="FACTOR-EFFICIENCY-RATIO-001",
    )
    assert_lookback_negative_raises(lambda value: _factor(lookback=value))
    assert_missing_close_raises(
        lambda: _factor(lookback=2),
        error_code="FACTOR-EFFICIENCY-RATIO-002",
        factor_name="efficiency_ratio",
    )
    factor = _factor(lookback=2)
    assert_protocol_and_immutability(factor, output_column="efficiency_ratio")
    assert_preserves_columns(factor, output_column="efficiency_ratio")
    assert_output_float64_nullable(factor, output_column="efficiency_ratio")
    assert_empty_and_single_row(
        lambda: _factor(lookback=2),
        output_column="efficiency_ratio",
    )
    assert_determinism(
        lambda: _factor(lookback=3),
        output_column="efficiency_ratio",
        frame=pl.DataFrame({"close": [10.0, 12.0, 11.0, 15.0, 14.0]}),
    )
    large = _factor(lookback=50).compute(pl.DataFrame({"close": [float(i) for i in range(1, 11)]}))
    assert all(value is None for value in large.get_column("efficiency_ratio").to_list())
    assert EfficiencyRatioFactor is EfficiencyRatioFactorDirect
