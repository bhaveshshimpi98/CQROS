"""Unit tests for CQROS ``UlcerIndexFactor``."""

from __future__ import annotations

import math

import polars as pl
import pytest

from cqros.factors.price import UlcerIndexFactor
from cqros.factors.price.ulcer_index import UlcerIndexFactor as UlcerIndexFactorDirect
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


def _factor(*, lookback: int = 20) -> UlcerIndexFactor:
    """Build an Ulcer Index factor with an optional lookback override."""
    return UlcerIndexFactor(lookback=lookback)


def _percent_drawdowns(closes: list[float], *, lookback: int) -> list[float | None]:
    """Return percent drawdowns from rolling peak."""
    values: list[float | None] = []
    for index in range(len(closes)):
        if index < lookback - 1:
            values.append(None)
            continue
        peak = max(closes[index - lookback + 1 : index + 1])
        if peak == 0.0:
            values.append(None)
        else:
            values.append(100.0 * (closes[index] / peak - 1.0))
    return values


def _ulcer_index(closes: list[float], *, lookback: int, index: int) -> float | None:
    """Return Ulcer Index at ``index`` for a fully observed window."""
    drawdowns = _percent_drawdowns(closes, lookback=lookback)
    if index < 2 * lookback - 2:
        return None
    window = drawdowns[index - lookback + 1 : index + 1]
    if any(value is None for value in window):
        return None
    mean_sq = sum((value or 0.0) ** 2 for value in window) / lookback
    return math.sqrt(mean_sq)


def test_ulcer_index_metadata() -> None:
    """UlcerIndexFactor exposes the fixed production metadata contract."""
    factor = _factor()
    assert factor.name == "ulcer_index"
    assert factor.version == "1.0.0"
    assert factor.category == "price"
    assert factor.required_features == ("close",)
    assert factor.produced_columns == ("ulcer_index",)
    assert factor.lookback == 20
    assert factor.metadata.name == "ulcer_index"


def test_ulcer_index_calculation_correctness() -> None:
    """Ulcer Index matches RMS of percent drawdowns from peak."""
    closes = [10.0, 12.0, 9.0, 11.0, 8.0, 10.0]
    values = (
        _factor(lookback=3)
        .compute(pl.DataFrame({"close": closes}))
        .get_column("ulcer_index")
        .to_list()
    )
    assert values[0] is None
    assert values[1] is None
    assert values[2] is None
    assert values[3] is None
    assert values[4] == pytest.approx(_ulcer_index(closes, lookback=3, index=4))
    assert values[5] == pytest.approx(_ulcer_index(closes, lookback=3, index=5))


def test_constant_prices_zero_ulcer() -> None:
    """Constant prices produce zero Ulcer Index after warm-up."""
    frame = pl.DataFrame({"close": [10.0, 10.0, 10.0, 10.0, 10.0]})
    values = _factor(lookback=2).compute(frame).get_column("ulcer_index").to_list()
    assert values[3] == pytest.approx(0.0)
    assert values[4] == pytest.approx(0.0)


def test_increasing_prices_zero_ulcer() -> None:
    """Strictly increasing prices have zero drawdown and zero Ulcer Index."""
    frame = pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0]})
    values = _factor(lookback=2).compute(frame).get_column("ulcer_index").to_list()
    assert values[3] == pytest.approx(0.0)
    assert values[4] == pytest.approx(0.0)


def test_decreasing_prices_positive_ulcer() -> None:
    """Strictly decreasing prices produce a positive Ulcer Index."""
    frame = pl.DataFrame({"close": [5.0, 4.0, 3.0, 2.0, 1.0]})
    values = _factor(lookback=2).compute(frame).get_column("ulcer_index").to_list()
    assert values[3] is not None
    assert values[3] > 0.0


def test_validation_immutability_schema_and_exports() -> None:
    """Validation, schema, determinism, and package export contracts."""
    assert_lookback_below_two_raises(
        lambda value: _factor(lookback=value),
        error_code="FACTOR-ULCER-INDEX-001",
    )
    assert_lookback_negative_raises(lambda value: _factor(lookback=value))
    assert_missing_close_raises(
        lambda: _factor(lookback=2),
        error_code="FACTOR-ULCER-INDEX-002",
        factor_name="ulcer_index",
    )
    factor = _factor(lookback=2)
    assert_protocol_and_immutability(factor, output_column="ulcer_index")
    assert_preserves_columns(factor, output_column="ulcer_index")
    assert_output_float64_nullable(factor, output_column="ulcer_index")
    assert_empty_and_single_row(lambda: _factor(lookback=2), output_column="ulcer_index")
    assert_determinism(
        lambda: _factor(lookback=3),
        output_column="ulcer_index",
        frame=pl.DataFrame({"close": [10.0, 12.0, 9.0, 11.0, 8.0]}),
    )
    large = _factor(lookback=50).compute(pl.DataFrame({"close": [float(i) for i in range(1, 11)]}))
    assert all(value is None for value in large.get_column("ulcer_index").to_list())
    assert UlcerIndexFactor is UlcerIndexFactorDirect
