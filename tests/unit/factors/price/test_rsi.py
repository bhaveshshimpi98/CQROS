"""Unit tests for CQROS ``RSIFactor``."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.factors.price import RSIFactor
from cqros.factors.price.rsi import RSIFactor as RSIFactorDirect
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


def _factor(*, lookback: int = 14) -> RSIFactor:
    """Build an RSI factor with an optional lookback override."""
    return RSIFactor(lookback=lookback)


def _expected_rsi(closes: list[float], *, lookback: int) -> list[float | None]:
    """Return RSI values matching the factor's Wilder ewm settings."""
    frame = pl.DataFrame({"close": closes})
    delta = pl.col("close").diff()
    gain = pl.when(delta.is_null()).then(None).when(delta > 0).then(delta).otherwise(0.0)
    loss = pl.when(delta.is_null()).then(None).when(delta < 0).then(-delta).otherwise(0.0)
    avg_gain = gain.ewm_mean(alpha=1.0 / lookback, adjust=False, min_samples=lookback)
    avg_loss = loss.ewm_mean(alpha=1.0 / lookback, adjust=False, min_samples=lookback)
    rsi = (
        pl.when(avg_loss.is_null() | avg_gain.is_null())
        .then(None)
        .when((avg_loss == 0) & (avg_gain == 0))
        .then(None)
        .when(avg_loss == 0)
        .then(100.0)
        .otherwise(100.0 - (100.0 / (1.0 + (avg_gain / avg_loss))))
    )
    return frame.select(rsi.alias("rsi")).get_column("rsi").to_list()


def test_rsi_metadata() -> None:
    """RSIFactor exposes the fixed production metadata contract."""
    factor = _factor()
    assert factor.name == "rsi"
    assert factor.version == "1.0.0"
    assert factor.category == "price"
    assert factor.required_features == ("close",)
    assert factor.produced_columns == ("rsi",)
    assert factor.lookback == 14
    assert factor.metadata.name == "rsi"


def test_rsi_calculation_correctness() -> None:
    """RSI matches Wilder ewm average gain and loss formulation."""
    closes = [10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5]
    values = (
        _factor(lookback=3).compute(pl.DataFrame({"close": closes})).get_column("rsi").to_list()
    )
    expected = _expected_rsi(closes, lookback=3)
    for value, expected_value in zip(values, expected, strict=True):
        if expected_value is None:
            assert value is None
        else:
            assert value == pytest.approx(expected_value)


def test_increasing_prices_rsi_is_high() -> None:
    """Strictly increasing prices drive RSI toward 100."""
    frame = pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    values = _factor(lookback=3).compute(frame).get_column("rsi").to_list()
    assert values[3] == pytest.approx(100.0)
    assert values[5] == pytest.approx(100.0)


def test_decreasing_prices_rsi_is_zero() -> None:
    """Strictly decreasing prices drive RSI toward 0."""
    frame = pl.DataFrame({"close": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]})
    values = _factor(lookback=3).compute(frame).get_column("rsi").to_list()
    assert values[3] == pytest.approx(0.0)
    assert values[5] == pytest.approx(0.0)


def test_constant_prices_after_warmup_are_null() -> None:
    """Constant prices make average gain and loss zero and RSI null."""
    frame = pl.DataFrame({"close": [10.0, 10.0, 10.0, 10.0, 10.0]})
    values = _factor(lookback=3).compute(frame).get_column("rsi").to_list()
    assert values[3] is None
    assert values[4] is None


def test_null_close_propagates() -> None:
    """Null close values leave incomplete RSI windows null."""
    frame = pl.DataFrame({"close": [10.0, None, 12.0, 13.0, 14.0, 15.0]})
    values = _factor(lookback=3).compute(frame).get_column("rsi").to_list()
    assert values[0] is None
    assert values[1] is None
    assert values[2] is None


def test_validation_immutability_schema_and_exports() -> None:
    """Validation, schema, determinism, and package export contracts."""
    assert_lookback_below_two_raises(
        lambda value: _factor(lookback=value),
        error_code="FACTOR-RSI-001",
    )
    assert_lookback_negative_raises(lambda value: _factor(lookback=value))
    assert_missing_close_raises(
        lambda: _factor(lookback=2),
        error_code="FACTOR-RSI-002",
        factor_name="rsi",
    )
    factor = _factor(lookback=2)
    assert_protocol_and_immutability(factor, output_column="rsi")
    assert_preserves_columns(factor, output_column="rsi")
    assert_output_float64_nullable(factor, output_column="rsi")
    assert_empty_and_single_row(lambda: _factor(lookback=2), output_column="rsi")
    assert_determinism(
        lambda: _factor(lookback=3),
        output_column="rsi",
        frame=pl.DataFrame({"close": [10.0, 11.0, 10.5, 12.0, 11.5]}),
    )
    large = _factor(lookback=50).compute(pl.DataFrame({"close": [float(i) for i in range(1, 11)]}))
    assert all(value is None for value in large.get_column("rsi").to_list())
    assert RSIFactor is RSIFactorDirect
