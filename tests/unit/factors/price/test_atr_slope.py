"""Unit tests for CQROS ``ATRSlopeFactor``."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.core.exceptions import ValidationError
from cqros.factors.exceptions import FactorError
from cqros.factors.price import ATRSlopeFactor
from cqros.factors.price.atr_slope import ATRSlopeFactor as ATRSlopeFactorDirect
from tests.unit.factors.price._helpers import (
    assert_determinism,
    assert_empty_and_single_row,
    assert_lookback_below_two_raises,
    assert_lookback_negative_raises,
    assert_output_float64_nullable,
    assert_preserves_columns,
    assert_protocol_and_immutability,
)


def _factor(*, lookback: int = 20) -> ATRSlopeFactor:
    """Build an ATR slope factor with an optional lookback override."""
    return ATRSlopeFactor(lookback=lookback)


def _ohlc_frame() -> pl.DataFrame:
    """Return a small OHLC fixture for ATR slope tests."""
    return pl.DataFrame(
        {
            "high": [12.0, 15.0, 14.0, 18.0, 20.0, 22.0],
            "low": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "close": [11.0, 14.0, 13.0, 16.0, 18.0, 20.0],
            "volume": [1, 2, 3, 4, 5, 6],
        }
    )


def _true_ranges(highs: list[float], lows: list[float], closes: list[float]) -> list[float]:
    """Return true range series matching CQROS ATRFeature semantics."""
    values: list[float] = []
    for index, (high, low) in enumerate(zip(highs, lows, strict=True)):
        candidates = [high - low]
        if index > 0:
            prev_close = closes[index - 1]
            candidates.append(abs(high - prev_close))
            candidates.append(abs(low - prev_close))
        values.append(max(candidates))
    return values


def _atr_series(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    *,
    lookback: int,
) -> list[float | None]:
    """Return rolling ATR values."""
    true_ranges = _true_ranges(highs, lows, closes)
    values: list[float | None] = []
    for index in range(len(true_ranges)):
        if index < lookback - 1:
            values.append(None)
        else:
            window = true_ranges[index - lookback + 1 : index + 1]
            values.append(sum(window) / lookback)
    return values


def _ols_slope(values: list[float]) -> float:
    """Return OLS slope of values on relative index 0..n-1."""
    n = len(values)
    xs = list(range(n))
    sum_x = sum(xs)
    sum_y = sum(values)
    sum_xy = sum(x * y for x, y in zip(xs, values, strict=True))
    sum_x2 = sum(x * x for x in xs)
    return (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)


def test_atr_slope_metadata() -> None:
    """ATRSlopeFactor exposes the fixed production metadata contract."""
    factor = _factor()
    assert factor.name == "atr_slope"
    assert factor.version == "1.0.0"
    assert factor.category == "price"
    assert factor.required_features == ("high", "low", "close")
    assert factor.produced_columns == ("atr_slope",)
    assert factor.lookback == 20
    assert factor.metadata.name == "atr_slope"


def test_atr_slope_calculation_correctness() -> None:
    """ATR slope matches rolling OLS slope of ATR."""
    frame = _ohlc_frame()
    highs = frame.get_column("high").to_list()
    lows = frame.get_column("low").to_list()
    closes = frame.get_column("close").to_list()
    atr = _atr_series(highs, lows, closes, lookback=2)
    values = _factor(lookback=2).compute(frame).get_column("atr_slope").to_list()
    assert values[0] is None
    assert values[1] is None
    assert atr[1] is not None and atr[2] is not None
    assert values[2] == pytest.approx(_ols_slope([atr[1], atr[2]]))
    assert atr[2] is not None and atr[3] is not None
    assert values[3] == pytest.approx(_ols_slope([atr[2], atr[3]]))


def test_constant_prices_zero_slope() -> None:
    """Constant OHLC yields near-zero ATR slope after warm-up."""
    frame = pl.DataFrame(
        {
            "high": [10.0, 10.0, 10.0, 10.0],
            "low": [10.0, 10.0, 10.0, 10.0],
            "close": [10.0, 10.0, 10.0, 10.0],
        }
    )
    values = _factor(lookback=2).compute(frame).get_column("atr_slope").to_list()
    assert values[3] == pytest.approx(0.0)


def test_increasing_volatility_positive_slope() -> None:
    """Widening ranges produce a positive ATR slope after warm-up."""
    frame = pl.DataFrame(
        {
            "high": [10.5, 11.0, 13.0, 16.0, 20.0],
            "low": [9.5, 10.0, 10.0, 10.0, 10.0],
            "close": [10.0, 10.5, 12.0, 14.0, 18.0],
        }
    )
    values = _factor(lookback=2).compute(frame).get_column("atr_slope").to_list()
    assert values[4] is not None
    assert values[4] > 0.0


def test_missing_required_columns_raise() -> None:
    """Missing high, low, or close raises FactorError."""
    with pytest.raises(FactorError, match="required column missing: high") as exc_info:
        _factor(lookback=2).compute(pl.DataFrame({"low": [1.0], "close": [1.0]}))
    assert exc_info.value.error_code == "FACTOR-ATR-SLOPE-002"

    with pytest.raises(FactorError, match="required column missing: close") as exc_info:
        _factor(lookback=2).compute(pl.DataFrame({"high": [1.0], "low": [1.0]}))
    assert exc_info.value.error_code == "FACTOR-ATR-SLOPE-002"


def test_validation_immutability_schema_and_exports() -> None:
    """Validation, schema, determinism, and package export contracts."""
    assert_lookback_below_two_raises(
        lambda value: _factor(lookback=value),
        error_code="FACTOR-ATR-SLOPE-001",
    )
    assert_lookback_negative_raises(lambda value: _factor(lookback=value))
    with pytest.raises(ValidationError):
        ATRSlopeFactor(lookback=0)

    factor = _factor(lookback=2)
    frame = _ohlc_frame()
    assert_protocol_and_immutability(factor, output_column="atr_slope", frame=frame)
    assert_preserves_columns(
        factor,
        output_column="atr_slope",
        frame=frame.select(["high", "low", "close", "volume"]),
    )
    assert_output_float64_nullable(factor, output_column="atr_slope")
    assert_empty_and_single_row(
        lambda: _factor(lookback=2),
        output_column="atr_slope",
        columns={"high": [], "low": [], "close": []},
    )
    assert_determinism(lambda: _factor(lookback=2), output_column="atr_slope", frame=frame)
    large = _factor(lookback=50).compute(frame)
    assert all(value is None for value in large.get_column("atr_slope").to_list())
    assert ATRSlopeFactor is ATRSlopeFactorDirect
