"""Unit tests for CQROS ``ATRPercentFactor``."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.core.exceptions import ValidationError
from cqros.factors.exceptions import FactorError
from cqros.factors.price import ATRPercentFactor
from cqros.factors.price.atr_percent import ATRPercentFactor as ATRPercentFactorDirect
from tests.unit.factors.price._helpers import (
    assert_determinism,
    assert_empty_and_single_row,
    assert_lookback_below_two_raises,
    assert_lookback_negative_raises,
    assert_output_float64_nullable,
    assert_preserves_columns,
    assert_protocol_and_immutability,
)


def _factor(*, lookback: int = 20) -> ATRPercentFactor:
    """Build an ATR percent factor with an optional lookback override."""
    return ATRPercentFactor(lookback=lookback)


def _ohlc_frame() -> pl.DataFrame:
    """Return a small OHLC fixture for ATR percent tests."""
    return pl.DataFrame(
        {
            "high": [12.0, 15.0, 14.0, 16.0, 17.0],
            "low": [10.0, 11.0, 12.0, 13.0, 14.0],
            "close": [11.0, 14.0, 13.0, 15.0, 16.0],
            "volume": [1, 2, 3, 4, 5],
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


def test_atr_percent_metadata() -> None:
    """ATRPercentFactor exposes the fixed production metadata contract."""
    factor = _factor()
    assert factor.name == "atr_percent"
    assert factor.version == "1.0.0"
    assert factor.category == "price"
    assert factor.required_features == ("high", "low", "close")
    assert factor.produced_columns == ("atr_percent",)
    assert factor.lookback == 20
    assert factor.metadata.name == "atr_percent"


def test_atr_percent_calculation_correctness() -> None:
    """ATR percent matches ATR divided by close."""
    frame = _ohlc_frame()
    highs = frame.get_column("high").to_list()
    lows = frame.get_column("low").to_list()
    closes = frame.get_column("close").to_list()
    true_ranges = _true_ranges(highs, lows, closes)
    values = _factor(lookback=2).compute(frame).get_column("atr_percent").to_list()
    assert values[0] is None
    atr1 = (true_ranges[0] + true_ranges[1]) / 2.0
    atr2 = (true_ranges[1] + true_ranges[2]) / 2.0
    assert values[1] == pytest.approx(atr1 / closes[1])
    assert values[2] == pytest.approx(atr2 / closes[2])


def test_zero_close_is_null() -> None:
    """Zero close returns null ATR percent."""
    frame = pl.DataFrame(
        {
            "high": [2.0, 3.0, 1.0],
            "low": [1.0, 1.0, 0.0],
            "close": [1.5, 2.0, 0.0],
        }
    )
    values = _factor(lookback=2).compute(frame).get_column("atr_percent").to_list()
    assert values[2] is None


def test_constant_prices_zero_percent() -> None:
    """Constant OHLC makes ATR zero and percent zero."""
    frame = pl.DataFrame(
        {
            "high": [10.0, 10.0, 10.0],
            "low": [10.0, 10.0, 10.0],
            "close": [10.0, 10.0, 10.0],
        }
    )
    values = _factor(lookback=2).compute(frame).get_column("atr_percent").to_list()
    assert values[1] == pytest.approx(0.0)
    assert values[2] == pytest.approx(0.0)


def test_increasing_and_decreasing_prices() -> None:
    """Rising and falling OHLC produce non-negative ATR percent."""
    rising = pl.DataFrame(
        {
            "high": [2.0, 3.0, 4.0, 5.0],
            "low": [1.0, 2.0, 3.0, 4.0],
            "close": [1.5, 2.5, 3.5, 4.5],
        }
    )
    falling = pl.DataFrame(
        {
            "high": [5.0, 4.0, 3.0, 2.0],
            "low": [4.0, 3.0, 2.0, 1.0],
            "close": [4.5, 3.5, 2.5, 1.5],
        }
    )
    assert _factor(lookback=2).compute(rising).get_column("atr_percent").to_list()[3] >= 0.0
    assert _factor(lookback=2).compute(falling).get_column("atr_percent").to_list()[3] >= 0.0


def test_missing_required_columns_raise() -> None:
    """Missing high, low, or close raises FactorError."""
    with pytest.raises(FactorError, match="required column missing: high") as exc_info:
        _factor(lookback=2).compute(pl.DataFrame({"low": [1.0], "close": [1.0]}))
    assert exc_info.value.error_code == "FACTOR-ATR-PERCENT-002"

    with pytest.raises(FactorError, match="required column missing: close") as exc_info:
        _factor(lookback=2).compute(pl.DataFrame({"high": [1.0], "low": [1.0]}))
    assert exc_info.value.error_code == "FACTOR-ATR-PERCENT-002"


def test_validation_immutability_schema_and_exports() -> None:
    """Validation, schema, determinism, and package export contracts."""
    assert_lookback_below_two_raises(
        lambda value: _factor(lookback=value),
        error_code="FACTOR-ATR-PERCENT-001",
    )
    assert_lookback_negative_raises(lambda value: _factor(lookback=value))
    with pytest.raises(ValidationError):
        ATRPercentFactor(lookback=0)

    factor = _factor(lookback=2)
    frame = _ohlc_frame()
    assert_protocol_and_immutability(factor, output_column="atr_percent", frame=frame)
    assert_preserves_columns(
        factor,
        output_column="atr_percent",
        frame=frame.select(["high", "low", "close", "volume"]),
    )
    assert_output_float64_nullable(factor, output_column="atr_percent")
    assert_empty_and_single_row(
        lambda: _factor(lookback=2),
        output_column="atr_percent",
        columns={"high": [], "low": [], "close": []},
    )
    assert_determinism(lambda: _factor(lookback=2), output_column="atr_percent", frame=frame)
    large = _factor(lookback=50).compute(frame)
    assert all(value is None for value in large.get_column("atr_percent").to_list())
    assert ATRPercentFactor is ATRPercentFactorDirect
