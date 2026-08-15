"""Unit tests for CQROS ``ChoppinessIndexFactor``."""

from __future__ import annotations

import math

import polars as pl
import pytest

from cqros.core.exceptions import ValidationError
from cqros.factors.exceptions import FactorError
from cqros.factors.price import ChoppinessIndexFactor
from cqros.factors.price.choppiness_index import (
    ChoppinessIndexFactor as ChoppinessIndexFactorDirect,
)
from tests.unit.factors.price._helpers import (
    assert_determinism,
    assert_empty_and_single_row,
    assert_lookback_below_two_raises,
    assert_lookback_negative_raises,
    assert_output_float64_nullable,
    assert_preserves_columns,
    assert_protocol_and_immutability,
)


def _factor(*, lookback: int = 20) -> ChoppinessIndexFactor:
    """Build a Choppiness Index factor with an optional lookback override."""
    return ChoppinessIndexFactor(lookback=lookback)


def _ohlc_frame() -> pl.DataFrame:
    """Return a small OHLC fixture for Choppiness Index tests."""
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


def _choppiness(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    *,
    lookback: int,
    index: int,
) -> float | None:
    """Return Choppiness Index at ``index`` for a fully observed window."""
    if index < lookback - 1:
        return None
    true_ranges = _true_ranges(highs, lows, closes)
    window_tr = true_ranges[index - lookback + 1 : index + 1]
    window_high = max(highs[index - lookback + 1 : index + 1])
    window_low = min(lows[index - lookback + 1 : index + 1])
    price_range = window_high - window_low
    if price_range == 0.0:
        return None
    return 100.0 * math.log10(sum(window_tr) / price_range) / math.log10(lookback)


def test_choppiness_index_metadata() -> None:
    """ChoppinessIndexFactor exposes the fixed production metadata contract."""
    factor = _factor()
    assert factor.name == "choppiness_index"
    assert factor.version == "1.0.0"
    assert factor.category == "price"
    assert factor.required_features == ("high", "low", "close")
    assert factor.produced_columns == ("choppiness_index",)
    assert factor.lookback == 20
    assert factor.metadata.name == "choppiness_index"


def test_choppiness_index_calculation_correctness() -> None:
    """Choppiness Index matches the standard formula."""
    frame = _ohlc_frame()
    highs = frame.get_column("high").to_list()
    lows = frame.get_column("low").to_list()
    closes = frame.get_column("close").to_list()
    values = _factor(lookback=3).compute(frame).get_column("choppiness_index").to_list()
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx(_choppiness(highs, lows, closes, lookback=3, index=2))
    assert values[3] == pytest.approx(_choppiness(highs, lows, closes, lookback=3, index=3))


def test_zero_range_is_null() -> None:
    """Zero high-low range returns null Choppiness Index."""
    frame = pl.DataFrame(
        {
            "high": [10.0, 10.0, 10.0],
            "low": [10.0, 10.0, 10.0],
            "close": [10.0, 10.0, 10.0],
        }
    )
    values = _factor(lookback=2).compute(frame).get_column("choppiness_index").to_list()
    assert values[1] is None
    assert values[2] is None


def test_increasing_and_decreasing_prices() -> None:
    """Trending OHLC produces finite Choppiness Index values."""
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
    assert (
        _factor(lookback=2).compute(rising).get_column("choppiness_index").to_list()[3] is not None
    )
    assert (
        _factor(lookback=2).compute(falling).get_column("choppiness_index").to_list()[3] is not None
    )


def test_missing_required_columns_raise() -> None:
    """Missing high, low, or close raises FactorError."""
    with pytest.raises(FactorError, match="required column missing: high") as exc_info:
        _factor(lookback=2).compute(pl.DataFrame({"low": [1.0], "close": [1.0]}))
    assert exc_info.value.error_code == "FACTOR-CHOPPINESS-INDEX-002"

    with pytest.raises(FactorError, match="required column missing: close") as exc_info:
        _factor(lookback=2).compute(pl.DataFrame({"high": [1.0], "low": [1.0]}))
    assert exc_info.value.error_code == "FACTOR-CHOPPINESS-INDEX-002"


def test_validation_immutability_schema_and_exports() -> None:
    """Validation, schema, determinism, and package export contracts."""
    assert_lookback_below_two_raises(
        lambda value: _factor(lookback=value),
        error_code="FACTOR-CHOPPINESS-INDEX-001",
    )
    assert_lookback_negative_raises(lambda value: _factor(lookback=value))
    with pytest.raises(ValidationError):
        ChoppinessIndexFactor(lookback=0)

    factor = _factor(lookback=2)
    frame = _ohlc_frame()
    assert_protocol_and_immutability(factor, output_column="choppiness_index", frame=frame)
    assert_preserves_columns(
        factor,
        output_column="choppiness_index",
        frame=frame.select(["high", "low", "close", "volume"]),
    )
    assert_output_float64_nullable(factor, output_column="choppiness_index")
    assert_empty_and_single_row(
        lambda: _factor(lookback=2),
        output_column="choppiness_index",
        columns={"high": [], "low": [], "close": []},
    )
    assert_determinism(
        lambda: _factor(lookback=3),
        output_column="choppiness_index",
        frame=frame,
    )
    large = _factor(lookback=50).compute(frame)
    assert all(value is None for value in large.get_column("choppiness_index").to_list())
    assert ChoppinessIndexFactor is ChoppinessIndexFactorDirect
