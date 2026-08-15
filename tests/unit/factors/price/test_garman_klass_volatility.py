"""Unit tests for CQROS ``GarmanKlassVolatilityFactor``."""

from __future__ import annotations

import math

import polars as pl
import pytest

from cqros.core.exceptions import ValidationError
from cqros.factors.exceptions import FactorError
from cqros.factors.price import GarmanKlassVolatilityFactor
from cqros.factors.price.garman_klass_volatility import (
    GarmanKlassVolatilityFactor as GarmanKlassVolatilityFactorDirect,
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


def _factor(*, lookback: int = 20) -> GarmanKlassVolatilityFactor:
    """Build a Garman-Klass volatility factor with an optional lookback override."""
    return GarmanKlassVolatilityFactor(lookback=lookback)


def _ohlc_frame() -> pl.DataFrame:
    """Return a small OHLC fixture for Garman-Klass tests."""
    return pl.DataFrame(
        {
            "open": [10.0, 11.0, 12.0, 13.0, 14.0],
            "high": [11.0, 13.0, 12.5, 15.0, 14.5],
            "low": [9.0, 10.0, 11.0, 12.0, 13.0],
            "close": [10.5, 12.0, 11.5, 14.0, 13.5],
            "volume": [1, 2, 3, 4, 5],
        }
    )


def _garman_klass(
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    *,
    lookback: int,
    index: int,
) -> float | None:
    """Return Garman-Klass volatility at ``index`` for a fully observed window."""
    if index < lookback - 1:
        return None
    coeff = 2.0 * math.log(2.0) - 1.0
    variances: list[float] = []
    for offset in range(index - lookback + 1, index + 1):
        log_hl_sq = math.log(highs[offset] / lows[offset]) ** 2
        log_co_sq = math.log(closes[offset] / opens[offset]) ** 2
        variances.append(0.5 * log_hl_sq - coeff * log_co_sq)
    mean_variance = sum(variances) / lookback
    if mean_variance < 0.0:
        return None
    return math.sqrt(mean_variance)


def test_garman_klass_volatility_metadata() -> None:
    """GarmanKlassVolatilityFactor exposes the fixed production metadata contract."""
    factor = _factor()
    assert factor.name == "garman_klass_volatility"
    assert factor.version == "1.0.0"
    assert factor.category == "price"
    assert factor.required_features == ("open", "high", "low", "close")
    assert factor.produced_columns == ("garman_klass_volatility",)
    assert factor.lookback == 20
    assert factor.metadata.required_features == ("open", "high", "low", "close")


def test_garman_klass_volatility_calculation_correctness() -> None:
    """Garman-Klass volatility matches the OHLC estimator."""
    frame = _ohlc_frame()
    opens = frame.get_column("open").to_list()
    highs = frame.get_column("high").to_list()
    lows = frame.get_column("low").to_list()
    closes = frame.get_column("close").to_list()
    values = _factor(lookback=3).compute(frame).get_column("garman_klass_volatility").to_list()
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx(
        _garman_klass(opens, highs, lows, closes, lookback=3, index=2)
    )
    assert values[3] == pytest.approx(
        _garman_klass(opens, highs, lows, closes, lookback=3, index=3)
    )


def test_constant_prices_zero_volatility() -> None:
    """Constant OHLC produces zero Garman-Klass volatility."""
    frame = pl.DataFrame(
        {
            "open": [10.0, 10.0, 10.0],
            "high": [10.0, 10.0, 10.0],
            "low": [10.0, 10.0, 10.0],
            "close": [10.0, 10.0, 10.0],
        }
    )
    values = _factor(lookback=2).compute(frame).get_column("garman_klass_volatility").to_list()
    assert values[1] == pytest.approx(0.0)
    assert values[2] == pytest.approx(0.0)


def test_increasing_and_decreasing_prices() -> None:
    """Trending OHLC produces non-negative Garman-Klass volatility."""
    rising = pl.DataFrame(
        {
            "open": [1.0, 2.0, 3.0, 4.0],
            "high": [1.5, 2.5, 3.5, 4.5],
            "low": [0.8, 1.8, 2.8, 3.8],
            "close": [1.2, 2.2, 3.2, 4.2],
        }
    )
    falling = pl.DataFrame(
        {
            "open": [4.0, 3.0, 2.0, 1.0],
            "high": [4.5, 3.5, 2.5, 1.5],
            "low": [3.8, 2.8, 1.8, 0.8],
            "close": [4.2, 3.2, 2.2, 1.2],
        }
    )
    assert (
        _factor(lookback=2).compute(rising).get_column("garman_klass_volatility").to_list()[3]
        >= 0.0
    )
    assert (
        _factor(lookback=2).compute(falling).get_column("garman_klass_volatility").to_list()[3]
        >= 0.0
    )


def test_missing_required_columns_raise() -> None:
    """Missing open, high, low, or close raises FactorError."""
    with pytest.raises(FactorError, match="required column missing: open") as exc_info:
        _factor(lookback=2).compute(pl.DataFrame({"high": [1.0], "low": [1.0], "close": [1.0]}))
    assert exc_info.value.error_code == "FACTOR-GARMAN-KLASS-VOLATILITY-002"

    with pytest.raises(FactorError, match="required column missing: close") as exc_info:
        _factor(lookback=2).compute(pl.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0]}))
    assert exc_info.value.error_code == "FACTOR-GARMAN-KLASS-VOLATILITY-002"
    assert exc_info.value.details["factor"] == "garman_klass_volatility"


def test_validation_immutability_schema_and_exports() -> None:
    """Validation, schema, determinism, and package export contracts."""
    assert_lookback_below_two_raises(
        lambda value: _factor(lookback=value),
        error_code="FACTOR-GARMAN-KLASS-VOLATILITY-001",
    )
    assert_lookback_negative_raises(lambda value: _factor(lookback=value))
    with pytest.raises(ValidationError):
        GarmanKlassVolatilityFactor(lookback=0)

    factor = _factor(lookback=2)
    frame = _ohlc_frame()
    assert_protocol_and_immutability(
        factor,
        output_column="garman_klass_volatility",
        frame=frame,
    )
    assert_preserves_columns(
        factor,
        output_column="garman_klass_volatility",
        frame=frame,
    )
    assert_output_float64_nullable(
        factor,
        output_column="garman_klass_volatility",
        frame=frame,
    )
    assert_empty_and_single_row(
        lambda: _factor(lookback=2),
        output_column="garman_klass_volatility",
        columns={"open": [], "high": [], "low": [], "close": []},
    )
    assert_determinism(
        lambda: _factor(lookback=3),
        output_column="garman_klass_volatility",
        frame=frame,
    )
    large = _factor(lookback=50).compute(frame)
    assert all(value is None for value in large.get_column("garman_klass_volatility").to_list())
    assert GarmanKlassVolatilityFactor is GarmanKlassVolatilityFactorDirect
