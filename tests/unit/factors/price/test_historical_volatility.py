"""Unit tests for CQROS ``HistoricalVolatilityFactor``."""

from __future__ import annotations

import math

import polars as pl
import pytest

from cqros.core.exceptions import ValidationError
from cqros.factors.price import HistoricalVolatilityFactor
from cqros.factors.price.historical_volatility import (
    HistoricalVolatilityFactor as HistoricalVolatilityFactorDirect,
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


def _factor(*, lookback: int = 20, annualization: int = 365) -> HistoricalVolatilityFactor:
    """Build a historical volatility factor with optional overrides."""
    return HistoricalVolatilityFactor(lookback=lookback, annualization=annualization)


def _historical_volatility(
    closes: list[float],
    *,
    lookback: int,
    annualization: int,
    index: int,
) -> float | None:
    """Return annualized historical volatility at ``index``."""
    if index < lookback:
        return None
    log_returns = [
        math.log(closes[offset] / closes[offset - 1])
        for offset in range(index - lookback + 1, index + 1)
    ]
    mean = sum(log_returns) / lookback
    variance = sum((value - mean) ** 2 for value in log_returns) / lookback
    return math.sqrt(variance) * math.sqrt(annualization)


def test_historical_volatility_metadata() -> None:
    """HistoricalVolatilityFactor exposes the fixed production metadata contract."""
    factor = _factor()
    assert factor.name == "historical_volatility"
    assert factor.version == "1.0.0"
    assert factor.category == "price"
    assert factor.required_features == ("close",)
    assert factor.produced_columns == ("historical_volatility",)
    assert factor.lookback == 20
    assert factor.annualization == 365
    assert factor.metadata.name == "historical_volatility"


def test_historical_volatility_calculation_correctness() -> None:
    """Historical volatility matches annualized rolling log-return std."""
    closes = [100.0, 102.0, 101.0, 105.0, 110.0, 108.0]
    values = (
        _factor(lookback=3, annualization=365)
        .compute(pl.DataFrame({"close": closes}))
        .get_column("historical_volatility")
        .to_list()
    )
    assert values[0] is None
    assert values[1] is None
    assert values[2] is None
    assert values[3] == pytest.approx(
        _historical_volatility(closes, lookback=3, annualization=365, index=3)
    )
    assert values[4] == pytest.approx(
        _historical_volatility(closes, lookback=3, annualization=365, index=4)
    )


def test_annualization_scales_output() -> None:
    """Changing annualization scales volatility by sqrt ratio."""
    closes = [100.0, 102.0, 101.0, 105.0, 110.0]
    frame = pl.DataFrame({"close": closes})
    values_365 = (
        _factor(lookback=3, annualization=365)
        .compute(frame)
        .get_column("historical_volatility")
        .to_list()
    )
    values_91 = (
        _factor(lookback=3, annualization=91)
        .compute(frame)
        .get_column("historical_volatility")
        .to_list()
    )
    assert values_365[4] is not None and values_91[4] is not None
    assert values_365[4] == pytest.approx(values_91[4] * math.sqrt(365.0 / 91.0))


def test_constant_prices_zero_volatility() -> None:
    """Constant prices produce zero historical volatility after warm-up."""
    frame = pl.DataFrame({"close": [10.0, 10.0, 10.0, 10.0]})
    values = _factor(lookback=2).compute(frame).get_column("historical_volatility").to_list()
    assert values[2] == pytest.approx(0.0)
    assert values[3] == pytest.approx(0.0)


def test_increasing_and_decreasing_prices() -> None:
    """Monotonic price paths produce positive historical volatility."""
    rising = _factor(lookback=2).compute(pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0]}))
    falling = _factor(lookback=2).compute(pl.DataFrame({"close": [4.0, 3.0, 2.0, 1.0]}))
    assert rising.get_column("historical_volatility").to_list()[3] > 0.0
    assert falling.get_column("historical_volatility").to_list()[3] > 0.0


def test_invalid_annualization_raises() -> None:
    """Non-positive annualization is rejected."""
    with pytest.raises(
        ValidationError,
        match="annualization must be an integer greater than or equal to 1",
    ) as exc_info:
        HistoricalVolatilityFactor(lookback=2, annualization=0)
    assert exc_info.value.error_code == "FACTOR-HISTORICAL-VOLATILITY-002"


def test_validation_immutability_schema_and_exports() -> None:
    """Validation, schema, determinism, and package export contracts."""
    assert_lookback_below_two_raises(
        lambda value: _factor(lookback=value),
        error_code="FACTOR-HISTORICAL-VOLATILITY-001",
    )
    assert_lookback_negative_raises(lambda value: _factor(lookback=value))
    assert_missing_close_raises(
        lambda: _factor(lookback=2),
        error_code="FACTOR-HISTORICAL-VOLATILITY-003",
        factor_name="historical_volatility",
    )
    factor = _factor(lookback=2)
    assert_protocol_and_immutability(factor, output_column="historical_volatility")
    assert_preserves_columns(factor, output_column="historical_volatility")
    assert_output_float64_nullable(factor, output_column="historical_volatility")
    assert_empty_and_single_row(
        lambda: _factor(lookback=2),
        output_column="historical_volatility",
    )
    assert_determinism(
        lambda: _factor(lookback=3),
        output_column="historical_volatility",
        frame=pl.DataFrame({"close": [100.0, 102.0, 101.0, 105.0]}),
    )
    large = _factor(lookback=50).compute(pl.DataFrame({"close": [float(i) for i in range(1, 11)]}))
    assert all(value is None for value in large.get_column("historical_volatility").to_list())
    assert HistoricalVolatilityFactor is HistoricalVolatilityFactorDirect
