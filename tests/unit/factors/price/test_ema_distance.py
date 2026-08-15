"""Unit tests for CQROS ``EMADistanceFactor``."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.factors.price import EMADistanceFactor
from cqros.factors.price.ema_distance import EMADistanceFactor as EMADistanceFactorDirect
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


def _factor(*, lookback: int = 20) -> EMADistanceFactor:
    """Build an EMA distance factor with an optional lookback override."""
    return EMADistanceFactor(lookback=lookback)


def _expected_ema_distance(closes: list[float], *, lookback: int) -> list[float | None]:
    """Compute expected EMA distance with the factor's EMA settings."""
    frame = pl.DataFrame({"close": closes})
    ema = (
        frame.select(
            pl.col("close").ewm_mean(span=lookback, adjust=False, min_samples=lookback).alias("ema")
        )
        .get_column("ema")
        .to_list()
    )
    output: list[float | None] = []
    for close, ema_value in zip(closes, ema, strict=True):
        if ema_value is None or ema_value == 0.0:
            output.append(None)
        else:
            output.append((close - ema_value) / ema_value)
    return output


def test_ema_distance_metadata() -> None:
    """EMADistanceFactor exposes the fixed production metadata contract."""
    factor = _factor()
    assert factor.name == "ema_distance"
    assert factor.version == "1.0.0"
    assert factor.category == "price"
    assert factor.required_features == ("close",)
    assert factor.produced_columns == ("ema_distance",)
    assert factor.lookback == 20
    assert factor.metadata.name == "ema_distance"


def test_ema_distance_calculation_correctness() -> None:
    """EMA distance matches (close - EMA) / EMA with span warm-up."""
    closes = [10.0, 11.0, 12.0, 13.0, 14.0]
    values = (
        _factor(lookback=3)
        .compute(pl.DataFrame({"close": closes}))
        .get_column("ema_distance")
        .to_list()
    )
    expected = _expected_ema_distance(closes, lookback=3)
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx(expected[2])
    assert values[3] == pytest.approx(expected[3])
    assert values[4] == pytest.approx(expected[4])


def test_warmup_nulls_match_lookback_minus_one() -> None:
    """The first lookback - 1 EMA distance values are null."""
    frame = pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0]})
    values = _factor(lookback=4).compute(frame).get_column("ema_distance").to_list()
    assert values[0] is None
    assert values[1] is None
    assert values[2] is None
    assert values[3] is not None


def test_constant_prices_zero_distance() -> None:
    """Constant prices yield zero EMA distance after warm-up."""
    frame = pl.DataFrame({"close": [10.0, 10.0, 10.0, 10.0]})
    values = _factor(lookback=3).compute(frame).get_column("ema_distance").to_list()
    assert values[3] == pytest.approx(0.0)


def test_validation_immutability_schema_and_exports() -> None:
    """Validation, schema, determinism, and package export contracts."""
    assert_lookback_below_two_raises(
        lambda value: _factor(lookback=value),
        error_code="FACTOR-EMA-DISTANCE-001",
    )
    assert_lookback_negative_raises(lambda value: _factor(lookback=value))
    assert_missing_close_raises(
        lambda: _factor(lookback=2),
        error_code="FACTOR-EMA-DISTANCE-002",
        factor_name="ema_distance",
    )
    factor = _factor(lookback=2)
    assert_protocol_and_immutability(factor, output_column="ema_distance")
    assert_preserves_columns(factor, output_column="ema_distance")
    assert_output_float64_nullable(factor, output_column="ema_distance")
    assert_empty_and_single_row(lambda: _factor(lookback=2), output_column="ema_distance")
    assert_determinism(
        lambda: _factor(lookback=3),
        output_column="ema_distance",
        frame=pl.DataFrame({"close": [10.0, 11.0, 12.0, 13.0]}),
    )
    assert EMADistanceFactor is EMADistanceFactorDirect
