"""Unit tests for CQROS ``MeanReversionScoreFactor``."""

from __future__ import annotations

import math

import polars as pl
import pytest

from cqros.factors.price import MeanReversionScoreFactor
from cqros.factors.price.mean_reversion_score import (
    MeanReversionScoreFactor as MeanReversionScoreFactorDirect,
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


def _factor(*, lookback: int = 20) -> MeanReversionScoreFactor:
    """Build a mean reversion score factor with an optional lookback override."""
    return MeanReversionScoreFactor(lookback=lookback)


def _mean_reversion_score(closes: list[float]) -> float | None:
    """Return negative population z-score for a fully observed window."""
    mean = sum(closes) / len(closes)
    variance = sum((value - mean) ** 2 for value in closes) / len(closes)
    std = math.sqrt(variance)
    if std == 0.0:
        return 0.0
    return -(closes[-1] - mean) / std


def test_mean_reversion_score_metadata() -> None:
    """MeanReversionScoreFactor exposes the fixed production metadata contract."""
    factor = _factor()
    assert factor.name == "mean_reversion_score"
    assert factor.version == "1.0.0"
    assert factor.category == "price"
    assert factor.required_features == ("close",)
    assert factor.produced_columns == ("mean_reversion_score",)
    assert factor.lookback == 20
    assert factor.metadata.name == "mean_reversion_score"


def test_mean_reversion_score_calculation_correctness() -> None:
    """Mean reversion score matches the negative rolling z-score."""
    closes = [10.0, 12.0, 11.0, 15.0, 14.0]
    values = (
        _factor(lookback=3)
        .compute(pl.DataFrame({"close": closes}))
        .get_column("mean_reversion_score")
        .to_list()
    )
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx(_mean_reversion_score(closes[0:3]))
    assert values[3] == pytest.approx(_mean_reversion_score(closes[1:4]))
    assert values[4] == pytest.approx(_mean_reversion_score(closes[2:5]))


def test_constant_prices_are_zero() -> None:
    """Constant prices make standard deviation zero and score 0.0."""
    frame = pl.DataFrame({"close": [10.0, 10.0, 10.0, 10.0]})
    values = _factor(lookback=3).compute(frame).get_column("mean_reversion_score").to_list()
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx(0.0)
    assert values[3] == pytest.approx(0.0)


def test_increasing_and_decreasing_prices() -> None:
    """Above-mean closes score negative and below-mean closes score positive."""
    rising = _factor(lookback=3).compute(pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0]}))
    falling = _factor(lookback=3).compute(pl.DataFrame({"close": [4.0, 3.0, 2.0, 1.0]}))
    assert rising.get_column("mean_reversion_score").to_list()[3] < 0.0
    assert falling.get_column("mean_reversion_score").to_list()[3] > 0.0


def test_is_negative_of_price_zscore() -> None:
    """Mean reversion score equals the negated price z-score on identical inputs."""
    from cqros.factors.price import PriceZScoreFactor

    frame = pl.DataFrame({"close": [10.0, 12.0, 11.0, 15.0, 14.0]})
    score = _factor(lookback=3).compute(frame).get_column("mean_reversion_score").to_list()
    zscore = PriceZScoreFactor(lookback=3).compute(frame).get_column("price_zscore").to_list()
    for score_value, zscore_value in zip(score, zscore, strict=True):
        if zscore_value is None:
            assert score_value is None
        else:
            assert score_value == pytest.approx(-zscore_value)


def test_validation_immutability_schema_and_exports() -> None:
    """Validation, schema, determinism, and package export contracts."""
    assert_lookback_below_two_raises(
        lambda value: _factor(lookback=value),
        error_code="FACTOR-MEAN-REVERSION-SCORE-001",
    )
    assert_lookback_negative_raises(lambda value: _factor(lookback=value))
    assert_missing_close_raises(
        lambda: _factor(lookback=2),
        error_code="FACTOR-MEAN-REVERSION-SCORE-002",
        factor_name="mean_reversion_score",
    )
    factor = _factor(lookback=2)
    assert_protocol_and_immutability(factor, output_column="mean_reversion_score")
    assert_preserves_columns(factor, output_column="mean_reversion_score")
    assert_output_float64_nullable(factor, output_column="mean_reversion_score")
    assert_empty_and_single_row(lambda: _factor(lookback=2), output_column="mean_reversion_score")
    assert_determinism(
        lambda: _factor(lookback=3),
        output_column="mean_reversion_score",
        frame=pl.DataFrame({"close": [10.0, 12.0, 11.0, 15.0]}),
    )
    large = _factor(lookback=50).compute(pl.DataFrame({"close": [float(i) for i in range(1, 11)]}))
    assert all(value is None for value in large.get_column("mean_reversion_score").to_list())
    assert MeanReversionScoreFactor is MeanReversionScoreFactorDirect
