"""Unit tests for CQROS long/short ratio features."""

from __future__ import annotations

import statistics

import polars as pl
import pytest

from cqros.features.exceptions import FeatureExecutionError
from cqros.features.long_short import (
    CrowdingScoreFeature,
    RatioChangeFeature,
    RatioMomentumFeature,
    RatioZScoreFeature,
)
from tests.unit.features._helpers import (
    assert_lookback_zero_raises,
    assert_null_prefix,
    assert_protocol_and_immutability,
)


def _ratios() -> pl.DataFrame:
    """Build a deterministic long/short ratio fixture."""
    return pl.DataFrame({"long_short_ratio": [0.8, 1.0, 1.2, 1.4, 1.6]})


def test_ratio_change() -> None:
    """Ratio change is the one-period absolute difference."""
    result = RatioChangeFeature().transform(_ratios())
    values = result.get_column("ratio_change").to_list()
    assert values[0] is None
    assert values[1:] == pytest.approx([0.2, 0.2, 0.2, 0.2])
    assert RatioChangeFeature().category == "long_short"
    assert RatioChangeFeature().lookback == 1


def test_ratio_momentum() -> None:
    """Ratio momentum is the absolute change over lookback."""
    result = RatioMomentumFeature(lookback=2).transform(_ratios())
    values = result.get_column("ratio_momentum").to_list()
    assert_null_prefix(values, 2)
    assert values[2] == pytest.approx(0.4)
    assert values[3] == pytest.approx(0.4)


def test_ratio_zscore() -> None:
    """Ratio z-score is (ratio - mean) / std; zero std yields null."""
    ratios = [0.8, 1.0, 1.2, 1.4]
    result = RatioZScoreFeature(lookback=3).transform(pl.DataFrame({"long_short_ratio": ratios}))
    values = result.get_column("ratio_zscore").to_list()
    assert_null_prefix(values, 2)
    mean = statistics.fmean(ratios[0:3])
    std = statistics.stdev(ratios[0:3])
    assert values[2] == pytest.approx((ratios[2] - mean) / std)
    constant = RatioZScoreFeature(lookback=3).transform(
        pl.DataFrame({"long_short_ratio": [1.0, 1.0, 1.0]})
    )
    assert constant.get_column("ratio_zscore").to_list()[2] is None


def test_crowding_score() -> None:
    """Crowding score is (ratio - 1.0) / rolling_std."""
    ratios = [0.8, 1.0, 1.2, 1.4]
    result = CrowdingScoreFeature(lookback=3).transform(pl.DataFrame({"long_short_ratio": ratios}))
    values = result.get_column("crowding_score").to_list()
    assert_null_prefix(values, 2)
    std = statistics.stdev(ratios[0:3])
    assert values[2] == pytest.approx((ratios[2] - 1.0) / std)
    constant = CrowdingScoreFeature(lookback=3).transform(
        pl.DataFrame({"long_short_ratio": [1.1, 1.1, 1.1]})
    )
    assert constant.get_column("crowding_score").to_list()[2] is None


@pytest.mark.parametrize(
    ("factory", "error_code"),
    [
        (RatioChangeFeature, "FEATURE-RATIO-CHANGE-001"),
        (lambda: RatioMomentumFeature(lookback=2), "FEATURE-RATIO-MOMENTUM-002"),
        (lambda: RatioZScoreFeature(lookback=2), "FEATURE-RATIO-ZSCORE-002"),
        (lambda: CrowdingScoreFeature(lookback=2), "FEATURE-CROWDING-SCORE-002"),
    ],
)
def test_long_short_missing_column(factory: object, error_code: str) -> None:
    """Missing long_short_ratio raises FeatureExecutionError."""
    feature = factory()  # type: ignore[operator]
    with pytest.raises(
        FeatureExecutionError,
        match="required column missing: long_short_ratio",
    ) as exc_info:
        feature.transform(pl.DataFrame({"long_account": [0.5, 0.6]}))
    assert exc_info.value.error_code == error_code


def test_long_short_lookback_validation() -> None:
    """Windowed long/short features reject zero lookback."""
    assert_lookback_zero_raises(
        lambda lookback: RatioMomentumFeature(lookback=lookback),
        error_code="FEATURE-RATIO-MOMENTUM-001",
    )
    assert_lookback_zero_raises(
        lambda lookback: RatioZScoreFeature(lookback=lookback),
        error_code="FEATURE-RATIO-ZSCORE-001",
    )
    assert_lookback_zero_raises(
        lambda lookback: CrowdingScoreFeature(lookback=lookback),
        error_code="FEATURE-CROWDING-SCORE-001",
    )


def test_long_short_immutability_and_exports() -> None:
    """Long/short transforms are immutable and package-exported."""
    assert_protocol_and_immutability(
        RatioChangeFeature(),
        output_column="ratio_change",
        frame=_ratios(),
    )
    import cqros.features as features_package
    import cqros.features.long_short as long_short_package

    for name in (
        "RatioChangeFeature",
        "RatioMomentumFeature",
        "RatioZScoreFeature",
        "CrowdingScoreFeature",
    ):
        assert name in long_short_package.__all__
        assert name in features_package.__all__
