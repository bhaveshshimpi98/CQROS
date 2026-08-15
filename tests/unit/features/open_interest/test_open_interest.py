"""Unit tests for CQROS open interest features."""

from __future__ import annotations

import statistics

import polars as pl
import pytest

from cqros.features.exceptions import FeatureExecutionError
from cqros.features.open_interest import (
    OIChangeFeature,
    OIMomentumFeature,
    OIPercentChangeFeature,
    OIRollingMeanFeature,
    OIZScoreFeature,
)
from tests.unit.features._helpers import (
    assert_lookback_zero_raises,
    assert_null_prefix,
    assert_protocol_and_immutability,
)


def _oi() -> pl.DataFrame:
    """Build a deterministic open-interest fixture."""
    return pl.DataFrame({"open_interest": [100.0, 110.0, 121.0, 133.1, 146.41]})


def test_oi_change() -> None:
    """OI change is the one-period absolute difference."""
    result = OIChangeFeature().transform(_oi())
    values = result.get_column("oi_change").to_list()
    assert values[0] is None
    assert values[1] == pytest.approx(10.0)
    assert values[2] == pytest.approx(11.0)
    assert OIChangeFeature().category == "open_interest"
    assert OIChangeFeature().lookback == 1


def test_oi_percent_change() -> None:
    """OI percent change matches (oi / previous) - 1."""
    result = OIPercentChangeFeature().transform(_oi())
    values = result.get_column("oi_percent_change").to_list()
    assert values[0] is None
    assert values[1] == pytest.approx(0.10)
    assert values[2] == pytest.approx(0.10)


def test_oi_rolling_mean() -> None:
    """OI rolling mean matches window arithmetic mean."""
    values_in = [100.0, 110.0, 120.0, 130.0]
    result = OIRollingMeanFeature(lookback=3).transform(pl.DataFrame({"open_interest": values_in}))
    values = result.get_column("oi_rolling_mean").to_list()
    assert_null_prefix(values, 2)
    assert values[2] == pytest.approx(statistics.fmean(values_in[0:3]))


def test_oi_zscore() -> None:
    """OI z-score is (oi - mean) / std; zero std yields null."""
    values_in = [100.0, 110.0, 120.0, 130.0]
    result = OIZScoreFeature(lookback=3).transform(pl.DataFrame({"open_interest": values_in}))
    values = result.get_column("oi_zscore").to_list()
    assert_null_prefix(values, 2)
    mean = statistics.fmean(values_in[0:3])
    std = statistics.stdev(values_in[0:3])
    assert values[2] == pytest.approx((values_in[2] - mean) / std)
    constant = OIZScoreFeature(lookback=3).transform(
        pl.DataFrame({"open_interest": [10.0, 10.0, 10.0]})
    )
    assert constant.get_column("oi_zscore").to_list()[2] is None


def test_oi_momentum() -> None:
    """OI momentum is percent change over lookback."""
    result = OIMomentumFeature(lookback=2).transform(_oi())
    values = result.get_column("oi_momentum").to_list()
    assert_null_prefix(values, 2)
    assert values[2] == pytest.approx((121.0 / 100.0) - 1.0)
    assert values[3] == pytest.approx((133.1 / 110.0) - 1.0)


@pytest.mark.parametrize(
    ("factory", "error_code"),
    [
        (OIChangeFeature, "FEATURE-OI-CHANGE-001"),
        (OIPercentChangeFeature, "FEATURE-OI-PERCENT-CHANGE-001"),
        (lambda: OIRollingMeanFeature(lookback=2), "FEATURE-OI-ROLLING-MEAN-002"),
        (lambda: OIZScoreFeature(lookback=2), "FEATURE-OI-ZSCORE-002"),
        (lambda: OIMomentumFeature(lookback=2), "FEATURE-OI-MOMENTUM-002"),
    ],
)
def test_oi_missing_column(factory: object, error_code: str) -> None:
    """Missing open_interest raises FeatureExecutionError."""
    feature = factory()  # type: ignore[operator]
    with pytest.raises(
        FeatureExecutionError,
        match="required column missing: open_interest",
    ) as exc_info:
        feature.transform(pl.DataFrame({"symbol": ["BTCUSDT", "BTCUSDT"]}))
    assert exc_info.value.error_code == error_code


def test_oi_lookback_validation() -> None:
    """Windowed OI features reject zero lookback."""
    assert_lookback_zero_raises(
        lambda lookback: OIRollingMeanFeature(lookback=lookback),
        error_code="FEATURE-OI-ROLLING-MEAN-001",
    )
    assert_lookback_zero_raises(
        lambda lookback: OIZScoreFeature(lookback=lookback),
        error_code="FEATURE-OI-ZSCORE-001",
    )
    assert_lookback_zero_raises(
        lambda lookback: OIMomentumFeature(lookback=lookback),
        error_code="FEATURE-OI-MOMENTUM-001",
    )


def test_oi_immutability_and_exports() -> None:
    """OI transforms are immutable and package-exported."""
    assert_protocol_and_immutability(
        OIChangeFeature(),
        output_column="oi_change",
        frame=_oi(),
    )
    import cqros.features as features_package
    import cqros.features.open_interest as oi_package

    for name in (
        "OIChangeFeature",
        "OIPercentChangeFeature",
        "OIRollingMeanFeature",
        "OIZScoreFeature",
        "OIMomentumFeature",
    ):
        assert name in oi_package.__all__
        assert name in features_package.__all__
