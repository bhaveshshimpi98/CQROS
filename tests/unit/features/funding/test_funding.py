"""Unit tests for CQROS funding features."""

from __future__ import annotations

import statistics

import polars as pl
import pytest

from cqros.features.exceptions import FeatureExecutionError
from cqros.features.funding import (
    FundingChangeFeature,
    FundingMomentumFeature,
    FundingRollingMeanFeature,
    FundingZScoreFeature,
)
from tests.unit.features._helpers import (
    assert_lookback_zero_raises,
    assert_null_prefix,
    assert_protocol_and_immutability,
)


def _rates() -> pl.DataFrame:
    """Build a deterministic funding-rate fixture."""
    return pl.DataFrame({"funding_rate": [0.01, 0.02, 0.03, 0.04, 0.05]})


def test_funding_change() -> None:
    """Funding change is the one-period absolute difference."""
    result = FundingChangeFeature().transform(_rates())
    values = result.get_column("funding_change").to_list()
    assert values[0] is None
    assert values[1:] == pytest.approx([0.01, 0.01, 0.01, 0.01])
    assert FundingChangeFeature().name == "funding_change"
    assert FundingChangeFeature().category == "funding"
    assert FundingChangeFeature().lookback == 1


def test_funding_rolling_mean() -> None:
    """Funding rolling mean matches window arithmetic mean."""
    rates = [0.01, 0.02, 0.03, 0.04, 0.05]
    result = FundingRollingMeanFeature(lookback=3).transform(pl.DataFrame({"funding_rate": rates}))
    values = result.get_column("funding_rolling_mean").to_list()
    assert_null_prefix(values, 2)
    assert values[2] == pytest.approx(statistics.fmean(rates[0:3]))
    assert values[4] == pytest.approx(statistics.fmean(rates[2:5]))


def test_funding_zscore() -> None:
    """Funding z-score is (rate - mean) / std; zero std yields 0.0."""
    rates = [0.01, 0.02, 0.03, 0.04, 0.05]
    result = FundingZScoreFeature(lookback=3).transform(pl.DataFrame({"funding_rate": rates}))
    values = result.get_column("funding_zscore").to_list()
    assert_null_prefix(values, 2)
    mean = statistics.fmean(rates[0:3])
    std = statistics.stdev(rates[0:3])
    assert values[2] == pytest.approx((rates[2] - mean) / std)

    constant = FundingZScoreFeature(lookback=3).transform(
        pl.DataFrame({"funding_rate": [0.01, 0.01, 0.01]})
    )
    assert constant.get_column("funding_zscore").to_list()[2] == pytest.approx(0.0)


def test_funding_zscore_constant_window_is_zero() -> None:
    """A fully constant lookback window produces funding_zscore == 0.0."""
    rates = [0.0001] * 8
    lookback = 4
    values = (
        FundingZScoreFeature(lookback=lookback)
        .transform(pl.DataFrame({"funding_rate": rates}))
        .get_column("funding_zscore")
        .to_list()
    )
    assert_null_prefix(values, lookback - 1)
    assert values[lookback - 1 :] == pytest.approx([0.0] * (len(rates) - lookback + 1))


def test_funding_zscore_nonzero_variance_unchanged() -> None:
    """Non-zero variance windows match (rate - mean) / std exactly."""
    rates = [0.01, 0.02, 0.04, 0.03, 0.05, 0.08]
    lookback = 3
    values = (
        FundingZScoreFeature(lookback=lookback)
        .transform(pl.DataFrame({"funding_rate": rates}))
        .get_column("funding_zscore")
        .to_list()
    )
    assert_null_prefix(values, lookback - 1)
    for index in range(lookback - 1, len(rates)):
        window = rates[index - lookback + 1 : index + 1]
        mean = statistics.fmean(window)
        std = statistics.stdev(window)
        assert std > 0
        assert values[index] == pytest.approx((rates[index] - mean) / std)


def test_funding_zscore_mixed_windows() -> None:
    """Zero-std windows emit 0.0 while varying windows keep standard z-scores."""
    # Windows of size 3:
    # idx2: [0.02, 0.02, 0.02] → std 0 → 0.0
    # idx3: [0.02, 0.02, 0.05] → std > 0 → standard z-score
    # idx4: [0.02, 0.05, 0.05] → std > 0 → standard z-score
    # idx5: [0.05, 0.05, 0.05] → std 0 → 0.0
    rates = [0.02, 0.02, 0.02, 0.05, 0.05, 0.05]
    lookback = 3
    values = (
        FundingZScoreFeature(lookback=lookback)
        .transform(pl.DataFrame({"funding_rate": rates}))
        .get_column("funding_zscore")
        .to_list()
    )
    assert_null_prefix(values, lookback - 1)
    assert values[2] == pytest.approx(0.0)
    assert values[5] == pytest.approx(0.0)

    for index in (3, 4):
        window = rates[index - lookback + 1 : index + 1]
        mean = statistics.fmean(window)
        std = statistics.stdev(window)
        assert std > 0
        assert values[index] == pytest.approx((rates[index] - mean) / std)


def test_funding_zscore_warmup_rows_unchanged() -> None:
    """Warm-up prefix remains null for the first lookback - 1 rows."""
    rates = [0.01, 0.01, 0.01, 0.02, 0.03]
    for lookback in (2, 3, 4):
        values = (
            FundingZScoreFeature(lookback=lookback)
            .transform(pl.DataFrame({"funding_rate": rates}))
            .get_column("funding_zscore")
            .to_list()
        )
        assert_null_prefix(values, lookback - 1)
        assert all(value is not None for value in values[lookback - 1 :])


def test_funding_zscore_introduces_no_new_nulls_after_warmup() -> None:
    """After warm-up, constant or mixed series never produce additional nulls."""
    cases = (
        [0.0] * 10,
        [0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.03],
        [0.01, 0.02, 0.03, 0.04, 0.05],
    )
    lookback = 3
    for rates in cases:
        values = (
            FundingZScoreFeature(lookback=lookback)
            .transform(pl.DataFrame({"funding_rate": rates}))
            .get_column("funding_zscore")
            .to_list()
        )
        assert_null_prefix(values, lookback - 1)
        assert None not in values[lookback - 1 :]
        assert all(isinstance(value, float) for value in values[lookback - 1 :])


def test_funding_momentum() -> None:
    """Funding momentum is the absolute change over lookback."""
    rates = [0.01, 0.02, 0.03, 0.04, 0.05]
    result = FundingMomentumFeature(lookback=2).transform(pl.DataFrame({"funding_rate": rates}))
    values = result.get_column("funding_momentum").to_list()
    assert_null_prefix(values, 2)
    assert values[2] == pytest.approx(0.02)
    assert values[3] == pytest.approx(0.02)
    assert values[4] == pytest.approx(0.02)


@pytest.mark.parametrize(
    ("factory", "error_code"),
    [
        (FundingChangeFeature, "FEATURE-FUNDING-CHANGE-001"),
        (lambda: FundingRollingMeanFeature(lookback=2), "FEATURE-FUNDING-ROLLING-MEAN-002"),
        (lambda: FundingZScoreFeature(lookback=2), "FEATURE-FUNDING-ZSCORE-002"),
        (lambda: FundingMomentumFeature(lookback=2), "FEATURE-FUNDING-MOMENTUM-002"),
    ],
)
def test_funding_missing_column(factory: object, error_code: str) -> None:
    """Missing funding_rate raises FeatureExecutionError."""
    feature = factory()  # type: ignore[operator]
    with pytest.raises(
        FeatureExecutionError,
        match="required column missing: funding_rate",
    ) as exc_info:
        feature.transform(pl.DataFrame({"mark_price": [1.0, 2.0]}))
    assert exc_info.value.error_code == error_code


def test_funding_lookback_validation() -> None:
    """Windowed funding features reject zero lookback."""
    assert_lookback_zero_raises(
        lambda lookback: FundingRollingMeanFeature(lookback=lookback),
        error_code="FEATURE-FUNDING-ROLLING-MEAN-001",
    )
    assert_lookback_zero_raises(
        lambda lookback: FundingZScoreFeature(lookback=lookback),
        error_code="FEATURE-FUNDING-ZSCORE-001",
    )
    assert_lookback_zero_raises(
        lambda lookback: FundingMomentumFeature(lookback=lookback),
        error_code="FEATURE-FUNDING-MOMENTUM-001",
    )


def test_funding_immutability_and_exports() -> None:
    """Funding transforms are immutable and package-exported."""
    assert_protocol_and_immutability(
        FundingChangeFeature(),
        output_column="funding_change",
        frame=_rates(),
    )
    import cqros.features as features_package
    import cqros.features.funding as funding_package

    for name in (
        "FundingChangeFeature",
        "FundingRollingMeanFeature",
        "FundingZScoreFeature",
        "FundingMomentumFeature",
    ):
        assert name in funding_package.__all__
        assert name in features_package.__all__
