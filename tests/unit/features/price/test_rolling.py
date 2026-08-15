"""Unit tests for CQROS rolling price features."""

from __future__ import annotations

import statistics

import polars as pl
import pytest

from cqros.core.exceptions import ValidationError
from cqros.features.price import (
    RollingMaxFeature,
    RollingMeanFeature,
    RollingMinFeature,
    RollingStdFeature,
)
from tests.unit.features._helpers import (
    assert_lookback_zero_raises,
    assert_null_prefix,
    assert_protocol_and_immutability,
)


def test_rolling_mean_calculation() -> None:
    """Rolling mean matches the arithmetic mean over the window."""
    closes = [10.0, 20.0, 30.0, 40.0, 50.0]
    result = RollingMeanFeature(lookback=3).transform(pl.DataFrame({"close": closes}))
    values = result.get_column("rolling_mean").to_list()
    assert_null_prefix(values, 2)
    assert values[2] == pytest.approx(statistics.fmean(closes[0:3]))
    assert values[3] == pytest.approx(statistics.fmean(closes[1:4]))
    assert values[4] == pytest.approx(statistics.fmean(closes[2:5]))


def test_rolling_std_calculation() -> None:
    """Rolling std matches sample standard deviation over the window."""
    closes = [10.0, 20.0, 30.0, 40.0, 50.0]
    result = RollingStdFeature(lookback=3).transform(pl.DataFrame({"close": closes}))
    values = result.get_column("rolling_std").to_list()
    assert_null_prefix(values, 2)
    assert values[2] == pytest.approx(statistics.stdev(closes[0:3]))
    assert values[4] == pytest.approx(statistics.stdev(closes[2:5]))


def test_rolling_max_calculation() -> None:
    """Rolling max returns the window maximum."""
    closes = [10.0, 40.0, 20.0, 50.0, 15.0]
    result = RollingMaxFeature(lookback=3).transform(pl.DataFrame({"close": closes}))
    values = result.get_column("rolling_max").to_list()
    assert_null_prefix(values, 2)
    assert values[2] == pytest.approx(40.0)
    assert values[3] == pytest.approx(50.0)
    assert values[4] == pytest.approx(50.0)


def test_rolling_min_calculation() -> None:
    """Rolling min returns the window minimum."""
    closes = [10.0, 40.0, 20.0, 50.0, 15.0]
    result = RollingMinFeature(lookback=3).transform(pl.DataFrame({"close": closes}))
    values = result.get_column("rolling_min").to_list()
    assert_null_prefix(values, 2)
    assert values[2] == pytest.approx(10.0)
    assert values[3] == pytest.approx(20.0)
    assert values[4] == pytest.approx(15.0)


@pytest.mark.parametrize(
    ("factory", "name", "output", "error_code"),
    [
        (RollingMeanFeature, "rolling_mean", "rolling_mean", "FEATURE-ROLLING-MEAN-001"),
        (RollingStdFeature, "rolling_std", "rolling_std", "FEATURE-ROLLING-STD-001"),
        (RollingMaxFeature, "rolling_max", "rolling_max", "FEATURE-ROLLING-MAX-001"),
        (RollingMinFeature, "rolling_min", "rolling_min", "FEATURE-ROLLING-MIN-001"),
    ],
)
def test_rolling_feature_metadata_and_lookback(
    factory: type,
    name: str,
    output: str,
    error_code: str,
) -> None:
    """Rolling features expose metadata and reject zero lookback."""
    feature = factory()
    assert feature.name == name
    assert feature.category == "price"
    assert feature.version == "1.0.0"
    assert feature.required_columns == ("close",)
    assert feature.produced_columns == (output,)
    assert feature.lookback == 20
    assert_lookback_zero_raises(lambda lookback: factory(lookback=lookback), error_code=error_code)
    with pytest.raises(ValidationError, match="lookback must be an integer greater than or equal"):
        factory(lookback=-1)


@pytest.mark.parametrize(
    ("factory", "output", "error_code"),
    [
        (RollingMeanFeature, "rolling_mean", "FEATURE-ROLLING-MEAN-002"),
        (RollingStdFeature, "rolling_std", "FEATURE-ROLLING-STD-002"),
        (RollingMaxFeature, "rolling_max", "FEATURE-ROLLING-MAX-002"),
        (RollingMinFeature, "rolling_min", "FEATURE-ROLLING-MIN-002"),
    ],
)
def test_rolling_missing_close_and_immutability(
    factory: type,
    output: str,
    error_code: str,
) -> None:
    """Rolling features fail on missing close and preserve input frames."""
    from cqros.features.exceptions import FeatureExecutionError

    feature = factory(lookback=2)
    with pytest.raises(FeatureExecutionError, match="required column missing: close") as exc_info:
        feature.transform(pl.DataFrame({"open": [1.0, 2.0, 3.0]}))
    assert exc_info.value.error_code == error_code
    assert_protocol_and_immutability(
        feature,
        output_column=output,
        frame=pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0]}),
    )
