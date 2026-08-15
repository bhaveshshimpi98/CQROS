"""Unit tests for CQROS ``ATRFeature`` and ``DollarVolumeFeature``."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.features.exceptions import FeatureExecutionError
from cqros.features.price import ATRFeature, DollarVolumeFeature
from tests.unit.features._helpers import (
    assert_lookback_zero_raises,
    assert_null_prefix,
    assert_protocol_and_immutability,
)


def test_atr_feature_metadata() -> None:
    """ATRFeature exposes the fixed production metadata contract."""
    feature = ATRFeature()
    assert feature.name == "atr"
    assert feature.version == "1.0.0"
    assert feature.category == "price"
    assert feature.required_columns == ("high", "low", "close")
    assert feature.produced_columns == ("atr",)
    assert feature.lookback == 14
    assert feature.dependencies == ()


def test_atr_calculation_correctness() -> None:
    """ATR is the rolling mean of true range."""
    frame = pl.DataFrame(
        {
            "high": [12.0, 15.0, 14.0, 16.0],
            "low": [10.0, 11.0, 12.0, 13.0],
            "close": [11.0, 14.0, 13.0, 15.0],
        }
    )
    result = ATRFeature(lookback=2).transform(frame)
    values = result.get_column("atr").to_list()
    # TR0 = high-low = 2 (prev close null; max_horizontal ignores nulls)
    # TR1 = max(4, |15-11|, |11-11|) = max(4, 4, 0) = 4
    # TR2 = max(2, |14-14|, |12-14|) = max(2, 0, 2) = 2
    # TR3 = max(3, |16-13|, |13-13|) = max(3, 3, 0) = 3
    assert_null_prefix(values, 1)
    assert values[1] == pytest.approx((2.0 + 4.0) / 2.0)
    assert values[2] == pytest.approx((4.0 + 2.0) / 2.0)
    assert values[3] == pytest.approx((2.0 + 3.0) / 2.0)


def test_atr_missing_column_raises() -> None:
    """Missing OHLC columns raise FeatureExecutionError."""
    frame = pl.DataFrame({"high": [1.0], "low": [0.5]})
    with pytest.raises(FeatureExecutionError, match="required column missing: close") as exc_info:
        ATRFeature(lookback=1).transform(frame)
    assert exc_info.value.error_code == "FEATURE-ATR-002"


def test_atr_lookback_zero_raises() -> None:
    """Zero lookback is rejected."""
    assert_lookback_zero_raises(
        lambda lookback: ATRFeature(lookback=lookback), error_code="FEATURE-ATR-001"
    )


def test_atr_immutability() -> None:
    """ATR transform does not mutate the input frame."""
    assert_protocol_and_immutability(
        ATRFeature(lookback=2),
        output_column="atr",
        frame=pl.DataFrame(
            {
                "high": [2.0, 3.0, 4.0],
                "low": [1.0, 1.5, 2.0],
                "close": [1.5, 2.5, 3.5],
            }
        ),
    )


def test_dollar_volume_feature_metadata() -> None:
    """DollarVolumeFeature exposes the fixed production metadata contract."""
    feature = DollarVolumeFeature()
    assert feature.name == "dollar_volume"
    assert feature.version == "1.0.0"
    assert feature.category == "price"
    assert feature.required_columns == ("close", "volume")
    assert feature.produced_columns == ("dollar_volume",)
    assert feature.lookback == 0


def test_dollar_volume_calculation() -> None:
    """Dollar volume equals close * volume."""
    frame = pl.DataFrame({"close": [100.0, 110.0], "volume": [2.0, 3.0]})
    result = DollarVolumeFeature().transform(frame)
    assert result.get_column("dollar_volume").to_list() == pytest.approx([200.0, 330.0])


def test_dollar_volume_missing_column_raises() -> None:
    """Missing volume raises FeatureExecutionError."""
    frame = pl.DataFrame({"close": [1.0, 2.0]})
    with pytest.raises(FeatureExecutionError, match="required column missing: volume") as exc_info:
        DollarVolumeFeature().transform(frame)
    assert exc_info.value.error_code == "FEATURE-DOLLAR-VOLUME-001"


def test_dollar_volume_immutability() -> None:
    """Dollar volume transform does not mutate the input frame."""
    assert_protocol_and_immutability(
        DollarVolumeFeature(),
        output_column="dollar_volume",
        frame=pl.DataFrame({"close": [1.0, 2.0], "volume": [10.0, 20.0]}),
    )
