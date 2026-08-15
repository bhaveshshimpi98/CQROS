"""Unit tests for CQROS ``LogReturnsFeature``."""

from __future__ import annotations

import math

import polars as pl
import pytest

from cqros.features.base import BaseFeature
from cqros.features.exceptions import FeatureExecutionError
from cqros.features.interfaces import Feature
from cqros.features.price import LogReturnsFeature
from cqros.features.price.log_returns import LogReturnsFeature as LogReturnsFeatureDirect
from tests.unit.features._helpers import assert_protocol_and_immutability


def _feature() -> LogReturnsFeature:
    """Build the default log returns feature instance."""
    return LogReturnsFeature()


def test_log_returns_feature_metadata() -> None:
    """LogReturnsFeature exposes the fixed production metadata contract."""
    feature = _feature()
    assert isinstance(feature, BaseFeature)
    assert isinstance(feature, Feature)
    assert feature.name == "log_returns"
    assert feature.version == "1.0.0"
    assert feature.category == "price"
    assert feature.required_columns == ("close",)
    assert feature.produced_columns == ("log_returns",)
    assert feature.lookback == 1
    assert feature.dependencies == ()


def test_log_returns_calculation_correctness() -> None:
    """Log returns match ln(close / close.shift(1))."""
    frame = pl.DataFrame({"close": [100.0, 110.0, 99.0]})
    result = _feature().transform(frame)
    values = result.get_column("log_returns").to_list()
    assert values[0] is None
    assert values[1] == pytest.approx(math.log(1.1))
    assert values[2] == pytest.approx(math.log(99.0 / 110.0))


def test_missing_close_column_raises() -> None:
    """Missing close column raises FeatureExecutionError."""
    frame = pl.DataFrame({"open": [1.0, 2.0]})
    with pytest.raises(FeatureExecutionError, match="required column missing: close") as exc_info:
        _feature().transform(frame)
    assert exc_info.value.error_code == "FEATURE-LOG-RETURNS-001"


def test_empty_dataframe() -> None:
    """An empty frame with close yields an empty log_returns column."""
    frame = pl.DataFrame({"close": pl.Series("close", [], dtype=pl.Float64)})
    result = _feature().transform(frame)
    assert result.height == 0
    assert "log_returns" in result.columns


def test_input_immutability() -> None:
    """transform does not mutate the caller-supplied DataFrame."""
    assert_protocol_and_immutability(
        _feature(),
        output_column="log_returns",
        frame=pl.DataFrame({"close": [1.0, 2.0, 3.0]}),
    )


def test_package_exports_log_returns_feature() -> None:
    """LogReturnsFeature is exported from the price and features packages."""
    assert LogReturnsFeature is LogReturnsFeatureDirect
    import cqros.features as features_package
    import cqros.features.price as price_package

    assert "LogReturnsFeature" in price_package.__all__
    assert "LogReturnsFeature" in features_package.__all__
