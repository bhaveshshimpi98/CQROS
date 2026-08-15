"""Unit tests for CQROS ``ReturnsFeature``."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.features.base import BaseFeature
from cqros.features.exceptions import FeatureExecutionError
from cqros.features.interfaces import Feature
from cqros.features.price import ReturnsFeature
from cqros.features.price.returns import ReturnsFeature as ReturnsFeatureDirect


def _feature() -> ReturnsFeature:
    """Build the default returns feature instance."""
    return ReturnsFeature()


def test_returns_feature_metadata() -> None:
    """ReturnsFeature exposes the fixed production metadata contract."""
    feature = _feature()
    assert isinstance(feature, BaseFeature)
    assert isinstance(feature, Feature)
    assert feature.name == "returns"
    assert feature.version == "1.0.0"
    assert feature.category == "price"
    assert feature.description == "Simple percentage returns computed from the close column."
    assert feature.required_columns == ("close",)
    assert feature.produced_columns == ("returns",)
    assert feature.dependencies == ()
    assert feature.lookback == 1


def test_returns_calculation_correctness() -> None:
    """Returns match (close / close.shift(1)) - 1 for subsequent rows."""
    frame = pl.DataFrame({"close": [100.0, 110.0, 99.0, 99.0]})
    result = _feature().transform(frame)
    values = result.get_column("returns").to_list()
    assert values[0] is None
    assert values[1] == pytest.approx(0.10)
    assert values[2] == pytest.approx((99.0 / 110.0) - 1.0)
    assert values[3] == pytest.approx(0.0)


def test_first_row_is_null() -> None:
    """The first return is null because the prior close is undefined."""
    frame = pl.DataFrame({"close": [10.0, 20.0]})
    result = _feature().transform(frame)
    assert result.get_column("returns")[0] is None
    assert result.get_column("returns")[1] == pytest.approx(1.0)


def test_missing_close_column_raises() -> None:
    """Missing close column raises FeatureExecutionError."""
    frame = pl.DataFrame({"open": [1.0, 2.0]})
    with pytest.raises(FeatureExecutionError, match="required column missing: close") as exc_info:
        _feature().transform(frame)
    error = exc_info.value
    assert error.error_code == "FEATURE-RETURNS-001"
    assert error.details["feature"] == "returns"
    assert error.details["required_column"] == "close"
    assert error.details["available_columns"] == ("open",)


def test_empty_dataframe() -> None:
    """An empty frame with close yields an empty returns column."""
    frame = pl.DataFrame({"close": pl.Series("close", [], dtype=pl.Float64)})
    result = _feature().transform(frame)
    assert result.height == 0
    assert "returns" in result.columns
    assert result.get_column("returns").dtype == pl.Float64


def test_single_row_dataframe() -> None:
    """A single-row frame produces a single null return."""
    frame = pl.DataFrame({"close": [42.0]})
    result = _feature().transform(frame)
    assert result.height == 1
    assert result.get_column("returns")[0] is None


def test_input_immutability() -> None:
    """transform does not mutate the caller-supplied DataFrame."""
    frame = pl.DataFrame({"close": [1.0, 2.0, 3.0]})
    original_columns = list(frame.columns)
    original_values = frame.get_column("close").to_list()
    result = _feature().transform(frame)
    assert list(frame.columns) == original_columns
    assert frame.get_column("close").to_list() == original_values
    assert "returns" not in frame.columns
    assert "returns" in result.columns
    assert result is not frame


def test_preserves_existing_columns() -> None:
    """Existing non-close columns are preserved alongside returns."""
    frame = pl.DataFrame({"close": [1.0, 2.0], "volume": [10, 20]})
    result = _feature().transform(frame)
    assert result.columns == ["close", "volume", "returns"]


def test_package_exports_returns_feature() -> None:
    """ReturnsFeature is exported from the price package."""
    assert ReturnsFeature is ReturnsFeatureDirect
    import cqros.features as features_package
    import cqros.features.price as price_package

    assert "ReturnsFeature" in price_package.__all__
    assert "ReturnsFeature" in features_package.__all__
    assert features_package.ReturnsFeature is ReturnsFeature
