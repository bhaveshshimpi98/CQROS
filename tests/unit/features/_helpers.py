"""Shared helpers for CQROS feature unit tests."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import polars as pl
import pytest

from cqros.core.exceptions import ValidationError
from cqros.features.base import BaseFeature
from cqros.features.exceptions import FeatureExecutionError
from cqros.features.interfaces import Feature


def assert_protocol_and_immutability(
    feature: BaseFeature,
    *,
    output_column: str,
    frame: pl.DataFrame,
) -> None:
    """Assert Feature protocol conformance and transform immutability."""
    assert isinstance(feature, BaseFeature)
    assert isinstance(feature, Feature)
    original_columns = list(frame.columns)
    original_data = {name: frame.get_column(name).to_list() for name in frame.columns}
    result = feature.transform(frame)
    assert list(frame.columns) == original_columns
    for name, values in original_data.items():
        assert frame.get_column(name).to_list() == values
    assert output_column not in original_columns or output_column in frame.columns
    assert output_column in result.columns
    assert result.height == frame.height
    assert result is not frame


def assert_missing_column_raises(
    factory: Callable[[], BaseFeature],
    *,
    frame: pl.DataFrame,
    missing_column: str,
    error_code: str,
    feature_name: str,
) -> None:
    """Assert a missing required column fails fast with FeatureExecutionError."""
    with pytest.raises(
        FeatureExecutionError,
        match=f"required column missing: {missing_column}",
    ) as exc_info:
        factory().transform(frame)
    error = exc_info.value
    assert error.error_code == error_code
    assert error.details["feature"] == feature_name
    assert error.details["required_column"] == missing_column
    assert error.details["available_columns"] == tuple(frame.columns)


def assert_lookback_zero_raises(
    factory: Callable[[int], BaseFeature],
    *,
    error_code: str,
) -> None:
    """Assert zero lookback is rejected."""
    with pytest.raises(ValidationError, match="lookback must be an integer greater than 0") as (
        exc_info
    ):
        factory(0)
    assert exc_info.value.error_code == error_code
    assert exc_info.value.details["parameter"] == "lookback"
    assert exc_info.value.details["value"] == 0


def assert_preserves_columns(
    feature: BaseFeature,
    *,
    frame: pl.DataFrame,
    output_column: str,
) -> None:
    """Assert existing columns are preserved and declared output is present."""
    result = feature.transform(frame)
    for column in frame.columns:
        if column != output_column:
            assert column in result.columns
            assert result.get_column(column).to_list() == frame.get_column(column).to_list()
    assert output_column in result.columns
    assert feature.produced_columns == (output_column,)


def assert_null_prefix(values: Sequence[object], count: int) -> None:
    """Assert the first ``count`` values are null."""
    assert all(value is None for value in values[:count])
