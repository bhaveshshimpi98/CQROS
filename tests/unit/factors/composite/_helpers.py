"""Shared helpers for CQROS composite factor unit tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import polars as pl
import pytest

from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError
from cqros.factors.interfaces import Factor


def feature_frame(values: Mapping[str, Sequence[float | None]]) -> pl.DataFrame:
    """Build a deterministic Float64 feature-output DataFrame for composite tests."""
    series: dict[str, pl.Series] = {}
    for name, column_values in values.items():
        series[name] = pl.Series(name, list(column_values), dtype=pl.Float64)
    return pl.DataFrame(series)


def assert_protocol_and_immutability(
    factor: BaseFactor,
    *,
    output_column: str,
    frame: pl.DataFrame,
) -> None:
    """Assert Factor protocol conformance and compute immutability."""
    assert isinstance(factor, BaseFactor)
    assert isinstance(factor, Factor)
    original_columns = list(frame.columns)
    original_rows = frame.to_dicts()
    result = factor.compute(frame)
    assert list(frame.columns) == original_columns
    assert frame.to_dicts() == original_rows
    assert output_column not in frame.columns
    assert output_column in result.columns
    assert result.height == frame.height
    assert result is not frame


def assert_missing_feature_raises(
    factory: Callable[[], BaseFactor],
    *,
    error_code: str,
    factor_name: str,
    missing_feature: str,
    present_features: Mapping[str, Sequence[float]],
) -> None:
    """Assert a missing required feature fails fast with FactorError."""
    frame = pl.DataFrame(dict(present_features))
    with pytest.raises(
        FactorError,
        match=f"required feature missing: {missing_feature}",
    ) as exc_info:
        factory().compute(frame)
    error = exc_info.value
    assert error.error_code == error_code
    assert error.details["factor"] == factor_name
    assert error.details["required_feature"] == missing_feature
    assert error.details["available_columns"] == tuple(frame.columns)


def assert_preserves_columns(
    factor: BaseFactor,
    *,
    output_column: str,
    frame: pl.DataFrame,
) -> None:
    """Assert existing columns are preserved and only declared output is added."""
    result = factor.compute(frame)
    assert result.columns == [*frame.columns, output_column]
    assert factor.produced_columns == (output_column,)


def assert_null_propagation(
    factor: BaseFactor,
    *,
    output_column: str,
    frame: pl.DataFrame,
) -> None:
    """Assert null inputs propagate into the composite output."""
    result = factor.compute(frame)
    values = result.get_column(output_column).to_list()
    assert any(value is None for value in values)
