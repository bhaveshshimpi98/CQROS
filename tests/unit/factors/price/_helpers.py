"""Shared helpers for CQROS price factor unit tests."""

from __future__ import annotations

from collections.abc import Callable

import polars as pl
import pytest

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError
from cqros.factors.interfaces import Factor


def assert_protocol_and_immutability(
    factor: BaseFactor,
    *,
    output_column: str,
    frame: pl.DataFrame | None = None,
) -> None:
    """Assert Factor protocol conformance and compute immutability."""
    assert isinstance(factor, BaseFactor)
    assert isinstance(factor, Factor)
    source = frame if frame is not None else pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0]})
    original_columns = list(source.columns)
    probe_column = original_columns[0]
    original_values = source.get_column(probe_column).to_list()
    result = factor.compute(source)
    assert list(source.columns) == original_columns
    assert source.get_column(probe_column).to_list() == original_values
    assert output_column not in source.columns
    assert output_column in result.columns
    assert result.height == source.height
    assert result is not source


def assert_missing_close_raises(
    factory: Callable[[], BaseFactor],
    *,
    error_code: str,
    factor_name: str,
) -> None:
    """Assert missing close fails fast with FactorError."""
    frame = pl.DataFrame({"open": [1.0, 2.0, 3.0]})
    with pytest.raises(FactorError, match="required column missing: close") as exc_info:
        factory().compute(frame)
    error = exc_info.value
    assert error.error_code == error_code
    assert error.details["factor"] == factor_name
    assert error.details["required_column"] == "close"
    assert error.details["available_columns"] == ("open",)


def assert_lookback_zero_raises(
    factory: Callable[[int], BaseFactor],
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


def assert_lookback_below_two_raises(
    factory: Callable[[int], BaseFactor],
    *,
    error_code: str,
    value: int = 1,
) -> None:
    """Assert lookback below 2 is rejected for Batch-1 factors."""
    with pytest.raises(
        ValidationError,
        match="lookback must be an integer greater than or equal to 2",
    ) as exc_info:
        factory(value)
    assert exc_info.value.error_code == error_code
    assert exc_info.value.details["parameter"] == "lookback"
    assert exc_info.value.details["value"] == value


def assert_lookback_negative_raises(factory: Callable[[int], BaseFactor]) -> None:
    """Assert negative lookback is rejected by BaseFactor."""
    with pytest.raises(
        ValidationError,
        match="lookback must be an integer greater than or equal to 0",
    ) as exc_info:
        factory(-1)
    assert exc_info.value.error_code == "FACTOR-BASE-008"
    assert exc_info.value.details["parameter"] == "lookback"
    assert exc_info.value.details["value"] == -1


def assert_preserves_columns(
    factor: BaseFactor,
    *,
    output_column: str,
    frame: pl.DataFrame | None = None,
) -> None:
    """Assert existing columns are preserved and only declared output is added."""
    source = (
        frame
        if frame is not None
        else pl.DataFrame({"close": [1.0, 2.0, 3.0], "volume": [10, 20, 30]})
    )
    result = factor.compute(source)
    assert result.columns == [*source.columns, output_column]
    assert factor.produced_columns == (output_column,)


def assert_output_float64_nullable(
    factor: BaseFactor,
    *,
    output_column: str,
    frame: pl.DataFrame | None = None,
) -> None:
    """Assert produced column is nullable Float64."""
    if frame is None:
        frame = pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0]})
        required = set(factor.required_features)
        if "high" in required or "low" in required or "open" in required:
            frame_data: dict[str, list[float]] = {
                "close": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
            if "open" in required:
                frame_data["open"] = [0.95, 1.95, 2.95, 3.95, 4.95]
            if "high" in required:
                frame_data["high"] = [1.1, 2.1, 3.1, 4.1, 5.1]
            if "low" in required:
                frame_data["low"] = [0.9, 1.9, 2.9, 3.9, 4.9]
            frame = pl.DataFrame(frame_data)
    result = factor.compute(frame)
    dtype = result.schema[output_column]
    assert dtype == pl.Float64


def assert_empty_and_single_row(
    factory: Callable[[], BaseFactor],
    *,
    output_column: str,
    columns: dict[str, list[float]] | None = None,
) -> None:
    """Assert empty and single-row frames produce null-safe outputs."""
    factor = factory()
    empty_data: dict[str, list[float]] = columns if columns is not None else {"close": []}
    empty = pl.DataFrame(
        {name: pl.Series(name, values, dtype=pl.Float64) for name, values in empty_data.items()}
    )
    empty_result = factor.compute(empty)
    assert empty_result.height == 0
    assert output_column in empty_result.columns

    single_data: dict[str, list[float]] = {name: [10.0] for name in empty_data}
    single = pl.DataFrame(single_data)
    single_result = factor.compute(single)
    assert single_result.height == 1
    assert single_result.get_column(output_column).to_list() == [None]


def assert_determinism(
    factory: Callable[[], BaseFactor],
    *,
    output_column: str,
    frame: pl.DataFrame,
) -> None:
    """Assert identical inputs produce identical outputs."""
    factor = factory()
    first = factor.compute(frame).get_column(output_column).to_list()
    second = factory().compute(frame).get_column(output_column).to_list()
    assert first == second
