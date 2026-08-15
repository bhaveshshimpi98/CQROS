"""Unit tests for CQROS processing ``BaseVerifier`` helpers."""

from __future__ import annotations

import math

import polars as pl
import pytest

from cqros.processing.verification.base import BaseVerifier
from cqros.processing.verification.exceptions import (
    ERROR_REQUIRED_COLUMNS,
    ProcessingValidationError,
)

_COL_TS: str = "open_time"
_COL_A: str = "a"
_COL_B: str = "b"
_COL_PRICE: str = "price"
_COL_VOLUME: str = "volume"


def _verifier() -> BaseVerifier:
    """Build a BaseVerifier instance for helper tests."""
    return BaseVerifier()


def test_validate_required_columns_success() -> None:
    """Required columns present do not raise."""
    frame = pl.DataFrame({_COL_TS: [1], _COL_A: [1.0]})
    _verifier()._validate_required_columns(frame, (_COL_TS, _COL_A))


def test_validate_required_columns_failure() -> None:
    """Missing required columns raise ProcessingValidationError."""
    frame = pl.DataFrame({_COL_TS: [1]})
    with pytest.raises(ProcessingValidationError) as exc_info:
        _verifier()._validate_required_columns(frame, (_COL_TS, _COL_A, _COL_B))
    assert exc_info.value.error_code == ERROR_REQUIRED_COLUMNS
    assert dict(exc_info.value.details)["missing_columns"] == (_COL_A, _COL_B)
    assert dict(exc_info.value.details)["required_columns"] == (_COL_TS, _COL_A, _COL_B)


def test_count_duplicate_timestamp_rows() -> None:
    """Duplicate timestamps use keep-first semantics."""
    frame = pl.DataFrame({_COL_TS: [1, 1, 2, 2, 2, 3]})
    assert _verifier()._count_duplicate_timestamp_rows(frame, _COL_TS) == 3


def test_count_duplicate_timestamp_rows_none() -> None:
    """Unique timestamps yield zero duplicates."""
    frame = pl.DataFrame({_COL_TS: [1, 2, 3]})
    assert _verifier()._count_duplicate_timestamp_rows(frame, _COL_TS) == 0


def test_count_null_rows() -> None:
    """Rows with any null among inspected columns are counted."""
    frame = pl.DataFrame(
        {
            _COL_A: [1.0, None, 3.0, None],
            _COL_B: [1.0, 2.0, None, None],
        }
    )
    assert _verifier()._count_null_rows(frame, (_COL_A, _COL_B)) == 3


def test_count_null_rows_empty_columns() -> None:
    """Empty column list yields zero null rows."""
    frame = pl.DataFrame({_COL_A: [None, 1.0]})
    assert _verifier()._count_null_rows(frame, ()) == 0


def test_count_nan_rows() -> None:
    """Rows with any NaN among floating columns are counted."""
    frame = pl.DataFrame(
        {
            _COL_PRICE: [1.0, math.nan, 3.0, math.nan],
            _COL_VOLUME: [1.0, 2.0, math.nan, math.nan],
            "trade_count": [1, 2, 3, 4],
        }
    )
    assert _verifier()._count_nan_rows(frame, (_COL_PRICE, _COL_VOLUME, "trade_count")) == 3


def test_count_nan_rows_ignores_non_floating_columns() -> None:
    """Integer columns are ignored even when listed as numeric."""
    frame = pl.DataFrame({"trade_count": [1, 2, None]})
    assert _verifier()._count_nan_rows(frame, ("trade_count",)) == 0


def test_count_nan_rows_empty_columns() -> None:
    """Empty numeric column list yields zero NaN rows."""
    frame = pl.DataFrame({_COL_PRICE: [math.nan]})
    assert _verifier()._count_nan_rows(frame, ()) == 0


def test_count_invalid_timestamp_rows_null_and_non_positive() -> None:
    """NULL and <= 0 integer timestamps are invalid."""
    frame = pl.DataFrame({_COL_TS: [100, 0, -5, None, 200]})
    assert _verifier()._count_invalid_timestamp_rows(frame, _COL_TS) == 3


def test_count_invalid_timestamp_rows_non_integer_dtype() -> None:
    """Non-integer timestamp dtype marks every row invalid."""
    frame = pl.DataFrame({_COL_TS: [1.0, 2.0, 3.0]})
    assert _verifier()._count_invalid_timestamp_rows(frame, _COL_TS) == 3


def test_is_sorted_true() -> None:
    """Strictly increasing timestamps are sorted."""
    frame = pl.DataFrame({_COL_TS: [1, 2, 3, 4]})
    assert _verifier()._is_sorted(frame, _COL_TS) is True


def test_is_sorted_false() -> None:
    """Out-of-order timestamps are not sorted."""
    frame = pl.DataFrame({_COL_TS: [1, 3, 2, 4]})
    assert _verifier()._is_sorted(frame, _COL_TS) is False


def test_is_sorted_allows_equal_adjacent_timestamps() -> None:
    """Equal adjacent timestamps remain sorted; duplicates are separate."""
    frame = pl.DataFrame({_COL_TS: [1, 1, 2, 2, 3]})
    verifier = _verifier()
    assert verifier._is_sorted(frame, _COL_TS) is True
    assert verifier._count_duplicate_timestamp_rows(frame, _COL_TS) == 2


def test_is_sorted_single_row() -> None:
    """A single-row frame is considered sorted."""
    frame = pl.DataFrame({_COL_TS: [42]})
    assert _verifier()._is_sorted(frame, _COL_TS) is True


def test_empty_dataframe_helpers() -> None:
    """Empty frames yield zero counts and are considered sorted."""
    frame = pl.DataFrame(
        schema={
            _COL_TS: pl.Int64,
            _COL_PRICE: pl.Float64,
            _COL_A: pl.Float64,
        }
    )
    verifier = _verifier()
    verifier._validate_required_columns(frame, (_COL_TS, _COL_PRICE))
    assert verifier._count_duplicate_timestamp_rows(frame, _COL_TS) == 0
    assert verifier._count_null_rows(frame, (_COL_TS, _COL_PRICE, _COL_A)) == 0
    assert verifier._count_nan_rows(frame, (_COL_PRICE,)) == 0
    assert verifier._count_invalid_timestamp_rows(frame, _COL_TS) == 0
    assert verifier._is_sorted(frame, _COL_TS) is True


def test_helpers_do_not_mutate_frame() -> None:
    """Helper methods leave the input DataFrame unchanged."""
    frame = pl.DataFrame(
        {
            _COL_TS: [1, 1, 0, None, 2],
            _COL_PRICE: [1.0, math.nan, 3.0, 4.0, None],
        }
    )
    original = frame.clone()
    verifier = _verifier()
    verifier._validate_required_columns(frame, (_COL_TS, _COL_PRICE))
    verifier._count_duplicate_timestamp_rows(frame, _COL_TS)
    verifier._count_null_rows(frame, (_COL_TS, _COL_PRICE))
    verifier._count_nan_rows(frame, (_COL_PRICE,))
    verifier._count_invalid_timestamp_rows(frame, _COL_TS)
    verifier._is_sorted(frame, _COL_TS)
    assert frame.equals(original)
