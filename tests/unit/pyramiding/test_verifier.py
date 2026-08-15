"""Unit tests for CQROS ``PyramidingVerifier``."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from cqros.pyramiding import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    ERROR_REQUIRED_COLUMNS,
    ERROR_SCHEMA_MISMATCH,
    MERGED_PYRAMIDING_SCHEMA,
    PyramidingReason,
    PyramidingValidationError,
    PyramidingVerifier,
)

_TIMEFRAME = "1h"
_MANAGER = "simple"


def _open_time(index: int) -> datetime:
    """Build a deterministic UTC open_time for row ``index``."""
    return datetime(2024, 1, 1, hour=index, tzinfo=UTC)


def _pyramiding_frame(
    *,
    symbols: list[str] | None = None,
    position_ids: list[str] | None = None,
    trade_ids: list[str] | None = None,
    open_times: list[datetime] | None = None,
    reasons: list[str] | None = None,
    allow_pyramid: list[bool] | None = None,
    managers: list[str] | None = None,
    entry_prices: list[float] | None = None,
    current_prices: list[float] | None = None,
    highest_prices: list[float] | None = None,
    position_sizes: list[float] | None = None,
    additional_sizes: list[float] | None = None,
    recommended_sizes: list[float] | None = None,
    profit_pcts: list[float] | None = None,
    add_numbers: list[int] | None = None,
    max_adds_list: list[int] | None = None,
) -> pl.DataFrame:
    """Build a canonical pyramiding frame for verifier tests."""
    symbols = symbols if symbols is not None else ["BTCUSDT", "ETHUSDT"]
    row_count = len(symbols)
    position_ids = (
        position_ids if position_ids is not None else [f"pos-{i + 1:08d}" for i in range(row_count)]
    )
    trade_ids = trade_ids if trade_ids is not None else list(position_ids)
    open_times = open_times if open_times is not None else [_open_time(i) for i in range(row_count)]
    reasons = (
        reasons if reasons is not None else [PyramidingReason.INSUFFICIENT_PROFIT.value] * row_count
    )
    allow_pyramid = allow_pyramid if allow_pyramid is not None else [False] * row_count
    managers = managers if managers is not None else [_MANAGER] * row_count
    entry_prices = entry_prices if entry_prices is not None else [100.0] * row_count
    current_prices = current_prices if current_prices is not None else [102.0] * row_count
    highest_prices = highest_prices if highest_prices is not None else [102.0] * row_count
    position_sizes = position_sizes if position_sizes is not None else [1.0] * row_count
    additional_sizes = additional_sizes if additional_sizes is not None else [0.0] * row_count
    recommended_sizes = recommended_sizes if recommended_sizes is not None else [1.0] * row_count
    profit_pcts = profit_pcts if profit_pcts is not None else [0.02] * row_count
    add_numbers = add_numbers if add_numbers is not None else [0] * row_count
    max_adds_list = max_adds_list if max_adds_list is not None else [3] * row_count
    return pl.DataFrame(
        {
            "manager": managers,
            "symbol": symbols,
            "timeframe": [_TIMEFRAME] * row_count,
            "open_time": open_times,
            "position_id": position_ids,
            "trade_id": trade_ids,
            "entry_price": entry_prices,
            "current_price": current_prices,
            "highest_price": highest_prices,
            "position_size": position_sizes,
            "add_number": add_numbers,
            "max_adds": max_adds_list,
            "additional_size": additional_sizes,
            "recommended_size": recommended_sizes,
            "profit_pct": profit_pcts,
            "allow_pyramid": allow_pyramid,
            "reason": reasons,
        },
        schema=dict(COLUMN_DTYPES),
    ).select(list(CANONICAL_COLUMN_ORDER))


def test_verifier_passes_canonical_frame() -> None:
    """A clean canonical pyramiding frame passes verification."""
    report = PyramidingVerifier().verify(_pyramiding_frame())
    assert report.passed is True
    assert report.rows_checked == 2
    assert report.duplicate_timestamp_rows == 0
    assert report.null_rows == 0
    assert report.warnings == ()


def test_verifier_requires_merged_schema_identity() -> None:
    """Frames cast to MERGED_PYRAMIDING_SCHEMA remain verifiable."""
    frame = _pyramiding_frame().cast(MERGED_PYRAMIDING_SCHEMA)
    report = PyramidingVerifier().verify(frame)
    assert report.passed is True


def test_verifier_passes_empty_dataset() -> None:
    """An empty frame with the correct schema passes verification."""
    empty = pl.DataFrame(schema=dict(COLUMN_DTYPES)).select(list(CANONICAL_COLUMN_ORDER))
    report = PyramidingVerifier().verify(empty)
    assert report.passed is True
    assert report.rows_checked == 0
    assert report.warnings == ()


def test_verifier_rejects_missing_columns() -> None:
    """Missing required columns raise PYR-VERIFICATION-001."""
    frame = _pyramiding_frame().drop("allow_pyramid")
    with pytest.raises(PyramidingValidationError) as exc_info:
        PyramidingVerifier().verify(frame)
    assert exc_info.value.error_code == ERROR_REQUIRED_COLUMNS
    assert "allow_pyramid" in exc_info.value.details["missing_columns"]


def test_verifier_rejects_missing_reason_column() -> None:
    """Missing 'reason' column raises PYR-VERIFICATION-001."""
    frame = _pyramiding_frame().drop("reason")
    with pytest.raises(PyramidingValidationError) as exc_info:
        PyramidingVerifier().verify(frame)
    assert exc_info.value.error_code == ERROR_REQUIRED_COLUMNS


def test_verifier_rejects_dtype_mismatch() -> None:
    """Column dtype mismatches raise PYR-VERIFICATION-002."""
    bad_dtype = _pyramiding_frame().with_columns(pl.col("entry_price").cast(pl.Int64))
    with pytest.raises(PyramidingValidationError) as exc_info:
        PyramidingVerifier().verify(bad_dtype)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH


def test_verifier_reports_duplicate_primary_keys() -> None:
    """Duplicate primary keys fail verification with a warning."""
    duplicates = _pyramiding_frame(
        symbols=["BTCUSDT", "BTCUSDT"],
        position_ids=["pos-00000001", "pos-00000001"],
        open_times=[_open_time(0), _open_time(0)],
    )
    report = PyramidingVerifier().verify(duplicates)
    assert report.passed is False
    assert report.duplicate_timestamp_rows == 1
    assert "Duplicate pyramiding primary keys detected." in report.warnings


def test_verifier_reports_null_rows() -> None:
    """Null values in required columns fail verification with a warning."""
    nulls = _pyramiding_frame().with_columns(
        pl.when(pl.arange(0, pl.len()) == 1).then(None).otherwise(pl.col("symbol")).alias("symbol")
    )
    report = PyramidingVerifier().verify(nulls)
    assert report.passed is False
    assert report.null_rows > 0
    assert "Rows containing NULL values." in report.warnings


def test_verifier_reports_invalid_enum_reason() -> None:
    """Invalid reason values fail verification with a warning."""
    invalid_reason = _pyramiding_frame(
        symbols=["BTCUSDT"],
        position_ids=["pos-00000001"],
        open_times=[_open_time(0)],
        reasons=["INVALID_REASON"],
    )
    report = PyramidingVerifier().verify(invalid_reason)
    assert report.passed is False
    assert "Invalid PyramidingReason values detected." in report.warnings


def test_verifier_reports_null_allow_pyramid() -> None:
    """Null allow_pyramid values fail verification with a warning."""
    null_bool = _pyramiding_frame().with_columns(
        pl.when(pl.arange(0, pl.len()) == 0)
        .then(None)
        .otherwise(pl.col("allow_pyramid"))
        .alias("allow_pyramid")
    )
    report = PyramidingVerifier().verify(null_bool)
    assert report.passed is False
    assert "Invalid allow_pyramid boolean values detected." in report.warnings


def test_verifier_reports_negative_sizes() -> None:
    """Negative size or add-counter values fail verification with a warning."""
    negative = _pyramiding_frame(
        symbols=["BTCUSDT"],
        position_ids=["pos-00000001"],
        open_times=[_open_time(0)],
        position_sizes=[-1.0],
    )
    report = PyramidingVerifier().verify(negative)
    assert report.passed is False
    assert "Negative size values detected." in report.warnings


def test_verifier_reports_negative_add_number() -> None:
    """Negative add_number fails verification with a negative-sizes warning."""
    frame = _pyramiding_frame(
        symbols=["BTCUSDT"],
        position_ids=["pos-00000001"],
        open_times=[_open_time(0)],
        add_numbers=[-1],
    )
    report = PyramidingVerifier().verify(frame)
    assert report.passed is False
    assert "Negative size values detected." in report.warnings


def test_verifier_reports_non_finite_values() -> None:
    """Non-finite numerics (inf) fail verification with a warning."""
    non_finite = _pyramiding_frame().with_columns(
        pl.when(pl.arange(0, pl.len()) == 0)
        .then(pl.lit(float("inf")))
        .otherwise(pl.col("entry_price"))
        .alias("entry_price")
    )
    report = PyramidingVerifier().verify(non_finite)
    assert report.passed is False
    assert report.invalid_numeric_rows > 0
    assert "Non-finite numeric values detected." in report.warnings


def test_verifier_reports_unsorted_open_time() -> None:
    """A frame not sorted by open_time fails verification with a warning."""
    unsorted = _pyramiding_frame(open_times=[_open_time(5), _open_time(0)])
    report = PyramidingVerifier().verify(unsorted)
    assert report.passed is False
    assert "Frame is not sorted by open_time." in report.warnings


def test_verifier_reports_non_canonical_column_order() -> None:
    """A non-canonical column order fails verification with a warning."""
    reordered = _pyramiding_frame().select(list(reversed(CANONICAL_COLUMN_ORDER)))
    report = PyramidingVerifier().verify(reordered)
    assert report.passed is False
    assert "Frame column order does not match canonical order." in report.warnings


def test_verifier_reports_incomplete_lineage() -> None:
    """Blank manager values fail verification with a lineage warning."""
    blank_manager = _pyramiding_frame(managers=[_MANAGER, ""])
    report = PyramidingVerifier().verify(blank_manager)
    assert report.passed is False
    assert "Incomplete lineage metadata detected." in report.warnings


def test_verifier_all_reason_values_are_valid() -> None:
    """All PyramidingReason enum values pass the enum check in the verifier."""
    for reason in PyramidingReason:
        frame = _pyramiding_frame(
            symbols=["BTCUSDT"],
            position_ids=["pos-00000001"],
            open_times=[_open_time(0)],
            reasons=[reason.value],
        )
        report = PyramidingVerifier().verify(frame)
        assert (
            "Invalid PyramidingReason values detected." not in report.warnings
        ), f"reason {reason.value} should be valid"
