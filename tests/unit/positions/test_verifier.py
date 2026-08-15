"""Unit tests for CQROS ``PositionVerifier``."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from cqros.positions import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    ERROR_REQUIRED_COLUMNS,
    ERROR_SCHEMA_MISMATCH,
    MERGED_POSITION_SCHEMA,
    PositionStatus,
    PositionValidationError,
    PositionVerifier,
)

_TIMEFRAME = "1h"
_MANAGER = "simple"


def _opened_at(index: int) -> datetime:
    """Build a deterministic UTC opened_at for row ``index``."""
    return datetime(2024, 1, 1, hour=index, tzinfo=UTC)


def _position_frame(
    *,
    symbols: list[str] | None = None,
    position_ids: list[str] | None = None,
    statuses: list[str] | None = None,
    quantities: list[float] | None = None,
    realized: list[float] | None = None,
    unrealized: list[float] | None = None,
    opened_ats: list[datetime] | None = None,
    closed_ats: list[datetime | None] | None = None,
) -> pl.DataFrame:
    """Build a canonical position frame for verifier tests."""
    symbols = symbols if symbols is not None else ["BTCUSDT", "ETHUSDT"]
    row_count = len(symbols)
    times = (
        opened_ats if opened_ats is not None else [_opened_at(index) for index in range(row_count)]
    )
    statuses = statuses if statuses is not None else [PositionStatus.OPEN.value] * row_count
    position_ids = (
        position_ids
        if position_ids is not None
        else [f"pos-{index + 1:08d}" for index in range(row_count)]
    )
    quantities = quantities if quantities is not None else [1.0] * row_count
    realized = realized if realized is not None else [0.0] * row_count
    unrealized = unrealized if unrealized is not None else [0.0] * row_count
    closed_ats = closed_ats if closed_ats is not None else [None] * row_count
    return pl.DataFrame(
        {
            "symbol": symbols,
            "timeframe": [_TIMEFRAME] * row_count,
            "position_id": position_ids,
            "side": ["LONG"] * row_count,
            "status": statuses,
            "quantity": quantities,
            "average_entry_price": [100.0] * row_count,
            "market_price": [100.0] * row_count,
            "realized_pnl": realized,
            "unrealized_pnl": unrealized,
            "fees_paid": [0.0] * row_count,
            "opened_at": times,
            "updated_at": times,
            "closed_at": closed_ats,
            "model_name": ["alpha-lgbm"] * row_count,
            "model_version": ["1.0.0"] * row_count,
            "optimizer": ["equal_weight"] * row_count,
            "policy": ["fixed_risk"] * row_count,
            "manager": [_MANAGER] * row_count,
        },
        schema=dict(COLUMN_DTYPES),
    ).select(list(CANONICAL_COLUMN_ORDER))


def test_verifier_passes_canonical_frame() -> None:
    """A clean canonical position frame passes verification."""
    report = PositionVerifier().verify(_position_frame())
    assert report.passed is True
    assert report.rows_checked == 2
    assert report.duplicate_timestamp_rows == 0
    assert report.null_rows == 0
    assert report.warnings == ()


def test_verifier_rejects_missing_columns_and_dtype_mismatch() -> None:
    """Missing columns and dtype mismatches raise verification errors."""
    frame = _position_frame().drop("status")
    with pytest.raises(PositionValidationError) as exc_info:
        PositionVerifier().verify(frame)
    assert exc_info.value.error_code == ERROR_REQUIRED_COLUMNS

    bad_dtype = _position_frame().with_columns(pl.col("fees_paid").cast(pl.Int64))
    with pytest.raises(PositionValidationError) as exc_info:
        PositionVerifier().verify(bad_dtype)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH


def test_verifier_reports_duplicates_and_invalid_values() -> None:
    """Duplicate ids, negative quantity, and non-finite PnL fail verification."""
    duplicates = _position_frame(
        symbols=["BTCUSDT", "BTCUSDT"],
        position_ids=["pos-00000001", "pos-00000001"],
        opened_ats=[_opened_at(0), _opened_at(1)],
    )
    report = PositionVerifier().verify(duplicates)
    assert report.passed is False
    assert report.duplicate_timestamp_rows == 1
    assert "Duplicate position ids detected." in report.warnings

    negative = _position_frame(quantities=[1.0, -0.5])
    report = PositionVerifier().verify(negative)
    assert report.passed is False
    assert report.invalid_numeric_rows == 1
    assert "Negative quantity values detected." in report.warnings

    non_finite = _position_frame(realized=[0.0, float("nan")])
    report = PositionVerifier().verify(non_finite)
    assert report.passed is False
    assert "Non-finite PnL values detected." in report.warnings or report.nan_rows > 0


def test_verifier_requires_merged_schema_identity() -> None:
    """Canonical frames cast to MERGED_POSITION_SCHEMA remain verifiable."""
    frame = _position_frame().cast(MERGED_POSITION_SCHEMA)
    report = PositionVerifier().verify(frame)
    assert report.passed is True
