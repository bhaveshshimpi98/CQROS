"""Unit tests for CQROS ``AccountingVerifier``."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from cqros.accounting import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    ERROR_REQUIRED_COLUMNS,
    ERROR_SCHEMA_MISMATCH,
    MERGED_ACCOUNTING_SCHEMA,
    AccountingValidationError,
    AccountingVerifier,
    PositionStatus,
)

_TIMEFRAME = "1h"
_MANAGER = "simple"


def _open_time(index: int) -> datetime:
    """Build a deterministic UTC open_time for row ``index``."""
    return datetime(2024, 1, 1, hour=index, tzinfo=UTC)


def _accounting_frame(
    *,
    symbols: list[str] | None = None,
    position_ids: list[str] | None = None,
    statuses: list[str] | None = None,
    quantities: list[float] | None = None,
    equities: list[float] | None = None,
    realized: list[float] | None = None,
    open_times: list[datetime] | None = None,
    managers: list[str] | None = None,
) -> pl.DataFrame:
    """Build a canonical accounting frame for verifier tests."""
    symbols = symbols if symbols is not None else ["BTCUSDT", "ETHUSDT"]
    row_count = len(symbols)
    open_times = (
        open_times if open_times is not None else [_open_time(index) for index in range(row_count)]
    )
    statuses = statuses if statuses is not None else [PositionStatus.OPEN.value] * row_count
    position_ids = (
        position_ids
        if position_ids is not None
        else [f"pos-{index + 1:08d}" for index in range(row_count)]
    )
    quantities = quantities if quantities is not None else [1.0] * row_count
    equities = equities if equities is not None else [110.0] * row_count
    realized = realized if realized is not None else [0.0] * row_count
    managers = managers if managers is not None else [_MANAGER] * row_count
    return pl.DataFrame(
        {
            "symbol": symbols,
            "timeframe": [_TIMEFRAME] * row_count,
            "open_time": open_times,
            "manager": managers,
            "position_id": position_ids,
            "position_status": statuses,
            "quantity": quantities,
            "average_entry_price": [100.0] * row_count,
            "mark_price": [110.0] * row_count,
            "position_value": [110.0] * row_count,
            "market_value": [110.0] * row_count,
            "cash": [0.0] * row_count,
            "realized_pnl": realized,
            "unrealized_pnl": [10.0] * row_count,
            "total_pnl": [10.0] * row_count,
            "gross_exposure": [220.0] * row_count,
            "net_exposure": [220.0] * row_count,
            "equity": equities,
            "return_pct": [10.0 / 110.0] * row_count,
            "model_name": ["alpha-lgbm"] * row_count,
            "model_version": ["1.0.0"] * row_count,
            "optimizer": ["equal_weight"] * row_count,
            "policy": ["fixed_risk"] * row_count,
        },
        schema=dict(COLUMN_DTYPES),
    ).select(list(CANONICAL_COLUMN_ORDER))


def test_verifier_passes_canonical_frame() -> None:
    """A clean canonical accounting frame passes verification."""
    report = AccountingVerifier().verify(_accounting_frame())
    assert report.passed is True
    assert report.rows_checked == 2
    assert report.duplicate_timestamp_rows == 0
    assert report.null_rows == 0
    assert report.warnings == ()


def test_verifier_requires_merged_schema_identity() -> None:
    """Canonical frames cast to MERGED_ACCOUNTING_SCHEMA remain verifiable."""
    frame = _accounting_frame().cast(MERGED_ACCOUNTING_SCHEMA)
    report = AccountingVerifier().verify(frame)
    assert report.passed is True


def test_verifier_rejects_missing_columns() -> None:
    """Missing required columns raise ACC-VERIFICATION-001."""
    frame = _accounting_frame().drop("equity")
    with pytest.raises(AccountingValidationError) as exc_info:
        AccountingVerifier().verify(frame)
    assert exc_info.value.error_code == ERROR_REQUIRED_COLUMNS


def test_verifier_rejects_dtype_mismatch() -> None:
    """Column dtype mismatches raise ACC-VERIFICATION-002."""
    bad_dtype = _accounting_frame().with_columns(pl.col("quantity").cast(pl.Int64))
    with pytest.raises(AccountingValidationError) as exc_info:
        AccountingVerifier().verify(bad_dtype)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH


def test_verifier_reports_duplicate_primary_keys() -> None:
    """Duplicate primary keys fail verification with a warning."""
    duplicates = _accounting_frame(
        symbols=["BTCUSDT", "BTCUSDT"],
        position_ids=["pos-00000001", "pos-00000001"],
        open_times=[_open_time(0), _open_time(0)],
    )
    report = AccountingVerifier().verify(duplicates)
    assert report.passed is False
    assert report.duplicate_timestamp_rows == 1
    assert "Duplicate accounting primary keys detected." in report.warnings


def test_verifier_reports_negative_quantity() -> None:
    """Negative quantity fails verification with a warning."""
    negative = _accounting_frame(quantities=[1.0, -0.5])
    report = AccountingVerifier().verify(negative)
    assert report.passed is False
    assert report.invalid_numeric_rows == 1
    assert "Negative quantity values detected." in report.warnings


def test_verifier_reports_negative_equity() -> None:
    """Negative equity fails verification with a warning."""
    negative = _accounting_frame(equities=[110.0, -5.0])
    report = AccountingVerifier().verify(negative)
    assert report.passed is False
    assert report.invalid_numeric_rows == 1
    assert "Negative equity values detected." in report.warnings


def test_verifier_reports_nan_values() -> None:
    """NaN numeric values fail verification."""
    non_finite = _accounting_frame(realized=[0.0, float("nan")])
    report = AccountingVerifier().verify(non_finite)
    assert report.passed is False
    assert report.nan_rows > 0 or "Non-finite numeric values detected." in report.warnings


def test_verifier_reports_incomplete_lineage() -> None:
    """Blank lineage metadata fails verification with a warning."""
    incomplete = _accounting_frame(managers=[_MANAGER, ""])
    report = AccountingVerifier().verify(incomplete)
    assert report.passed is False
    assert "Incomplete lineage metadata detected." in report.warnings


def test_verifier_reports_non_canonical_order() -> None:
    """A non-canonical column order fails verification with a warning."""
    reordered = _accounting_frame().select(list(reversed(CANONICAL_COLUMN_ORDER)))
    report = AccountingVerifier().verify(reordered)
    assert report.passed is False
    assert "Frame column order does not match canonical order." in report.warnings


def test_verifier_reports_unsorted_open_time() -> None:
    """A frame not sorted by open_time fails verification with a warning."""
    unsorted = _accounting_frame(open_times=[_open_time(3), _open_time(1)])
    report = AccountingVerifier().verify(unsorted)
    assert report.passed is False
    assert "Frame is not sorted by open_time." in report.warnings


def test_verifier_reports_invalid_status() -> None:
    """Invalid position_status values fail verification with a warning."""
    invalid = _accounting_frame(statuses=[PositionStatus.OPEN.value, "PENDING"])
    report = AccountingVerifier().verify(invalid)
    assert report.passed is False
    assert "Invalid PositionStatus values detected." in report.warnings
