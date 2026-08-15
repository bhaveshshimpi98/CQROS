"""Unit tests for CQROS ``PortfolioRiskVerifier``."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from cqros.portfolio_risk import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    ERROR_REQUIRED_COLUMNS,
    ERROR_SCHEMA_MISMATCH,
    MERGED_PORTFOLIO_RISK_SCHEMA,
    PortfolioRiskState,
    PortfolioRiskValidationError,
    PortfolioRiskVerifier,
    ShutdownReason,
)

_TIMEFRAME = "1h"
_MANAGER = "simple"
_EQUITY = 1000.0


def _open_time(index: int) -> datetime:
    """Build a deterministic UTC open_time for row ``index``."""
    return datetime(2024, 1, 1, hour=index, tzinfo=UTC)


def _portfolio_risk_frame(
    *,
    symbols: list[str] | None = None,
    position_ids: list[str] | None = None,
    open_times: list[datetime] | None = None,
    states: list[str] | None = None,
    reasons: list[str] | None = None,
    allow_entries: list[bool] | None = None,
    cooldown_untils: list[datetime | None] | None = None,
    equities: list[float] | None = None,
    totals: list[float] | None = None,
    managers: list[str] | None = None,
) -> pl.DataFrame:
    """Build a canonical portfolio-risk frame for verifier tests."""
    symbols = symbols if symbols is not None else ["BTCUSDT", "ETHUSDT"]
    row_count = len(symbols)
    open_times = (
        open_times if open_times is not None else [_open_time(index) for index in range(row_count)]
    )
    position_ids = (
        position_ids
        if position_ids is not None
        else [f"pos-{index + 1:08d}" for index in range(row_count)]
    )
    states = states if states is not None else [PortfolioRiskState.NORMAL.value] * row_count
    reasons = reasons if reasons is not None else [ShutdownReason.NONE.value] * row_count
    allow_entries = allow_entries if allow_entries is not None else [True] * row_count
    cooldown_untils = cooldown_untils if cooldown_untils is not None else [None] * row_count
    equities = equities if equities is not None else [_EQUITY] * row_count
    totals = totals if totals is not None else [0.0] * row_count
    managers = managers if managers is not None else [_MANAGER] * row_count
    return pl.DataFrame(
        {
            "symbol": symbols,
            "timeframe": [_TIMEFRAME] * row_count,
            "open_time": open_times,
            "manager": managers,
            "position_id": position_ids,
            "equity": equities,
            "gross_exposure": [500.0] * row_count,
            "net_exposure": [500.0] * row_count,
            "daily_realized_pnl": [0.0] * row_count,
            "daily_unrealized_pnl": [0.0] * row_count,
            "daily_total_pnl": totals,
            "daily_return_pct": [0.0] * row_count,
            "daily_drawdown_pct": [0.0] * row_count,
            "portfolio_risk_state": states,
            "allow_new_entries": allow_entries,
            "shutdown_reason": reasons,
            "cooldown_until": cooldown_untils,
            "model_name": ["alpha-lgbm"] * row_count,
            "model_version": ["1.0.0"] * row_count,
            "optimizer": ["equal_weight"] * row_count,
            "policy": ["fixed_risk"] * row_count,
        },
        schema=dict(COLUMN_DTYPES),
    ).select(list(CANONICAL_COLUMN_ORDER))


def test_verifier_passes_canonical_frame() -> None:
    """A clean canonical portfolio-risk frame passes verification."""
    report = PortfolioRiskVerifier().verify(_portfolio_risk_frame())
    assert report.passed is True
    assert report.rows_checked == 2
    assert report.duplicate_timestamp_rows == 0
    assert report.null_rows == 0
    assert report.warnings == ()


def test_verifier_requires_merged_schema_identity() -> None:
    """Canonical frames cast to MERGED_PORTFOLIO_RISK_SCHEMA remain verifiable."""
    frame = _portfolio_risk_frame().cast(MERGED_PORTFOLIO_RISK_SCHEMA)
    report = PortfolioRiskVerifier().verify(frame)
    assert report.passed is True


def test_verifier_rejects_missing_columns() -> None:
    """Missing required columns raise PRISK-VERIFICATION-001."""
    frame = _portfolio_risk_frame().drop("equity")
    with pytest.raises(PortfolioRiskValidationError) as exc_info:
        PortfolioRiskVerifier().verify(frame)
    assert exc_info.value.error_code == ERROR_REQUIRED_COLUMNS


def test_verifier_rejects_dtype_mismatch() -> None:
    """Column dtype mismatches raise PRISK-VERIFICATION-002."""
    bad_dtype = _portfolio_risk_frame().with_columns(pl.col("equity").cast(pl.Int64))
    with pytest.raises(PortfolioRiskValidationError) as exc_info:
        PortfolioRiskVerifier().verify(bad_dtype)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH


def test_verifier_reports_duplicate_primary_keys() -> None:
    """Duplicate primary keys fail verification with a warning."""
    duplicates = _portfolio_risk_frame(
        symbols=["BTCUSDT", "BTCUSDT"],
        position_ids=["pos-00000001", "pos-00000001"],
        open_times=[_open_time(0), _open_time(0)],
    )
    report = PortfolioRiskVerifier().verify(duplicates)
    assert report.passed is False
    assert report.duplicate_timestamp_rows == 1
    assert "Duplicate portfolio-risk primary keys detected." in report.warnings


def test_verifier_reports_null_rows() -> None:
    """Null values in required columns fail verification with a warning."""
    nulls = _portfolio_risk_frame().with_columns(
        pl.when(pl.arange(0, pl.len()) == 1).then(None).otherwise(pl.col("symbol")).alias("symbol")
    )
    report = PortfolioRiskVerifier().verify(nulls)
    assert report.passed is False
    assert report.null_rows > 0
    assert "Rows containing NULL values." in report.warnings


def test_verifier_reports_invalid_enums() -> None:
    """Invalid portfolio_risk_state and shutdown_reason values fail verification."""
    invalid_state = _portfolio_risk_frame(
        states=[PortfolioRiskState.NORMAL.value, "PANIC"],
    )
    report = PortfolioRiskVerifier().verify(invalid_state)
    assert report.passed is False
    assert "Invalid PortfolioRiskState values detected." in report.warnings

    invalid_reason = _portfolio_risk_frame(
        reasons=[ShutdownReason.NONE.value, "UNKNOWN_REASON"],
    )
    report = PortfolioRiskVerifier().verify(invalid_reason)
    assert report.passed is False
    assert "Invalid ShutdownReason values detected." in report.warnings


def test_verifier_reports_boolean_nulls() -> None:
    """Null allow_new_entries values fail verification with a warning."""
    null_bool = _portfolio_risk_frame().with_columns(
        pl.when(pl.arange(0, pl.len()) == 1)
        .then(None)
        .otherwise(pl.col("allow_new_entries"))
        .alias("allow_new_entries")
    )
    report = PortfolioRiskVerifier().verify(null_bool)
    assert report.passed is False
    assert "Invalid allow_new_entries boolean values detected." in report.warnings


def test_verifier_reports_cooldown_inconsistency() -> None:
    """Inconsistent cooldown_until values fail verification with a warning."""
    # DAILY_LOSS_LIMIT requires a non-null cooldown_until strictly after open_time.
    inconsistent = _portfolio_risk_frame(
        symbols=["BTCUSDT"],
        position_ids=["pos-00000001"],
        open_times=[_open_time(0)],
        states=[PortfolioRiskState.SHUTDOWN.value],
        reasons=[ShutdownReason.DAILY_LOSS_LIMIT.value],
        allow_entries=[False],
        cooldown_untils=[None],
    )
    report = PortfolioRiskVerifier().verify(inconsistent)
    assert report.passed is False
    assert "Inconsistent cooldown_until values detected." in report.warnings

    # NONE must not carry a cooldown_until value.
    present_when_none = _portfolio_risk_frame(
        symbols=["BTCUSDT"],
        position_ids=["pos-00000001"],
        open_times=[_open_time(0)],
        cooldown_untils=[_open_time(0) + timedelta(hours=24)],
    )
    report = PortfolioRiskVerifier().verify(present_when_none)
    assert report.passed is False
    assert "Inconsistent cooldown_until values detected." in report.warnings


def test_verifier_reports_incomplete_lineage() -> None:
    """Blank lineage metadata fails verification with a warning."""
    incomplete = _portfolio_risk_frame(managers=[_MANAGER, ""])
    report = PortfolioRiskVerifier().verify(incomplete)
    assert report.passed is False
    assert "Incomplete lineage metadata detected." in report.warnings


def test_verifier_reports_unsorted_open_time() -> None:
    """A frame not sorted by open_time fails verification with a warning."""
    unsorted = _portfolio_risk_frame(open_times=[_open_time(3), _open_time(1)])
    report = PortfolioRiskVerifier().verify(unsorted)
    assert report.passed is False
    assert "Frame is not sorted by open_time." in report.warnings


def test_verifier_reports_non_canonical_order() -> None:
    """A non-canonical column order fails verification with a warning."""
    reordered = _portfolio_risk_frame().select(list(reversed(CANONICAL_COLUMN_ORDER)))
    report = PortfolioRiskVerifier().verify(reordered)
    assert report.passed is False
    assert "Frame column order does not match canonical order." in report.warnings
