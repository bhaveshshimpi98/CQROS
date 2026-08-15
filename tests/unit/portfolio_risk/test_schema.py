"""Unit tests for CQROS merged portfolio risk decision schema."""

from __future__ import annotations

import polars as pl

from cqros.portfolio_risk import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    DEFAULT_COOLDOWN_HOURS,
    DEFAULT_DAILY_LOSS_LIMIT,
    DEFAULT_GROSS_EXPOSURE_LIMIT,
    MERGED_PORTFOLIO_RISK_SCHEMA,
    METADATA_COLUMNS,
    PORTFOLIO_RISK_COLUMNS,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    PortfolioRiskState,
    ShutdownReason,
    portfolio_risk_states,
    shutdown_reasons,
    values,
)
from cqros.portfolio_risk.schema import (
    MERGED_PORTFOLIO_RISK_SCHEMA as MERGED_PORTFOLIO_RISK_SCHEMA_DIRECT,
)


def test_primary_key_and_required_columns() -> None:
    """Primary key and required columns match the canonical portfolio-risk contract."""
    assert PRIMARY_KEY_COLUMNS == ("symbol", "timeframe", "open_time", "position_id")
    assert REQUIRED_COLUMNS == CANONICAL_COLUMN_ORDER
    assert "model_name" in METADATA_COLUMNS
    assert "model_version" in METADATA_COLUMNS
    assert "optimizer" in METADATA_COLUMNS
    assert "policy" in METADATA_COLUMNS


def test_portfolio_risk_columns_contain_required_domain_columns() -> None:
    """PORTFOLIO_RISK_COLUMNS enumerates identity, exposure, and decision fields."""
    for column in (
        "manager",
        "position_id",
        "equity",
        "gross_exposure",
        "net_exposure",
        "daily_realized_pnl",
        "daily_unrealized_pnl",
        "daily_total_pnl",
        "daily_return_pct",
        "daily_drawdown_pct",
        "portfolio_risk_state",
        "allow_new_entries",
        "shutdown_reason",
        "cooldown_until",
    ):
        assert column in PORTFOLIO_RISK_COLUMNS


def test_column_dtypes_and_merged_schema() -> None:
    """Merged schema dtypes match COLUMN_DTYPES in canonical order."""
    assert MERGED_PORTFOLIO_RISK_SCHEMA is MERGED_PORTFOLIO_RISK_SCHEMA_DIRECT
    assert MERGED_PORTFOLIO_RISK_SCHEMA.names() == list(CANONICAL_COLUMN_ORDER)
    for column in CANONICAL_COLUMN_ORDER:
        assert MERGED_PORTFOLIO_RISK_SCHEMA[column] == COLUMN_DTYPES[column]
    assert COLUMN_DTYPES["open_time"] == pl.Datetime("us", "UTC")
    assert COLUMN_DTYPES["cooldown_until"] == pl.Datetime("us", "UTC")
    assert COLUMN_DTYPES["equity"] == pl.Float64
    assert COLUMN_DTYPES["gross_exposure"] == pl.Float64
    assert COLUMN_DTYPES["allow_new_entries"] == pl.Boolean
    assert COLUMN_DTYPES["portfolio_risk_state"] == pl.Utf8
    assert COLUMN_DTYPES["symbol"] == pl.Utf8


def test_canonical_order_ends_with_metadata_columns() -> None:
    """Canonical column order terminates with the lineage metadata columns."""
    assert CANONICAL_COLUMN_ORDER[-len(METADATA_COLUMNS) :] == METADATA_COLUMNS
    assert CANONICAL_COLUMN_ORDER[0] == "symbol"
    assert CANONICAL_COLUMN_ORDER[1] == "timeframe"
    assert CANONICAL_COLUMN_ORDER[2] == "open_time"


def test_portfolio_risk_state_and_shutdown_reason_helpers() -> None:
    """State and reason helpers expose only the v1 enumeration members."""
    assert PortfolioRiskState.NORMAL.value == "NORMAL"
    assert PortfolioRiskState.WARNING.value == "WARNING"
    assert PortfolioRiskState.SHUTDOWN.value == "SHUTDOWN"
    assert portfolio_risk_states() == (
        PortfolioRiskState.NORMAL,
        PortfolioRiskState.WARNING,
        PortfolioRiskState.SHUTDOWN,
    )
    assert values(PortfolioRiskState) == ("NORMAL", "WARNING", "SHUTDOWN")

    assert ShutdownReason.NONE.value == ""
    assert ShutdownReason.DAILY_LOSS_LIMIT.value == "DAILY_LOSS_LIMIT"
    assert ShutdownReason.COOLDOWN.value == "COOLDOWN"
    assert ShutdownReason.EXPOSURE_LIMIT.value == "EXPOSURE_LIMIT"
    assert shutdown_reasons() == (
        ShutdownReason.NONE,
        ShutdownReason.DAILY_LOSS_LIMIT,
        ShutdownReason.COOLDOWN,
        ShutdownReason.EXPOSURE_LIMIT,
    )
    assert values(ShutdownReason) == (
        "",
        "DAILY_LOSS_LIMIT",
        "COOLDOWN",
        "EXPOSURE_LIMIT",
    )


def test_default_limit_constants() -> None:
    """Default portfolio-risk limit constants match the v1 contract."""
    assert DEFAULT_DAILY_LOSS_LIMIT == 0.02
    assert DEFAULT_GROSS_EXPOSURE_LIMIT == 1.00
    assert DEFAULT_COOLDOWN_HOURS == 24
