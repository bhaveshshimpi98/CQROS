"""Unit tests for CQROS ``PortfolioRiskPipeline``."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.portfolio_risk import (
    CANONICAL_COLUMN_ORDER,
    MERGED_PORTFOLIO_RISK_SCHEMA,
    PortfolioRiskManagerRegistry,
    PortfolioRiskPipeline,
    PortfolioRiskState,
    PortfolioRiskValidationError,
    ShutdownReason,
    SimplePortfolioRiskManager,
)
from cqros.portfolio_risk.pipeline import PortfolioRiskPipeline as PortfolioRiskPipelineDirect

_TIMEFRAME = "1h"
_MODEL_NAME = "alpha-lgbm"
_MODEL_VERSION = "1.0.0"
_POLICY = "fixed_risk"
_OPTIMIZER = "equal_weight"
_MANAGER = "simple"
_EQUITY = 1000.0


def _open_time(index: int) -> datetime:
    """Build a deterministic UTC open_time timestamp for row ``index``."""
    return datetime(2024, 1, 1, hour=index, tzinfo=UTC)


def _accounting_frame(
    *,
    symbols: list[str] | None = None,
    position_ids: list[str] | None = None,
    totals: list[float] | None = None,
    gross_exposures: list[float] | None = None,
) -> pl.DataFrame:
    """Build an accounting-shaped frame for pipeline tests."""
    symbols = symbols if symbols is not None else ["BTCUSDT"]
    row_count = len(symbols)
    position_ids = (
        position_ids
        if position_ids is not None
        else [f"pos-{index + 1:08d}" for index in range(row_count)]
    )
    totals = totals if totals is not None else [0.0] * row_count
    gross_exposures = gross_exposures if gross_exposures is not None else [500.0] * row_count
    return pl.DataFrame(
        {
            "symbol": symbols,
            "timeframe": [_TIMEFRAME] * row_count,
            "open_time": [_open_time(index) for index in range(row_count)],
            "position_id": position_ids,
            "equity": [_EQUITY] * row_count,
            "gross_exposure": gross_exposures,
            "net_exposure": [500.0] * row_count,
            "realized_pnl": [0.0] * row_count,
            "unrealized_pnl": [0.0] * row_count,
            "total_pnl": totals,
            "model_name": [_MODEL_NAME] * row_count,
            "model_version": [_MODEL_VERSION] * row_count,
            "optimizer": [_OPTIMIZER] * row_count,
            "policy": [_POLICY] * row_count,
        }
    )


def _position_frame(
    *,
    symbols: list[str] | None = None,
    position_ids: list[str] | None = None,
) -> pl.DataFrame:
    """Build a minimal position identity frame for pipeline tests."""
    symbols = symbols if symbols is not None else ["BTCUSDT"]
    row_count = len(symbols)
    position_ids = (
        position_ids
        if position_ids is not None
        else [f"pos-{index + 1:08d}" for index in range(row_count)]
    )
    return pl.DataFrame(
        {
            "symbol": symbols,
            "timeframe": [_TIMEFRAME] * row_count,
            "position_id": position_ids,
        }
    )


def _risk_like_frame() -> pl.DataFrame:
    """Return a canonical portfolio-risk-shaped frame for stub managers."""
    return SimplePortfolioRiskManager().evaluate(
        _accounting_frame(),
        _position_frame(),
        manager=_MANAGER,
    )


class _RecordingManager:
    """Manager stub that records evaluate calls and returns a fixed frame."""

    def __init__(self, frame: pl.DataFrame) -> None:
        self.frame = frame
        self.calls: list[tuple[pl.DataFrame, pl.DataFrame, str]] = []

    def evaluate(
        self,
        accounting: pl.DataFrame,
        positions: pl.DataFrame,
        *,
        manager: str,
    ) -> pl.DataFrame:
        self.calls.append((accounting, positions, manager))
        return self.frame


def test_portfolio_risk_pipeline_is_exported_from_package() -> None:
    """Package export matches the pipeline module by identity."""
    assert PortfolioRiskPipeline is PortfolioRiskPipelineDirect


def test_pipeline_resolves_simple_manager_and_finalizes_schema() -> None:
    """Pipeline resolves the simple manager, stamps manager, finalizes schema."""
    registry = PortfolioRiskManagerRegistry()
    registry.register("simple", SimplePortfolioRiskManager())
    pipeline = PortfolioRiskPipeline(registry)
    accounting = _accounting_frame()
    positions = _position_frame()
    original_accounting = accounting.clone()
    original_positions = positions.clone()
    result = pipeline.run(
        accounting,
        positions,
        manager=_MANAGER,
        risk_manager_name="simple",
    )
    assert_frame_equal(accounting, original_accounting)
    assert_frame_equal(positions, original_positions)
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result.schema == MERGED_PORTFOLIO_RISK_SCHEMA
    assert result["manager"].to_list() == [_MANAGER]
    assert result["portfolio_risk_state"].to_list() == [PortfolioRiskState.NORMAL.value]
    assert result["shutdown_reason"].to_list() == [ShutdownReason.NONE.value]


def test_pipeline_default_risk_manager_name_is_simple() -> None:
    """The pipeline defaults to the ``simple`` risk-manager name."""
    registry = PortfolioRiskManagerRegistry()
    registry.register("simple", SimplePortfolioRiskManager())
    pipeline = PortfolioRiskPipeline(registry)
    result = pipeline.run(_accounting_frame(), _position_frame(), manager=_MANAGER)
    assert result.height == 1


def test_pipeline_rejects_blank_names() -> None:
    """Blank risk-manager names and blank managers raise validation errors."""
    registry = PortfolioRiskManagerRegistry()
    registry.register("simple", SimplePortfolioRiskManager())
    pipeline = PortfolioRiskPipeline(registry)
    accounting = _accounting_frame()
    positions = _position_frame()
    with pytest.raises(PortfolioRiskValidationError, match="non-blank") as exc_info:
        pipeline.run(
            accounting,
            positions,
            manager=_MANAGER,
            risk_manager_name="",
        )
    assert exc_info.value.error_code == "PRISK_PIPE_NAME_BLANK"
    with pytest.raises(PortfolioRiskValidationError, match="non-blank") as exc_info:
        pipeline.run(accounting, positions, manager="", risk_manager_name="simple")
    assert exc_info.value.error_code == "PRISK_PIPE_MANAGER_BLANK"


def test_pipeline_rejects_unknown_risk_manager() -> None:
    """Unknown risk-manager names raise validation errors."""
    registry = PortfolioRiskManagerRegistry()
    registry.register("simple", SimplePortfolioRiskManager())
    pipeline = PortfolioRiskPipeline(registry)
    with pytest.raises(PortfolioRiskValidationError, match="not registered") as exc_info:
        pipeline.run(
            _accounting_frame(),
            _position_frame(),
            manager=_MANAGER,
            risk_manager_name="missing",
        )
    assert exc_info.value.error_code == "PRISK_REG_UNKNOWN"


def test_pipeline_finalizes_manager_output() -> None:
    """Pipeline reorders and casts manager output to the merged schema."""
    risk_like = _risk_like_frame()
    reordered = risk_like.select(list(reversed(risk_like.columns)))
    stub = _RecordingManager(reordered)
    registry = PortfolioRiskManagerRegistry()
    registry.register("stub", stub)
    pipeline = PortfolioRiskPipeline(registry)
    result = pipeline.run(
        _accounting_frame(),
        _position_frame(),
        manager=_MANAGER,
        risk_manager_name="stub",
    )
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result.schema == MERGED_PORTFOLIO_RISK_SCHEMA
    assert len(stub.calls) == 1
    assert stub.calls[0][2] == _MANAGER


def test_pipeline_rejects_duplicate_primary_keys() -> None:
    """Duplicate primary keys in manager output raise validation errors."""
    base = _risk_like_frame()
    duplicate = pl.concat([base, base])
    stub = _RecordingManager(duplicate)
    registry = PortfolioRiskManagerRegistry()
    registry.register("stub", stub)
    pipeline = PortfolioRiskPipeline(registry)
    with pytest.raises(PortfolioRiskValidationError, match="duplicate primary keys") as exc_info:
        pipeline.run(
            _accounting_frame(),
            _position_frame(),
            manager=_MANAGER,
            risk_manager_name="stub",
        )
    assert exc_info.value.error_code == "PRISK_PIPE_DUPLICATE_KEYS"


def test_pipeline_rejects_missing_columns() -> None:
    """Missing required portfolio-risk columns on manager output are rejected."""
    incomplete = _risk_like_frame().drop("allow_new_entries")
    stub = _RecordingManager(incomplete)
    registry = PortfolioRiskManagerRegistry()
    registry.register("stub", stub)
    pipeline = PortfolioRiskPipeline(registry)
    with pytest.raises(PortfolioRiskValidationError, match="missing required columns") as exc_info:
        pipeline.run(
            _accounting_frame(),
            _position_frame(),
            manager=_MANAGER,
            risk_manager_name="stub",
        )
    assert exc_info.value.error_code == "PRISK_PIPE_MISSING_COLUMNS"
    assert "allow_new_entries" in exc_info.value.details["missing_columns"]


def test_pipeline_preserves_input_immutability() -> None:
    """Pipeline must not mutate caller-supplied accounting or position frames."""
    registry = PortfolioRiskManagerRegistry()
    registry.register("simple", SimplePortfolioRiskManager())
    pipeline = PortfolioRiskPipeline(registry)
    accounting = _accounting_frame(totals=[-5.0])
    positions = _position_frame()
    accounting_before = accounting.clone()
    positions_before = positions.clone()
    pipeline.run(accounting, positions, manager=_MANAGER)
    assert_frame_equal(accounting, accounting_before)
    assert_frame_equal(positions, positions_before)
