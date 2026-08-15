"""Unit tests for CQROS ``AccountingPipeline``."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.accounting import (
    CANONICAL_COLUMN_ORDER,
    MERGED_ACCOUNTING_SCHEMA,
    AccountingEngineRegistry,
    AccountingPipeline,
    AccountingValidationError,
    PositionStatus,
    SimplePortfolioAccountingEngine,
)
from cqros.accounting.pipeline import AccountingPipeline as AccountingPipelineDirect

_TIMEFRAME = "1h"
_MODEL_NAME = "alpha-lgbm"
_MODEL_VERSION = "1.0.0"
_POLICY = "fixed_risk"
_OPTIMIZER = "equal_weight"
_MANAGER = "simple"


def _opened_at(index: int) -> datetime:
    """Build a deterministic UTC opened_at timestamp for row ``index``."""
    return datetime(2024, 1, 1, hour=index, tzinfo=UTC)


def _position_frame(
    *,
    symbols: list[str] | None = None,
    quantities: list[float] | None = None,
    market_prices: list[float] | None = None,
) -> pl.DataFrame:
    """Build a canonical-position-shaped frame for pipeline tests."""
    symbols = symbols if symbols is not None else ["BTCUSDT"]
    row_count = len(symbols)
    quantities = quantities if quantities is not None else [1.0] * row_count
    market_prices = market_prices if market_prices is not None else [100.0] * row_count
    return pl.DataFrame(
        {
            "symbol": symbols,
            "timeframe": [_TIMEFRAME] * row_count,
            "position_id": [f"pos-{index + 1:08d}" for index in range(row_count)],
            "side": ["LONG"] * row_count,
            "status": [PositionStatus.OPEN.value] * row_count,
            "quantity": quantities,
            "average_entry_price": [100.0] * row_count,
            "market_price": market_prices,
            "realized_pnl": [0.0] * row_count,
            "unrealized_pnl": [0.0] * row_count,
            "opened_at": [_opened_at(index) for index in range(row_count)],
            "model_name": [_MODEL_NAME] * row_count,
            "model_version": [_MODEL_VERSION] * row_count,
            "optimizer": [_OPTIMIZER] * row_count,
            "policy": [_POLICY] * row_count,
            "manager": [_MANAGER] * row_count,
        }
    )


class _RecordingEngine:
    """Engine stub that records build calls and returns a fixed frame."""

    def __init__(self, frame: pl.DataFrame) -> None:
        self.frame = frame
        self.calls: list[tuple[pl.DataFrame, str]] = []

    def build(self, positions: pl.DataFrame, *, manager: str) -> pl.DataFrame:
        self.calls.append((positions, manager))
        return self.frame


def test_accounting_pipeline_is_exported_from_package() -> None:
    """Package export matches the pipeline module by identity."""
    assert AccountingPipeline is AccountingPipelineDirect


def test_pipeline_resolves_simple_engine_and_finalizes_schema() -> None:
    """Pipeline resolves the simple engine, stamps manager, finalizes schema."""
    registry = AccountingEngineRegistry()
    registry.register("simple", SimplePortfolioAccountingEngine())
    pipeline = AccountingPipeline(registry)
    positions = _position_frame(quantities=[2.0], market_prices=[110.0])
    original = positions.clone()
    accounting = pipeline.run(positions, manager=_MANAGER, engine_name="simple")
    assert_frame_equal(positions, original)
    assert tuple(accounting.columns) == CANONICAL_COLUMN_ORDER
    assert accounting.schema == MERGED_ACCOUNTING_SCHEMA
    assert accounting["manager"].to_list() == [_MANAGER]
    assert accounting["market_value"].to_list() == [220.0]


def test_pipeline_default_engine_name_is_simple() -> None:
    """The pipeline defaults to the ``simple`` engine name."""
    registry = AccountingEngineRegistry()
    registry.register("simple", SimplePortfolioAccountingEngine())
    pipeline = AccountingPipeline(registry)
    accounting = pipeline.run(_position_frame(), manager=_MANAGER)
    assert accounting.height == 1


def test_pipeline_rejects_unknown_engine_and_blank_manager() -> None:
    """Unknown engine names and blank managers raise validation errors."""
    registry = AccountingEngineRegistry()
    registry.register("simple", SimplePortfolioAccountingEngine())
    pipeline = AccountingPipeline(registry)
    positions = _position_frame()
    with pytest.raises(AccountingValidationError, match="not registered") as exc_info:
        pipeline.run(positions, manager=_MANAGER, engine_name="missing")
    assert exc_info.value.error_code == "ACC_REG_UNKNOWN"
    with pytest.raises(AccountingValidationError, match="non-blank") as exc_info:
        pipeline.run(positions, manager="", engine_name="simple")
    assert exc_info.value.error_code == "ACC_PIPE_MANAGER_BLANK"


def test_pipeline_finalizes_engine_output() -> None:
    """Pipeline reorders and casts engine output to the merged schema."""
    positions = _position_frame()
    accounting_like = SimplePortfolioAccountingEngine().build(positions, manager=_MANAGER)
    reordered = accounting_like.select(list(reversed(accounting_like.columns)))
    stub = _RecordingEngine(reordered)
    registry = AccountingEngineRegistry()
    registry.register("stub", stub)
    pipeline = AccountingPipeline(registry)
    result = pipeline.run(positions, manager=_MANAGER, engine_name="stub")
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result.schema == MERGED_ACCOUNTING_SCHEMA
    assert len(stub.calls) == 1
    assert stub.calls[0][1] == _MANAGER
