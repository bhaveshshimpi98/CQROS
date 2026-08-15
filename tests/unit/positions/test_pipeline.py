"""Unit tests for CQROS ``PositionPipeline``."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.positions import (
    CANONICAL_COLUMN_ORDER,
    MERGED_POSITION_SCHEMA,
    AverageCostPositionEngine,
    PositionEngineRegistry,
    PositionPipeline,
    PositionStatus,
    PositionValidationError,
)
from cqros.positions.pipeline import PositionPipeline as PositionPipelineDirect

_TIMEFRAME = "1h"
_MODEL_NAME = "alpha-lgbm"
_MODEL_VERSION = "1.0.0"
_POLICY = "fixed_risk"
_OPTIMIZER = "equal_weight"
_MANAGER = "simple"


def _execution_time(index: int) -> datetime:
    """Build a deterministic UTC execution_time for row ``index``."""
    return datetime(2024, 1, 1, hour=index, tzinfo=UTC)


def _trade_frame(
    *,
    sides: list[str],
    quantities: list[float],
    prices: list[float],
) -> pl.DataFrame:
    """Build an executed-trade-shaped frame for pipeline tests."""
    row_count = len(sides)
    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * row_count,
            "timeframe": [_TIMEFRAME] * row_count,
            "open_time": [_execution_time(index) for index in range(row_count)],
            "model_name": [_MODEL_NAME] * row_count,
            "model_version": [_MODEL_VERSION] * row_count,
            "optimizer": [_OPTIMIZER] * row_count,
            "policy": [_POLICY] * row_count,
            "manager": [_MANAGER] * row_count,
            "signal": sides,
            "side": sides,
            "order_type": ["MARKET"] * row_count,
            "requested_quantity": quantities,
            "executed_quantity": quantities,
            "requested_price": prices,
            "executed_price": prices,
            "fees": [0.0] * row_count,
            "slippage": [0.0] * row_count,
            "status": ["FILLED"] * row_count,
            "execution_time": [_execution_time(index) for index in range(row_count)],
        }
    )


class _RecordingEngine:
    """Engine stub that records build calls and returns a frame."""

    def __init__(self, frame: pl.DataFrame) -> None:
        self.frame = frame
        self.calls: list[tuple[pl.DataFrame, str]] = []

    def build(self, trades: pl.DataFrame, *, manager: str) -> pl.DataFrame:
        self.calls.append((trades, manager))
        return self.frame


def test_position_pipeline_is_exported_from_package() -> None:
    """Package export matches the pipeline module by identity."""
    assert PositionPipeline is PositionPipelineDirect


def test_pipeline_runs_registered_engine() -> None:
    """Pipeline resolves the engine, stamps manager, and finalizes schema."""
    registry = PositionEngineRegistry()
    registry.register("average_cost", AverageCostPositionEngine())
    pipeline = PositionPipeline(registry)
    trades = _trade_frame(sides=["BUY"], quantities=[1.25], prices=[50.0])
    original = trades.clone()
    positions = pipeline.run(trades, manager=_MANAGER, engine_name="average_cost")
    assert_frame_equal(trades, original)
    assert tuple(positions.columns) == CANONICAL_COLUMN_ORDER
    assert positions.schema == MERGED_POSITION_SCHEMA
    assert positions["status"].to_list() == [PositionStatus.OPEN.value]
    assert positions["manager"].to_list() == [_MANAGER]
    assert positions["quantity"].to_list() == [1.25]
    assert positions["average_entry_price"].to_list() == [50.0]


def test_pipeline_rejects_unknown_engine_and_blank_manager() -> None:
    """Unknown engine names and blank managers raise validation errors."""
    registry = PositionEngineRegistry()
    registry.register("average_cost", AverageCostPositionEngine())
    pipeline = PositionPipeline(registry)
    trades = _trade_frame(sides=["BUY"], quantities=[1.0], prices=[100.0])
    with pytest.raises(PositionValidationError, match="not registered") as exc_info:
        pipeline.run(trades, manager=_MANAGER, engine_name="missing")
    assert exc_info.value.error_code == "POS_REG_UNKNOWN"
    with pytest.raises(PositionValidationError, match="non-blank") as exc_info:
        pipeline.run(trades, manager="", engine_name="average_cost")
    assert exc_info.value.error_code == "POS_PIPE_MANAGER_BLANK"


def test_pipeline_finalizes_engine_output() -> None:
    """Pipeline reorders and casts engine output to the merged schema."""
    trades = _trade_frame(sides=["BUY"], quantities=[1.0], prices=[100.0])
    position_like = AverageCostPositionEngine().build(trades, manager=_MANAGER)
    reordered = position_like.select(list(reversed(position_like.columns)))
    stub = _RecordingEngine(reordered)
    registry = PositionEngineRegistry()
    registry.register("stub", stub)
    pipeline = PositionPipeline(registry)
    result = pipeline.run(trades, manager=_MANAGER, engine_name="stub")
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result.schema == MERGED_POSITION_SCHEMA
    assert len(stub.calls) == 1
    assert stub.calls[0][1] == _MANAGER
