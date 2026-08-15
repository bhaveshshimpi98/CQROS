"""Unit tests for CQROS ``ExecutionPipeline``."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.execution import (
    CANONICAL_COLUMN_ORDER,
    MERGED_TRADE_SCHEMA,
    ExecutionPipeline,
    ExecutionSimulatorRegistry,
    ExecutionStatus,
    ExecutionValidationError,
    SimpleExecutionSimulator,
)
from cqros.execution.pipeline import ExecutionPipeline as ExecutionPipelineDirect

_TIMEFRAME = "1h"
_MODEL_NAME = "alpha-lgbm"
_MODEL_VERSION = "1.0.0"
_POLICY = "fixed_risk"
_OPTIMIZER = "equal_weight"
_MANAGER = "simple"
_FIXED_NOW = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)


def _open_time(index: int) -> datetime:
    """Build a deterministic UTC open_time for row ``index``."""
    return datetime(2024, 1, 1, tzinfo=UTC).replace(microsecond=index)


def _order_frame(
    *,
    symbols: list[str],
    sides: list[str],
    quantities: list[float],
    limit_prices: list[float | None] | None = None,
) -> pl.DataFrame:
    """Build an OMS order-shaped frame for pipeline tests."""
    row_count = len(symbols)
    ids = [f"order-{index}" for index in range(row_count)]
    return pl.DataFrame(
        {
            "symbol": symbols,
            "timeframe": [_TIMEFRAME] * row_count,
            "open_time": [_open_time(index) for index in range(row_count)],
            "order_id": ids,
            "parent_order_id": ids,
            "model_name": [_MODEL_NAME] * row_count,
            "model_version": [_MODEL_VERSION] * row_count,
            "policy": [_POLICY] * row_count,
            "optimizer": [_OPTIMIZER] * row_count,
            "side": sides,
            "order_type": ["MARKET"] * row_count,
            "quantity": quantities,
            "limit_price": (limit_prices if limit_prices is not None else [100.0] * row_count),
            "stop_price": [None] * row_count,
            "filled_quantity": [0.0] * row_count,
            "average_fill_price": [None] * row_count,
            "status": ["PENDING"] * row_count,
            "created_at": [_FIXED_NOW] * row_count,
            "updated_at": [_FIXED_NOW] * row_count,
        }
    )


class _RecordingSimulator:
    """Simulator stub that records execute calls and returns a frame."""

    def __init__(self, frame: pl.DataFrame) -> None:
        self.frame = frame
        self.calls: list[tuple[pl.DataFrame, str]] = []

    def execute(self, orders: pl.DataFrame, *, manager: str) -> pl.DataFrame:
        self.calls.append((orders, manager))
        return self.frame


def test_execution_pipeline_is_exported_from_package() -> None:
    """Package export matches the pipeline module by identity."""
    assert ExecutionPipeline is ExecutionPipelineDirect


def test_pipeline_runs_registered_simulator() -> None:
    """Pipeline resolves the simulator, stamps manager, and finalizes schema."""
    registry = ExecutionSimulatorRegistry()
    registry.register("simple", SimpleExecutionSimulator())
    pipeline = ExecutionPipeline(registry)
    orders = _order_frame(
        symbols=["BTCUSDT"],
        sides=["BUY"],
        quantities=[1.25],
        limit_prices=[50.0],
    )
    original = orders.clone()
    trades = pipeline.run(orders, manager=_MANAGER, simulator_name="simple")
    assert_frame_equal(orders, original)
    assert tuple(trades.columns) == CANONICAL_COLUMN_ORDER
    assert trades.schema == MERGED_TRADE_SCHEMA
    assert trades["status"].to_list() == [ExecutionStatus.FILLED.value]
    assert trades["manager"].to_list() == [_MANAGER]
    assert trades["executed_quantity"].to_list() == [1.25]
    assert trades["executed_price"].to_list() == [50.0]


def test_pipeline_rejects_unknown_simulator_and_blank_manager() -> None:
    """Unknown simulator names and blank managers raise validation errors."""
    registry = ExecutionSimulatorRegistry()
    registry.register("simple", SimpleExecutionSimulator())
    pipeline = ExecutionPipeline(registry)
    orders = _order_frame(symbols=["BTCUSDT"], sides=["BUY"], quantities=[1.0])
    with pytest.raises(ExecutionValidationError, match="not registered") as exc_info:
        pipeline.run(orders, manager=_MANAGER, simulator_name="missing")
    assert exc_info.value.error_code == "EXEC_REG_UNKNOWN"
    with pytest.raises(ExecutionValidationError, match="non-blank") as exc_info:
        pipeline.run(orders, manager="", simulator_name="simple")
    assert exc_info.value.error_code == "EXEC_PIPE_MANAGER_BLANK"


def test_pipeline_finalizes_simulator_output() -> None:
    """Pipeline reorders and casts simulator output to the merged schema."""
    orders = _order_frame(symbols=["BTCUSDT"], sides=["BUY"], quantities=[1.0])
    trade_like = SimpleExecutionSimulator().execute(orders, manager=_MANAGER)
    # Deliberately reorder columns before returning from the stub.
    reordered = trade_like.select(list(reversed(trade_like.columns)))
    stub = _RecordingSimulator(reordered)
    registry = ExecutionSimulatorRegistry()
    registry.register("stub", stub)
    pipeline = ExecutionPipeline(registry)
    result = pipeline.run(orders, manager=_MANAGER, simulator_name="stub")
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result.schema == MERGED_TRADE_SCHEMA
    assert len(stub.calls) == 1
    assert stub.calls[0][1] == _MANAGER
