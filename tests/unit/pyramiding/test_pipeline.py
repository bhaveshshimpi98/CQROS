"""Unit tests for CQROS ``PyramidingPipeline``."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.pyramiding import (
    CANONICAL_COLUMN_ORDER,
    MERGED_PYRAMIDING_SCHEMA,
    PyramidingPipeline,
    PyramidingRegistry,
    PyramidingValidationError,
    SimplePyramidingEngine,
)
from cqros.pyramiding.pipeline import PyramidingPipeline as PyramidingPipelineDirect

_TIMEFRAME = "1h"
_MANAGER = "simple"
_POSITION_ID = "pos-00000001"


def _open_time(index: int) -> datetime:
    """Build a deterministic UTC open_time for row ``index``."""
    return datetime(2024, 1, 1, hour=index, tzinfo=UTC)


def _positions_frame() -> pl.DataFrame:
    """Build a minimal positions frame for pipeline tests."""
    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timeframe": [_TIMEFRAME],
            "position_id": [_POSITION_ID],
            "side": ["LONG"],
        }
    )


def _accounting_frame() -> pl.DataFrame:
    """Build an accounting frame for pipeline tests."""
    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timeframe": [_TIMEFRAME],
            "open_time": [_open_time(0)],
            "position_id": [_POSITION_ID],
            "position_status": ["OPEN"],
            "quantity": [1.0],
            "average_entry_price": [100.0],
        }
    )


def _portfolio_risk_frame() -> pl.DataFrame:
    """Build a portfolio-risk frame for pipeline tests."""
    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timeframe": [_TIMEFRAME],
            "open_time": [_open_time(0)],
            "position_id": [_POSITION_ID],
            "portfolio_risk_state": ["NORMAL"],
            "shutdown_reason": [None],
            "cooldown_until": [None],
        }
    )


def _trade_management_frame() -> pl.DataFrame:
    """Build a trade-management frame for pipeline tests."""
    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timeframe": [_TIMEFRAME],
            "open_time": [_open_time(0)],
            "position_id": [_POSITION_ID],
            "management_action": ["NONE"],
            "action_reason": [None],
        }
    )


def _market_prices_frame(*, price: float = 105.0) -> pl.DataFrame:
    """Build a market-price frame for pipeline tests."""
    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timeframe": [_TIMEFRAME],
            "open_time": [_open_time(0)],
            "price": [price],
            "high": [price],
        }
    )


def _pyramiding_like_frame() -> pl.DataFrame:
    """Return a canonical pyramiding frame via the simple engine."""
    return SimplePyramidingEngine().evaluate(
        _positions_frame(),
        _accounting_frame(),
        _portfolio_risk_frame(),
        _trade_management_frame(),
        _market_prices_frame(),
        manager=_MANAGER,
    )


class _RecordingEngine:
    """Engine stub that records evaluate calls and returns a fixed frame."""

    def __init__(self, frame: pl.DataFrame) -> None:
        self.frame = frame
        self.calls: list[
            tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, str]
        ] = []

    def evaluate(
        self,
        positions: pl.DataFrame,
        accounting: pl.DataFrame,
        portfolio_risk: pl.DataFrame,
        trade_management: pl.DataFrame,
        market_prices: pl.DataFrame,
        *,
        manager: str,
    ) -> pl.DataFrame:
        self.calls.append(
            (positions, accounting, portfolio_risk, trade_management, market_prices, manager)
        )
        return self.frame


def test_pyramiding_pipeline_is_exported_from_package() -> None:
    """Package export matches the pipeline module by identity."""
    assert PyramidingPipeline is PyramidingPipelineDirect


def test_pipeline_resolves_simple_engine_and_finalizes_schema() -> None:
    """Pipeline resolves the simple engine, stamps manager, finalizes schema."""
    registry = PyramidingRegistry()
    registry.register("simple", SimplePyramidingEngine())
    pipeline = PyramidingPipeline(registry)

    positions = _positions_frame()
    accounting = _accounting_frame()
    portfolio_risk = _portfolio_risk_frame()
    trade_management = _trade_management_frame()
    market_prices = _market_prices_frame()
    original_positions = positions.clone()
    original_accounting = accounting.clone()

    result = pipeline.run(
        positions,
        accounting,
        portfolio_risk,
        trade_management,
        market_prices,
        manager=_MANAGER,
        engine_name="simple",
    )

    assert_frame_equal(positions, original_positions)
    assert_frame_equal(accounting, original_accounting)
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result.schema == MERGED_PYRAMIDING_SCHEMA
    assert result["manager"].to_list() == [_MANAGER]
    assert result.height == 1


def test_pipeline_default_engine_name_is_simple() -> None:
    """The pipeline defaults to the ``simple`` engine name."""
    registry = PyramidingRegistry()
    registry.register("simple", SimplePyramidingEngine())
    pipeline = PyramidingPipeline(registry)
    result = pipeline.run(
        _positions_frame(),
        _accounting_frame(),
        _portfolio_risk_frame(),
        _trade_management_frame(),
        _market_prices_frame(),
        manager=_MANAGER,
    )
    assert result.height == 1


def test_pipeline_rejects_blank_engine_name() -> None:
    """Blank engine_name raises PyramidingValidationError."""
    registry = PyramidingRegistry()
    registry.register("simple", SimplePyramidingEngine())
    pipeline = PyramidingPipeline(registry)
    with pytest.raises(PyramidingValidationError, match="non-blank") as exc_info:
        pipeline.run(
            _positions_frame(),
            _accounting_frame(),
            _portfolio_risk_frame(),
            _trade_management_frame(),
            _market_prices_frame(),
            manager=_MANAGER,
            engine_name="",
        )
    assert exc_info.value.error_code == "PYR_PIPE_NAME_BLANK"


def test_pipeline_rejects_blank_manager() -> None:
    """Blank manager raises PyramidingValidationError."""
    registry = PyramidingRegistry()
    registry.register("simple", SimplePyramidingEngine())
    pipeline = PyramidingPipeline(registry)
    with pytest.raises(PyramidingValidationError, match="non-blank") as exc_info:
        pipeline.run(
            _positions_frame(),
            _accounting_frame(),
            _portfolio_risk_frame(),
            _trade_management_frame(),
            _market_prices_frame(),
            manager="",
            engine_name="simple",
        )
    assert exc_info.value.error_code == "PYR_PIPE_MANAGER_BLANK"


def test_pipeline_rejects_unknown_engine() -> None:
    """Unknown engine names raise PyramidingValidationError."""
    registry = PyramidingRegistry()
    registry.register("simple", SimplePyramidingEngine())
    pipeline = PyramidingPipeline(registry)
    with pytest.raises(PyramidingValidationError, match="not registered") as exc_info:
        pipeline.run(
            _positions_frame(),
            _accounting_frame(),
            _portfolio_risk_frame(),
            _trade_management_frame(),
            _market_prices_frame(),
            manager=_MANAGER,
            engine_name="missing",
        )
    assert exc_info.value.error_code == "PYR_REG_UNKNOWN"


def test_pipeline_finalizes_engine_output_columns_and_schema() -> None:
    """Pipeline reorders and casts engine output to the canonical merged schema."""
    pyr_like = _pyramiding_like_frame()
    reordered = pyr_like.select(list(reversed(pyr_like.columns)))
    stub = _RecordingEngine(reordered)
    registry = PyramidingRegistry()
    registry.register("stub", stub)
    pipeline = PyramidingPipeline(registry)
    result = pipeline.run(
        _positions_frame(),
        _accounting_frame(),
        _portfolio_risk_frame(),
        _trade_management_frame(),
        _market_prices_frame(),
        manager=_MANAGER,
        engine_name="stub",
    )
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result.schema == MERGED_PYRAMIDING_SCHEMA
    assert len(stub.calls) == 1
    assert stub.calls[0][5] == _MANAGER


def test_pipeline_rejects_duplicate_primary_keys() -> None:
    """Duplicate primary keys in engine output raise PyramidingValidationError."""
    base = _pyramiding_like_frame()
    duplicate = pl.concat([base, base])
    stub = _RecordingEngine(duplicate)
    registry = PyramidingRegistry()
    registry.register("stub", stub)
    pipeline = PyramidingPipeline(registry)
    with pytest.raises(PyramidingValidationError, match="duplicate primary keys") as exc_info:
        pipeline.run(
            _positions_frame(),
            _accounting_frame(),
            _portfolio_risk_frame(),
            _trade_management_frame(),
            _market_prices_frame(),
            manager=_MANAGER,
            engine_name="stub",
        )
    assert exc_info.value.error_code == "PYR_PIPE_DUPLICATE_KEYS"


def test_pipeline_rejects_missing_schema_columns() -> None:
    """Missing required columns on engine output raise PyramidingValidationError."""
    incomplete = _pyramiding_like_frame().drop("allow_pyramid")
    stub = _RecordingEngine(incomplete)
    registry = PyramidingRegistry()
    registry.register("stub", stub)
    pipeline = PyramidingPipeline(registry)
    with pytest.raises(PyramidingValidationError, match="missing required columns") as exc_info:
        pipeline.run(
            _positions_frame(),
            _accounting_frame(),
            _portfolio_risk_frame(),
            _trade_management_frame(),
            _market_prices_frame(),
            manager=_MANAGER,
            engine_name="stub",
        )
    assert exc_info.value.error_code == "PYR_PIPE_MISSING_COLUMNS"
    assert "allow_pyramid" in exc_info.value.details["missing_columns"]


def test_pipeline_preserves_input_immutability() -> None:
    """Pipeline must not mutate caller-supplied input frames."""
    registry = PyramidingRegistry()
    registry.register("simple", SimplePyramidingEngine())
    pipeline = PyramidingPipeline(registry)

    positions = _positions_frame()
    accounting = _accounting_frame()
    portfolio_risk = _portfolio_risk_frame()
    trade_management = _trade_management_frame()
    market_prices = _market_prices_frame()

    positions_before = positions.clone()
    accounting_before = accounting.clone()
    risk_before = portfolio_risk.clone()
    tm_before = trade_management.clone()
    prices_before = market_prices.clone()

    pipeline.run(
        positions,
        accounting,
        portfolio_risk,
        trade_management,
        market_prices,
        manager=_MANAGER,
    )

    assert_frame_equal(positions, positions_before)
    assert_frame_equal(accounting, accounting_before)
    assert_frame_equal(portfolio_risk, risk_before)
    assert_frame_equal(trade_management, tm_before)
    assert_frame_equal(market_prices, prices_before)


def test_canonical_column_order_and_merged_schema_are_aligned() -> None:
    """CANONICAL_COLUMN_ORDER length matches MERGED_PYRAMIDING_SCHEMA field count."""
    assert len(CANONICAL_COLUMN_ORDER) == len(MERGED_PYRAMIDING_SCHEMA.names())
    assert list(CANONICAL_COLUMN_ORDER) == MERGED_PYRAMIDING_SCHEMA.names()
