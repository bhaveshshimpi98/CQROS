"""Unit tests for CQROS ``TradeManagementPipeline``."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.trade_management import (
    CANONICAL_COLUMN_ORDER,
    MERGED_TRADE_MANAGEMENT_SCHEMA,
    SimpleTradeManagementManager,
    TradeManagementManagerRegistry,
    TradeManagementPipeline,
    TradeManagementValidationError,
)
from cqros.trade_management.pipeline import (
    TradeManagementPipeline as TradeManagementPipelineDirect,
)

_TIMEFRAME = "1h"
_MODEL_NAME = "alpha-lgbm"
_MODEL_VERSION = "1.0.0"
_POLICY = "fixed_risk"
_OPTIMIZER = "equal_weight"
_MANAGER = "simple"


def _open_time(index: int) -> datetime:
    """Build a deterministic UTC open_time timestamp for row ``index``."""
    return datetime(2024, 1, 1, hour=index, tzinfo=UTC)


def _position_frame() -> pl.DataFrame:
    """Build a minimal position identity frame."""
    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timeframe": [_TIMEFRAME],
            "position_id": ["pos-00000001"],
        }
    )


def _accounting_frame() -> pl.DataFrame:
    """Build an accounting-shaped frame for pipeline tests."""
    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timeframe": [_TIMEFRAME],
            "open_time": [_open_time(0)],
            "position_id": ["pos-00000001"],
            "position_status": ["OPEN"],
            "quantity": [1.0],
            "average_entry_price": [100.0],
            "unrealized_pnl": [0.0],
            "model_name": [_MODEL_NAME],
            "model_version": [_MODEL_VERSION],
            "optimizer": [_OPTIMIZER],
            "policy": [_POLICY],
        }
    )


def _portfolio_risk_frame() -> pl.DataFrame:
    """Build a portfolio-risk-shaped frame for pipeline tests."""
    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timeframe": [_TIMEFRAME],
            "open_time": [_open_time(0)],
            "position_id": ["pos-00000001"],
            "portfolio_risk_state": ["NORMAL"],
            "allow_new_entries": [True],
        }
    )


def _market_prices_frame(*, price: float = 104.0) -> pl.DataFrame:
    """Build a market-price frame for pipeline tests."""
    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timeframe": [_TIMEFRAME],
            "open_time": [_open_time(0)],
            "price": [price],
        }
    )


def _trade_management_like_frame() -> pl.DataFrame:
    """Return a canonical trade-management-shaped frame via the simple manager."""
    return SimpleTradeManagementManager().evaluate(
        _position_frame(),
        _accounting_frame(),
        _portfolio_risk_frame(),
        _market_prices_frame(),
        manager=_MANAGER,
    )


class _RecordingManager:
    """Manager stub that records evaluate calls and returns a fixed frame."""

    def __init__(self, frame: pl.DataFrame) -> None:
        self.frame = frame
        self.calls: list[tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, str]] = []

    def evaluate(
        self,
        positions: pl.DataFrame,
        accounting: pl.DataFrame,
        portfolio_risk: pl.DataFrame,
        market_prices: pl.DataFrame,
        *,
        manager: str,
    ) -> pl.DataFrame:
        self.calls.append((positions, accounting, portfolio_risk, market_prices, manager))
        return self.frame


def test_trade_management_pipeline_is_exported_from_package() -> None:
    """Package export matches the pipeline module by identity."""
    assert TradeManagementPipeline is TradeManagementPipelineDirect


def test_pipeline_resolves_simple_manager_and_finalizes_schema() -> None:
    """Pipeline resolves the simple manager, stamps manager, finalizes schema."""
    registry = TradeManagementManagerRegistry()
    registry.register("simple", SimpleTradeManagementManager())
    pipeline = TradeManagementPipeline(registry)
    positions = _position_frame()
    accounting = _accounting_frame()
    portfolio_risk = _portfolio_risk_frame()
    market_prices = _market_prices_frame()
    original_positions = positions.clone()
    original_accounting = accounting.clone()
    result = pipeline.run(
        positions,
        accounting,
        portfolio_risk,
        market_prices,
        manager=_MANAGER,
        trade_manager_name="simple",
    )
    assert_frame_equal(positions, original_positions)
    assert_frame_equal(accounting, original_accounting)
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result.schema == MERGED_TRADE_MANAGEMENT_SCHEMA
    assert result["manager"].to_list() == [_MANAGER]
    assert result.height == 1


def test_pipeline_default_trade_manager_name_is_simple() -> None:
    """The pipeline defaults to the ``simple`` trade-manager name."""
    registry = TradeManagementManagerRegistry()
    registry.register("simple", SimpleTradeManagementManager())
    pipeline = TradeManagementPipeline(registry)
    result = pipeline.run(
        _position_frame(),
        _accounting_frame(),
        _portfolio_risk_frame(),
        _market_prices_frame(),
        manager=_MANAGER,
    )
    assert result.height == 1


def test_pipeline_rejects_blank_names() -> None:
    """Blank trade-manager names and blank managers raise validation errors."""
    registry = TradeManagementManagerRegistry()
    registry.register("simple", SimpleTradeManagementManager())
    pipeline = TradeManagementPipeline(registry)
    with pytest.raises(TradeManagementValidationError, match="non-blank") as exc_info:
        pipeline.run(
            _position_frame(),
            _accounting_frame(),
            _portfolio_risk_frame(),
            _market_prices_frame(),
            manager=_MANAGER,
            trade_manager_name="",
        )
    assert exc_info.value.error_code == "TME_PIPE_NAME_BLANK"
    with pytest.raises(TradeManagementValidationError, match="non-blank") as exc_info:
        pipeline.run(
            _position_frame(),
            _accounting_frame(),
            _portfolio_risk_frame(),
            _market_prices_frame(),
            manager="",
            trade_manager_name="simple",
        )
    assert exc_info.value.error_code == "TME_PIPE_MANAGER_BLANK"


def test_pipeline_rejects_unknown_trade_manager() -> None:
    """Unknown trade-manager names raise validation errors."""
    registry = TradeManagementManagerRegistry()
    registry.register("simple", SimpleTradeManagementManager())
    pipeline = TradeManagementPipeline(registry)
    with pytest.raises(TradeManagementValidationError, match="not registered") as exc_info:
        pipeline.run(
            _position_frame(),
            _accounting_frame(),
            _portfolio_risk_frame(),
            _market_prices_frame(),
            manager=_MANAGER,
            trade_manager_name="missing",
        )
    assert exc_info.value.error_code == "TME_REG_UNKNOWN"


def test_pipeline_finalizes_manager_output() -> None:
    """Pipeline reorders and casts manager output to the merged schema."""
    trade_like = _trade_management_like_frame()
    reordered = trade_like.select(list(reversed(trade_like.columns)))
    stub = _RecordingManager(reordered)
    registry = TradeManagementManagerRegistry()
    registry.register("stub", stub)
    pipeline = TradeManagementPipeline(registry)
    result = pipeline.run(
        _position_frame(),
        _accounting_frame(),
        _portfolio_risk_frame(),
        _market_prices_frame(),
        manager=_MANAGER,
        trade_manager_name="stub",
    )
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result.schema == MERGED_TRADE_MANAGEMENT_SCHEMA
    assert len(stub.calls) == 1
    assert stub.calls[0][4] == _MANAGER


def test_pipeline_rejects_duplicate_primary_keys() -> None:
    """Duplicate primary keys in manager output raise validation errors."""
    base = _trade_management_like_frame()
    duplicate = pl.concat([base, base])
    stub = _RecordingManager(duplicate)
    registry = TradeManagementManagerRegistry()
    registry.register("stub", stub)
    pipeline = TradeManagementPipeline(registry)
    with pytest.raises(TradeManagementValidationError, match="duplicate primary keys") as exc_info:
        pipeline.run(
            _position_frame(),
            _accounting_frame(),
            _portfolio_risk_frame(),
            _market_prices_frame(),
            manager=_MANAGER,
            trade_manager_name="stub",
        )
    assert exc_info.value.error_code == "TME_PIPE_DUPLICATE_KEYS"


def test_pipeline_rejects_missing_columns() -> None:
    """Missing required trade-management columns on manager output are rejected."""
    incomplete = _trade_management_like_frame().drop("allow_pyramid")
    stub = _RecordingManager(incomplete)
    registry = TradeManagementManagerRegistry()
    registry.register("stub", stub)
    pipeline = TradeManagementPipeline(registry)
    with pytest.raises(
        TradeManagementValidationError, match="missing required columns"
    ) as exc_info:
        pipeline.run(
            _position_frame(),
            _accounting_frame(),
            _portfolio_risk_frame(),
            _market_prices_frame(),
            manager=_MANAGER,
            trade_manager_name="stub",
        )
    assert exc_info.value.error_code == "TME_PIPE_MISSING_COLUMNS"
    assert "allow_pyramid" in exc_info.value.details["missing_columns"]


def test_pipeline_preserves_input_immutability() -> None:
    """Pipeline must not mutate caller-supplied input frames."""
    registry = TradeManagementManagerRegistry()
    registry.register("simple", SimpleTradeManagementManager())
    pipeline = TradeManagementPipeline(registry)
    positions = _position_frame()
    accounting = _accounting_frame()
    portfolio_risk = _portfolio_risk_frame()
    market_prices = _market_prices_frame()
    positions_before = positions.clone()
    accounting_before = accounting.clone()
    risk_before = portfolio_risk.clone()
    prices_before = market_prices.clone()
    pipeline.run(
        positions,
        accounting,
        portfolio_risk,
        market_prices,
        manager=_MANAGER,
    )
    assert_frame_equal(positions, positions_before)
    assert_frame_equal(accounting, accounting_before)
    assert_frame_equal(portfolio_risk, risk_before)
    assert_frame_equal(market_prices, prices_before)
