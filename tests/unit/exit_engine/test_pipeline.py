"""Unit tests for CQROS ``ExitEnginePipeline``."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from cqros.exit_engine import (
    CANONICAL_COLUMN_ORDER,
    ExitAction,
    ExitEnginePipeline,
    ExitEngineRegistry,
    ExitEngineValidationError,
    ExitReason,
    SimpleExitEngine,
)

_TIMEFRAME = "1h"
_MANAGER = "simple"
_POSITION_ID = "pos-00000001"
_OPEN_TIME = datetime(2024, 1, 1, tzinfo=UTC)


def _positions_frame(
    *,
    symbol: str = "BTCUSDT",
    position_id: str = _POSITION_ID,
    side: str = "LONG",
) -> pl.DataFrame:
    """Build a minimal position frame for pipeline tests."""
    return pl.DataFrame(
        {
            "symbol": [symbol],
            "timeframe": [_TIMEFRAME],
            "position_id": [position_id],
            "side": [side],
        }
    )


def _accounting_frame(
    *,
    symbol: str = "BTCUSDT",
    position_id: str = _POSITION_ID,
    position_status: str = "OPEN",
    quantity: float = 1.0,
    entry_price: float = 100.0,
) -> pl.DataFrame:
    """Build a minimal accounting frame for pipeline tests."""
    return pl.DataFrame(
        {
            "symbol": [symbol],
            "timeframe": [_TIMEFRAME],
            "open_time": [_OPEN_TIME],
            "position_id": [position_id],
            "position_status": [position_status],
            "quantity": [quantity],
            "average_entry_price": [entry_price],
        }
    )


def _portfolio_risk_frame(
    *,
    symbol: str = "BTCUSDT",
    position_id: str = _POSITION_ID,
    risk_state: str = "NORMAL",
    shutdown_reason: str | None = None,
) -> pl.DataFrame:
    """Build a minimal portfolio-risk frame for pipeline tests."""
    return pl.DataFrame(
        {
            "symbol": [symbol],
            "timeframe": [_TIMEFRAME],
            "open_time": [_OPEN_TIME],
            "position_id": [position_id],
            "portfolio_risk_state": [risk_state],
            "shutdown_reason": [shutdown_reason],
            "cooldown_until": [None],
        }
    )


def _trade_management_frame(
    *,
    symbol: str = "BTCUSDT",
    position_id: str = _POSITION_ID,
    current_price: float = 102.0,
    management_action: str = "NONE",
    action_reason: str | None = None,
) -> pl.DataFrame:
    """Build a minimal trade-management frame for pipeline tests."""
    return pl.DataFrame(
        {
            "symbol": [symbol],
            "timeframe": [_TIMEFRAME],
            "open_time": [_OPEN_TIME],
            "position_id": [position_id],
            "current_price": [current_price],
            "management_action": [management_action],
            "action_reason": [action_reason],
        }
    )


def _pyramiding_frame(
    *,
    symbol: str = "BTCUSDT",
    position_id: str = _POSITION_ID,
    reason: str = "INSUFFICIENT_PROFIT",
) -> pl.DataFrame:
    """Build a minimal pyramiding frame for pipeline tests."""
    return pl.DataFrame(
        {
            "symbol": [symbol],
            "timeframe": [_TIMEFRAME],
            "open_time": [_OPEN_TIME],
            "position_id": [position_id],
            "reason": [reason],
        }
    )


def _build_registry(engine_name: str = "simple") -> ExitEngineRegistry:
    """Build a registry with SimpleExitEngine under engine_name."""
    registry = ExitEngineRegistry()
    registry.register(engine_name, SimpleExitEngine())
    return registry


def _run(
    pipeline: ExitEnginePipeline,
    *,
    engine_name: str = "simple",
    manager: str = _MANAGER,
    current_price: float = 102.0,
) -> pl.DataFrame:
    """Run the pipeline with default frames."""
    return pipeline.run(
        _positions_frame(),
        _accounting_frame(),
        _portfolio_risk_frame(),
        _trade_management_frame(current_price=current_price),
        _pyramiding_frame(),
        manager=manager,
        engine_name=engine_name,
    )


# ---------------------------------------------------------------------------
# Engine-name validation
# ---------------------------------------------------------------------------


def test_run_rejects_blank_engine_name() -> None:
    """Blank or whitespace engine names raise EXIT_PIPE_NAME_BLANK."""
    pipeline = ExitEnginePipeline(_build_registry())
    for blank in ("", "   "):
        with pytest.raises(ExitEngineValidationError) as exc_info:
            pipeline.run(
                _positions_frame(),
                _accounting_frame(),
                _portfolio_risk_frame(),
                _trade_management_frame(),
                _pyramiding_frame(),
                manager=_MANAGER,
                engine_name=blank,
            )
        assert exc_info.value.error_code == "EXIT_PIPE_NAME_BLANK"


def test_run_rejects_unknown_engine_name() -> None:
    """Unregistered engine name causes EXIT_REG_UNKNOWN from registry lookup."""
    pipeline = ExitEnginePipeline(_build_registry())
    with pytest.raises(ExitEngineValidationError) as exc_info:
        _run(pipeline, engine_name="unknown-engine")
    assert exc_info.value.error_code == "EXIT_REG_UNKNOWN"


# ---------------------------------------------------------------------------
# Manager validation
# ---------------------------------------------------------------------------


def test_run_rejects_blank_manager() -> None:
    """Blank manager raises EXIT_PIPE_MANAGER_BLANK."""
    pipeline = ExitEnginePipeline(_build_registry())
    for blank in ("", "   "):
        with pytest.raises(ExitEngineValidationError) as exc_info:
            _run(pipeline, manager=blank)
        assert exc_info.value.error_code == "EXIT_PIPE_MANAGER_BLANK"


# ---------------------------------------------------------------------------
# Duplicate primary key rejection
# ---------------------------------------------------------------------------


def test_run_rejects_duplicate_primary_keys() -> None:
    """Duplicate primary keys in engine output raise EXIT_PIPE_DUPLICATE_KEYS."""

    class _DuplicatingEngine:
        """Test engine that returns two rows with identical primary keys."""

        def evaluate(self, *args: object, **kwargs: object) -> pl.DataFrame:
            row = pl.DataFrame(
                {
                    "symbol": ["BTCUSDT", "BTCUSDT"],
                    "timeframe": [_TIMEFRAME, _TIMEFRAME],
                    "open_time": [_OPEN_TIME, _OPEN_TIME],
                    "position_id": [_POSITION_ID, _POSITION_ID],
                    "manager": [_MANAGER, _MANAGER],
                    "entry_price": [100.0, 100.0],
                    "current_price": [102.0, 102.0],
                    "quantity": [1.0, 1.0],
                    "risk_reward_ratio": [0.4, 0.4],
                    "risk_state": ["NORMAL", "NORMAL"],
                    "trade_state": ["NONE", "NONE"],
                    "pyramid_state": ["NONE", "NONE"],
                    "exit_action": ["HOLD", "HOLD"],
                    "exit_reason": ["NONE", "NONE"],
                    "recommended_quantity": [0.0, 0.0],
                    "recommended_percent": [0.0, 0.0],
                    "priority": [0, 0],
                    "created_at": [_OPEN_TIME, _OPEN_TIME],
                }
            )
            return row

    registry = ExitEngineRegistry()
    registry.register("duplicating", _DuplicatingEngine())  # type: ignore[arg-type]
    pipeline = ExitEnginePipeline(registry)

    with pytest.raises(ExitEngineValidationError) as exc_info:
        pipeline.run(
            _positions_frame(),
            _accounting_frame(),
            _portfolio_risk_frame(),
            _trade_management_frame(),
            _pyramiding_frame(),
            manager=_MANAGER,
            engine_name="duplicating",
        )
    assert exc_info.value.error_code == "EXIT_PIPE_DUPLICATE_KEYS"


# ---------------------------------------------------------------------------
# Invalid engine output types
# ---------------------------------------------------------------------------


def test_run_rejects_non_dataframe_engine_output() -> None:
    """Engine returning a non-DataFrame raises EXIT_PIPE_INVALID_OUTPUT."""

    class _BadEngine:
        def evaluate(self, *args: object, **kwargs: object) -> object:
            return "not-a-dataframe"

    registry = ExitEngineRegistry()
    registry.register("bad", _BadEngine())  # type: ignore[arg-type]
    pipeline = ExitEnginePipeline(registry)

    with pytest.raises(ExitEngineValidationError) as exc_info:
        pipeline.run(
            _positions_frame(),
            _accounting_frame(),
            _portfolio_risk_frame(),
            _trade_management_frame(),
            _pyramiding_frame(),
            manager=_MANAGER,
            engine_name="bad",
        )
    assert exc_info.value.error_code == "EXIT_PIPE_INVALID_OUTPUT"


def test_run_rejects_missing_schema_columns_in_engine_output() -> None:
    """Engine returning a DataFrame missing required columns raises EXIT_PIPE_MISSING_COLUMNS."""

    class _IncompleteEngine:
        def evaluate(self, *args: object, **kwargs: object) -> pl.DataFrame:
            return pl.DataFrame({"symbol": ["BTCUSDT"], "exit_action": ["HOLD"]})

    registry = ExitEngineRegistry()
    registry.register("incomplete", _IncompleteEngine())  # type: ignore[arg-type]
    pipeline = ExitEnginePipeline(registry)

    with pytest.raises(ExitEngineValidationError) as exc_info:
        pipeline.run(
            _positions_frame(),
            _accounting_frame(),
            _portfolio_risk_frame(),
            _trade_management_frame(),
            _pyramiding_frame(),
            manager=_MANAGER,
            engine_name="incomplete",
        )
    assert exc_info.value.error_code == "EXIT_PIPE_MISSING_COLUMNS"


# ---------------------------------------------------------------------------
# Happy path: single HOLD row
# ---------------------------------------------------------------------------


def test_run_produces_canonical_hold_row() -> None:
    """Pipeline with default HOLD inputs produces one canonical HOLD output row."""
    pipeline = ExitEnginePipeline(_build_registry())
    result = _run(pipeline, current_price=102.0)
    assert result.height == 1
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result["exit_action"].to_list() == [ExitAction.HOLD.value]
    assert result["exit_reason"].to_list() == [ExitReason.NONE.value]


# ---------------------------------------------------------------------------
# Happy path: empty output for closed positions
# ---------------------------------------------------------------------------


def test_run_returns_empty_frame_for_closed_positions() -> None:
    """Pipeline returns empty frame with canonical schema when no OPEN positions."""
    pipeline = ExitEnginePipeline(_build_registry())
    result = pipeline.run(
        _positions_frame(),
        _accounting_frame(position_status="CLOSED"),
        _portfolio_risk_frame(),
        _trade_management_frame(),
        _pyramiding_frame(),
        manager=_MANAGER,
        engine_name="simple",
    )
    assert result.height == 0
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER


# ---------------------------------------------------------------------------
# Default engine name
# ---------------------------------------------------------------------------


def test_run_defaults_to_simple_engine() -> None:
    """Pipeline defaults to 'simple' engine when engine_name is omitted."""
    pipeline = ExitEnginePipeline(_build_registry())
    result = pipeline.run(
        _positions_frame(),
        _accounting_frame(),
        _portfolio_risk_frame(),
        _trade_management_frame(current_price=102.0),
        _pyramiding_frame(),
        manager=_MANAGER,
    )
    assert result.height == 1
    assert result["exit_action"].to_list() == [ExitAction.HOLD.value]
