"""Unit tests for CQROS ``BacktestingPipeline``."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from cqros.backtesting import (
    CANONICAL_COLUMN_ORDER,
    BacktestingPipeline,
    BacktestingRegistry,
    BacktestingStatus,
    BacktestingValidationError,
    SimpleBacktestingEngine,
)

_TIMEFRAME = "1h"
_MANAGER = "simple"
_POSITION_ID = "pos-00000001"
_OPEN_TIME = datetime(2024, 1, 1, tzinfo=UTC)


def _accounting_frame(
    *,
    symbol: str = "BTCUSDT",
    position_id: str = _POSITION_ID,
    position_status: str = "OPEN",
    cash: float = 10000.0,
    unrealized_pnl: float = 500.0,
) -> pl.DataFrame:
    """Build a minimal accounting frame for pipeline tests."""
    return pl.DataFrame(
        {
            "symbol": [symbol],
            "timeframe": [_TIMEFRAME],
            "open_time": [_OPEN_TIME],
            "cash": [cash],
            "position_value": [500.0],
            "realized_pnl": [0.0],
            "unrealized_pnl": [unrealized_pnl],
            "position_id": [position_id],
            "position_status": [position_status],
        }
    )


def _positions_frame(
    *,
    symbol: str = "BTCUSDT",
    position_id: str = _POSITION_ID,
    status: str = "OPEN",
) -> pl.DataFrame:
    """Build a minimal positions frame for pipeline tests."""
    return pl.DataFrame(
        {
            "symbol": [symbol],
            "timeframe": [_TIMEFRAME],
            "position_id": [position_id],
            "status": [status],
            "realized_pnl": [0.0],
            "opened_at": [_OPEN_TIME],
            "updated_at": [None],
            "closed_at": [None],
        }
    )


def _exit_engine_frame(
    *,
    symbol: str = "BTCUSDT",
    position_id: str = _POSITION_ID,
    exit_action: str = "HOLD",
) -> pl.DataFrame:
    """Build a minimal exit-engine frame for pipeline tests."""
    return pl.DataFrame(
        {
            "symbol": [symbol],
            "timeframe": [_TIMEFRAME],
            "open_time": [_OPEN_TIME],
            "position_id": [position_id],
            "exit_action": [exit_action],
        }
    )


def _build_registry(engine_name: str = "simple") -> BacktestingRegistry:
    """Build a registry with SimpleBacktestingEngine under engine_name."""
    registry = BacktestingRegistry()
    registry.register(engine_name, SimpleBacktestingEngine())
    return registry


def _run(
    pipeline: BacktestingPipeline,
    *,
    engine_name: str = "simple",
    manager: str = _MANAGER,
) -> pl.DataFrame:
    """Run the pipeline with default frames."""
    return pipeline.run(
        _accounting_frame(),
        _positions_frame(),
        _exit_engine_frame(),
        manager=manager,
        engine_name=engine_name,
    )


# ---------------------------------------------------------------------------
# Engine-name validation
# ---------------------------------------------------------------------------


def test_run_rejects_blank_engine_name() -> None:
    """Blank or whitespace engine names raise BT_PIPE_NAME_BLANK."""
    pipeline = BacktestingPipeline(_build_registry())
    for blank in ("", "   "):
        with pytest.raises(BacktestingValidationError) as exc_info:
            pipeline.run(
                _accounting_frame(),
                _positions_frame(),
                _exit_engine_frame(),
                manager=_MANAGER,
                engine_name=blank,
            )
        assert exc_info.value.error_code == "BT_PIPE_NAME_BLANK"


def test_run_rejects_unknown_engine_name() -> None:
    """Unregistered engine name causes BT_REG_UNKNOWN from registry lookup."""
    pipeline = BacktestingPipeline(_build_registry())
    with pytest.raises(BacktestingValidationError) as exc_info:
        _run(pipeline, engine_name="unknown-engine")
    assert exc_info.value.error_code == "BT_REG_UNKNOWN"


# ---------------------------------------------------------------------------
# Manager validation
# ---------------------------------------------------------------------------


def test_run_rejects_blank_manager() -> None:
    """Blank manager raises BT_PIPE_MANAGER_BLANK."""
    pipeline = BacktestingPipeline(_build_registry())
    for blank in ("", "   "):
        with pytest.raises(BacktestingValidationError) as exc_info:
            _run(pipeline, manager=blank)
        assert exc_info.value.error_code == "BT_PIPE_MANAGER_BLANK"


# ---------------------------------------------------------------------------
# Duplicate primary key rejection
# ---------------------------------------------------------------------------


def test_run_rejects_duplicate_primary_keys() -> None:
    """Duplicate primary keys in engine output raise BT_PIPE_DUPLICATE_KEYS."""

    class _DuplicatingEngine:
        """Test engine that returns two rows with identical primary keys."""

        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            return pl.DataFrame(
                {
                    "symbol": ["BTCUSDT", "BTCUSDT"],
                    "timeframe": [_TIMEFRAME, _TIMEFRAME],
                    "open_time": [_OPEN_TIME, _OPEN_TIME],
                    "manager": [_MANAGER, _MANAGER],
                    "equity": [10500.0, 10500.0],
                    "cash": [10000.0, 10000.0],
                    "position_value": [500.0, 500.0],
                    "realized_pnl": [0.0, 0.0],
                    "unrealized_pnl": [500.0, 500.0],
                    "total_pnl": [500.0, 500.0],
                    "drawdown": [0.0, 0.0],
                    "peak_equity": [10500.0, 10500.0],
                    "daily_return": [0.0, 0.0],
                    "cumulative_return": [0.0, 0.0],
                    "trade_count": [0, 0],
                    "winning_trades": [0, 0],
                    "losing_trades": [0, 0],
                    "win_rate": [0.0, 0.0],
                    "profit_factor": [None, None],
                    "sharpe_stub": [None, None],
                    "sortino_stub": [None, None],
                    "max_drawdown": [0.0, 0.0],
                    "status": [
                        BacktestingStatus.ACTIVE.value,
                        BacktestingStatus.FINISHED.value,
                    ],
                }
            )

    registry = BacktestingRegistry()
    registry.register("duplicating", _DuplicatingEngine())  # type: ignore[arg-type]
    pipeline = BacktestingPipeline(registry)

    with pytest.raises(BacktestingValidationError) as exc_info:
        pipeline.run(
            _accounting_frame(),
            _positions_frame(),
            _exit_engine_frame(),
            manager=_MANAGER,
            engine_name="duplicating",
        )
    assert exc_info.value.error_code == "BT_PIPE_DUPLICATE_KEYS"


# ---------------------------------------------------------------------------
# Invalid engine output types
# ---------------------------------------------------------------------------


def test_run_rejects_non_dataframe_engine_output() -> None:
    """Engine returning a non-DataFrame raises BT_PIPE_INVALID_OUTPUT."""

    class _BadEngine:
        def build(self, *args: object, **kwargs: object) -> object:
            return "not-a-dataframe"

    registry = BacktestingRegistry()
    registry.register("bad", _BadEngine())  # type: ignore[arg-type]
    pipeline = BacktestingPipeline(registry)

    with pytest.raises(BacktestingValidationError) as exc_info:
        pipeline.run(
            _accounting_frame(),
            _positions_frame(),
            _exit_engine_frame(),
            manager=_MANAGER,
            engine_name="bad",
        )
    assert exc_info.value.error_code == "BT_PIPE_INVALID_OUTPUT"


def test_run_rejects_empty_engine_output() -> None:
    """Engine returning an empty DataFrame raises BT_PIPE_OUTPUT_EMPTY."""

    class _EmptyEngine:
        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            return pl.DataFrame({"symbol": []})

    registry = BacktestingRegistry()
    registry.register("empty", _EmptyEngine())  # type: ignore[arg-type]
    pipeline = BacktestingPipeline(registry)

    with pytest.raises(BacktestingValidationError) as exc_info:
        pipeline.run(
            _accounting_frame(),
            _positions_frame(),
            _exit_engine_frame(),
            manager=_MANAGER,
            engine_name="empty",
        )
    assert exc_info.value.error_code == "BT_PIPE_OUTPUT_EMPTY"


def test_run_rejects_missing_schema_columns_in_engine_output() -> None:
    """Engine returning a DataFrame missing required columns raises BT_PIPE_MISSING_COLUMNS."""

    class _IncompleteEngine:
        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            return pl.DataFrame({"symbol": ["BTCUSDT"], "equity": [10000.0]})

    registry = BacktestingRegistry()
    registry.register("incomplete", _IncompleteEngine())  # type: ignore[arg-type]
    pipeline = BacktestingPipeline(registry)

    with pytest.raises(BacktestingValidationError) as exc_info:
        pipeline.run(
            _accounting_frame(),
            _positions_frame(),
            _exit_engine_frame(),
            manager=_MANAGER,
            engine_name="incomplete",
        )
    assert exc_info.value.error_code == "BT_PIPE_MISSING_COLUMNS"


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_run_produces_canonical_performance_row() -> None:
    """Pipeline with default inputs produces one canonical performance output row."""
    pipeline = BacktestingPipeline(_build_registry())
    result = _run(pipeline)
    assert result.height == 1
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result["status"].to_list() == [BacktestingStatus.FINISHED.value]
    assert result["equity"].to_list()[0] == pytest.approx(10500.0)


def test_run_defaults_to_simple_engine() -> None:
    """Pipeline defaults to 'simple' engine when engine_name is omitted."""
    pipeline = BacktestingPipeline(_build_registry())
    result = pipeline.run(
        _accounting_frame(),
        _positions_frame(),
        _exit_engine_frame(),
        manager=_MANAGER,
    )
    assert result.height == 1
    assert result["manager"].to_list() == [_MANAGER]


def test_run_does_not_mutate_input_frames() -> None:
    """Pipeline run must not mutate caller-supplied input frames."""
    from polars.testing import assert_frame_equal

    pipeline = BacktestingPipeline(_build_registry())
    accounting = _accounting_frame()
    positions = _positions_frame()
    exit_engine = _exit_engine_frame()

    accounting_before = accounting.clone()
    positions_before = positions.clone()
    exit_before = exit_engine.clone()

    pipeline.run(accounting, positions, exit_engine, manager=_MANAGER)

    assert_frame_equal(accounting, accounting_before)
    assert_frame_equal(positions, positions_before)
    assert_frame_equal(exit_engine, exit_before)
