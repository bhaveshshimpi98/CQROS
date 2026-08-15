"""Unit tests for CQROS position-generation CLI."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

import cqros.cli.generate_positions as generate_positions_module
from cqros.cli.generate_positions import (
    DiscoveredWorkItem,
    PositionGenerationOptions,
    PositionGenerationSummary,
    build_default_engine,
    build_engine_registry,
    build_options,
    build_parser,
    build_position_pipeline,
    discover_work,
    format_summary,
    run_generation,
)
from cqros.config.models import ResearchConfig
from cqros.core.constants import (
    DEFAULT_STORAGE_ROOT,
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
    STORAGE_DIR_EXECUTIONS,
    STORAGE_DIR_POSITIONS,
)
from cqros.core.exceptions import ValidationError
from cqros.execution import CANONICAL_COLUMN_ORDER as TRADE_COLUMNS
from cqros.execution import COLUMN_DTYPES as TRADE_DTYPES
from cqros.execution import ExecutionStatus, TradeRepository
from cqros.positions import (
    AverageCostPositionEngine,
    PositionEngineRegistry,
    PositionPipeline,
    PositionRepository,
)
from cqros.storage import ParquetStore, StorageLayout

_MANAGER = "simple"
_ENGINE = "average_cost"
_MODEL = "alpha-lgbm"
_VERSION = "1.0.0"
_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1h"
_YEAR = 2026


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def _options(
    *,
    storage_root: Path,
    manager: str = _MANAGER,
    engine: str = _ENGINE,
    model: str | None = _MODEL,
    version: str | None = _VERSION,
    symbols: tuple[str, ...] | None = None,
    timeframes: tuple[str, ...] | None = None,
    years: tuple[int, ...] | None = None,
    overwrite: bool = False,
    workers: int = ResearchConfig().worker_count,
    verbose: bool = False,
    debug: bool = False,
) -> PositionGenerationOptions:
    """Build options for tests against a temporary storage root."""
    return PositionGenerationOptions(
        storage_root=storage_root,
        manager=manager,
        engine=engine,
        model=model,
        version=version,
        symbols=symbols,
        timeframes=timeframes,
        years=years,
        overwrite=overwrite,
        workers=workers,
        verbose=verbose,
        debug=debug,
    )


def _trade_frame() -> pl.DataFrame:
    """Return a canonical executed-trade frame."""
    open_time = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
    return pl.DataFrame(
        {
            "symbol": [_SYMBOL],
            "timeframe": [_TIMEFRAME],
            "open_time": [open_time],
            "model_name": [_MODEL],
            "model_version": [_VERSION],
            "optimizer": ["equal_weight"],
            "policy": ["fixed_risk"],
            "manager": [_MANAGER],
            "signal": ["BUY"],
            "side": ["BUY"],
            "order_type": ["MARKET"],
            "requested_quantity": [1.0],
            "executed_quantity": [1.0],
            "requested_price": [100.0],
            "executed_price": [100.0],
            "fees": [0.0],
            "slippage": [0.0],
            "status": [ExecutionStatus.FILLED.value],
            "execution_time": [open_time],
        },
        schema=dict(TRADE_DTYPES),
    ).select(list(TRADE_COLUMNS))


def test_build_parser_requires_manager() -> None:
    """Parser requires --manager and defaults engine to average_cost."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(["--manager", "simple"])
    assert args.manager == "simple"
    assert args.engine == "average_cost"


def test_build_options_validates_workers_and_manager(tmp_path: Path) -> None:
    """build_options rejects non-positive workers and blank manager."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "simple", "--workers", "0"])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-POSITIONS-001"

    args = parser.parse_args(["--manager", "   "])
    with patch.object(generate_positions_module, "DEFAULT_STORAGE_ROOT", tmp_path):
        with pytest.raises(ValidationError) as exc_info:
            build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-POSITIONS-006"


def test_build_default_engine_and_registry() -> None:
    """Default registry contains AverageCostPositionEngine under average_cost."""
    assert isinstance(build_default_engine(), AverageCostPositionEngine)
    registry = build_engine_registry()
    assert registry.exists("average_cost")
    assert isinstance(registry.get("average_cost"), AverageCostPositionEngine)


def test_build_position_pipeline_wires_registry() -> None:
    """Pipeline composition uses the default engine registry."""
    options = _options(storage_root=Path(DEFAULT_STORAGE_ROOT))
    pipeline = build_position_pipeline(options)
    assert isinstance(pipeline, PositionPipeline)


def test_discover_work_groups_execution_partitions(tmp_path: Path) -> None:
    """discover_work returns sorted manager/symbol/timeframe work items."""
    layout = StorageLayout(tmp_path)
    trade_repository = TradeRepository(layout, ParquetStore())
    frame = _trade_frame()
    for symbol, year in (("ETHUSDT", 2025), ("BTCUSDT", 2026), ("BTCUSDT", 2025)):
        trade_repository.save(
            frame,
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            symbol=symbol,
            timeframe=_TIMEFRAME,
            year=year,
        )
    work = discover_work(
        trade_repository,
        _options(storage_root=tmp_path, model=None, version=None),
    )
    assert work == (
        DiscoveredWorkItem(
            manager=_MANAGER,
            symbol="BTCUSDT",
            timeframe=_TIMEFRAME,
            years=(2025, 2026),
        ),
        DiscoveredWorkItem(
            manager=_MANAGER,
            symbol="ETHUSDT",
            timeframe=_TIMEFRAME,
            years=(2025,),
        ),
    )


def test_format_summary_includes_core_fields() -> None:
    """format_summary renders manager, engine, and task counters."""
    summary = PositionGenerationSummary(
        manager=_MANAGER,
        engine=_ENGINE,
        version=_VERSION,
        symbols_discovered=2,
        symbols_processed=2,
        timeframes_processed=1,
        successful_tasks=1,
        failed_tasks=1,
        skipped_tasks=0,
        rows_generated=10,
        duration_seconds=1.5,
        output_directory=Path("data") / STORAGE_DIR_POSITIONS,
        failed_task_labels=("BTCUSDT 1h 2026",),
    )
    text = format_summary(summary)
    assert "CQROS Position Generation Summary" in text
    assert f"Manager: {_MANAGER}" in text
    assert f"Engine: {_ENGINE}" in text
    assert "Failed Tasks" in text
    assert "BTCUSDT 1h 2026" in text


def test_run_generation_empty_work(tmp_path: Path) -> None:
    """Empty work produces a zeroed success summary."""
    options = _options(storage_root=tmp_path)
    pipeline = PositionPipeline(build_engine_registry())
    summary = _run(
        run_generation(
            pipeline=pipeline,
            trade_repository=TradeRepository(StorageLayout(tmp_path), ParquetStore()),
            position_repository=PositionRepository(StorageLayout(tmp_path), ParquetStore()),
            options=options,
            work=(),
        )
    )
    assert summary.successful_tasks == 0
    assert summary.failed_tasks == 0
    assert summary.rows_generated == 0
    assert summary.output_directory == tmp_path / STORAGE_DIR_POSITIONS


def test_run_generation_persists_positions(tmp_path: Path) -> None:
    """Generation loads trades, runs the pipeline, and writes positions."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    trade_repository = TradeRepository(layout, datastore)
    position_repository = PositionRepository(layout, datastore)
    trade_repository.save(
        _trade_frame(),
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    options = _options(storage_root=tmp_path, model=None, version=None)
    work = discover_work(trade_repository, options)
    registry = PositionEngineRegistry()
    registry.register(_ENGINE, AverageCostPositionEngine())
    summary = _run(
        run_generation(
            pipeline=PositionPipeline(registry),
            trade_repository=trade_repository,
            position_repository=position_repository,
            options=options,
            work=work,
        )
    )
    assert summary.successful_tasks == 1
    assert summary.failed_tasks == 0
    assert summary.rows_generated == 1
    assert position_repository.exists(
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert STORAGE_DIR_EXECUTIONS in str(tmp_path / STORAGE_DIR_EXECUTIONS)
