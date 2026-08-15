"""Unit tests for CQROS accounting-generation CLI."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

import cqros.cli.generate_accounting as generate_accounting_module
from cqros.accounting import (
    AccountingEngineRegistry,
    AccountingPipeline,
    AccountingRepository,
    SimplePortfolioAccountingEngine,
)
from cqros.cli.generate_accounting import (
    AccountingGenerationOptions,
    AccountingGenerationSummary,
    DiscoveredWorkItem,
    build_accounting_pipeline,
    build_default_engine,
    build_engine_registry,
    build_options,
    build_parser,
    discover_work,
    format_summary,
    run_generation,
)
from cqros.config.models import ResearchConfig
from cqros.core.constants import (
    DEFAULT_STORAGE_ROOT,
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
    STORAGE_DIR_ACCOUNTING,
    STORAGE_DIR_POSITIONS,
)
from cqros.core.exceptions import ValidationError
from cqros.positions import (
    CANONICAL_COLUMN_ORDER as POSITION_COLUMNS,
)
from cqros.positions import (
    COLUMN_DTYPES as POSITION_DTYPES,
)
from cqros.positions import (
    PositionRepository,
    PositionStatus,
)
from cqros.storage import ParquetStore, StorageLayout

_MANAGER = "simple"
_ENGINE = "simple"
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
    cash: float = 0.0,
    verbose: bool = False,
    debug: bool = False,
) -> AccountingGenerationOptions:
    """Build options for tests against a temporary storage root."""
    return AccountingGenerationOptions(
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
        cash=cash,
        verbose=verbose,
        debug=debug,
    )


def _position_frame(
    *,
    symbol: str = _SYMBOL,
    model_name: str = _MODEL,
    model_version: str = _VERSION,
) -> pl.DataFrame:
    """Return a canonical position frame for generation input."""
    opened_at = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
    return pl.DataFrame(
        {
            "symbol": [symbol],
            "timeframe": [_TIMEFRAME],
            "position_id": ["pos-00000001"],
            "side": ["LONG"],
            "status": [PositionStatus.OPEN.value],
            "quantity": [2.0],
            "average_entry_price": [100.0],
            "market_price": [110.0],
            "realized_pnl": [0.0],
            "unrealized_pnl": [20.0],
            "fees_paid": [0.0],
            "opened_at": [opened_at],
            "updated_at": [opened_at],
            "closed_at": [None],
            "model_name": [model_name],
            "model_version": [model_version],
            "optimizer": ["equal_weight"],
            "policy": ["fixed_risk"],
            "manager": [_MANAGER],
        },
        schema=dict(POSITION_DTYPES),
    ).select(list(POSITION_COLUMNS))


def test_build_parser_requires_manager() -> None:
    """Parser requires --manager and defaults engine to simple."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(["--manager", "simple"])
    assert args.manager == "simple"
    assert args.engine == "simple"
    assert args.cash == 0.0


def test_build_options_validates_workers_manager_and_cash(tmp_path: Path) -> None:
    """build_options rejects non-positive workers, blank manager, and bad cash."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "simple", "--workers", "0"])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-ACCOUNTING-001"

    args = parser.parse_args(["--manager", "   "])
    with patch.object(generate_accounting_module, "DEFAULT_STORAGE_ROOT", tmp_path):
        with pytest.raises(ValidationError) as exc_info:
            build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-ACCOUNTING-006"

    args = parser.parse_args(["--manager", "simple", "--cash", "nan"])
    with patch.object(generate_accounting_module, "DEFAULT_STORAGE_ROOT", tmp_path):
        with pytest.raises(ValidationError) as exc_info:
            build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-ACCOUNTING-008"


def test_build_default_engine_and_registry() -> None:
    """Default registry contains SimplePortfolioAccountingEngine under simple."""
    assert isinstance(build_default_engine(), SimplePortfolioAccountingEngine)
    assert isinstance(build_default_engine(cash=25.0), SimplePortfolioAccountingEngine)
    registry = build_engine_registry(cash=10.0)
    assert registry.exists("simple")
    assert isinstance(registry.get("simple"), SimplePortfolioAccountingEngine)


def test_build_accounting_pipeline_wires_registry() -> None:
    """Pipeline composition uses the default engine registry."""
    options = _options(storage_root=Path(DEFAULT_STORAGE_ROOT), cash=5.0)
    pipeline = build_accounting_pipeline(options)
    assert isinstance(pipeline, AccountingPipeline)


def test_discover_work_groups_position_partitions(tmp_path: Path) -> None:
    """discover_work returns sorted manager/symbol/timeframe work items."""
    layout = StorageLayout(tmp_path)
    position_repository = PositionRepository(layout, ParquetStore())
    frame = _position_frame()
    for symbol, year in (("ETHUSDT", 2025), ("BTCUSDT", 2026), ("BTCUSDT", 2025)):
        position_repository.save(
            frame,
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            symbol=symbol,
            timeframe=_TIMEFRAME,
            year=year,
        )
    work = discover_work(
        position_repository,
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
    summary = AccountingGenerationSummary(
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
        output_directory=Path("data") / STORAGE_DIR_ACCOUNTING,
        failed_task_labels=("BTCUSDT 1h 2026",),
    )
    text = format_summary(summary)
    assert "CQROS Accounting Generation Summary" in text
    assert f"Manager: {_MANAGER}" in text
    assert f"Engine: {_ENGINE}" in text
    assert "Failed Tasks" in text
    assert "BTCUSDT 1h 2026" in text
    assert STORAGE_DIR_ACCOUNTING in text


def test_run_generation_empty_work(tmp_path: Path) -> None:
    """Empty work produces a zeroed success summary."""
    options = _options(storage_root=tmp_path)
    pipeline = AccountingPipeline(build_engine_registry())
    summary = _run(
        run_generation(
            pipeline=pipeline,
            position_repository=PositionRepository(StorageLayout(tmp_path), ParquetStore()),
            accounting_repository=AccountingRepository(
                StorageLayout(tmp_path),
                ParquetStore(),
            ),
            options=options,
            work=(),
        )
    )
    assert summary.successful_tasks == 0
    assert summary.failed_tasks == 0
    assert summary.rows_generated == 0
    assert summary.output_directory == tmp_path / STORAGE_DIR_ACCOUNTING


def test_run_generation_persists_accounting(tmp_path: Path) -> None:
    """Generation loads positions, runs the pipeline, and writes accounting."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    position_repository = PositionRepository(layout, datastore)
    accounting_repository = AccountingRepository(layout, datastore)
    position_repository.save(
        _position_frame(),
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    options = _options(storage_root=tmp_path, model=None, version=None, cash=50.0)
    work = discover_work(position_repository, options)
    registry = AccountingEngineRegistry()
    registry.register(_ENGINE, SimplePortfolioAccountingEngine(cash=50.0))
    summary = _run(
        run_generation(
            pipeline=AccountingPipeline(registry),
            position_repository=position_repository,
            accounting_repository=accounting_repository,
            options=options,
            work=work,
        )
    )
    assert summary.successful_tasks == 1
    assert summary.failed_tasks == 0
    assert summary.rows_generated == 1
    assert accounting_repository.exists(
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    frame = accounting_repository.load(
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert frame["market_value"].to_list() == [220.0]
    assert frame["cash"].to_list() == [50.0]
    assert frame["equity"].to_list() == [270.0]
    assert STORAGE_DIR_POSITIONS in str(tmp_path / STORAGE_DIR_POSITIONS)
    assert STORAGE_DIR_ACCOUNTING in str(tmp_path / STORAGE_DIR_ACCOUNTING)


def test_run_generation_skips_existing_without_overwrite(tmp_path: Path) -> None:
    """Existing accounting partitions are skipped when overwrite is false."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    position_repository = PositionRepository(layout, datastore)
    accounting_repository = AccountingRepository(layout, datastore)
    position_repository.save(
        _position_frame(),
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    accounting_repository.save(
        AccountingPipeline(build_engine_registry()).run(
            _position_frame(),
            manager=_MANAGER,
        ),
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    options = _options(storage_root=tmp_path, model=None, version=None, overwrite=False)
    work = discover_work(position_repository, options)
    summary = _run(
        run_generation(
            pipeline=AccountingPipeline(build_engine_registry()),
            position_repository=position_repository,
            accounting_repository=accounting_repository,
            options=options,
            work=work,
        )
    )
    assert summary.skipped_tasks == 1
    assert summary.successful_tasks == 0
    assert summary.failed_tasks == 0
