"""Unit tests for CQROS backtesting generation CLI."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

import cqros.cli.generate_backtesting as generate_backtesting_module
from cqros.accounting import AccountingRepository, PositionStatus
from cqros.backtesting import (
    BacktestingPipeline,
    BacktestingRegistry,
    BacktestingRepository,
    SimpleBacktestingEngine,
)
from cqros.cli.generate_backtesting import (
    BacktestingGenerationOptions,
    BacktestingGenerationSummary,
    DiscoveredWorkItem,
    build_backtesting_pipeline,
    build_default_engine,
    build_options,
    build_parser,
    build_registry,
    discover_work,
    format_summary,
    main,
    run_generation,
)
from cqros.config.models import ResearchConfig
from cqros.core.constants import (
    DEFAULT_STORAGE_ROOT,
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
    STORAGE_DIR_BACKTESTING,
)
from cqros.core.exceptions import ValidationError
from cqros.exit_engine import ExitAction, ExitReason, ExitRepository
from cqros.positions import PositionRepository
from cqros.positions import PositionStatus as PositionLifecycleStatus
from cqros.storage import ParquetStore, StorageLayout

_MANAGER = "simple"
_ENGINE = "simple"
_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1h"
_YEAR = 2026
_OPEN_TIME = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
_POSITION_ID = "pos-00000001"


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def _options(
    *,
    storage_root: Path,
    manager: str = _MANAGER,
    engine: str = _ENGINE,
    model: str | None = None,
    version: str | None = None,
    symbols: tuple[str, ...] | None = None,
    timeframes: tuple[str, ...] | None = None,
    years: tuple[int, ...] | None = None,
    overwrite: bool = False,
    workers: int = ResearchConfig().worker_count,
    verbose: bool = False,
    debug: bool = False,
) -> BacktestingGenerationOptions:
    """Build BacktestingGenerationOptions against a temporary storage root."""
    return BacktestingGenerationOptions(
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


def _accounting_frame(
    *,
    symbol: str = _SYMBOL,
    open_times: list[datetime] | None = None,
    position_ids: list[str] | None = None,
    cash: float = 1000.0,
    position_value: float = 100.0,
    unrealized_pnl: float = 15.0,
) -> pl.DataFrame:
    """Return a minimal accounting frame for backtesting generation tests."""
    open_times = open_times if open_times is not None else [_OPEN_TIME]
    position_ids = position_ids if position_ids is not None else [_POSITION_ID]
    row_count = len(open_times)
    return pl.DataFrame(
        {
            "symbol": [symbol] * row_count,
            "timeframe": [_TIMEFRAME] * row_count,
            "open_time": open_times,
            "cash": [cash] * row_count,
            "position_value": [position_value] * row_count,
            "realized_pnl": [0.0] * row_count,
            "unrealized_pnl": [unrealized_pnl] * row_count,
            "position_id": position_ids,
            "position_status": [PositionStatus.OPEN.value] * row_count,
            "model_name": ["alpha-lgbm"] * row_count,
            "model_version": ["1.0.0"] * row_count,
        }
    )


def _position_frame(*, symbol: str = _SYMBOL) -> pl.DataFrame:
    """Return a minimal position frame for backtesting generation tests."""
    return pl.DataFrame(
        {
            "symbol": [symbol],
            "timeframe": [_TIMEFRAME],
            "position_id": [_POSITION_ID],
            "status": [PositionLifecycleStatus.OPEN.value],
            "realized_pnl": [0.0],
            "opened_at": [_OPEN_TIME],
            "updated_at": [_OPEN_TIME],
            "closed_at": [None],
        }
    )


def _exit_engine_frame(
    *,
    symbol: str = _SYMBOL,
    open_times: list[datetime] | None = None,
    position_ids: list[str] | None = None,
    exit_action: str = ExitAction.HOLD.value,
) -> pl.DataFrame:
    """Return a minimal exit-engine frame for backtesting generation tests."""
    open_times = open_times if open_times is not None else [_OPEN_TIME]
    position_ids = position_ids if position_ids is not None else [_POSITION_ID]
    row_count = len(open_times)
    return pl.DataFrame(
        {
            "symbol": [symbol] * row_count,
            "timeframe": [_TIMEFRAME] * row_count,
            "open_time": open_times,
            "position_id": position_ids,
            "manager": [_MANAGER] * row_count,
            "entry_price": [100.0] * row_count,
            "current_price": [115.0] * row_count,
            "quantity": [1.0] * row_count,
            "risk_reward_ratio": [0.4] * row_count,
            "risk_state": ["NORMAL"] * row_count,
            "trade_state": ["NONE"] * row_count,
            "pyramid_state": ["READY_TO_ADD"] * row_count,
            "exit_action": [exit_action] * row_count,
            "exit_reason": [ExitReason.NONE.value] * row_count,
            "recommended_quantity": [0.0] * row_count,
            "recommended_percent": [0.0] * row_count,
            "priority": [0] * row_count,
            "created_at": open_times,
        }
    )


def _seed_generation_inputs(
    *,
    layout: StorageLayout,
    datastore: ParquetStore,
    symbol: str = _SYMBOL,
    open_times: list[datetime] | None = None,
    position_ids: list[str] | None = None,
    cash: float = 1000.0,
    unrealized_pnl: float = 15.0,
) -> None:
    """Persist all inputs needed for backtesting generation for one symbol."""
    open_times = open_times if open_times is not None else [_OPEN_TIME]
    position_ids = position_ids if position_ids is not None else [_POSITION_ID]

    AccountingRepository(layout, datastore).save(
        _accounting_frame(
            symbol=symbol,
            open_times=open_times,
            position_ids=position_ids,
            cash=cash,
            unrealized_pnl=unrealized_pnl,
        ),
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=symbol,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    PositionRepository(layout, datastore).save(
        _position_frame(symbol=symbol),
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=symbol,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    ExitRepository(layout, datastore).save(
        _exit_engine_frame(
            symbol=symbol,
            open_times=open_times,
            position_ids=position_ids,
        ),
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=symbol,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )


# ---------------------------------------------------------------------------
# Parser and options
# ---------------------------------------------------------------------------


def test_build_parser_requires_manager() -> None:
    """Parser requires --manager and defaults engine to simple."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(["--manager", "simple"])
    assert args.manager == "simple"
    assert args.engine == "simple"


def test_build_parser_accepts_all_flags(tmp_path: Path) -> None:
    """Parser correctly maps all supported CLI flags."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "--manager",
            "ledger",
            "--engine",
            "custom",
            "--workers",
            "4",
            "--overwrite",
            "--storage-root",
            str(tmp_path),
        ]
    )
    assert args.manager == "ledger"
    assert args.engine == "custom"
    assert args.workers == 4
    assert args.overwrite is True
    assert args.storage_root == tmp_path


def test_build_options_rejects_non_positive_workers() -> None:
    """build_options rejects workers <= 0 with CLI-GENERATE-BACKTESTING-001."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "simple", "--workers", "0"])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-BACKTESTING-001"


def test_build_options_rejects_blank_manager() -> None:
    """build_options rejects a blank manager identity with CLI-GENERATE-BACKTESTING-006."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "   "])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-BACKTESTING-006"


def test_build_options_rejects_blank_engine() -> None:
    """build_options rejects a blank engine identity with CLI-GENERATE-BACKTESTING-007."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "simple", "--engine", "   "])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-BACKTESTING-007"


def test_build_options_uses_storage_root(tmp_path: Path) -> None:
    """build_options honors an explicit storage root override."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "simple", "--storage-root", str(tmp_path)])
    options = build_options(args)
    assert options.storage_root == tmp_path


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def test_build_default_engine_returns_simple_backtesting_engine() -> None:
    """build_default_engine returns a SimpleBacktestingEngine instance."""
    assert isinstance(build_default_engine(), SimpleBacktestingEngine)


def test_build_registry_contains_simple_engine() -> None:
    """Default registry contains SimpleBacktestingEngine under 'simple'."""
    registry = build_registry()
    assert registry.exists("simple")
    assert isinstance(registry.get("simple"), SimpleBacktestingEngine)


def test_build_backtesting_pipeline_wires_registry() -> None:
    """build_backtesting_pipeline returns a fully wired BacktestingPipeline."""
    options = _options(storage_root=Path(DEFAULT_STORAGE_ROOT))
    pipeline = build_backtesting_pipeline(options)
    assert isinstance(pipeline, BacktestingPipeline)


# ---------------------------------------------------------------------------
# discover_work
# ---------------------------------------------------------------------------


def test_discover_work_groups_exit_engine_partitions(tmp_path: Path) -> None:
    """discover_work returns sorted work items from exit-engine tier."""
    layout = StorageLayout(tmp_path)
    exit_repository = ExitRepository(layout, ParquetStore())
    for symbol, year in (("ETHUSDT", 2025), ("BTCUSDT", 2026), ("BTCUSDT", 2025)):
        exit_repository.save(
            _exit_engine_frame(symbol=symbol),
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            symbol=symbol,
            timeframe=_TIMEFRAME,
            year=year,
        )
    work = discover_work(
        exit_repository,
        _options(storage_root=tmp_path),
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


def test_discover_work_returns_empty_when_no_exit_partitions(tmp_path: Path) -> None:
    """discover_work returns empty tuple when no exit-engine partitions exist."""
    layout = StorageLayout(tmp_path)
    exit_repository = ExitRepository(layout, ParquetStore())
    work = discover_work(exit_repository, _options(storage_root=tmp_path))
    assert work == ()


# ---------------------------------------------------------------------------
# format_summary
# ---------------------------------------------------------------------------


def test_format_summary_includes_core_fields() -> None:
    """format_summary renders manager, engine, and performance aggregates."""
    summary = BacktestingGenerationSummary(
        manager=_MANAGER,
        engine=_ENGINE,
        symbols=2,
        rows=10,
        trades=3,
        total_return=0.1234,
        max_dd=0.0567,
        successful_tasks=2,
        failed_tasks=1,
        skipped_tasks=0,
        duration_seconds=2.5,
        output_directory=Path("data") / STORAGE_DIR_BACKTESTING,
        failed_task_labels=("BTCUSDT 1h 2026",),
    )
    text = format_summary(summary)
    assert "CQROS Backtesting Generation Summary" in text
    assert f"Manager: {_MANAGER}" in text
    assert f"Engine: {_ENGINE}" in text
    assert "Symbols: 2" in text
    assert "Rows: 10" in text
    assert "Trades: 3" in text
    assert "Total Return: 0.1234" in text
    assert "Max DD: 0.0567" in text
    assert "Failed Tasks" in text
    assert "BTCUSDT 1h 2026" in text
    assert STORAGE_DIR_BACKTESTING in text


# ---------------------------------------------------------------------------
# run_generation — empty work
# ---------------------------------------------------------------------------


def test_run_generation_empty_work(tmp_path: Path) -> None:
    """Empty work produces a zeroed success summary."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    options = _options(storage_root=tmp_path)
    registry = build_registry()
    summary = _run(
        run_generation(
            pipeline=BacktestingPipeline(registry),
            accounting_repository=AccountingRepository(layout, datastore),
            position_repository=PositionRepository(layout, datastore),
            exit_repository=ExitRepository(layout, datastore),
            backtesting_repository=BacktestingRepository(layout, datastore),
            options=options,
            work=(),
        )
    )
    assert summary.successful_tasks == 0
    assert summary.failed_tasks == 0
    assert summary.skipped_tasks == 0
    assert summary.rows == 0
    assert summary.trades == 0
    assert summary.total_return == 0.0
    assert summary.max_dd == 0.0
    assert summary.output_directory == tmp_path / STORAGE_DIR_BACKTESTING


# ---------------------------------------------------------------------------
# run_generation — persists performance ledgers
# ---------------------------------------------------------------------------


def test_run_generation_persists_backtesting_partitions(tmp_path: Path) -> None:
    """Generation loads inputs, runs the pipeline, and persists backtesting output."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()

    _seed_generation_inputs(layout=layout, datastore=datastore, symbol="BTCUSDT")
    _seed_generation_inputs(
        layout=layout,
        datastore=datastore,
        symbol="ETHUSDT",
        cash=2000.0,
        unrealized_pnl=25.0,
    )

    options = _options(storage_root=tmp_path)
    exit_repository = ExitRepository(layout, datastore)
    work = discover_work(exit_repository, options)

    registry = BacktestingRegistry()
    registry.register(_ENGINE, SimpleBacktestingEngine())
    summary = _run(
        run_generation(
            pipeline=BacktestingPipeline(registry),
            accounting_repository=AccountingRepository(layout, datastore),
            position_repository=PositionRepository(layout, datastore),
            exit_repository=exit_repository,
            backtesting_repository=BacktestingRepository(layout, datastore),
            options=options,
            work=work,
        )
    )
    assert summary.successful_tasks == 2
    assert summary.failed_tasks == 0
    assert summary.rows == 2
    assert summary.trades == 0
    assert summary.total_return == 0.0
    assert summary.max_dd == 0.0

    backtesting_repo = BacktestingRepository(layout, datastore)
    assert backtesting_repo.exists(
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol="BTCUSDT",
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert backtesting_repo.exists(
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol="ETHUSDT",
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )


# ---------------------------------------------------------------------------
# run_generation — skip existing without overwrite
# ---------------------------------------------------------------------------


def test_run_generation_skips_existing_without_overwrite(tmp_path: Path) -> None:
    """Existing backtesting partitions are skipped when overwrite is False."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    _seed_generation_inputs(layout=layout, datastore=datastore)

    options = _options(storage_root=tmp_path, overwrite=False)
    exit_repository = ExitRepository(layout, datastore)
    work = discover_work(exit_repository, options)

    registry = build_registry()
    first_summary = _run(
        run_generation(
            pipeline=BacktestingPipeline(registry),
            accounting_repository=AccountingRepository(layout, datastore),
            position_repository=PositionRepository(layout, datastore),
            exit_repository=exit_repository,
            backtesting_repository=BacktestingRepository(layout, datastore),
            options=options,
            work=work,
        )
    )
    assert first_summary.successful_tasks == 1

    second_summary = _run(
        run_generation(
            pipeline=BacktestingPipeline(build_registry()),
            accounting_repository=AccountingRepository(layout, datastore),
            position_repository=PositionRepository(layout, datastore),
            exit_repository=exit_repository,
            backtesting_repository=BacktestingRepository(layout, datastore),
            options=options,
            work=work,
        )
    )
    assert second_summary.skipped_tasks == 1
    assert second_summary.successful_tasks == 0
    assert second_summary.failed_tasks == 0


# ---------------------------------------------------------------------------
# run_generation — failure when accounting missing
# ---------------------------------------------------------------------------


def test_run_generation_fails_when_accounting_missing(tmp_path: Path) -> None:
    """Generation fails a task when accounting partition is absent."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    exit_repository = ExitRepository(layout, datastore)
    exit_repository.save(
        _exit_engine_frame(),
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    PositionRepository(layout, datastore).save(
        _position_frame(),
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )

    options = _options(storage_root=tmp_path)
    work = discover_work(exit_repository, options)
    summary = _run(
        run_generation(
            pipeline=BacktestingPipeline(build_registry()),
            accounting_repository=AccountingRepository(layout, datastore),
            position_repository=PositionRepository(layout, datastore),
            exit_repository=exit_repository,
            backtesting_repository=BacktestingRepository(layout, datastore),
            options=options,
            work=work,
        )
    )
    assert summary.failed_tasks == 1
    assert summary.successful_tasks == 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_returns_success_for_empty_universe(tmp_path: Path) -> None:
    """main returns 0 when no exit-engine partitions are discovered."""
    with patch.object(generate_backtesting_module, "DEFAULT_STORAGE_ROOT", tmp_path):
        code = _run(main(["--manager", "simple", "--storage-root", str(tmp_path)]))
    assert code == 0


def test_main_validation_error_exit_code() -> None:
    """main returns 1 when build_options raises ValidationError."""
    code = _run(main(["--manager", "simple", "--workers", "0"]))
    assert code == 1


def test_main_returns_failure_when_generation_fails(tmp_path: Path) -> None:
    """main returns 1 when any generation task fails."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    ExitRepository(layout, datastore).save(
        _exit_engine_frame(),
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    with patch.object(generate_backtesting_module, "DEFAULT_STORAGE_ROOT", tmp_path):
        code = _run(main(["--manager", "simple", "--storage-root", str(tmp_path)]))
    assert code == 1


def test_main_returns_success_after_generation(tmp_path: Path) -> None:
    """main returns 0 after successful backtesting generation."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    _seed_generation_inputs(layout=layout, datastore=datastore)
    with patch.object(generate_backtesting_module, "DEFAULT_STORAGE_ROOT", tmp_path):
        code = _run(main(["--manager", "simple", "--storage-root", str(tmp_path)]))
    assert code == 0
