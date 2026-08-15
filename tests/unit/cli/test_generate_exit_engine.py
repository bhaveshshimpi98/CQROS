"""Unit tests for CQROS exit-engine generation CLI."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

import cqros.cli.generate_exit_engine as generate_exit_engine_module
from cqros.accounting import CANONICAL_COLUMN_ORDER as ACCOUNTING_COLUMNS
from cqros.accounting import COLUMN_DTYPES as ACCOUNTING_DTYPES
from cqros.accounting import (
    AccountingRepository,
    PositionStatus,
)
from cqros.cli.generate_exit_engine import (
    DiscoveredWorkItem,
    ExitEngineGenerationOptions,
    ExitEngineGenerationSummary,
    build_default_engine,
    build_exit_engine_pipeline,
    build_options,
    build_parser,
    build_registry,
    discover_work,
    format_summary,
    run_generation,
)
from cqros.config.models import ResearchConfig
from cqros.core.constants import (
    DEFAULT_STORAGE_ROOT,
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
    STORAGE_DIR_EXIT_ENGINE,
)
from cqros.core.exceptions import ValidationError
from cqros.exit_engine import (
    ExitEnginePipeline,
    ExitEngineRegistry,
    ExitRepository,
    SimpleExitEngine,
)
from cqros.portfolio_risk import CANONICAL_COLUMN_ORDER as PORTFOLIO_RISK_COLUMNS
from cqros.portfolio_risk import COLUMN_DTYPES as PORTFOLIO_RISK_DTYPES
from cqros.portfolio_risk import (
    PortfolioRiskRepository,
    PortfolioRiskState,
)
from cqros.portfolio_risk import ShutdownReason as PortfolioRiskShutdownReason
from cqros.positions import CANONICAL_COLUMN_ORDER as POSITION_COLUMNS
from cqros.positions import COLUMN_DTYPES as POSITION_DTYPES
from cqros.positions import PositionRepository
from cqros.positions import PositionStatus as PositionLifecycleStatus
from cqros.pyramiding import CANONICAL_COLUMN_ORDER as PYRAMIDING_COLUMNS
from cqros.pyramiding import COLUMN_DTYPES as PYRAMIDING_DTYPES
from cqros.pyramiding import (
    PyramidingReason,
    PyramidingRepository,
)
from cqros.storage import ParquetStore, StorageLayout
from cqros.trade_management import CANONICAL_COLUMN_ORDER as TM_COLUMNS
from cqros.trade_management import COLUMN_DTYPES as TM_DTYPES
from cqros.trade_management import ManagementAction, TradeManagementRepository

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
) -> ExitEngineGenerationOptions:
    """Build ExitEngineGenerationOptions against a temporary storage root."""
    return ExitEngineGenerationOptions(
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
    current_price: float = 115.0,
) -> pl.DataFrame:
    """Return a canonical accounting frame for generation tests."""
    open_times = open_times if open_times is not None else [_OPEN_TIME]
    position_ids = position_ids if position_ids is not None else [_POSITION_ID]
    row_count = len(open_times)
    return pl.DataFrame(
        {
            "symbol": [symbol] * row_count,
            "timeframe": [_TIMEFRAME] * row_count,
            "open_time": open_times,
            "manager": [_MANAGER] * row_count,
            "position_id": position_ids,
            "position_status": [PositionStatus.OPEN.value] * row_count,
            "quantity": [1.0] * row_count,
            "average_entry_price": [100.0] * row_count,
            "mark_price": [current_price] * row_count,
            "position_value": [current_price] * row_count,
            "market_value": [current_price] * row_count,
            "cash": [0.0] * row_count,
            "realized_pnl": [0.0] * row_count,
            "unrealized_pnl": [current_price - 100.0] * row_count,
            "total_pnl": [current_price - 100.0] * row_count,
            "gross_exposure": [500.0] * row_count,
            "net_exposure": [500.0] * row_count,
            "equity": [1000.0] * row_count,
            "return_pct": [0.0] * row_count,
            "model_name": ["alpha-lgbm"] * row_count,
            "model_version": ["1.0.0"] * row_count,
            "optimizer": ["equal_weight"] * row_count,
            "policy": ["fixed_risk"] * row_count,
        },
        schema=dict(ACCOUNTING_DTYPES),
    ).select(list(ACCOUNTING_COLUMNS))


def _position_frame(*, symbol: str = _SYMBOL) -> pl.DataFrame:
    """Return a canonical position frame for generation tests."""
    return pl.DataFrame(
        {
            "symbol": [symbol],
            "timeframe": [_TIMEFRAME],
            "position_id": [_POSITION_ID],
            "side": ["LONG"],
            "status": [PositionLifecycleStatus.OPEN.value],
            "quantity": [1.0],
            "average_entry_price": [100.0],
            "market_price": [115.0],
            "realized_pnl": [0.0],
            "unrealized_pnl": [15.0],
            "fees_paid": [0.0],
            "opened_at": [_OPEN_TIME],
            "updated_at": [_OPEN_TIME],
            "closed_at": [None],
            "model_name": ["alpha-lgbm"],
            "model_version": ["1.0.0"],
            "optimizer": ["equal_weight"],
            "policy": ["fixed_risk"],
            "manager": [_MANAGER],
        },
        schema=dict(POSITION_DTYPES),
    ).select(list(POSITION_COLUMNS))


def _portfolio_risk_frame(
    *,
    symbol: str = _SYMBOL,
    open_times: list[datetime] | None = None,
    position_ids: list[str] | None = None,
    risk_state: str = PortfolioRiskState.NORMAL.value,
    shutdown_reason: str = PortfolioRiskShutdownReason.NONE.value,
) -> pl.DataFrame:
    """Return a canonical portfolio-risk frame for generation tests."""
    open_times = open_times if open_times is not None else [_OPEN_TIME]
    position_ids = position_ids if position_ids is not None else [_POSITION_ID]
    row_count = len(open_times)
    return pl.DataFrame(
        {
            "symbol": [symbol] * row_count,
            "timeframe": [_TIMEFRAME] * row_count,
            "open_time": open_times,
            "manager": [_MANAGER] * row_count,
            "position_id": position_ids,
            "equity": [1000.0] * row_count,
            "gross_exposure": [500.0] * row_count,
            "net_exposure": [500.0] * row_count,
            "daily_realized_pnl": [0.0] * row_count,
            "daily_unrealized_pnl": [0.0] * row_count,
            "daily_total_pnl": [0.0] * row_count,
            "daily_return_pct": [0.0] * row_count,
            "daily_drawdown_pct": [0.0] * row_count,
            "portfolio_risk_state": [risk_state] * row_count,
            "allow_new_entries": [True] * row_count,
            "shutdown_reason": [shutdown_reason] * row_count,
            "cooldown_until": [None] * row_count,
            "model_name": ["alpha-lgbm"] * row_count,
            "model_version": ["1.0.0"] * row_count,
            "optimizer": ["equal_weight"] * row_count,
            "policy": ["fixed_risk"] * row_count,
        },
        schema=dict(PORTFOLIO_RISK_DTYPES),
    ).select(list(PORTFOLIO_RISK_COLUMNS))


def _trade_management_frame(
    *,
    symbol: str = _SYMBOL,
    open_times: list[datetime] | None = None,
    position_ids: list[str] | None = None,
    current_price: float = 115.0,
    action_reason: str = "NONE",
) -> pl.DataFrame:
    """Return a canonical trade-management frame for generation tests."""
    open_times = open_times if open_times is not None else [_OPEN_TIME]
    position_ids = position_ids if position_ids is not None else [_POSITION_ID]
    row_count = len(open_times)
    return pl.DataFrame(
        {
            "symbol": [symbol] * row_count,
            "timeframe": [_TIMEFRAME] * row_count,
            "open_time": open_times,
            "manager": [_MANAGER] * row_count,
            "position_id": position_ids,
            "position_status": ["OPEN"] * row_count,
            "quantity": [1.0] * row_count,
            "entry_price": [100.0] * row_count,
            "current_price": [current_price] * row_count,
            "highest_price": [current_price] * row_count,
            "lowest_price": [100.0] * row_count,
            "unrealized_pnl": [current_price - 100.0] * row_count,
            "risk_state": ["NORMAL"] * row_count,
            "management_action": [ManagementAction.NONE.value] * row_count,
            "action_reason": [action_reason] * row_count,
            "stop_price": [None] * row_count,
            "take_profit_price": [None] * row_count,
            "trail_price": [90.0] * row_count,
            "breakeven_price": [None] * row_count,
            "allow_pyramid": [True] * row_count,
            "exit_quantity": [0.0] * row_count,
            "model_name": ["alpha-lgbm"] * row_count,
            "model_version": ["1.0.0"] * row_count,
            "optimizer": ["equal_weight"] * row_count,
            "policy": ["fixed_risk"] * row_count,
        },
        schema=dict(TM_DTYPES),
    ).select(list(TM_COLUMNS))


def _pyramiding_frame(
    *,
    symbol: str = _SYMBOL,
    open_times: list[datetime] | None = None,
    position_ids: list[str] | None = None,
    reason: str = PyramidingReason.READY_TO_ADD.value,
) -> pl.DataFrame:
    """Return a canonical pyramiding frame for generation tests."""
    open_times = open_times if open_times is not None else [_OPEN_TIME]
    position_ids = position_ids if position_ids is not None else [_POSITION_ID]
    row_count = len(open_times)
    return pl.DataFrame(
        {
            "manager": [_MANAGER] * row_count,
            "symbol": [symbol] * row_count,
            "timeframe": [_TIMEFRAME] * row_count,
            "open_time": open_times,
            "position_id": position_ids,
            "trade_id": position_ids,
            "entry_price": [100.0] * row_count,
            "current_price": [115.0] * row_count,
            "highest_price": [115.0] * row_count,
            "position_size": [1.0] * row_count,
            "add_number": [1] * row_count,
            "max_adds": [3] * row_count,
            "additional_size": [0.5] * row_count,
            "recommended_size": [1.5] * row_count,
            "profit_pct": [0.15] * row_count,
            "allow_pyramid": [True] * row_count,
            "reason": [reason] * row_count,
        },
        schema=dict(PYRAMIDING_DTYPES),
    ).select(list(PYRAMIDING_COLUMNS))


def _seed_generation_inputs(
    *,
    layout: StorageLayout,
    datastore: ParquetStore,
    symbol: str = _SYMBOL,
    open_times: list[datetime] | None = None,
    position_ids: list[str] | None = None,
    current_price: float = 115.0,
    action_reason: str = "NONE",
) -> None:
    """Persist all inputs needed for exit-engine generation for one symbol."""
    open_times = open_times if open_times is not None else [_OPEN_TIME]
    position_ids = position_ids if position_ids is not None else [_POSITION_ID]

    AccountingRepository(layout, datastore).save(
        _accounting_frame(
            symbol=symbol,
            open_times=open_times,
            position_ids=position_ids,
            current_price=current_price,
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
    PortfolioRiskRepository(layout, datastore).save(
        _portfolio_risk_frame(
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
    TradeManagementRepository(layout, datastore).save(
        _trade_management_frame(
            symbol=symbol,
            open_times=open_times,
            position_ids=position_ids,
            current_price=current_price,
            action_reason=action_reason,
        ),
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=symbol,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    PyramidingRepository(layout, datastore).save(
        _pyramiding_frame(
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


def test_build_parser_defaults_manager_and_engine() -> None:
    """Parser defaults manager to simple and engine to simple when omitted."""
    parser = build_parser()
    args = parser.parse_args([])
    assert args.manager == "simple"
    assert args.engine == "simple"


def test_build_parser_accepts_all_flags() -> None:
    """Parser correctly maps all supported CLI flags."""
    parser = build_parser()
    args = parser.parse_args(
        ["--manager", "ledger", "--engine", "custom", "--workers", "4", "--overwrite"]
    )
    assert args.manager == "ledger"
    assert args.engine == "custom"
    assert args.workers == 4
    assert args.overwrite is True


def test_build_options_rejects_non_positive_workers() -> None:
    """build_options rejects workers <= 0 with CLI-GENERATE-EXIT-ENGINE-001."""
    parser = build_parser()
    args = parser.parse_args(["--workers", "0"])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-EXIT-ENGINE-001"


def test_build_options_rejects_blank_manager() -> None:
    """build_options rejects a blank manager identity with CLI-GENERATE-EXIT-ENGINE-006."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "   "])
    with patch.object(generate_exit_engine_module, "DEFAULT_STORAGE_ROOT", "/tmp"):
        with pytest.raises(ValidationError) as exc_info:
            build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-EXIT-ENGINE-006"


def test_build_options_rejects_blank_engine() -> None:
    """build_options rejects a blank engine identity with CLI-GENERATE-EXIT-ENGINE-007."""
    parser = build_parser()
    args = parser.parse_args(["--engine", "   "])
    with patch.object(generate_exit_engine_module, "DEFAULT_STORAGE_ROOT", "/tmp"):
        with pytest.raises(ValidationError) as exc_info:
            build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-EXIT-ENGINE-007"


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def test_build_default_engine_returns_simple_exit_engine() -> None:
    """build_default_engine returns a SimpleExitEngine instance."""
    assert isinstance(build_default_engine(), SimpleExitEngine)


def test_build_registry_contains_simple_engine() -> None:
    """Default registry contains SimpleExitEngine under 'simple'."""
    registry = build_registry()
    assert registry.exists("simple")
    assert isinstance(registry.get("simple"), SimpleExitEngine)


def test_build_exit_engine_pipeline_wires_registry() -> None:
    """build_exit_engine_pipeline returns a fully wired ExitEnginePipeline."""
    options = _options(storage_root=Path(DEFAULT_STORAGE_ROOT))
    pipeline = build_exit_engine_pipeline(options)
    assert isinstance(pipeline, ExitEnginePipeline)


# ---------------------------------------------------------------------------
# discover_work
# ---------------------------------------------------------------------------


def test_discover_work_groups_pyramiding_partitions(tmp_path: Path) -> None:
    """discover_work returns sorted manager/symbol/timeframe work items from pyramiding tier."""
    layout = StorageLayout(tmp_path)
    pyr_repository = PyramidingRepository(layout, ParquetStore())
    for symbol, year in (("ETHUSDT", 2025), ("BTCUSDT", 2026), ("BTCUSDT", 2025)):
        pyr_repository.save(
            _pyramiding_frame(symbol=symbol),
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            symbol=symbol,
            timeframe=_TIMEFRAME,
            year=year,
        )
    work = discover_work(
        pyr_repository,
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


def test_discover_work_returns_empty_when_no_pyramiding_partitions(tmp_path: Path) -> None:
    """discover_work returns empty tuple when no pyramiding partitions exist."""
    layout = StorageLayout(tmp_path)
    pyr_repository = PyramidingRepository(layout, ParquetStore())
    work = discover_work(pyr_repository, _options(storage_root=tmp_path))
    assert work == ()


# ---------------------------------------------------------------------------
# format_summary
# ---------------------------------------------------------------------------


def test_format_summary_includes_core_fields() -> None:
    """format_summary renders manager, engine, and all exit-action/reason counts."""
    summary = ExitEngineGenerationSummary(
        manager=_MANAGER,
        engine=_ENGINE,
        symbols_discovered=2,
        symbols_processed=2,
        timeframes_processed=1,
        successful_tasks=3,
        failed_tasks=1,
        skipped_tasks=0,
        rows_generated=10,
        hold_count=5,
        partial_exit_count=3,
        full_exit_count=2,
        take_profit_count=3,
        trailing_stop_count=1,
        break_even_count=1,
        portfolio_shutdown_count=0,
        duration_seconds=2.5,
        output_directory=Path("data") / STORAGE_DIR_EXIT_ENGINE,
        failed_task_labels=("BTCUSDT 1h 2026",),
    )
    text = format_summary(summary)
    assert "CQROS Exit Engine Generation Summary" in text
    assert f"Manager: {_MANAGER}" in text
    assert f"Engine: {_ENGINE}" in text
    assert "HOLD: 5" in text
    assert "PARTIAL_EXIT: 3" in text
    assert "FULL_EXIT: 2" in text
    assert "TAKE_PROFIT: 3" in text
    assert "TRAILING_STOP: 1" in text
    assert "BREAK_EVEN: 1" in text
    assert "PORTFOLIO_SHUTDOWN: 0" in text
    assert "Failed Tasks" in text
    assert "BTCUSDT 1h 2026" in text
    assert STORAGE_DIR_EXIT_ENGINE in text


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
            pipeline=ExitEnginePipeline(registry),
            accounting_repository=AccountingRepository(layout, datastore),
            position_repository=PositionRepository(layout, datastore),
            portfolio_risk_repository=PortfolioRiskRepository(layout, datastore),
            trade_management_repository=TradeManagementRepository(layout, datastore),
            pyramiding_repository=PyramidingRepository(layout, datastore),
            exit_repository=ExitRepository(layout, datastore),
            options=options,
            work=(),
        )
    )
    assert summary.successful_tasks == 0
    assert summary.failed_tasks == 0
    assert summary.skipped_tasks == 0
    assert summary.rows_generated == 0
    assert summary.hold_count == 0
    assert summary.partial_exit_count == 0
    assert summary.full_exit_count == 0
    assert summary.take_profit_count == 0
    assert summary.output_directory == tmp_path / STORAGE_DIR_EXIT_ENGINE


# ---------------------------------------------------------------------------
# run_generation — persists and counts exit action/reason events
# ---------------------------------------------------------------------------


def test_run_generation_persists_and_counts_exit_actions(tmp_path: Path) -> None:
    """Generation loads inputs, runs the pipeline, persists, and counts exit actions."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()

    # BTCUSDT: price=115, entry=100, RR=(115-100)/(100*0.05)=3.0 → PARTIAL_EXIT/TAKE_PROFIT
    _seed_generation_inputs(
        layout=layout,
        datastore=datastore,
        symbol="BTCUSDT",
        current_price=115.0,
    )
    # ETHUSDT: price=102, entry=100, RR=0.4 → HOLD/NONE
    _seed_generation_inputs(
        layout=layout,
        datastore=datastore,
        symbol="ETHUSDT",
        current_price=102.0,
    )

    options = _options(storage_root=tmp_path)
    pyr_repository = PyramidingRepository(layout, datastore)
    work = discover_work(pyr_repository, options)

    registry = ExitEngineRegistry()
    registry.register(_ENGINE, SimpleExitEngine())
    summary = _run(
        run_generation(
            pipeline=ExitEnginePipeline(registry),
            accounting_repository=AccountingRepository(layout, datastore),
            position_repository=PositionRepository(layout, datastore),
            portfolio_risk_repository=PortfolioRiskRepository(layout, datastore),
            trade_management_repository=TradeManagementRepository(layout, datastore),
            pyramiding_repository=pyr_repository,
            exit_repository=ExitRepository(layout, datastore),
            options=options,
            work=work,
        )
    )
    assert summary.successful_tasks == 2
    assert summary.failed_tasks == 0
    assert summary.rows_generated == 2
    assert summary.partial_exit_count == 1
    assert summary.take_profit_count == 1
    assert summary.hold_count == 1

    exit_repo = ExitRepository(layout, datastore)
    assert exit_repo.exists(
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol="BTCUSDT",
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert exit_repo.exists(
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
    """Existing exit-engine partitions are skipped when overwrite is False."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    _seed_generation_inputs(layout=layout, datastore=datastore, current_price=115.0)

    options = _options(storage_root=tmp_path, overwrite=False)
    pyr_repository = PyramidingRepository(layout, datastore)
    work = discover_work(pyr_repository, options)

    registry = build_registry()
    first_summary = _run(
        run_generation(
            pipeline=ExitEnginePipeline(registry),
            accounting_repository=AccountingRepository(layout, datastore),
            position_repository=PositionRepository(layout, datastore),
            portfolio_risk_repository=PortfolioRiskRepository(layout, datastore),
            trade_management_repository=TradeManagementRepository(layout, datastore),
            pyramiding_repository=pyr_repository,
            exit_repository=ExitRepository(layout, datastore),
            options=options,
            work=work,
        )
    )
    assert first_summary.successful_tasks == 1

    second_summary = _run(
        run_generation(
            pipeline=ExitEnginePipeline(build_registry()),
            accounting_repository=AccountingRepository(layout, datastore),
            position_repository=PositionRepository(layout, datastore),
            portfolio_risk_repository=PortfolioRiskRepository(layout, datastore),
            trade_management_repository=TradeManagementRepository(layout, datastore),
            pyramiding_repository=pyr_repository,
            exit_repository=ExitRepository(layout, datastore),
            options=options,
            work=work,
        )
    )
    assert second_summary.skipped_tasks == 1
    assert second_summary.successful_tasks == 0
    assert second_summary.failed_tasks == 0


# ---------------------------------------------------------------------------
# run_generation — full exit (trailing stop)
# ---------------------------------------------------------------------------


def test_run_generation_produces_full_exit_for_trailing_stop(tmp_path: Path) -> None:
    """Trailing stop action_reason produces FULL_EXIT rows in the output."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    _seed_generation_inputs(
        layout=layout,
        datastore=datastore,
        current_price=102.0,
        action_reason="TRAILING_STOP",
    )

    options = _options(storage_root=tmp_path)
    pyr_repository = PyramidingRepository(layout, datastore)
    work = discover_work(pyr_repository, options)

    summary = _run(
        run_generation(
            pipeline=ExitEnginePipeline(build_registry()),
            accounting_repository=AccountingRepository(layout, datastore),
            position_repository=PositionRepository(layout, datastore),
            portfolio_risk_repository=PortfolioRiskRepository(layout, datastore),
            trade_management_repository=TradeManagementRepository(layout, datastore),
            pyramiding_repository=pyr_repository,
            exit_repository=ExitRepository(layout, datastore),
            options=options,
            work=work,
        )
    )
    assert summary.successful_tasks == 1
    assert summary.full_exit_count == 1
    assert summary.trailing_stop_count == 1
