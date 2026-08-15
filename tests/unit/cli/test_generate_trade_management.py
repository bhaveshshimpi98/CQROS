"""Unit tests for CQROS trade-management-generation CLI."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

import cqros.cli.generate_trade_management as generate_trade_management_module
from cqros.accounting import CANONICAL_COLUMN_ORDER as ACCOUNTING_COLUMNS
from cqros.accounting import COLUMN_DTYPES as ACCOUNTING_DTYPES
from cqros.accounting import (
    AccountingRepository,
    PositionStatus,
)
from cqros.cli.generate_trade_management import (
    DiscoveredWorkItem,
    TradeManagementGenerationOptions,
    TradeManagementGenerationSummary,
    build_default_manager,
    build_manager_registry,
    build_options,
    build_parser,
    build_trade_management_pipeline,
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
    STORAGE_DIR_PORTFOLIO_RISK,
    STORAGE_DIR_POSITIONS,
    STORAGE_DIR_TRADE_MANAGEMENT,
)
from cqros.core.exceptions import ValidationError
from cqros.portfolio_risk import CANONICAL_COLUMN_ORDER as PORTFOLIO_RISK_COLUMNS
from cqros.portfolio_risk import COLUMN_DTYPES as PORTFOLIO_RISK_DTYPES
from cqros.portfolio_risk import (
    PortfolioRiskRepository,
    PortfolioRiskState,
)
from cqros.portfolio_risk import ShutdownReason as PortfolioRiskShutdownReason
from cqros.positions import CANONICAL_COLUMN_ORDER as POSITION_COLUMNS
from cqros.positions import COLUMN_DTYPES as POSITION_DTYPES
from cqros.positions import (
    PositionRepository,
)
from cqros.positions import PositionStatus as PositionLifecycleStatus
from cqros.storage import ParquetStore, ProcessedMarketDataRepository, StorageLayout
from cqros.trade_management import (
    SimpleTradeManagementManager,
    TradeManagementManagerRegistry,
    TradeManagementPipeline,
    TradeManagementRepository,
)

_MANAGER = "simple"
_TRADE_MANAGER = "simple"
_MODEL = "alpha-lgbm"
_VERSION = "1.0.0"
_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1h"
_YEAR = 2026
_OPEN_TIME = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def _options(
    *,
    storage_root: Path,
    manager: str = _MANAGER,
    trade_manager: str = _TRADE_MANAGER,
    model: str | None = _MODEL,
    version: str | None = _VERSION,
    symbols: tuple[str, ...] | None = None,
    timeframes: tuple[str, ...] | None = None,
    years: tuple[int, ...] | None = None,
    overwrite: bool = False,
    workers: int = ResearchConfig().worker_count,
    verbose: bool = False,
    debug: bool = False,
) -> TradeManagementGenerationOptions:
    """Build options for tests against a temporary storage root."""
    return TradeManagementGenerationOptions(
        storage_root=storage_root,
        manager=manager,
        trade_manager=trade_manager,
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
    model_name: str = _MODEL,
    model_version: str = _VERSION,
    open_times: list[datetime] | None = None,
    position_ids: list[str] | None = None,
    entry_price: float = 100.0,
) -> pl.DataFrame:
    """Return a canonical accounting frame for generation input."""
    open_times = open_times if open_times is not None else [_OPEN_TIME]
    position_ids = position_ids if position_ids is not None else ["pos-00000001"]
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
            "average_entry_price": [entry_price] * row_count,
            "mark_price": [110.0] * row_count,
            "position_value": [110.0] * row_count,
            "market_value": [110.0] * row_count,
            "cash": [0.0] * row_count,
            "realized_pnl": [0.0] * row_count,
            "unrealized_pnl": [0.0] * row_count,
            "total_pnl": [0.0] * row_count,
            "gross_exposure": [500.0] * row_count,
            "net_exposure": [500.0] * row_count,
            "equity": [1000.0] * row_count,
            "return_pct": [0.0] * row_count,
            "model_name": [model_name] * row_count,
            "model_version": [model_version] * row_count,
            "optimizer": ["equal_weight"] * row_count,
            "policy": ["fixed_risk"] * row_count,
        },
        schema=dict(ACCOUNTING_DTYPES),
    ).select(list(ACCOUNTING_COLUMNS))


def _position_frame(
    *,
    symbol: str = _SYMBOL,
    model_name: str = _MODEL,
    model_version: str = _VERSION,
) -> pl.DataFrame:
    """Return a canonical position frame for generation input."""
    return pl.DataFrame(
        {
            "symbol": [symbol],
            "timeframe": [_TIMEFRAME],
            "position_id": ["pos-00000001"],
            "side": ["LONG"],
            "status": [PositionLifecycleStatus.OPEN.value],
            "quantity": [1.0],
            "average_entry_price": [100.0],
            "market_price": [110.0],
            "realized_pnl": [0.0],
            "unrealized_pnl": [0.0],
            "fees_paid": [0.0],
            "opened_at": [_OPEN_TIME],
            "updated_at": [_OPEN_TIME],
            "closed_at": [None],
            "model_name": [model_name],
            "model_version": [model_version],
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
) -> pl.DataFrame:
    """Return a canonical portfolio-risk frame aligned with accounting rows."""
    open_times = open_times if open_times is not None else [_OPEN_TIME]
    position_ids = position_ids if position_ids is not None else ["pos-00000001"]
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
            "portfolio_risk_state": [PortfolioRiskState.NORMAL.value] * row_count,
            "allow_new_entries": [True] * row_count,
            "shutdown_reason": [PortfolioRiskShutdownReason.NONE.value] * row_count,
            "cooldown_until": [None] * row_count,
            "model_name": [_MODEL] * row_count,
            "model_version": [_VERSION] * row_count,
            "optimizer": ["equal_weight"] * row_count,
            "policy": ["fixed_risk"] * row_count,
        },
        schema=dict(PORTFOLIO_RISK_DTYPES),
    ).select(list(PORTFOLIO_RISK_COLUMNS))


def _ohlcv_frame(
    *,
    symbol: str = _SYMBOL,
    open_times: list[datetime] | None = None,
    closes: list[float] | None = None,
) -> pl.DataFrame:
    """Return a processed OHLCV frame with UTC open_time and close prices."""
    open_times = open_times if open_times is not None else [_OPEN_TIME]
    closes = closes if closes is not None else [104.0]
    row_count = len(open_times)
    return pl.DataFrame(
        {
            "symbol": [symbol] * row_count,
            "timeframe": [_TIMEFRAME] * row_count,
            "open_time": open_times,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [10.0] * row_count,
            "quote_volume": [15.0] * row_count,
            "trade_count": [1] * row_count,
            "close_time": open_times,
        }
    )


def _seed_generation_inputs(
    *,
    layout: StorageLayout,
    datastore: ParquetStore,
    accounting: pl.DataFrame,
    symbol: str = _SYMBOL,
    open_times: list[datetime] | None = None,
    closes: list[float] | None = None,
) -> None:
    """Persist accounting, positions, portfolio risk, and OHLCV for one symbol."""
    open_times = open_times if open_times is not None else [_OPEN_TIME]
    position_ids = accounting["position_id"].to_list()
    AccountingRepository(layout, datastore).save(
        accounting,
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
        _portfolio_risk_frame(symbol=symbol, open_times=open_times, position_ids=position_ids),
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=symbol,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    ProcessedMarketDataRepository(layout, datastore).save_ohlcv(
        _ohlcv_frame(symbol=symbol, open_times=open_times, closes=closes),
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=symbol,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )


def test_build_parser_requires_manager() -> None:
    """Parser requires --manager and defaults trade-manager to simple."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(["--manager", "simple"])
    assert args.manager == "simple"
    assert args.trade_manager == "simple"


def test_build_options_validates_workers_manager_and_trade_manager(tmp_path: Path) -> None:
    """build_options rejects non-positive workers and blank identities."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "simple", "--workers", "0"])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-TRADE-MANAGEMENT-001"

    args = parser.parse_args(["--manager", "   "])
    with patch.object(generate_trade_management_module, "DEFAULT_STORAGE_ROOT", tmp_path):
        with pytest.raises(ValidationError) as exc_info:
            build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-TRADE-MANAGEMENT-006"

    args = parser.parse_args(["--manager", "simple", "--trade-manager", "   "])
    with patch.object(generate_trade_management_module, "DEFAULT_STORAGE_ROOT", tmp_path):
        with pytest.raises(ValidationError) as exc_info:
            build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-TRADE-MANAGEMENT-007"


def test_build_default_manager_and_registry() -> None:
    """Default registry contains SimpleTradeManagementManager under simple."""
    assert isinstance(build_default_manager(), SimpleTradeManagementManager)
    registry = build_manager_registry()
    assert registry.exists("simple")
    assert isinstance(registry.get("simple"), SimpleTradeManagementManager)


def test_build_trade_management_pipeline_wires_registry() -> None:
    """Pipeline composition uses the default manager registry."""
    options = _options(storage_root=Path(DEFAULT_STORAGE_ROOT))
    pipeline = build_trade_management_pipeline(options)
    assert isinstance(pipeline, TradeManagementPipeline)


def test_discover_work_groups_accounting_partitions(tmp_path: Path) -> None:
    """discover_work returns sorted manager/symbol/timeframe work items."""
    layout = StorageLayout(tmp_path)
    accounting_repository = AccountingRepository(layout, ParquetStore())
    frame = _accounting_frame()
    for symbol, year in (("ETHUSDT", 2025), ("BTCUSDT", 2026), ("BTCUSDT", 2025)):
        accounting_repository.save(
            frame,
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            symbol=symbol,
            timeframe=_TIMEFRAME,
            year=year,
        )
    work = discover_work(
        accounting_repository,
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
    """format_summary renders manager, management events, and task counters."""
    summary = TradeManagementGenerationSummary(
        manager=_MANAGER,
        trade_manager=_TRADE_MANAGER,
        symbols_discovered=2,
        symbols_processed=2,
        timeframes_processed=1,
        successful_tasks=1,
        failed_tasks=1,
        skipped_tasks=0,
        rows_generated=10,
        trailing_stop_updates=3,
        breakeven_updates=2,
        duration_seconds=1.5,
        output_directory=Path("data") / STORAGE_DIR_TRADE_MANAGEMENT,
        failed_task_labels=("BTCUSDT 1h 2026",),
    )
    text = format_summary(summary)
    assert "CQROS Trade Management Generation Summary" in text
    assert f"Manager: {_MANAGER}" in text
    assert f"Trade Manager: {_TRADE_MANAGER}" in text
    assert "Trailing stop updates: 3" in text
    assert "Breakeven updates: 2" in text
    assert "Failed Tasks" in text
    assert "BTCUSDT 1h 2026" in text
    assert STORAGE_DIR_TRADE_MANAGEMENT in text


def test_run_generation_empty_work(tmp_path: Path) -> None:
    """Empty work produces a zeroed success summary."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    options = _options(storage_root=tmp_path)
    pipeline = TradeManagementPipeline(build_manager_registry())
    summary = _run(
        run_generation(
            pipeline=pipeline,
            layout=layout,
            datastore=datastore,
            accounting_repository=AccountingRepository(layout, datastore),
            position_repository=PositionRepository(layout, datastore),
            portfolio_risk_repository=PortfolioRiskRepository(layout, datastore),
            processed_market_data_repository=ProcessedMarketDataRepository(layout, datastore),
            trade_management_repository=TradeManagementRepository(layout, datastore),
            options=options,
            work=(),
        )
    )
    assert summary.successful_tasks == 0
    assert summary.failed_tasks == 0
    assert summary.rows_generated == 0
    assert summary.trailing_stop_updates == 0
    assert summary.breakeven_updates == 0
    assert summary.output_directory == tmp_path / STORAGE_DIR_TRADE_MANAGEMENT


def test_run_generation_persists_and_counts_events(tmp_path: Path) -> None:
    """Generation loads inputs, runs the pipeline, persists, and counts events."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    trailing_open_times = [
        _OPEN_TIME,
        datetime(2024, 6, 15, 13, 0, 0, tzinfo=UTC),
    ]
    _seed_generation_inputs(
        layout=layout,
        datastore=datastore,
        accounting=_accounting_frame(
            open_times=trailing_open_times,
            position_ids=["pos-00000001", "pos-00000001"],
        ),
        closes=[120.0, 110.0],
        open_times=trailing_open_times,
    )
    _seed_generation_inputs(
        layout=layout,
        datastore=datastore,
        accounting=_accounting_frame(symbol="ETHUSDT"),
        symbol="ETHUSDT",
        closes=[106.0],
    )

    options = _options(storage_root=tmp_path, model=None, version=None)
    work = discover_work(AccountingRepository(layout, datastore), options)
    registry = TradeManagementManagerRegistry()
    registry.register(_TRADE_MANAGER, SimpleTradeManagementManager())
    summary = _run(
        run_generation(
            pipeline=TradeManagementPipeline(registry),
            layout=layout,
            datastore=datastore,
            accounting_repository=AccountingRepository(layout, datastore),
            position_repository=PositionRepository(layout, datastore),
            portfolio_risk_repository=PortfolioRiskRepository(layout, datastore),
            processed_market_data_repository=ProcessedMarketDataRepository(layout, datastore),
            trade_management_repository=TradeManagementRepository(layout, datastore),
            options=options,
            work=work,
        )
    )
    assert summary.successful_tasks == 2
    assert summary.failed_tasks == 0
    assert summary.rows_generated == 3
    assert summary.trailing_stop_updates == 1
    assert summary.breakeven_updates == 2
    trade_management_repository = TradeManagementRepository(layout, datastore)
    assert trade_management_repository.exists(
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert STORAGE_DIR_ACCOUNTING in str(tmp_path / STORAGE_DIR_ACCOUNTING)
    assert STORAGE_DIR_POSITIONS in str(tmp_path / STORAGE_DIR_POSITIONS)
    assert STORAGE_DIR_PORTFOLIO_RISK in str(tmp_path / STORAGE_DIR_PORTFOLIO_RISK)
    assert STORAGE_DIR_TRADE_MANAGEMENT in str(tmp_path / STORAGE_DIR_TRADE_MANAGEMENT)


def test_run_generation_skips_existing_without_overwrite(tmp_path: Path) -> None:
    """Existing trade-management partitions are skipped when overwrite is false."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    _seed_generation_inputs(
        layout=layout,
        datastore=datastore,
        accounting=_accounting_frame(),
    )
    existing = TradeManagementPipeline(build_manager_registry()).run(
        PositionRepository(layout, datastore).load(
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        ),
        _accounting_frame(),
        PortfolioRiskRepository(layout, datastore).load(
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        ),
        ProcessedMarketDataRepository(layout, datastore)
        .load_ohlcv(
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )
        .select(
            pl.col("symbol"),
            pl.col("timeframe"),
            pl.col("open_time"),
            pl.col("close").alias("price"),
        ),
        manager=_MANAGER,
    )
    TradeManagementRepository(layout, datastore).save(
        existing,
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    options = _options(storage_root=tmp_path, model=None, version=None, overwrite=False)
    work = discover_work(AccountingRepository(layout, datastore), options)
    summary = _run(
        run_generation(
            pipeline=TradeManagementPipeline(build_manager_registry()),
            layout=layout,
            datastore=datastore,
            accounting_repository=AccountingRepository(layout, datastore),
            position_repository=PositionRepository(layout, datastore),
            portfolio_risk_repository=PortfolioRiskRepository(layout, datastore),
            processed_market_data_repository=ProcessedMarketDataRepository(layout, datastore),
            trade_management_repository=TradeManagementRepository(layout, datastore),
            options=options,
            work=work,
        )
    )
    assert summary.skipped_tasks == 1
    assert summary.successful_tasks == 0
    assert summary.failed_tasks == 0
