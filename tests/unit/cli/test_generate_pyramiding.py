"""Unit tests for CQROS pyramiding-generation CLI."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

import cqros.cli.generate_pyramiding as generate_pyramiding_module
from cqros.accounting import CANONICAL_COLUMN_ORDER as ACCOUNTING_COLUMNS
from cqros.accounting import COLUMN_DTYPES as ACCOUNTING_DTYPES
from cqros.accounting import (
    AccountingRepository,
    PositionStatus,
)
from cqros.cli.generate_pyramiding import (
    DiscoveredWorkItem,
    PyramidingGenerationOptions,
    PyramidingGenerationSummary,
    build_default_engine,
    build_options,
    build_parser,
    build_pyramiding_pipeline,
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
    STORAGE_DIR_PYRAMIDING,
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
from cqros.positions import PositionRepository
from cqros.positions import PositionStatus as PositionLifecycleStatus
from cqros.pyramiding import (
    PyramidingPipeline,
    PyramidingRegistry,
    PyramidingRepository,
    SimplePyramidingEngine,
)
from cqros.storage import ParquetStore, ProcessedMarketDataRepository, StorageLayout
from cqros.trade_management import CANONICAL_COLUMN_ORDER as TM_COLUMNS
from cqros.trade_management import COLUMN_DTYPES as TM_DTYPES
from cqros.trade_management import ManagementAction, TradeManagementRepository

_MANAGER = "simple"
_ENGINE = "simple"
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
) -> PyramidingGenerationOptions:
    """Build PyramidingGenerationOptions against a temporary storage root."""
    return PyramidingGenerationOptions(
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
    model_name: str = _MODEL,
    model_version: str = _VERSION,
    open_times: list[datetime] | None = None,
    position_ids: list[str] | None = None,
    entry_price: float = 100.0,
) -> pl.DataFrame:
    """Return a canonical accounting frame aligned to pyramiding inputs."""
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


def _trade_management_frame(
    *,
    symbol: str = _SYMBOL,
    open_times: list[datetime] | None = None,
    position_ids: list[str] | None = None,
) -> pl.DataFrame:
    """Return a canonical trade-management frame for generation input."""
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
            "position_status": ["OPEN"] * row_count,
            "quantity": [1.0] * row_count,
            "entry_price": [100.0] * row_count,
            "current_price": [106.0] * row_count,
            "highest_price": [106.0] * row_count,
            "lowest_price": [100.0] * row_count,
            "unrealized_pnl": [6.0] * row_count,
            "risk_state": ["NORMAL"] * row_count,
            "management_action": [ManagementAction.NONE.value] * row_count,
            "action_reason": ["NONE"] * row_count,
            "stop_price": [None] * row_count,
            "take_profit_price": [None] * row_count,
            "trail_price": [95.0] * row_count,
            "breakeven_price": [None] * row_count,
            "allow_pyramid": [False] * row_count,
            "exit_quantity": [0.0] * row_count,
            "model_name": [_MODEL] * row_count,
            "model_version": [_VERSION] * row_count,
            "optimizer": ["equal_weight"] * row_count,
            "policy": ["fixed_risk"] * row_count,
        },
        schema=dict(TM_DTYPES),
    ).select(list(TM_COLUMNS))


def _ohlcv_frame(
    *,
    symbol: str = _SYMBOL,
    open_times: list[datetime] | None = None,
    closes: list[float] | None = None,
    highs: list[float] | None = None,
) -> pl.DataFrame:
    """Return a processed OHLCV frame with UTC open_time, close, and high prices."""
    open_times = open_times if open_times is not None else [_OPEN_TIME]
    closes = closes if closes is not None else [106.0]
    highs = highs if highs is not None else closes
    row_count = len(open_times)
    return pl.DataFrame(
        {
            "symbol": [symbol] * row_count,
            "timeframe": [_TIMEFRAME] * row_count,
            "open_time": open_times,
            "open": closes,
            "high": highs,
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
    highs: list[float] | None = None,
) -> None:
    """Persist all inputs needed for pyramiding generation for one symbol."""
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
    TradeManagementRepository(layout, datastore).save(
        _trade_management_frame(symbol=symbol, open_times=open_times, position_ids=position_ids),
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=symbol,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    ProcessedMarketDataRepository(layout, datastore).save_ohlcv(
        _ohlcv_frame(symbol=symbol, open_times=open_times, closes=closes, highs=highs),
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


def test_build_options_rejects_non_positive_workers(tmp_path: Path) -> None:
    """build_options rejects workers <= 0."""
    parser = build_parser()
    args = parser.parse_args(["--workers", "0"])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-PYRAMIDING-001"


def test_build_options_rejects_blank_manager(tmp_path: Path) -> None:
    """build_options rejects a blank manager identity."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "   "])
    with patch.object(generate_pyramiding_module, "DEFAULT_STORAGE_ROOT", tmp_path):
        with pytest.raises(ValidationError) as exc_info:
            build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-PYRAMIDING-006"


def test_build_options_rejects_blank_engine(tmp_path: Path) -> None:
    """build_options rejects a blank engine identity."""
    parser = build_parser()
    args = parser.parse_args(["--engine", "   "])
    with patch.object(generate_pyramiding_module, "DEFAULT_STORAGE_ROOT", tmp_path):
        with pytest.raises(ValidationError) as exc_info:
            build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-PYRAMIDING-007"


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def test_build_default_engine_and_registry() -> None:
    """Default registry contains SimplePyramidingEngine under simple."""
    assert isinstance(build_default_engine(), SimplePyramidingEngine)
    registry = build_registry()
    assert registry.exists("simple")
    assert isinstance(registry.get("simple"), SimplePyramidingEngine)


def test_build_pyramiding_pipeline_wires_registry() -> None:
    """Pipeline composition uses the default engine registry."""
    options = _options(storage_root=Path(DEFAULT_STORAGE_ROOT))
    pipeline = build_pyramiding_pipeline(options)
    assert isinstance(pipeline, PyramidingPipeline)


# ---------------------------------------------------------------------------
# discover_work
# ---------------------------------------------------------------------------


def test_discover_work_groups_trade_management_partitions(tmp_path: Path) -> None:
    """discover_work returns sorted manager/symbol/timeframe work items from TM tier."""
    layout = StorageLayout(tmp_path)
    tm_repository = TradeManagementRepository(layout, ParquetStore())
    frame = _trade_management_frame()
    for symbol, year in (("ETHUSDT", 2025), ("BTCUSDT", 2026), ("BTCUSDT", 2025)):
        tm_repository.save(
            frame,
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            symbol=symbol,
            timeframe=_TIMEFRAME,
            year=year,
        )
    work = discover_work(
        tm_repository,
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


# ---------------------------------------------------------------------------
# format_summary
# ---------------------------------------------------------------------------


def test_format_summary_includes_core_fields() -> None:
    """format_summary renders manager, engine, READY_TO_ADD / NOT_ELIGIBLE / MAX_ADDS."""
    summary = PyramidingGenerationSummary(
        manager=_MANAGER,
        engine=_ENGINE,
        symbols_discovered=2,
        symbols_processed=2,
        timeframes_processed=1,
        successful_tasks=1,
        failed_tasks=1,
        skipped_tasks=0,
        rows_generated=10,
        ready_to_add=3,
        not_eligible=5,
        max_adds_reached=2,
        duration_seconds=1.5,
        output_directory=Path("data") / STORAGE_DIR_PYRAMIDING,
        failed_task_labels=("BTCUSDT 1h 2026",),
    )
    text = format_summary(summary)
    assert "CQROS Pyramiding Generation Summary" in text
    assert f"Manager: {_MANAGER}" in text
    assert f"Engine: {_ENGINE}" in text
    assert "READY_TO_ADD: 3" in text
    assert "NOT_ELIGIBLE: 5" in text
    assert "MAX_ADDS_REACHED: 2" in text
    assert "Failed Tasks" in text
    assert "BTCUSDT 1h 2026" in text
    assert STORAGE_DIR_PYRAMIDING in text


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
            pipeline=PyramidingPipeline(registry),
            layout=layout,
            datastore=datastore,
            accounting_repository=AccountingRepository(layout, datastore),
            position_repository=PositionRepository(layout, datastore),
            portfolio_risk_repository=PortfolioRiskRepository(layout, datastore),
            processed_market_data_repository=ProcessedMarketDataRepository(layout, datastore),
            trade_management_repository=TradeManagementRepository(layout, datastore),
            pyramiding_repository=PyramidingRepository(layout, datastore),
            options=options,
            work=(),
        )
    )
    assert summary.successful_tasks == 0
    assert summary.failed_tasks == 0
    assert summary.rows_generated == 0
    assert summary.ready_to_add == 0
    assert summary.not_eligible == 0
    assert summary.max_adds_reached == 0
    assert summary.output_directory == tmp_path / STORAGE_DIR_PYRAMIDING


# ---------------------------------------------------------------------------
# run_generation — persists and counts reason events
# ---------------------------------------------------------------------------


def test_run_generation_persists_and_counts_reasons(tmp_path: Path) -> None:
    """Generation loads inputs, runs the pipeline, persists, and counts reasons."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()

    # BTCUSDT: price=106, profit=6% > 5% threshold → READY_TO_ADD
    _seed_generation_inputs(
        layout=layout,
        datastore=datastore,
        accounting=_accounting_frame(),
        closes=[106.0],
        highs=[106.0],
    )
    # ETHUSDT: price=102, profit=2% < 5% threshold → INSUFFICIENT_PROFIT
    _seed_generation_inputs(
        layout=layout,
        datastore=datastore,
        accounting=_accounting_frame(symbol="ETHUSDT"),
        symbol="ETHUSDT",
        closes=[102.0],
        highs=[102.0],
    )

    options = _options(storage_root=tmp_path, model=None, version=None)
    tm_repository = TradeManagementRepository(layout, datastore)
    work = discover_work(tm_repository, options)

    registry = PyramidingRegistry()
    registry.register(_ENGINE, SimplePyramidingEngine())
    summary = _run(
        run_generation(
            pipeline=PyramidingPipeline(registry),
            layout=layout,
            datastore=datastore,
            accounting_repository=AccountingRepository(layout, datastore),
            position_repository=PositionRepository(layout, datastore),
            portfolio_risk_repository=PortfolioRiskRepository(layout, datastore),
            processed_market_data_repository=ProcessedMarketDataRepository(layout, datastore),
            trade_management_repository=tm_repository,
            pyramiding_repository=PyramidingRepository(layout, datastore),
            options=options,
            work=work,
        )
    )
    assert summary.successful_tasks == 2
    assert summary.failed_tasks == 0
    assert summary.rows_generated == 2
    assert summary.ready_to_add == 1
    assert summary.not_eligible == 0
    assert summary.max_adds_reached == 0

    pyr_repository = PyramidingRepository(layout, datastore)
    assert pyr_repository.exists(
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert STORAGE_DIR_PYRAMIDING in str(tmp_path / STORAGE_DIR_PYRAMIDING)


# ---------------------------------------------------------------------------
# run_generation — skip existing without overwrite
# ---------------------------------------------------------------------------


def test_run_generation_skips_existing_without_overwrite(tmp_path: Path) -> None:
    """Existing pyramiding partitions are skipped when overwrite is False."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    _seed_generation_inputs(
        layout=layout,
        datastore=datastore,
        accounting=_accounting_frame(),
        closes=[106.0],
        highs=[106.0],
    )

    options_no_model = _options(storage_root=tmp_path, model=None, version=None, overwrite=False)
    tm_repository = TradeManagementRepository(layout, datastore)
    work = discover_work(tm_repository, options_no_model)

    registry = build_registry()
    first_summary = _run(
        run_generation(
            pipeline=PyramidingPipeline(registry),
            layout=layout,
            datastore=datastore,
            accounting_repository=AccountingRepository(layout, datastore),
            position_repository=PositionRepository(layout, datastore),
            portfolio_risk_repository=PortfolioRiskRepository(layout, datastore),
            processed_market_data_repository=ProcessedMarketDataRepository(layout, datastore),
            trade_management_repository=tm_repository,
            pyramiding_repository=PyramidingRepository(layout, datastore),
            options=options_no_model,
            work=work,
        )
    )
    assert first_summary.successful_tasks == 1

    second_summary = _run(
        run_generation(
            pipeline=PyramidingPipeline(build_registry()),
            layout=layout,
            datastore=datastore,
            accounting_repository=AccountingRepository(layout, datastore),
            position_repository=PositionRepository(layout, datastore),
            portfolio_risk_repository=PortfolioRiskRepository(layout, datastore),
            processed_market_data_repository=ProcessedMarketDataRepository(layout, datastore),
            trade_management_repository=tm_repository,
            pyramiding_repository=PyramidingRepository(layout, datastore),
            options=options_no_model,
            work=work,
        )
    )
    assert second_summary.skipped_tasks == 1
    assert second_summary.successful_tasks == 0
    assert second_summary.failed_tasks == 0
