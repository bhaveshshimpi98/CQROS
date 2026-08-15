"""Unit tests for CQROS portfolio-risk-generation CLI."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

import cqros.cli.generate_portfolio_risk as generate_portfolio_risk_module
from cqros.accounting import (
    CANONICAL_COLUMN_ORDER as ACCOUNTING_COLUMNS,
)
from cqros.accounting import (
    COLUMN_DTYPES as ACCOUNTING_DTYPES,
)
from cqros.accounting import (
    AccountingRepository,
    PositionStatus,
)
from cqros.cli.generate_portfolio_risk import (
    DiscoveredWorkItem,
    PortfolioRiskGenerationOptions,
    PortfolioRiskGenerationSummary,
    build_default_manager,
    build_manager_registry,
    build_options,
    build_parser,
    build_portfolio_risk_pipeline,
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
)
from cqros.core.exceptions import ValidationError
from cqros.portfolio_risk import (
    PortfolioRiskManagerRegistry,
    PortfolioRiskPipeline,
    PortfolioRiskRepository,
    SimplePortfolioRiskManager,
)
from cqros.positions import (
    CANONICAL_COLUMN_ORDER as POSITION_COLUMNS,
)
from cqros.positions import (
    COLUMN_DTYPES as POSITION_DTYPES,
)
from cqros.positions import (
    PositionRepository,
)
from cqros.positions import (
    PositionStatus as PositionLifecycleStatus,
)
from cqros.storage import ParquetStore, StorageLayout

_MANAGER = "simple"
_RISK_MANAGER = "simple"
_MODEL = "alpha-lgbm"
_VERSION = "1.0.0"
_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1h"
_YEAR = 2026
_EQUITY = 1000.0


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def _options(
    *,
    storage_root: Path,
    manager: str = _MANAGER,
    risk_manager: str = _RISK_MANAGER,
    model: str | None = _MODEL,
    version: str | None = _VERSION,
    symbols: tuple[str, ...] | None = None,
    timeframes: tuple[str, ...] | None = None,
    years: tuple[int, ...] | None = None,
    overwrite: bool = False,
    workers: int = ResearchConfig().worker_count,
    verbose: bool = False,
    debug: bool = False,
) -> PortfolioRiskGenerationOptions:
    """Build options for tests against a temporary storage root."""
    return PortfolioRiskGenerationOptions(
        storage_root=storage_root,
        manager=manager,
        risk_manager=risk_manager,
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
    total_pnl: float = 0.0,
    gross_exposure: float = 500.0,
    equity: float = _EQUITY,
) -> pl.DataFrame:
    """Return a canonical accounting frame for generation input."""
    open_time = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
    return pl.DataFrame(
        {
            "symbol": [symbol],
            "timeframe": [_TIMEFRAME],
            "open_time": [open_time],
            "manager": [_MANAGER],
            "position_id": ["pos-00000001"],
            "position_status": [PositionStatus.OPEN.value],
            "quantity": [1.0],
            "average_entry_price": [100.0],
            "mark_price": [110.0],
            "position_value": [110.0],
            "market_value": [110.0],
            "cash": [0.0],
            "realized_pnl": [0.0],
            "unrealized_pnl": [0.0],
            "total_pnl": [total_pnl],
            "gross_exposure": [gross_exposure],
            "net_exposure": [gross_exposure],
            "equity": [equity],
            "return_pct": [total_pnl / equity if equity else 0.0],
            "model_name": [model_name],
            "model_version": [model_version],
            "optimizer": ["equal_weight"],
            "policy": ["fixed_risk"],
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
    opened_at = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
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
    """Parser requires --manager and defaults risk-manager to simple."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(["--manager", "simple"])
    assert args.manager == "simple"
    assert args.risk_manager == "simple"


def test_build_options_validates_workers_manager_and_risk_manager(tmp_path: Path) -> None:
    """build_options rejects non-positive workers and blank identities."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "simple", "--workers", "0"])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-PORTFOLIO-RISK-001"

    args = parser.parse_args(["--manager", "   "])
    with patch.object(generate_portfolio_risk_module, "DEFAULT_STORAGE_ROOT", tmp_path):
        with pytest.raises(ValidationError) as exc_info:
            build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-PORTFOLIO-RISK-006"

    args = parser.parse_args(["--manager", "simple", "--risk-manager", "   "])
    with patch.object(generate_portfolio_risk_module, "DEFAULT_STORAGE_ROOT", tmp_path):
        with pytest.raises(ValidationError) as exc_info:
            build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-PORTFOLIO-RISK-007"


def test_build_default_manager_and_registry() -> None:
    """Default registry contains SimplePortfolioRiskManager under simple."""
    assert isinstance(build_default_manager(), SimplePortfolioRiskManager)
    registry = build_manager_registry()
    assert registry.exists("simple")
    assert isinstance(registry.get("simple"), SimplePortfolioRiskManager)


def test_build_portfolio_risk_pipeline_wires_registry() -> None:
    """Pipeline composition uses the default manager registry."""
    options = _options(storage_root=Path(DEFAULT_STORAGE_ROOT))
    pipeline = build_portfolio_risk_pipeline(options)
    assert isinstance(pipeline, PortfolioRiskPipeline)


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
    """format_summary renders manager, risk events, and task counters."""
    summary = PortfolioRiskGenerationSummary(
        manager=_MANAGER,
        risk_manager=_RISK_MANAGER,
        version=_VERSION,
        symbols_discovered=2,
        symbols_processed=2,
        timeframes_processed=1,
        successful_tasks=1,
        failed_tasks=1,
        skipped_tasks=0,
        rows_generated=10,
        shutdown_events=3,
        cooldown_events=2,
        exposure_warnings=1,
        duration_seconds=1.5,
        output_directory=Path("data") / STORAGE_DIR_PORTFOLIO_RISK,
        failed_task_labels=("BTCUSDT 1h 2026",),
    )
    text = format_summary(summary)
    assert "CQROS Portfolio Risk Generation Summary" in text
    assert f"Manager: {_MANAGER}" in text
    assert f"Risk Manager: {_RISK_MANAGER}" in text
    assert "Shutdown events: 3" in text
    assert "Cooldown events: 2" in text
    assert "Exposure warnings: 1" in text
    assert "Failed Tasks" in text
    assert "BTCUSDT 1h 2026" in text
    assert STORAGE_DIR_PORTFOLIO_RISK in text


def test_run_generation_empty_work(tmp_path: Path) -> None:
    """Empty work produces a zeroed success summary."""
    options = _options(storage_root=tmp_path)
    pipeline = PortfolioRiskPipeline(build_manager_registry())
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    summary = _run(
        run_generation(
            pipeline=pipeline,
            accounting_repository=AccountingRepository(layout, datastore),
            position_repository=PositionRepository(layout, datastore),
            portfolio_risk_repository=PortfolioRiskRepository(layout, datastore),
            options=options,
            work=(),
        )
    )
    assert summary.successful_tasks == 0
    assert summary.failed_tasks == 0
    assert summary.rows_generated == 0
    assert summary.shutdown_events == 0
    assert summary.cooldown_events == 0
    assert summary.exposure_warnings == 0
    assert summary.output_directory == tmp_path / STORAGE_DIR_PORTFOLIO_RISK


def test_run_generation_persists_and_counts_events(tmp_path: Path) -> None:
    """Generation loads inputs, runs the pipeline, persists, and counts events."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    accounting_repository = AccountingRepository(layout, datastore)
    position_repository = PositionRepository(layout, datastore)
    portfolio_risk_repository = PortfolioRiskRepository(layout, datastore)

    # Daily-loss shutdown row (-2% of equity) and an exposure-warning row.
    accounting_shutdown = _accounting_frame(total_pnl=-20.0, gross_exposure=500.0)
    accounting_exposure = _accounting_frame(
        symbol="ETHUSDT",
        total_pnl=0.0,
        gross_exposure=1100.0,
    )
    accounting_repository.save(
        accounting_shutdown,
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    accounting_repository.save(
        accounting_exposure,
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol="ETHUSDT",
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    position_repository.save(
        _position_frame(),
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    position_repository.save(
        _position_frame(symbol="ETHUSDT"),
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol="ETHUSDT",
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )

    options = _options(storage_root=tmp_path, model=None, version=None)
    work = discover_work(accounting_repository, options)
    registry = PortfolioRiskManagerRegistry()
    registry.register(_RISK_MANAGER, SimplePortfolioRiskManager())
    summary = _run(
        run_generation(
            pipeline=PortfolioRiskPipeline(registry),
            accounting_repository=accounting_repository,
            position_repository=position_repository,
            portfolio_risk_repository=portfolio_risk_repository,
            options=options,
            work=work,
        )
    )
    assert summary.successful_tasks == 2
    assert summary.failed_tasks == 0
    assert summary.rows_generated == 2
    assert summary.shutdown_events == 1
    assert summary.cooldown_events == 0
    assert summary.exposure_warnings == 1
    assert portfolio_risk_repository.exists(
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


def test_run_generation_skips_existing_without_overwrite(tmp_path: Path) -> None:
    """Existing portfolio-risk partitions are skipped when overwrite is false."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    accounting_repository = AccountingRepository(layout, datastore)
    position_repository = PositionRepository(layout, datastore)
    portfolio_risk_repository = PortfolioRiskRepository(layout, datastore)

    accounting_repository.save(
        _accounting_frame(),
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    position_repository.save(
        _position_frame(),
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    existing = PortfolioRiskPipeline(build_manager_registry()).run(
        _accounting_frame(),
        _position_frame(),
        manager=_MANAGER,
    )
    portfolio_risk_repository.save(
        existing,
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    options = _options(storage_root=tmp_path, model=None, version=None, overwrite=False)
    work = discover_work(accounting_repository, options)
    summary = _run(
        run_generation(
            pipeline=PortfolioRiskPipeline(build_manager_registry()),
            accounting_repository=accounting_repository,
            position_repository=position_repository,
            portfolio_risk_repository=portfolio_risk_repository,
            options=options,
            work=work,
        )
    )
    assert summary.skipped_tasks == 1
    assert summary.successful_tasks == 0
    assert summary.failed_tasks == 0
