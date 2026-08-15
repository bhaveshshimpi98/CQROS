"""Unit tests for CQROS analytics generation CLI."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

import cqros.cli.generate_analytics as generate_analytics_module
from cqros.analytics import (
    AnalyticsEngineRegistry,
    AnalyticsPipeline,
    AnalyticsRepository,
    SimpleAnalyticsEngine,
)
from cqros.cli.generate_analytics import (
    AnalyticsGenerationOptions,
    AnalyticsGenerationSummary,
    DiscoveredWorkItem,
    build_analytics_pipeline,
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
    STORAGE_DIR_ANALYTICS,
)
from cqros.core.exceptions import ValidationError
from cqros.performance import PerformanceRepository, PerformanceStatus
from cqros.performance.schema import CANONICAL_COLUMN_ORDER, COLUMN_DTYPES
from cqros.storage import ParquetStore, StorageLayout

_MANAGER = "simple"
_ENGINE = "simple"
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
    symbols: tuple[str, ...] | None = None,
    timeframes: tuple[str, ...] | None = None,
    years: tuple[int, ...] | None = None,
    overwrite: bool = False,
    workers: int = ResearchConfig().worker_count,
    verbose: bool = False,
    debug: bool = False,
) -> AnalyticsGenerationOptions:
    """Build AnalyticsGenerationOptions against a temporary storage root."""
    return AnalyticsGenerationOptions(
        storage_root=storage_root,
        manager=manager,
        engine=engine,
        symbols=symbols,
        timeframes=timeframes,
        years=years,
        overwrite=overwrite,
        workers=workers,
        verbose=verbose,
        debug=debug,
    )


def _performance_frame(
    *,
    symbol: str = _SYMBOL,
    total_return: float = 0.05,
    max_drawdown: float = 0.02,
) -> pl.DataFrame:
    """Return a canonical performance frame for analytics generation tests."""
    return pl.DataFrame(
        {
            "symbol": [symbol],
            "timeframe": [_TIMEFRAME],
            "open_time": [_OPEN_TIME],
            "manager": [_MANAGER],
            "total_return": [total_return],
            "cagr": [0.0],
            "volatility": [0.1],
            "downside_volatility": [0.0],
            "max_drawdown": [max_drawdown],
            "drawdown_duration": [0],
            "sharpe_ratio": [1.0],
            "sortino_ratio": [1.2],
            "calmar_ratio": [None],
            "total_trades": [0],
            "winning_trades": [0],
            "losing_trades": [0],
            "win_rate": [0.0],
            "average_win": [None],
            "average_loss": [None],
            "profit_factor": [None],
            "expectancy": [0.0],
            "starting_equity": [10000.0],
            "ending_equity": [10500.0],
            "net_profit": [500.0],
            "gross_profit": [0.0],
            "gross_loss": [0.0],
            "first_trade_time": [None],
            "last_trade_time": [None],
            "status": [PerformanceStatus.FINISHED.value],
        },
        schema=dict(COLUMN_DTYPES),
    ).select(list(CANONICAL_COLUMN_ORDER))


def _seed_generation_inputs(
    *,
    layout: StorageLayout,
    datastore: ParquetStore,
    symbol: str = _SYMBOL,
    total_return: float = 0.05,
) -> None:
    """Persist a performance partition needed for analytics generation."""
    PerformanceRepository(layout, datastore).save(
        _performance_frame(symbol=symbol, total_return=total_return),
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
            "--verbose",
            "--debug",
            "--storage-root",
            str(tmp_path),
        ]
    )
    assert args.manager == "ledger"
    assert args.engine == "custom"
    assert args.workers == 4
    assert args.overwrite is True
    assert args.verbose is True
    assert args.debug is True
    assert args.storage_root == tmp_path


def test_build_options_rejects_non_positive_workers() -> None:
    """build_options rejects workers <= 0 with CLI-GENERATE-ANALYTICS-001."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "simple", "--workers", "0"])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-ANALYTICS-001"


def test_build_options_rejects_blank_manager() -> None:
    """build_options rejects a blank manager identity with CLI-GENERATE-ANALYTICS-004."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "   "])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-ANALYTICS-004"


def test_build_options_rejects_blank_engine() -> None:
    """build_options rejects a blank engine identity with CLI-GENERATE-ANALYTICS-005."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "simple", "--engine", "   "])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-ANALYTICS-005"


def test_build_options_uses_storage_root(tmp_path: Path) -> None:
    """build_options honors an explicit storage root override."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "simple", "--storage-root", str(tmp_path)])
    options = build_options(args)
    assert options.storage_root == tmp_path


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def test_build_default_engine_returns_simple_analytics_engine() -> None:
    """build_default_engine returns a SimpleAnalyticsEngine instance."""
    assert isinstance(build_default_engine(), SimpleAnalyticsEngine)


def test_build_registry_contains_simple_engine() -> None:
    """Default registry contains SimpleAnalyticsEngine under 'simple'."""
    registry = build_registry()
    assert registry.exists("simple")
    assert isinstance(registry.get("simple"), SimpleAnalyticsEngine)


def test_build_analytics_pipeline_wires_registry() -> None:
    """build_analytics_pipeline returns a fully wired AnalyticsPipeline."""
    options = _options(storage_root=Path(DEFAULT_STORAGE_ROOT))
    pipeline = build_analytics_pipeline(options)
    assert isinstance(pipeline, AnalyticsPipeline)


# ---------------------------------------------------------------------------
# discover_work
# ---------------------------------------------------------------------------


def test_discover_work_groups_performance_partitions(tmp_path: Path) -> None:
    """discover_work returns sorted work items from the performance tier."""
    layout = StorageLayout(tmp_path)
    performance_repository = PerformanceRepository(layout, ParquetStore())
    for symbol, year in (("ETHUSDT", 2025), ("BTCUSDT", 2026), ("BTCUSDT", 2025)):
        performance_repository.save(
            _performance_frame(symbol=symbol),
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            symbol=symbol,
            timeframe=_TIMEFRAME,
            year=year,
        )
    work = discover_work(
        performance_repository,
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


def test_discover_work_returns_empty_when_no_performance_partitions(tmp_path: Path) -> None:
    """discover_work returns empty tuple when no performance partitions exist."""
    layout = StorageLayout(tmp_path)
    performance_repository = PerformanceRepository(layout, ParquetStore())
    work = discover_work(performance_repository, _options(storage_root=tmp_path))
    assert work == ()


# ---------------------------------------------------------------------------
# format_summary
# ---------------------------------------------------------------------------


def test_format_summary_includes_core_fields() -> None:
    """format_summary renders manager, engine, and analytics aggregates."""
    summary = AnalyticsGenerationSummary(
        manager=_MANAGER,
        engine=_ENGINE,
        symbols=2,
        rows=10,
        rolling_return=0.1234,
        max_dd=0.0567,
        successful_tasks=2,
        failed_tasks=1,
        skipped_tasks=0,
        duration_seconds=2.5,
        output_directory=Path("data") / STORAGE_DIR_ANALYTICS,
        failed_task_labels=("BTCUSDT 1h 2026",),
    )
    text = format_summary(summary)
    assert "CQROS Analytics Generation Summary" in text
    assert f"Manager: {_MANAGER}" in text
    assert f"Engine: {_ENGINE}" in text
    assert "Symbols: 2" in text
    assert "Rows: 10" in text
    assert "Rolling Return: 0.1234" in text
    assert "Max DD: 0.0567" in text
    assert "Failed Tasks" in text
    assert "BTCUSDT 1h 2026" in text
    assert STORAGE_DIR_ANALYTICS in text


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
            pipeline=AnalyticsPipeline(registry),
            performance_repository=PerformanceRepository(layout, datastore),
            analytics_repository=AnalyticsRepository(layout, datastore),
            options=options,
            work=(),
        )
    )
    assert summary.successful_tasks == 0
    assert summary.failed_tasks == 0
    assert summary.skipped_tasks == 0
    assert summary.rows == 0
    assert summary.rolling_return == 0.0
    assert summary.max_dd == 0.0
    assert summary.output_directory == tmp_path / STORAGE_DIR_ANALYTICS


# ---------------------------------------------------------------------------
# run_generation — persists analytics ledgers
# ---------------------------------------------------------------------------


def test_run_generation_persists_analytics_partitions(tmp_path: Path) -> None:
    """Generation loads performance inputs, runs the pipeline, and persists output."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()

    _seed_generation_inputs(layout=layout, datastore=datastore, symbol="BTCUSDT")
    _seed_generation_inputs(
        layout=layout,
        datastore=datastore,
        symbol="ETHUSDT",
        total_return=0.1,
    )

    options = _options(storage_root=tmp_path)
    performance_repository = PerformanceRepository(layout, datastore)
    work = discover_work(performance_repository, options)

    registry = AnalyticsEngineRegistry()
    registry.register(_ENGINE, SimpleAnalyticsEngine())
    summary = _run(
        run_generation(
            pipeline=AnalyticsPipeline(registry),
            performance_repository=performance_repository,
            analytics_repository=AnalyticsRepository(layout, datastore),
            options=options,
            work=work,
        )
    )
    assert summary.successful_tasks == 2
    assert summary.failed_tasks == 0
    assert summary.rows == 2

    analytics_repo = AnalyticsRepository(layout, datastore)
    assert analytics_repo.exists(
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol="BTCUSDT",
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert analytics_repo.exists(
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
    """Existing analytics partitions are skipped when overwrite is False."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    _seed_generation_inputs(layout=layout, datastore=datastore)

    options = _options(storage_root=tmp_path, overwrite=False)
    performance_repository = PerformanceRepository(layout, datastore)
    work = discover_work(performance_repository, options)

    registry = build_registry()
    first_summary = _run(
        run_generation(
            pipeline=AnalyticsPipeline(registry),
            performance_repository=performance_repository,
            analytics_repository=AnalyticsRepository(layout, datastore),
            options=options,
            work=work,
        )
    )
    assert first_summary.successful_tasks == 1

    second_summary = _run(
        run_generation(
            pipeline=AnalyticsPipeline(build_registry()),
            performance_repository=performance_repository,
            analytics_repository=AnalyticsRepository(layout, datastore),
            options=options,
            work=work,
        )
    )
    assert second_summary.skipped_tasks == 1
    assert second_summary.successful_tasks == 0
    assert second_summary.failed_tasks == 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_returns_success_for_empty_universe(tmp_path: Path) -> None:
    """main returns 0 when no performance partitions are discovered."""
    with patch.object(generate_analytics_module, "DEFAULT_STORAGE_ROOT", tmp_path):
        code = _run(main(["--manager", "simple", "--storage-root", str(tmp_path)]))
    assert code == 0


def test_main_validation_error_exit_code() -> None:
    """main returns 1 when build_options raises ValidationError."""
    code = _run(main(["--manager", "simple", "--workers", "0"]))
    assert code == 1


def test_main_returns_success_after_generation(tmp_path: Path) -> None:
    """main returns 0 after successful analytics generation."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    _seed_generation_inputs(layout=layout, datastore=datastore)
    with patch.object(generate_analytics_module, "DEFAULT_STORAGE_ROOT", tmp_path):
        code = _run(main(["--manager", "simple", "--storage-root", str(tmp_path)]))
    assert code == 0
