"""Unit tests for CQROS reporting generation CLI."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

import cqros.cli.generate_reporting as generate_reporting_module
from cqros.analytics import AnalyticsRepository, AnalyticsStatus
from cqros.analytics.schema import CANONICAL_COLUMN_ORDER, COLUMN_DTYPES
from cqros.cli.generate_reporting import (
    DiscoveredWorkItem,
    ReportingGenerationOptions,
    ReportingGenerationSummary,
    build_default_engine,
    build_options,
    build_parser,
    build_registry,
    build_reporting_pipeline,
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
    STORAGE_DIR_REPORTING,
)
from cqros.core.exceptions import ValidationError
from cqros.reporting import (
    ReportingEngineRegistry,
    ReportingPipeline,
    ReportingRepository,
    SimpleReportingEngine,
)
from cqros.storage import ParquetStore, StorageLayout

_MANAGER = "simple"
_ENGINE = "simple"
_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1h"
_YEAR = 2026
_OPEN_TIME_MS = int(datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC).timestamp() * 1000.0)


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
) -> ReportingGenerationOptions:
    """Build ReportingGenerationOptions against a temporary storage root."""
    return ReportingGenerationOptions(
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


def _analytics_frame(
    *,
    symbol: str = _SYMBOL,
    rolling_return: float = 0.15,
    rolling_max_drawdown: float = 0.02,
) -> pl.DataFrame:
    """Return a canonical analytics frame for reporting generation tests."""
    return pl.DataFrame(
        {
            "symbol": [symbol],
            "timeframe": [_TIMEFRAME],
            "open_time": [_OPEN_TIME_MS],
            "manager": [_MANAGER],
            "rolling_return": [rolling_return],
            "rolling_volatility": [0.0],
            "rolling_sharpe": [None],
            "rolling_sortino": [None],
            "rolling_max_drawdown": [rolling_max_drawdown],
            "rolling_win_rate": [0.0],
            "rolling_profit_factor": [None],
            "rolling_expectancy": [0.0],
            "rolling_cagr": [0.0],
            "rolling_calmar": [None],
            "rolling_recovery_factor": [None],
            "benchmark_return": [0.0],
            "benchmark_alpha": [0.0],
            "benchmark_beta": [0.0],
            "benchmark_correlation": [0.0],
            "benchmark_tracking_error": [0.0],
            "benchmark_information_ratio": [0.0],
            "status": [AnalyticsStatus.FINISHED.value],
        },
        schema=dict(COLUMN_DTYPES),
    ).select(list(CANONICAL_COLUMN_ORDER))


def _seed_generation_inputs(
    *,
    layout: StorageLayout,
    datastore: ParquetStore,
    symbol: str = _SYMBOL,
    rolling_return: float = 0.15,
) -> None:
    """Persist an analytics partition needed for reporting generation."""
    AnalyticsRepository(layout, datastore).save(
        _analytics_frame(symbol=symbol, rolling_return=rolling_return),
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
    """build_options rejects workers <= 0 with CLI-GENERATE-REPORTING-001."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "simple", "--workers", "0"])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-REPORTING-001"


def test_build_options_rejects_blank_manager() -> None:
    """build_options rejects a blank manager identity with CLI-GENERATE-REPORTING-004."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "   "])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-REPORTING-004"


def test_build_options_rejects_blank_engine() -> None:
    """build_options rejects a blank engine identity with CLI-GENERATE-REPORTING-005."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "simple", "--engine", "   "])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-REPORTING-005"


def test_build_options_uses_storage_root(tmp_path: Path) -> None:
    """build_options honors an explicit storage root override."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "simple", "--storage-root", str(tmp_path)])
    options = build_options(args)
    assert options.storage_root == tmp_path


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def test_build_default_engine_returns_simple_reporting_engine() -> None:
    """build_default_engine returns a SimpleReportingEngine instance."""
    assert isinstance(build_default_engine(), SimpleReportingEngine)


def test_build_registry_contains_simple_engine() -> None:
    """Default registry contains SimpleReportingEngine under 'simple'."""
    registry = build_registry()
    assert registry.exists("simple")
    assert isinstance(registry.get("simple"), SimpleReportingEngine)


def test_build_reporting_pipeline_wires_registry() -> None:
    """build_reporting_pipeline returns a fully wired ReportingPipeline."""
    options = _options(storage_root=Path(DEFAULT_STORAGE_ROOT))
    pipeline = build_reporting_pipeline(options)
    assert isinstance(pipeline, ReportingPipeline)


# ---------------------------------------------------------------------------
# discover_work
# ---------------------------------------------------------------------------


def test_discover_work_groups_analytics_partitions(tmp_path: Path) -> None:
    """discover_work returns sorted work items from the analytics tier."""
    layout = StorageLayout(tmp_path)
    analytics_repository = AnalyticsRepository(layout, ParquetStore())
    for symbol, year in (("ETHUSDT", 2025), ("BTCUSDT", 2026), ("BTCUSDT", 2025)):
        analytics_repository.save(
            _analytics_frame(symbol=symbol),
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            symbol=symbol,
            timeframe=_TIMEFRAME,
            year=year,
        )
    work = discover_work(
        analytics_repository,
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


def test_discover_work_returns_empty_when_no_analytics_partitions(tmp_path: Path) -> None:
    """discover_work returns empty tuple when no analytics partitions exist."""
    layout = StorageLayout(tmp_path)
    analytics_repository = AnalyticsRepository(layout, ParquetStore())
    work = discover_work(analytics_repository, _options(storage_root=tmp_path))
    assert work == ()


# ---------------------------------------------------------------------------
# format_summary
# ---------------------------------------------------------------------------


def test_format_summary_includes_core_fields() -> None:
    """format_summary renders manager, engine, and reporting aggregates."""
    summary = ReportingGenerationSummary(
        manager=_MANAGER,
        engine=_ENGINE,
        symbols=2,
        rows=10,
        generated_rows=8,
        failed_status_rows=2,
        successful_tasks=2,
        failed_tasks=1,
        skipped_tasks=0,
        duration_seconds=2.5,
        output_directory=Path("data") / STORAGE_DIR_REPORTING,
        failed_task_labels=("BTCUSDT 1h 2026",),
    )
    text = format_summary(summary)
    assert "CQROS Reporting Generation Summary" in text
    assert f"Manager: {_MANAGER}" in text
    assert f"Engine: {_ENGINE}" in text
    assert "Symbols: 2" in text
    assert "Rows: 10" in text
    assert "Generated: 8" in text
    assert "Failed Status: 2" in text
    assert "Rolling Return" not in text
    assert "Max DD" not in text
    assert "Failed Tasks" in text
    assert "BTCUSDT 1h 2026" in text
    assert STORAGE_DIR_REPORTING in text


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
            pipeline=ReportingPipeline(registry),
            analytics_repository=AnalyticsRepository(layout, datastore),
            reporting_repository=ReportingRepository(layout, datastore),
            options=options,
            work=(),
        )
    )
    assert summary.successful_tasks == 0
    assert summary.failed_tasks == 0
    assert summary.skipped_tasks == 0
    assert summary.rows == 0
    assert summary.generated_rows == 0
    assert summary.failed_status_rows == 0
    assert summary.output_directory == tmp_path / STORAGE_DIR_REPORTING


# ---------------------------------------------------------------------------
# run_generation — persists reporting ledgers
# ---------------------------------------------------------------------------


def test_run_generation_persists_reporting_partitions(tmp_path: Path) -> None:
    """Generation loads analytics inputs, runs the pipeline, and persists output."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()

    _seed_generation_inputs(layout=layout, datastore=datastore, symbol="BTCUSDT")
    _seed_generation_inputs(
        layout=layout,
        datastore=datastore,
        symbol="ETHUSDT",
        rolling_return=0.1,
    )

    options = _options(storage_root=tmp_path)
    analytics_repository = AnalyticsRepository(layout, datastore)
    work = discover_work(analytics_repository, options)

    registry = ReportingEngineRegistry()
    registry.register(_ENGINE, SimpleReportingEngine())
    summary = _run(
        run_generation(
            pipeline=ReportingPipeline(registry),
            analytics_repository=analytics_repository,
            reporting_repository=ReportingRepository(layout, datastore),
            options=options,
            work=work,
        )
    )
    assert summary.successful_tasks == 2
    assert summary.failed_tasks == 0
    assert summary.rows == 2
    assert summary.generated_rows == 2
    assert summary.failed_status_rows == 0

    reporting_repo = ReportingRepository(layout, datastore)
    assert reporting_repo.exists(
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol="BTCUSDT",
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert reporting_repo.exists(
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
    """Existing reporting partitions are skipped when overwrite is False."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    _seed_generation_inputs(layout=layout, datastore=datastore)

    options = _options(storage_root=tmp_path, overwrite=False)
    analytics_repository = AnalyticsRepository(layout, datastore)
    work = discover_work(analytics_repository, options)

    registry = build_registry()
    first_summary = _run(
        run_generation(
            pipeline=ReportingPipeline(registry),
            analytics_repository=analytics_repository,
            reporting_repository=ReportingRepository(layout, datastore),
            options=options,
            work=work,
        )
    )
    assert first_summary.successful_tasks == 1

    second_summary = _run(
        run_generation(
            pipeline=ReportingPipeline(build_registry()),
            analytics_repository=analytics_repository,
            reporting_repository=ReportingRepository(layout, datastore),
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
    """main returns 0 when no analytics partitions are discovered."""
    with patch.object(generate_reporting_module, "DEFAULT_STORAGE_ROOT", tmp_path):
        code = _run(main(["--manager", "simple", "--storage-root", str(tmp_path)]))
    assert code == 0


def test_main_validation_error_exit_code() -> None:
    """main returns 1 when build_options raises ValidationError."""
    code = _run(main(["--manager", "simple", "--workers", "0"]))
    assert code == 1


def test_main_returns_success_after_generation(tmp_path: Path) -> None:
    """main returns 0 after successful reporting generation."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    _seed_generation_inputs(layout=layout, datastore=datastore)
    with patch.object(generate_reporting_module, "DEFAULT_STORAGE_ROOT", tmp_path):
        code = _run(main(["--manager", "simple", "--storage-root", str(tmp_path)]))
    assert code == 0
