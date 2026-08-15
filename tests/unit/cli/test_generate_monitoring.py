"""Unit tests for CQROS monitoring generation CLI."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

import cqros.cli.generate_monitoring as generate_monitoring_module
from cqros.cli.generate_monitoring import (
    DiscoveredWorkItem,
    MonitoringGenerationOptions,
    MonitoringGenerationSummary,
    build_default_engine,
    build_monitoring_pipeline,
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
    STORAGE_DIR_MONITORING,
)
from cqros.core.exceptions import ValidationError
from cqros.monitoring import (
    MonitoringEngineRegistry,
    MonitoringPipeline,
    MonitoringRepository,
    SimpleMonitoringEngine,
)
from cqros.reporting import ReportingRepository, ReportingStatus
from cqros.reporting.schema import CANONICAL_COLUMN_ORDER, COLUMN_DTYPES
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
) -> MonitoringGenerationOptions:
    """Build MonitoringGenerationOptions against a temporary storage root."""
    return MonitoringGenerationOptions(
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


def _reporting_frame(
    *,
    symbol: str = _SYMBOL,
    report_name: str = "performance_report",
) -> pl.DataFrame:
    """Return a canonical reporting frame for monitoring generation tests."""
    return pl.DataFrame(
        {
            "symbol": [symbol],
            "timeframe": [_TIMEFRAME],
            "open_time": [_OPEN_TIME_MS],
            "manager": [_MANAGER],
            "report_name": [report_name],
            "report_type": ["analytics"],
            "report_format": ["parquet"],
            "report_version": ["v1"],
            "report_path": [""],
            "generated_at": [_OPEN_TIME_MS],
            "status": [ReportingStatus.GENERATED.value],
        },
        schema=dict(COLUMN_DTYPES),
    ).select(list(CANONICAL_COLUMN_ORDER))


def _seed_generation_inputs(
    *,
    layout: StorageLayout,
    datastore: ParquetStore,
    symbol: str = _SYMBOL,
    report_name: str = "performance_report",
) -> None:
    """Persist a reporting partition needed for monitoring generation."""
    ReportingRepository(layout, datastore).save(
        _reporting_frame(symbol=symbol, report_name=report_name),
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
    """build_options rejects workers <= 0 with CLI-GENERATE-MONITORING-001."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "simple", "--workers", "0"])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-MONITORING-001"


def test_build_options_rejects_blank_manager() -> None:
    """build_options rejects a blank manager identity with CLI-GENERATE-MONITORING-004."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "   "])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-MONITORING-004"


def test_build_options_rejects_blank_engine() -> None:
    """build_options rejects a blank engine identity with CLI-GENERATE-MONITORING-005."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "simple", "--engine", "   "])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-MONITORING-005"


def test_build_options_uses_storage_root(tmp_path: Path) -> None:
    """build_options honors an explicit storage root override."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "simple", "--storage-root", str(tmp_path)])
    options = build_options(args)
    assert options.storage_root == tmp_path


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def test_build_default_engine_returns_simple_monitoring_engine() -> None:
    """build_default_engine returns a SimpleMonitoringEngine instance."""
    assert isinstance(build_default_engine(), SimpleMonitoringEngine)


def test_build_registry_contains_simple_engine() -> None:
    """Default registry contains SimpleMonitoringEngine under 'simple'."""
    registry = build_registry()
    assert registry.exists("simple")
    assert isinstance(registry.get("simple"), SimpleMonitoringEngine)


def test_build_monitoring_pipeline_wires_registry() -> None:
    """build_monitoring_pipeline returns a fully wired MonitoringPipeline."""
    options = _options(storage_root=Path(DEFAULT_STORAGE_ROOT))
    pipeline = build_monitoring_pipeline(options)
    assert isinstance(pipeline, MonitoringPipeline)


# ---------------------------------------------------------------------------
# discover_work
# ---------------------------------------------------------------------------


def test_discover_work_groups_reporting_partitions(tmp_path: Path) -> None:
    """discover_work returns sorted work items from the reporting tier."""
    layout = StorageLayout(tmp_path)
    reporting_repository = ReportingRepository(layout, ParquetStore())
    for symbol, year in (("ETHUSDT", 2025), ("BTCUSDT", 2026), ("BTCUSDT", 2025)):
        reporting_repository.save(
            _reporting_frame(symbol=symbol),
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            symbol=symbol,
            timeframe=_TIMEFRAME,
            year=year,
        )
    work = discover_work(
        reporting_repository,
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


def test_discover_work_returns_empty_when_no_reporting_partitions(tmp_path: Path) -> None:
    """discover_work returns empty tuple when no reporting partitions exist."""
    layout = StorageLayout(tmp_path)
    reporting_repository = ReportingRepository(layout, ParquetStore())
    work = discover_work(reporting_repository, _options(storage_root=tmp_path))
    assert work == ()


# ---------------------------------------------------------------------------
# format_summary
# ---------------------------------------------------------------------------


def test_format_summary_includes_core_fields() -> None:
    """format_summary renders manager, engine, and monitoring aggregates."""
    summary = MonitoringGenerationSummary(
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
        output_directory=Path("data") / STORAGE_DIR_MONITORING,
        failed_task_labels=("BTCUSDT 1h 2026",),
    )
    text = format_summary(summary)
    assert "CQROS Monitoring Generation Summary" in text
    assert f"Manager: {_MANAGER}" in text
    assert f"Engine: {_ENGINE}" in text
    assert "Symbols: 2" in text
    assert "Rows: 10" in text
    assert "Generated: 8" in text
    assert "Failed Status: 2" in text
    assert "Failed Tasks" in text
    assert "BTCUSDT 1h 2026" in text
    assert STORAGE_DIR_MONITORING in text


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
            pipeline=MonitoringPipeline(registry),
            reporting_repository=ReportingRepository(layout, datastore),
            monitoring_repository=MonitoringRepository(layout, datastore),
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
    assert summary.output_directory == tmp_path / STORAGE_DIR_MONITORING


# ---------------------------------------------------------------------------
# run_generation — persists monitoring ledgers
# ---------------------------------------------------------------------------


def test_run_generation_persists_monitoring_partitions(tmp_path: Path) -> None:
    """Generation loads reporting inputs, runs the pipeline, and persists output."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()

    _seed_generation_inputs(layout=layout, datastore=datastore, symbol="BTCUSDT")
    _seed_generation_inputs(
        layout=layout,
        datastore=datastore,
        symbol="ETHUSDT",
        report_name="risk_report",
    )

    options = _options(storage_root=tmp_path)
    reporting_repository = ReportingRepository(layout, datastore)
    work = discover_work(reporting_repository, options)

    registry = MonitoringEngineRegistry()
    registry.register(_ENGINE, SimpleMonitoringEngine())
    summary = _run(
        run_generation(
            pipeline=MonitoringPipeline(registry),
            reporting_repository=reporting_repository,
            monitoring_repository=MonitoringRepository(layout, datastore),
            options=options,
            work=work,
        )
    )
    assert summary.successful_tasks == 2
    assert summary.failed_tasks == 0
    assert summary.rows == 2
    assert summary.generated_rows == 2
    assert summary.failed_status_rows == 0

    monitoring_repo = MonitoringRepository(layout, datastore)
    assert monitoring_repo.exists(
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol="BTCUSDT",
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert monitoring_repo.exists(
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
    """Existing monitoring partitions are skipped when overwrite is False."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    _seed_generation_inputs(layout=layout, datastore=datastore)

    options = _options(storage_root=tmp_path, overwrite=False)
    reporting_repository = ReportingRepository(layout, datastore)
    work = discover_work(reporting_repository, options)

    registry = build_registry()
    first_summary = _run(
        run_generation(
            pipeline=MonitoringPipeline(registry),
            reporting_repository=reporting_repository,
            monitoring_repository=MonitoringRepository(layout, datastore),
            options=options,
            work=work,
        )
    )
    assert first_summary.successful_tasks == 1

    second_summary = _run(
        run_generation(
            pipeline=MonitoringPipeline(build_registry()),
            reporting_repository=reporting_repository,
            monitoring_repository=MonitoringRepository(layout, datastore),
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
    """main returns 0 when no reporting partitions are discovered."""
    with patch.object(generate_monitoring_module, "DEFAULT_STORAGE_ROOT", tmp_path):
        code = _run(main(["--manager", "simple", "--storage-root", str(tmp_path)]))
    assert code == 0


def test_main_validation_error_exit_code() -> None:
    """main returns 1 when build_options raises ValidationError."""
    code = _run(main(["--manager", "simple", "--workers", "0"]))
    assert code == 1


def test_main_returns_success_after_generation(tmp_path: Path) -> None:
    """main returns 0 after successful monitoring generation."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    _seed_generation_inputs(layout=layout, datastore=datastore)
    with patch.object(generate_monitoring_module, "DEFAULT_STORAGE_ROOT", tmp_path):
        code = _run(main(["--manager", "simple", "--storage-root", str(tmp_path)]))
    assert code == 0
