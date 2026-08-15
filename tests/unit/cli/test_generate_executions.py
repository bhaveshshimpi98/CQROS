"""Unit tests for CQROS execution-generation CLI."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import cqros.cli.generate_executions as generate_executions_module
from cqros.cli.generate_executions import (
    DiscoveredWorkItem,
    ExecutionGenerationOptions,
    ExecutionGenerationSummary,
    ExecutionTaskResult,
    build_default_simulator,
    build_execution_pipeline,
    build_options,
    build_parser,
    build_simulator_registry,
    discover_work,
    format_summary,
    main,
    run_generation,
)
from cqros.config.models import ResearchConfig
from cqros.core.constants import (
    DEFAULT_STORAGE_ROOT,
    STORAGE_DIR_EXECUTIONS,
    STORAGE_DIR_ORDERS,
)
from cqros.core.exceptions import ValidationError
from cqros.execution import (
    ExecutionPipeline,
    ExecutionSimulatorRegistry,
    SimpleExecutionSimulator,
)
from cqros.storage import OrderRepository, ParquetStore, StorageLayout

_MANAGER = "simple"
_SIMULATOR = "simple"
_MODEL = "alpha-lgbm"
_VERSION = "1.0.0"


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def _options(
    *,
    storage_root: Path,
    manager: str = _MANAGER,
    simulator: str = _SIMULATOR,
    model: str | None = _MODEL,
    version: str | None = _VERSION,
    symbols: tuple[str, ...] | None = None,
    timeframes: tuple[str, ...] | None = None,
    years: tuple[int, ...] | None = None,
    overwrite: bool = False,
    workers: int = ResearchConfig().worker_count,
    verbose: bool = False,
    debug: bool = False,
) -> ExecutionGenerationOptions:
    """Build options for tests against a temporary storage root."""
    return ExecutionGenerationOptions(
        storage_root=storage_root,
        manager=manager,
        simulator=simulator,
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


def _touch_order(
    root: Path,
    *,
    manager: str,
    symbol: str,
    timeframe: str,
    year: int,
) -> Path:
    """Create an empty order year partition path on disk."""
    path = (
        root
        / STORAGE_DIR_ORDERS
        / manager
        / "binance"
        / "usdt_perpetual"
        / symbol
        / timeframe
        / f"{year}.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_build_parser_requires_manager() -> None:
    """Parser requires --manager and defaults simulator to simple."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(["--manager", "simple"])
    assert args.manager == "simple"
    assert args.simulator == "simple"


def test_build_options_validates_workers_and_manager(tmp_path: Path) -> None:
    """build_options rejects non-positive workers and blank manager."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "simple", "--workers", "0"])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-EXECUTIONS-001"

    args = parser.parse_args(["--manager", "   "])
    with patch.object(generate_executions_module, "DEFAULT_STORAGE_ROOT", tmp_path):
        with pytest.raises(ValidationError) as exc_info:
            build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-EXECUTIONS-006"


def test_build_default_simulator_and_registry() -> None:
    """Default registry contains SimpleExecutionSimulator under simple."""
    assert isinstance(build_default_simulator(), SimpleExecutionSimulator)
    registry = build_simulator_registry()
    assert registry.exists("simple")
    assert isinstance(registry.get("simple"), SimpleExecutionSimulator)


def test_build_execution_pipeline_wires_registry() -> None:
    """Pipeline composition uses the default simulator registry."""
    options = _options(storage_root=Path(DEFAULT_STORAGE_ROOT))
    pipeline = build_execution_pipeline(options)
    assert isinstance(pipeline, ExecutionPipeline)


def test_discover_work_groups_order_partitions(tmp_path: Path) -> None:
    """discover_work returns sorted manager/symbol/timeframe work items."""
    _touch_order(tmp_path, manager=_MANAGER, symbol="ETHUSDT", timeframe="1h", year=2025)
    _touch_order(tmp_path, manager=_MANAGER, symbol="BTCUSDT", timeframe="1h", year=2026)
    _touch_order(tmp_path, manager=_MANAGER, symbol="BTCUSDT", timeframe="1h", year=2025)
    repository = OrderRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(repository, _options(storage_root=tmp_path, model=None, version=None))
    assert work == (
        DiscoveredWorkItem(
            manager=_MANAGER,
            symbol="BTCUSDT",
            timeframe="1h",
            years=(2025, 2026),
        ),
        DiscoveredWorkItem(
            manager=_MANAGER,
            symbol="ETHUSDT",
            timeframe="1h",
            years=(2025,),
        ),
    )


def test_format_summary_includes_core_fields() -> None:
    """format_summary renders manager, simulator, and task counters."""
    summary = ExecutionGenerationSummary(
        manager=_MANAGER,
        simulator=_SIMULATOR,
        version=_VERSION,
        symbols_discovered=2,
        symbols_processed=2,
        timeframes_processed=1,
        successful_tasks=1,
        failed_tasks=1,
        skipped_tasks=0,
        rows_generated=10,
        duration_seconds=1.5,
        output_directory=Path("data") / STORAGE_DIR_EXECUTIONS,
        failed_task_labels=("BTCUSDT 1h 2026",),
    )
    text = format_summary(summary)
    assert "CQROS Execution Generation Summary" in text
    assert f"Manager: {_MANAGER}" in text
    assert f"Simulator: {_SIMULATOR}" in text
    assert "Failed Tasks" in text
    assert "BTCUSDT 1h 2026" in text


def test_run_generation_empty_work(tmp_path: Path) -> None:
    """Empty work produces a zeroed success summary."""
    options = _options(storage_root=tmp_path)
    pipeline = ExecutionPipeline(build_simulator_registry())
    summary = _run(
        run_generation(
            pipeline=pipeline,
            order_repository=MagicMock(),
            trade_repository=MagicMock(),
            options=options,
            work=(),
        )
    )
    assert summary.successful_tasks == 0
    assert summary.failed_tasks == 0
    assert summary.output_directory == tmp_path / STORAGE_DIR_EXECUTIONS


def test_main_returns_success_for_empty_universe(tmp_path: Path) -> None:
    """main returns 0 when no order partitions are discovered."""
    with patch.object(generate_executions_module, "DEFAULT_STORAGE_ROOT", tmp_path):
        code = _run(main(["--manager", "simple"]))
    assert code == 0


def test_task_result_statuses_are_reported() -> None:
    """Task result dataclass accepts succeeded/failed/skipped statuses."""
    result = ExecutionTaskResult(
        symbol="BTCUSDT",
        timeframe="1h",
        year=2026,
        status="skipped",
    )
    assert result.rows_generated is None
    assert result.status == "skipped"


def test_build_simulator_registry_accepts_injection() -> None:
    """Custom simulator mappings are registered atomically."""
    custom = SimpleExecutionSimulator()
    registry = build_simulator_registry(simulators={"custom": custom})
    assert isinstance(registry, ExecutionSimulatorRegistry)
    assert registry.get("custom") is custom
    assert registry.exists("simple") is False
