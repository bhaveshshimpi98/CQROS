"""CQROS walk-forward-generation CLI.

Purpose:
    Provide an argparse-based production entry point that discovers canonical
    Factor Selection panels and executes ``WalkForwardPipeline`` across
    timeframe/year panels with bounded concurrency, persisting walk-forward
    metrics through ``WalkForwardRepository``.

Responsibilities:
    - Parse CLI arguments for walk-forward dataset generation
    - Discover available Factor Selection partitions through
      ``FactorSelectionRepository``
    - Load matching Factor Selection panels for each discovered partition
    - Enrich evaluation input with Labels ``future_return_1`` through
      ``WalkForwardInputBuilder`` (Factors + Labels join)
    - Resolve ``--engine`` through ``WalkForwardEngineRegistry``
    - Execute ``WalkForwardPipeline`` and persist via ``WalkForwardRepository``
    - Honor ``--overwrite``, worker concurrency, and debug logging
    - Print deterministic progress and a final summary

Dependencies:
    ``argparse``, ``asyncio``, ``polars``, ``cqros.config``, ``cqros.core``,
    ``cqros.factor_selection``, ``cqros.factors``, ``cqros.walk_forward``, and
    ``cqros.storage``.

Public API:
    ``build_parser``, ``build_options``, ``build_default_engine``,
    ``build_registry``, ``build_walk_forward_pipeline``,
    ``discover_work``, ``format_summary``, ``run_generation``, and ``main``.

Notes:
    This module is a thin composition root. It does not implement walk-forward
    math, schema validation, or repository filesystem walks beyond calling
    repository discovery and load/save APIs. Metric computation is delegated
    exclusively to ``WalkForwardPipeline``. Persistence remains in the CLI
    because ``WalkForwardPipeline`` does not own a repository.
    Canonical Factor Selection partitions are never mutated; ``future_return_1``
    exists only on the evaluation-input frame passed to the engine.
    Partitions are cross-sectional panels keyed by manager/timeframe/year
    (no symbol).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import polars as pl

from cqros.config.models import ResearchConfig
from cqros.core.constants import (
    DEFAULT_STORAGE_ROOT,
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
    STORAGE_DIR_WALK_FORWARD,
    SUPPORTED_TIMEFRAMES,
)
from cqros.core.exceptions import CQROSError, ValidationError
from cqros.core.types import Timeframe
from cqros.factor_selection import FactorSelectionPartitionRef, FactorSelectionRepository
from cqros.factors import FactorsRepository
from cqros.storage import LabelRepository, ParquetStore, StorageLayout
from cqros.walk_forward import (
    FULL_PANEL_EXECUTION_MODE,
    MEMORY_EFFICIENT_EXECUTION_MODE,
    MemoryEfficientExecutionConfig,
    MemoryEfficientWalkForwardExecutor,
    SimpleWalkForwardEngine,
    WalkForwardEngine,
    WalkForwardEngineRegistry,
    WalkForwardInputBuilder,
    WalkForwardPipeline,
    WalkForwardRepository,
    WalkForwardStatus,
)

__all__ = [
    "DiscoveredWorkItem",
    "WalkForwardGenerationOptions",
    "WalkForwardGenerationSummary",
    "WalkForwardTaskResult",
    "build_default_engine",
    "build_options",
    "build_parser",
    "build_registry",
    "build_walk_forward_pipeline",
    "discover_work",
    "format_summary",
    "main",
    "run_generation",
]

_logger = logging.getLogger(__name__)

_EXIT_SUCCESS: Final[int] = 0
_EXIT_FAILURE: Final[int] = 1

_DEFAULT_WORKER_COUNT: Final[int] = ResearchConfig().worker_count
_DEFAULT_ENGINE: Final[str] = "simple"
_DEFAULT_EXECUTION_MODE: Final[str] = FULL_PANEL_EXECUTION_MODE
_DEFAULT_MEMORY_BUDGET_MB: Final[int] = 256

_EXCHANGE: Final[str] = EXCHANGE_BINANCE
_MARKET: Final[str] = MARKET_USDT_PERPETUAL

_ERROR_WORKERS: Final[str] = "CLI-GENERATE-WALK-FORWARD-001"
_ERROR_TIMEFRAME: Final[str] = "CLI-GENERATE-WALK-FORWARD-002"
_ERROR_YEAR: Final[str] = "CLI-GENERATE-WALK-FORWARD-003"
_ERROR_MANAGER: Final[str] = "CLI-GENERATE-WALK-FORWARD-004"
_ERROR_ENGINE: Final[str] = "CLI-GENERATE-WALK-FORWARD-005"
_ERROR_FACTOR_SELECTION_MISSING: Final[str] = "CLI-GENERATE-WALK-FORWARD-006"
_ERROR_EXECUTION_MODE: Final[str] = "CLI-GENERATE-WALK-FORWARD-007"
_ERROR_MEMORY_MODE_COMBINATION: Final[str] = "CLI-GENERATE-WALK-FORWARD-008"

_COL_STATUS: Final[str] = "status"
_COL_SELECTED_FACTORS: Final[str] = "selected_factors"


@dataclass(frozen=True, slots=True)
class WalkForwardGenerationOptions:
    """Immutable CLI options for walk-forward dataset generation.

    Attributes:
        storage_root: Storage root containing ``factor_selection`` and
            ``walk_forward``.
        manager: Order manager identity used for discovery and walk-forward
            lineage.
        engine: Registry key of the walk-forward engine to execute.
        timeframes: Optional timeframe allowlist. ``None`` discovers all.
        years: Optional year allowlist. ``None`` discovers all.
        overwrite: When ``True``, regenerate existing walk-forward partitions.
        workers: Maximum concurrent panels.
        verbose: When ``True``, enable INFO logging.
        debug: When ``True``, enable DEBUG logging and log full failure
            tracebacks.
        execution_mode: Physical executor. ``full_panel`` is the unchanged
            default; ``memory_efficient`` is explicit opt-in.
        spill_parent: Optional parent for unique bounded-execution spill runs.
        memory_budget_mb: External-merge cursor budget in MiB.
    """

    storage_root: Path
    manager: str
    engine: str
    timeframes: tuple[Timeframe, ...] | None
    years: tuple[int, ...] | None
    overwrite: bool
    workers: int
    verbose: bool
    debug: bool
    execution_mode: str = _DEFAULT_EXECUTION_MODE
    spill_parent: Path | None = None
    memory_budget_mb: int = _DEFAULT_MEMORY_BUDGET_MB


@dataclass(frozen=True, slots=True)
class DiscoveredWorkItem:
    """One discovered Factor Selection panel group ready for walk-forward generation.

    Attributes:
        manager: Order manager identifier of the source partitions.
        timeframe: Available bar interval.
        years: Calendar years with existing Factor Selection parquet partitions.
    """

    manager: str
    timeframe: Timeframe
    years: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class WalkForwardTaskResult:
    """Immutable result for one timeframe/year panel generation task.

    Attributes:
        timeframe: Bar interval.
        year: Calendar year of the partition.
        status: ``succeeded``, ``failed``, or ``skipped``.
        rows_generated: Output row count when succeeded.
        selected_factors: Sum of ``selected_factors`` when succeeded.
        pass_rows: Count of rows with ``PASS`` status on success.
        fail_rows: Count of rows with ``FAIL`` status on success.
        error_type: Exception type name when failed.
        error_message: Exception message when failed.
        error_code: Optional CQROS error code when failed.
    """

    timeframe: Timeframe
    year: int
    status: str
    rows_generated: int | None = None
    selected_factors: int | None = None
    pass_rows: int | None = None
    fail_rows: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class WalkForwardGenerationSummary:
    """Immutable aggregate summary for a walk-forward-generation run.

    Attributes:
        manager: Order manager identity used for generation.
        engine: Walk-forward engine registry key used for generation.
        panels: Unique timeframe/year panels for which generation was attempted.
        rows: Sum of output rows across successes.
        selected_factors: Sum of ``selected_factors`` across successes.
        pass_rows: Sum of rows with ``PASS`` status across successes.
        fail_rows: Sum of rows with ``FAIL`` status across successes.
        successful_tasks: Count of succeeded partition tasks.
        failed_tasks: Count of failed partition tasks.
        skipped_tasks: Count of skipped existing partitions.
        duration_seconds: Wall-clock generation duration.
        output_directory: Walk-forward-tier output directory.
        failed_task_labels: Deterministic failed-task labels for reporting.
        execution_mode: Physical execution mode recorded for the run.
    """

    manager: str
    engine: str
    panels: int
    rows: int
    selected_factors: int
    pass_rows: int
    fail_rows: int
    successful_tasks: int
    failed_tasks: int
    skipped_tasks: int
    duration_seconds: float
    output_directory: Path
    failed_task_labels: tuple[str, ...]
    execution_mode: str = _DEFAULT_EXECUTION_MODE


def build_parser() -> argparse.ArgumentParser:
    """Create the walk-forward-generation argument parser.

    Returns:
        Configured ``ArgumentParser`` for walk-forward-generation flags.
    """
    parser = argparse.ArgumentParser(
        prog="cqros-generate-walk-forward",
        description=(
            "Generate CQROS walk-forward datasets from discovered "
            "Factor Selection panels and an injected walk-forward engine."
        ),
    )
    parser.add_argument(
        "--manager",
        dest="manager",
        required=True,
        metavar="NAME",
        help="Order manager identity used for discovery and walk-forward lineage.",
    )
    parser.add_argument(
        "--engine",
        dest="engine",
        default=_DEFAULT_ENGINE,
        metavar="NAME",
        help=f"Walk-forward engine registry key (default: {_DEFAULT_ENGINE}).",
    )
    parser.add_argument(
        "--timeframes",
        dest="timeframes",
        nargs="*",
        default=None,
        metavar="TIMEFRAME",
        help="Optional timeframe allowlist (0..N values). Omit to discover all.",
    )
    parser.add_argument(
        "--years",
        dest="years",
        nargs="*",
        default=None,
        metavar="YEAR",
        help="Optional calendar-year allowlist (0..N values). Omit to discover all.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate walk-forward partitions that already exist.",
    )
    parser.add_argument(
        "--workers",
        dest="workers",
        type=int,
        default=_DEFAULT_WORKER_COUNT,
        metavar="INT",
        help=f"Maximum concurrent panels (default: {_DEFAULT_WORKER_COUNT}).",
    )
    parser.add_argument(
        "--execution-mode",
        choices=(FULL_PANEL_EXECUTION_MODE, MEMORY_EFFICIENT_EXECUTION_MODE),
        default=_DEFAULT_EXECUTION_MODE,
        help=(
            "Physical execution mode; full_panel remains the default reference " "implementation."
        ),
    )
    parser.add_argument(
        "--spill-parent",
        type=Path,
        default=None,
        metavar="PATH",
        help="Parent directory for run-scoped memory-efficient temporary spill.",
    )
    parser.add_argument(
        "--memory-budget-mb",
        type=int,
        default=_DEFAULT_MEMORY_BUDGET_MB,
        metavar="INT",
        help="Memory-efficient external-merge cursor budget in MiB (default: 256).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO logging.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging and log complete failure tracebacks with logger.exception.",
    )
    parser.add_argument(
        "--storage-root",
        dest="storage_root",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"Storage root for dataset tiers (default: {DEFAULT_STORAGE_ROOT}).",
    )
    return parser


def build_options(args: argparse.Namespace) -> WalkForwardGenerationOptions:
    """Map parsed CLI arguments onto ``WalkForwardGenerationOptions``.

    Args:
        args: Namespace produced by ``build_parser().parse_args(...)``.

    Returns:
        Immutable generation options.

    Raises:
        ValidationError: If ``workers`` is not positive or filters are invalid.
    """
    workers = int(args.workers)
    if workers <= 0:
        raise ValidationError(
            "workers must be greater than 0",
            error_code=_ERROR_WORKERS,
            details={"parameter": "workers", "value": workers},
        )

    manager = str(args.manager).strip()
    if manager == "":
        raise ValidationError(
            "manager must be a non-empty string",
            error_code=_ERROR_MANAGER,
            details={"parameter": "manager", "value": args.manager},
        )

    engine = str(args.engine).strip()
    if engine == "":
        raise ValidationError(
            "engine must be a non-empty string",
            error_code=_ERROR_ENGINE,
            details={"parameter": "engine", "value": args.engine},
        )

    storage_root = (
        Path(args.storage_root) if args.storage_root is not None else Path(DEFAULT_STORAGE_ROOT)
    )
    execution_mode = str(args.execution_mode)
    memory_budget_mb = int(args.memory_budget_mb)
    if memory_budget_mb <= 0:
        raise ValidationError(
            "memory_budget_mb must be greater than 0",
            error_code=_ERROR_EXECUTION_MODE,
            details={"parameter": "memory_budget_mb", "value": memory_budget_mb},
        )
    if execution_mode == MEMORY_EFFICIENT_EXECUTION_MODE and (
        workers != 1 or engine != _DEFAULT_ENGINE
    ):
        raise ValidationError(
            "memory_efficient execution requires --workers 1 and --engine simple",
            error_code=_ERROR_MEMORY_MODE_COMBINATION,
            details={"workers": workers, "engine": engine},
        )

    return WalkForwardGenerationOptions(
        storage_root=storage_root,
        manager=manager,
        engine=engine,
        timeframes=_normalize_timeframes(args.timeframes),
        years=_normalize_years(args.years),
        overwrite=bool(args.overwrite),
        workers=workers,
        verbose=bool(args.verbose),
        debug=bool(args.debug),
        execution_mode=execution_mode,
        spill_parent=(Path(args.spill_parent) if args.spill_parent is not None else None),
        memory_budget_mb=memory_budget_mb,
    )


def build_default_engine() -> SimpleWalkForwardEngine:
    """Compose the default production walk-forward engine for the CLI.

    Returns:
        ``SimpleWalkForwardEngine`` instance.
    """
    return SimpleWalkForwardEngine()


def build_registry(
    *,
    engines: Mapping[str, WalkForwardEngine] | None = None,
) -> WalkForwardEngineRegistry:
    """Compose a registry with default or injected walk-forward engine implementations.

    Args:
        engines: Optional mapping of registry names to engine instances.
            When ``None``, registers ``SimpleWalkForwardEngine`` under
            ``simple``.

    Returns:
        Fully populated ``WalkForwardEngineRegistry``.
    """
    registry = WalkForwardEngineRegistry()
    if engines is None:
        registry.register(_DEFAULT_ENGINE, build_default_engine())
    else:
        for name, engine in engines.items():
            registry.register(name, engine)
    return registry


def build_walk_forward_pipeline(
    options: WalkForwardGenerationOptions,
    *,
    engine_registry: WalkForwardEngineRegistry | None = None,
) -> WalkForwardPipeline:
    """Compose ``WalkForwardPipeline`` from injected engine registry dependencies.

    Args:
        options: Immutable generation options providing the engine name.
        engine_registry: Optional engine registry. When ``None``, a default
            registry containing ``SimpleWalkForwardEngine`` is built.

    Returns:
        Fully wired ``WalkForwardPipeline``.
    """
    if engine_registry is None:
        engine_registry = build_registry()
    elif options.engine == _DEFAULT_ENGINE and not engine_registry.exists(options.engine):
        engine_registry.register(options.engine, build_default_engine())
    return WalkForwardPipeline(engine_registry)


def discover_work(
    factor_selection_repository: FactorSelectionRepository,
    options: WalkForwardGenerationOptions,
) -> tuple[DiscoveredWorkItem, ...]:
    """Discover walk-forward-ready Factor Selection panels matching CLI filters.

    Only Factor Selection partitions that exist are scheduled. Missing Factor
    Selection partitions are never invented.

    Args:
        factor_selection_repository: Factor Selection repository providing
            discovery APIs.
        options: CLI filters for manager, timeframe, and year.

    Returns:
        Deterministically ordered discovered work items.
    """
    partitions = factor_selection_repository.discover_partitions(
        managers=(options.manager,),
        timeframes=options.timeframes,
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    return _group_partitions(partitions, year_filter=options.years)


def format_summary(summary: WalkForwardGenerationSummary) -> str:
    """Render a deterministic walk-forward-generation summary report.

    Args:
        summary: Aggregate generation summary.

    Returns:
        Multi-line summary text suitable for stdout.
    """
    lines = [
        "=====================================",
        "CQROS Walk-Forward Generation Summary",
        "=====================================",
        "",
        f"Manager: {summary.manager}",
        f"Engine: {summary.engine}",
        f"Execution mode: {summary.execution_mode}",
        "",
        f"Panels: {summary.panels}",
        f"Rows: {summary.rows}",
        f"Selected Factors: {summary.selected_factors}",
        f"Pass Rows: {summary.pass_rows}",
        f"Fail Rows: {summary.fail_rows}",
        "",
        f"Duration: {_format_duration(summary.duration_seconds)}",
        "",
        f"Output directory: {_format_output_directory(summary.output_directory)}",
    ]
    if summary.failed_task_labels:
        lines.extend(["", "Failed Tasks", ""])
        lines.extend(f"- {label}" for label in summary.failed_task_labels)
    return "\n".join(lines) + "\n"


async def main(argv: Sequence[str] | None = None) -> int:
    """Run the walk-forward-generation CLI.

    Args:
        argv: Optional argument vector. When ``None``, ``sys.argv[1:]`` is used.

    Returns:
        ``0`` on completion; ``1`` when a fatal CLI error occurs or any task
        failed.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        options = build_options(args)
        _configure_logging(verbose=options.verbose, debug=options.debug)
        layout = StorageLayout(options.storage_root)
        datastore = ParquetStore()
        factor_selection_repository = FactorSelectionRepository(layout, datastore)
        walk_forward_repository = WalkForwardRepository(layout, datastore)
        factors_repository = FactorsRepository(layout, datastore)
        label_repository = LabelRepository(layout, datastore)
        walk_forward_input_builder = WalkForwardInputBuilder(
            factors_repository,
            label_repository,
        )
        memory_efficient_executor = (
            MemoryEfficientWalkForwardExecutor(
                layout,
                factors_repository,
                label_repository,
                build_default_engine(),
                MemoryEfficientExecutionConfig(
                    spill_parent=(
                        options.spill_parent
                        if options.spill_parent is not None
                        else options.storage_root / ".cqros_tmp" / "walk_forward_spill"
                    ),
                    memory_budget_mb=options.memory_budget_mb,
                ),
            )
            if options.execution_mode == MEMORY_EFFICIENT_EXECUTION_MODE
            else None
        )
        pipeline = build_walk_forward_pipeline(options)
        work = discover_work(factor_selection_repository, options)
        summary = await run_generation(
            pipeline=pipeline,
            factor_selection_repository=factor_selection_repository,
            walk_forward_repository=walk_forward_repository,
            walk_forward_input_builder=walk_forward_input_builder,
            options=options,
            work=work,
            memory_efficient_executor=memory_efficient_executor,
        )
    except CQROSError as exc:
        print(str(exc), file=sys.stderr)
        return _EXIT_FAILURE
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return _EXIT_FAILURE

    print(format_summary(summary), end="")
    return _EXIT_SUCCESS if summary.failed_tasks == 0 else _EXIT_FAILURE


async def run_generation(
    *,
    pipeline: WalkForwardPipeline,
    factor_selection_repository: FactorSelectionRepository,
    walk_forward_repository: WalkForwardRepository,
    walk_forward_input_builder: WalkForwardInputBuilder,
    options: WalkForwardGenerationOptions,
    work: Sequence[DiscoveredWorkItem],
    memory_efficient_executor: MemoryEfficientWalkForwardExecutor | None = None,
) -> WalkForwardGenerationSummary:
    """Execute discovered work through a bounded panel worker pool.

    Args:
        pipeline: Injected walk-forward pipeline.
        factor_selection_repository: Factor Selection partition repository.
        walk_forward_repository: Walk-forward partition repository.
        walk_forward_input_builder: Adapter that attaches Labels
            ``future_return_1`` onto Factor Selection observations.
        options: Immutable generation options.
        work: Discovered work items.

    Returns:
        Aggregate immutable summary.
    """
    started = time.perf_counter()
    output_directory = options.storage_root / STORAGE_DIR_WALK_FORWARD
    if options.execution_mode not in (
        FULL_PANEL_EXECUTION_MODE,
        MEMORY_EFFICIENT_EXECUTION_MODE,
    ):
        raise ValidationError(
            "unsupported walk-forward execution mode",
            error_code=_ERROR_EXECUTION_MODE,
            details={"execution_mode": options.execution_mode},
        )
    _logger.info(
        "Configured walk-forward generation",
        extra={
            "execution_mode": options.execution_mode,
            "workers": options.workers,
            "memory_budget_mb": options.memory_budget_mb,
        },
    )

    if len(work) == 0:
        return WalkForwardGenerationSummary(
            manager=options.manager,
            engine=options.engine,
            panels=0,
            rows=0,
            selected_factors=0,
            pass_rows=0,
            fail_rows=0,
            successful_tasks=0,
            failed_tasks=0,
            skipped_tasks=0,
            duration_seconds=time.perf_counter() - started,
            output_directory=output_directory,
            failed_task_labels=(),
            execution_mode=options.execution_mode,
        )

    work_by_timeframe = _group_work_by_timeframe(work)
    results = await _run_worker_pool(
        pipeline=pipeline,
        factor_selection_repository=factor_selection_repository,
        walk_forward_repository=walk_forward_repository,
        walk_forward_input_builder=walk_forward_input_builder,
        work_by_timeframe=work_by_timeframe,
        worker_count=options.workers,
        overwrite=options.overwrite,
        debug=options.debug,
        manager_name=options.manager,
        engine_name=options.engine,
        execution_mode=options.execution_mode,
        memory_efficient_executor=memory_efficient_executor,
    )
    return _build_summary(
        options=options,
        work=work,
        results=results,
        duration_seconds=time.perf_counter() - started,
        output_directory=output_directory,
    )


def _configure_logging(*, verbose: bool, debug: bool) -> None:
    """Configure process logging for the CLI entry point."""
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger("cqros").setLevel(level)


def _normalize_timeframes(
    values: Sequence[str] | None,
) -> tuple[Timeframe, ...] | None:
    """Validate and freeze optional timeframe filters."""
    if values is None:
        return None
    normalized: list[Timeframe] = []
    for timeframe in values:
        if timeframe not in SUPPORTED_TIMEFRAMES:
            raise ValidationError(
                f"unsupported timeframe: {timeframe}",
                error_code=_ERROR_TIMEFRAME,
                details={"parameter": "timeframes", "value": timeframe},
            )
        if timeframe not in normalized:
            normalized.append(timeframe)
    return tuple(normalized) if normalized else None


def _normalize_years(values: Sequence[str] | None) -> tuple[int, ...] | None:
    """Validate and freeze optional year filters."""
    if values is None:
        return None
    normalized: list[int] = []
    for raw in values:
        try:
            year = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"invalid year: {raw}",
                error_code=_ERROR_YEAR,
                details={"parameter": "years", "value": raw},
            ) from exc
        if year < 1:
            raise ValidationError(
                f"invalid year: {raw}",
                error_code=_ERROR_YEAR,
                details={"parameter": "years", "value": raw},
            )
        if year not in normalized:
            normalized.append(year)
    return tuple(sorted(normalized)) if normalized else None


def _group_partitions(
    partitions: Sequence[FactorSelectionPartitionRef],
    *,
    year_filter: tuple[int, ...] | None,
) -> tuple[DiscoveredWorkItem, ...]:
    """Group Factor Selection year partitions into manager/timeframe work items."""
    year_allowlist = set(year_filter) if year_filter is not None else None
    grouped: dict[tuple[str, str], list[int]] = {}
    for partition in partitions:
        if year_allowlist is not None and partition.year not in year_allowlist:
            continue
        key = (partition.manager, partition.timeframe)
        grouped.setdefault(key, []).append(partition.year)

    items: list[DiscoveredWorkItem] = []
    for (manager, timeframe), years in grouped.items():
        items.append(
            DiscoveredWorkItem(
                manager=manager,
                timeframe=timeframe,
                years=tuple(sorted(years)),
            )
        )
    return tuple(
        sorted(
            items,
            key=lambda item: (item.manager, item.timeframe),
        )
    )


def _group_work_by_timeframe(
    work: Sequence[DiscoveredWorkItem],
) -> dict[Timeframe, tuple[DiscoveredWorkItem, ...]]:
    """Group discovered work by timeframe while preserving discovery order."""
    grouped: dict[Timeframe, list[DiscoveredWorkItem]] = {}
    for item in work:
        grouped.setdefault(item.timeframe, []).append(item)
    return {timeframe: tuple(items) for timeframe, items in grouped.items()}


async def _run_worker_pool(
    *,
    pipeline: WalkForwardPipeline,
    factor_selection_repository: FactorSelectionRepository,
    walk_forward_repository: WalkForwardRepository,
    walk_forward_input_builder: WalkForwardInputBuilder,
    work_by_timeframe: Mapping[Timeframe, Sequence[DiscoveredWorkItem]],
    worker_count: int,
    overwrite: bool,
    debug: bool,
    manager_name: str,
    engine_name: str,
    execution_mode: str,
    memory_efficient_executor: MemoryEfficientWalkForwardExecutor | None,
) -> tuple[WalkForwardTaskResult, ...]:
    """Drain timeframes through a bounded asyncio worker pool."""
    timeframes = tuple(work_by_timeframe.keys())
    if len(timeframes) == 0:
        return ()

    queue: asyncio.Queue[Timeframe | None] = asyncio.Queue()
    for timeframe in timeframes:
        queue.put_nowait(timeframe)
    for _ in range(worker_count):
        queue.put_nowait(None)

    collected: list[WalkForwardTaskResult] = []
    lock = asyncio.Lock()

    async def worker() -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                results = await _generate_timeframe_work(
                    pipeline=pipeline,
                    factor_selection_repository=factor_selection_repository,
                    walk_forward_repository=walk_forward_repository,
                    walk_forward_input_builder=walk_forward_input_builder,
                    timeframe=item,
                    work_items=work_by_timeframe[item],
                    overwrite=overwrite,
                    debug=debug,
                    manager_name=manager_name,
                    engine_name=engine_name,
                    execution_mode=execution_mode,
                    memory_efficient_executor=memory_efficient_executor,
                )
                async with lock:
                    collected.extend(results)
            finally:
                queue.task_done()

    worker_tasks = [
        asyncio.create_task(worker(), name=f"generate-walk-forward-worker-{index}")
        for index in range(worker_count)
    ]
    try:
        await asyncio.gather(*worker_tasks)
    finally:
        for task in worker_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*worker_tasks, return_exceptions=True)

    return tuple(
        sorted(
            collected,
            key=lambda result: (result.timeframe, result.year),
        )
    )


async def _generate_timeframe_work(
    *,
    pipeline: WalkForwardPipeline,
    factor_selection_repository: FactorSelectionRepository,
    walk_forward_repository: WalkForwardRepository,
    walk_forward_input_builder: WalkForwardInputBuilder,
    timeframe: Timeframe,
    work_items: Sequence[DiscoveredWorkItem],
    overwrite: bool,
    debug: bool,
    manager_name: str,
    engine_name: str,
    execution_mode: str,
    memory_efficient_executor: MemoryEfficientWalkForwardExecutor | None,
) -> tuple[WalkForwardTaskResult, ...]:
    """Generate walk-forward datasets for every discovered year for one timeframe."""
    results: list[WalkForwardTaskResult] = []
    for item in work_items:
        for year in item.years:
            result = await asyncio.to_thread(
                _generate_partition,
                pipeline,
                factor_selection_repository,
                walk_forward_repository,
                walk_forward_input_builder,
                manager=item.manager,
                timeframe=timeframe,
                year=year,
                overwrite=overwrite,
                debug=debug,
                manager_name=manager_name,
                engine_name=engine_name,
                execution_mode=execution_mode,
                memory_efficient_executor=memory_efficient_executor,
            )
            _print_progress(result)
            results.append(result)
    return tuple(results)


def _generate_partition(
    pipeline: WalkForwardPipeline,
    factor_selection_repository: FactorSelectionRepository,
    walk_forward_repository: WalkForwardRepository,
    walk_forward_input_builder: WalkForwardInputBuilder,
    *,
    manager: str,
    timeframe: Timeframe,
    year: int,
    overwrite: bool,
    debug: bool,
    manager_name: str,
    engine_name: str,
    execution_mode: str,
    memory_efficient_executor: MemoryEfficientWalkForwardExecutor | None,
) -> WalkForwardTaskResult:
    """Generate one walk-forward year partition synchronously."""
    if not overwrite and walk_forward_repository.exists(
        manager=manager_name,
        exchange=_EXCHANGE,
        market=_MARKET,
        timeframe=timeframe,
        year=year,
    ):
        return WalkForwardTaskResult(
            timeframe=timeframe,
            year=year,
            status="skipped",
        )

    try:
        if not factor_selection_repository.exists(
            manager=manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            timeframe=timeframe,
            year=year,
        ):
            raise ValidationError(
                f"factor selection partition missing for {manager}/{timeframe}/{year}",
                error_code=_ERROR_FACTOR_SELECTION_MISSING,
                details={
                    "manager": manager,
                    "timeframe": timeframe,
                    "year": year,
                    "missing": "factor_selection",
                },
            )

        factor_selection = factor_selection_repository.load(
            manager=manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            timeframe=timeframe,
            year=year,
        )
        if execution_mode == MEMORY_EFFICIENT_EXECUTION_MODE:
            if memory_efficient_executor is None:
                raise ValidationError(
                    "memory_efficient executor is not configured",
                    error_code=_ERROR_MEMORY_MODE_COMBINATION,
                    details={"execution_mode": execution_mode},
                )
            output = memory_efficient_executor.execute(
                factor_selection,
                manager=manager,
                exchange=_EXCHANGE,
                market=_MARKET,
                timeframe=timeframe,
                year=year,
            )
        else:
            evaluation_input = walk_forward_input_builder.build(
                factor_selection,
                manager=manager,
                exchange=_EXCHANGE,
                market=_MARKET,
                timeframe=timeframe,
                year=year,
            )
            output = pipeline.run(engine_name, evaluation_input)
        walk_forward_repository.save(
            output,
            manager=manager_name,
            exchange=_EXCHANGE,
            market=_MARKET,
            timeframe=timeframe,
            year=year,
        )
    except Exception as exc:
        _log_partition_failure(
            timeframe=timeframe,
            year=year,
            exc=exc,
            debug=debug,
        )
        return WalkForwardTaskResult(
            timeframe=timeframe,
            year=year,
            status="failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
            error_code=exc.error_code if isinstance(exc, CQROSError) else None,
        )

    rows_generated, selected_factors, pass_rows, fail_rows = _extract_partition_stats(output)
    return WalkForwardTaskResult(
        timeframe=timeframe,
        year=year,
        status="succeeded",
        rows_generated=rows_generated,
        selected_factors=selected_factors,
        pass_rows=pass_rows,
        fail_rows=fail_rows,
    )


def _extract_partition_stats(frame: pl.DataFrame) -> tuple[int, int, int, int]:
    """Extract row count and walk-forward aggregates from one metrics frame.

    Args:
        frame: Finalized walk-forward output DataFrame.

    Returns:
        A 4-tuple of ``(rows, selected_factors, pass_rows, fail_rows)``.
    """
    if frame.height == 0:
        return 0, 0, 0, 0

    selected_factors = int(frame.select(pl.col(_COL_SELECTED_FACTORS).sum()).item())
    pass_rows = int(
        frame.select((pl.col(_COL_STATUS) == WalkForwardStatus.PASS.value).sum()).item()
    )
    fail_rows = int(
        frame.select((pl.col(_COL_STATUS) == WalkForwardStatus.FAIL.value).sum()).item()
    )
    return frame.height, selected_factors, pass_rows, fail_rows


def _print_progress(result: WalkForwardTaskResult) -> None:
    """Print a deterministic one-line progress record for a task result."""
    label = f"{result.timeframe} {result.year}"
    if result.status == "succeeded":
        rows = result.rows_generated if result.rows_generated is not None else 0
        message = f"OK {label} rows={rows}"
    elif result.status == "skipped":
        message = f"SKIP {label}"
    else:
        error_type = result.error_type if result.error_type is not None else "Exception"
        message = f"FAIL {label} {error_type}"
    print(message, flush=True)


def _log_partition_failure(
    *,
    timeframe: Timeframe,
    year: int,
    exc: BaseException,
    debug: bool,
) -> None:
    """Log a partition generation failure without aborting the run."""
    log_extra = {
        "timeframe": timeframe,
        "year": year,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "error_code": exc.error_code if isinstance(exc, CQROSError) else None,
    }
    if debug:
        _logger.exception(
            "Failed walk-forward generation partition; continuing",
            extra=log_extra,
            exc_info=True,
        )
    else:
        _logger.warning(
            "Failed walk-forward generation partition; continuing",
            extra=log_extra,
        )


def _build_summary(
    *,
    options: WalkForwardGenerationOptions,
    work: Sequence[DiscoveredWorkItem],
    results: Sequence[WalkForwardTaskResult],
    duration_seconds: float,
    output_directory: Path,
) -> WalkForwardGenerationSummary:
    """Aggregate task results into a generation report."""
    panels_discovered = sum(len(item.years) for item in work)
    panels_processed: set[tuple[Timeframe, int]] = set()
    successful_tasks = 0
    failed_tasks = 0
    skipped_tasks = 0
    rows = 0
    selected_factors = 0
    pass_rows = 0
    fail_rows = 0
    failed_labels: set[str] = set()

    for result in results:
        panels_processed.add((result.timeframe, result.year))
        if result.status == "succeeded":
            successful_tasks += 1
            if result.rows_generated is not None:
                rows += result.rows_generated
            if result.selected_factors is not None:
                selected_factors += result.selected_factors
            if result.pass_rows is not None:
                pass_rows += result.pass_rows
            if result.fail_rows is not None:
                fail_rows += result.fail_rows
        elif result.status == "skipped":
            skipped_tasks += 1
        else:
            failed_tasks += 1
            failed_labels.add(f"{result.timeframe} {result.year}")

    return WalkForwardGenerationSummary(
        manager=options.manager,
        engine=options.engine,
        panels=len(panels_processed) if results else panels_discovered,
        rows=rows,
        selected_factors=selected_factors,
        pass_rows=pass_rows,
        fail_rows=fail_rows,
        successful_tasks=successful_tasks,
        failed_tasks=failed_tasks,
        skipped_tasks=skipped_tasks,
        duration_seconds=duration_seconds,
        output_directory=output_directory,
        failed_task_labels=tuple(sorted(failed_labels)),
        execution_mode=options.execution_mode,
    )


def _format_duration(duration_seconds: float) -> str:
    """Format duration seconds for the summary report."""
    return f"{duration_seconds:.3f}s"


def _format_output_directory(path: Path) -> str:
    """Format the output directory using POSIX separators."""
    return path.as_posix()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
