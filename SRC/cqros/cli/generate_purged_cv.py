"""CQROS purged-CV-generation CLI.

Purpose:
    Provide an argparse-based production entry point that discovers canonical
    Walk-Forward panels and executes ``PurgedCVPipeline`` across
    timeframe/year panels with bounded concurrency, persisting purged-CV
    metrics through ``PurgedCVRepository``.

Responsibilities:
    - Parse CLI arguments for purged-CV dataset generation
    - Discover available Walk-Forward partitions through
      ``WalkForwardRepository``
    - Load matching Walk-Forward panels for each discovered partition
    - Resolve ``--engine`` through ``PurgedCVEngineRegistry``
    - Execute ``PurgedCVPipeline`` and persist via ``PurgedCVRepository``
    - Honor ``--overwrite``, worker concurrency, and debug logging
    - Print deterministic progress and a final summary

Dependencies:
    ``argparse``, ``asyncio``, ``polars``, ``cqros.config``, ``cqros.core``,
    ``cqros.walk_forward``, ``cqros.purged_cv``, and ``cqros.storage``.

Public API:
    ``build_parser``, ``build_options``, ``build_default_engine``,
    ``build_registry``, ``build_purged_cv_pipeline``,
    ``discover_work``, ``format_summary``, ``run_generation``, and ``main``.

Notes:
    This module is a thin composition root. It does not implement purge
    math, schema validation, or repository filesystem walks beyond calling
    repository discovery and load/save APIs. Metric computation is delegated
    exclusively to ``PurgedCVPipeline``. Persistence remains in the CLI
    because ``PurgedCVPipeline`` does not own a repository.
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
    STORAGE_DIR_PURGED_CV,
    SUPPORTED_TIMEFRAMES,
)
from cqros.core.exceptions import CQROSError, ValidationError
from cqros.core.types import Timeframe
from cqros.purged_cv import (
    PurgedCVEngine,
    PurgedCVEngineRegistry,
    PurgedCVPipeline,
    PurgedCVRepository,
    PurgedCVStatus,
    SimplePurgedCVEngine,
)
from cqros.storage import ParquetStore, StorageLayout
from cqros.walk_forward import WalkForwardPartitionRef, WalkForwardRepository

__all__ = [
    "DiscoveredWorkItem",
    "PurgedCVGenerationOptions",
    "PurgedCVGenerationSummary",
    "PurgedCVTaskResult",
    "build_default_engine",
    "build_options",
    "build_parser",
    "build_purged_cv_pipeline",
    "build_registry",
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

_EXCHANGE: Final[str] = EXCHANGE_BINANCE
_MARKET: Final[str] = MARKET_USDT_PERPETUAL

_ERROR_WORKERS: Final[str] = "CLI-GENERATE-PURGED-CV-001"
_ERROR_TIMEFRAME: Final[str] = "CLI-GENERATE-PURGED-CV-002"
_ERROR_YEAR: Final[str] = "CLI-GENERATE-PURGED-CV-003"
_ERROR_MANAGER: Final[str] = "CLI-GENERATE-PURGED-CV-004"
_ERROR_ENGINE: Final[str] = "CLI-GENERATE-PURGED-CV-005"
_ERROR_WALK_FORWARD_MISSING: Final[str] = "CLI-GENERATE-PURGED-CV-006"

_COL_STATUS: Final[str] = "status"


@dataclass(frozen=True, slots=True)
class PurgedCVGenerationOptions:
    """Immutable CLI options for purged-CV dataset generation.

    Attributes:
        storage_root: Storage root containing ``walk_forward`` and
            ``purged_cv``.
        manager: Order manager identity used for discovery and purged-CV
            lineage.
        engine: Registry key of the purged-CV engine to execute.
        timeframes: Optional timeframe allowlist. ``None`` discovers all.
        years: Optional year allowlist. ``None`` discovers all.
        overwrite: When ``True``, regenerate existing purged-CV partitions.
        workers: Maximum concurrent panels.
        verbose: When ``True``, enable INFO logging.
        debug: When ``True``, enable DEBUG logging and log full failure
            tracebacks.
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


@dataclass(frozen=True, slots=True)
class DiscoveredWorkItem:
    """One discovered Walk-Forward panel group ready for purged-CV generation.

    Attributes:
        manager: Order manager identifier of the source partitions.
        timeframe: Available bar interval.
        years: Calendar years with existing Walk-Forward parquet partitions.
    """

    manager: str
    timeframe: Timeframe
    years: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PurgedCVTaskResult:
    """Immutable result for one timeframe/year panel generation task.

    Attributes:
        timeframe: Bar interval.
        year: Calendar year of the partition.
        status: ``succeeded``, ``failed``, or ``skipped``.
        rows_generated: Output row count when succeeded.
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
    pass_rows: int | None = None
    fail_rows: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class PurgedCVGenerationSummary:
    """Immutable aggregate summary for a purged-CV-generation run.

    Attributes:
        manager: Order manager identity used for generation.
        engine: Purged-CV engine registry key used for generation.
        panels: Unique timeframe/year panels for which generation was attempted.
        rows: Sum of output rows across successes.
        pass_rows: Sum of rows with ``PASS`` status across successes.
        fail_rows: Sum of rows with ``FAIL`` status across successes.
        status: Aggregate run status (``SUCCESS`` or ``FAILED``).
        successful_tasks: Count of succeeded partition tasks.
        failed_tasks: Count of failed partition tasks.
        skipped_tasks: Count of skipped existing partitions.
        duration_seconds: Wall-clock generation duration.
        output_directory: Purged-CV-tier output directory.
        failed_task_labels: Deterministic failed-task labels for reporting.
    """

    manager: str
    engine: str
    panels: int
    rows: int
    pass_rows: int
    fail_rows: int
    status: str
    successful_tasks: int
    failed_tasks: int
    skipped_tasks: int
    duration_seconds: float
    output_directory: Path
    failed_task_labels: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    """Create the purged-CV-generation argument parser.

    Returns:
        Configured ``ArgumentParser`` for purged-CV-generation flags.
    """
    parser = argparse.ArgumentParser(
        prog="cqros-generate-purged-cv",
        description=(
            "Generate CQROS purged cross-validation datasets from discovered "
            "Walk-Forward panels and an injected purged-CV engine."
        ),
    )
    parser.add_argument(
        "--manager",
        dest="manager",
        required=True,
        metavar="NAME",
        help="Order manager identity used for discovery and purged-CV lineage.",
    )
    parser.add_argument(
        "--engine",
        dest="engine",
        default=_DEFAULT_ENGINE,
        metavar="NAME",
        help=f"Purged-CV engine registry key (default: {_DEFAULT_ENGINE}).",
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
        help="Regenerate purged-CV partitions that already exist.",
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


def build_options(args: argparse.Namespace) -> PurgedCVGenerationOptions:
    """Map parsed CLI arguments onto ``PurgedCVGenerationOptions``.

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

    return PurgedCVGenerationOptions(
        storage_root=storage_root,
        manager=manager,
        engine=engine,
        timeframes=_normalize_timeframes(args.timeframes),
        years=_normalize_years(args.years),
        overwrite=bool(args.overwrite),
        workers=workers,
        verbose=bool(args.verbose),
        debug=bool(args.debug),
    )


def build_default_engine() -> PurgedCVEngine:
    """Compose the default production purged-CV engine for the CLI.

    Returns:
        ``SimplePurgedCVEngine`` instance.
    """
    return SimplePurgedCVEngine()


def build_registry(
    *,
    engines: Mapping[str, PurgedCVEngine] | None = None,
) -> PurgedCVEngineRegistry:
    """Compose a registry with default or injected purged-CV engine implementations.

    Args:
        engines: Optional mapping of registry names to engine instances.
            When ``None``, registers ``SimplePurgedCVEngine`` under
            ``simple``.

    Returns:
        Fully populated ``PurgedCVEngineRegistry``.
    """
    registry = PurgedCVEngineRegistry()
    if engines is None:
        registry.register(_DEFAULT_ENGINE, build_default_engine())
    else:
        for name, engine in engines.items():
            registry.register(name, engine)
    return registry


def build_purged_cv_pipeline(
    options: PurgedCVGenerationOptions,
    *,
    engine_registry: PurgedCVEngineRegistry | None = None,
) -> PurgedCVPipeline:
    """Compose ``PurgedCVPipeline`` from injected engine registry dependencies.

    Args:
        options: Immutable generation options providing the engine name.
        engine_registry: Optional engine registry. When ``None``, a default
            registry containing ``SimplePurgedCVEngine`` is built.

    Returns:
        Fully wired ``PurgedCVPipeline``.
    """
    if engine_registry is None:
        engine_registry = build_registry()
    elif options.engine == _DEFAULT_ENGINE and not engine_registry.exists(options.engine):
        engine_registry.register(options.engine, build_default_engine())
    return PurgedCVPipeline(engine_registry)


def discover_work(
    walk_forward_repository: WalkForwardRepository,
    options: PurgedCVGenerationOptions,
) -> tuple[DiscoveredWorkItem, ...]:
    """Discover purged-CV-ready Walk-Forward panels matching CLI filters.

    Only Walk-Forward partitions that exist are scheduled. Missing Walk-Forward
    partitions are never invented.

    Args:
        walk_forward_repository: Walk-Forward repository providing discovery
            APIs.
        options: CLI filters for manager, timeframe, and year.

    Returns:
        Deterministically ordered discovered work items.
    """
    partitions = walk_forward_repository.discover_partitions(
        managers=(options.manager,),
        timeframes=options.timeframes,
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    return _group_partitions(partitions, year_filter=options.years)


def format_summary(summary: PurgedCVGenerationSummary) -> str:
    """Render a deterministic purged-CV-generation summary report.

    Args:
        summary: Aggregate generation summary.

    Returns:
        Multi-line summary text suitable for stdout.
    """
    lines = [
        "=====================================",
        "CQROS Purged-CV Generation Summary",
        "=====================================",
        "",
        f"Manager: {summary.manager}",
        f"Engine: {summary.engine}",
        "",
        f"Panels: {summary.panels}",
        f"Rows: {summary.rows}",
        f"Passed: {summary.pass_rows}",
        f"Failed: {summary.fail_rows}",
        f"Status: {summary.status}",
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
    """Run the purged-CV-generation CLI.

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
        walk_forward_repository = WalkForwardRepository(layout, datastore)
        purged_cv_repository = PurgedCVRepository(layout, datastore)
        pipeline = build_purged_cv_pipeline(options)
        work = discover_work(walk_forward_repository, options)
        summary = await run_generation(
            pipeline=pipeline,
            walk_forward_repository=walk_forward_repository,
            purged_cv_repository=purged_cv_repository,
            options=options,
            work=work,
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
    pipeline: PurgedCVPipeline,
    walk_forward_repository: WalkForwardRepository,
    purged_cv_repository: PurgedCVRepository,
    options: PurgedCVGenerationOptions,
    work: Sequence[DiscoveredWorkItem],
) -> PurgedCVGenerationSummary:
    """Execute discovered work through a bounded panel worker pool.

    Args:
        pipeline: Injected purged-CV pipeline.
        walk_forward_repository: Walk-Forward partition repository.
        purged_cv_repository: Purged-CV partition repository.
        options: Immutable generation options.
        work: Discovered work items.

    Returns:
        Aggregate immutable summary.
    """
    started = time.perf_counter()
    output_directory = options.storage_root / STORAGE_DIR_PURGED_CV

    if len(work) == 0:
        return PurgedCVGenerationSummary(
            manager=options.manager,
            engine=options.engine,
            panels=0,
            rows=0,
            pass_rows=0,
            fail_rows=0,
            status="SUCCESS",
            successful_tasks=0,
            failed_tasks=0,
            skipped_tasks=0,
            duration_seconds=time.perf_counter() - started,
            output_directory=output_directory,
            failed_task_labels=(),
        )

    work_by_timeframe = _group_work_by_timeframe(work)
    results = await _run_worker_pool(
        pipeline=pipeline,
        walk_forward_repository=walk_forward_repository,
        purged_cv_repository=purged_cv_repository,
        work_by_timeframe=work_by_timeframe,
        worker_count=options.workers,
        overwrite=options.overwrite,
        debug=options.debug,
        manager_name=options.manager,
        engine_name=options.engine,
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
    partitions: Sequence[WalkForwardPartitionRef],
    *,
    year_filter: tuple[int, ...] | None,
) -> tuple[DiscoveredWorkItem, ...]:
    """Group Walk-Forward year partitions into manager/timeframe work items."""
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
    pipeline: PurgedCVPipeline,
    walk_forward_repository: WalkForwardRepository,
    purged_cv_repository: PurgedCVRepository,
    work_by_timeframe: Mapping[Timeframe, Sequence[DiscoveredWorkItem]],
    worker_count: int,
    overwrite: bool,
    debug: bool,
    manager_name: str,
    engine_name: str,
) -> tuple[PurgedCVTaskResult, ...]:
    """Drain timeframes through a bounded asyncio worker pool."""
    timeframes = tuple(work_by_timeframe.keys())
    if len(timeframes) == 0:
        return ()

    queue: asyncio.Queue[Timeframe | None] = asyncio.Queue()
    for timeframe in timeframes:
        queue.put_nowait(timeframe)
    for _ in range(worker_count):
        queue.put_nowait(None)

    collected: list[PurgedCVTaskResult] = []
    lock = asyncio.Lock()

    async def worker() -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                results = await _generate_timeframe_work(
                    pipeline=pipeline,
                    walk_forward_repository=walk_forward_repository,
                    purged_cv_repository=purged_cv_repository,
                    timeframe=item,
                    work_items=work_by_timeframe[item],
                    overwrite=overwrite,
                    debug=debug,
                    manager_name=manager_name,
                    engine_name=engine_name,
                )
                async with lock:
                    collected.extend(results)
            finally:
                queue.task_done()

    worker_tasks = [
        asyncio.create_task(worker(), name=f"generate-purged-cv-worker-{index}")
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
    pipeline: PurgedCVPipeline,
    walk_forward_repository: WalkForwardRepository,
    purged_cv_repository: PurgedCVRepository,
    timeframe: Timeframe,
    work_items: Sequence[DiscoveredWorkItem],
    overwrite: bool,
    debug: bool,
    manager_name: str,
    engine_name: str,
) -> tuple[PurgedCVTaskResult, ...]:
    """Generate purged-CV datasets for every discovered year for one timeframe."""
    results: list[PurgedCVTaskResult] = []
    for item in work_items:
        for year in item.years:
            result = await asyncio.to_thread(
                _generate_partition,
                pipeline,
                walk_forward_repository,
                purged_cv_repository,
                manager=item.manager,
                timeframe=timeframe,
                year=year,
                overwrite=overwrite,
                debug=debug,
                manager_name=manager_name,
                engine_name=engine_name,
            )
            _print_progress(result)
            results.append(result)
    return tuple(results)


def _generate_partition(
    pipeline: PurgedCVPipeline,
    walk_forward_repository: WalkForwardRepository,
    purged_cv_repository: PurgedCVRepository,
    *,
    manager: str,
    timeframe: Timeframe,
    year: int,
    overwrite: bool,
    debug: bool,
    manager_name: str,
    engine_name: str,
) -> PurgedCVTaskResult:
    """Generate one purged-CV year partition synchronously."""
    if not overwrite and purged_cv_repository.exists(
        manager=manager_name,
        exchange=_EXCHANGE,
        market=_MARKET,
        timeframe=timeframe,
        year=year,
    ):
        return PurgedCVTaskResult(
            timeframe=timeframe,
            year=year,
            status="skipped",
        )

    try:
        if not walk_forward_repository.exists(
            manager=manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            timeframe=timeframe,
            year=year,
        ):
            raise ValidationError(
                f"walk-forward partition missing for {manager}/{timeframe}/{year}",
                error_code=_ERROR_WALK_FORWARD_MISSING,
                details={
                    "manager": manager,
                    "timeframe": timeframe,
                    "year": year,
                    "missing": "walk_forward",
                },
            )

        walk_forward = walk_forward_repository.load(
            manager=manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            timeframe=timeframe,
            year=year,
        )
        output = pipeline.build(walk_forward, engine=engine_name)
        purged_cv_repository.save(
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
        return PurgedCVTaskResult(
            timeframe=timeframe,
            year=year,
            status="failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
            error_code=exc.error_code if isinstance(exc, CQROSError) else None,
        )

    rows_generated, pass_rows, fail_rows = _extract_partition_stats(output)
    return PurgedCVTaskResult(
        timeframe=timeframe,
        year=year,
        status="succeeded",
        rows_generated=rows_generated,
        pass_rows=pass_rows,
        fail_rows=fail_rows,
    )


def _extract_partition_stats(frame: pl.DataFrame) -> tuple[int, int, int]:
    """Extract row count and purged-CV aggregates from one metrics frame.

    Args:
        frame: Finalized purged-CV output DataFrame.

    Returns:
        A 3-tuple of ``(rows, pass_rows, fail_rows)``.
    """
    if frame.height == 0:
        return 0, 0, 0

    pass_rows = int(frame.select((pl.col(_COL_STATUS) == PurgedCVStatus.PASS.value).sum()).item())
    fail_rows = int(frame.select((pl.col(_COL_STATUS) == PurgedCVStatus.FAIL.value).sum()).item())
    return frame.height, pass_rows, fail_rows


def _print_progress(result: PurgedCVTaskResult) -> None:
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
            "Failed purged-CV generation partition; continuing",
            extra=log_extra,
            exc_info=True,
        )
    else:
        _logger.warning(
            "Failed purged-CV generation partition; continuing",
            extra=log_extra,
        )


def _build_summary(
    *,
    options: PurgedCVGenerationOptions,
    work: Sequence[DiscoveredWorkItem],
    results: Sequence[PurgedCVTaskResult],
    duration_seconds: float,
    output_directory: Path,
) -> PurgedCVGenerationSummary:
    """Aggregate task results into a generation report."""
    panels_discovered = sum(len(item.years) for item in work)
    panels_processed: set[tuple[Timeframe, int]] = set()
    successful_tasks = 0
    failed_tasks = 0
    skipped_tasks = 0
    rows = 0
    pass_rows = 0
    fail_rows = 0
    failed_labels: set[str] = set()

    for result in results:
        panels_processed.add((result.timeframe, result.year))
        if result.status == "succeeded":
            successful_tasks += 1
            if result.rows_generated is not None:
                rows += result.rows_generated
            if result.pass_rows is not None:
                pass_rows += result.pass_rows
            if result.fail_rows is not None:
                fail_rows += result.fail_rows
        elif result.status == "skipped":
            skipped_tasks += 1
        else:
            failed_tasks += 1
            failed_labels.add(f"{result.timeframe} {result.year}")

    return PurgedCVGenerationSummary(
        manager=options.manager,
        engine=options.engine,
        panels=len(panels_processed) if results else panels_discovered,
        rows=rows,
        pass_rows=pass_rows,
        fail_rows=fail_rows,
        status="SUCCESS" if failed_tasks == 0 else "FAILED",
        successful_tasks=successful_tasks,
        failed_tasks=failed_tasks,
        skipped_tasks=skipped_tasks,
        duration_seconds=duration_seconds,
        output_directory=output_directory,
        failed_task_labels=tuple(sorted(failed_labels)),
    )


def _format_duration(duration_seconds: float) -> str:
    """Format duration seconds for the summary report."""
    return f"{duration_seconds:.3f}s"


def _format_output_directory(path: Path) -> str:
    """Format the output directory using POSIX separators."""
    return path.as_posix()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
