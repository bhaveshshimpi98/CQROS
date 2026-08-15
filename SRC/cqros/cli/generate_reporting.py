"""CQROS reporting-generation CLI.

Purpose:
    Provide an argparse-based production entry point that discovers canonical
    analytics partitions and executes ``ReportingPipeline`` across the
    universe with bounded symbol concurrency, persisting reporting metadata
    through ``ReportingRepository``.

Responsibilities:
    - Parse CLI arguments for reporting dataset generation
    - Discover available analytics partitions through ``AnalyticsRepository``
    - Load matching analytics partitions for each discovered partition
    - Resolve ``--engine`` through ``ReportingEngineRegistry``
    - Execute ``ReportingPipeline`` and persist via ``ReportingRepository``
    - Honor ``--overwrite``, worker concurrency, and debug logging
    - Print deterministic progress and a final summary

Dependencies:
    ``argparse``, ``asyncio``, ``polars``, ``cqros.config``, ``cqros.core``,
    ``cqros.analytics``, ``cqros.reporting``, and ``cqros.storage``.

Public API:
    ``build_parser``, ``build_options``, ``build_default_engine``,
    ``build_registry``, ``build_reporting_pipeline``, ``discover_work``,
    ``format_summary``, ``run_generation``, and ``main``.

Notes:
    This module is a thin composition root. It does not implement reporting
    logic, schema validation, or repository filesystem walks beyond calling
    repository discovery and load/save APIs. Metadata assembly is delegated
    exclusively to ``ReportingPipeline``. Persistence remains in the CLI
    because ``ReportingPipeline`` does not own a repository.
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

from cqros.analytics import AnalyticsRepository
from cqros.analytics.repository import AnalyticsPartitionRef
from cqros.config.models import ResearchConfig
from cqros.core.constants import (
    DEFAULT_STORAGE_ROOT,
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
    STORAGE_DIR_REPORTING,
    SUPPORTED_TIMEFRAMES,
)
from cqros.core.exceptions import CQROSError, ValidationError
from cqros.core.types import Symbol, Timeframe
from cqros.reporting import (
    ReportingEngine,
    ReportingEngineRegistry,
    ReportingPipeline,
    ReportingRepository,
    ReportingStatus,
    SimpleReportingEngine,
)
from cqros.storage import ParquetStore, StorageLayout

__all__ = [
    "DiscoveredWorkItem",
    "ReportingGenerationOptions",
    "ReportingGenerationSummary",
    "ReportingTaskResult",
    "build_default_engine",
    "build_options",
    "build_parser",
    "build_registry",
    "build_reporting_pipeline",
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

_ERROR_WORKERS: Final[str] = "CLI-GENERATE-REPORTING-001"
_ERROR_TIMEFRAME: Final[str] = "CLI-GENERATE-REPORTING-002"
_ERROR_YEAR: Final[str] = "CLI-GENERATE-REPORTING-003"
_ERROR_MANAGER: Final[str] = "CLI-GENERATE-REPORTING-004"
_ERROR_ENGINE: Final[str] = "CLI-GENERATE-REPORTING-005"
_ERROR_ANALYTICS_MISSING: Final[str] = "CLI-GENERATE-REPORTING-006"

_COL_STATUS: Final[str] = "status"


@dataclass(frozen=True, slots=True)
class ReportingGenerationOptions:
    """Immutable CLI options for reporting dataset generation.

    Attributes:
        storage_root: Storage root containing ``analytics`` and
            ``reporting``.
        manager: Order manager identity used for discovery and reporting
            lineage.
        engine: Registry key of the reporting engine to execute.
        symbols: Optional symbol allowlist. ``None`` discovers all.
        timeframes: Optional timeframe allowlist. ``None`` discovers all.
        years: Optional year allowlist. ``None`` discovers all.
        overwrite: When ``True``, regenerate existing reporting partitions.
        workers: Maximum concurrent symbols.
        verbose: When ``True``, enable INFO logging.
        debug: When ``True``, enable DEBUG logging and log full failure
            tracebacks.
    """

    storage_root: Path
    manager: str
    engine: str
    symbols: tuple[Symbol, ...] | None
    timeframes: tuple[Timeframe, ...] | None
    years: tuple[int, ...] | None
    overwrite: bool
    workers: int
    verbose: bool
    debug: bool


@dataclass(frozen=True, slots=True)
class DiscoveredWorkItem:
    """One discovered analytics partition group ready for reporting generation.

    Attributes:
        manager: Order manager identifier of the source partitions.
        symbol: Tradeable symbol.
        timeframe: Available bar interval.
        years: Calendar years with existing analytics parquet partitions.
    """

    manager: str
    symbol: Symbol
    timeframe: Timeframe
    years: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ReportingTaskResult:
    """Immutable result for one symbol/timeframe/year generation task.

    Attributes:
        symbol: Tradeable symbol.
        timeframe: Bar interval.
        year: Calendar year of the partition.
        status: ``succeeded``, ``failed``, or ``skipped``.
        rows_generated: Output row count when succeeded.
        generated_rows: Count of rows with ``GENERATED`` status on success.
        failed_status_rows: Count of rows with ``FAILED`` status on success.
        error_type: Exception type name when failed.
        error_message: Exception message when failed.
        error_code: Optional CQROS error code when failed.
    """

    symbol: Symbol
    timeframe: Timeframe
    year: int
    status: str
    rows_generated: int | None = None
    generated_rows: int | None = None
    failed_status_rows: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class ReportingGenerationSummary:
    """Immutable aggregate summary for a reporting-generation run.

    Attributes:
        manager: Order manager identity used for generation.
        engine: Reporting engine registry key used for generation.
        symbols: Unique symbols for which generation was attempted.
        rows: Sum of output rows across successes.
        generated_rows: Sum of rows with ``GENERATED`` status across successes.
        failed_status_rows: Sum of rows with ``FAILED`` status across
            successes.
        successful_tasks: Count of succeeded partition tasks.
        failed_tasks: Count of failed partition tasks.
        skipped_tasks: Count of skipped existing partitions.
        duration_seconds: Wall-clock generation duration.
        output_directory: Reporting-tier output directory.
        failed_task_labels: Deterministic failed-task labels for reporting.
    """

    manager: str
    engine: str
    symbols: int
    rows: int
    generated_rows: int
    failed_status_rows: int
    successful_tasks: int
    failed_tasks: int
    skipped_tasks: int
    duration_seconds: float
    output_directory: Path
    failed_task_labels: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    """Create the reporting-generation argument parser.

    Returns:
        Configured ``ArgumentParser`` for reporting-generation flags.
    """
    parser = argparse.ArgumentParser(
        prog="cqros-generate-reporting",
        description=(
            "Generate CQROS reporting metadata datasets from discovered "
            "analytics partitions and an injected reporting engine."
        ),
    )
    parser.add_argument(
        "--manager",
        dest="manager",
        required=True,
        metavar="NAME",
        help="Order manager identity used for discovery and reporting lineage.",
    )
    parser.add_argument(
        "--engine",
        dest="engine",
        default=_DEFAULT_ENGINE,
        metavar="NAME",
        help=f"Reporting engine registry key (default: {_DEFAULT_ENGINE}).",
    )
    parser.add_argument(
        "--symbols",
        dest="symbols",
        nargs="*",
        default=None,
        metavar="SYMBOL",
        help="Optional symbol allowlist (0..N values). Omit to discover all.",
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
        help="Regenerate reporting partitions that already exist.",
    )
    parser.add_argument(
        "--workers",
        dest="workers",
        type=int,
        default=_DEFAULT_WORKER_COUNT,
        metavar="INT",
        help=f"Maximum concurrent symbols (default: {_DEFAULT_WORKER_COUNT}).",
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


def build_options(args: argparse.Namespace) -> ReportingGenerationOptions:
    """Map parsed CLI arguments onto ``ReportingGenerationOptions``.

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

    return ReportingGenerationOptions(
        storage_root=storage_root,
        manager=manager,
        engine=engine,
        symbols=_normalize_symbols(args.symbols),
        timeframes=_normalize_timeframes(args.timeframes),
        years=_normalize_years(args.years),
        overwrite=bool(args.overwrite),
        workers=workers,
        verbose=bool(args.verbose),
        debug=bool(args.debug),
    )


def build_default_engine() -> ReportingEngine:
    """Compose the default production reporting engine for the CLI.

    Returns:
        ``SimpleReportingEngine`` instance.
    """
    return SimpleReportingEngine()


def build_registry(
    *,
    engines: Mapping[str, ReportingEngine] | None = None,
) -> ReportingEngineRegistry:
    """Compose a registry with default or injected reporting engine implementations.

    Args:
        engines: Optional mapping of registry names to engine instances.
            When ``None``, registers ``SimpleReportingEngine`` under
            ``simple``.

    Returns:
        Fully populated ``ReportingEngineRegistry``.
    """
    registry = ReportingEngineRegistry()
    if engines is None:
        registry.register(_DEFAULT_ENGINE, build_default_engine())
    else:
        for name, engine in engines.items():
            registry.register(name, engine)
    return registry


def build_reporting_pipeline(
    options: ReportingGenerationOptions,
    *,
    engine_registry: ReportingEngineRegistry | None = None,
) -> ReportingPipeline:
    """Compose ``ReportingPipeline`` from injected engine registry dependencies.

    Args:
        options: Immutable generation options providing the engine name.
        engine_registry: Optional engine registry. When ``None``, a default
            registry containing ``SimpleReportingEngine`` is built.

    Returns:
        Fully wired ``ReportingPipeline``.
    """
    if engine_registry is None:
        engine_registry = build_registry()
    elif options.engine == _DEFAULT_ENGINE and not engine_registry.exists(options.engine):
        engine_registry.register(options.engine, build_default_engine())
    return ReportingPipeline(engine_registry)


def discover_work(
    analytics_repository: AnalyticsRepository,
    options: ReportingGenerationOptions,
) -> tuple[DiscoveredWorkItem, ...]:
    """Discover reporting-ready analytics partitions matching CLI filters.

    Only analytics partitions that exist are scheduled. Missing analytics
    partitions are never invented.

    Args:
        analytics_repository: Analytics repository providing discovery APIs.
        options: CLI filters for manager, symbol, timeframe, and year.

    Returns:
        Deterministically ordered discovered work items.
    """
    partitions = analytics_repository.discover_partitions(
        managers=(options.manager,),
        symbols=options.symbols,
        timeframes=options.timeframes,
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    return _group_partitions(partitions, year_filter=options.years)


def format_summary(summary: ReportingGenerationSummary) -> str:
    """Render a deterministic reporting-generation summary report.

    Args:
        summary: Aggregate generation summary.

    Returns:
        Multi-line summary text suitable for stdout.
    """
    lines = [
        "=====================================",
        "CQROS Reporting Generation Summary",
        "=====================================",
        "",
        f"Manager: {summary.manager}",
        f"Engine: {summary.engine}",
        "",
        f"Symbols: {summary.symbols}",
        f"Rows: {summary.rows}",
        f"Generated: {summary.generated_rows}",
        f"Failed Status: {summary.failed_status_rows}",
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
    """Run the reporting-generation CLI.

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
        analytics_repository = AnalyticsRepository(layout, datastore)
        reporting_repository = ReportingRepository(layout, datastore)
        pipeline = build_reporting_pipeline(options)
        work = discover_work(analytics_repository, options)
        summary = await run_generation(
            pipeline=pipeline,
            analytics_repository=analytics_repository,
            reporting_repository=reporting_repository,
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
    pipeline: ReportingPipeline,
    analytics_repository: AnalyticsRepository,
    reporting_repository: ReportingRepository,
    options: ReportingGenerationOptions,
    work: Sequence[DiscoveredWorkItem],
) -> ReportingGenerationSummary:
    """Execute discovered work through a bounded symbol worker pool.

    Args:
        pipeline: Injected reporting pipeline.
        analytics_repository: Analytics partition repository (input tier).
        reporting_repository: Reporting partition repository (output tier).
        options: Immutable generation options.
        work: Discovered work items.

    Returns:
        Aggregate immutable summary.
    """
    started = time.perf_counter()
    output_directory = options.storage_root / STORAGE_DIR_REPORTING

    if len(work) == 0:
        return ReportingGenerationSummary(
            manager=options.manager,
            engine=options.engine,
            symbols=0,
            rows=0,
            generated_rows=0,
            failed_status_rows=0,
            successful_tasks=0,
            failed_tasks=0,
            skipped_tasks=0,
            duration_seconds=time.perf_counter() - started,
            output_directory=output_directory,
            failed_task_labels=(),
        )

    work_by_symbol = _group_work_by_symbol(work)
    results = await _run_worker_pool(
        pipeline=pipeline,
        analytics_repository=analytics_repository,
        reporting_repository=reporting_repository,
        work_by_symbol=work_by_symbol,
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


def _normalize_symbols(values: Sequence[str] | None) -> tuple[Symbol, ...] | None:
    """Validate and freeze optional symbol filters."""
    if values is None:
        return None
    normalized: list[Symbol] = []
    for symbol in values:
        stripped = symbol.strip()
        if stripped == "":
            continue
        if stripped not in normalized:
            normalized.append(stripped)
    return tuple(normalized) if normalized else None


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
    partitions: Sequence[AnalyticsPartitionRef],
    *,
    year_filter: tuple[int, ...] | None,
) -> tuple[DiscoveredWorkItem, ...]:
    """Group analytics year partitions into manager/symbol/timeframe work items."""
    year_allowlist = set(year_filter) if year_filter is not None else None
    grouped: dict[tuple[str, str, str], list[int]] = {}
    for partition in partitions:
        if year_allowlist is not None and partition.year not in year_allowlist:
            continue
        key = (partition.manager, partition.symbol, partition.timeframe)
        grouped.setdefault(key, []).append(partition.year)

    items: list[DiscoveredWorkItem] = []
    for (manager, symbol, timeframe), years in grouped.items():
        items.append(
            DiscoveredWorkItem(
                manager=manager,
                symbol=symbol,
                timeframe=timeframe,
                years=tuple(sorted(years)),
            )
        )
    return tuple(
        sorted(
            items,
            key=lambda item: (item.manager, item.symbol, item.timeframe),
        )
    )


def _group_work_by_symbol(
    work: Sequence[DiscoveredWorkItem],
) -> dict[Symbol, tuple[DiscoveredWorkItem, ...]]:
    """Group discovered work by symbol while preserving discovery order."""
    grouped: dict[Symbol, list[DiscoveredWorkItem]] = {}
    for item in work:
        grouped.setdefault(item.symbol, []).append(item)
    return {symbol: tuple(items) for symbol, items in grouped.items()}


async def _run_worker_pool(
    *,
    pipeline: ReportingPipeline,
    analytics_repository: AnalyticsRepository,
    reporting_repository: ReportingRepository,
    work_by_symbol: Mapping[Symbol, Sequence[DiscoveredWorkItem]],
    worker_count: int,
    overwrite: bool,
    debug: bool,
    manager_name: str,
    engine_name: str,
) -> tuple[ReportingTaskResult, ...]:
    """Drain symbols through a bounded asyncio worker pool."""
    symbols = tuple(work_by_symbol.keys())
    if len(symbols) == 0:
        return ()

    queue: asyncio.Queue[Symbol | None] = asyncio.Queue()
    for symbol in symbols:
        queue.put_nowait(symbol)
    for _ in range(worker_count):
        queue.put_nowait(None)

    collected: list[ReportingTaskResult] = []
    lock = asyncio.Lock()

    async def worker() -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                results = await _generate_symbol_work(
                    pipeline=pipeline,
                    analytics_repository=analytics_repository,
                    reporting_repository=reporting_repository,
                    symbol=item,
                    work_items=work_by_symbol[item],
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
        asyncio.create_task(worker(), name=f"generate-reporting-worker-{index}")
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
            key=lambda result: (result.symbol, result.timeframe, result.year),
        )
    )


async def _generate_symbol_work(
    *,
    pipeline: ReportingPipeline,
    analytics_repository: AnalyticsRepository,
    reporting_repository: ReportingRepository,
    symbol: Symbol,
    work_items: Sequence[DiscoveredWorkItem],
    overwrite: bool,
    debug: bool,
    manager_name: str,
    engine_name: str,
) -> tuple[ReportingTaskResult, ...]:
    """Generate reporting datasets for every discovered year for one symbol."""
    results: list[ReportingTaskResult] = []
    for item in work_items:
        for year in item.years:
            result = await asyncio.to_thread(
                _generate_partition,
                pipeline,
                analytics_repository,
                reporting_repository,
                manager=item.manager,
                symbol=symbol,
                timeframe=item.timeframe,
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
    pipeline: ReportingPipeline,
    analytics_repository: AnalyticsRepository,
    reporting_repository: ReportingRepository,
    *,
    manager: str,
    symbol: Symbol,
    timeframe: Timeframe,
    year: int,
    overwrite: bool,
    debug: bool,
    manager_name: str,
    engine_name: str,
) -> ReportingTaskResult:
    """Generate one reporting year partition synchronously."""
    if not overwrite and reporting_repository.exists(
        manager=manager_name,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=symbol,
        timeframe=timeframe,
        year=year,
    ):
        return ReportingTaskResult(
            symbol=symbol,
            timeframe=timeframe,
            year=year,
            status="skipped",
        )

    try:
        if not analytics_repository.exists(
            manager=manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        ):
            raise ValidationError(
                f"analytics partition missing for {manager}/{symbol}/{timeframe}/{year}",
                error_code=_ERROR_ANALYTICS_MISSING,
                details={
                    "manager": manager,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "year": year,
                    "missing": "analytics",
                },
            )

        analytics = analytics_repository.load(
            manager=manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        )
        output = pipeline.run(engine_name, analytics)
        reporting_repository.save(
            output,
            manager=manager_name,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        )
    except Exception as exc:
        _log_partition_failure(
            symbol=symbol,
            timeframe=timeframe,
            year=year,
            exc=exc,
            debug=debug,
        )
        return ReportingTaskResult(
            symbol=symbol,
            timeframe=timeframe,
            year=year,
            status="failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
            error_code=exc.error_code if isinstance(exc, CQROSError) else None,
        )

    rows_generated, generated_rows, failed_status_rows = _extract_partition_stats(output)
    return ReportingTaskResult(
        symbol=symbol,
        timeframe=timeframe,
        year=year,
        status="succeeded",
        rows_generated=rows_generated,
        generated_rows=generated_rows,
        failed_status_rows=failed_status_rows,
    )


def _extract_partition_stats(frame: pl.DataFrame) -> tuple[int, int, int]:
    """Extract row count and reporting-status counts from one reporting frame.

    Args:
        frame: Finalized reporting output DataFrame.

    Returns:
        A 3-tuple of ``(rows, generated_rows, failed_status_rows)``.
    """
    if frame.height == 0:
        return 0, 0, 0

    generated_rows = int(
        frame.select((pl.col(_COL_STATUS) == ReportingStatus.GENERATED.value).sum()).item()
    )
    failed_status_rows = int(
        frame.select((pl.col(_COL_STATUS) == ReportingStatus.FAILED.value).sum()).item()
    )
    return frame.height, generated_rows, failed_status_rows


def _print_progress(result: ReportingTaskResult) -> None:
    """Print a deterministic one-line progress record for a task result."""
    label = f"{result.symbol} {result.timeframe} {result.year}"
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
    symbol: Symbol,
    timeframe: Timeframe,
    year: int,
    exc: BaseException,
    debug: bool,
) -> None:
    """Log a partition generation failure without aborting the run."""
    log_extra = {
        "symbol": symbol,
        "timeframe": timeframe,
        "year": year,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "error_code": exc.error_code if isinstance(exc, CQROSError) else None,
    }
    if debug:
        _logger.exception(
            "Failed reporting generation partition; continuing",
            extra=log_extra,
            exc_info=True,
        )
    else:
        _logger.warning(
            "Failed reporting generation partition; continuing",
            extra=log_extra,
        )


def _build_summary(
    *,
    options: ReportingGenerationOptions,
    work: Sequence[DiscoveredWorkItem],
    results: Sequence[ReportingTaskResult],
    duration_seconds: float,
    output_directory: Path,
) -> ReportingGenerationSummary:
    """Aggregate task results into a generation report."""
    symbols_discovered = {item.symbol for item in work}
    symbols_processed: set[Symbol] = set()
    successful_tasks = 0
    failed_tasks = 0
    skipped_tasks = 0
    rows = 0
    generated_rows = 0
    failed_status_rows = 0
    failed_labels: set[str] = set()

    for result in results:
        symbols_processed.add(result.symbol)
        if result.status == "succeeded":
            successful_tasks += 1
            if result.rows_generated is not None:
                rows += result.rows_generated
            if result.generated_rows is not None:
                generated_rows += result.generated_rows
            if result.failed_status_rows is not None:
                failed_status_rows += result.failed_status_rows
        elif result.status == "skipped":
            skipped_tasks += 1
        else:
            failed_tasks += 1
            failed_labels.add(f"{result.symbol} {result.timeframe} {result.year}")

    return ReportingGenerationSummary(
        manager=options.manager,
        engine=options.engine,
        symbols=len(symbols_processed) if results else len(symbols_discovered),
        rows=rows,
        generated_rows=generated_rows,
        failed_status_rows=failed_status_rows,
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
