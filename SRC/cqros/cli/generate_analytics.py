"""CQROS analytics-generation CLI.

Purpose:
    Provide an argparse-based production entry point that discovers canonical
    performance partitions and executes ``AnalyticsPipeline`` across the
    universe with bounded symbol concurrency, persisting analytics metrics
    through ``AnalyticsRepository``.

Responsibilities:
    - Parse CLI arguments for analytics dataset generation
    - Discover available performance partitions through ``PerformanceRepository``
    - Load matching performance partitions for each discovered partition
    - Resolve ``--engine`` through ``AnalyticsEngineRegistry``
    - Execute ``AnalyticsPipeline`` and persist via ``AnalyticsRepository``
    - Honor ``--overwrite``, worker concurrency, and debug logging
    - Print deterministic progress and a final summary

Dependencies:
    ``argparse``, ``asyncio``, ``polars``, ``cqros.config``, ``cqros.core``,
    ``cqros.analytics``, ``cqros.performance``, and ``cqros.storage``.

Public API:
    ``build_parser``, ``build_options``, ``build_default_engine``,
    ``build_registry``, ``build_analytics_pipeline``, ``discover_work``,
    ``format_summary``, ``run_generation``, and ``main``.

Notes:
    This module is a thin composition root. It does not implement analytics
    math, schema validation, or repository filesystem walks beyond calling
    repository discovery and load/save APIs. Metric computation is delegated
    exclusively to ``AnalyticsPipeline``. Persistence remains in the CLI
    because ``AnalyticsPipeline`` does not own a repository.
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

from cqros.analytics import (
    AnalyticsEngine,
    AnalyticsEngineRegistry,
    AnalyticsPipeline,
    AnalyticsRepository,
    SimpleAnalyticsEngine,
)
from cqros.config.models import ResearchConfig
from cqros.core.constants import (
    DEFAULT_STORAGE_ROOT,
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
    STORAGE_DIR_ANALYTICS,
    SUPPORTED_TIMEFRAMES,
)
from cqros.core.exceptions import CQROSError, ValidationError
from cqros.core.types import Symbol, Timeframe
from cqros.performance import PerformanceRepository
from cqros.performance.repository import PerformancePartitionRef
from cqros.storage import ParquetStore, StorageLayout

__all__ = [
    "AnalyticsGenerationOptions",
    "AnalyticsGenerationSummary",
    "AnalyticsTaskResult",
    "DiscoveredWorkItem",
    "build_analytics_pipeline",
    "build_default_engine",
    "build_options",
    "build_parser",
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

_ERROR_WORKERS: Final[str] = "CLI-GENERATE-ANALYTICS-001"
_ERROR_TIMEFRAME: Final[str] = "CLI-GENERATE-ANALYTICS-002"
_ERROR_YEAR: Final[str] = "CLI-GENERATE-ANALYTICS-003"
_ERROR_MANAGER: Final[str] = "CLI-GENERATE-ANALYTICS-004"
_ERROR_ENGINE: Final[str] = "CLI-GENERATE-ANALYTICS-005"
_ERROR_PERFORMANCE_MISSING: Final[str] = "CLI-GENERATE-ANALYTICS-006"

_COL_ROLLING_RETURN: Final[str] = "rolling_return"
_COL_ROLLING_MAX_DRAWDOWN: Final[str] = "rolling_max_drawdown"


@dataclass(frozen=True, slots=True)
class AnalyticsGenerationOptions:
    """Immutable CLI options for analytics dataset generation.

    Attributes:
        storage_root: Storage root containing ``performance`` and
            ``analytics``.
        manager: Order manager identity used for discovery and analytics
            lineage.
        engine: Registry key of the analytics engine to execute.
        symbols: Optional symbol allowlist. ``None`` discovers all.
        timeframes: Optional timeframe allowlist. ``None`` discovers all.
        years: Optional year allowlist. ``None`` discovers all.
        overwrite: When ``True``, regenerate existing analytics partitions.
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
    """One discovered performance partition group ready for analytics generation.

    Attributes:
        manager: Order manager identifier of the source partitions.
        symbol: Tradeable symbol.
        timeframe: Available bar interval.
        years: Calendar years with existing performance parquet partitions.
    """

    manager: str
    symbol: Symbol
    timeframe: Timeframe
    years: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class AnalyticsTaskResult:
    """Immutable result for one symbol/timeframe/year generation task.

    Attributes:
        symbol: Tradeable symbol.
        timeframe: Bar interval.
        year: Calendar year of the partition.
        status: ``succeeded``, ``failed``, or ``skipped``.
        rows_generated: Output row count when succeeded.
        rolling_return: Last ``rolling_return`` when succeeded.
        max_dd: Last ``rolling_max_drawdown`` when succeeded.
        error_type: Exception type name when failed.
        error_message: Exception message when failed.
        error_code: Optional CQROS error code when failed.
    """

    symbol: Symbol
    timeframe: Timeframe
    year: int
    status: str
    rows_generated: int | None = None
    rolling_return: float | None = None
    max_dd: float | None = None
    error_type: str | None = None
    error_message: str | None = None
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class AnalyticsGenerationSummary:
    """Immutable aggregate summary for an analytics-generation run.

    Attributes:
        manager: Order manager identity used for generation.
        engine: Analytics engine registry key used for generation.
        symbols: Unique symbols for which generation was attempted.
        rows: Sum of output rows across successes.
        rolling_return: Sum of last ``rolling_return`` values across successes.
        max_dd: Maximum last ``rolling_max_drawdown`` across successes.
        successful_tasks: Count of succeeded partition tasks.
        failed_tasks: Count of failed partition tasks.
        skipped_tasks: Count of skipped existing partitions.
        duration_seconds: Wall-clock generation duration.
        output_directory: Analytics-tier output directory.
        failed_task_labels: Deterministic failed-task labels for reporting.
    """

    manager: str
    engine: str
    symbols: int
    rows: int
    rolling_return: float
    max_dd: float
    successful_tasks: int
    failed_tasks: int
    skipped_tasks: int
    duration_seconds: float
    output_directory: Path
    failed_task_labels: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    """Create the analytics-generation argument parser.

    Returns:
        Configured ``ArgumentParser`` for analytics-generation flags.
    """
    parser = argparse.ArgumentParser(
        prog="cqros-generate-analytics",
        description=(
            "Generate CQROS analytics metrics datasets from discovered "
            "performance partitions and an injected analytics engine."
        ),
    )
    parser.add_argument(
        "--manager",
        dest="manager",
        required=True,
        metavar="NAME",
        help="Order manager identity used for discovery and analytics lineage.",
    )
    parser.add_argument(
        "--engine",
        dest="engine",
        default=_DEFAULT_ENGINE,
        metavar="NAME",
        help=f"Analytics engine registry key (default: {_DEFAULT_ENGINE}).",
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
        help="Regenerate analytics partitions that already exist.",
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


def build_options(args: argparse.Namespace) -> AnalyticsGenerationOptions:
    """Map parsed CLI arguments onto ``AnalyticsGenerationOptions``.

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

    return AnalyticsGenerationOptions(
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


def build_default_engine() -> AnalyticsEngine:
    """Compose the default production analytics engine for the CLI.

    Returns:
        ``SimpleAnalyticsEngine`` instance.
    """
    return SimpleAnalyticsEngine()


def build_registry(
    *,
    engines: Mapping[str, AnalyticsEngine] | None = None,
) -> AnalyticsEngineRegistry:
    """Compose a registry with default or injected analytics engine implementations.

    Args:
        engines: Optional mapping of registry names to engine instances.
            When ``None``, registers ``SimpleAnalyticsEngine`` under
            ``simple``.

    Returns:
        Fully populated ``AnalyticsEngineRegistry``.
    """
    registry = AnalyticsEngineRegistry()
    if engines is None:
        registry.register(_DEFAULT_ENGINE, build_default_engine())
    else:
        for name, engine in engines.items():
            registry.register(name, engine)
    return registry


def build_analytics_pipeline(
    options: AnalyticsGenerationOptions,
    *,
    engine_registry: AnalyticsEngineRegistry | None = None,
) -> AnalyticsPipeline:
    """Compose ``AnalyticsPipeline`` from injected engine registry dependencies.

    Args:
        options: Immutable generation options providing the engine name.
        engine_registry: Optional engine registry. When ``None``, a default
            registry containing ``SimpleAnalyticsEngine`` is built.

    Returns:
        Fully wired ``AnalyticsPipeline``.
    """
    if engine_registry is None:
        engine_registry = build_registry()
    elif options.engine == _DEFAULT_ENGINE and not engine_registry.exists(options.engine):
        engine_registry.register(options.engine, build_default_engine())
    return AnalyticsPipeline(engine_registry)


def discover_work(
    performance_repository: PerformanceRepository,
    options: AnalyticsGenerationOptions,
) -> tuple[DiscoveredWorkItem, ...]:
    """Discover analytics-ready performance partitions matching CLI filters.

    Only performance partitions that exist are scheduled. Missing performance
    partitions are never invented.

    Args:
        performance_repository: Performance repository providing discovery APIs.
        options: CLI filters for manager, symbol, timeframe, and year.

    Returns:
        Deterministically ordered discovered work items.
    """
    partitions = performance_repository.discover_partitions(
        managers=(options.manager,),
        symbols=options.symbols,
        timeframes=options.timeframes,
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    return _group_partitions(partitions, year_filter=options.years)


def format_summary(summary: AnalyticsGenerationSummary) -> str:
    """Render a deterministic analytics-generation summary report.

    Args:
        summary: Aggregate generation summary.

    Returns:
        Multi-line summary text suitable for stdout.
    """
    lines = [
        "=====================================",
        "CQROS Analytics Generation Summary",
        "=====================================",
        "",
        f"Manager: {summary.manager}",
        f"Engine: {summary.engine}",
        "",
        f"Symbols: {summary.symbols}",
        f"Rows: {summary.rows}",
        f"Rolling Return: {summary.rolling_return:.4f}",
        f"Max DD: {summary.max_dd:.4f}",
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
    """Run the analytics-generation CLI.

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
        performance_repository = PerformanceRepository(layout, datastore)
        analytics_repository = AnalyticsRepository(layout, datastore)
        pipeline = build_analytics_pipeline(options)
        work = discover_work(performance_repository, options)
        summary = await run_generation(
            pipeline=pipeline,
            performance_repository=performance_repository,
            analytics_repository=analytics_repository,
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
    pipeline: AnalyticsPipeline,
    performance_repository: PerformanceRepository,
    analytics_repository: AnalyticsRepository,
    options: AnalyticsGenerationOptions,
    work: Sequence[DiscoveredWorkItem],
) -> AnalyticsGenerationSummary:
    """Execute discovered work through a bounded symbol worker pool.

    Args:
        pipeline: Injected analytics pipeline.
        performance_repository: Performance partition repository.
        analytics_repository: Analytics partition repository.
        options: Immutable generation options.
        work: Discovered work items.

    Returns:
        Aggregate immutable summary.
    """
    started = time.perf_counter()
    output_directory = options.storage_root / STORAGE_DIR_ANALYTICS

    if len(work) == 0:
        return AnalyticsGenerationSummary(
            manager=options.manager,
            engine=options.engine,
            symbols=0,
            rows=0,
            rolling_return=0.0,
            max_dd=0.0,
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
        performance_repository=performance_repository,
        analytics_repository=analytics_repository,
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
    partitions: Sequence[PerformancePartitionRef],
    *,
    year_filter: tuple[int, ...] | None,
) -> tuple[DiscoveredWorkItem, ...]:
    """Group performance year partitions into manager/symbol/timeframe work items."""
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
    pipeline: AnalyticsPipeline,
    performance_repository: PerformanceRepository,
    analytics_repository: AnalyticsRepository,
    work_by_symbol: Mapping[Symbol, Sequence[DiscoveredWorkItem]],
    worker_count: int,
    overwrite: bool,
    debug: bool,
    manager_name: str,
    engine_name: str,
) -> tuple[AnalyticsTaskResult, ...]:
    """Drain symbols through a bounded asyncio worker pool."""
    symbols = tuple(work_by_symbol.keys())
    if len(symbols) == 0:
        return ()

    queue: asyncio.Queue[Symbol | None] = asyncio.Queue()
    for symbol in symbols:
        queue.put_nowait(symbol)
    for _ in range(worker_count):
        queue.put_nowait(None)

    collected: list[AnalyticsTaskResult] = []
    lock = asyncio.Lock()

    async def worker() -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                results = await _generate_symbol_work(
                    pipeline=pipeline,
                    performance_repository=performance_repository,
                    analytics_repository=analytics_repository,
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
        asyncio.create_task(worker(), name=f"generate-analytics-worker-{index}")
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
    pipeline: AnalyticsPipeline,
    performance_repository: PerformanceRepository,
    analytics_repository: AnalyticsRepository,
    symbol: Symbol,
    work_items: Sequence[DiscoveredWorkItem],
    overwrite: bool,
    debug: bool,
    manager_name: str,
    engine_name: str,
) -> tuple[AnalyticsTaskResult, ...]:
    """Generate analytics datasets for every discovered year for one symbol."""
    results: list[AnalyticsTaskResult] = []
    for item in work_items:
        for year in item.years:
            result = await asyncio.to_thread(
                _generate_partition,
                pipeline,
                performance_repository,
                analytics_repository,
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
    pipeline: AnalyticsPipeline,
    performance_repository: PerformanceRepository,
    analytics_repository: AnalyticsRepository,
    *,
    manager: str,
    symbol: Symbol,
    timeframe: Timeframe,
    year: int,
    overwrite: bool,
    debug: bool,
    manager_name: str,
    engine_name: str,
) -> AnalyticsTaskResult:
    """Generate one analytics year partition synchronously."""
    if not overwrite and analytics_repository.exists(
        manager=manager_name,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=symbol,
        timeframe=timeframe,
        year=year,
    ):
        return AnalyticsTaskResult(
            symbol=symbol,
            timeframe=timeframe,
            year=year,
            status="skipped",
        )

    try:
        if not performance_repository.exists(
            manager=manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        ):
            raise ValidationError(
                f"performance partition missing for {manager}/{symbol}/{timeframe}/{year}",
                error_code=_ERROR_PERFORMANCE_MISSING,
                details={
                    "manager": manager,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "year": year,
                    "missing": "performance",
                },
            )

        performance = performance_repository.load(
            manager=manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        )
        output = pipeline.run(engine_name, performance)
        analytics_repository.save(
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
        return AnalyticsTaskResult(
            symbol=symbol,
            timeframe=timeframe,
            year=year,
            status="failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
            error_code=exc.error_code if isinstance(exc, CQROSError) else None,
        )

    rows_generated, rolling_return, max_dd = _extract_partition_stats(output)
    return AnalyticsTaskResult(
        symbol=symbol,
        timeframe=timeframe,
        year=year,
        status="succeeded",
        rows_generated=rows_generated,
        rolling_return=rolling_return,
        max_dd=max_dd,
    )


def _extract_partition_stats(frame: pl.DataFrame) -> tuple[int, float, float]:
    """Extract row count and terminal analytics metrics from one metrics frame.

    Args:
        frame: Finalized analytics output DataFrame.

    Returns:
        A 3-tuple of ``(rows, rolling_return, max_dd)``.
    """
    if frame.height == 0:
        return 0, 0.0, 0.0

    last_row = frame.sort("open_time", maintain_order=True).tail(1)
    rolling_return = float(last_row[_COL_ROLLING_RETURN][0])
    max_dd = float(last_row[_COL_ROLLING_MAX_DRAWDOWN][0])
    return frame.height, rolling_return, max_dd


def _print_progress(result: AnalyticsTaskResult) -> None:
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
            "Failed analytics generation partition; continuing",
            extra=log_extra,
            exc_info=True,
        )
    else:
        _logger.warning(
            "Failed analytics generation partition; continuing",
            extra=log_extra,
        )


def _build_summary(
    *,
    options: AnalyticsGenerationOptions,
    work: Sequence[DiscoveredWorkItem],
    results: Sequence[AnalyticsTaskResult],
    duration_seconds: float,
    output_directory: Path,
) -> AnalyticsGenerationSummary:
    """Aggregate task results into a generation report."""
    symbols_discovered = {item.symbol for item in work}
    symbols_processed: set[Symbol] = set()
    successful_tasks = 0
    failed_tasks = 0
    skipped_tasks = 0
    rows = 0
    rolling_return = 0.0
    max_dd = 0.0
    failed_labels: set[str] = set()

    for result in results:
        symbols_processed.add(result.symbol)
        if result.status == "succeeded":
            successful_tasks += 1
            if result.rows_generated is not None:
                rows += result.rows_generated
            if result.rolling_return is not None:
                rolling_return += result.rolling_return
            if result.max_dd is not None:
                max_dd = max(max_dd, result.max_dd)
        elif result.status == "skipped":
            skipped_tasks += 1
        else:
            failed_tasks += 1
            failed_labels.add(f"{result.symbol} {result.timeframe} {result.year}")

    return AnalyticsGenerationSummary(
        manager=options.manager,
        engine=options.engine,
        symbols=len(symbols_processed) if results else len(symbols_discovered),
        rows=rows,
        rolling_return=rolling_return,
        max_dd=max_dd,
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
