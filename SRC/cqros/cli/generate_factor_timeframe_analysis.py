"""CQROS factor-timeframe-analysis generation CLI.

Purpose:
    Provide an argparse-based production entry point that discovers canonical
    Factor Selection panels and executes ``FactorTimeframeAnalysisPipeline``
    across manager/year work items with bounded concurrency, persisting
    factor timeframe analysis metrics through
    ``FactorTimeframeAnalysisRepository``.

Responsibilities:
    - Parse CLI arguments for factor timeframe analysis dataset generation
    - Discover available Factor Selection partitions through
      ``FactorSelectionRepository`` and group them by manager/year
    - Load all available Factor Selection timeframe partitions for each year
      via ``load_factor_selection_for_analysis``
    - Resolve ``SimpleFactorTimeframeAnalysisEngine`` with
      ``source_selection_version=str(year)``
    - Execute ``FactorTimeframeAnalysisPipeline`` and persist via
      ``FactorTimeframeAnalysisRepository`` (no symbol, no timeframe in path)
    - Honor ``--overwrite``, worker concurrency, and debug logging
    - Print deterministic progress and a final summary

Dependencies:
    ``argparse``, ``asyncio``, ``polars``, ``cqros.config``, ``cqros.core``,
    ``cqros.factor_selection``, ``cqros.factor_timeframe_analysis``,
    and ``cqros.storage``.

Public API:
    ``DiscoveredWorkItem``, ``FactorTimeframeAnalysisGenerationOptions``,
    ``FactorTimeframeAnalysisGenerationSummary``,
    ``FactorTimeframeAnalysisTaskResult``, ``build_options``, ``build_parser``,
    ``discover_work``, ``format_summary``, ``main``, ``run_generation``.

Notes:
    This module is a thin composition root. It does not implement analysis
    math, schema validation, or repository filesystem walks beyond calling
    repository discovery and load/save APIs. Metric computation is delegated
    exclusively to ``FactorTimeframeAnalysisPipeline``. Persistence remains
    in the CLI because the pipeline does not own a repository.
    Optional ``--export-detailed-csv`` writes audit CSVs via
    ``cqros.factor_timeframe_analysis.detailed_export`` without replacing
    Parquet. FTA partitions are cross-sectional panels keyed by
    manager/year (no symbol, no source timeframe).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import polars as pl

from cqros.config.models import ResearchConfig
from cqros.core.constants import (
    DEFAULT_STORAGE_ROOT,
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
    STORAGE_DIR_FACTOR_TIMEFRAME_ANALYSIS,
    SUPPORTED_TIMEFRAMES,
)
from cqros.core.exceptions import CQROSError, ValidationError
from cqros.core.types import Timeframe
from cqros.factor_selection import FactorSelectionRepository
from cqros.factor_selection.repository import FactorSelectionPartitionRef
from cqros.factor_timeframe_analysis import (
    FactorTimeframeAnalysisEngineRegistry,
    FactorTimeframeAnalysisPipeline,
    FactorTimeframeAnalysisRepository,
    SimpleFactorTimeframeAnalysisEngine,
    build_detailed_audit_frame,
    detailed_csv_path,
    discover_selection_timeframes,
    load_factor_selection_for_analysis,
    write_detailed_csv,
)
from cqros.storage import ParquetStore, StorageLayout

__all__ = [
    "DiscoveredWorkItem",
    "FactorTimeframeAnalysisGenerationOptions",
    "FactorTimeframeAnalysisGenerationSummary",
    "FactorTimeframeAnalysisTaskResult",
    "build_options",
    "build_parser",
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

_ERROR_WORKERS: Final[str] = "CLI-GENERATE-FTA-001"
_ERROR_YEAR: Final[str] = "CLI-GENERATE-FTA-002"
_ERROR_MANAGER: Final[str] = "CLI-GENERATE-FTA-003"
_ERROR_TIMEFRAME: Final[str] = "CLI-GENERATE-FTA-004"


@dataclass(frozen=True, slots=True)
class FactorTimeframeAnalysisGenerationOptions:
    """Immutable CLI options for factor timeframe analysis dataset generation.

    Attributes:
        storage_root: Storage root containing ``factor_selection`` and
            ``factor_timeframe_analysis``.
        manager: Order manager identity used for discovery and FTA lineage.
        engine: Registry key of the engine to execute (default ``simple``).
        timeframes: Optional Factor Selection source timeframe allowlist.
            ``None`` loads all available timeframes for each year.
        years: Optional year allowlist. ``None`` discovers all.
        overwrite: When ``True``, regenerate existing FTA partitions.
        export_detailed_csv: When ``True``, write per-year detailed audit
            CSV exports alongside canonical Parquet output.
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
    export_detailed_csv: bool
    workers: int
    verbose: bool
    debug: bool


@dataclass(frozen=True, slots=True)
class DiscoveredWorkItem:
    """One discovered manager/year work item ready for FTA generation.

    Attributes:
        manager: Order manager identifier of the source Factor Selection
            partitions.
        year: Calendar year of the Factor Selection partitions to analyse.
    """

    manager: str
    year: int


@dataclass(frozen=True, slots=True)
class FactorTimeframeAnalysisTaskResult:
    """Immutable result for one manager/year FTA generation task.

    Attributes:
        year: Calendar year of the partition.
        status: ``succeeded``, ``failed``, or ``skipped``.
        rows_generated: Output row count when succeeded.
        selected_rows: Count of rows with ``selected==True`` on success.
        detailed_audit: Optional detailed audit frame when CSV export is
            enabled.
        error_type: Exception type name when failed.
        error_message: Exception message when failed.
        error_code: Optional CQROS error code when failed.
    """

    year: int
    status: str
    rows_generated: int | None = None
    selected_rows: int | None = None
    detailed_audit: pl.DataFrame | None = None
    error_type: str | None = None
    error_message: str | None = None
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class FactorTimeframeAnalysisGenerationSummary:
    """Immutable aggregate summary for a FTA-generation run.

    Attributes:
        manager: Order manager identity used for generation.
        engine: Engine registry key used for generation.
        panels: Unique manager/year panels for which generation was attempted.
        rows: Sum of output rows across successes.
        selected_rows: Sum of selected-factor rows across successes.
        successful_tasks: Count of succeeded partition tasks.
        failed_tasks: Count of failed partition tasks.
        skipped_tasks: Count of skipped existing partitions.
        duration_seconds: Wall-clock generation duration.
        output_directory: FTA-tier output directory.
        failed_task_labels: Deterministic failed-task labels for reporting.
    """

    manager: str
    engine: str
    panels: int
    rows: int
    selected_rows: int
    successful_tasks: int
    failed_tasks: int
    skipped_tasks: int
    duration_seconds: float
    output_directory: Path
    failed_task_labels: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    """Create the factor-timeframe-analysis-generation argument parser.

    Returns:
        Configured ``ArgumentParser`` for FTA-generation flags.
    """
    parser = argparse.ArgumentParser(
        prog="cqros-generate-factor-timeframe-analysis",
        description=(
            "Generate CQROS factor timeframe analysis datasets from discovered "
            "Factor Selection panels and the simple FTA engine."
        ),
    )
    parser.add_argument(
        "--manager",
        dest="manager",
        required=True,
        metavar="NAME",
        help="Order manager identity used for discovery and FTA lineage.",
    )
    parser.add_argument(
        "--engine",
        dest="engine",
        default=_DEFAULT_ENGINE,
        metavar="NAME",
        help=f"FTA engine registry key (default: {_DEFAULT_ENGINE}).",
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
        "--timeframes",
        dest="timeframes",
        nargs="*",
        default=None,
        metavar="TIMEFRAME",
        help=(
            "Optional Factor Selection source timeframe allowlist (0..N values). "
            "Omit to load all available timeframes for each year."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate FTA partitions that already exist.",
    )
    parser.add_argument(
        "--export-detailed-csv",
        dest="export_detailed_csv",
        action="store_true",
        help=(
            "Write detailed audit CSV exports (per-year) "
            "alongside canonical Parquet FTA datasets."
        ),
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


def build_options(args: argparse.Namespace) -> FactorTimeframeAnalysisGenerationOptions:
    """Map parsed CLI arguments onto ``FactorTimeframeAnalysisGenerationOptions``.

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
            error_code="CLI-GENERATE-FTA-005",
            details={"parameter": "engine", "value": args.engine},
        )

    storage_root = (
        Path(args.storage_root) if args.storage_root is not None else Path(DEFAULT_STORAGE_ROOT)
    )

    return FactorTimeframeAnalysisGenerationOptions(
        storage_root=storage_root,
        manager=manager,
        engine=engine,
        timeframes=_normalize_timeframes(args.timeframes),
        years=_normalize_years(args.years),
        overwrite=bool(args.overwrite),
        export_detailed_csv=bool(args.export_detailed_csv),
        workers=workers,
        verbose=bool(args.verbose),
        debug=bool(args.debug),
    )


def discover_work(
    factor_selection_repository: FactorSelectionRepository,
    options: FactorTimeframeAnalysisGenerationOptions,
) -> tuple[DiscoveredWorkItem, ...]:
    """Discover FTA-ready manager/year work items from Factor Selection partitions.

    Uses ``FactorSelectionRepository`` to find all available partitions for
    the manager and groups them by year. Only years with at least one Factor
    Selection partition are scheduled.

    Args:
        factor_selection_repository: Factor Selection repository providing
            discovery APIs.
        options: CLI filters for manager, timeframe, and year.

    Returns:
        Deterministically ordered discovered work items (manager, year).
    """
    partitions = factor_selection_repository.discover_partitions(
        managers=(options.manager,),
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    return _group_partitions_by_year(partitions, year_filter=options.years)


def format_summary(summary: FactorTimeframeAnalysisGenerationSummary) -> str:
    """Render a deterministic FTA-generation summary report.

    Args:
        summary: Aggregate generation summary.

    Returns:
        Multi-line summary text suitable for stdout.
    """
    lines = [
        "=====================================",
        "CQROS Factor Timeframe Analysis Generation Summary",
        "=====================================",
        "",
        f"Manager: {summary.manager}",
        f"Engine: {summary.engine}",
        "",
        f"Panels: {summary.panels}",
        f"Rows: {summary.rows}",
        f"Selected: {summary.selected_rows}",
        "",
        f"Successful: {summary.successful_tasks}",
        f"Failed: {summary.failed_tasks}",
        f"Skipped: {summary.skipped_tasks}",
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
    """Run the factor-timeframe-analysis-generation CLI.

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
        fta_repository = FactorTimeframeAnalysisRepository(layout, datastore)
        work = discover_work(factor_selection_repository, options)
        summary = await run_generation(
            layout=layout,
            factor_selection_repository=factor_selection_repository,
            fta_repository=fta_repository,
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
    layout: StorageLayout,
    factor_selection_repository: FactorSelectionRepository,
    fta_repository: FactorTimeframeAnalysisRepository,
    options: FactorTimeframeAnalysisGenerationOptions,
    work: Sequence[DiscoveredWorkItem],
) -> FactorTimeframeAnalysisGenerationSummary:
    """Execute discovered work through a bounded panel worker pool."""
    started = time.perf_counter()
    output_directory = options.storage_root / STORAGE_DIR_FACTOR_TIMEFRAME_ANALYSIS

    if len(work) == 0:
        return FactorTimeframeAnalysisGenerationSummary(
            manager=options.manager,
            engine=options.engine,
            panels=0,
            rows=0,
            selected_rows=0,
            successful_tasks=0,
            failed_tasks=0,
            skipped_tasks=0,
            duration_seconds=time.perf_counter() - started,
            output_directory=output_directory,
            failed_task_labels=(),
        )

    results = await _run_worker_pool(
        layout=layout,
        factor_selection_repository=factor_selection_repository,
        fta_repository=fta_repository,
        work=work,
        worker_count=options.workers,
        overwrite=options.overwrite,
        export_detailed_csv=options.export_detailed_csv,
        debug=options.debug,
        manager=options.manager,
        engine=options.engine,
        timeframes=options.timeframes,
        storage_root=options.storage_root,
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


def _group_partitions_by_year(
    partitions: Sequence[FactorSelectionPartitionRef],
    *,
    year_filter: tuple[int, ...] | None,
) -> tuple[DiscoveredWorkItem, ...]:
    """Group Factor Selection partitions into manager/year work items."""
    year_allowlist = set(year_filter) if year_filter is not None else None
    seen: dict[tuple[str, int], None] = {}
    for partition in partitions:
        if year_allowlist is not None and partition.year not in year_allowlist:
            continue
        key = (partition.manager, partition.year)
        seen[key] = None

    items: list[DiscoveredWorkItem] = []
    for manager, year in seen:
        items.append(DiscoveredWorkItem(manager=manager, year=year))
    return tuple(
        sorted(
            items,
            key=lambda item: (item.manager, item.year),
        )
    )


async def _run_worker_pool(
    *,
    layout: StorageLayout,
    factor_selection_repository: FactorSelectionRepository,
    fta_repository: FactorTimeframeAnalysisRepository,
    work: Sequence[DiscoveredWorkItem],
    worker_count: int,
    overwrite: bool,
    export_detailed_csv: bool,
    debug: bool,
    manager: str,
    engine: str,
    timeframes: tuple[Timeframe, ...] | None,
    storage_root: Path,
) -> tuple[FactorTimeframeAnalysisTaskResult, ...]:
    """Drain work items through a bounded asyncio worker pool."""
    if len(work) == 0:
        return ()

    queue: asyncio.Queue[DiscoveredWorkItem | None] = asyncio.Queue()
    for item in work:
        queue.put_nowait(item)
    for _ in range(worker_count):
        queue.put_nowait(None)

    collected: list[FactorTimeframeAnalysisTaskResult] = []
    lock = asyncio.Lock()

    async def worker() -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                result = await asyncio.to_thread(
                    _generate_partition,
                    layout,
                    factor_selection_repository,
                    fta_repository,
                    manager=item.manager,
                    year=item.year,
                    overwrite=overwrite,
                    export_detailed_csv=export_detailed_csv,
                    debug=debug,
                    engine=engine,
                    timeframes=timeframes,
                    storage_root=storage_root,
                )
                _print_progress(result)
                async with lock:
                    collected.append(result)
            finally:
                queue.task_done()

    worker_tasks = [
        asyncio.create_task(worker(), name=f"generate-fta-worker-{index}")
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
            key=lambda result: result.year,
        )
    )


def _generate_partition(
    layout: StorageLayout,
    factor_selection_repository: FactorSelectionRepository,
    fta_repository: FactorTimeframeAnalysisRepository,
    *,
    manager: str,
    year: int,
    overwrite: bool,
    export_detailed_csv: bool,
    debug: bool,
    engine: str,
    timeframes: tuple[Timeframe, ...] | None,
    storage_root: Path,
) -> FactorTimeframeAnalysisTaskResult:
    """Generate one FTA year partition synchronously."""
    if not overwrite and fta_repository.exists(
        manager=manager,
        exchange=_EXCHANGE,
        market=_MARKET,
        year=year,
    ):
        return FactorTimeframeAnalysisTaskResult(year=year, status="skipped")

    try:
        factor_selection_frame = load_factor_selection_for_analysis(
            factor_selection_repository,
            manager=manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            year=year,
            timeframes=timeframes,
        )

        fta_engine = SimpleFactorTimeframeAnalysisEngine(
            source_selection_version=str(year),
        )
        registry = FactorTimeframeAnalysisEngineRegistry()
        registry.register(engine, fta_engine)
        pipeline = FactorTimeframeAnalysisPipeline(registry)
        output = pipeline.run(engine, factor_selection_frame)

        fta_repository.save(
            output,
            manager=manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            year=year,
        )

        detailed_audit: pl.DataFrame | None = None
        if export_detailed_csv:
            source_timeframes_list = discover_selection_timeframes(
                factor_selection_repository,
                manager=manager,
                exchange=_EXCHANGE,
                market=_MARKET,
                year=year,
            )
            if timeframes is not None:
                tf_filter = frozenset(timeframes)
                source_timeframes_list = tuple(
                    tf for tf in source_timeframes_list if tf in tf_filter
                )
            source_timeframes_str = ",".join(sorted(source_timeframes_list))
            detailed_audit = build_detailed_audit_frame(
                output,
                manager=manager,
                exchange=_EXCHANGE,
                market=_MARKET,
                year=year,
                source_timeframes=source_timeframes_str,
            )
            write_detailed_csv(
                detailed_audit,
                detailed_csv_path(
                    storage_root,
                    manager=manager,
                    exchange=_EXCHANGE,
                    market=_MARKET,
                    year=year,
                ),
            )
    except Exception as exc:
        _log_partition_failure(year=year, exc=exc, debug=debug)
        return FactorTimeframeAnalysisTaskResult(
            year=year,
            status="failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
            error_code=exc.error_code if isinstance(exc, CQROSError) else None,
        )

    rows_generated = output.height
    selected_rows = int(output.select(pl.col("selected").fill_null(value=False).sum()).item())
    return FactorTimeframeAnalysisTaskResult(
        year=year,
        status="succeeded",
        rows_generated=rows_generated,
        selected_rows=selected_rows,
        detailed_audit=detailed_audit,
    )


def _print_progress(result: FactorTimeframeAnalysisTaskResult) -> None:
    """Print a deterministic one-line progress record for a task result."""
    label = str(result.year)
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
    year: int,
    exc: BaseException,
    debug: bool,
) -> None:
    """Log a partition generation failure without aborting the run."""
    log_extra = {
        "year": year,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "error_code": exc.error_code if isinstance(exc, CQROSError) else None,
    }
    if debug:
        _logger.exception(
            "Failed FTA generation partition; continuing",
            extra=log_extra,
            exc_info=True,
        )
    else:
        _logger.warning(
            "Failed FTA generation partition; continuing",
            extra=log_extra,
        )


def _build_summary(
    *,
    options: FactorTimeframeAnalysisGenerationOptions,
    work: Sequence[DiscoveredWorkItem],
    results: Sequence[FactorTimeframeAnalysisTaskResult],
    duration_seconds: float,
    output_directory: Path,
) -> FactorTimeframeAnalysisGenerationSummary:
    """Aggregate task results into a generation report."""
    panels_discovered = len(work)
    panels_processed: set[int] = set()
    successful_tasks = 0
    failed_tasks = 0
    skipped_tasks = 0
    rows = 0
    selected_rows = 0
    failed_labels: set[str] = set()

    for result in results:
        panels_processed.add(result.year)
        if result.status == "succeeded":
            successful_tasks += 1
            if result.rows_generated is not None:
                rows += result.rows_generated
            if result.selected_rows is not None:
                selected_rows += result.selected_rows
        elif result.status == "skipped":
            skipped_tasks += 1
        else:
            failed_tasks += 1
            failed_labels.add(str(result.year))

    return FactorTimeframeAnalysisGenerationSummary(
        manager=options.manager,
        engine=options.engine,
        panels=len(panels_processed) if results else panels_discovered,
        rows=rows,
        selected_rows=selected_rows,
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
