"""CQROS factor-combination generation CLI.

Purpose:
    Provide an argparse-based production entry point that discovers canonical
    Factor Timeframe Analysis panels and executes
    ``SimpleFactorCombinationEngine`` across manager/year work items with
    bounded concurrency, partitioning combination output by the resolved
    ``timeframe`` column and persisting each partition through
    ``FactorCombinationRepository``.

Responsibilities:
    - Parse CLI arguments for factor combination dataset generation
    - Discover available FTA partitions through
      ``FactorTimeframeAnalysisRepository``
    - Load FTA frames and build combinations via
      ``SimpleFactorCombinationEngine`` (never from Factor Selection directly)
    - Partition combination output by the ``timeframe`` column
    - Save each timeframe partition via ``FactorCombinationRepository``
      (no symbol, no source timeframe in FTA load)
    - Honor ``--overwrite`` per timeframe partition, worker concurrency,
      and debug logging
    - Print deterministic progress and a final summary

Dependencies:
    ``argparse``, ``asyncio``, ``polars``, ``cqros.config``, ``cqros.core``,
    ``cqros.factor_timeframe_analysis``, ``cqros.factor_combination``,
    and ``cqros.storage``.

Public API:
    ``DiscoveredWorkItem``, ``FactorCombinationGenerationOptions``,
    ``FactorCombinationGenerationSummary``, ``FactorCombinationTaskResult``,
    ``build_options``, ``build_parser``, ``discover_work``, ``format_summary``,
    ``main``, ``run_generation``.

Notes:
    This module is a thin composition root. It does not implement combination
    math, schema validation, or repository filesystem walks beyond calling
    repository discovery and load/save APIs. Combination semantics are
    delegated exclusively to ``SimpleFactorCombinationEngine``. FTA is the
    only input; Factor Selection is never loaded in this CLI.
    Persistence remains in the CLI because the engine does not own a
    repository. Optional ``--export-detailed-csv`` writes audit CSVs via
    ``cqros.factor_combination.detailed_export`` without replacing Parquet.
    Combination partitions are keyed by manager/timeframe/year.
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
    STORAGE_DIR_FACTOR_COMBINATION,
)
from cqros.core.exceptions import CQROSError, ValidationError
from cqros.core.types import Timeframe
from cqros.factor_combination import (
    FactorCombinationRepository,
    SimpleFactorCombinationEngine,
    combined_detailed_csv_path,
    write_combined_detailed_csv,
)
from cqros.factor_combination import (
    build_detailed_audit_frame as build_combination_detailed_audit_frame,
)
from cqros.factor_combination import (
    detailed_csv_path as combination_detailed_csv_path,
)
from cqros.factor_combination import (
    write_detailed_csv as write_combination_detailed_csv,
)
from cqros.factor_timeframe_analysis import (
    FactorTimeframeAnalysisPartitionRef,
    FactorTimeframeAnalysisRepository,
)
from cqros.storage import ParquetStore, StorageLayout

__all__ = [
    "DiscoveredWorkItem",
    "FactorCombinationGenerationOptions",
    "FactorCombinationGenerationSummary",
    "FactorCombinationTaskResult",
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

_EXCHANGE: Final[str] = EXCHANGE_BINANCE
_MARKET: Final[str] = MARKET_USDT_PERPETUAL

_ERROR_WORKERS: Final[str] = "CLI-GENERATE-FCOMB-001"
_ERROR_YEAR: Final[str] = "CLI-GENERATE-FCOMB-002"
_ERROR_MANAGER: Final[str] = "CLI-GENERATE-FCOMB-003"

_COL_TIMEFRAME: Final[str] = "timeframe"
_COL_SOURCE_SELECTION_VERSION: Final[str] = "source_selection_version"


@dataclass(frozen=True, slots=True)
class FactorCombinationGenerationOptions:
    """Immutable CLI options for factor combination dataset generation.

    Attributes:
        storage_root: Storage root containing ``factor_timeframe_analysis``
            and ``factor_combination``.
        manager: Order manager identity used for discovery and lineage.
        years: Optional year allowlist. ``None`` discovers all.
        overwrite: When ``True``, regenerate existing combination partitions.
        export_detailed_csv: When ``True``, write per-timeframe detailed audit
            CSV exports alongside canonical Parquet output.
        workers: Maximum concurrent year work items.
        verbose: When ``True``, enable INFO logging.
        debug: When ``True``, enable DEBUG logging and log full failure
            tracebacks.
    """

    storage_root: Path
    manager: str
    years: tuple[int, ...] | None
    overwrite: bool
    export_detailed_csv: bool
    workers: int
    verbose: bool
    debug: bool


@dataclass(frozen=True, slots=True)
class DiscoveredWorkItem:
    """One discovered manager/year FTA panel ready for combination generation.

    Attributes:
        manager: Order manager identifier.
        year: Calendar year of the FTA partition to combine.
    """

    manager: str
    year: int


@dataclass(frozen=True, slots=True)
class FactorCombinationTaskResult:
    """Immutable result for one manager/year/timeframe combination partition task.

    Attributes:
        year: Calendar year of the partition.
        timeframe: Resolved combination timeframe partition.
        status: ``succeeded``, ``failed``, or ``skipped``.
        rows_generated: Output row count when succeeded.
        detailed_audit: Optional detailed audit frame when CSV export is
            enabled.
        error_type: Exception type name when failed.
        error_message: Exception message when failed.
        error_code: Optional CQROS error code when failed.
    """

    year: int
    timeframe: Timeframe
    status: str
    rows_generated: int | None = None
    detailed_audit: pl.DataFrame | None = None
    error_type: str | None = None
    error_message: str | None = None
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class FactorCombinationGenerationSummary:
    """Immutable aggregate summary for a combination-generation run.

    Attributes:
        manager: Order manager identity used for generation.
        panels: Unique manager/year/timeframe panels attempted.
        rows: Sum of output rows across successes.
        successful_tasks: Count of succeeded partition tasks.
        failed_tasks: Count of failed partition tasks.
        skipped_tasks: Count of skipped existing partitions.
        duration_seconds: Wall-clock generation duration.
        output_directory: Factor-combination-tier output directory.
        failed_task_labels: Deterministic failed-task labels for reporting.
    """

    manager: str
    panels: int
    rows: int
    successful_tasks: int
    failed_tasks: int
    skipped_tasks: int
    duration_seconds: float
    output_directory: Path
    failed_task_labels: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    """Create the factor-combination-generation argument parser.

    Returns:
        Configured ``ArgumentParser`` for combination-generation flags.
    """
    parser = argparse.ArgumentParser(
        prog="cqros-generate-factor-combination",
        description=(
            "Generate CQROS factor combination datasets from discovered "
            "Factor Timeframe Analysis panels."
        ),
    )
    parser.add_argument(
        "--manager",
        dest="manager",
        required=True,
        metavar="NAME",
        help="Order manager identity used for discovery and combination lineage.",
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
        help="Regenerate combination partitions that already exist.",
    )
    parser.add_argument(
        "--export-detailed-csv",
        dest="export_detailed_csv",
        action="store_true",
        help=(
            "Write detailed audit CSV exports (per-timeframe and combined) "
            "alongside canonical Parquet combination datasets."
        ),
    )
    parser.add_argument(
        "--workers",
        dest="workers",
        type=int,
        default=_DEFAULT_WORKER_COUNT,
        metavar="INT",
        help=f"Maximum concurrent year panels (default: {_DEFAULT_WORKER_COUNT}).",
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


def build_options(args: argparse.Namespace) -> FactorCombinationGenerationOptions:
    """Map parsed CLI arguments onto ``FactorCombinationGenerationOptions``.

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

    storage_root = (
        Path(args.storage_root) if args.storage_root is not None else Path(DEFAULT_STORAGE_ROOT)
    )

    return FactorCombinationGenerationOptions(
        storage_root=storage_root,
        manager=manager,
        years=_normalize_years(args.years),
        overwrite=bool(args.overwrite),
        export_detailed_csv=bool(args.export_detailed_csv),
        workers=workers,
        verbose=bool(args.verbose),
        debug=bool(args.debug),
    )


def discover_work(
    fta_repository: FactorTimeframeAnalysisRepository,
    options: FactorCombinationGenerationOptions,
) -> tuple[DiscoveredWorkItem, ...]:
    """Discover combination-ready manager/year work items from FTA partitions.

    Uses ``FactorTimeframeAnalysisRepository`` to find all available FTA
    partitions for the manager and returns them as year work items.

    Args:
        fta_repository: FTA repository providing discovery APIs.
        options: CLI filters for manager and year.

    Returns:
        Deterministically ordered discovered work items (manager, year).
    """
    partitions = fta_repository.discover_partitions(
        managers=(options.manager,),
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    return _group_partitions_by_year(partitions, year_filter=options.years)


def format_summary(summary: FactorCombinationGenerationSummary) -> str:
    """Render a deterministic combination-generation summary report.

    Args:
        summary: Aggregate generation summary.

    Returns:
        Multi-line summary text suitable for stdout.
    """
    lines = [
        "=====================================",
        "CQROS Factor Combination Generation Summary",
        "=====================================",
        "",
        f"Manager: {summary.manager}",
        "",
        f"Panels: {summary.panels}",
        f"Rows: {summary.rows}",
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
    """Run the factor-combination-generation CLI.

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
        fta_repository = FactorTimeframeAnalysisRepository(layout, datastore)
        combination_repository = FactorCombinationRepository(layout, datastore)
        work = discover_work(fta_repository, options)
        summary = await run_generation(
            fta_repository=fta_repository,
            combination_repository=combination_repository,
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
    fta_repository: FactorTimeframeAnalysisRepository,
    combination_repository: FactorCombinationRepository,
    options: FactorCombinationGenerationOptions,
    work: Sequence[DiscoveredWorkItem],
) -> FactorCombinationGenerationSummary:
    """Execute discovered work through a bounded panel worker pool."""
    started = time.perf_counter()
    output_directory = options.storage_root / STORAGE_DIR_FACTOR_COMBINATION

    if len(work) == 0:
        return FactorCombinationGenerationSummary(
            manager=options.manager,
            panels=0,
            rows=0,
            successful_tasks=0,
            failed_tasks=0,
            skipped_tasks=0,
            duration_seconds=time.perf_counter() - started,
            output_directory=output_directory,
            failed_task_labels=(),
        )

    results = await _run_worker_pool(
        fta_repository=fta_repository,
        combination_repository=combination_repository,
        work=work,
        worker_count=options.workers,
        overwrite=options.overwrite,
        export_detailed_csv=options.export_detailed_csv,
        debug=options.debug,
        manager=options.manager,
        storage_root=options.storage_root,
    )
    if options.export_detailed_csv:
        _write_combined_detailed_export(
            results=results,
            storage_root=options.storage_root,
            manager=options.manager,
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
    partitions: Sequence[FactorTimeframeAnalysisPartitionRef],
    *,
    year_filter: tuple[int, ...] | None,
) -> tuple[DiscoveredWorkItem, ...]:
    """Group FTA partitions into manager/year work items."""
    year_allowlist = set(year_filter) if year_filter is not None else None
    items: list[DiscoveredWorkItem] = []
    for partition in partitions:
        if year_allowlist is not None and partition.year not in year_allowlist:
            continue
        items.append(DiscoveredWorkItem(manager=partition.manager, year=partition.year))
    return tuple(
        sorted(
            items,
            key=lambda item: (item.manager, item.year),
        )
    )


async def _run_worker_pool(
    *,
    fta_repository: FactorTimeframeAnalysisRepository,
    combination_repository: FactorCombinationRepository,
    work: Sequence[DiscoveredWorkItem],
    worker_count: int,
    overwrite: bool,
    export_detailed_csv: bool,
    debug: bool,
    manager: str,
    storage_root: Path,
) -> tuple[FactorCombinationTaskResult, ...]:
    """Drain work items through a bounded asyncio worker pool."""
    if len(work) == 0:
        return ()

    queue: asyncio.Queue[DiscoveredWorkItem | None] = asyncio.Queue()
    for item in work:
        queue.put_nowait(item)
    for _ in range(worker_count):
        queue.put_nowait(None)

    collected: list[FactorCombinationTaskResult] = []
    lock = asyncio.Lock()

    async def worker() -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                results = await asyncio.to_thread(
                    _generate_year_partitions,
                    fta_repository,
                    combination_repository,
                    manager=item.manager,
                    year=item.year,
                    overwrite=overwrite,
                    export_detailed_csv=export_detailed_csv,
                    debug=debug,
                    storage_root=storage_root,
                )
                for result in results:
                    _print_progress(result)
                async with lock:
                    collected.extend(results)
            finally:
                queue.task_done()

    worker_tasks = [
        asyncio.create_task(worker(), name=f"generate-combination-worker-{index}")
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
            key=lambda result: (result.year, result.timeframe),
        )
    )


def _generate_year_partitions(
    fta_repository: FactorTimeframeAnalysisRepository,
    combination_repository: FactorCombinationRepository,
    *,
    manager: str,
    year: int,
    overwrite: bool,
    export_detailed_csv: bool,
    debug: bool,
    storage_root: Path,
) -> tuple[FactorCombinationTaskResult, ...]:
    """Generate combination partitions for one FTA year synchronously.

    Loads the FTA frame, builds all combinations, partitions output by
    the ``timeframe`` column, then saves or skips each partition.

    Returns one ``FactorCombinationTaskResult`` per timeframe partition
    in the combination output, plus a single FAIL result for the whole
    year if loading or building raises before any partition can be saved.
    """
    try:
        fta_frame = fta_repository.load(
            manager=manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            year=year,
        )
        engine = SimpleFactorCombinationEngine()
        combination_output = engine.build(fta_frame)
    except Exception as exc:
        _log_year_failure(manager=manager, year=year, exc=exc, debug=debug)
        return (
            FactorCombinationTaskResult(
                year=year,
                timeframe="",
                status="failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
                error_code=exc.error_code if isinstance(exc, CQROSError) else None,
            ),
        )

    source_selection_version = _extract_source_selection_version(fta_frame)

    results: list[FactorCombinationTaskResult] = []
    timeframes = combination_output.select(_COL_TIMEFRAME).unique().to_series().to_list()
    for timeframe in sorted(timeframes):
        partition_frame = combination_output.filter(pl.col(_COL_TIMEFRAME) == timeframe)
        result = _save_combination_partition(
            combination_repository=combination_repository,
            partition_frame=partition_frame,
            fta_frame=fta_frame,
            manager=manager,
            year=year,
            timeframe=timeframe,
            overwrite=overwrite,
            export_detailed_csv=export_detailed_csv,
            debug=debug,
            storage_root=storage_root,
            source_selection_version=source_selection_version,
        )
        results.append(result)

    return tuple(results)


def _save_combination_partition(
    *,
    combination_repository: FactorCombinationRepository,
    partition_frame: pl.DataFrame,
    fta_frame: pl.DataFrame,
    manager: str,
    year: int,
    timeframe: Timeframe,
    overwrite: bool,
    export_detailed_csv: bool,
    debug: bool,
    storage_root: Path,
    source_selection_version: str,
) -> FactorCombinationTaskResult:
    """Save one timeframe combination partition synchronously."""
    if not overwrite and combination_repository.exists(
        manager=manager,
        exchange=_EXCHANGE,
        market=_MARKET,
        timeframe=timeframe,
        year=year,
    ):
        return FactorCombinationTaskResult(year=year, timeframe=timeframe, status="skipped")

    try:
        combination_repository.save(
            partition_frame,
            manager=manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            timeframe=timeframe,
            year=year,
        )
        detailed_audit: pl.DataFrame | None = None
        if export_detailed_csv:
            detailed_audit = build_combination_detailed_audit_frame(
                partition_frame,
                manager=manager,
                exchange=_EXCHANGE,
                market=_MARKET,
                year=year,
                source_fta_version=str(year),
                source_selection_version=source_selection_version,
            )
            write_combination_detailed_csv(
                detailed_audit,
                combination_detailed_csv_path(
                    storage_root,
                    manager=manager,
                    exchange=_EXCHANGE,
                    market=_MARKET,
                    timeframe=timeframe,
                    year=year,
                ),
            )
    except Exception as exc:
        _log_partition_failure(
            manager=manager,
            year=year,
            timeframe=timeframe,
            exc=exc,
            debug=debug,
        )
        return FactorCombinationTaskResult(
            year=year,
            timeframe=timeframe,
            status="failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
            error_code=exc.error_code if isinstance(exc, CQROSError) else None,
        )

    return FactorCombinationTaskResult(
        year=year,
        timeframe=timeframe,
        status="succeeded",
        rows_generated=partition_frame.height,
        detailed_audit=detailed_audit,
    )


def _extract_source_selection_version(fta_frame: pl.DataFrame) -> str:
    """Extract the first non-null source_selection_version from the FTA frame."""
    if _COL_SOURCE_SELECTION_VERSION not in fta_frame.columns:
        return "unspecified"
    series = fta_frame.select(_COL_SOURCE_SELECTION_VERSION).drop_nulls().to_series()
    if series.is_empty():
        return "unspecified"
    return str(series[0])


def _write_combined_detailed_export(
    *,
    results: Sequence[FactorCombinationTaskResult],
    storage_root: Path,
    manager: str,
) -> None:
    """Write the combined detailed audit CSV from successful partition exports."""
    frames = [
        result.detailed_audit
        for result in results
        if result.status == "succeeded" and result.detailed_audit is not None
    ]
    if len(frames) == 0:
        return
    write_combined_detailed_csv(
        frames,
        combined_detailed_csv_path(
            storage_root,
            manager=manager,
            exchange=_EXCHANGE,
            market=_MARKET,
        ),
    )


def _print_progress(result: FactorCombinationTaskResult) -> None:
    """Print a deterministic one-line progress record for a task result."""
    label = f"{result.year}/{result.timeframe}" if result.timeframe else str(result.year)
    if result.status == "succeeded":
        rows = result.rows_generated if result.rows_generated is not None else 0
        message = f"OK {label} rows={rows}"
    elif result.status == "skipped":
        message = f"SKIP {label}"
    else:
        error_type = result.error_type if result.error_type is not None else "Exception"
        message = f"FAIL {label} {error_type}"
    print(message, flush=True)


def _log_year_failure(
    *,
    manager: str,
    year: int,
    exc: BaseException,
    debug: bool,
) -> None:
    """Log a year-level combination generation failure."""
    log_extra = {
        "manager": manager,
        "year": year,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "error_code": exc.error_code if isinstance(exc, CQROSError) else None,
    }
    if debug:
        _logger.exception(
            "Failed combination generation for year; continuing",
            extra=log_extra,
            exc_info=True,
        )
    else:
        _logger.warning(
            "Failed combination generation for year; continuing",
            extra=log_extra,
        )


def _log_partition_failure(
    *,
    manager: str,
    year: int,
    timeframe: Timeframe,
    exc: BaseException,
    debug: bool,
) -> None:
    """Log a partition-level combination save failure without aborting the run."""
    log_extra = {
        "manager": manager,
        "year": year,
        "timeframe": timeframe,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "error_code": exc.error_code if isinstance(exc, CQROSError) else None,
    }
    if debug:
        _logger.exception(
            "Failed combination generation partition; continuing",
            extra=log_extra,
            exc_info=True,
        )
    else:
        _logger.warning(
            "Failed combination generation partition; continuing",
            extra=log_extra,
        )


def _build_summary(
    *,
    options: FactorCombinationGenerationOptions,
    work: Sequence[DiscoveredWorkItem],
    results: Sequence[FactorCombinationTaskResult],
    duration_seconds: float,
    output_directory: Path,
) -> FactorCombinationGenerationSummary:
    """Aggregate task results into a generation report."""
    panels_processed: set[tuple[int, Timeframe]] = set()
    successful_tasks = 0
    failed_tasks = 0
    skipped_tasks = 0
    rows = 0
    failed_labels: set[str] = set()

    for result in results:
        if result.timeframe:
            panels_processed.add((result.year, result.timeframe))
        if result.status == "succeeded":
            successful_tasks += 1
            if result.rows_generated is not None:
                rows += result.rows_generated
        elif result.status == "skipped":
            skipped_tasks += 1
        else:
            failed_tasks += 1
            label = f"{result.year}/{result.timeframe}" if result.timeframe else str(result.year)
            failed_labels.add(label)

    panels = len(panels_processed) if results else len(work)

    return FactorCombinationGenerationSummary(
        manager=options.manager,
        panels=panels,
        rows=rows,
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
