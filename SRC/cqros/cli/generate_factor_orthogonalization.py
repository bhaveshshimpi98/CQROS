"""CQROS factor-orthogonalization generation CLI.

Purpose:
    Provide an argparse-based production entry point that discovers canonical
    Factor Combination panels and executes
    ``SimpleFactorOrthogonalizationEngine`` across manager/timeframe/year
    work items with bounded concurrency.

Responsibilities:
    - Parse CLI arguments for factor orthogonalization dataset generation
    - Discover available Combination partitions through
      ``FactorCombinationRepository``
    - Resolve validation windows from Factor Validation partitions
    - Load combination frames and build orthogonalization via
      ``SimpleFactorOrthogonalizationEngine`` using ``FactorsObservationLoader``
    - Save each partition via ``FactorOrthogonalizationRepository``
    - Honor ``--overwrite``, worker concurrency, and debug logging
    - Print deterministic progress and a final summary

Dependencies:
    ``argparse``, ``asyncio``, ``polars``, ``cqros.config``, ``cqros.core``,
    ``cqros.factor_combination``, ``cqros.factor_orthogonalization``,
    ``cqros.factor_selection``, ``cqros.factor_timeframe_analysis``,
    ``cqros.factor_validation``, and ``cqros.storage``.

Public API:
    ``DiscoveredWorkItem``, ``FactorOrthogonalizationGenerationOptions``,
    ``FactorOrthogonalizationGenerationSummary``,
    ``FactorOrthogonalizationTaskResult``, ``build_options``, ``build_parser``,
    ``discover_work``, ``format_summary``, ``main``, ``run_generation``.
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
    STORAGE_DIR_FACTOR_ORTHOGONALIZATION,
)
from cqros.core.exceptions import CQROSError, ValidationError
from cqros.core.types import Timeframe
from cqros.factor_combination import FactorCombinationRepository
from cqros.factor_combination.repository import FactorCombinationPartitionRef
from cqros.factor_orthogonalization import (
    DEFAULT_MAX_COMBINATION_CORRELATION,
    DEFAULT_MIN_CORRELATION_OVERLAP,
    FactorOrthogonalizationRepository,
    LineageContext,
    SimpleFactorOrthogonalizationEngine,
    combined_detailed_csv_path,
    write_combined_detailed_csv,
)
from cqros.factor_orthogonalization import (
    build_detailed_audit_frame as build_ortho_detailed_audit_frame,
)
from cqros.factor_orthogonalization import (
    detailed_csv_path as ortho_detailed_csv_path,
)
from cqros.factor_orthogonalization import (
    write_detailed_csv as write_ortho_detailed_csv,
)
from cqros.factor_selection import FactorsObservationLoader
from cqros.factor_timeframe_analysis import FactorTimeframeAnalysisRepository
from cqros.factor_validation import FactorValidationRepository
from cqros.storage import ParquetStore, StorageLayout

__all__ = [
    "DiscoveredWorkItem",
    "FactorOrthogonalizationGenerationOptions",
    "FactorOrthogonalizationGenerationSummary",
    "FactorOrthogonalizationTaskResult",
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

_ERROR_WORKERS: Final[str] = "CLI-GENERATE-FORTH-001"
_ERROR_YEAR: Final[str] = "CLI-GENERATE-FORTH-002"
_ERROR_MANAGER: Final[str] = "CLI-GENERATE-FORTH-003"
_ERROR_CORR: Final[str] = "CLI-GENERATE-FORTH-004"
_ERROR_OVERLAP: Final[str] = "CLI-GENERATE-FORTH-005"
_ERROR_VALIDATION_WINDOW: Final[str] = "CLI-GENERATE-FORTH-006"

_COL_SOURCE_SELECTION_VERSION: Final[str] = "source_selection_version"


@dataclass(frozen=True, slots=True)
class FactorOrthogonalizationGenerationOptions:
    """Immutable CLI options for factor orthogonalization dataset generation."""

    storage_root: Path
    manager: str
    years: tuple[int, ...] | None
    overwrite: bool
    export_detailed_csv: bool
    max_combination_correlation: float
    min_correlation_overlap: int
    workers: int
    verbose: bool
    debug: bool


@dataclass(frozen=True, slots=True)
class DiscoveredWorkItem:
    """One discovered manager/timeframe/year combination partition."""

    manager: str
    timeframe: Timeframe
    year: int


@dataclass(frozen=True, slots=True)
class FactorOrthogonalizationTaskResult:
    """Immutable result for one orthogonalization partition task."""

    year: int
    timeframe: Timeframe
    status: str
    rows_generated: int | None = None
    detailed_audit: pl.DataFrame | None = None
    error_type: str | None = None
    error_message: str | None = None
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class FactorOrthogonalizationGenerationSummary:
    """Immutable aggregate summary for an orthogonalization-generation run."""

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
    """Create the factor-orthogonalization-generation argument parser."""
    parser = argparse.ArgumentParser(
        prog="cqros-generate-factor-orthogonalization",
        description=(
            "Generate CQROS factor orthogonalization datasets from discovered "
            "Factor Combination panels."
        ),
    )
    parser.add_argument(
        "--manager",
        dest="manager",
        required=True,
        metavar="NAME",
        help="Order manager identity used for discovery and orthogonalization lineage.",
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
        help="Regenerate orthogonalization partitions that already exist.",
    )
    parser.add_argument(
        "--export-detailed-csv",
        dest="export_detailed_csv",
        action="store_true",
        help=(
            "Write detailed audit CSV exports (per-timeframe and combined) "
            "alongside canonical Parquet orthogonalization datasets."
        ),
    )
    parser.add_argument(
        "--max-combination-correlation",
        dest="max_combination_correlation",
        type=float,
        default=DEFAULT_MAX_COMBINATION_CORRELATION,
        metavar="FLOAT",
        help=(
            "Absolute Pearson redundancy threshold for combinations "
            f"(default: {DEFAULT_MAX_COMBINATION_CORRELATION})."
        ),
    )
    parser.add_argument(
        "--min-correlation-overlap",
        dest="min_correlation_overlap",
        type=int,
        default=DEFAULT_MIN_CORRELATION_OVERLAP,
        metavar="INT",
        help=(
            "Minimum pairwise complete observations for a correlation check "
            f"(default: {DEFAULT_MIN_CORRELATION_OVERLAP})."
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


def build_options(args: argparse.Namespace) -> FactorOrthogonalizationGenerationOptions:
    """Map parsed CLI arguments onto generation options."""
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

    max_corr = float(args.max_combination_correlation)
    if max_corr <= 0.0 or max_corr >= 1.0:
        raise ValidationError(
            "max_combination_correlation must be in (0, 1)",
            error_code=_ERROR_CORR,
            details={"parameter": "max_combination_correlation", "value": max_corr},
        )

    min_overlap = int(args.min_correlation_overlap)
    if min_overlap <= 0:
        raise ValidationError(
            "min_correlation_overlap must be a positive integer",
            error_code=_ERROR_OVERLAP,
            details={"parameter": "min_correlation_overlap", "value": min_overlap},
        )

    storage_root = (
        Path(args.storage_root) if args.storage_root is not None else Path(DEFAULT_STORAGE_ROOT)
    )

    return FactorOrthogonalizationGenerationOptions(
        storage_root=storage_root,
        manager=manager,
        years=_normalize_years(args.years),
        overwrite=bool(args.overwrite),
        export_detailed_csv=bool(args.export_detailed_csv),
        max_combination_correlation=max_corr,
        min_correlation_overlap=min_overlap,
        workers=workers,
        verbose=bool(args.verbose),
        debug=bool(args.debug),
    )


def discover_work(
    combination_repository: FactorCombinationRepository,
    options: FactorOrthogonalizationGenerationOptions,
) -> tuple[DiscoveredWorkItem, ...]:
    """Discover orthogonalization-ready combination partitions."""
    partitions = combination_repository.discover_partitions(
        managers=(options.manager,),
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    return _group_partitions(partitions, year_filter=options.years)


def format_summary(summary: FactorOrthogonalizationGenerationSummary) -> str:
    """Render a deterministic orthogonalization-generation summary report."""
    lines = [
        "=====================================",
        "CQROS Factor Orthogonalization Generation Summary",
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
    """Run the factor-orthogonalization-generation CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        options = build_options(args)
        _configure_logging(verbose=options.verbose, debug=options.debug)
        layout = StorageLayout(options.storage_root)
        datastore = ParquetStore()
        combination_repository = FactorCombinationRepository(layout, datastore)
        orthogonalization_repository = FactorOrthogonalizationRepository(layout, datastore)
        validation_repository = FactorValidationRepository(layout, datastore)
        fta_repository = FactorTimeframeAnalysisRepository(layout, datastore)
        work = discover_work(combination_repository, options)
        summary = await run_generation(
            combination_repository=combination_repository,
            orthogonalization_repository=orthogonalization_repository,
            validation_repository=validation_repository,
            fta_repository=fta_repository,
            layout=layout,
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
    combination_repository: FactorCombinationRepository,
    orthogonalization_repository: FactorOrthogonalizationRepository,
    validation_repository: FactorValidationRepository,
    fta_repository: FactorTimeframeAnalysisRepository,
    layout: StorageLayout,
    options: FactorOrthogonalizationGenerationOptions,
    work: Sequence[DiscoveredWorkItem],
) -> FactorOrthogonalizationGenerationSummary:
    """Execute discovered work through a bounded panel worker pool."""
    started = time.perf_counter()
    output_directory = options.storage_root / STORAGE_DIR_FACTOR_ORTHOGONALIZATION

    if len(work) == 0:
        return FactorOrthogonalizationGenerationSummary(
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
        combination_repository=combination_repository,
        orthogonalization_repository=orthogonalization_repository,
        validation_repository=validation_repository,
        fta_repository=fta_repository,
        layout=layout,
        work=work,
        options=options,
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


def _group_partitions(
    partitions: Sequence[FactorCombinationPartitionRef],
    *,
    year_filter: tuple[int, ...] | None,
) -> tuple[DiscoveredWorkItem, ...]:
    """Group combination partitions into discovered work items."""
    year_allowlist = set(year_filter) if year_filter is not None else None
    items: list[DiscoveredWorkItem] = []
    for partition in partitions:
        if year_allowlist is not None and partition.year not in year_allowlist:
            continue
        items.append(
            DiscoveredWorkItem(
                manager=partition.manager,
                timeframe=partition.timeframe,
                year=partition.year,
            )
        )
    return tuple(
        sorted(
            items,
            key=lambda item: (item.manager, item.timeframe, item.year),
        )
    )


async def _run_worker_pool(
    *,
    combination_repository: FactorCombinationRepository,
    orthogonalization_repository: FactorOrthogonalizationRepository,
    validation_repository: FactorValidationRepository,
    fta_repository: FactorTimeframeAnalysisRepository,
    layout: StorageLayout,
    work: Sequence[DiscoveredWorkItem],
    options: FactorOrthogonalizationGenerationOptions,
) -> tuple[FactorOrthogonalizationTaskResult, ...]:
    """Drain work items through a bounded asyncio worker pool."""
    if len(work) == 0:
        return ()

    queue: asyncio.Queue[DiscoveredWorkItem | None] = asyncio.Queue()
    for item in work:
        queue.put_nowait(item)
    for _ in range(options.workers):
        queue.put_nowait(None)

    collected: list[FactorOrthogonalizationTaskResult] = []
    lock = asyncio.Lock()

    async def worker() -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                result = await asyncio.to_thread(
                    _generate_partition,
                    combination_repository,
                    orthogonalization_repository,
                    validation_repository,
                    fta_repository,
                    layout,
                    manager=item.manager,
                    timeframe=item.timeframe,
                    year=item.year,
                    options=options,
                )
                _print_progress(result)
                async with lock:
                    collected.append(result)
            finally:
                queue.task_done()

    worker_tasks = [
        asyncio.create_task(worker(), name=f"generate-orthogonalization-worker-{index}")
        for index in range(options.workers)
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


def _generate_partition(
    combination_repository: FactorCombinationRepository,
    orthogonalization_repository: FactorOrthogonalizationRepository,
    validation_repository: FactorValidationRepository,
    fta_repository: FactorTimeframeAnalysisRepository,
    layout: StorageLayout,
    *,
    manager: str,
    timeframe: Timeframe,
    year: int,
    options: FactorOrthogonalizationGenerationOptions,
) -> FactorOrthogonalizationTaskResult:
    """Generate one orthogonalization partition synchronously."""
    if not options.overwrite and orthogonalization_repository.exists(
        manager=manager,
        exchange=_EXCHANGE,
        market=_MARKET,
        timeframe=timeframe,
        year=year,
    ):
        return FactorOrthogonalizationTaskResult(
            year=year,
            timeframe=timeframe,
            status="skipped",
        )

    try:
        combination_frame = combination_repository.load(
            manager=manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            timeframe=timeframe,
            year=year,
        )
        validation_start, validation_end = _resolve_validation_window(
            validation_repository,
            manager=manager,
            timeframe=timeframe,
            year=year,
        )
        source_selection_version = _resolve_source_selection_version(
            fta_repository,
            manager=manager,
            year=year,
        )
        observation_source = FactorsObservationLoader(
            layout,
            manager=manager,
            year=year,
            exchange=_EXCHANGE,
            market=_MARKET,
        )
        engine = SimpleFactorOrthogonalizationEngine(
            observation_source=observation_source,
            max_combination_correlation=options.max_combination_correlation,
            min_overlap=options.min_correlation_overlap,
        )
        lineage = LineageContext(
            validation_start_time=validation_start,
            validation_end_time=validation_end,
            source_combination_version=str(year),
            source_fta_version=str(year),
            source_selection_version=source_selection_version,
            dataset_version=str(year),
        )
        output = engine.build(combination_frame, lineage=lineage)
        orthogonalization_repository.save(
            output,
            manager=manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            timeframe=timeframe,
            year=year,
        )
        detailed_audit: pl.DataFrame | None = None
        if options.export_detailed_csv:
            detailed_audit = build_ortho_detailed_audit_frame(
                output,
                manager=manager,
                exchange=_EXCHANGE,
                market=_MARKET,
                year=year,
            )
            write_ortho_detailed_csv(
                detailed_audit,
                ortho_detailed_csv_path(
                    options.storage_root,
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
            debug=options.debug,
        )
        return FactorOrthogonalizationTaskResult(
            year=year,
            timeframe=timeframe,
            status="failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
            error_code=exc.error_code if isinstance(exc, CQROSError) else None,
        )

    return FactorOrthogonalizationTaskResult(
        year=year,
        timeframe=timeframe,
        status="succeeded",
        rows_generated=output.height,
        detailed_audit=detailed_audit,
    )


def _resolve_validation_window(
    validation_repository: FactorValidationRepository,
    *,
    manager: str,
    timeframe: Timeframe,
    year: int,
) -> tuple[int, int]:
    """Resolve inclusive validation-window bounds from Factor Validation."""
    if not validation_repository.exists(
        manager=manager,
        exchange=_EXCHANGE,
        market=_MARKET,
        timeframe=timeframe,
        year=year,
    ):
        raise ValidationError(
            "factor validation partition required for orthogonalization window",
            error_code=_ERROR_VALIDATION_WINDOW,
            details={
                "manager": manager,
                "timeframe": timeframe,
                "year": year,
            },
        )
    validation = validation_repository.load(
        manager=manager,
        exchange=_EXCHANGE,
        market=_MARKET,
        timeframe=timeframe,
        year=year,
    )
    if validation.height == 0:
        raise ValidationError(
            "factor validation partition is empty; cannot resolve window",
            error_code=_ERROR_VALIDATION_WINDOW,
            details={
                "manager": manager,
                "timeframe": timeframe,
                "year": year,
            },
        )
    start_raw = validation["validation_start_time"].min()
    end_raw = validation["validation_end_time"].max()
    if start_raw is None or end_raw is None:
        raise ValidationError(
            "factor validation partition missing validation window bounds",
            error_code=_ERROR_VALIDATION_WINDOW,
            details={
                "validation_start_time": start_raw,
                "validation_end_time": end_raw,
            },
        )
    try:
        start_time = int(start_raw)  # type: ignore[arg-type]
        end_time = int(end_raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            "factor validation window bounds must be integer epoch milliseconds",
            error_code=_ERROR_VALIDATION_WINDOW,
            details={
                "validation_start_time": start_raw,
                "validation_end_time": end_raw,
            },
        ) from exc
    if start_time > end_time:
        raise ValidationError(
            "validation window start_time must be <= end_time",
            error_code=_ERROR_VALIDATION_WINDOW,
            details={
                "validation_start_time": start_time,
                "validation_end_time": end_time,
            },
        )
    return start_time, end_time


def _resolve_source_selection_version(
    fta_repository: FactorTimeframeAnalysisRepository,
    *,
    manager: str,
    year: int,
) -> str:
    """Extract source_selection_version from the FTA year panel when present."""
    if not fta_repository.exists(
        manager=manager,
        exchange=_EXCHANGE,
        market=_MARKET,
        year=year,
    ):
        return "unspecified"
    fta_frame = fta_repository.load(
        manager=manager,
        exchange=_EXCHANGE,
        market=_MARKET,
        year=year,
    )
    if _COL_SOURCE_SELECTION_VERSION not in fta_frame.columns:
        return "unspecified"
    series = fta_frame.select(_COL_SOURCE_SELECTION_VERSION).drop_nulls().to_series()
    if series.is_empty():
        return "unspecified"
    return str(series[0])


def _write_combined_detailed_export(
    *,
    results: Sequence[FactorOrthogonalizationTaskResult],
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


def _print_progress(result: FactorOrthogonalizationTaskResult) -> None:
    """Print a deterministic one-line progress record for a task result."""
    label = f"{result.year}/{result.timeframe}"
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
    manager: str,
    year: int,
    timeframe: Timeframe,
    exc: BaseException,
    debug: bool,
) -> None:
    """Log a partition-level orthogonalization failure without aborting the run."""
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
            "Failed orthogonalization generation partition; continuing",
            extra=log_extra,
            exc_info=True,
        )
    else:
        _logger.warning(
            "Failed orthogonalization generation partition; continuing",
            extra=log_extra,
        )


def _build_summary(
    *,
    options: FactorOrthogonalizationGenerationOptions,
    work: Sequence[DiscoveredWorkItem],
    results: Sequence[FactorOrthogonalizationTaskResult],
    duration_seconds: float,
    output_directory: Path,
) -> FactorOrthogonalizationGenerationSummary:
    """Aggregate task results into a generation report."""
    panels_processed: set[tuple[int, Timeframe]] = set()
    successful_tasks = 0
    failed_tasks = 0
    skipped_tasks = 0
    rows = 0
    failed_labels: set[str] = set()

    for result in results:
        panels_processed.add((result.year, result.timeframe))
        if result.status == "succeeded":
            successful_tasks += 1
            if result.rows_generated is not None:
                rows += result.rows_generated
        elif result.status == "skipped":
            skipped_tasks += 1
        else:
            failed_tasks += 1
            failed_labels.add(f"{result.year}/{result.timeframe}")

    panels = len(panels_processed) if results else len(work)

    return FactorOrthogonalizationGenerationSummary(
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
