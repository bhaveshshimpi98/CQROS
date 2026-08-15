"""CQROS Alpha generation CLI.

Purpose:
    Provide an argparse-based production entry point that discovers canonical
    Factor Orthogonalization partitions and executes combination-unit Alpha
    generation across manager/timeframe/year/symbol work items with bounded
    concurrency.

Responsibilities:
    - Parse CLI arguments for Alpha dataset generation
    - Discover Factor Orthogonalization partitions through
      ``FactorOrthogonalizationRepository``
    - Discover symbols through ``FactorsRepository.discover_symbols``
    - Generate Alpha via ``SimpleAlphaEngine`` / ``AlphaRegistry`` using
      ``FactorsObservationLoader``
    - Persist non-empty partitions through ``AlphaRepository`` (same save
      contract used by ``AlphaPipeline``)
    - Honor ``--overwrite``, ``--symbols``, worker concurrency, and debug
      logging
    - Report generated, skipped, empty, and failed scopes
    - Print deterministic progress and a final summary

Dependencies:
    ``argparse``, ``asyncio``, ``polars``, ``cqros.alpha``, ``cqros.config``,
    ``cqros.core``, ``cqros.factor_orthogonalization``,
    ``cqros.factor_selection``, ``cqros.factors``, and ``cqros.storage``.

Public API:
    ``DiscoveredWorkItem``, ``AlphaGenerationOptions``,
    ``AlphaGenerationSummary``, ``AlphaTaskResult``, ``build_options``,
    ``build_parser``, ``discover_work``, ``format_summary``, ``main``,
    ``run_generation``.

Notes:
    This module is a thin composition root. Alpha semantics remain exclusively
    in ``SimpleAlphaEngine``. Factor Orthogonalization partitions are never
    exploded into member factors. Persistence layout remains
    ``alpha/{manager}/{exchange}/{market}/{symbol}/{timeframe}/{year}.parquet``.
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

from cqros.alpha import (
    AlphaRegistry,
    AlphaRepository,
    SimpleAlphaEngine,
)
from cqros.alpha.detailed_export import (
    combined_detailed_csv_path,
    detailed_csv_path,
    write_combined_detailed_csv,
    write_detailed_csv,
)
from cqros.alpha.exceptions import AlphaError
from cqros.config.models import ResearchConfig
from cqros.core.constants import (
    DEFAULT_STORAGE_ROOT,
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
    STORAGE_DIR_ALPHA,
)
from cqros.core.exceptions import CQROSError, ValidationError
from cqros.core.types import Symbol, Timeframe
from cqros.factor_orthogonalization import FactorOrthogonalizationRepository
from cqros.factor_orthogonalization.repository import FactorOrthogonalizationPartitionRef
from cqros.factor_selection import FactorsObservationLoader
from cqros.factors import FactorsRepository
from cqros.storage import ParquetStore, StorageLayout

__all__ = [
    "AlphaGenerationOptions",
    "AlphaGenerationSummary",
    "AlphaTaskResult",
    "DiscoveredWorkItem",
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

_ERROR_WORKERS: Final[str] = "CLI-GENERATE-ALPHA-001"
_ERROR_YEAR: Final[str] = "CLI-GENERATE-ALPHA-002"
_ERROR_MANAGER: Final[str] = "CLI-GENERATE-ALPHA-003"
_ERROR_SYMBOL: Final[str] = "CLI-GENERATE-ALPHA-004"

_ERROR_NO_COMBINATIONS: Final[str] = "ALPHA_NO_COMBINATIONS"


@dataclass(frozen=True, slots=True)
class AlphaGenerationOptions:
    """Immutable CLI options for Alpha dataset generation."""

    storage_root: Path
    manager: str
    years: tuple[int, ...] | None
    symbols: tuple[Symbol, ...] | None
    overwrite: bool
    export_detailed_csv: bool
    workers: int
    verbose: bool
    debug: bool


@dataclass(frozen=True, slots=True)
class DiscoveredWorkItem:
    """One discovered manager/timeframe/year/symbol Alpha generation scope."""

    manager: str
    timeframe: Timeframe
    year: int
    symbol: Symbol


@dataclass(frozen=True, slots=True)
class AlphaTaskResult:
    """Immutable result for one Alpha partition task."""

    year: int
    timeframe: Timeframe
    symbol: Symbol
    status: str
    rows_generated: int | None = None
    detailed_frame: pl.DataFrame | None = None
    error_type: str | None = None
    error_message: str | None = None
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class AlphaGenerationSummary:
    """Immutable aggregate summary for an Alpha-generation run."""

    manager: str
    panels: int
    rows: int
    successful_tasks: int
    failed_tasks: int
    skipped_tasks: int
    empty_tasks: int
    duration_seconds: float
    output_directory: Path
    failed_task_labels: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    """Create the Alpha-generation argument parser."""
    parser = argparse.ArgumentParser(
        prog="cqros-generate-factor-alpha",
        description=(
            "Generate CQROS Alpha datasets from discovered Factor " "Orthogonalization partitions."
        ),
    )
    parser.add_argument(
        "--manager",
        dest="manager",
        required=True,
        metavar="NAME",
        help="Order manager identity used for discovery and Alpha lineage.",
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
        "--symbols",
        dest="symbols",
        nargs="*",
        default=None,
        metavar="SYMBOL",
        help="Optional symbol allowlist (0..N values). Omit to discover all.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate Alpha partitions that already exist.",
    )
    parser.add_argument(
        "--export-detailed-csv",
        dest="export_detailed_csv",
        action="store_true",
        help=(
            "Write detailed CSV exports (per-partition and combined) "
            "alongside canonical Parquet Alpha datasets."
        ),
    )
    parser.add_argument(
        "--workers",
        dest="workers",
        type=int,
        default=_DEFAULT_WORKER_COUNT,
        metavar="INT",
        help=f"Maximum concurrent Alpha partitions (default: {_DEFAULT_WORKER_COUNT}).",
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


def build_options(args: argparse.Namespace) -> AlphaGenerationOptions:
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

    storage_root = (
        Path(args.storage_root) if args.storage_root is not None else Path(DEFAULT_STORAGE_ROOT)
    )

    return AlphaGenerationOptions(
        storage_root=storage_root,
        manager=manager,
        years=_normalize_years(args.years),
        symbols=_normalize_symbols(args.symbols),
        overwrite=bool(args.overwrite),
        export_detailed_csv=bool(args.export_detailed_csv),
        workers=workers,
        verbose=bool(args.verbose),
        debug=bool(args.debug),
    )


def discover_work(
    orthogonalization_repository: FactorOrthogonalizationRepository,
    factors_repository: FactorsRepository,
    options: AlphaGenerationOptions,
) -> tuple[DiscoveredWorkItem, ...]:
    """Discover Alpha-ready FO partitions crossed with Factors symbols."""
    partitions = orthogonalization_repository.discover_partitions(
        managers=(options.manager,),
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    symbols = _resolve_symbols(factors_repository, options)
    return _expand_work_items(
        partitions,
        symbols=symbols,
        year_filter=options.years,
    )


def format_summary(summary: AlphaGenerationSummary) -> str:
    """Render a deterministic Alpha-generation summary report."""
    lines = [
        "=====================================",
        "CQROS Alpha Generation Summary",
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
        f"Empty: {summary.empty_tasks}",
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
    """Run the Alpha-generation CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        options = build_options(args)
        _configure_logging(verbose=options.verbose, debug=options.debug)
        layout = StorageLayout(options.storage_root)
        datastore = ParquetStore()
        orthogonalization_repository = FactorOrthogonalizationRepository(layout, datastore)
        factors_repository = FactorsRepository(layout, datastore)
        alpha_repository = AlphaRepository(layout, datastore)
        work = discover_work(orthogonalization_repository, factors_repository, options)
        summary = await run_generation(
            orthogonalization_repository=orthogonalization_repository,
            factors_repository=factors_repository,
            alpha_repository=alpha_repository,
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
    orthogonalization_repository: FactorOrthogonalizationRepository,
    factors_repository: FactorsRepository,
    alpha_repository: AlphaRepository,
    layout: StorageLayout,
    options: AlphaGenerationOptions,
    work: Sequence[DiscoveredWorkItem],
) -> AlphaGenerationSummary:
    """Execute discovered work through a bounded worker pool."""
    started = time.perf_counter()
    output_directory = options.storage_root / STORAGE_DIR_ALPHA

    if len(work) == 0:
        return AlphaGenerationSummary(
            manager=options.manager,
            panels=0,
            rows=0,
            successful_tasks=0,
            failed_tasks=0,
            skipped_tasks=0,
            empty_tasks=0,
            duration_seconds=time.perf_counter() - started,
            output_directory=output_directory,
            failed_task_labels=(),
        )

    results = await _run_worker_pool(
        orthogonalization_repository=orthogonalization_repository,
        factors_repository=factors_repository,
        alpha_repository=alpha_repository,
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


def _normalize_symbols(values: Sequence[str] | None) -> tuple[Symbol, ...] | None:
    """Validate and freeze optional symbol filters."""
    if values is None:
        return None
    normalized: list[Symbol] = []
    for symbol in values:
        stripped = str(symbol).strip()
        if stripped == "":
            continue
        if stripped not in normalized:
            normalized.append(stripped)
    if len(values) > 0 and len(normalized) == 0:
        raise ValidationError(
            "symbols must contain at least one non-empty symbol",
            error_code=_ERROR_SYMBOL,
            details={"parameter": "symbols", "value": values},
        )
    return tuple(normalized) if normalized else None


def _resolve_symbols(
    factors_repository: FactorsRepository,
    options: AlphaGenerationOptions,
) -> tuple[Symbol, ...]:
    """Resolve the symbol universe for Alpha generation."""
    if options.symbols is not None:
        return options.symbols
    return factors_repository.discover_symbols(
        manager=options.manager,
        exchange=_EXCHANGE,
        market=_MARKET,
    )


def _expand_work_items(
    partitions: Sequence[FactorOrthogonalizationPartitionRef],
    *,
    symbols: Sequence[Symbol],
    year_filter: tuple[int, ...] | None,
) -> tuple[DiscoveredWorkItem, ...]:
    """Cross FO partitions with symbols into discovered work items."""
    year_allowlist = set(year_filter) if year_filter is not None else None
    items: list[DiscoveredWorkItem] = []
    for partition in partitions:
        if year_allowlist is not None and partition.year not in year_allowlist:
            continue
        for symbol in symbols:
            items.append(
                DiscoveredWorkItem(
                    manager=partition.manager,
                    timeframe=partition.timeframe,
                    year=partition.year,
                    symbol=symbol,
                )
            )
    return tuple(
        sorted(
            items,
            key=lambda item: (item.manager, item.timeframe, item.year, item.symbol),
        )
    )


async def _run_worker_pool(
    *,
    orthogonalization_repository: FactorOrthogonalizationRepository,
    factors_repository: FactorsRepository,
    alpha_repository: AlphaRepository,
    layout: StorageLayout,
    work: Sequence[DiscoveredWorkItem],
    options: AlphaGenerationOptions,
) -> tuple[AlphaTaskResult, ...]:
    """Drain work items through a bounded asyncio worker pool."""
    if len(work) == 0:
        return ()

    queue: asyncio.Queue[DiscoveredWorkItem | None] = asyncio.Queue()
    for item in work:
        queue.put_nowait(item)
    for _ in range(options.workers):
        queue.put_nowait(None)

    collected: list[AlphaTaskResult] = []
    lock = asyncio.Lock()

    async def worker() -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                result = await asyncio.to_thread(
                    _generate_partition,
                    orthogonalization_repository,
                    factors_repository,
                    alpha_repository,
                    layout,
                    manager=item.manager,
                    timeframe=item.timeframe,
                    year=item.year,
                    symbol=item.symbol,
                    options=options,
                )
                _print_progress(result)
                async with lock:
                    collected.append(result)
            finally:
                queue.task_done()

    worker_tasks = [
        asyncio.create_task(worker(), name=f"generate-alpha-worker-{index}")
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
            key=lambda result: (result.year, result.timeframe, result.symbol),
        )
    )


def _generate_partition(
    orthogonalization_repository: FactorOrthogonalizationRepository,
    factors_repository: FactorsRepository,
    alpha_repository: AlphaRepository,
    layout: StorageLayout,
    *,
    manager: str,
    timeframe: Timeframe,
    year: int,
    symbol: Symbol,
    options: AlphaGenerationOptions,
) -> AlphaTaskResult:
    """Generate one Alpha partition synchronously."""
    if not options.overwrite and alpha_repository.exists(
        manager=manager,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=symbol,
        timeframe=timeframe,
        year=year,
    ):
        return AlphaTaskResult(
            year=year,
            timeframe=timeframe,
            symbol=symbol,
            status="skipped",
        )

    if not factors_repository.exists(
        manager=manager,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=symbol,
        timeframe=timeframe,
        year=year,
    ):
        return AlphaTaskResult(
            year=year,
            timeframe=timeframe,
            symbol=symbol,
            status="empty",
            rows_generated=0,
        )

    try:
        orthogonalization_frame = orthogonalization_repository.load(
            manager=manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            timeframe=timeframe,
            year=year,
        )
        observation_source = FactorsObservationLoader(
            layout,
            manager=manager,
            year=year,
            exchange=_EXCHANGE,
            market=_MARKET,
        )
        engine = SimpleAlphaEngine(observation_source=observation_source)
        registry = AlphaRegistry(engine=engine)

        try:
            created = registry.build(orthogonalization_frame, symbol=symbol)
        except AlphaError as exc:
            if exc.error_code == _ERROR_NO_COMBINATIONS:
                return AlphaTaskResult(
                    year=year,
                    timeframe=timeframe,
                    symbol=symbol,
                    status="empty",
                    rows_generated=0,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    error_code=exc.error_code,
                )
            raise

        if created.height == 0:
            return AlphaTaskResult(
                year=year,
                timeframe=timeframe,
                symbol=symbol,
                status="empty",
                rows_generated=0,
            )

        alpha_repository.save(
            created,
            manager=manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        )
        persisted = created

        detailed_frame: pl.DataFrame | None = None
        if options.export_detailed_csv:
            detailed_frame = persisted
            write_detailed_csv(
                detailed_frame,
                detailed_csv_path(
                    options.storage_root,
                    manager=manager,
                    exchange=_EXCHANGE,
                    market=_MARKET,
                    symbol=symbol,
                    timeframe=timeframe,
                    year=year,
                ),
            )
    except Exception as exc:
        _log_partition_failure(
            manager=manager,
            year=year,
            timeframe=timeframe,
            symbol=symbol,
            exc=exc,
            debug=options.debug,
        )
        return AlphaTaskResult(
            year=year,
            timeframe=timeframe,
            symbol=symbol,
            status="failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
            error_code=exc.error_code if isinstance(exc, CQROSError) else None,
        )

    return AlphaTaskResult(
        year=year,
        timeframe=timeframe,
        symbol=symbol,
        status="succeeded",
        rows_generated=persisted.height,
        detailed_frame=detailed_frame,
    )


def _write_combined_detailed_export(
    *,
    results: Sequence[AlphaTaskResult],
    storage_root: Path,
    manager: str,
) -> None:
    """Write the combined detailed CSV from successful partition exports."""
    frames = [
        result.detailed_frame
        for result in results
        if result.status == "succeeded" and result.detailed_frame is not None
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


def _print_progress(result: AlphaTaskResult) -> None:
    """Print a deterministic one-line progress record for a task result."""
    label = f"{result.year}/{result.timeframe}/{result.symbol}"
    if result.status == "succeeded":
        rows = result.rows_generated if result.rows_generated is not None else 0
        message = f"OK {label} rows={rows}"
    elif result.status == "skipped":
        message = f"SKIP {label}"
    elif result.status == "empty":
        message = f"EMPTY {label}"
    else:
        error_type = result.error_type if result.error_type is not None else "Exception"
        message = f"FAIL {label} {error_type}"
    print(message, flush=True)


def _log_partition_failure(
    *,
    manager: str,
    year: int,
    timeframe: Timeframe,
    symbol: Symbol,
    exc: BaseException,
    debug: bool,
) -> None:
    """Log a partition-level Alpha failure without aborting the run."""
    log_extra = {
        "manager": manager,
        "year": year,
        "timeframe": timeframe,
        "symbol": symbol,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "error_code": exc.error_code if isinstance(exc, CQROSError) else None,
    }
    if debug:
        _logger.exception(
            "Failed Alpha generation partition; continuing",
            extra=log_extra,
            exc_info=True,
        )
    else:
        _logger.warning(
            "Failed Alpha generation partition; continuing",
            extra=log_extra,
        )


def _build_summary(
    *,
    options: AlphaGenerationOptions,
    work: Sequence[DiscoveredWorkItem],
    results: Sequence[AlphaTaskResult],
    duration_seconds: float,
    output_directory: Path,
) -> AlphaGenerationSummary:
    """Aggregate task results into a generation report."""
    panels_processed: set[tuple[int, Timeframe, Symbol]] = set()
    successful_tasks = 0
    failed_tasks = 0
    skipped_tasks = 0
    empty_tasks = 0
    rows = 0
    failed_labels: set[str] = set()

    for result in results:
        panels_processed.add((result.year, result.timeframe, result.symbol))
        if result.status == "succeeded":
            successful_tasks += 1
            if result.rows_generated is not None:
                rows += result.rows_generated
        elif result.status == "skipped":
            skipped_tasks += 1
        elif result.status == "empty":
            empty_tasks += 1
        else:
            failed_tasks += 1
            failed_labels.add(f"{result.year}/{result.timeframe}/{result.symbol}")

    panels = len(panels_processed) if results else len(work)

    return AlphaGenerationSummary(
        manager=options.manager,
        panels=panels,
        rows=rows,
        successful_tasks=successful_tasks,
        failed_tasks=failed_tasks,
        skipped_tasks=skipped_tasks,
        empty_tasks=empty_tasks,
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
