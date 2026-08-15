"""CQROS training-dataset build CLI.

Purpose:
    Provide an argparse-based production entry point that discovers feature
    and label partitions and executes ``TrainingPipeline`` across the universe
    with bounded symbol concurrency.

Responsibilities:
    - Parse CLI arguments for training dataset generation
    - Discover available feature and label partitions with intersection logic
    - Load feature and label frames into ``TrainingPipeline``
    - Execute ``TrainingPipeline`` and persist via ``TrainingRepository``
    - Honor ``--overwrite``, worker concurrency, and debug logging
    - Print deterministic progress and a final summary

Dependencies:
    ``argparse``, ``asyncio``, ``polars``, ``cqros.config``, ``cqros.core``,
    ``cqros.training``, and ``cqros.storage``.

Public API:
    ``build_parser``, ``build_options``, ``build_training_pipeline``,
    ``discover_work``, ``format_summary``, ``run_generation``, and ``main``.

Notes:
    This module is a thin composition root. It does not implement joins,
    schema validation, or repository filesystem walks beyond calling
    repository discovery and load APIs.
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

from cqros.config.models import ResearchConfig
from cqros.core.constants import (
    DEFAULT_STORAGE_ROOT,
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
    STORAGE_DIR_TRAINING,
    SUPPORTED_TIMEFRAMES,
)
from cqros.core.exceptions import CQROSError, ValidationError
from cqros.core.types import Symbol, Timeframe
from cqros.storage import (
    FeatureRepository,
    LabelRepository,
    ParquetStore,
    StorageLayout,
    TrainingRepository,
)
from cqros.training import TrainingPipeline

__all__ = [
    "DiscoveredWorkItem",
    "TrainingDatasetOptions",
    "TrainingDatasetSummary",
    "TrainingTaskResult",
    "build_options",
    "build_parser",
    "build_training_pipeline",
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

_ERROR_WORKERS: Final[str] = "CLI-BUILD-TRAINING-001"
_ERROR_TIMEFRAME: Final[str] = "CLI-BUILD-TRAINING-002"
_ERROR_YEAR: Final[str] = "CLI-BUILD-TRAINING-003"


@dataclass(frozen=True, slots=True)
class TrainingDatasetOptions:
    """Immutable CLI options for training dataset generation.

    Attributes:
        storage_root: Storage root containing ``features``, ``labels``, and
            ``training``.
        symbols: Optional symbol allowlist. ``None`` discovers all.
        timeframes: Optional timeframe allowlist. ``None`` discovers all.
        years: Optional year allowlist. ``None`` discovers all.
        overwrite: When ``True``, regenerate existing training partitions.
        workers: Maximum concurrent symbols.
        verbose: When ``True``, enable INFO logging.
        debug: When ``True``, enable DEBUG logging and log full failure
            tracebacks.
    """

    storage_root: Path
    symbols: tuple[Symbol, ...] | None
    timeframes: tuple[Timeframe, ...] | None
    years: tuple[int, ...] | None
    overwrite: bool
    workers: int
    verbose: bool
    debug: bool


@dataclass(frozen=True, slots=True)
class DiscoveredWorkItem:
    """One discovered partition group ready for training dataset generation.

    Attributes:
        symbol: Tradeable symbol.
        timeframe: Available bar interval.
        years: Calendar years with existing feature and label parquet partitions.
    """

    symbol: Symbol
    timeframe: Timeframe
    years: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TrainingTaskResult:
    """Immutable result for one symbol/timeframe/year generation task.

    Attributes:
        symbol: Tradeable symbol.
        timeframe: Bar interval.
        year: Calendar year of the partition.
        status: ``succeeded``, ``failed``, or ``skipped``.
        rows_generated: Output row count when succeeded.
        error_type: Exception type name when failed.
        error_message: Exception message when failed.
        error_code: Optional CQROS error code when failed.
    """

    symbol: Symbol
    timeframe: Timeframe
    year: int
    status: str
    rows_generated: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class TrainingDatasetSummary:
    """Immutable aggregate summary for a training dataset generation run.

    Attributes:
        symbols_discovered: Unique symbols discovered from feature/label storage.
        symbols_processed: Unique symbols for which generation was attempted.
        timeframes_processed: Unique timeframes attempted.
        successful_tasks: Count of succeeded partition tasks.
        failed_tasks: Count of failed partition tasks.
        skipped_tasks: Count of skipped existing partitions.
        rows_generated: Sum of output rows across successes.
        duration_seconds: Wall-clock generation duration.
        output_directory: Training-tier output directory.
        failed_task_labels: Deterministic failed-task labels for reporting.
    """

    symbols_discovered: int
    symbols_processed: int
    timeframes_processed: int
    successful_tasks: int
    failed_tasks: int
    skipped_tasks: int
    rows_generated: int
    duration_seconds: float
    output_directory: Path
    failed_task_labels: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    """Create the training-dataset build argument parser.

    Returns:
        Configured ``ArgumentParser`` for training dataset build flags.
    """
    parser = argparse.ArgumentParser(
        prog="cqros-build-training-dataset",
        description=(
            "Build CQROS merged training datasets from discovered feature " "and label partitions."
        ),
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
        help="Regenerate training partitions that already exist.",
    )
    parser.add_argument(
        "--workers",
        dest="workers",
        type=int,
        default=_DEFAULT_WORKER_COUNT,
        metavar="INT",
        help=("Maximum concurrent symbols " f"(default: {_DEFAULT_WORKER_COUNT})."),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO logging.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=("Enable DEBUG logging and log complete failure tracebacks " "with logger.exception."),
    )
    return parser


def build_options(args: argparse.Namespace) -> TrainingDatasetOptions:
    """Map parsed CLI arguments onto ``TrainingDatasetOptions``.

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

    return TrainingDatasetOptions(
        storage_root=Path(DEFAULT_STORAGE_ROOT),
        symbols=_normalize_symbols(args.symbols),
        timeframes=_normalize_timeframes(args.timeframes),
        years=_normalize_years(args.years),
        overwrite=bool(args.overwrite),
        workers=workers,
        verbose=bool(args.verbose),
        debug=bool(args.debug),
    )


def build_training_pipeline(
    options: TrainingDatasetOptions,
    *,
    logger: logging.Logger | None = None,
) -> TrainingPipeline:
    """Compose ``TrainingPipeline`` from shared storage dependencies.

    Args:
        options: Immutable generation options providing the storage root.
        logger: Optional logger forwarded to the pipeline.

    Returns:
        Fully wired ``TrainingPipeline``.
    """
    layout = StorageLayout(options.storage_root)
    datastore = ParquetStore()
    training_repository = TrainingRepository(layout, datastore)
    return TrainingPipeline(
        training_repository,
        logger=logger if logger is not None else _logger,
    )


def discover_work(
    feature_repository: FeatureRepository,
    label_repository: LabelRepository,
    options: TrainingDatasetOptions,
) -> tuple[DiscoveredWorkItem, ...]:
    """Discover training-ready partitions present in both features and labels.

    Only ``(symbol, timeframe, year)`` tuples that exist in both repositories
    are included. Feature-only or label-only partitions are never scheduled.

    Args:
        feature_repository: Feature repository providing discovery APIs.
        label_repository: Label repository providing discovery APIs.
        options: CLI filters for symbol, timeframe, and year.

    Returns:
        Deterministically ordered discovered work items.
    """
    feature_partitions = feature_repository.discover_partitions(
        symbols=options.symbols,
        timeframes=options.timeframes,
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    label_partitions = label_repository.discover_partitions(
        symbols=options.symbols,
        timeframes=options.timeframes,
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    feature_keys = {
        (partition.symbol, partition.timeframe, partition.year) for partition in feature_partitions
    }
    label_keys = {
        (partition.symbol, partition.timeframe, partition.year) for partition in label_partitions
    }
    common_keys = feature_keys & label_keys
    return _group_intersection(common_keys, year_filter=options.years)


def format_summary(summary: TrainingDatasetSummary) -> str:
    """Render a deterministic training-dataset generation summary report.

    Args:
        summary: Aggregate generation summary.

    Returns:
        Multi-line summary text suitable for stdout.
    """
    lines = [
        "=====================================",
        "CQROS Training Dataset Summary",
        "=====================================",
        "",
        f"Symbols discovered: {summary.symbols_discovered}",
        f"Symbols processed: {summary.symbols_processed}",
        f"Timeframes processed: {summary.timeframes_processed}",
        f"Successful tasks: {summary.successful_tasks}",
        f"Failed tasks: {summary.failed_tasks}",
        f"Skipped tasks: {summary.skipped_tasks}",
        f"Rows generated: {summary.rows_generated}",
        f"Generation duration: {_format_duration(summary.duration_seconds)}",
        f"Output directory: {_format_output_directory(summary.output_directory)}",
    ]
    if summary.failed_task_labels:
        lines.extend(["", "Failed Tasks", ""])
        lines.extend(f"- {label}" for label in summary.failed_task_labels)
    return "\n".join(lines) + "\n"


async def main(argv: Sequence[str] | None = None) -> int:
    """Run the training-dataset build CLI.

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
        feature_repository = FeatureRepository(layout, datastore)
        label_repository = LabelRepository(layout, datastore)
        training_repository = TrainingRepository(layout, datastore)
        pipeline = build_training_pipeline(options)
        work = discover_work(feature_repository, label_repository, options)
        summary = await run_generation(
            pipeline=pipeline,
            feature_repository=feature_repository,
            label_repository=label_repository,
            training_repository=training_repository,
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
    pipeline: TrainingPipeline,
    feature_repository: FeatureRepository,
    label_repository: LabelRepository,
    training_repository: TrainingRepository,
    options: TrainingDatasetOptions,
    work: Sequence[DiscoveredWorkItem],
) -> TrainingDatasetSummary:
    """Execute discovered work through a bounded symbol worker pool.

    Args:
        pipeline: Injected training pipeline.
        feature_repository: Feature partition repository.
        label_repository: Label partition repository.
        training_repository: Training partition repository.
        options: Immutable generation options.
        work: Discovered work items.

    Returns:
        Aggregate immutable summary.
    """
    started = time.perf_counter()
    output_directory = options.storage_root / STORAGE_DIR_TRAINING

    if len(work) == 0:
        return TrainingDatasetSummary(
            symbols_discovered=0,
            symbols_processed=0,
            timeframes_processed=0,
            successful_tasks=0,
            failed_tasks=0,
            skipped_tasks=0,
            rows_generated=0,
            duration_seconds=time.perf_counter() - started,
            output_directory=output_directory,
            failed_task_labels=(),
        )

    work_by_symbol = _group_work_by_symbol(work)
    results = await _run_worker_pool(
        pipeline=pipeline,
        feature_repository=feature_repository,
        label_repository=label_repository,
        training_repository=training_repository,
        work_by_symbol=work_by_symbol,
        worker_count=options.workers,
        overwrite=options.overwrite,
        debug=options.debug,
    )
    return _build_summary(
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


def _group_intersection(
    common_keys: set[tuple[Symbol, Timeframe, int]],
    *,
    year_filter: tuple[int, ...] | None,
) -> tuple[DiscoveredWorkItem, ...]:
    """Group intersected partition keys into symbol/timeframe work items."""
    year_allowlist = set(year_filter) if year_filter is not None else None
    grouped: dict[tuple[str, str], list[int]] = {}
    for symbol, timeframe, year in common_keys:
        if year_allowlist is not None and year not in year_allowlist:
            continue
        key = (symbol, timeframe)
        grouped.setdefault(key, []).append(year)

    items: list[DiscoveredWorkItem] = []
    for (symbol, timeframe), years in grouped.items():
        items.append(
            DiscoveredWorkItem(
                symbol=symbol,
                timeframe=timeframe,
                years=tuple(sorted(years)),
            )
        )
    return tuple(
        sorted(
            items,
            key=lambda item: (item.symbol, item.timeframe),
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
    pipeline: TrainingPipeline,
    feature_repository: FeatureRepository,
    label_repository: LabelRepository,
    training_repository: TrainingRepository,
    work_by_symbol: Mapping[Symbol, Sequence[DiscoveredWorkItem]],
    worker_count: int,
    overwrite: bool,
    debug: bool,
) -> tuple[TrainingTaskResult, ...]:
    """Drain symbols through a bounded asyncio worker pool."""
    symbols = tuple(work_by_symbol.keys())
    if len(symbols) == 0:
        return ()

    queue: asyncio.Queue[Symbol | None] = asyncio.Queue()
    for symbol in symbols:
        queue.put_nowait(symbol)
    for _ in range(worker_count):
        queue.put_nowait(None)

    collected: list[TrainingTaskResult] = []
    lock = asyncio.Lock()

    async def worker() -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                results = await _generate_symbol_work(
                    pipeline=pipeline,
                    feature_repository=feature_repository,
                    label_repository=label_repository,
                    training_repository=training_repository,
                    symbol=item,
                    work_items=work_by_symbol[item],
                    overwrite=overwrite,
                    debug=debug,
                )
                async with lock:
                    collected.extend(results)
            finally:
                queue.task_done()

    worker_tasks = [
        asyncio.create_task(worker(), name=f"build-training-worker-{index}")
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
    pipeline: TrainingPipeline,
    feature_repository: FeatureRepository,
    label_repository: LabelRepository,
    training_repository: TrainingRepository,
    symbol: Symbol,
    work_items: Sequence[DiscoveredWorkItem],
    overwrite: bool,
    debug: bool,
) -> tuple[TrainingTaskResult, ...]:
    """Generate training datasets for every discovered year for one symbol."""
    results: list[TrainingTaskResult] = []
    for item in work_items:
        for year in item.years:
            result = await asyncio.to_thread(
                _generate_partition,
                pipeline,
                feature_repository,
                label_repository,
                training_repository,
                symbol=symbol,
                timeframe=item.timeframe,
                year=year,
                overwrite=overwrite,
                debug=debug,
            )
            _print_progress(result)
            results.append(result)
    return tuple(results)


def _generate_partition(
    pipeline: TrainingPipeline,
    feature_repository: FeatureRepository,
    label_repository: LabelRepository,
    training_repository: TrainingRepository,
    *,
    symbol: Symbol,
    timeframe: Timeframe,
    year: int,
    overwrite: bool,
    debug: bool,
) -> TrainingTaskResult:
    """Generate one training year partition synchronously."""
    if not overwrite and training_repository.exists(
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=symbol,
        timeframe=timeframe,
        year=year,
    ):
        return TrainingTaskResult(
            symbol=symbol,
            timeframe=timeframe,
            year=year,
            status="skipped",
        )

    try:
        features = feature_repository.load(
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        )
        labels = label_repository.load(
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        )
        output = pipeline.run(
            features,
            labels,
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
        return TrainingTaskResult(
            symbol=symbol,
            timeframe=timeframe,
            year=year,
            status="failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
            error_code=exc.error_code if isinstance(exc, CQROSError) else None,
        )

    return TrainingTaskResult(
        symbol=symbol,
        timeframe=timeframe,
        year=year,
        status="succeeded",
        rows_generated=output.height,
    )


def _print_progress(result: TrainingTaskResult) -> None:
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
            "Failed training dataset generation partition; continuing",
            extra=log_extra,
            exc_info=True,
        )
    else:
        _logger.warning(
            "Failed training dataset generation partition; continuing",
            extra=log_extra,
        )


def _build_summary(
    *,
    work: Sequence[DiscoveredWorkItem],
    results: Sequence[TrainingTaskResult],
    duration_seconds: float,
    output_directory: Path,
) -> TrainingDatasetSummary:
    """Aggregate task results into a generation report."""
    symbols_discovered = {item.symbol for item in work}
    symbols_processed: set[Symbol] = set()
    timeframes_processed: set[Timeframe] = set()
    successful_tasks = 0
    failed_tasks = 0
    skipped_tasks = 0
    rows_generated = 0
    failed_labels: set[str] = set()

    for result in results:
        symbols_processed.add(result.symbol)
        timeframes_processed.add(result.timeframe)
        if result.status == "succeeded":
            successful_tasks += 1
            if result.rows_generated is not None:
                rows_generated += result.rows_generated
        elif result.status == "skipped":
            skipped_tasks += 1
        else:
            failed_tasks += 1
            failed_labels.add(f"{result.symbol} {result.timeframe} {result.year}")

    return TrainingDatasetSummary(
        symbols_discovered=len(symbols_discovered),
        symbols_processed=len(symbols_processed),
        timeframes_processed=len(timeframes_processed),
        successful_tasks=successful_tasks,
        failed_tasks=failed_tasks,
        skipped_tasks=skipped_tasks,
        rows_generated=rows_generated,
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
