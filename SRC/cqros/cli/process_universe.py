"""CQROS processing universe CLI.

Purpose:
    Provide an argparse-based production entry point that discovers downloaded
    raw market partitions and executes ``ProcessingRunner`` across the
    universe with bounded symbol concurrency.

Responsibilities:
    - Parse CLI arguments for universe processing
    - Build storage, pipeline, cleaner, and runner dependencies
    - Discover available raw symbols, datasets, timeframes, and years
    - Invoke ``ProcessingRunner`` per discovered work unit
    - Print a deterministic final summary

Dependencies:
    ``argparse``, ``asyncio``, ``cqros.core``, ``cqros.processing``, and
    ``cqros.storage``.

Public API:
    ``build_parser``, ``build_options``, ``build_processing_runner``,
    ``discover_work``, ``format_summary``, ``run_universe``, and ``main``.

Notes:
    This module is a thin composition root. It does not implement validation,
    cleaning, repository I/O, or processing logic beyond discovery and
    orchestration.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from cqros.core.constants import (
    DEFAULT_STORAGE_ROOT,
    EXCHANGE_BINANCE,
    FILE_EXTENSION_PARQUET,
    MARKET_USDT_PERPETUAL,
    STORAGE_DIR_PROCESSED,
    STORAGE_DIR_RAW,
    SUPPORTED_TIMEFRAMES,
)
from cqros.core.exceptions import CQROSError, ValidationError
from cqros.core.types import Symbol, Timeframe
from cqros.processing import (
    FundingCleaner,
    LongShortCleaner,
    OHLCVCleaner,
    OpenInterestCleaner,
    ProcessingPipeline,
    ProcessingRunner,
    ProcessingSummary,
    ProcessingTaskResult,
    TakerVolumeCleaner,
)
from cqros.storage import (
    MarketDataRepository,
    ParquetStore,
    ProcessedMarketDataRepository,
    StorageLayout,
)

__all__ = [
    "CLI_DATASETS",
    "DEFAULT_PROCESS_UNIVERSE_WORKERS",
    "DiscoveredWorkItem",
    "ProcessUniverseOptions",
    "ProcessUniverseSummary",
    "build_options",
    "build_parser",
    "build_processing_runner",
    "discover_work",
    "format_summary",
    "main",
    "run_universe",
]

_logger = logging.getLogger(__name__)

_EXIT_SUCCESS: Final[int] = 0
_EXIT_FAILURE: Final[int] = 1

DEFAULT_PROCESS_UNIVERSE_WORKERS: Final[int] = 4

_EXCHANGE: Final[str] = EXCHANGE_BINANCE
_MARKET: Final[str] = MARKET_USDT_PERPETUAL

_CLI_DATASET_OHLCV: Final[str] = "ohlcv"
_CLI_DATASET_FUNDING: Final[str] = "funding"
_CLI_DATASET_OPEN_INTEREST: Final[str] = "open_interest"
_CLI_DATASET_TAKER_VOLUME: Final[str] = "taker_volume"
_CLI_DATASET_LONG_SHORT: Final[str] = "long_short"

CLI_DATASETS: Final[tuple[str, ...]] = (
    _CLI_DATASET_OHLCV,
    _CLI_DATASET_FUNDING,
    _CLI_DATASET_OPEN_INTEREST,
    _CLI_DATASET_TAKER_VOLUME,
    _CLI_DATASET_LONG_SHORT,
)

_STORAGE_DATASET_GLOBAL_LS: Final[str] = "global_long_short_account_ratio"
_STORAGE_DATASET_TOP_LS_ACCOUNT: Final[str] = "top_long_short_account_ratio"
_STORAGE_DATASET_TOP_LS_POSITION: Final[str] = "top_long_short_position_ratio"

_CLI_TO_STORAGE_DATASETS: Final[Mapping[str, tuple[str, ...]]] = {
    _CLI_DATASET_OHLCV: (_CLI_DATASET_OHLCV,),
    _CLI_DATASET_FUNDING: (_CLI_DATASET_FUNDING,),
    _CLI_DATASET_OPEN_INTEREST: (_CLI_DATASET_OPEN_INTEREST,),
    _CLI_DATASET_TAKER_VOLUME: (_CLI_DATASET_TAKER_VOLUME,),
    _CLI_DATASET_LONG_SHORT: (
        _STORAGE_DATASET_GLOBAL_LS,
        _STORAGE_DATASET_TOP_LS_ACCOUNT,
        _STORAGE_DATASET_TOP_LS_POSITION,
    ),
}

_STORAGE_TO_CLI_DATASET: Final[Mapping[str, str]] = {
    _CLI_DATASET_OHLCV: _CLI_DATASET_OHLCV,
    _CLI_DATASET_FUNDING: _CLI_DATASET_FUNDING,
    _CLI_DATASET_OPEN_INTEREST: _CLI_DATASET_OPEN_INTEREST,
    _CLI_DATASET_TAKER_VOLUME: _CLI_DATASET_TAKER_VOLUME,
    _STORAGE_DATASET_GLOBAL_LS: _CLI_DATASET_LONG_SHORT,
    _STORAGE_DATASET_TOP_LS_ACCOUNT: _CLI_DATASET_LONG_SHORT,
    _STORAGE_DATASET_TOP_LS_POSITION: _CLI_DATASET_LONG_SHORT,
}

_ERROR_WORKERS: Final[str] = "CLI-PROCESS-UNIVERSE-001"
_ERROR_DATASET: Final[str] = "CLI-PROCESS-UNIVERSE-002"
_ERROR_TIMEFRAME: Final[str] = "CLI-PROCESS-UNIVERSE-003"

type _RunnerMethod = Callable[..., ProcessingSummary]


@dataclass(frozen=True, slots=True)
class ProcessUniverseOptions:
    """Immutable CLI options for universe processing.

    Attributes:
        storage_root: Storage root containing ``raw`` and ``processed``.
        symbol: Optional single-symbol filter. ``None`` discovers all.
        timeframes: Optional timeframe allowlist. ``None`` uses every
            available timeframe discovered on disk.
        datasets: Optional CLI dataset allowlist. ``None`` uses every
            available dataset discovered on disk.
        workers: Maximum concurrent symbols.
        verbose: When ``True``, enable INFO logging.
        dry_run: When ``True``, discover work without invoking the runner.
    """

    storage_root: Path
    symbol: Symbol | None
    timeframes: tuple[Timeframe, ...] | None
    datasets: tuple[str, ...] | None
    workers: int
    verbose: bool
    dry_run: bool


@dataclass(frozen=True, slots=True)
class DiscoveredWorkItem:
    """One discovered raw partition group ready for ``ProcessingRunner``.

    Attributes:
        symbol: Downloaded symbol.
        storage_dataset: Storage / runner dataset name.
        cli_dataset: CLI dataset name that produced this work item.
        timeframe: Available bar interval.
        years: Calendar years with existing raw parquet partitions.
    """

    symbol: Symbol
    storage_dataset: str
    cli_dataset: str
    timeframe: Timeframe
    years: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ProcessUniverseSummary:
    """Immutable aggregate summary for a universe processing run.

    Attributes:
        symbols_discovered: Unique symbols discovered from raw storage.
        symbols_processed: Unique symbols for which processing was attempted.
        datasets_processed: Unique CLI datasets attempted.
        timeframes_processed: Unique timeframes attempted.
        successful_tasks: Count of succeeded partition tasks.
        failed_tasks: Count of failed partition tasks.
        rows_processed: Sum of cleaned ``rows_before`` across successes.
        rows_removed: Sum of ``rows_before - rows_after`` across successes.
        duration_seconds: Wall-clock processing duration.
        output_directory: Processed-data output directory.
        failed_task_labels: Deterministic failed-task labels for reporting.
        dry_run: Whether the run was discovery-only.
    """

    symbols_discovered: int
    symbols_processed: int
    datasets_processed: int
    timeframes_processed: int
    successful_tasks: int
    failed_tasks: int
    rows_processed: int
    rows_removed: int
    duration_seconds: float
    output_directory: Path
    failed_task_labels: tuple[str, ...]
    dry_run: bool


def build_parser() -> argparse.ArgumentParser:
    """Create the processing-universe argument parser.

    Returns:
        Configured ``ArgumentParser`` for universe processing flags.
    """
    parser = argparse.ArgumentParser(
        prog="cqros-process-universe",
        description=(
            "Process downloaded CQROS raw market data into research-ready "
            "processed market data across the discovered universe."
        ),
    )
    parser.add_argument(
        "--symbol",
        dest="symbol",
        default=None,
        metavar="SYMBOL",
        help="Process only one symbol (default: every discovered symbol).",
    )
    parser.add_argument(
        "--timeframe",
        dest="timeframes",
        action="append",
        default=None,
        metavar="TIMEFRAME",
        help=(
            "Timeframe to process. Repeat for multiple intervals. "
            "Default: every available timeframe."
        ),
    )
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        default=None,
        choices=CLI_DATASETS,
        metavar="DATASET",
        help=(
            "Dataset to process. Repeat for multiple datasets. "
            f"Allowed: {', '.join(CLI_DATASETS)}. Default: every dataset."
        ),
    )
    parser.add_argument(
        "--workers",
        dest="workers",
        type=int,
        default=DEFAULT_PROCESS_UNIVERSE_WORKERS,
        metavar="INT",
        help=("Maximum concurrent symbols " f"(default: {DEFAULT_PROCESS_UNIVERSE_WORKERS})."),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO logging.",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Discover work only; do not process.",
    )
    return parser


def build_options(args: argparse.Namespace) -> ProcessUniverseOptions:
    """Map parsed CLI arguments onto ``ProcessUniverseOptions``.

    Args:
        args: Namespace produced by ``build_parser().parse_args(...)``.

    Returns:
        Immutable processing options.

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

    timeframes = _normalize_timeframes(args.timeframes)
    datasets = _normalize_datasets(args.datasets)
    symbol = args.symbol.strip() if isinstance(args.symbol, str) else None
    if symbol is not None and symbol == "":
        symbol = None

    return ProcessUniverseOptions(
        storage_root=Path(DEFAULT_STORAGE_ROOT),
        symbol=symbol,
        timeframes=timeframes,
        datasets=datasets,
        workers=workers,
        verbose=bool(args.verbose),
        dry_run=bool(args.dry_run),
    )


def build_processing_runner(
    options: ProcessUniverseOptions,
    *,
    logger: logging.Logger | None = None,
) -> ProcessingRunner:
    """Compose ``ProcessingRunner`` from shared storage and cleaner dependencies.

    Args:
        options: Immutable processing options providing the storage root.
        logger: Optional logger forwarded to the runner.

    Returns:
        Fully wired ``ProcessingRunner``.
    """
    layout = StorageLayout(options.storage_root)
    datastore = ParquetStore()
    raw_repository = MarketDataRepository(layout, datastore)
    processed_repository = ProcessedMarketDataRepository(layout, datastore)
    pipeline = ProcessingPipeline(())
    return ProcessingRunner(
        raw_repository,
        processed_repository,
        pipeline,
        ohlcv_cleaner=OHLCVCleaner(),
        funding_cleaner=FundingCleaner(),
        open_interest_cleaner=OpenInterestCleaner(),
        taker_volume_cleaner=TakerVolumeCleaner(),
        long_short_cleaner=LongShortCleaner(),
        logger=logger if logger is not None else _logger,
    )


def discover_work(
    layout: StorageLayout,
    options: ProcessUniverseOptions,
) -> tuple[DiscoveredWorkItem, ...]:
    """Discover raw partitions matching the CLI filters.

    Missing dataset trees are skipped. Only year partitions that exist as
    ``{year}.parquet`` files are included.

    Args:
        layout: Storage layout used to resolve the raw data root.
        options: CLI filters for symbol, timeframe, and dataset.

    Returns:
        Deterministically ordered discovered work items.
    """
    cli_datasets = options.datasets if options.datasets is not None else CLI_DATASETS
    items: list[DiscoveredWorkItem] = []

    for cli_dataset in cli_datasets:
        for storage_dataset in _CLI_TO_STORAGE_DATASETS[cli_dataset]:
            items.extend(
                _discover_dataset_work(
                    layout=layout,
                    cli_dataset=cli_dataset,
                    storage_dataset=storage_dataset,
                    symbol_filter=options.symbol,
                    timeframe_filter=options.timeframes,
                )
            )

    return tuple(
        sorted(
            items,
            key=lambda item: (
                item.symbol,
                item.cli_dataset,
                item.storage_dataset,
                item.timeframe,
            ),
        )
    )


def format_summary(summary: ProcessUniverseSummary) -> str:
    """Render a deterministic processing summary report.

    Args:
        summary: Aggregate universe processing summary.

    Returns:
        Multi-line summary text suitable for stdout.
    """
    lines = [
        "=====================================",
        "CQROS Processing Summary",
        "=====================================",
        "",
        f"Symbols discovered: {summary.symbols_discovered}",
        f"Symbols processed: {summary.symbols_processed}",
        f"Datasets processed: {summary.datasets_processed}",
        f"Timeframes processed: {summary.timeframes_processed}",
        f"Successful tasks: {summary.successful_tasks}",
        f"Failed tasks: {summary.failed_tasks}",
        f"Rows processed: {summary.rows_processed}",
        f"Rows removed: {summary.rows_removed}",
        f"Processing duration: {_format_duration(summary.duration_seconds)}",
        f"Output directory: {_format_output_directory(summary.output_directory)}",
    ]
    if summary.failed_task_labels:
        lines.extend(["", "Failed Tasks", ""])
        lines.extend(f"- {label}" for label in summary.failed_task_labels)
    return "\n".join(lines) + "\n"


async def main(argv: Sequence[str] | None = None) -> int:
    """Run the processing-universe CLI.

    Args:
        argv: Optional argument vector. When ``None``, ``sys.argv[1:]`` is used.

    Returns:
        ``0`` on completion; ``1`` when a fatal CLI error occurs.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        options = build_options(args)
        _configure_logging(verbose=options.verbose)
        layout = StorageLayout(options.storage_root)
        work = discover_work(layout, options)
        runner = build_processing_runner(options)
        summary = await run_universe(runner=runner, options=options, work=work)
    except CQROSError as exc:
        print(str(exc), file=sys.stderr)
        return _EXIT_FAILURE
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return _EXIT_FAILURE

    print(format_summary(summary), end="")
    return _EXIT_SUCCESS


async def run_universe(
    *,
    runner: ProcessingRunner,
    options: ProcessUniverseOptions,
    work: Sequence[DiscoveredWorkItem],
) -> ProcessUniverseSummary:
    """Execute discovered work through a bounded symbol worker pool.

    Args:
        runner: Injected processing runner.
        options: Immutable processing options.
        work: Discovered work items.

    Returns:
        Aggregate immutable summary.
    """
    started = time.perf_counter()
    symbols_discovered = tuple(sorted({item.symbol for item in work}))
    output_directory = options.storage_root / STORAGE_DIR_PROCESSED

    if options.dry_run or len(work) == 0:
        return ProcessUniverseSummary(
            symbols_discovered=len(symbols_discovered),
            symbols_processed=0,
            datasets_processed=0,
            timeframes_processed=0,
            successful_tasks=0,
            failed_tasks=0,
            rows_processed=0,
            rows_removed=0,
            duration_seconds=time.perf_counter() - started,
            output_directory=output_directory,
            failed_task_labels=(),
            dry_run=options.dry_run,
        )

    work_by_symbol = _group_work_by_symbol(work)
    summaries = await _run_worker_pool(
        runner=runner,
        work_by_symbol=work_by_symbol,
        worker_count=options.workers,
    )
    return _build_summary(
        work=work,
        summaries=summaries,
        duration_seconds=time.perf_counter() - started,
        output_directory=output_directory,
        dry_run=False,
    )


def _configure_logging(*, verbose: bool) -> None:
    """Configure process logging for the CLI entry point."""
    level = logging.INFO if verbose else logging.WARNING
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
    return tuple(normalized)


def _normalize_datasets(values: Sequence[str] | None) -> tuple[str, ...] | None:
    """Validate and freeze optional dataset filters."""
    if values is None:
        return None
    normalized: list[str] = []
    for dataset in values:
        if dataset not in _CLI_TO_STORAGE_DATASETS:
            raise ValidationError(
                f"unsupported dataset: {dataset}",
                error_code=_ERROR_DATASET,
                details={"parameter": "datasets", "value": dataset},
            )
        if dataset not in normalized:
            normalized.append(dataset)
    return tuple(normalized)


def _discover_dataset_work(
    *,
    layout: StorageLayout,
    cli_dataset: str,
    storage_dataset: str,
    symbol_filter: Symbol | None,
    timeframe_filter: tuple[Timeframe, ...] | None,
) -> list[DiscoveredWorkItem]:
    """Discover work items for one storage dataset tree."""
    base = layout.root / STORAGE_DIR_RAW / storage_dataset / _EXCHANGE / _MARKET
    if not base.is_dir():
        _logger.info(
            "Skipping missing raw dataset tree",
            extra={"dataset": storage_dataset, "path": str(base)},
        )
        return []

    items: list[DiscoveredWorkItem] = []
    for symbol_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        symbol = symbol_dir.name
        if symbol_filter is not None and symbol != symbol_filter:
            continue
        for timeframe_dir in sorted(path for path in symbol_dir.iterdir() if path.is_dir()):
            timeframe = timeframe_dir.name
            if timeframe_filter is not None and timeframe not in timeframe_filter:
                continue
            years = _discover_years(timeframe_dir)
            if not years:
                continue
            items.append(
                DiscoveredWorkItem(
                    symbol=symbol,
                    storage_dataset=storage_dataset,
                    cli_dataset=cli_dataset,
                    timeframe=timeframe,
                    years=years,
                )
            )
    return items


def _discover_years(timeframe_dir: Path) -> tuple[int, ...]:
    """Return sorted calendar years present as parquet partitions."""
    years: list[int] = []
    for path in sorted(timeframe_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix != FILE_EXTENSION_PARQUET:
            continue
        stem = path.stem
        if not stem.isdigit():
            continue
        year = int(stem)
        if year >= 1:
            years.append(year)
    return tuple(years)


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
    runner: ProcessingRunner,
    work_by_symbol: Mapping[Symbol, Sequence[DiscoveredWorkItem]],
    worker_count: int,
) -> tuple[ProcessingSummary, ...]:
    """Drain symbols through a bounded asyncio worker pool."""
    symbols = tuple(work_by_symbol.keys())
    if len(symbols) == 0:
        return ()

    queue: asyncio.Queue[Symbol | None] = asyncio.Queue()
    for symbol in symbols:
        queue.put_nowait(symbol)
    for _ in range(worker_count):
        queue.put_nowait(None)

    collected: list[ProcessingSummary] = []
    lock = asyncio.Lock()

    async def worker() -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                summaries = await _process_symbol_work(
                    runner=runner,
                    symbol=item,
                    work_items=work_by_symbol[item],
                )
                async with lock:
                    collected.extend(summaries)
            finally:
                queue.task_done()

    worker_tasks = [
        asyncio.create_task(worker(), name=f"process-universe-worker-{index}")
        for index in range(worker_count)
    ]
    try:
        await asyncio.gather(*worker_tasks)
    finally:
        for task in worker_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*worker_tasks, return_exceptions=True)

    return tuple(collected)


async def _process_symbol_work(
    *,
    runner: ProcessingRunner,
    symbol: Symbol,
    work_items: Sequence[DiscoveredWorkItem],
) -> tuple[ProcessingSummary, ...]:
    """Process every discovered work item for one symbol sequentially."""
    summaries: list[ProcessingSummary] = []
    for item in work_items:
        try:
            summary = await asyncio.to_thread(
                _invoke_runner,
                runner,
                item,
            )
        except Exception as exc:
            _logger.warning(
                "Failed symbol dataset processing; continuing",
                extra={
                    "symbol": symbol,
                    "dataset": item.storage_dataset,
                    "timeframe": item.timeframe,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            summaries.append(
                ProcessingSummary(
                    dataset=item.storage_dataset,
                    exchange=_EXCHANGE,
                    market=_MARKET,
                    results=(
                        ProcessingTaskResult(
                            symbol=symbol,
                            timeframe=item.timeframe,
                            year=item.years[0] if item.years else 0,
                            status="failed",
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                        ),
                    ),
                )
            )
            continue
        summaries.append(summary)
    return tuple(summaries)


def _invoke_runner(
    runner: ProcessingRunner,
    item: DiscoveredWorkItem,
) -> ProcessingSummary:
    """Invoke the appropriate ``ProcessingRunner`` method for one work item."""
    method = _runner_method(runner, item.storage_dataset)
    return method(
        symbols=(item.symbol,),
        timeframes=(item.timeframe,),
        years=item.years,
        exchange=_EXCHANGE,
        market=_MARKET,
    )


def _runner_method(runner: ProcessingRunner, storage_dataset: str) -> _RunnerMethod:
    """Resolve the runner method for a storage dataset name."""
    mapping: dict[str, _RunnerMethod] = {
        _CLI_DATASET_OHLCV: runner.process_ohlcv,
        _CLI_DATASET_FUNDING: runner.process_funding,
        _CLI_DATASET_OPEN_INTEREST: runner.process_open_interest,
        _CLI_DATASET_TAKER_VOLUME: runner.process_taker_volume,
        _STORAGE_DATASET_GLOBAL_LS: runner.process_global_long_short_account_ratio,
        _STORAGE_DATASET_TOP_LS_ACCOUNT: runner.process_top_long_short_account_ratio,
        _STORAGE_DATASET_TOP_LS_POSITION: runner.process_top_long_short_position_ratio,
    }
    return mapping[storage_dataset]


def _build_summary(
    *,
    work: Sequence[DiscoveredWorkItem],
    summaries: Sequence[ProcessingSummary],
    duration_seconds: float,
    output_directory: Path,
    dry_run: bool,
) -> ProcessUniverseSummary:
    """Aggregate runner summaries into a universe report."""
    symbols_discovered = {item.symbol for item in work}
    symbols_processed: set[Symbol] = set()
    datasets_processed: set[str] = set()
    timeframes_processed: set[Timeframe] = set()
    successful_tasks = 0
    failed_tasks = 0
    rows_processed = 0
    rows_removed = 0
    failed_labels: set[str] = set()

    for summary in summaries:
        cli_dataset = _STORAGE_TO_CLI_DATASET.get(summary.dataset, summary.dataset)
        for result in summary.results:
            symbols_processed.add(result.symbol)
            datasets_processed.add(cli_dataset)
            timeframes_processed.add(result.timeframe)
            if result.status == "succeeded":
                successful_tasks += 1
                if result.cleaning_report is not None:
                    rows_processed += result.cleaning_report.rows_before
                    rows_removed += (
                        result.cleaning_report.rows_before - result.cleaning_report.rows_after
                    )
                elif result.rows_loaded is not None:
                    rows_processed += result.rows_loaded
            else:
                failed_tasks += 1
                failed_labels.add(f"{result.symbol} {result.timeframe} {cli_dataset}")

    return ProcessUniverseSummary(
        symbols_discovered=len(symbols_discovered),
        symbols_processed=len(symbols_processed),
        datasets_processed=len(datasets_processed),
        timeframes_processed=len(timeframes_processed),
        successful_tasks=successful_tasks,
        failed_tasks=failed_tasks,
        rows_processed=rows_processed,
        rows_removed=rows_removed,
        duration_seconds=duration_seconds,
        output_directory=output_directory,
        failed_task_labels=tuple(sorted(failed_labels)),
        dry_run=dry_run,
    )


def _format_duration(duration_seconds: float) -> str:
    """Format duration seconds for the summary report."""
    return f"{duration_seconds:.3f}s"


def _format_output_directory(path: Path) -> str:
    """Format the output directory using POSIX separators."""
    return path.as_posix()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
