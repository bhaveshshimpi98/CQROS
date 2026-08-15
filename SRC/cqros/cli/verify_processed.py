"""CQROS processed-data verification CLI.

Purpose:
    Provide an argparse-based production entry point that discovers processed
    market partitions and executes ``VerificationRunner`` across the universe
    with bounded symbol concurrency.

Responsibilities:
    - Parse CLI arguments for processed-data verification
    - Discover available processed partitions through the repository
    - Create ``VerificationRunner`` and schedule verification tasks
    - Aggregate ``VerificationSummary`` results into a final report
    - Print structured per-partition failure diagnostics
    - Print the report and return an exit code

Dependencies:
    ``argparse``, ``asyncio``, ``cqros.config``, ``cqros.core``,
    ``cqros.processing.verification``, and ``cqros.storage``.

Public API:
    ``build_parser``, ``build_options``, ``build_verification_runner``,
    ``discover_work``, ``format_partition_failure``, ``format_summary``,
    ``run_verification``, and ``main``.

Notes:
    This module is a thin composition root. It does not implement verification
    logic, Polars transforms, or repository filesystem walks beyond calling
    repository discovery APIs.
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

from cqros.config.models import ResearchConfig
from cqros.core.constants import (
    DEFAULT_STORAGE_ROOT,
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
    SUPPORTED_TIMEFRAMES,
)
from cqros.core.exceptions import CQROSError, ValidationError
from cqros.core.types import Symbol, Timeframe
from cqros.processing.verification import (
    FundingVerifier,
    LongShortVerifier,
    OHLCVVerifier,
    OpenInterestVerifier,
    TakerVolumeVerifier,
    VerificationRunner,
    VerificationSummary,
    VerificationTaskResult,
)
from cqros.storage import (
    ParquetStore,
    ProcessedMarketDataRepository,
    ProcessedPartitionRef,
    StorageLayout,
)

__all__ = [
    "CLI_DATASETS",
    "DiscoveredWorkItem",
    "VerifyProcessedOptions",
    "VerifyProcessedSummary",
    "build_options",
    "build_parser",
    "build_verification_runner",
    "discover_work",
    "format_partition_failure",
    "format_summary",
    "main",
    "run_verification",
]

_logger = logging.getLogger(__name__)

_EXIT_SUCCESS: Final[int] = 0
_EXIT_FAILURE: Final[int] = 1

_DEFAULT_WORKER_COUNT: Final[int] = ResearchConfig().worker_count

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

_DATASET_DISPLAY_NAMES: Final[Mapping[str, str]] = {
    _CLI_DATASET_OHLCV: "OHLCV",
    _CLI_DATASET_FUNDING: "Funding",
    _CLI_DATASET_OPEN_INTEREST: "Open Interest",
    _CLI_DATASET_TAKER_VOLUME: "Taker Volume",
    _STORAGE_DATASET_GLOBAL_LS: "Global Long Short Account Ratio",
    _STORAGE_DATASET_TOP_LS_ACCOUNT: "Top Long Short Account Ratio",
    _STORAGE_DATASET_TOP_LS_POSITION: "Top Long Short Position Ratio",
}

_DATASET_VERIFIER_NAMES: Final[Mapping[str, str]] = {
    _CLI_DATASET_OHLCV: "OHLCVVerifier",
    _CLI_DATASET_FUNDING: "FundingVerifier",
    _CLI_DATASET_OPEN_INTEREST: "OpenInterestVerifier",
    _CLI_DATASET_TAKER_VOLUME: "TakerVolumeVerifier",
    _STORAGE_DATASET_GLOBAL_LS: "LongShortVerifier",
    _STORAGE_DATASET_TOP_LS_ACCOUNT: "LongShortVerifier",
    _STORAGE_DATASET_TOP_LS_POSITION: "LongShortVerifier",
}

_ERROR_WORKERS: Final[str] = "CLI-VERIFY-PROCESSED-001"
_ERROR_DATASET: Final[str] = "CLI-VERIFY-PROCESSED-002"
_ERROR_TIMEFRAME: Final[str] = "CLI-VERIFY-PROCESSED-003"

type _RunnerMethod = Callable[..., VerificationSummary]


@dataclass(frozen=True, slots=True)
class VerifyProcessedOptions:
    """Immutable CLI options for processed-data verification.

    Attributes:
        storage_root: Storage root containing ``processed``.
        symbols: Optional symbol allowlist. ``None`` discovers all.
        timeframes: Optional timeframe allowlist. ``None`` discovers all.
        datasets: Optional CLI dataset allowlist. ``None`` discovers all.
        workers: Maximum concurrent symbols.
        verbose: When ``True``, enable INFO logging.
        debug: When ``True``, enable DEBUG logging and log full failure
            tracebacks.
    """

    storage_root: Path
    symbols: tuple[Symbol, ...] | None
    timeframes: tuple[Timeframe, ...] | None
    datasets: tuple[str, ...] | None
    workers: int
    verbose: bool
    debug: bool


@dataclass(frozen=True, slots=True)
class DiscoveredWorkItem:
    """One discovered processed partition group ready for verification.

    Attributes:
        symbol: Processed symbol.
        storage_dataset: Storage / runner dataset name.
        cli_dataset: CLI dataset name that produced this work item.
        timeframe: Available bar interval.
        years: Calendar years with existing processed parquet partitions.
    """

    symbol: Symbol
    storage_dataset: str
    cli_dataset: str
    timeframe: Timeframe
    years: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class VerifyProcessedSummary:
    """Immutable aggregate summary for a processed verification run.

    Attributes:
        symbols_verified: Unique symbols for which verification was attempted.
        datasets_verified: Unique CLI datasets attempted.
        timeframes_verified: Unique timeframes attempted.
        successful_tasks: Count of succeeded partition tasks.
        failed_tasks: Count of failed partition tasks.
        rows_checked: Sum of verifier ``rows_checked`` across successes.
        duplicate_timestamps: Sum of duplicate-timestamp counters.
        null_rows: Sum of null-row counters.
        nan_rows: Sum of NaN-row counters.
        invalid_timestamps: Sum of invalid-timestamp counters.
        invalid_numeric_rows: Sum of invalid-numeric counters.
        warnings: Sum of warning counts across successes.
        duration_seconds: Wall-clock verification duration.
        repository_passed: Whether the repository status is PASS.
    """

    symbols_verified: int
    datasets_verified: int
    timeframes_verified: int
    successful_tasks: int
    failed_tasks: int
    rows_checked: int
    duplicate_timestamps: int
    null_rows: int
    nan_rows: int
    invalid_timestamps: int
    invalid_numeric_rows: int
    warnings: int
    duration_seconds: float
    repository_passed: bool


def build_parser() -> argparse.ArgumentParser:
    """Create the processed-verification argument parser.

    Returns:
        Configured ``ArgumentParser`` for verification flags.
    """
    parser = argparse.ArgumentParser(
        prog="cqros-verify-processed",
        description=(
            "Verify CQROS processed market data across the discovered " "processed universe."
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
            "Dataset to verify. Repeat for multiple datasets. "
            f"Allowed: {', '.join(CLI_DATASETS)}. Default: every dataset."
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


def build_options(args: argparse.Namespace) -> VerifyProcessedOptions:
    """Map parsed CLI arguments onto ``VerifyProcessedOptions``.

    Args:
        args: Namespace produced by ``build_parser().parse_args(...)``.

    Returns:
        Immutable verification options.

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
    symbols = _normalize_symbols(args.symbols)

    return VerifyProcessedOptions(
        storage_root=Path(DEFAULT_STORAGE_ROOT),
        symbols=symbols,
        timeframes=timeframes,
        datasets=datasets,
        workers=workers,
        verbose=bool(args.verbose),
        debug=bool(args.debug),
    )


def build_verification_runner(
    options: VerifyProcessedOptions,
    *,
    logger: logging.Logger | None = None,
) -> VerificationRunner:
    """Compose ``VerificationRunner`` from shared storage dependencies.

    Args:
        options: Immutable verification options providing the storage root.
        logger: Optional logger forwarded to the runner.

    Returns:
        Fully wired ``VerificationRunner``.
    """
    layout = StorageLayout(options.storage_root)
    datastore = ParquetStore()
    processed_repository = ProcessedMarketDataRepository(layout, datastore)
    return VerificationRunner(
        processed_repository,
        ohlcv_verifier=OHLCVVerifier(),
        funding_verifier=FundingVerifier(),
        open_interest_verifier=OpenInterestVerifier(),
        taker_volume_verifier=TakerVolumeVerifier(),
        long_short_verifier=LongShortVerifier(),
        logger=logger if logger is not None else _logger,
    )


def discover_work(
    repository: ProcessedMarketDataRepository,
    options: VerifyProcessedOptions,
) -> tuple[DiscoveredWorkItem, ...]:
    """Discover processed partitions matching the CLI filters.

    Args:
        repository: Processed repository providing discovery APIs.
        options: CLI filters for symbol, timeframe, and dataset.

    Returns:
        Deterministically ordered discovered work items.
    """
    cli_datasets = options.datasets if options.datasets is not None else CLI_DATASETS
    storage_datasets: list[str] = []
    for cli_dataset in cli_datasets:
        storage_datasets.extend(_CLI_TO_STORAGE_DATASETS[cli_dataset])

    partitions = repository.discover_partitions(
        datasets=tuple(storage_datasets),
        symbols=options.symbols,
        timeframes=options.timeframes,
    )
    return _group_partitions(partitions)


def format_partition_failure(
    *,
    dataset: str,
    symbol: Symbol,
    timeframe: Timeframe,
    partition: str,
    verifier: str,
    exception_type: str,
    message: str,
    code: str | None = None,
) -> str:
    """Render a structured per-partition verification failure report.

    Args:
        dataset: Human-readable dataset label.
        symbol: Failed symbol.
        timeframe: Failed timeframe.
        partition: Partition identifier (for example ``2025.parquet``).
        verifier: Verifier class name responsible for the partition.
        exception_type: Exception type name.
        message: Human-readable failure message.
        code: Optional CQROS / processing validation error code.

    Returns:
        Multi-line failure text suitable for stdout. Does not include a
        traceback.
    """
    lines = [
        "FAILED",
        "",
        f"Dataset: {dataset}",
        f"Symbol: {symbol}",
        f"Timeframe: {timeframe}",
        f"Partition: {partition}",
        f"Verifier: {verifier}",
        f"Exception: {exception_type}",
    ]
    if code is not None:
        lines.append(f"Code: {code}")
    lines.append(f"Message: {message}")
    return "\n".join(lines) + "\n"


def format_summary(summary: VerifyProcessedSummary) -> str:
    """Render a deterministic verification summary report.

    Args:
        summary: Aggregate verification summary.

    Returns:
        Multi-line summary text suitable for stdout.
    """
    status = "PASS" if summary.repository_passed else "FAIL"
    lines = [
        "=====================================",
        "CQROS Verification Summary",
        "=====================================",
        "",
        f"Symbols verified: {summary.symbols_verified}",
        f"Datasets verified: {summary.datasets_verified}",
        f"Timeframes verified: {summary.timeframes_verified}",
        f"Successful tasks: {summary.successful_tasks}",
        f"Failed tasks: {summary.failed_tasks}",
        f"Rows checked: {summary.rows_checked}",
        f"Duplicate timestamps: {summary.duplicate_timestamps}",
        f"NULL rows: {summary.null_rows}",
        f"NaN rows: {summary.nan_rows}",
        f"Invalid timestamps: {summary.invalid_timestamps}",
        f"Invalid numeric rows: {summary.invalid_numeric_rows}",
        f"Warnings: {summary.warnings}",
        f"Verification duration: {_format_duration(summary.duration_seconds)}",
        "",
        "Repository status:",
        "",
        status,
        "",
        "=====================================",
    ]
    return "\n".join(lines) + "\n"


async def main(argv: Sequence[str] | None = None) -> int:
    """Run the processed-verification CLI.

    Args:
        argv: Optional argument vector. When ``None``, ``sys.argv[1:]`` is used.

    Returns:
        ``0`` when the repository passed; ``1`` on failure or fatal CLI error.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        options = build_options(args)
        _configure_logging(verbose=options.verbose, debug=options.debug)
        layout = StorageLayout(options.storage_root)
        repository = ProcessedMarketDataRepository(layout, ParquetStore())
        work = discover_work(repository, options)
        runner = build_verification_runner(options)
        summary = await run_verification(runner=runner, options=options, work=work)
    except CQROSError as exc:
        print(str(exc), file=sys.stderr)
        return _EXIT_FAILURE
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return _EXIT_FAILURE

    print(format_summary(summary), end="")
    return _EXIT_SUCCESS if summary.repository_passed else _EXIT_FAILURE


async def run_verification(
    *,
    runner: VerificationRunner,
    options: VerifyProcessedOptions,
    work: Sequence[DiscoveredWorkItem],
) -> VerifyProcessedSummary:
    """Execute discovered work through a bounded symbol worker pool.

    Args:
        runner: Injected verification runner.
        options: Immutable verification options.
        work: Discovered work items.

    Returns:
        Aggregate immutable summary.
    """
    started = time.perf_counter()
    if len(work) == 0:
        return VerifyProcessedSummary(
            symbols_verified=0,
            datasets_verified=0,
            timeframes_verified=0,
            successful_tasks=0,
            failed_tasks=0,
            rows_checked=0,
            duplicate_timestamps=0,
            null_rows=0,
            nan_rows=0,
            invalid_timestamps=0,
            invalid_numeric_rows=0,
            warnings=0,
            duration_seconds=time.perf_counter() - started,
            repository_passed=True,
        )

    work_by_symbol = _group_work_by_symbol(work)
    summaries = await _run_worker_pool(
        runner=runner,
        work_by_symbol=work_by_symbol,
        worker_count=options.workers,
        debug=options.debug,
    )
    return _build_summary(
        summaries=summaries,
        duration_seconds=time.perf_counter() - started,
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


def _group_partitions(
    partitions: Sequence[ProcessedPartitionRef],
) -> tuple[DiscoveredWorkItem, ...]:
    """Group year partitions into symbol/dataset/timeframe work items."""
    grouped: dict[tuple[str, str, str, str], list[int]] = {}
    for partition in partitions:
        cli_dataset = _STORAGE_TO_CLI_DATASET.get(partition.dataset, partition.dataset)
        key = (partition.symbol, partition.dataset, cli_dataset, partition.timeframe)
        grouped.setdefault(key, []).append(partition.year)

    items: list[DiscoveredWorkItem] = []
    for (symbol, storage_dataset, cli_dataset, timeframe), years in grouped.items():
        items.append(
            DiscoveredWorkItem(
                symbol=symbol,
                storage_dataset=storage_dataset,
                cli_dataset=cli_dataset,
                timeframe=timeframe,
                years=tuple(sorted(years)),
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
    runner: VerificationRunner,
    work_by_symbol: Mapping[Symbol, Sequence[DiscoveredWorkItem]],
    worker_count: int,
    debug: bool,
) -> tuple[VerificationSummary, ...]:
    """Drain symbols through a bounded asyncio worker pool."""
    symbols = tuple(work_by_symbol.keys())
    if len(symbols) == 0:
        return ()

    queue: asyncio.Queue[Symbol | None] = asyncio.Queue()
    for symbol in symbols:
        queue.put_nowait(symbol)
    for _ in range(worker_count):
        queue.put_nowait(None)

    collected: list[VerificationSummary] = []
    lock = asyncio.Lock()

    async def worker() -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                summaries = await _verify_symbol_work(
                    runner=runner,
                    symbol=item,
                    work_items=work_by_symbol[item],
                    debug=debug,
                )
                async with lock:
                    collected.extend(summaries)
            finally:
                queue.task_done()

    worker_tasks = [
        asyncio.create_task(worker(), name=f"verify-processed-worker-{index}")
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


async def _verify_symbol_work(
    *,
    runner: VerificationRunner,
    symbol: Symbol,
    work_items: Sequence[DiscoveredWorkItem],
    debug: bool,
) -> tuple[VerificationSummary, ...]:
    """Verify every discovered work item for one symbol sequentially."""
    summaries: list[VerificationSummary] = []
    for item in work_items:
        try:
            summary = await asyncio.to_thread(
                _invoke_runner,
                runner,
                item,
            )
        except Exception as exc:
            year = item.years[0] if item.years else 0
            _report_exception_failure(
                item=item,
                year=year,
                exc=exc,
                debug=debug,
            )
            summaries.append(
                VerificationSummary(
                    dataset=item.storage_dataset,
                    exchange=_EXCHANGE,
                    market=_MARKET,
                    results=(
                        VerificationTaskResult(
                            symbol=symbol,
                            timeframe=item.timeframe,
                            year=year,
                            status="failed",
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                            error_code=(exc.error_code if isinstance(exc, CQROSError) else None),
                        ),
                    ),
                )
            )
            continue
        _report_summary_failures(summary=summary)
        summaries.append(summary)
    return tuple(summaries)


def _report_exception_failure(
    *,
    item: DiscoveredWorkItem,
    year: int,
    exc: BaseException,
    debug: bool,
) -> None:
    """Print structured failure diagnostics for a raised verification exception."""
    print(
        _format_exception_failure(item=item, year=year, exc=exc),
        end="",
        flush=True,
    )
    log_extra = {
        "symbol": item.symbol,
        "dataset": item.storage_dataset,
        "timeframe": item.timeframe,
        "year": year,
        "verifier": _verifier_name(item.storage_dataset),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "error_code": exc.error_code if isinstance(exc, CQROSError) else None,
    }
    if debug:
        _logger.exception(
            "Failed symbol dataset verification; continuing",
            extra=log_extra,
            exc_info=True,
        )
    else:
        _logger.warning(
            "Failed symbol dataset verification; continuing",
            extra=log_extra,
        )


def _report_summary_failures(*, summary: VerificationSummary) -> None:
    """Print structured diagnostics for failed partitions returned by the runner."""
    for result in summary.failed:
        print(
            _format_task_result_failure(dataset=summary.dataset, result=result),
            end="",
            flush=True,
        )


def _format_exception_failure(
    *,
    item: DiscoveredWorkItem,
    year: int,
    exc: BaseException,
) -> str:
    """Build a structured failure report from a raised exception."""
    code: str | None = None
    message = str(exc)
    if isinstance(exc, CQROSError):
        code = exc.error_code
        message = exc.message
    return format_partition_failure(
        dataset=_dataset_display_name(item.storage_dataset),
        symbol=item.symbol,
        timeframe=item.timeframe,
        partition=_partition_label(year),
        verifier=_verifier_name(item.storage_dataset),
        exception_type=type(exc).__name__,
        message=message,
        code=code,
    )


def _format_task_result_failure(
    *,
    dataset: str,
    result: VerificationTaskResult,
) -> str:
    """Build a structured failure report from a runner task result."""
    return format_partition_failure(
        dataset=_dataset_display_name(dataset),
        symbol=result.symbol,
        timeframe=result.timeframe,
        partition=_partition_label(result.year),
        verifier=_verifier_name(dataset),
        exception_type=result.error_type if result.error_type is not None else "Exception",
        message=result.error_message if result.error_message is not None else "",
        code=result.error_code,
    )


def _dataset_display_name(storage_dataset: str) -> str:
    """Return the human-readable dataset label for diagnostics."""
    return _DATASET_DISPLAY_NAMES.get(storage_dataset, storage_dataset)


def _verifier_name(storage_dataset: str) -> str:
    """Return the verifier class name associated with a storage dataset."""
    return _DATASET_VERIFIER_NAMES.get(storage_dataset, "DataVerifier")


def _partition_label(year: int) -> str:
    """Return the partition filename identifier for a calendar year."""
    return f"{year}.parquet"


def _invoke_runner(
    runner: VerificationRunner,
    item: DiscoveredWorkItem,
) -> VerificationSummary:
    """Invoke the appropriate ``VerificationRunner`` method for one work item."""
    method = _runner_method(runner, item.storage_dataset)
    return method(
        symbols=(item.symbol,),
        timeframes=(item.timeframe,),
        years=item.years,
        exchange=_EXCHANGE,
        market=_MARKET,
    )


def _runner_method(runner: VerificationRunner, storage_dataset: str) -> _RunnerMethod:
    """Resolve the runner method for a storage dataset name."""
    mapping: dict[str, _RunnerMethod] = {
        _CLI_DATASET_OHLCV: runner.verify_ohlcv,
        _CLI_DATASET_FUNDING: runner.verify_funding,
        _CLI_DATASET_OPEN_INTEREST: runner.verify_open_interest,
        _CLI_DATASET_TAKER_VOLUME: runner.verify_taker_volume,
        _STORAGE_DATASET_GLOBAL_LS: runner.verify_global_long_short_account_ratio,
        _STORAGE_DATASET_TOP_LS_ACCOUNT: runner.verify_top_long_short_account_ratio,
        _STORAGE_DATASET_TOP_LS_POSITION: runner.verify_top_long_short_position_ratio,
    }
    return mapping[storage_dataset]


def _build_summary(
    *,
    summaries: Sequence[VerificationSummary],
    duration_seconds: float,
) -> VerifyProcessedSummary:
    """Aggregate runner summaries into a verification report."""
    symbols_verified: set[Symbol] = set()
    datasets_verified: set[str] = set()
    timeframes_verified: set[Timeframe] = set()
    successful_tasks = 0
    failed_tasks = 0
    rows_checked = 0
    duplicate_timestamps = 0
    null_rows = 0
    nan_rows = 0
    invalid_timestamps = 0
    invalid_numeric_rows = 0
    warnings = 0

    for summary in summaries:
        cli_dataset = _STORAGE_TO_CLI_DATASET.get(summary.dataset, summary.dataset)
        for result in summary.results:
            symbols_verified.add(result.symbol)
            datasets_verified.add(cli_dataset)
            timeframes_verified.add(result.timeframe)
            if result.status == "succeeded":
                successful_tasks += 1
                report = result.report
                if report is not None:
                    rows_checked += report.rows_checked
                    duplicate_timestamps += report.duplicate_timestamp_rows
                    null_rows += report.null_rows
                    nan_rows += report.nan_rows
                    invalid_timestamps += report.invalid_timestamp_rows
                    invalid_numeric_rows += report.invalid_numeric_rows
                    warnings += len(report.warnings)
            else:
                failed_tasks += 1

    repository_passed = (
        failed_tasks == 0
        and duplicate_timestamps == 0
        and null_rows == 0
        and nan_rows == 0
        and invalid_timestamps == 0
        and invalid_numeric_rows == 0
        and warnings == 0
    )

    return VerifyProcessedSummary(
        symbols_verified=len(symbols_verified),
        datasets_verified=len(datasets_verified),
        timeframes_verified=len(timeframes_verified),
        successful_tasks=successful_tasks,
        failed_tasks=failed_tasks,
        rows_checked=rows_checked,
        duplicate_timestamps=duplicate_timestamps,
        null_rows=null_rows,
        nan_rows=nan_rows,
        invalid_timestamps=invalid_timestamps,
        invalid_numeric_rows=invalid_numeric_rows,
        warnings=warnings,
        duration_seconds=duration_seconds,
        repository_passed=repository_passed,
    )


def _format_duration(duration_seconds: float) -> str:
    """Format duration seconds for the summary report."""
    return f"{duration_seconds:.3f}s"


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
