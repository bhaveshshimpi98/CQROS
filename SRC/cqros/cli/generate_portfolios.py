"""CQROS portfolio-generation CLI.

Purpose:
    Provide an argparse-based production entry point that discovers signal
    partitions and executes ``PortfolioPipeline`` across the universe with
    bounded symbol concurrency, persisting outputs through
    ``PortfolioRepository``.

Responsibilities:
    - Parse CLI arguments for portfolio dataset generation
    - Discover available signal partitions
    - Filter loaded signal frames by ``--model`` / ``--version``
    - Resolve ``--optimizer`` through ``PortfolioOptimizerRegistry``
    - Execute ``PortfolioPipeline`` and persist via ``PortfolioRepository``
    - Honor ``--overwrite``, worker concurrency, and debug logging
    - Print deterministic progress and a final summary

Dependencies:
    ``argparse``, ``asyncio``, ``polars``, ``cqros.config``, ``cqros.core``,
    ``cqros.portfolio``, and ``cqros.storage``.

Public API:
    ``build_parser``, ``build_options``, ``build_default_optimizer``,
    ``build_optimizer_registry``, ``build_portfolio_pipeline``,
    ``discover_work``, ``format_summary``, ``run_generation``, and ``main``.

Notes:
    This module is a thin composition root. It does not implement allocation
    semantics, schema validation, or repository filesystem walks beyond
    calling repository discovery and load/save APIs. Portfolio creation is
    delegated exclusively to ``PortfolioPipeline``. Persistence remains in
    the CLI because ``PortfolioPipeline`` does not own a repository.
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
    STORAGE_DIR_PORTFOLIOS,
    SUPPORTED_TIMEFRAMES,
)
from cqros.core.exceptions import CQROSError, ValidationError
from cqros.core.types import Symbol, Timeframe
from cqros.portfolio import (
    EqualWeightOptimizer,
    OptimizerStrategy,
    PortfolioOptimizer,
    PortfolioOptimizerRegistry,
    PortfolioPipeline,
)
from cqros.storage import (
    ParquetStore,
    PortfolioRepository,
    SignalPartitionRef,
    SignalRepository,
    StorageLayout,
)

__all__ = [
    "DiscoveredWorkItem",
    "PortfolioGenerationOptions",
    "PortfolioGenerationSummary",
    "PortfolioTaskResult",
    "build_default_optimizer",
    "build_optimizer_registry",
    "build_options",
    "build_parser",
    "build_portfolio_pipeline",
    "discover_work",
    "format_summary",
    "main",
    "run_generation",
]

_logger = logging.getLogger(__name__)

_EXIT_SUCCESS: Final[int] = 0
_EXIT_FAILURE: Final[int] = 1

_DEFAULT_WORKER_COUNT: Final[int] = ResearchConfig().worker_count
_DEFAULT_OPTIMIZER: Final[str] = OptimizerStrategy.EQUAL_WEIGHT.value

_EXCHANGE: Final[str] = EXCHANGE_BINANCE
_MARKET: Final[str] = MARKET_USDT_PERPETUAL

_ERROR_WORKERS: Final[str] = "CLI-GENERATE-PORTFOLIOS-001"
_ERROR_TIMEFRAME: Final[str] = "CLI-GENERATE-PORTFOLIOS-002"
_ERROR_YEAR: Final[str] = "CLI-GENERATE-PORTFOLIOS-003"
_ERROR_MODEL: Final[str] = "CLI-GENERATE-PORTFOLIOS-004"
_ERROR_VERSION: Final[str] = "CLI-GENERATE-PORTFOLIOS-005"
_ERROR_OPTIMIZER: Final[str] = "CLI-GENERATE-PORTFOLIOS-006"


@dataclass(frozen=True, slots=True)
class PortfolioGenerationOptions:
    """Immutable CLI options for portfolio dataset generation.

    Attributes:
        storage_root: Storage root containing ``signals`` and ``portfolios``.
        optimizer: Registry key of the portfolio optimizer to execute.
        model: Stable model identifier used to filter signal rows.
        version: Model version identifier used to filter signal rows.
        symbols: Optional symbol allowlist. ``None`` discovers all.
        timeframes: Optional timeframe allowlist. ``None`` discovers all.
        years: Optional year allowlist. ``None`` discovers all.
        overwrite: When ``True``, regenerate existing portfolio partitions.
        workers: Maximum concurrent symbols.
        verbose: When ``True``, enable INFO logging.
        debug: When ``True``, enable DEBUG logging and log full failure
            tracebacks.
    """

    storage_root: Path
    optimizer: str
    model: str
    version: str
    symbols: tuple[Symbol, ...] | None
    timeframes: tuple[Timeframe, ...] | None
    years: tuple[int, ...] | None
    overwrite: bool
    workers: int
    verbose: bool
    debug: bool


@dataclass(frozen=True, slots=True)
class DiscoveredWorkItem:
    """One discovered signal partition group ready for portfolio generation.

    Attributes:
        symbol: Tradeable symbol.
        timeframe: Available bar interval.
        years: Calendar years with existing signal parquet partitions.
    """

    symbol: Symbol
    timeframe: Timeframe
    years: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PortfolioTaskResult:
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
class PortfolioGenerationSummary:
    """Immutable aggregate summary for a portfolio generation run.

    Attributes:
        optimizer: Optimizer registry key used for generation.
        model: Stable model identifier used for generation.
        version: Model version used for generation.
        symbols_discovered: Unique symbols discovered from signal storage.
        symbols_processed: Unique symbols for which generation was attempted.
        timeframes_processed: Unique timeframes attempted.
        successful_tasks: Count of succeeded partition tasks.
        failed_tasks: Count of failed partition tasks.
        skipped_tasks: Count of skipped existing partitions.
        rows_generated: Sum of output rows across successes.
        duration_seconds: Wall-clock generation duration.
        output_directory: Portfolios-tier output directory.
        failed_task_labels: Deterministic failed-task labels for reporting.
    """

    optimizer: str
    model: str
    version: str
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
    """Create the portfolio-generation argument parser.

    Returns:
        Configured ``ArgumentParser`` for portfolio generation flags.
    """
    parser = argparse.ArgumentParser(
        prog="cqros-generate-portfolios",
        description=(
            "Generate CQROS portfolio datasets from discovered signal "
            "partitions and an injected portfolio optimizer."
        ),
    )
    parser.add_argument(
        "--optimizer",
        dest="optimizer",
        default=_DEFAULT_OPTIMIZER,
        metavar="NAME",
        help=("Portfolio optimizer registry key " f"(default: {_DEFAULT_OPTIMIZER})."),
    )
    parser.add_argument(
        "--model",
        dest="model",
        required=True,
        metavar="NAME",
        help="Stable model identifier used to filter signal rows.",
    )
    parser.add_argument(
        "--version",
        dest="version",
        required=True,
        metavar="VERSION",
        help="Model version identifier used to filter signal rows.",
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
        help="Regenerate portfolio partitions that already exist.",
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


def build_options(args: argparse.Namespace) -> PortfolioGenerationOptions:
    """Map parsed CLI arguments onto ``PortfolioGenerationOptions``.

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

    optimizer = str(args.optimizer).strip()
    if optimizer == "":
        raise ValidationError(
            "optimizer must be a non-empty string",
            error_code=_ERROR_OPTIMIZER,
            details={"parameter": "optimizer", "value": args.optimizer},
        )
    model = str(args.model).strip()
    if model == "":
        raise ValidationError(
            "model must be a non-empty string",
            error_code=_ERROR_MODEL,
            details={"parameter": "model", "value": args.model},
        )
    version = str(args.version).strip()
    if version == "":
        raise ValidationError(
            "version must be a non-empty string",
            error_code=_ERROR_VERSION,
            details={"parameter": "version", "value": args.version},
        )

    return PortfolioGenerationOptions(
        storage_root=Path(DEFAULT_STORAGE_ROOT),
        optimizer=optimizer,
        model=model,
        version=version,
        symbols=_normalize_symbols(args.symbols),
        timeframes=_normalize_timeframes(args.timeframes),
        years=_normalize_years(args.years),
        overwrite=bool(args.overwrite),
        workers=workers,
        verbose=bool(args.verbose),
        debug=bool(args.debug),
    )


def build_default_optimizer() -> PortfolioOptimizer:
    """Compose the default production portfolio optimizer for the CLI.

    Returns:
        ``EqualWeightOptimizer`` instance.
    """
    return EqualWeightOptimizer()


def build_optimizer_registry(
    *,
    optimizers: Mapping[str, PortfolioOptimizer] | None = None,
) -> PortfolioOptimizerRegistry:
    """Compose a registry with default or injected optimizer implementations.

    Args:
        optimizers: Optional mapping of registry names to optimizer instances.
            When ``None``, registers ``EqualWeightOptimizer`` under
            ``OptimizerStrategy.EQUAL_WEIGHT``.

    Returns:
        Fully populated ``PortfolioOptimizerRegistry``.
    """
    registry = PortfolioOptimizerRegistry()
    if optimizers is None:
        registry.register(_DEFAULT_OPTIMIZER, build_default_optimizer())
    else:
        registry.register_many(optimizers)
    return registry


def build_portfolio_pipeline(
    options: PortfolioGenerationOptions,
    *,
    optimizer_registry: PortfolioOptimizerRegistry | None = None,
) -> PortfolioPipeline:
    """Compose ``PortfolioPipeline`` from injected optimizer registry deps.

    Args:
        options: Immutable generation options providing the optimizer name.
        optimizer_registry: Optional optimizer registry. When ``None``, a
            default registry containing ``EqualWeightOptimizer`` is built.

    Returns:
        Fully wired ``PortfolioPipeline``.
    """
    if optimizer_registry is None:
        optimizer_registry = build_optimizer_registry()
    elif options.optimizer == _DEFAULT_OPTIMIZER and not optimizer_registry.exists(
        options.optimizer
    ):
        optimizer_registry.register(options.optimizer, build_default_optimizer())
    return PortfolioPipeline(optimizer_registry)


def discover_work(
    signal_repository: SignalRepository,
    options: PortfolioGenerationOptions,
) -> tuple[DiscoveredWorkItem, ...]:
    """Discover portfolio-ready signal partitions matching CLI filters.

    Only signal partitions that exist are scheduled. Missing signal partitions
    are never invented. Partial portfolio datasets are never generated.

    Args:
        signal_repository: Signal repository providing discovery APIs.
        options: CLI filters for symbol, timeframe, and year.

    Returns:
        Deterministically ordered discovered work items.
    """
    partitions = signal_repository.discover_partitions(
        symbols=options.symbols,
        timeframes=options.timeframes,
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    return _group_partitions(partitions, year_filter=options.years)


def format_summary(summary: PortfolioGenerationSummary) -> str:
    """Render a deterministic portfolio-generation summary report.

    Args:
        summary: Aggregate generation summary.

    Returns:
        Multi-line summary text suitable for stdout.
    """
    lines = [
        "=====================================",
        "CQROS Portfolio Generation Summary",
        "=====================================",
        "",
        f"Optimizer: {summary.optimizer}",
        f"Model: {summary.model}",
        f"Version: {summary.version}",
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
    """Run the portfolio-generation CLI.

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
        signal_repository = SignalRepository(layout, datastore)
        portfolio_repository = PortfolioRepository(layout, datastore)
        pipeline = build_portfolio_pipeline(options)
        work = discover_work(signal_repository, options)
        summary = await run_generation(
            pipeline=pipeline,
            signal_repository=signal_repository,
            portfolio_repository=portfolio_repository,
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
    pipeline: PortfolioPipeline,
    signal_repository: SignalRepository,
    portfolio_repository: PortfolioRepository,
    options: PortfolioGenerationOptions,
    work: Sequence[DiscoveredWorkItem],
) -> PortfolioGenerationSummary:
    """Execute discovered work through a bounded symbol worker pool.

    Args:
        pipeline: Injected portfolio pipeline.
        signal_repository: Signal partition repository.
        portfolio_repository: Portfolio partition repository.
        options: Immutable generation options.
        work: Discovered work items.

    Returns:
        Aggregate immutable summary.
    """
    started = time.perf_counter()
    output_directory = options.storage_root / STORAGE_DIR_PORTFOLIOS

    if len(work) == 0:
        return PortfolioGenerationSummary(
            optimizer=options.optimizer,
            model=options.model,
            version=options.version,
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
        signal_repository=signal_repository,
        portfolio_repository=portfolio_repository,
        work_by_symbol=work_by_symbol,
        worker_count=options.workers,
        overwrite=options.overwrite,
        debug=options.debug,
        optimizer_name=options.optimizer,
        model_name=options.model,
        model_version=options.version,
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
    partitions: Sequence[SignalPartitionRef],
    *,
    year_filter: tuple[int, ...] | None,
) -> tuple[DiscoveredWorkItem, ...]:
    """Group signal year partitions into symbol/timeframe work items."""
    year_allowlist = set(year_filter) if year_filter is not None else None
    grouped: dict[tuple[str, str], list[int]] = {}
    for partition in partitions:
        if year_allowlist is not None and partition.year not in year_allowlist:
            continue
        key = (partition.symbol, partition.timeframe)
        grouped.setdefault(key, []).append(partition.year)

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
    pipeline: PortfolioPipeline,
    signal_repository: SignalRepository,
    portfolio_repository: PortfolioRepository,
    work_by_symbol: Mapping[Symbol, Sequence[DiscoveredWorkItem]],
    worker_count: int,
    overwrite: bool,
    debug: bool,
    optimizer_name: str,
    model_name: str,
    model_version: str,
) -> tuple[PortfolioTaskResult, ...]:
    """Drain symbols through a bounded asyncio worker pool."""
    symbols = tuple(work_by_symbol.keys())
    if len(symbols) == 0:
        return ()

    queue: asyncio.Queue[Symbol | None] = asyncio.Queue()
    for symbol in symbols:
        queue.put_nowait(symbol)
    for _ in range(worker_count):
        queue.put_nowait(None)

    collected: list[PortfolioTaskResult] = []
    lock = asyncio.Lock()

    async def worker() -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                results = await _generate_symbol_work(
                    pipeline=pipeline,
                    signal_repository=signal_repository,
                    portfolio_repository=portfolio_repository,
                    symbol=item,
                    work_items=work_by_symbol[item],
                    overwrite=overwrite,
                    debug=debug,
                    optimizer_name=optimizer_name,
                    model_name=model_name,
                    model_version=model_version,
                )
                async with lock:
                    collected.extend(results)
            finally:
                queue.task_done()

    worker_tasks = [
        asyncio.create_task(worker(), name=f"generate-portfolios-worker-{index}")
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
    pipeline: PortfolioPipeline,
    signal_repository: SignalRepository,
    portfolio_repository: PortfolioRepository,
    symbol: Symbol,
    work_items: Sequence[DiscoveredWorkItem],
    overwrite: bool,
    debug: bool,
    optimizer_name: str,
    model_name: str,
    model_version: str,
) -> tuple[PortfolioTaskResult, ...]:
    """Generate portfolio datasets for every discovered year for one symbol."""
    results: list[PortfolioTaskResult] = []
    for item in work_items:
        for year in item.years:
            result = await asyncio.to_thread(
                _generate_partition,
                pipeline,
                signal_repository,
                portfolio_repository,
                symbol=symbol,
                timeframe=item.timeframe,
                year=year,
                overwrite=overwrite,
                debug=debug,
                optimizer_name=optimizer_name,
                model_name=model_name,
                model_version=model_version,
            )
            _print_progress(result)
            results.append(result)
    return tuple(results)


def _generate_partition(
    pipeline: PortfolioPipeline,
    signal_repository: SignalRepository,
    portfolio_repository: PortfolioRepository,
    *,
    symbol: Symbol,
    timeframe: Timeframe,
    year: int,
    overwrite: bool,
    debug: bool,
    optimizer_name: str,
    model_name: str,
    model_version: str,
) -> PortfolioTaskResult:
    """Generate one portfolio year partition synchronously."""
    if not overwrite and portfolio_repository.exists(
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=symbol,
        timeframe=timeframe,
        year=year,
    ):
        return PortfolioTaskResult(
            symbol=symbol,
            timeframe=timeframe,
            year=year,
            status="skipped",
        )

    try:
        signals = signal_repository.load(
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        )
        filtered = _filter_signals_for_model(
            signals,
            model_name=model_name,
            model_version=model_version,
        )
        output = pipeline.run(optimizer_name, filtered)
        portfolio_repository.save(
            output,
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
        return PortfolioTaskResult(
            symbol=symbol,
            timeframe=timeframe,
            year=year,
            status="failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
            error_code=exc.error_code if isinstance(exc, CQROSError) else None,
        )

    return PortfolioTaskResult(
        symbol=symbol,
        timeframe=timeframe,
        year=year,
        status="succeeded",
        rows_generated=output.height,
    )


def _filter_signals_for_model(
    signals: pl.DataFrame,
    *,
    model_name: str,
    model_version: str,
) -> pl.DataFrame:
    """Return signal rows matching the requested model identity.

    Args:
        signals: Loaded signal partition frame.
        model_name: Required model_name column value.
        model_version: Required model_version column value.

    Returns:
        A new DataFrame containing only matching rows.
    """
    return signals.filter(
        (pl.col("model_name") == model_name) & (pl.col("model_version") == model_version)
    )


def _print_progress(result: PortfolioTaskResult) -> None:
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
            "Failed portfolio generation partition; continuing",
            extra=log_extra,
            exc_info=True,
        )
    else:
        _logger.warning(
            "Failed portfolio generation partition; continuing",
            extra=log_extra,
        )


def _build_summary(
    *,
    options: PortfolioGenerationOptions,
    work: Sequence[DiscoveredWorkItem],
    results: Sequence[PortfolioTaskResult],
    duration_seconds: float,
    output_directory: Path,
) -> PortfolioGenerationSummary:
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

    return PortfolioGenerationSummary(
        optimizer=options.optimizer,
        model=options.model,
        version=options.version,
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
