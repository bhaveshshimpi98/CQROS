"""CQROS backtesting-generation CLI.

Purpose:
    Provide an argparse-based production entry point that discovers canonical
    exit-engine partitions and executes ``BacktestingPipeline`` across the
    universe with bounded symbol concurrency, persisting performance ledgers
    through ``BacktestingRepository``.

Responsibilities:
    - Parse CLI arguments for backtesting dataset generation
    - Discover available exit-engine partitions through ``ExitRepository``
    - Load matching accounting, position, and exit-engine partitions for each
      discovered partition
    - Filter loaded accounting frames by optional ``--model`` / ``--version``
    - Resolve ``--engine`` through ``BacktestingRegistry``
    - Execute ``BacktestingPipeline`` and persist via ``BacktestingRepository``
    - Honor ``--overwrite``, worker concurrency, and debug logging
    - Print deterministic progress and a final summary

Dependencies:
    ``argparse``, ``asyncio``, ``polars``, ``cqros.config``, ``cqros.core``,
    ``cqros.accounting``, ``cqros.positions``, ``cqros.exit_engine``,
    ``cqros.backtesting``, and ``cqros.storage``.

Public API:
    ``build_parser``, ``build_options``, ``build_default_engine``,
    ``build_registry``, ``build_backtesting_pipeline``, ``discover_work``,
    ``format_summary``, ``run_generation``, and ``main``.

Notes:
    This module is a thin composition root. It does not implement backtesting
    logic, schema validation, or repository filesystem walks beyond calling
    repository discovery and load/save APIs. Performance reconstruction is
    delegated exclusively to ``BacktestingPipeline``. Persistence remains in
    the CLI because ``BacktestingPipeline`` does not own a repository.
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

from cqros.accounting import AccountingRepository
from cqros.backtesting import (
    BacktestingEngine,
    BacktestingPipeline,
    BacktestingRegistry,
    BacktestingRepository,
    SimpleBacktestingEngine,
)
from cqros.config.models import ResearchConfig
from cqros.core.constants import (
    DEFAULT_STORAGE_ROOT,
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
    STORAGE_DIR_BACKTESTING,
    SUPPORTED_TIMEFRAMES,
)
from cqros.core.exceptions import CQROSError, ValidationError
from cqros.core.types import Symbol, Timeframe
from cqros.exit_engine import ExitEnginePartitionRef, ExitRepository
from cqros.positions import PositionRepository
from cqros.storage import ParquetStore, StorageLayout

__all__ = [
    "BacktestingGenerationOptions",
    "BacktestingGenerationSummary",
    "BacktestingTaskResult",
    "DiscoveredWorkItem",
    "build_backtesting_pipeline",
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

_ERROR_WORKERS: Final[str] = "CLI-GENERATE-BACKTESTING-001"
_ERROR_TIMEFRAME: Final[str] = "CLI-GENERATE-BACKTESTING-002"
_ERROR_YEAR: Final[str] = "CLI-GENERATE-BACKTESTING-003"
_ERROR_MODEL: Final[str] = "CLI-GENERATE-BACKTESTING-004"
_ERROR_VERSION: Final[str] = "CLI-GENERATE-BACKTESTING-005"
_ERROR_MANAGER: Final[str] = "CLI-GENERATE-BACKTESTING-006"
_ERROR_ENGINE: Final[str] = "CLI-GENERATE-BACKTESTING-007"
_ERROR_ACCOUNTING_MISSING: Final[str] = "CLI-GENERATE-BACKTESTING-008"
_ERROR_POSITIONS_MISSING: Final[str] = "CLI-GENERATE-BACKTESTING-009"
_ERROR_EXIT_ENGINE_MISSING: Final[str] = "CLI-GENERATE-BACKTESTING-010"

_COL_CUMULATIVE_RETURN: Final[str] = "cumulative_return"
_COL_MAX_DRAWDOWN: Final[str] = "max_drawdown"
_COL_TRADE_COUNT: Final[str] = "trade_count"


@dataclass(frozen=True, slots=True)
class BacktestingGenerationOptions:
    """Immutable CLI options for backtesting dataset generation.

    Attributes:
        storage_root: Storage root containing ``accounting``, ``positions``,
            ``exit_engine``, and ``backtesting``.
        manager: Order manager identity used for discovery and backtesting
            lineage.
        engine: Registry key of the backtesting engine to execute.
        model: Optional model identifier used to filter accounting rows.
        version: Optional model version used to filter accounting rows.
        symbols: Optional symbol allowlist. ``None`` discovers all.
        timeframes: Optional timeframe allowlist. ``None`` discovers all.
        years: Optional year allowlist. ``None`` discovers all.
        overwrite: When ``True``, regenerate existing backtesting partitions.
        workers: Maximum concurrent symbols.
        verbose: When ``True``, enable INFO logging.
        debug: When ``True``, enable DEBUG logging and log full failure
            tracebacks.
    """

    storage_root: Path
    manager: str
    engine: str
    model: str | None
    version: str | None
    symbols: tuple[Symbol, ...] | None
    timeframes: tuple[Timeframe, ...] | None
    years: tuple[int, ...] | None
    overwrite: bool
    workers: int
    verbose: bool
    debug: bool


@dataclass(frozen=True, slots=True)
class DiscoveredWorkItem:
    """One discovered exit-engine partition group ready for backtesting generation.

    Attributes:
        manager: Order manager identifier of the source partitions.
        symbol: Tradeable symbol.
        timeframe: Available bar interval.
        years: Calendar years with existing exit-engine parquet partitions.
    """

    manager: str
    symbol: Symbol
    timeframe: Timeframe
    years: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class BacktestingTaskResult:
    """Immutable result for one symbol/timeframe/year generation task.

    Attributes:
        symbol: Tradeable symbol.
        timeframe: Bar interval.
        year: Calendar year of the partition.
        status: ``succeeded``, ``failed``, or ``skipped``.
        rows_generated: Output row count when succeeded.
        total_return: Last ``cumulative_return`` when succeeded.
        max_dd: Last ``max_drawdown`` when succeeded.
        trades: Last ``trade_count`` when succeeded.
        error_type: Exception type name when failed.
        error_message: Exception message when failed.
        error_code: Optional CQROS error code when failed.
    """

    symbol: Symbol
    timeframe: Timeframe
    year: int
    status: str
    rows_generated: int | None = None
    total_return: float | None = None
    max_dd: float | None = None
    trades: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class BacktestingGenerationSummary:
    """Immutable aggregate summary for a backtesting-generation run.

    Attributes:
        manager: Order manager identity used for generation.
        engine: Backtesting engine registry key used for generation.
        symbols: Unique symbols for which generation was attempted.
        rows: Sum of output rows across successes.
        trades: Sum of last ``trade_count`` values across successes.
        total_return: Sum of last ``cumulative_return`` values across
            successes.
        max_dd: Maximum last ``max_drawdown`` across successes.
        successful_tasks: Count of succeeded partition tasks.
        failed_tasks: Count of failed partition tasks.
        skipped_tasks: Count of skipped existing partitions.
        duration_seconds: Wall-clock generation duration.
        output_directory: Backtesting-tier output directory.
        failed_task_labels: Deterministic failed-task labels for reporting.
    """

    manager: str
    engine: str
    symbols: int
    rows: int
    trades: int
    total_return: float
    max_dd: float
    successful_tasks: int
    failed_tasks: int
    skipped_tasks: int
    duration_seconds: float
    output_directory: Path
    failed_task_labels: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    """Create the backtesting-generation argument parser.

    Returns:
        Configured ``ArgumentParser`` for backtesting-generation flags.
    """
    parser = argparse.ArgumentParser(
        prog="cqros-generate-backtesting",
        description=(
            "Generate CQROS backtesting performance datasets from discovered "
            "exit-engine partitions and an injected backtesting engine."
        ),
    )
    parser.add_argument(
        "--manager",
        dest="manager",
        required=True,
        metavar="NAME",
        help="Order manager identity used for discovery and backtesting lineage.",
    )
    parser.add_argument(
        "--engine",
        dest="engine",
        default=_DEFAULT_ENGINE,
        metavar="NAME",
        help=f"Backtesting engine registry key (default: {_DEFAULT_ENGINE}).",
    )
    parser.add_argument(
        "--model",
        dest="model",
        default=None,
        metavar="NAME",
        help="Optional stable model identifier used to filter accounting rows.",
    )
    parser.add_argument(
        "--version",
        dest="version",
        default=None,
        metavar="VERSION",
        help="Optional model version identifier used to filter accounting rows.",
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
        help="Regenerate backtesting partitions that already exist.",
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


def build_options(args: argparse.Namespace) -> BacktestingGenerationOptions:
    """Map parsed CLI arguments onto ``BacktestingGenerationOptions``.

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

    model = _optional_non_empty_string(
        args.model,
        parameter="model",
        error_code=_ERROR_MODEL,
    )
    version = _optional_non_empty_string(
        args.version,
        parameter="version",
        error_code=_ERROR_VERSION,
    )

    storage_root = (
        Path(args.storage_root) if args.storage_root is not None else Path(DEFAULT_STORAGE_ROOT)
    )

    return BacktestingGenerationOptions(
        storage_root=storage_root,
        manager=manager,
        engine=engine,
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


def build_default_engine() -> BacktestingEngine:
    """Compose the default production backtesting engine for the CLI.

    Returns:
        ``SimpleBacktestingEngine`` instance.
    """
    return SimpleBacktestingEngine()


def build_registry(
    *,
    engines: Mapping[str, BacktestingEngine] | None = None,
) -> BacktestingRegistry:
    """Compose a registry with default or injected backtesting engine implementations.

    Args:
        engines: Optional mapping of registry names to engine instances.
            When ``None``, registers ``SimpleBacktestingEngine`` under
            ``simple``.

    Returns:
        Fully populated ``BacktestingRegistry``.
    """
    registry = BacktestingRegistry()
    if engines is None:
        registry.register(_DEFAULT_ENGINE, build_default_engine())
    else:
        for name, engine in engines.items():
            registry.register(name, engine)
    return registry


def build_backtesting_pipeline(
    options: BacktestingGenerationOptions,
    *,
    engine_registry: BacktestingRegistry | None = None,
) -> BacktestingPipeline:
    """Compose ``BacktestingPipeline`` from injected engine registry dependencies.

    Args:
        options: Immutable generation options providing the engine name.
        engine_registry: Optional engine registry. When ``None``, a default
            registry containing ``SimpleBacktestingEngine`` is built.

    Returns:
        Fully wired ``BacktestingPipeline``.
    """
    if engine_registry is None:
        engine_registry = build_registry()
    elif options.engine == _DEFAULT_ENGINE and not engine_registry.exists(options.engine):
        engine_registry.register(options.engine, build_default_engine())
    return BacktestingPipeline(engine_registry)


def discover_work(
    exit_repository: ExitRepository,
    options: BacktestingGenerationOptions,
) -> tuple[DiscoveredWorkItem, ...]:
    """Discover backtesting-ready exit-engine partitions matching CLI filters.

    Only exit-engine partitions that exist are scheduled. Missing exit-engine
    partitions are never invented. Matching accounting and position partitions
    are validated at generation time; missing dependencies fail the individual
    task.

    Args:
        exit_repository: Exit-engine repository providing discovery APIs.
        options: CLI filters for manager, symbol, timeframe, and year.

    Returns:
        Deterministically ordered discovered work items.
    """
    partitions = exit_repository.discover_partitions(
        managers=(options.manager,),
        symbols=options.symbols,
        timeframes=options.timeframes,
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    return _group_partitions(partitions, year_filter=options.years)


def format_summary(summary: BacktestingGenerationSummary) -> str:
    """Render a deterministic backtesting-generation summary report.

    Args:
        summary: Aggregate generation summary.

    Returns:
        Multi-line summary text suitable for stdout.
    """
    lines = [
        "=====================================",
        "CQROS Backtesting Generation Summary",
        "=====================================",
        "",
        f"Manager: {summary.manager}",
        f"Engine: {summary.engine}",
        "",
        f"Symbols: {summary.symbols}",
        f"Rows: {summary.rows}",
        f"Trades: {summary.trades}",
        f"Total Return: {summary.total_return:.4f}",
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
    """Run the backtesting-generation CLI.

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
        accounting_repository = AccountingRepository(layout, datastore)
        position_repository = PositionRepository(layout, datastore)
        exit_repository = ExitRepository(layout, datastore)
        backtesting_repository = BacktestingRepository(layout, datastore)
        pipeline = build_backtesting_pipeline(options)
        work = discover_work(exit_repository, options)
        summary = await run_generation(
            pipeline=pipeline,
            accounting_repository=accounting_repository,
            position_repository=position_repository,
            exit_repository=exit_repository,
            backtesting_repository=backtesting_repository,
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
    pipeline: BacktestingPipeline,
    accounting_repository: AccountingRepository,
    position_repository: PositionRepository,
    exit_repository: ExitRepository,
    backtesting_repository: BacktestingRepository,
    options: BacktestingGenerationOptions,
    work: Sequence[DiscoveredWorkItem],
) -> BacktestingGenerationSummary:
    """Execute discovered work through a bounded symbol worker pool.

    Args:
        pipeline: Injected backtesting pipeline.
        accounting_repository: Accounting partition repository.
        position_repository: Position partition repository.
        exit_repository: Exit-engine partition repository.
        backtesting_repository: Backtesting partition repository.
        options: Immutable generation options.
        work: Discovered work items.

    Returns:
        Aggregate immutable summary.
    """
    started = time.perf_counter()
    output_directory = options.storage_root / STORAGE_DIR_BACKTESTING

    if len(work) == 0:
        return BacktestingGenerationSummary(
            manager=options.manager,
            engine=options.engine,
            symbols=0,
            rows=0,
            trades=0,
            total_return=0.0,
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
        accounting_repository=accounting_repository,
        position_repository=position_repository,
        exit_repository=exit_repository,
        backtesting_repository=backtesting_repository,
        work_by_symbol=work_by_symbol,
        worker_count=options.workers,
        overwrite=options.overwrite,
        debug=options.debug,
        manager_name=options.manager,
        engine_name=options.engine,
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


def _optional_non_empty_string(
    value: object | None,
    *,
    parameter: str,
    error_code: str,
) -> str | None:
    """Validate an optional non-blank string filter."""
    if value is None:
        return None
    stripped = str(value).strip()
    if stripped == "":
        raise ValidationError(
            f"{parameter} must be a non-empty string",
            error_code=error_code,
            details={"parameter": parameter, "value": value},
        )
    return stripped


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
    partitions: Sequence[ExitEnginePartitionRef],
    *,
    year_filter: tuple[int, ...] | None,
) -> tuple[DiscoveredWorkItem, ...]:
    """Group exit-engine year partitions into manager/symbol/timeframe work items."""
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
    pipeline: BacktestingPipeline,
    accounting_repository: AccountingRepository,
    position_repository: PositionRepository,
    exit_repository: ExitRepository,
    backtesting_repository: BacktestingRepository,
    work_by_symbol: Mapping[Symbol, Sequence[DiscoveredWorkItem]],
    worker_count: int,
    overwrite: bool,
    debug: bool,
    manager_name: str,
    engine_name: str,
    model_name: str | None,
    model_version: str | None,
) -> tuple[BacktestingTaskResult, ...]:
    """Drain symbols through a bounded asyncio worker pool."""
    symbols = tuple(work_by_symbol.keys())
    if len(symbols) == 0:
        return ()

    queue: asyncio.Queue[Symbol | None] = asyncio.Queue()
    for symbol in symbols:
        queue.put_nowait(symbol)
    for _ in range(worker_count):
        queue.put_nowait(None)

    collected: list[BacktestingTaskResult] = []
    lock = asyncio.Lock()

    async def worker() -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                results = await _generate_symbol_work(
                    pipeline=pipeline,
                    accounting_repository=accounting_repository,
                    position_repository=position_repository,
                    exit_repository=exit_repository,
                    backtesting_repository=backtesting_repository,
                    symbol=item,
                    work_items=work_by_symbol[item],
                    overwrite=overwrite,
                    debug=debug,
                    manager_name=manager_name,
                    engine_name=engine_name,
                    model_name=model_name,
                    model_version=model_version,
                )
                async with lock:
                    collected.extend(results)
            finally:
                queue.task_done()

    worker_tasks = [
        asyncio.create_task(worker(), name=f"generate-backtesting-worker-{index}")
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
    pipeline: BacktestingPipeline,
    accounting_repository: AccountingRepository,
    position_repository: PositionRepository,
    exit_repository: ExitRepository,
    backtesting_repository: BacktestingRepository,
    symbol: Symbol,
    work_items: Sequence[DiscoveredWorkItem],
    overwrite: bool,
    debug: bool,
    manager_name: str,
    engine_name: str,
    model_name: str | None,
    model_version: str | None,
) -> tuple[BacktestingTaskResult, ...]:
    """Generate backtesting datasets for every discovered year for one symbol."""
    results: list[BacktestingTaskResult] = []
    for item in work_items:
        for year in item.years:
            result = await asyncio.to_thread(
                _generate_partition,
                pipeline,
                accounting_repository,
                position_repository,
                exit_repository,
                backtesting_repository,
                manager=item.manager,
                symbol=symbol,
                timeframe=item.timeframe,
                year=year,
                overwrite=overwrite,
                debug=debug,
                manager_name=manager_name,
                engine_name=engine_name,
                model_name=model_name,
                model_version=model_version,
            )
            _print_progress(result)
            results.append(result)
    return tuple(results)


def _generate_partition(
    pipeline: BacktestingPipeline,
    accounting_repository: AccountingRepository,
    position_repository: PositionRepository,
    exit_repository: ExitRepository,
    backtesting_repository: BacktestingRepository,
    *,
    manager: str,
    symbol: Symbol,
    timeframe: Timeframe,
    year: int,
    overwrite: bool,
    debug: bool,
    manager_name: str,
    engine_name: str,
    model_name: str | None,
    model_version: str | None,
) -> BacktestingTaskResult:
    """Generate one backtesting year partition synchronously."""
    if not overwrite and backtesting_repository.exists(
        manager=manager_name,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=symbol,
        timeframe=timeframe,
        year=year,
    ):
        return BacktestingTaskResult(
            symbol=symbol,
            timeframe=timeframe,
            year=year,
            status="skipped",
        )

    try:
        if not accounting_repository.exists(
            manager=manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        ):
            raise ValidationError(
                f"accounting partition missing for {manager}/{symbol}/{timeframe}/{year}",
                error_code=_ERROR_ACCOUNTING_MISSING,
                details={
                    "manager": manager,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "year": year,
                    "missing": "accounting",
                },
            )

        if not position_repository.exists(
            manager=manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        ):
            raise ValidationError(
                f"positions partition missing for {manager}/{symbol}/{timeframe}/{year}",
                error_code=_ERROR_POSITIONS_MISSING,
                details={
                    "manager": manager,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "year": year,
                    "missing": "positions",
                },
            )

        if not exit_repository.exists(
            manager=manager_name,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        ):
            raise ValidationError(
                (
                    "exit_engine partition missing for "
                    f"{manager_name}/{symbol}/{timeframe}/{year}"
                ),
                error_code=_ERROR_EXIT_ENGINE_MISSING,
                details={
                    "manager": manager_name,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "year": year,
                    "missing": "exit_engine",
                },
            )

        accounting = accounting_repository.load(
            manager=manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        )
        filtered_accounting = _filter_accounting_for_model(
            accounting,
            model_name=model_name,
            model_version=model_version,
        )
        positions = position_repository.load(
            manager=manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        )
        exit_engine = exit_repository.load(
            manager=manager_name,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        )
        output = pipeline.run(
            filtered_accounting,
            positions,
            exit_engine,
            manager=manager_name,
            engine_name=engine_name,
        )
        backtesting_repository.save(
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
        return BacktestingTaskResult(
            symbol=symbol,
            timeframe=timeframe,
            year=year,
            status="failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
            error_code=exc.error_code if isinstance(exc, CQROSError) else None,
        )

    rows_generated, total_return, max_dd, trades = _extract_partition_stats(output)
    return BacktestingTaskResult(
        symbol=symbol,
        timeframe=timeframe,
        year=year,
        status="succeeded",
        rows_generated=rows_generated,
        total_return=total_return,
        max_dd=max_dd,
        trades=trades,
    )


def _filter_accounting_for_model(
    accounting: pl.DataFrame,
    *,
    model_name: str | None,
    model_version: str | None,
) -> pl.DataFrame:
    """Return accounting rows matching optional model identity filters."""
    if model_name is None and model_version is None:
        return accounting
    predicate = pl.lit(True)
    if model_name is not None:
        predicate = predicate & (pl.col("model_name") == model_name)
    if model_version is not None:
        predicate = predicate & (pl.col("model_version") == model_version)
    return accounting.filter(predicate)


def _extract_partition_stats(frame: pl.DataFrame) -> tuple[int, float, float, int]:
    """Extract row count and terminal performance metrics from one ledger frame.

    Args:
        frame: Finalized backtesting output DataFrame.

    Returns:
        A 4-tuple of ``(rows, total_return, max_dd, trades)``.
    """
    if frame.height == 0:
        return 0, 0.0, 0.0, 0

    last_row = frame.sort("open_time", maintain_order=True).tail(1)
    total_return = float(last_row[_COL_CUMULATIVE_RETURN][0])
    max_dd = float(last_row[_COL_MAX_DRAWDOWN][0])
    trades = int(last_row[_COL_TRADE_COUNT][0])
    return frame.height, total_return, max_dd, trades


def _print_progress(result: BacktestingTaskResult) -> None:
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
            "Failed backtesting generation partition; continuing",
            extra=log_extra,
            exc_info=True,
        )
    else:
        _logger.warning(
            "Failed backtesting generation partition; continuing",
            extra=log_extra,
        )


def _build_summary(
    *,
    options: BacktestingGenerationOptions,
    work: Sequence[DiscoveredWorkItem],
    results: Sequence[BacktestingTaskResult],
    duration_seconds: float,
    output_directory: Path,
) -> BacktestingGenerationSummary:
    """Aggregate task results into a generation report."""
    symbols_discovered = {item.symbol for item in work}
    symbols_processed: set[Symbol] = set()
    successful_tasks = 0
    failed_tasks = 0
    skipped_tasks = 0
    rows = 0
    trades = 0
    total_return = 0.0
    max_dd = 0.0
    failed_labels: set[str] = set()

    for result in results:
        symbols_processed.add(result.symbol)
        if result.status == "succeeded":
            successful_tasks += 1
            if result.rows_generated is not None:
                rows += result.rows_generated
            if result.trades is not None:
                trades += result.trades
            if result.total_return is not None:
                total_return += result.total_return
            if result.max_dd is not None:
                max_dd = max(max_dd, result.max_dd)
        elif result.status == "skipped":
            skipped_tasks += 1
        else:
            failed_tasks += 1
            failed_labels.add(f"{result.symbol} {result.timeframe} {result.year}")

    return BacktestingGenerationSummary(
        manager=options.manager,
        engine=options.engine,
        symbols=len(symbols_processed) if results else len(symbols_discovered),
        rows=rows,
        trades=trades,
        total_return=total_return,
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
