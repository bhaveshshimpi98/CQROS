"""CQROS OMS order-generation CLI.

Purpose:
    Provide an argparse-based production entry point that discovers risk
    partitions and executes ``OrderPipeline`` across the universe with bounded
    symbol concurrency, persisting outputs through ``OrderRepository``.

Responsibilities:
    - Parse CLI arguments for OMS order dataset generation
    - Discover available risk partitions through ``RiskRepository``
    - Filter loaded risk frames by optional ``--model`` / ``--version``
    - Resolve ``--manager`` through ``OrderManagerRegistry``
    - Execute ``OrderPipeline`` and persist via ``OrderRepository``
    - Honor ``--overwrite``, worker concurrency, and debug logging
    - Print deterministic progress and a final summary

Dependencies:
    ``argparse``, ``asyncio``, ``polars``, ``cqros.config``, ``cqros.core``,
    ``cqros.oms``, and ``cqros.storage``.

Public API:
    ``build_parser``, ``build_options``, ``build_default_manager``,
    ``build_manager_registry``, ``build_order_pipeline``, ``discover_work``,
    ``format_summary``, ``run_generation``, and ``main``.

Notes:
    This module is a thin composition root. It does not implement OMS order
    logic, schema validation, or repository filesystem walks beyond calling
    repository discovery and load/save APIs. Order creation is delegated
    exclusively to ``OrderPipeline``. Persistence remains in the CLI because
    ``OrderPipeline`` does not own a repository.
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
    STORAGE_DIR_ORDERS,
    SUPPORTED_TIMEFRAMES,
)
from cqros.core.exceptions import CQROSError, ValidationError
from cqros.core.types import Symbol, Timeframe
from cqros.oms import (
    OrderManager,
    OrderManagerRegistry,
    OrderManagerType,
    OrderPipeline,
    SimpleOrderManager,
)
from cqros.storage import (
    OrderRepository,
    ParquetStore,
    RiskPartitionRef,
    RiskRepository,
    StorageLayout,
)

__all__ = [
    "DiscoveredWorkItem",
    "OrderGenerationOptions",
    "OrderGenerationSummary",
    "OrderTaskResult",
    "build_default_manager",
    "build_manager_registry",
    "build_options",
    "build_order_pipeline",
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
_DEFAULT_MANAGER: Final[str] = OrderManagerType.SIMPLE.value

_EXCHANGE: Final[str] = EXCHANGE_BINANCE
_MARKET: Final[str] = MARKET_USDT_PERPETUAL

_ERROR_WORKERS: Final[str] = "CLI-GENERATE-ORDERS-001"
_ERROR_TIMEFRAME: Final[str] = "CLI-GENERATE-ORDERS-002"
_ERROR_YEAR: Final[str] = "CLI-GENERATE-ORDERS-003"
_ERROR_MODEL: Final[str] = "CLI-GENERATE-ORDERS-004"
_ERROR_VERSION: Final[str] = "CLI-GENERATE-ORDERS-005"
_ERROR_MANAGER: Final[str] = "CLI-GENERATE-ORDERS-006"
_ERROR_POLICY: Final[str] = "CLI-GENERATE-ORDERS-007"


@dataclass(frozen=True, slots=True)
class OrderGenerationOptions:
    """Immutable CLI options for OMS order dataset generation.

    Attributes:
        storage_root: Storage root containing ``risks`` and ``orders``.
        manager: Registry key of the order manager to execute.
        policy: Optional risk-policy filter for discovery. ``None`` discovers
            every policy present under the risks tier.
        model: Optional model identifier used to filter risk rows.
        version: Optional model version used to filter risk rows.
        symbols: Optional symbol allowlist. ``None`` discovers all.
        timeframes: Optional timeframe allowlist. ``None`` discovers all.
        years: Optional year allowlist. ``None`` discovers all.
        overwrite: When ``True``, regenerate existing order partitions.
        workers: Maximum concurrent symbols.
        verbose: When ``True``, enable INFO logging.
        debug: When ``True``, enable DEBUG logging and log full failure
            tracebacks.
    """

    storage_root: Path
    manager: str
    policy: str | None
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
    """One discovered risk partition group ready for order generation.

    Attributes:
        policy: Risk policy identifier of the source partitions.
        symbol: Tradeable symbol.
        timeframe: Available bar interval.
        years: Calendar years with existing risk parquet partitions.
    """

    policy: str
    symbol: Symbol
    timeframe: Timeframe
    years: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class OrderTaskResult:
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
class OrderGenerationSummary:
    """Immutable aggregate summary for an order-generation run.

    Attributes:
        manager: Order manager registry key used for generation.
        version: Optional model version used for generation.
        symbols_discovered: Unique symbols discovered from risk storage.
        symbols_processed: Unique symbols for which generation was attempted.
        timeframes_processed: Unique timeframes attempted.
        successful_tasks: Count of succeeded partition tasks.
        failed_tasks: Count of failed partition tasks.
        skipped_tasks: Count of skipped existing partitions.
        rows_generated: Sum of output rows across successes.
        duration_seconds: Wall-clock generation duration.
        output_directory: Orders-tier output directory.
        failed_task_labels: Deterministic failed-task labels for reporting.
    """

    manager: str
    version: str | None
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
    """Create the order-generation argument parser.

    Returns:
        Configured ``ArgumentParser`` for order-generation flags.
    """
    parser = argparse.ArgumentParser(
        prog="cqros-generate-orders",
        description=(
            "Generate CQROS OMS order datasets from discovered risk "
            "partitions and an injected order manager."
        ),
    )
    parser.add_argument(
        "--manager",
        dest="manager",
        required=True,
        metavar="NAME",
        help="Order manager registry key (for example simple).",
    )
    parser.add_argument(
        "--policy",
        dest="policy",
        default=None,
        metavar="NAME",
        help="Optional risk-policy filter used during risk discovery.",
    )
    parser.add_argument(
        "--model",
        dest="model",
        default=None,
        metavar="NAME",
        help="Optional stable model identifier used to filter risk rows.",
    )
    parser.add_argument(
        "--version",
        dest="version",
        default=None,
        metavar="VERSION",
        help="Optional model version identifier used to filter risk rows.",
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
        help="Regenerate order partitions that already exist.",
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


def build_options(args: argparse.Namespace) -> OrderGenerationOptions:
    """Map parsed CLI arguments onto ``OrderGenerationOptions``.

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

    policy = _optional_non_empty_string(
        args.policy,
        parameter="policy",
        error_code=_ERROR_POLICY,
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

    return OrderGenerationOptions(
        storage_root=Path(DEFAULT_STORAGE_ROOT),
        manager=manager,
        policy=policy,
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


def build_default_manager() -> OrderManager:
    """Compose the default production order manager for the CLI.

    Returns:
        ``SimpleOrderManager`` instance.
    """
    return SimpleOrderManager()


def build_manager_registry(
    *,
    managers: Mapping[str, OrderManager] | None = None,
) -> OrderManagerRegistry:
    """Compose a registry with default or injected order-manager implementations.

    Args:
        managers: Optional mapping of registry names to order-manager instances.
            When ``None``, registers ``SimpleOrderManager`` under
            ``OrderManagerType.SIMPLE``.

    Returns:
        Fully populated ``OrderManagerRegistry``.
    """
    registry = OrderManagerRegistry()
    if managers is None:
        registry.register(_DEFAULT_MANAGER, build_default_manager())
    else:
        registry.register_many(managers)
    return registry


def build_order_pipeline(
    options: OrderGenerationOptions,
    *,
    manager_registry: OrderManagerRegistry | None = None,
) -> OrderPipeline:
    """Compose ``OrderPipeline`` from injected manager registry deps.

    Args:
        options: Immutable generation options providing the manager name.
        manager_registry: Optional manager registry. When ``None``, a default
            registry containing ``SimpleOrderManager`` is built.

    Returns:
        Fully wired ``OrderPipeline``.
    """
    if manager_registry is None:
        manager_registry = build_manager_registry()
    elif options.manager == _DEFAULT_MANAGER and not manager_registry.exists(options.manager):
        manager_registry.register(options.manager, build_default_manager())
    return OrderPipeline(manager_registry)


def discover_work(
    risk_repository: RiskRepository,
    options: OrderGenerationOptions,
) -> tuple[DiscoveredWorkItem, ...]:
    """Discover order-ready risk partitions matching CLI filters.

    Only risk partitions that exist are scheduled. Missing risk partitions
    are never invented. Partial order datasets are never generated.

    Args:
        risk_repository: Risk repository providing discovery APIs.
        options: CLI filters for policy, symbol, timeframe, and year.

    Returns:
        Deterministically ordered discovered work items.
    """
    policies = (options.policy,) if options.policy is not None else None
    partitions = risk_repository.discover_partitions(
        policies=policies,
        symbols=options.symbols,
        timeframes=options.timeframes,
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    return _group_partitions(partitions, year_filter=options.years)


def format_summary(summary: OrderGenerationSummary) -> str:
    """Render a deterministic order-generation summary report.

    Args:
        summary: Aggregate generation summary.

    Returns:
        Multi-line summary text suitable for stdout.
    """
    version_text = summary.version if summary.version is not None else ""
    lines = [
        "=====================================",
        "CQROS Order Generation Summary",
        "=====================================",
        "",
        f"Manager: {summary.manager}",
        f"Version: {version_text}",
        "",
        f"Symbols discovered: {summary.symbols_discovered}",
        f"Symbols processed: {summary.symbols_processed}",
        f"Timeframes processed: {summary.timeframes_processed}",
        "",
        f"Successful tasks: {summary.successful_tasks}",
        f"Failed tasks: {summary.failed_tasks}",
        f"Skipped tasks: {summary.skipped_tasks}",
        "",
        f"Rows generated: {summary.rows_generated}",
        "",
        f"Generation duration: {_format_duration(summary.duration_seconds)}",
        "",
        f"Output directory: {_format_output_directory(summary.output_directory)}",
    ]
    if summary.failed_task_labels:
        lines.extend(["", "Failed Tasks", ""])
        lines.extend(f"- {label}" for label in summary.failed_task_labels)
    return "\n".join(lines) + "\n"


async def main(argv: Sequence[str] | None = None) -> int:
    """Run the order-generation CLI.

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
        risk_repository = RiskRepository(layout, datastore)
        order_repository = OrderRepository(layout, datastore)
        pipeline = build_order_pipeline(options)
        work = discover_work(risk_repository, options)
        summary = await run_generation(
            pipeline=pipeline,
            risk_repository=risk_repository,
            order_repository=order_repository,
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
    pipeline: OrderPipeline,
    risk_repository: RiskRepository,
    order_repository: OrderRepository,
    options: OrderGenerationOptions,
    work: Sequence[DiscoveredWorkItem],
) -> OrderGenerationSummary:
    """Execute discovered work through a bounded symbol worker pool.

    Args:
        pipeline: Injected order pipeline.
        risk_repository: Risk partition repository.
        order_repository: Order partition repository.
        options: Immutable generation options.
        work: Discovered work items.

    Returns:
        Aggregate immutable summary.
    """
    started = time.perf_counter()
    output_directory = options.storage_root / STORAGE_DIR_ORDERS

    if len(work) == 0:
        return OrderGenerationSummary(
            manager=options.manager,
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
        risk_repository=risk_repository,
        order_repository=order_repository,
        work_by_symbol=work_by_symbol,
        worker_count=options.workers,
        overwrite=options.overwrite,
        debug=options.debug,
        manager_name=options.manager,
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
    partitions: Sequence[RiskPartitionRef],
    *,
    year_filter: tuple[int, ...] | None,
) -> tuple[DiscoveredWorkItem, ...]:
    """Group risk year partitions into policy/symbol/timeframe work items."""
    year_allowlist = set(year_filter) if year_filter is not None else None
    grouped: dict[tuple[str, str, str], list[int]] = {}
    for partition in partitions:
        if year_allowlist is not None and partition.year not in year_allowlist:
            continue
        key = (partition.policy, partition.symbol, partition.timeframe)
        grouped.setdefault(key, []).append(partition.year)

    items: list[DiscoveredWorkItem] = []
    for (policy, symbol, timeframe), years in grouped.items():
        items.append(
            DiscoveredWorkItem(
                policy=policy,
                symbol=symbol,
                timeframe=timeframe,
                years=tuple(sorted(years)),
            )
        )
    return tuple(
        sorted(
            items,
            key=lambda item: (item.policy, item.symbol, item.timeframe),
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
    pipeline: OrderPipeline,
    risk_repository: RiskRepository,
    order_repository: OrderRepository,
    work_by_symbol: Mapping[Symbol, Sequence[DiscoveredWorkItem]],
    worker_count: int,
    overwrite: bool,
    debug: bool,
    manager_name: str,
    model_name: str | None,
    model_version: str | None,
) -> tuple[OrderTaskResult, ...]:
    """Drain symbols through a bounded asyncio worker pool."""
    symbols = tuple(work_by_symbol.keys())
    if len(symbols) == 0:
        return ()

    queue: asyncio.Queue[Symbol | None] = asyncio.Queue()
    for symbol in symbols:
        queue.put_nowait(symbol)
    for _ in range(worker_count):
        queue.put_nowait(None)

    collected: list[OrderTaskResult] = []
    lock = asyncio.Lock()

    async def worker() -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                results = await _generate_symbol_work(
                    pipeline=pipeline,
                    risk_repository=risk_repository,
                    order_repository=order_repository,
                    symbol=item,
                    work_items=work_by_symbol[item],
                    overwrite=overwrite,
                    debug=debug,
                    manager_name=manager_name,
                    model_name=model_name,
                    model_version=model_version,
                )
                async with lock:
                    collected.extend(results)
            finally:
                queue.task_done()

    worker_tasks = [
        asyncio.create_task(worker(), name=f"generate-orders-worker-{index}")
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
    pipeline: OrderPipeline,
    risk_repository: RiskRepository,
    order_repository: OrderRepository,
    symbol: Symbol,
    work_items: Sequence[DiscoveredWorkItem],
    overwrite: bool,
    debug: bool,
    manager_name: str,
    model_name: str | None,
    model_version: str | None,
) -> tuple[OrderTaskResult, ...]:
    """Generate order datasets for every discovered year for one symbol."""
    results: list[OrderTaskResult] = []
    for item in work_items:
        for year in item.years:
            result = await asyncio.to_thread(
                _generate_partition,
                pipeline,
                risk_repository,
                order_repository,
                policy=item.policy,
                symbol=symbol,
                timeframe=item.timeframe,
                year=year,
                overwrite=overwrite,
                debug=debug,
                manager_name=manager_name,
                model_name=model_name,
                model_version=model_version,
            )
            _print_progress(result)
            results.append(result)
    return tuple(results)


def _generate_partition(
    pipeline: OrderPipeline,
    risk_repository: RiskRepository,
    order_repository: OrderRepository,
    *,
    policy: str,
    symbol: Symbol,
    timeframe: Timeframe,
    year: int,
    overwrite: bool,
    debug: bool,
    manager_name: str,
    model_name: str | None,
    model_version: str | None,
) -> OrderTaskResult:
    """Generate one order year partition synchronously."""
    if not overwrite and order_repository.exists(
        manager=manager_name,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=symbol,
        timeframe=timeframe,
        year=year,
    ):
        return OrderTaskResult(
            symbol=symbol,
            timeframe=timeframe,
            year=year,
            status="skipped",
        )

    try:
        risks = risk_repository.load(
            policy=policy,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        )
        filtered = _filter_risks_for_model(
            risks,
            model_name=model_name,
            model_version=model_version,
        )
        output = pipeline.run(manager_name, filtered)
        order_repository.save(
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
        return OrderTaskResult(
            symbol=symbol,
            timeframe=timeframe,
            year=year,
            status="failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
            error_code=exc.error_code if isinstance(exc, CQROSError) else None,
        )

    return OrderTaskResult(
        symbol=symbol,
        timeframe=timeframe,
        year=year,
        status="succeeded",
        rows_generated=output.height,
    )


def _filter_risks_for_model(
    risks: pl.DataFrame,
    *,
    model_name: str | None,
    model_version: str | None,
) -> pl.DataFrame:
    """Return risk rows matching optional model identity filters.

    Args:
        risks: Loaded risk partition frame.
        model_name: Optional model_name column value.
        model_version: Optional model_version column value.

    Returns:
        A new DataFrame containing only matching rows, or ``risks`` when no
        filters are supplied.
    """
    if model_name is None and model_version is None:
        return risks
    predicate = pl.lit(True)
    if model_name is not None:
        predicate = predicate & (pl.col("model_name") == model_name)
    if model_version is not None:
        predicate = predicate & (pl.col("model_version") == model_version)
    return risks.filter(predicate)


def _print_progress(result: OrderTaskResult) -> None:
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
            "Failed order generation partition; continuing",
            extra=log_extra,
            exc_info=True,
        )
    else:
        _logger.warning(
            "Failed order generation partition; continuing",
            extra=log_extra,
        )


def _build_summary(
    *,
    options: OrderGenerationOptions,
    work: Sequence[DiscoveredWorkItem],
    results: Sequence[OrderTaskResult],
    duration_seconds: float,
    output_directory: Path,
) -> OrderGenerationSummary:
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

    return OrderGenerationSummary(
        manager=options.manager,
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
