"""CQROS pyramiding-generation CLI.

Purpose:
    Provide an argparse-based production entry point that discovers canonical
    trade-management partitions and executes ``PyramidingPipeline`` across the
    universe with bounded symbol concurrency, persisting outputs through
    ``PyramidingRepository``.

Responsibilities:
    - Parse CLI arguments for pyramiding dataset generation
    - Discover available trade-management partitions through
      ``TradeManagementRepository``
    - Load matching position, accounting, portfolio-risk, trade-management, and
      processed OHLCV partitions for each trade-management partition
    - Filter loaded accounting frames by optional ``--model`` / ``--version``
    - Resolve ``--engine`` through ``PyramidingRegistry``
    - Execute ``PyramidingPipeline`` and persist via ``PyramidingRepository``
    - Honor ``--overwrite``, worker concurrency, and debug logging
    - Print deterministic progress and a final summary

Dependencies:
    ``argparse``, ``asyncio``, ``polars``, ``cqros.config``, ``cqros.core``,
    ``cqros.accounting``, ``cqros.portfolio_risk``, ``cqros.positions``,
    ``cqros.trade_management``, ``cqros.pyramiding``, and ``cqros.storage``.

Public API:
    ``build_parser``, ``build_options``, ``build_default_engine``,
    ``build_registry``, ``build_pyramiding_pipeline``, ``discover_work``,
    ``format_summary``, ``run_generation``, and ``main``.

Notes:
    This module is a thin composition root. It does not implement
    pyramiding logic, schema validation, or repository filesystem walks
    beyond calling repository discovery and load/save APIs. Pyramiding
    evaluation is delegated exclusively to ``PyramidingPipeline``.
    Persistence remains in the CLI because ``PyramidingPipeline`` does
    not own a repository.
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
from cqros.config.models import ResearchConfig
from cqros.core.constants import (
    DEFAULT_STORAGE_ROOT,
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
    STORAGE_DIR_PYRAMIDING,
    SUPPORTED_TIMEFRAMES,
)
from cqros.core.exceptions import CQROSError, ValidationError
from cqros.core.types import Symbol, Timeframe
from cqros.portfolio_risk import PortfolioRiskRepository
from cqros.positions import PositionRepository
from cqros.pyramiding import (
    PyramidingEngine,
    PyramidingPipeline,
    PyramidingReason,
    PyramidingRegistry,
    PyramidingRepository,
    SimplePyramidingEngine,
)
from cqros.storage import ParquetStore, ProcessedMarketDataRepository, StorageLayout
from cqros.trade_management import (
    TradeManagementPartitionRef,
    TradeManagementRepository,
)

__all__ = [
    "DiscoveredWorkItem",
    "PyramidingGenerationOptions",
    "PyramidingGenerationSummary",
    "PyramidingTaskResult",
    "build_default_engine",
    "build_options",
    "build_parser",
    "build_pyramiding_pipeline",
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
_DEFAULT_MANAGER: Final[str] = "simple"
_DEFAULT_ENGINE: Final[str] = "simple"

_EXCHANGE: Final[str] = EXCHANGE_BINANCE
_MARKET: Final[str] = MARKET_USDT_PERPETUAL

_ERROR_WORKERS: Final[str] = "CLI-GENERATE-PYRAMIDING-001"
_ERROR_TIMEFRAME: Final[str] = "CLI-GENERATE-PYRAMIDING-002"
_ERROR_YEAR: Final[str] = "CLI-GENERATE-PYRAMIDING-003"
_ERROR_MODEL: Final[str] = "CLI-GENERATE-PYRAMIDING-004"
_ERROR_VERSION: Final[str] = "CLI-GENERATE-PYRAMIDING-005"
_ERROR_MANAGER: Final[str] = "CLI-GENERATE-PYRAMIDING-006"
_ERROR_ENGINE: Final[str] = "CLI-GENERATE-PYRAMIDING-007"
_ERROR_POSITIONS_MISSING: Final[str] = "CLI-GENERATE-PYRAMIDING-008"
_ERROR_PORTFOLIO_RISK_MISSING: Final[str] = "CLI-GENERATE-PYRAMIDING-009"
_ERROR_MARKET_MISSING: Final[str] = "CLI-GENERATE-PYRAMIDING-010"
_ERROR_OHLCV_COLUMNS: Final[str] = "CLI-GENERATE-PYRAMIDING-011"

_COL_CLOSE: Final[str] = "close"
_COL_HIGH: Final[str] = "high"
_COL_OPEN_TIME: Final[str] = "open_time"
_COL_PRICE: Final[str] = "price"
_COL_REASON: Final[str] = "reason"
_COL_SYMBOL: Final[str] = "symbol"
_COL_TIMEFRAME: Final[str] = "timeframe"

_REASON_READY_TO_ADD: Final[str] = PyramidingReason.READY_TO_ADD.value
_REASON_NOT_ELIGIBLE: Final[str] = PyramidingReason.NOT_ELIGIBLE.value
_REASON_MAX_ADDS_REACHED: Final[str] = PyramidingReason.MAX_ADDS_REACHED.value


@dataclass(frozen=True, slots=True)
class PyramidingGenerationOptions:
    """Immutable CLI options for pyramiding dataset generation.

    Attributes:
        storage_root: Storage root containing ``accounting``, ``positions``,
            ``portfolio_risk``, ``trade_management``, processed OHLCV, and
            ``pyramiding``.
        manager: Order manager identity used for discovery and pyramiding
            lineage.
        engine: Registry key of the pyramiding engine to execute.
        model: Optional model identifier used to filter accounting rows.
        version: Optional model version used to filter accounting rows.
        symbols: Optional symbol allowlist. ``None`` discovers all.
        timeframes: Optional timeframe allowlist. ``None`` discovers all.
        years: Optional year allowlist. ``None`` discovers all.
        overwrite: When ``True``, regenerate existing pyramiding partitions.
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
    """One discovered trade-management partition group ready for pyramiding generation.

    Attributes:
        manager: Order manager identifier of the source partitions.
        symbol: Tradeable symbol.
        timeframe: Available bar interval.
        years: Calendar years with existing trade-management parquet partitions.
    """

    manager: str
    symbol: Symbol
    timeframe: Timeframe
    years: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PyramidingTaskResult:
    """Immutable result for one symbol/timeframe/year generation task.

    Attributes:
        symbol: Tradeable symbol.
        timeframe: Bar interval.
        year: Calendar year of the partition.
        status: ``succeeded``, ``failed``, or ``skipped``.
        rows_generated: Output row count when succeeded.
        ready_to_add: Count of rows with ``reason=READY_TO_ADD`` when succeeded.
        not_eligible: Count of rows with ``reason=NOT_ELIGIBLE`` when succeeded.
        max_adds_reached: Count of rows with ``reason=MAX_ADDS_REACHED`` when
            succeeded.
        error_type: Exception type name when failed.
        error_message: Exception message when failed.
        error_code: Optional CQROS error code when failed.
    """

    symbol: Symbol
    timeframe: Timeframe
    year: int
    status: str
    rows_generated: int | None = None
    ready_to_add: int | None = None
    not_eligible: int | None = None
    max_adds_reached: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class PyramidingGenerationSummary:
    """Immutable aggregate summary for a pyramiding-generation run.

    Attributes:
        manager: Order manager identity used for generation.
        engine: Pyramiding engine registry key used for generation.
        symbols_discovered: Unique symbols discovered from trade-management storage.
        symbols_processed: Unique symbols for which generation was attempted.
        timeframes_processed: Unique timeframes attempted.
        successful_tasks: Count of succeeded partition tasks.
        failed_tasks: Count of failed partition tasks.
        skipped_tasks: Count of skipped existing partitions.
        rows_generated: Sum of output rows across successes.
        ready_to_add: Sum of READY_TO_ADD rows across successes.
        not_eligible: Sum of NOT_ELIGIBLE rows across successes.
        max_adds_reached: Sum of MAX_ADDS_REACHED rows across successes.
        duration_seconds: Wall-clock generation duration.
        output_directory: Pyramiding-tier output directory.
        failed_task_labels: Deterministic failed-task labels for reporting.
    """

    manager: str
    engine: str
    symbols_discovered: int
    symbols_processed: int
    timeframes_processed: int
    successful_tasks: int
    failed_tasks: int
    skipped_tasks: int
    rows_generated: int
    ready_to_add: int
    not_eligible: int
    max_adds_reached: int
    duration_seconds: float
    output_directory: Path
    failed_task_labels: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    """Create the pyramiding-generation argument parser.

    Returns:
        Configured ``ArgumentParser`` for pyramiding-generation flags.
    """
    parser = argparse.ArgumentParser(
        prog="cqros-generate-pyramiding",
        description=(
            "Generate CQROS pyramiding datasets from discovered trade-management "
            "partitions and an injected pyramiding engine."
        ),
    )
    parser.add_argument(
        "--manager",
        dest="manager",
        default=_DEFAULT_MANAGER,
        metavar="NAME",
        help=(
            "Order manager identity used for discovery and pyramiding lineage "
            f"(default: {_DEFAULT_MANAGER})."
        ),
    )
    parser.add_argument(
        "--engine",
        dest="engine",
        default=_DEFAULT_ENGINE,
        metavar="NAME",
        help=f"Pyramiding engine registry key (default: {_DEFAULT_ENGINE}).",
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
        help="Regenerate pyramiding partitions that already exist.",
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
    return parser


def build_options(args: argparse.Namespace) -> PyramidingGenerationOptions:
    """Map parsed CLI arguments onto ``PyramidingGenerationOptions``.

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

    return PyramidingGenerationOptions(
        storage_root=Path(DEFAULT_STORAGE_ROOT),
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


def build_default_engine() -> PyramidingEngine:
    """Compose the default production pyramiding engine for the CLI.

    Returns:
        ``SimplePyramidingEngine`` instance.
    """
    return SimplePyramidingEngine()


def build_registry(
    *,
    engines: Mapping[str, PyramidingEngine] | None = None,
) -> PyramidingRegistry:
    """Compose a registry with default or injected pyramiding engine implementations.

    Args:
        engines: Optional mapping of registry names to engine instances.
            When ``None``, registers ``SimplePyramidingEngine`` under
            ``simple``.

    Returns:
        Fully populated ``PyramidingRegistry``.
    """
    registry = PyramidingRegistry()
    if engines is None:
        registry.register(_DEFAULT_ENGINE, build_default_engine())
    else:
        for name, engine in engines.items():
            registry.register(name, engine)
    return registry


def build_pyramiding_pipeline(
    options: PyramidingGenerationOptions,
    *,
    engine_registry: PyramidingRegistry | None = None,
) -> PyramidingPipeline:
    """Compose ``PyramidingPipeline`` from injected engine registry dependencies.

    Args:
        options: Immutable generation options providing the engine name.
        engine_registry: Optional engine registry. When ``None``, a default
            registry containing ``SimplePyramidingEngine`` is built.

    Returns:
        Fully wired ``PyramidingPipeline``.
    """
    if engine_registry is None:
        engine_registry = build_registry()
    elif options.engine == _DEFAULT_ENGINE and not engine_registry.exists(options.engine):
        engine_registry.register(options.engine, build_default_engine())
    return PyramidingPipeline(engine_registry)


def discover_work(
    trade_management_repository: TradeManagementRepository,
    options: PyramidingGenerationOptions,
) -> tuple[DiscoveredWorkItem, ...]:
    """Discover pyramiding-ready trade-management partitions matching CLI filters.

    Only trade-management partitions that exist are scheduled. Missing
    trade-management partitions are never invented. Matching position,
    accounting, portfolio-risk, and processed OHLCV partitions are validated
    at generation time; missing dependencies fail the individual task.

    Args:
        trade_management_repository: Trade-management repository providing
            discovery APIs.
        options: CLI filters for manager, symbol, timeframe, and year.

    Returns:
        Deterministically ordered discovered work items.
    """
    partitions = trade_management_repository.discover_partitions(
        managers=(options.manager,),
        symbols=options.symbols,
        timeframes=options.timeframes,
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    return _group_partitions(partitions, year_filter=options.years)


def format_summary(summary: PyramidingGenerationSummary) -> str:
    """Render a deterministic pyramiding-generation summary report.

    Args:
        summary: Aggregate generation summary.

    Returns:
        Multi-line summary text suitable for stdout.
    """
    lines = [
        "=====================================",
        "CQROS Pyramiding Generation Summary",
        "=====================================",
        "",
        f"Manager: {summary.manager}",
        f"Engine: {summary.engine}",
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
        f"READY_TO_ADD: {summary.ready_to_add}",
        f"NOT_ELIGIBLE: {summary.not_eligible}",
        f"MAX_ADDS_REACHED: {summary.max_adds_reached}",
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
    """Run the pyramiding-generation CLI.

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
        portfolio_risk_repository = PortfolioRiskRepository(layout, datastore)
        processed_market_data_repository = ProcessedMarketDataRepository(layout, datastore)
        trade_management_repository = TradeManagementRepository(layout, datastore)
        pyramiding_repository = PyramidingRepository(layout, datastore)
        pipeline = build_pyramiding_pipeline(options)
        work = discover_work(trade_management_repository, options)
        summary = await run_generation(
            pipeline=pipeline,
            layout=layout,
            datastore=datastore,
            accounting_repository=accounting_repository,
            position_repository=position_repository,
            portfolio_risk_repository=portfolio_risk_repository,
            processed_market_data_repository=processed_market_data_repository,
            trade_management_repository=trade_management_repository,
            pyramiding_repository=pyramiding_repository,
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
    pipeline: PyramidingPipeline,
    layout: StorageLayout,
    datastore: ParquetStore,
    accounting_repository: AccountingRepository,
    position_repository: PositionRepository,
    portfolio_risk_repository: PortfolioRiskRepository,
    processed_market_data_repository: ProcessedMarketDataRepository,
    trade_management_repository: TradeManagementRepository,
    pyramiding_repository: PyramidingRepository,
    options: PyramidingGenerationOptions,
    work: Sequence[DiscoveredWorkItem],
) -> PyramidingGenerationSummary:
    """Execute discovered work through a bounded symbol worker pool.

    Args:
        pipeline: Injected pyramiding pipeline.
        layout: Storage layout for processed OHLCV existence checks.
        datastore: Datastore for processed OHLCV existence checks.
        accounting_repository: Accounting partition repository.
        position_repository: Position partition repository.
        portfolio_risk_repository: Portfolio-risk partition repository.
        processed_market_data_repository: Processed market-data repository.
        trade_management_repository: Trade-management partition repository.
        pyramiding_repository: Pyramiding partition repository.
        options: Immutable generation options.
        work: Discovered work items.

    Returns:
        Aggregate immutable summary.
    """
    started = time.perf_counter()
    output_directory = options.storage_root / STORAGE_DIR_PYRAMIDING

    if len(work) == 0:
        return PyramidingGenerationSummary(
            manager=options.manager,
            engine=options.engine,
            symbols_discovered=0,
            symbols_processed=0,
            timeframes_processed=0,
            successful_tasks=0,
            failed_tasks=0,
            skipped_tasks=0,
            rows_generated=0,
            ready_to_add=0,
            not_eligible=0,
            max_adds_reached=0,
            duration_seconds=time.perf_counter() - started,
            output_directory=output_directory,
            failed_task_labels=(),
        )

    work_by_symbol = _group_work_by_symbol(work)
    results = await _run_worker_pool(
        pipeline=pipeline,
        layout=layout,
        datastore=datastore,
        accounting_repository=accounting_repository,
        position_repository=position_repository,
        portfolio_risk_repository=portfolio_risk_repository,
        processed_market_data_repository=processed_market_data_repository,
        trade_management_repository=trade_management_repository,
        pyramiding_repository=pyramiding_repository,
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
    partitions: Sequence[TradeManagementPartitionRef],
    *,
    year_filter: tuple[int, ...] | None,
) -> tuple[DiscoveredWorkItem, ...]:
    """Group trade-management year partitions into manager/symbol/timeframe work items."""
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
    pipeline: PyramidingPipeline,
    layout: StorageLayout,
    datastore: ParquetStore,
    accounting_repository: AccountingRepository,
    position_repository: PositionRepository,
    portfolio_risk_repository: PortfolioRiskRepository,
    processed_market_data_repository: ProcessedMarketDataRepository,
    trade_management_repository: TradeManagementRepository,
    pyramiding_repository: PyramidingRepository,
    work_by_symbol: Mapping[Symbol, Sequence[DiscoveredWorkItem]],
    worker_count: int,
    overwrite: bool,
    debug: bool,
    manager_name: str,
    engine_name: str,
    model_name: str | None,
    model_version: str | None,
) -> tuple[PyramidingTaskResult, ...]:
    """Drain symbols through a bounded asyncio worker pool."""
    symbols = tuple(work_by_symbol.keys())
    if len(symbols) == 0:
        return ()

    queue: asyncio.Queue[Symbol | None] = asyncio.Queue()
    for symbol in symbols:
        queue.put_nowait(symbol)
    for _ in range(worker_count):
        queue.put_nowait(None)

    collected: list[PyramidingTaskResult] = []
    lock = asyncio.Lock()

    async def worker() -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                results = await _generate_symbol_work(
                    pipeline=pipeline,
                    layout=layout,
                    datastore=datastore,
                    accounting_repository=accounting_repository,
                    position_repository=position_repository,
                    portfolio_risk_repository=portfolio_risk_repository,
                    processed_market_data_repository=processed_market_data_repository,
                    trade_management_repository=trade_management_repository,
                    pyramiding_repository=pyramiding_repository,
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
        asyncio.create_task(worker(), name=f"generate-pyramiding-worker-{index}")
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
    pipeline: PyramidingPipeline,
    layout: StorageLayout,
    datastore: ParquetStore,
    accounting_repository: AccountingRepository,
    position_repository: PositionRepository,
    portfolio_risk_repository: PortfolioRiskRepository,
    processed_market_data_repository: ProcessedMarketDataRepository,
    trade_management_repository: TradeManagementRepository,
    pyramiding_repository: PyramidingRepository,
    symbol: Symbol,
    work_items: Sequence[DiscoveredWorkItem],
    overwrite: bool,
    debug: bool,
    manager_name: str,
    engine_name: str,
    model_name: str | None,
    model_version: str | None,
) -> tuple[PyramidingTaskResult, ...]:
    """Generate pyramiding datasets for every discovered year for one symbol."""
    results: list[PyramidingTaskResult] = []
    for item in work_items:
        for year in item.years:
            result = await asyncio.to_thread(
                _generate_partition,
                pipeline,
                layout,
                datastore,
                accounting_repository,
                position_repository,
                portfolio_risk_repository,
                processed_market_data_repository,
                trade_management_repository,
                pyramiding_repository,
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
    pipeline: PyramidingPipeline,
    layout: StorageLayout,
    datastore: ParquetStore,
    accounting_repository: AccountingRepository,
    position_repository: PositionRepository,
    portfolio_risk_repository: PortfolioRiskRepository,
    processed_market_data_repository: ProcessedMarketDataRepository,
    trade_management_repository: TradeManagementRepository,
    pyramiding_repository: PyramidingRepository,
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
) -> PyramidingTaskResult:
    """Generate one pyramiding year partition synchronously."""
    if not overwrite and pyramiding_repository.exists(
        manager=manager_name,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=symbol,
        timeframe=timeframe,
        year=year,
    ):
        return PyramidingTaskResult(
            symbol=symbol,
            timeframe=timeframe,
            year=year,
            status="skipped",
        )

    try:
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

        if not portfolio_risk_repository.exists(
            manager=manager_name,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        ):
            raise ValidationError(
                (
                    "portfolio_risk partition missing for "
                    f"{manager_name}/{symbol}/{timeframe}/{year}"
                ),
                error_code=_ERROR_PORTFOLIO_RISK_MISSING,
                details={
                    "manager": manager_name,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "year": year,
                    "missing": "portfolio_risk",
                },
            )

        ohlcv_path = layout.processed_ohlcv_path(
            _EXCHANGE,
            _MARKET,
            symbol,
            timeframe,
            year,
        )
        if not datastore.exists(ohlcv_path):
            raise ValidationError(
                f"processed OHLCV partition missing for {symbol}/{timeframe}/{year}",
                error_code=_ERROR_MARKET_MISSING,
                details={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "year": year,
                    "missing": "ohlcv",
                },
            )

        positions = position_repository.load(
            manager=manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
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
        portfolio_risk = portfolio_risk_repository.load(
            manager=manager_name,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        )
        trade_management = trade_management_repository.load(
            manager=manager_name,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        )
        ohlcv = processed_market_data_repository.load_ohlcv(
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        )
        market_prices = _prepare_market_prices(ohlcv, symbol=symbol, timeframe=timeframe)
        output = pipeline.run(
            positions,
            filtered_accounting,
            portfolio_risk,
            trade_management,
            market_prices,
            manager=manager_name,
            engine_name=engine_name,
        )
        pyramiding_repository.save(
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
        return PyramidingTaskResult(
            symbol=symbol,
            timeframe=timeframe,
            year=year,
            status="failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
            error_code=exc.error_code if isinstance(exc, CQROSError) else None,
        )

    ready_to_add, not_eligible, max_adds_reached = _count_pyramiding_reasons(output)
    return PyramidingTaskResult(
        symbol=symbol,
        timeframe=timeframe,
        year=year,
        status="succeeded",
        rows_generated=output.height,
        ready_to_add=ready_to_add,
        not_eligible=not_eligible,
        max_adds_reached=max_adds_reached,
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


def _prepare_market_prices(
    frame: pl.DataFrame,
    *,
    symbol: Symbol,
    timeframe: Timeframe,
) -> pl.DataFrame:
    """Map processed OHLCV close and high prices to pyramiding market inputs."""
    required = (_COL_OPEN_TIME, _COL_CLOSE, _COL_HIGH)
    missing = tuple(name for name in required if name not in frame.columns)
    if missing:
        raise ValidationError(
            f"processed OHLCV missing required columns: {list(missing)}",
            error_code=_ERROR_OHLCV_COLUMNS,
            details={
                "missing_columns": missing,
                "available_columns": tuple(frame.columns),
            },
        )

    working = frame
    if _COL_SYMBOL not in working.columns:
        working = working.with_columns(pl.lit(symbol).alias(_COL_SYMBOL))
    if _COL_TIMEFRAME not in working.columns:
        working = working.with_columns(pl.lit(timeframe).alias(_COL_TIMEFRAME))
    return working.select(
        pl.col(_COL_SYMBOL),
        pl.col(_COL_TIMEFRAME),
        pl.col(_COL_OPEN_TIME),
        pl.col(_COL_CLOSE).alias(_COL_PRICE),
        pl.col(_COL_HIGH),
    )


def _count_pyramiding_reasons(frame: pl.DataFrame) -> tuple[int, int, int]:
    """Count READY_TO_ADD, NOT_ELIGIBLE, and MAX_ADDS_REACHED reason rows."""
    if frame.height == 0 or _COL_REASON not in frame.columns:
        return 0, 0, 0
    reason = pl.col(_COL_REASON)
    ready_to_add = int(frame.select((reason == _REASON_READY_TO_ADD).sum()).item())
    not_eligible = int(frame.select((reason == _REASON_NOT_ELIGIBLE).sum()).item())
    max_adds_reached = int(frame.select((reason == _REASON_MAX_ADDS_REACHED).sum()).item())
    return ready_to_add, not_eligible, max_adds_reached


def _print_progress(result: PyramidingTaskResult) -> None:
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
            "Failed pyramiding generation partition; continuing",
            extra=log_extra,
            exc_info=True,
        )
    else:
        _logger.warning(
            "Failed pyramiding generation partition; continuing",
            extra=log_extra,
        )


def _build_summary(
    *,
    options: PyramidingGenerationOptions,
    work: Sequence[DiscoveredWorkItem],
    results: Sequence[PyramidingTaskResult],
    duration_seconds: float,
    output_directory: Path,
) -> PyramidingGenerationSummary:
    """Aggregate task results into a generation report."""
    symbols_discovered = {item.symbol for item in work}
    symbols_processed: set[Symbol] = set()
    timeframes_processed: set[Timeframe] = set()
    successful_tasks = 0
    failed_tasks = 0
    skipped_tasks = 0
    rows_generated = 0
    ready_to_add = 0
    not_eligible = 0
    max_adds_reached = 0
    failed_labels: set[str] = set()

    for result in results:
        symbols_processed.add(result.symbol)
        timeframes_processed.add(result.timeframe)
        if result.status == "succeeded":
            successful_tasks += 1
            if result.rows_generated is not None:
                rows_generated += result.rows_generated
            if result.ready_to_add is not None:
                ready_to_add += result.ready_to_add
            if result.not_eligible is not None:
                not_eligible += result.not_eligible
            if result.max_adds_reached is not None:
                max_adds_reached += result.max_adds_reached
        elif result.status == "skipped":
            skipped_tasks += 1
        else:
            failed_tasks += 1
            failed_labels.add(f"{result.symbol} {result.timeframe} {result.year}")

    return PyramidingGenerationSummary(
        manager=options.manager,
        engine=options.engine,
        symbols_discovered=len(symbols_discovered),
        symbols_processed=len(symbols_processed),
        timeframes_processed=len(timeframes_processed),
        successful_tasks=successful_tasks,
        failed_tasks=failed_tasks,
        skipped_tasks=skipped_tasks,
        rows_generated=rows_generated,
        ready_to_add=ready_to_add,
        not_eligible=not_eligible,
        max_adds_reached=max_adds_reached,
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
