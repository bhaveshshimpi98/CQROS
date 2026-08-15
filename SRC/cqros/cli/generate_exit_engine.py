"""CQROS exit-engine generation CLI.

Purpose:
    Provide an argparse-based production entry point that discovers canonical
    pyramiding partitions and executes ``ExitEnginePipeline`` across the
    universe with bounded symbol concurrency, persisting exit recommendations
    through ``ExitRepository``.

Responsibilities:
    - Parse CLI arguments for exit-engine dataset generation
    - Discover available pyramiding partitions through
      ``PyramidingRepository``
    - Load matching position, accounting, portfolio-risk, trade-management, and
      pyramiding partitions for each pyramiding partition
    - Filter loaded accounting frames by optional ``--model`` / ``--version``
    - Resolve ``--engine`` through ``ExitEngineRegistry``
    - Execute ``ExitEnginePipeline`` and persist via ``ExitRepository``
    - Honor ``--overwrite``, worker concurrency, and debug logging
    - Print deterministic progress and a final summary

Dependencies:
    ``argparse``, ``asyncio``, ``polars``, ``cqros.config``, ``cqros.core``,
    ``cqros.accounting``, ``cqros.portfolio_risk``, ``cqros.positions``,
    ``cqros.trade_management``, ``cqros.pyramiding``, ``cqros.exit_engine``,
    and ``cqros.storage``.

Public API:
    ``build_parser``, ``build_options``, ``build_default_engine``,
    ``build_registry``, ``build_exit_engine_pipeline``, ``discover_work``,
    ``format_summary``, ``run_generation``, and ``main``.

Notes:
    This module is a thin composition root. It does not implement exit-engine
    logic, schema validation, or repository filesystem walks beyond calling
    repository discovery and load/save APIs. Exit-engine evaluation is
    delegated exclusively to ``ExitEnginePipeline``. Exit recommendations are
    advisory only and never execute orders. Persistence remains in the CLI
    because ``ExitEnginePipeline`` does not own a repository.
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
    STORAGE_DIR_EXIT_ENGINE,
    SUPPORTED_TIMEFRAMES,
)
from cqros.core.exceptions import CQROSError, ValidationError
from cqros.core.types import Symbol, Timeframe
from cqros.exit_engine import (
    ExitAction,
    ExitEngine,
    ExitEnginePipeline,
    ExitEngineRegistry,
    ExitReason,
    ExitRepository,
    SimpleExitEngine,
)
from cqros.portfolio_risk import PortfolioRiskRepository
from cqros.positions import PositionRepository
from cqros.pyramiding import (
    PyramidingPartitionRef,
    PyramidingRepository,
)
from cqros.storage import ParquetStore, StorageLayout
from cqros.trade_management import TradeManagementRepository

__all__ = [
    "DiscoveredWorkItem",
    "ExitEngineGenerationOptions",
    "ExitEngineGenerationSummary",
    "ExitEngineTaskResult",
    "build_default_engine",
    "build_exit_engine_pipeline",
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
_DEFAULT_MANAGER: Final[str] = "simple"
_DEFAULT_ENGINE: Final[str] = "simple"

_EXCHANGE: Final[str] = EXCHANGE_BINANCE
_MARKET: Final[str] = MARKET_USDT_PERPETUAL

_ERROR_WORKERS: Final[str] = "CLI-GENERATE-EXIT-ENGINE-001"
_ERROR_TIMEFRAME: Final[str] = "CLI-GENERATE-EXIT-ENGINE-002"
_ERROR_YEAR: Final[str] = "CLI-GENERATE-EXIT-ENGINE-003"
_ERROR_MODEL: Final[str] = "CLI-GENERATE-EXIT-ENGINE-004"
_ERROR_VERSION: Final[str] = "CLI-GENERATE-EXIT-ENGINE-005"
_ERROR_MANAGER: Final[str] = "CLI-GENERATE-EXIT-ENGINE-006"
_ERROR_ENGINE: Final[str] = "CLI-GENERATE-EXIT-ENGINE-007"
_ERROR_POSITIONS_MISSING: Final[str] = "CLI-GENERATE-EXIT-ENGINE-008"
_ERROR_PORTFOLIO_RISK_MISSING: Final[str] = "CLI-GENERATE-EXIT-ENGINE-009"
_ERROR_TRADE_MANAGEMENT_MISSING: Final[str] = "CLI-GENERATE-EXIT-ENGINE-010"
_ERROR_PYRAMIDING_MISSING: Final[str] = "CLI-GENERATE-EXIT-ENGINE-011"

_COL_EXIT_ACTION: Final[str] = "exit_action"
_COL_EXIT_REASON: Final[str] = "exit_reason"

_ACTION_HOLD: Final[str] = ExitAction.HOLD.value
_ACTION_PARTIAL_EXIT: Final[str] = ExitAction.PARTIAL_EXIT.value
_ACTION_FULL_EXIT: Final[str] = ExitAction.FULL_EXIT.value

_REASON_TAKE_PROFIT: Final[str] = ExitReason.TAKE_PROFIT.value
_REASON_TRAILING_STOP: Final[str] = ExitReason.TRAILING_STOP.value
_REASON_BREAK_EVEN: Final[str] = ExitReason.BREAK_EVEN.value
_REASON_PORTFOLIO_SHUTDOWN: Final[str] = ExitReason.PORTFOLIO_SHUTDOWN.value


@dataclass(frozen=True, slots=True)
class ExitEngineGenerationOptions:
    """Immutable CLI options for exit-engine dataset generation.

    Attributes:
        storage_root: Storage root containing ``accounting``, ``positions``,
            ``portfolio_risk``, ``trade_management``, ``pyramiding``, and
            ``exit_engine``.
        manager: Order manager identity used for discovery and exit-engine
            lineage.
        engine: Registry key of the exit engine to execute.
        model: Optional model identifier used to filter accounting rows.
        version: Optional model version used to filter accounting rows.
        symbols: Optional symbol allowlist. ``None`` discovers all.
        timeframes: Optional timeframe allowlist. ``None`` discovers all.
        years: Optional year allowlist. ``None`` discovers all.
        overwrite: When ``True``, regenerate existing exit-engine partitions.
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
    """One discovered pyramiding partition group ready for exit-engine generation.

    Attributes:
        manager: Order manager identifier of the source partitions.
        symbol: Tradeable symbol.
        timeframe: Available bar interval.
        years: Calendar years with existing pyramiding parquet partitions.
    """

    manager: str
    symbol: Symbol
    timeframe: Timeframe
    years: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ExitEngineTaskResult:
    """Immutable result for one symbol/timeframe/year generation task.

    Attributes:
        symbol: Tradeable symbol.
        timeframe: Bar interval.
        year: Calendar year of the partition.
        status: ``succeeded``, ``failed``, or ``skipped``.
        rows_generated: Output row count when succeeded.
        hold_count: Count of rows with ``exit_action=HOLD`` when succeeded.
        partial_exit_count: Count of rows with ``exit_action=PARTIAL_EXIT``
            when succeeded.
        full_exit_count: Count of rows with ``exit_action=FULL_EXIT`` when
            succeeded.
        take_profit_count: Count of rows with ``exit_reason=TAKE_PROFIT``
            when succeeded.
        trailing_stop_count: Count of rows with ``exit_reason=TRAILING_STOP``
            when succeeded.
        break_even_count: Count of rows with ``exit_reason=BREAK_EVEN`` when
            succeeded.
        portfolio_shutdown_count: Count of rows with
            ``exit_reason=PORTFOLIO_SHUTDOWN`` when succeeded.
        error_type: Exception type name when failed.
        error_message: Exception message when failed.
        error_code: Optional CQROS error code when failed.
    """

    symbol: Symbol
    timeframe: Timeframe
    year: int
    status: str
    rows_generated: int | None = None
    hold_count: int | None = None
    partial_exit_count: int | None = None
    full_exit_count: int | None = None
    take_profit_count: int | None = None
    trailing_stop_count: int | None = None
    break_even_count: int | None = None
    portfolio_shutdown_count: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class ExitEngineGenerationSummary:
    """Immutable aggregate summary for an exit-engine-generation run.

    Attributes:
        manager: Order manager identity used for generation.
        engine: Exit engine registry key used for generation.
        symbols_discovered: Unique symbols discovered from pyramiding storage.
        symbols_processed: Unique symbols for which generation was attempted.
        timeframes_processed: Unique timeframes attempted.
        successful_tasks: Count of succeeded partition tasks.
        failed_tasks: Count of failed partition tasks.
        skipped_tasks: Count of skipped existing partitions.
        rows_generated: Sum of output rows across successes.
        hold_count: Sum of HOLD rows across successes.
        partial_exit_count: Sum of PARTIAL_EXIT rows across successes.
        full_exit_count: Sum of FULL_EXIT rows across successes.
        take_profit_count: Sum of TAKE_PROFIT reason rows across successes.
        trailing_stop_count: Sum of TRAILING_STOP reason rows across successes.
        break_even_count: Sum of BREAK_EVEN reason rows across successes.
        portfolio_shutdown_count: Sum of PORTFOLIO_SHUTDOWN reason rows across
            successes.
        duration_seconds: Wall-clock generation duration.
        output_directory: Exit-engine-tier output directory.
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
    hold_count: int
    partial_exit_count: int
    full_exit_count: int
    take_profit_count: int
    trailing_stop_count: int
    break_even_count: int
    portfolio_shutdown_count: int
    duration_seconds: float
    output_directory: Path
    failed_task_labels: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    """Create the exit-engine-generation argument parser.

    Returns:
        Configured ``ArgumentParser`` for exit-engine-generation flags.
    """
    parser = argparse.ArgumentParser(
        prog="cqros-generate-exit-engine",
        description=(
            "Generate CQROS exit-engine recommendation datasets from discovered pyramiding "
            "partitions and an injected exit engine."
        ),
    )
    parser.add_argument(
        "--manager",
        dest="manager",
        default=_DEFAULT_MANAGER,
        metavar="NAME",
        help=(
            "Order manager identity used for discovery and exit-engine lineage "
            f"(default: {_DEFAULT_MANAGER})."
        ),
    )
    parser.add_argument(
        "--engine",
        dest="engine",
        default=_DEFAULT_ENGINE,
        metavar="NAME",
        help=f"Exit engine registry key (default: {_DEFAULT_ENGINE}).",
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
        help="Regenerate exit-engine partitions that already exist.",
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


def build_options(args: argparse.Namespace) -> ExitEngineGenerationOptions:
    """Map parsed CLI arguments onto ``ExitEngineGenerationOptions``.

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

    return ExitEngineGenerationOptions(
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


def build_default_engine() -> ExitEngine:
    """Compose the default production exit engine for the CLI.

    Returns:
        ``SimpleExitEngine`` instance.
    """
    return SimpleExitEngine()


def build_registry(
    *,
    engines: Mapping[str, ExitEngine] | None = None,
) -> ExitEngineRegistry:
    """Compose a registry with default or injected exit engine implementations.

    Args:
        engines: Optional mapping of registry names to engine instances.
            When ``None``, registers ``SimpleExitEngine`` under ``simple``.

    Returns:
        Fully populated ``ExitEngineRegistry``.
    """
    registry = ExitEngineRegistry()
    if engines is None:
        registry.register(_DEFAULT_ENGINE, build_default_engine())
    else:
        for name, engine in engines.items():
            registry.register(name, engine)
    return registry


def build_exit_engine_pipeline(
    options: ExitEngineGenerationOptions,
    *,
    engine_registry: ExitEngineRegistry | None = None,
) -> ExitEnginePipeline:
    """Compose ``ExitEnginePipeline`` from injected engine registry dependencies.

    Args:
        options: Immutable generation options providing the engine name.
        engine_registry: Optional engine registry. When ``None``, a default
            registry containing ``SimpleExitEngine`` is built.

    Returns:
        Fully wired ``ExitEnginePipeline``.
    """
    if engine_registry is None:
        engine_registry = build_registry()
    elif options.engine == _DEFAULT_ENGINE and not engine_registry.exists(options.engine):
        engine_registry.register(options.engine, build_default_engine())
    return ExitEnginePipeline(engine_registry)


def discover_work(
    pyramiding_repository: PyramidingRepository,
    options: ExitEngineGenerationOptions,
) -> tuple[DiscoveredWorkItem, ...]:
    """Discover exit-engine-ready pyramiding partitions matching CLI filters.

    Only pyramiding partitions that exist are scheduled. Missing pyramiding
    partitions are never invented. Matching position, accounting,
    portfolio-risk, and trade-management partitions are validated at
    generation time; missing dependencies fail the individual task.

    Args:
        pyramiding_repository: Pyramiding repository providing discovery APIs.
        options: CLI filters for manager, symbol, timeframe, and year.

    Returns:
        Deterministically ordered discovered work items.
    """
    partitions = pyramiding_repository.discover_partitions(
        managers=(options.manager,),
        symbols=options.symbols,
        timeframes=options.timeframes,
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    return _group_partitions(partitions, year_filter=options.years)


def format_summary(summary: ExitEngineGenerationSummary) -> str:
    """Render a deterministic exit-engine-generation summary report.

    Args:
        summary: Aggregate generation summary.

    Returns:
        Multi-line summary text suitable for stdout.
    """
    lines = [
        "=====================================",
        "CQROS Exit Engine Generation Summary",
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
        f"HOLD: {summary.hold_count}",
        f"PARTIAL_EXIT: {summary.partial_exit_count}",
        f"FULL_EXIT: {summary.full_exit_count}",
        "",
        f"TAKE_PROFIT: {summary.take_profit_count}",
        f"TRAILING_STOP: {summary.trailing_stop_count}",
        f"BREAK_EVEN: {summary.break_even_count}",
        f"PORTFOLIO_SHUTDOWN: {summary.portfolio_shutdown_count}",
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
    """Run the exit-engine-generation CLI.

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
        trade_management_repository = TradeManagementRepository(layout, datastore)
        pyramiding_repository = PyramidingRepository(layout, datastore)
        exit_repository = ExitRepository(layout, datastore)
        pipeline = build_exit_engine_pipeline(options)
        work = discover_work(pyramiding_repository, options)
        summary = await run_generation(
            pipeline=pipeline,
            accounting_repository=accounting_repository,
            position_repository=position_repository,
            portfolio_risk_repository=portfolio_risk_repository,
            trade_management_repository=trade_management_repository,
            pyramiding_repository=pyramiding_repository,
            exit_repository=exit_repository,
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
    pipeline: ExitEnginePipeline,
    accounting_repository: AccountingRepository,
    position_repository: PositionRepository,
    portfolio_risk_repository: PortfolioRiskRepository,
    trade_management_repository: TradeManagementRepository,
    pyramiding_repository: PyramidingRepository,
    exit_repository: ExitRepository,
    options: ExitEngineGenerationOptions,
    work: Sequence[DiscoveredWorkItem],
) -> ExitEngineGenerationSummary:
    """Execute discovered work through a bounded symbol worker pool.

    Args:
        pipeline: Injected exit-engine pipeline.
        accounting_repository: Accounting partition repository.
        position_repository: Position partition repository.
        portfolio_risk_repository: Portfolio-risk partition repository.
        trade_management_repository: Trade-management partition repository.
        pyramiding_repository: Pyramiding partition repository.
        exit_repository: Exit-engine partition repository.
        options: Immutable generation options.
        work: Discovered work items.

    Returns:
        Aggregate immutable summary.
    """
    started = time.perf_counter()
    output_directory = options.storage_root / STORAGE_DIR_EXIT_ENGINE

    if len(work) == 0:
        return ExitEngineGenerationSummary(
            manager=options.manager,
            engine=options.engine,
            symbols_discovered=0,
            symbols_processed=0,
            timeframes_processed=0,
            successful_tasks=0,
            failed_tasks=0,
            skipped_tasks=0,
            rows_generated=0,
            hold_count=0,
            partial_exit_count=0,
            full_exit_count=0,
            take_profit_count=0,
            trailing_stop_count=0,
            break_even_count=0,
            portfolio_shutdown_count=0,
            duration_seconds=time.perf_counter() - started,
            output_directory=output_directory,
            failed_task_labels=(),
        )

    work_by_symbol = _group_work_by_symbol(work)
    results = await _run_worker_pool(
        pipeline=pipeline,
        accounting_repository=accounting_repository,
        position_repository=position_repository,
        portfolio_risk_repository=portfolio_risk_repository,
        trade_management_repository=trade_management_repository,
        pyramiding_repository=pyramiding_repository,
        exit_repository=exit_repository,
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
    partitions: Sequence[PyramidingPartitionRef],
    *,
    year_filter: tuple[int, ...] | None,
) -> tuple[DiscoveredWorkItem, ...]:
    """Group pyramiding year partitions into manager/symbol/timeframe work items."""
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
    pipeline: ExitEnginePipeline,
    accounting_repository: AccountingRepository,
    position_repository: PositionRepository,
    portfolio_risk_repository: PortfolioRiskRepository,
    trade_management_repository: TradeManagementRepository,
    pyramiding_repository: PyramidingRepository,
    exit_repository: ExitRepository,
    work_by_symbol: Mapping[Symbol, Sequence[DiscoveredWorkItem]],
    worker_count: int,
    overwrite: bool,
    debug: bool,
    manager_name: str,
    engine_name: str,
    model_name: str | None,
    model_version: str | None,
) -> tuple[ExitEngineTaskResult, ...]:
    """Drain symbols through a bounded asyncio worker pool."""
    symbols = tuple(work_by_symbol.keys())
    if len(symbols) == 0:
        return ()

    queue: asyncio.Queue[Symbol | None] = asyncio.Queue()
    for symbol in symbols:
        queue.put_nowait(symbol)
    for _ in range(worker_count):
        queue.put_nowait(None)

    collected: list[ExitEngineTaskResult] = []
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
                    portfolio_risk_repository=portfolio_risk_repository,
                    trade_management_repository=trade_management_repository,
                    pyramiding_repository=pyramiding_repository,
                    exit_repository=exit_repository,
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
        asyncio.create_task(worker(), name=f"generate-exit-engine-worker-{index}")
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
    pipeline: ExitEnginePipeline,
    accounting_repository: AccountingRepository,
    position_repository: PositionRepository,
    portfolio_risk_repository: PortfolioRiskRepository,
    trade_management_repository: TradeManagementRepository,
    pyramiding_repository: PyramidingRepository,
    exit_repository: ExitRepository,
    symbol: Symbol,
    work_items: Sequence[DiscoveredWorkItem],
    overwrite: bool,
    debug: bool,
    manager_name: str,
    engine_name: str,
    model_name: str | None,
    model_version: str | None,
) -> tuple[ExitEngineTaskResult, ...]:
    """Generate exit-engine datasets for every discovered year for one symbol."""
    results: list[ExitEngineTaskResult] = []
    for item in work_items:
        for year in item.years:
            result = await asyncio.to_thread(
                _generate_partition,
                pipeline,
                accounting_repository,
                position_repository,
                portfolio_risk_repository,
                trade_management_repository,
                pyramiding_repository,
                exit_repository,
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
    pipeline: ExitEnginePipeline,
    accounting_repository: AccountingRepository,
    position_repository: PositionRepository,
    portfolio_risk_repository: PortfolioRiskRepository,
    trade_management_repository: TradeManagementRepository,
    pyramiding_repository: PyramidingRepository,
    exit_repository: ExitRepository,
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
) -> ExitEngineTaskResult:
    """Generate one exit-engine year partition synchronously."""
    if not overwrite and exit_repository.exists(
        manager=manager_name,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=symbol,
        timeframe=timeframe,
        year=year,
    ):
        return ExitEngineTaskResult(
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

        if not trade_management_repository.exists(
            manager=manager_name,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        ):
            raise ValidationError(
                (
                    "trade_management partition missing for "
                    f"{manager_name}/{symbol}/{timeframe}/{year}"
                ),
                error_code=_ERROR_TRADE_MANAGEMENT_MISSING,
                details={
                    "manager": manager_name,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "year": year,
                    "missing": "trade_management",
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
        pyramiding = pyramiding_repository.load(
            manager=manager_name,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        )
        output = pipeline.run(
            positions,
            filtered_accounting,
            portfolio_risk,
            trade_management,
            pyramiding,
            manager=manager_name,
            engine_name=engine_name,
        )
        exit_repository.save(
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
        return ExitEngineTaskResult(
            symbol=symbol,
            timeframe=timeframe,
            year=year,
            status="failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
            error_code=exc.error_code if isinstance(exc, CQROSError) else None,
        )

    (
        hold_count,
        partial_exit_count,
        full_exit_count,
        take_profit_count,
        trailing_stop_count,
        break_even_count,
        portfolio_shutdown_count,
    ) = _count_exit_stats(output)
    return ExitEngineTaskResult(
        symbol=symbol,
        timeframe=timeframe,
        year=year,
        status="succeeded",
        rows_generated=output.height,
        hold_count=hold_count,
        partial_exit_count=partial_exit_count,
        full_exit_count=full_exit_count,
        take_profit_count=take_profit_count,
        trailing_stop_count=trailing_stop_count,
        break_even_count=break_even_count,
        portfolio_shutdown_count=portfolio_shutdown_count,
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


def _count_exit_stats(
    frame: pl.DataFrame,
) -> tuple[int, int, int, int, int, int, int]:
    """Count exit-action and exit-reason rows for the generation summary.

    Args:
        frame: Finalized exit-engine output DataFrame.

    Returns:
        A 7-tuple of ``(hold, partial_exit, full_exit, take_profit,
        trailing_stop, break_even, portfolio_shutdown)`` counts.
    """
    if frame.height == 0:
        return 0, 0, 0, 0, 0, 0, 0

    hold = 0
    partial_exit = 0
    full_exit = 0
    take_profit = 0
    trailing_stop = 0
    break_even = 0
    portfolio_shutdown = 0

    if _COL_EXIT_ACTION in frame.columns:
        action = pl.col(_COL_EXIT_ACTION)
        hold = int(frame.select((action == _ACTION_HOLD).sum()).item())
        partial_exit = int(frame.select((action == _ACTION_PARTIAL_EXIT).sum()).item())
        full_exit = int(frame.select((action == _ACTION_FULL_EXIT).sum()).item())

    if _COL_EXIT_REASON in frame.columns:
        reason = pl.col(_COL_EXIT_REASON)
        take_profit = int(frame.select((reason == _REASON_TAKE_PROFIT).sum()).item())
        trailing_stop = int(frame.select((reason == _REASON_TRAILING_STOP).sum()).item())
        break_even = int(frame.select((reason == _REASON_BREAK_EVEN).sum()).item())
        portfolio_shutdown = int(frame.select((reason == _REASON_PORTFOLIO_SHUTDOWN).sum()).item())

    return hold, partial_exit, full_exit, take_profit, trailing_stop, break_even, portfolio_shutdown


def _print_progress(result: ExitEngineTaskResult) -> None:
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
            "Failed exit-engine generation partition; continuing",
            extra=log_extra,
            exc_info=True,
        )
    else:
        _logger.warning(
            "Failed exit-engine generation partition; continuing",
            extra=log_extra,
        )


def _build_summary(
    *,
    options: ExitEngineGenerationOptions,
    work: Sequence[DiscoveredWorkItem],
    results: Sequence[ExitEngineTaskResult],
    duration_seconds: float,
    output_directory: Path,
) -> ExitEngineGenerationSummary:
    """Aggregate task results into a generation report."""
    symbols_discovered = {item.symbol for item in work}
    symbols_processed: set[Symbol] = set()
    timeframes_processed: set[Timeframe] = set()
    successful_tasks = 0
    failed_tasks = 0
    skipped_tasks = 0
    rows_generated = 0
    hold_count = 0
    partial_exit_count = 0
    full_exit_count = 0
    take_profit_count = 0
    trailing_stop_count = 0
    break_even_count = 0
    portfolio_shutdown_count = 0
    failed_labels: set[str] = set()

    for result in results:
        symbols_processed.add(result.symbol)
        timeframes_processed.add(result.timeframe)
        if result.status == "succeeded":
            successful_tasks += 1
            if result.rows_generated is not None:
                rows_generated += result.rows_generated
            if result.hold_count is not None:
                hold_count += result.hold_count
            if result.partial_exit_count is not None:
                partial_exit_count += result.partial_exit_count
            if result.full_exit_count is not None:
                full_exit_count += result.full_exit_count
            if result.take_profit_count is not None:
                take_profit_count += result.take_profit_count
            if result.trailing_stop_count is not None:
                trailing_stop_count += result.trailing_stop_count
            if result.break_even_count is not None:
                break_even_count += result.break_even_count
            if result.portfolio_shutdown_count is not None:
                portfolio_shutdown_count += result.portfolio_shutdown_count
        elif result.status == "skipped":
            skipped_tasks += 1
        else:
            failed_tasks += 1
            failed_labels.add(f"{result.symbol} {result.timeframe} {result.year}")

    return ExitEngineGenerationSummary(
        manager=options.manager,
        engine=options.engine,
        symbols_discovered=len(symbols_discovered),
        symbols_processed=len(symbols_processed),
        timeframes_processed=len(timeframes_processed),
        successful_tasks=successful_tasks,
        failed_tasks=failed_tasks,
        skipped_tasks=skipped_tasks,
        rows_generated=rows_generated,
        hold_count=hold_count,
        partial_exit_count=partial_exit_count,
        full_exit_count=full_exit_count,
        take_profit_count=take_profit_count,
        trailing_stop_count=trailing_stop_count,
        break_even_count=break_even_count,
        portfolio_shutdown_count=portfolio_shutdown_count,
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
