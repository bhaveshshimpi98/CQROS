"""CQROS risk-decision generation CLI.

Purpose:
    Provide an argparse-based production entry point that discovers portfolio
    partitions and executes ``RiskPipeline`` across the universe with bounded
    symbol concurrency, persisting outputs through ``RiskRepository``.

Responsibilities:
    - Parse CLI arguments for risk-decision dataset generation
    - Discover available portfolio partitions
    - Filter loaded portfolio frames by optional ``--model`` / ``--version``
    - Resolve ``--policy`` through ``RiskPolicyRegistry``
    - Execute ``RiskPipeline`` and persist via ``RiskRepository``
    - Honor ``--overwrite``, worker concurrency, and debug logging
    - Print deterministic progress and a final summary

Dependencies:
    ``argparse``, ``asyncio``, ``polars``, ``cqros.config``, ``cqros.core``,
    ``cqros.risk``, and ``cqros.storage``.

Public API:
    ``build_parser``, ``build_options``, ``build_default_policy``,
    ``build_policy_registry``, ``build_risk_pipeline``, ``discover_work``,
    ``format_summary``, ``run_generation``, and ``main``.

Notes:
    This module is a thin composition root. It does not implement risk
    calculations, policy semantics, schema validation, or repository
    filesystem walks beyond calling repository discovery and load/save APIs.
    Risk evaluation is delegated exclusively to ``RiskPipeline``. Persistence
    remains in the CLI because ``RiskPipeline`` does not own a repository.
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
    STORAGE_DIR_RISKS,
    SUPPORTED_TIMEFRAMES,
)
from cqros.core.exceptions import CQROSError, ValidationError
from cqros.core.types import Symbol, Timeframe
from cqros.risk import (
    FixedRiskPolicy,
    RiskManager,
    RiskPipeline,
    RiskPolicy,
    RiskPolicyRegistry,
)
from cqros.storage import (
    ParquetStore,
    PortfolioPartitionRef,
    PortfolioRepository,
    RiskRepository,
    StorageLayout,
)

__all__ = [
    "DiscoveredWorkItem",
    "RiskGenerationOptions",
    "RiskGenerationSummary",
    "RiskTaskResult",
    "build_default_policy",
    "build_options",
    "build_parser",
    "build_policy_registry",
    "build_risk_pipeline",
    "discover_work",
    "format_summary",
    "main",
    "run_generation",
]

_logger = logging.getLogger(__name__)

_EXIT_SUCCESS: Final[int] = 0
_EXIT_FAILURE: Final[int] = 1

_DEFAULT_WORKER_COUNT: Final[int] = ResearchConfig().worker_count
_DEFAULT_POLICY: Final[str] = RiskPolicy.FIXED_RISK.value

_EXCHANGE: Final[str] = EXCHANGE_BINANCE
_MARKET: Final[str] = MARKET_USDT_PERPETUAL

_ERROR_WORKERS: Final[str] = "CLI-GENERATE-RISK-001"
_ERROR_TIMEFRAME: Final[str] = "CLI-GENERATE-RISK-002"
_ERROR_YEAR: Final[str] = "CLI-GENERATE-RISK-003"
_ERROR_MODEL: Final[str] = "CLI-GENERATE-RISK-004"
_ERROR_VERSION: Final[str] = "CLI-GENERATE-RISK-005"
_ERROR_POLICY: Final[str] = "CLI-GENERATE-RISK-006"


@dataclass(frozen=True, slots=True)
class RiskGenerationOptions:
    """Immutable CLI options for risk-decision dataset generation.

    Attributes:
        storage_root: Storage root containing ``portfolios`` and ``risks``.
        policy: Registry key of the risk manager to execute.
        model: Optional model identifier used to filter portfolio rows.
        version: Optional model version used to filter portfolio rows.
        symbols: Optional symbol allowlist. ``None`` discovers all.
        timeframes: Optional timeframe allowlist. ``None`` discovers all.
        years: Optional year allowlist. ``None`` discovers all.
        overwrite: When ``True``, regenerate existing risk partitions.
        workers: Maximum concurrent symbols.
        verbose: When ``True``, enable INFO logging.
        debug: When ``True``, enable DEBUG logging and log full failure
            tracebacks.
    """

    storage_root: Path
    policy: str
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
    """One discovered portfolio partition group ready for risk generation.

    Attributes:
        symbol: Tradeable symbol.
        timeframe: Available bar interval.
        years: Calendar years with existing portfolio parquet partitions.
    """

    symbol: Symbol
    timeframe: Timeframe
    years: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RiskTaskResult:
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
class RiskGenerationSummary:
    """Immutable aggregate summary for a risk-generation run.

    Attributes:
        policy: Risk policy registry key used for generation.
        version: Optional model version used for generation.
        symbols_discovered: Unique symbols discovered from portfolio storage.
        symbols_processed: Unique symbols for which generation was attempted.
        timeframes_processed: Unique timeframes attempted.
        successful_tasks: Count of succeeded partition tasks.
        failed_tasks: Count of failed partition tasks.
        skipped_tasks: Count of skipped existing partitions.
        rows_generated: Sum of output rows across successes.
        duration_seconds: Wall-clock generation duration.
        output_directory: Risks-tier output directory.
        failed_task_labels: Deterministic failed-task labels for reporting.
    """

    policy: str
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
    """Create the risk-generation argument parser.

    Returns:
        Configured ``ArgumentParser`` for risk-generation flags.
    """
    parser = argparse.ArgumentParser(
        prog="cqros-generate-risk",
        description=(
            "Generate CQROS risk-decision datasets from discovered portfolio "
            "partitions and an injected risk policy."
        ),
    )
    parser.add_argument(
        "--policy",
        dest="policy",
        required=True,
        metavar="NAME",
        help="Risk policy registry key (for example fixed_risk).",
    )
    parser.add_argument(
        "--model",
        dest="model",
        default=None,
        metavar="NAME",
        help="Optional stable model identifier used to filter portfolio rows.",
    )
    parser.add_argument(
        "--version",
        dest="version",
        default=None,
        metavar="VERSION",
        help="Optional model version identifier used to filter portfolio rows.",
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
        help="Regenerate risk partitions that already exist.",
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


def build_options(args: argparse.Namespace) -> RiskGenerationOptions:
    """Map parsed CLI arguments onto ``RiskGenerationOptions``.

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

    policy = str(args.policy).strip()
    if policy == "":
        raise ValidationError(
            "policy must be a non-empty string",
            error_code=_ERROR_POLICY,
            details={"parameter": "policy", "value": args.policy},
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

    return RiskGenerationOptions(
        storage_root=Path(DEFAULT_STORAGE_ROOT),
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


def build_default_policy() -> RiskManager:
    """Compose the default production risk policy for the CLI.

    Returns:
        ``FixedRiskPolicy`` instance.
    """
    return FixedRiskPolicy()


def build_policy_registry(
    *,
    policies: Mapping[str, RiskManager] | None = None,
) -> RiskPolicyRegistry:
    """Compose a registry with default or injected risk-manager implementations.

    Args:
        policies: Optional mapping of registry names to risk-manager instances.
            When ``None``, registers ``FixedRiskPolicy`` under
            ``RiskPolicy.FIXED_RISK``.

    Returns:
        Fully populated ``RiskPolicyRegistry``.
    """
    registry = RiskPolicyRegistry()
    if policies is None:
        registry.register(_DEFAULT_POLICY, build_default_policy())
    else:
        registry.register_many(policies)
    return registry


def build_risk_pipeline(
    options: RiskGenerationOptions,
    *,
    policy_registry: RiskPolicyRegistry | None = None,
) -> RiskPipeline:
    """Compose ``RiskPipeline`` from injected policy registry deps.

    Args:
        options: Immutable generation options providing the policy name.
        policy_registry: Optional policy registry. When ``None``, a default
            registry containing ``FixedRiskPolicy`` is built.

    Returns:
        Fully wired ``RiskPipeline``.
    """
    if policy_registry is None:
        policy_registry = build_policy_registry()
    elif options.policy == _DEFAULT_POLICY and not policy_registry.exists(options.policy):
        policy_registry.register(options.policy, build_default_policy())
    return RiskPipeline(policy_registry)


def discover_work(
    portfolio_repository: PortfolioRepository,
    options: RiskGenerationOptions,
) -> tuple[DiscoveredWorkItem, ...]:
    """Discover risk-ready portfolio partitions matching CLI filters.

    Only portfolio partitions that exist are scheduled. Missing portfolio
    partitions are never invented. Partial risk datasets are never generated.

    Args:
        portfolio_repository: Portfolio repository providing discovery APIs.
        options: CLI filters for symbol, timeframe, and year.

    Returns:
        Deterministically ordered discovered work items.
    """
    partitions = portfolio_repository.discover_partitions(
        symbols=options.symbols,
        timeframes=options.timeframes,
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    return _group_partitions(partitions, year_filter=options.years)


def format_summary(summary: RiskGenerationSummary) -> str:
    """Render a deterministic risk-generation summary report.

    Args:
        summary: Aggregate generation summary.

    Returns:
        Multi-line summary text suitable for stdout.
    """
    version_text = summary.version if summary.version is not None else ""
    lines = [
        "=====================================",
        "CQROS Risk Generation Summary",
        "=====================================",
        "",
        f"Policy: {summary.policy}",
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
    """Run the risk-generation CLI.

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
        portfolio_repository = PortfolioRepository(layout, datastore)
        risk_repository = RiskRepository(layout, datastore)
        pipeline = build_risk_pipeline(options)
        work = discover_work(portfolio_repository, options)
        summary = await run_generation(
            pipeline=pipeline,
            portfolio_repository=portfolio_repository,
            risk_repository=risk_repository,
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
    pipeline: RiskPipeline,
    portfolio_repository: PortfolioRepository,
    risk_repository: RiskRepository,
    options: RiskGenerationOptions,
    work: Sequence[DiscoveredWorkItem],
) -> RiskGenerationSummary:
    """Execute discovered work through a bounded symbol worker pool.

    Args:
        pipeline: Injected risk pipeline.
        portfolio_repository: Portfolio partition repository.
        risk_repository: Risk partition repository.
        options: Immutable generation options.
        work: Discovered work items.

    Returns:
        Aggregate immutable summary.
    """
    started = time.perf_counter()
    output_directory = options.storage_root / STORAGE_DIR_RISKS

    if len(work) == 0:
        return RiskGenerationSummary(
            policy=options.policy,
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
        portfolio_repository=portfolio_repository,
        risk_repository=risk_repository,
        work_by_symbol=work_by_symbol,
        worker_count=options.workers,
        overwrite=options.overwrite,
        debug=options.debug,
        policy_name=options.policy,
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
    partitions: Sequence[PortfolioPartitionRef],
    *,
    year_filter: tuple[int, ...] | None,
) -> tuple[DiscoveredWorkItem, ...]:
    """Group portfolio year partitions into symbol/timeframe work items."""
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
    pipeline: RiskPipeline,
    portfolio_repository: PortfolioRepository,
    risk_repository: RiskRepository,
    work_by_symbol: Mapping[Symbol, Sequence[DiscoveredWorkItem]],
    worker_count: int,
    overwrite: bool,
    debug: bool,
    policy_name: str,
    model_name: str | None,
    model_version: str | None,
) -> tuple[RiskTaskResult, ...]:
    """Drain symbols through a bounded asyncio worker pool."""
    symbols = tuple(work_by_symbol.keys())
    if len(symbols) == 0:
        return ()

    queue: asyncio.Queue[Symbol | None] = asyncio.Queue()
    for symbol in symbols:
        queue.put_nowait(symbol)
    for _ in range(worker_count):
        queue.put_nowait(None)

    collected: list[RiskTaskResult] = []
    lock = asyncio.Lock()

    async def worker() -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                results = await _generate_symbol_work(
                    pipeline=pipeline,
                    portfolio_repository=portfolio_repository,
                    risk_repository=risk_repository,
                    symbol=item,
                    work_items=work_by_symbol[item],
                    overwrite=overwrite,
                    debug=debug,
                    policy_name=policy_name,
                    model_name=model_name,
                    model_version=model_version,
                )
                async with lock:
                    collected.extend(results)
            finally:
                queue.task_done()

    worker_tasks = [
        asyncio.create_task(worker(), name=f"generate-risk-worker-{index}")
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
    pipeline: RiskPipeline,
    portfolio_repository: PortfolioRepository,
    risk_repository: RiskRepository,
    symbol: Symbol,
    work_items: Sequence[DiscoveredWorkItem],
    overwrite: bool,
    debug: bool,
    policy_name: str,
    model_name: str | None,
    model_version: str | None,
) -> tuple[RiskTaskResult, ...]:
    """Generate risk datasets for every discovered year for one symbol."""
    results: list[RiskTaskResult] = []
    for item in work_items:
        for year in item.years:
            result = await asyncio.to_thread(
                _generate_partition,
                pipeline,
                portfolio_repository,
                risk_repository,
                symbol=symbol,
                timeframe=item.timeframe,
                year=year,
                overwrite=overwrite,
                debug=debug,
                policy_name=policy_name,
                model_name=model_name,
                model_version=model_version,
            )
            _print_progress(result)
            results.append(result)
    return tuple(results)


def _generate_partition(
    pipeline: RiskPipeline,
    portfolio_repository: PortfolioRepository,
    risk_repository: RiskRepository,
    *,
    symbol: Symbol,
    timeframe: Timeframe,
    year: int,
    overwrite: bool,
    debug: bool,
    policy_name: str,
    model_name: str | None,
    model_version: str | None,
) -> RiskTaskResult:
    """Generate one risk year partition synchronously."""
    if not overwrite and risk_repository.exists(
        policy=policy_name,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=symbol,
        timeframe=timeframe,
        year=year,
    ):
        return RiskTaskResult(
            symbol=symbol,
            timeframe=timeframe,
            year=year,
            status="skipped",
        )

    try:
        portfolios = portfolio_repository.load(
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        )
        filtered = _filter_portfolios_for_model(
            portfolios,
            model_name=model_name,
            model_version=model_version,
        )
        output = pipeline.run(policy_name, filtered)
        risk_repository.save(
            output,
            policy=policy_name,
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
        return RiskTaskResult(
            symbol=symbol,
            timeframe=timeframe,
            year=year,
            status="failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
            error_code=exc.error_code if isinstance(exc, CQROSError) else None,
        )

    return RiskTaskResult(
        symbol=symbol,
        timeframe=timeframe,
        year=year,
        status="succeeded",
        rows_generated=output.height,
    )


def _filter_portfolios_for_model(
    portfolios: pl.DataFrame,
    *,
    model_name: str | None,
    model_version: str | None,
) -> pl.DataFrame:
    """Return portfolio rows matching optional model identity filters.

    Args:
        portfolios: Loaded portfolio partition frame.
        model_name: Optional model_name column value.
        model_version: Optional model_version column value.

    Returns:
        A new DataFrame containing only matching rows, or ``portfolios`` when
        no filters are supplied.
    """
    if model_name is None and model_version is None:
        return portfolios
    predicate = pl.lit(True)
    if model_name is not None:
        predicate = predicate & (pl.col("model_name") == model_name)
    if model_version is not None:
        predicate = predicate & (pl.col("model_version") == model_version)
    return portfolios.filter(predicate)


def _print_progress(result: RiskTaskResult) -> None:
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
            "Failed risk generation partition; continuing",
            extra=log_extra,
            exc_info=True,
        )
    else:
        _logger.warning(
            "Failed risk generation partition; continuing",
            extra=log_extra,
        )


def _build_summary(
    *,
    options: RiskGenerationOptions,
    work: Sequence[DiscoveredWorkItem],
    results: Sequence[RiskTaskResult],
    duration_seconds: float,
    output_directory: Path,
) -> RiskGenerationSummary:
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

    return RiskGenerationSummary(
        policy=options.policy,
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
