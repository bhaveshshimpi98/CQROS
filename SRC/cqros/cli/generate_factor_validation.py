"""CQROS factor-validation-generation CLI.

Purpose:
    Provide an argparse-based production entry point that discovers canonical
    Factors partitions and executes ``FactorValidationPipeline`` across
    cross-sectional ``(timeframe, year)`` panels with bounded concurrency,
    persisting factor validation metrics through
    ``FactorValidationRepository``.

Responsibilities:
    - Parse CLI arguments for factor validation dataset generation
    - Discover available Factors partitions through ``FactorsRepository``
    - Group discovered partitions into ``(manager, timeframe, year)`` panels
    - Wire ``ValidationDatasetBuilder`` with Factors and Labels repositories
    - Resolve ``--engine`` through ``FactorValidationEngineRegistry``
    - Execute ``FactorValidationPipeline`` and persist via
      ``FactorValidationRepository``
    - Honor ``--overwrite``, worker concurrency, and debug logging
    - Print deterministic progress and a final summary without letting
      console encoding failures abort generation

Dependencies:
    ``argparse``, ``asyncio``, ``polars``, ``cqros.config``, ``cqros.core``,
    ``cqros.factors``, ``cqros.factor_validation``, and ``cqros.storage``.

Public API:
    ``build_parser``, ``build_options``, ``build_default_engine``,
    ``build_registry``, ``build_factor_validation_pipeline``,
    ``discover_work``, ``format_summary``, ``run_generation``, and ``main``.

Notes:
    This module is a thin composition root. It does not implement validation
    math, Factors/Labels joins, schema validation, or repository filesystem
    walks beyond calling repository discovery and save APIs. Dataset assembly
    belongs to ``ValidationDatasetBuilder``. Metric computation is delegated
    exclusively to ``FactorValidationPipeline``. Persistence remains in the
    CLI because ``FactorValidationPipeline`` does not own an output repository.

    Scheduling is panel-based::

        for timeframe
            for year
                build full symbol panel
                validate panel

    Console rendering (progress, summary, fatal messages, debug tracebacks) is
    isolated from computation and persistence. ``UnicodeEncodeError``,
    ``OSError``, and ``BrokenPipeError`` raised while writing to stdout/stderr
    are logged at DEBUG and swallowed so asyncio workers keep running.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TextIO

import polars as pl

from cqros.config.models import ResearchConfig
from cqros.core.constants import (
    DEFAULT_STORAGE_ROOT,
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
    STORAGE_DIR_FACTOR_VALIDATION,
    SUPPORTED_TIMEFRAMES,
)
from cqros.core.exceptions import CQROSError, ValidationError
from cqros.core.types import Symbol, Timeframe
from cqros.factor_validation import (
    FactorValidationEngine,
    FactorValidationEngineRegistry,
    FactorValidationPipeline,
    FactorValidationRepository,
    FactorValidationStatus,
    SimpleFactorValidationEngine,
    ValidationDatasetBuilder,
)
from cqros.factor_validation.memory_efficient import (
    FactorValidationExecutionConfig,
    FactorValidationExecutionMode,
)
from cqros.factors import FactorPartitionRef, FactorsRepository
from cqros.storage import LabelRepository, ParquetStore, StorageLayout

__all__ = [
    "DiscoveredWorkItem",
    "FactorValidationGenerationOptions",
    "FactorValidationGenerationSummary",
    "FactorValidationTaskResult",
    "build_default_engine",
    "build_factor_validation_pipeline",
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

_ERROR_WORKERS: Final[str] = "CLI-GENERATE-FACTOR-VALIDATION-001"
_ERROR_TIMEFRAME: Final[str] = "CLI-GENERATE-FACTOR-VALIDATION-002"
_ERROR_YEAR: Final[str] = "CLI-GENERATE-FACTOR-VALIDATION-003"
_ERROR_MANAGER: Final[str] = "CLI-GENERATE-FACTOR-VALIDATION-004"
_ERROR_ENGINE: Final[str] = "CLI-GENERATE-FACTOR-VALIDATION-005"
_ERROR_FACTORS_MISSING: Final[str] = "CLI-GENERATE-FACTOR-VALIDATION-006"
_ERROR_EXECUTION_MODE: Final[str] = "CLI-GENERATE-FACTOR-VALIDATION-007"
_ERROR_FACTOR_BATCH_SIZE: Final[str] = "CLI-GENERATE-FACTOR-VALIDATION-008"

_COL_STATUS: Final[str] = "status"
_DEFAULT_EXECUTION_MODE: Final[str] = FactorValidationExecutionMode.MEMORY_EFFICIENT.value
_DEFAULT_FACTOR_BATCH_SIZE: Final[int] = 1


@dataclass(frozen=True, slots=True)
class FactorValidationGenerationOptions:
    """Immutable CLI options for factor validation dataset generation.

    Attributes:
        storage_root: Storage root containing ``factors`` and
            ``factor_validation``.
        manager: Order manager identity used for discovery and factor
            validation lineage.
        engine: Registry key of the factor validation engine to execute.
        symbols: Optional symbol allowlist applied when assembling panels.
            ``None`` includes every discovered symbol.
        timeframes: Optional timeframe allowlist. ``None`` discovers all.
        years: Optional year allowlist. ``None`` discovers all.
        overwrite: When ``True``, regenerate existing factor validation
            partitions.
        workers: Maximum concurrent panels.
        execution_mode: ``memory_efficient`` (default) or ``full_panel``.
        factor_batch_size: Factor identities per engine batch when using
            ``memory_efficient`` mode.
        verbose: When ``True``, enable INFO logging.
        debug: When ``True``, enable DEBUG logging and log full failure
            tracebacks.
    """

    storage_root: Path
    manager: str
    engine: str
    symbols: tuple[Symbol, ...] | None
    timeframes: tuple[Timeframe, ...] | None
    years: tuple[int, ...] | None
    overwrite: bool
    workers: int
    execution_mode: FactorValidationExecutionMode
    factor_batch_size: int
    verbose: bool
    debug: bool


@dataclass(frozen=True, slots=True)
class DiscoveredWorkItem:
    """One discovered Factors panel ready for validation generation.

    Attributes:
        manager: Order manager identifier of the source partitions.
        timeframe: Available bar interval.
        year: Calendar year of the panel.
        symbols: Deterministically ordered symbols contributing Factors for
            this panel key.
    """

    manager: str
    timeframe: Timeframe
    year: int
    symbols: tuple[Symbol, ...]


@dataclass(frozen=True, slots=True)
class FactorValidationTaskResult:
    """Immutable result for one timeframe/year panel generation task.

    Attributes:
        timeframe: Bar interval.
        year: Calendar year of the partition.
        symbols: Count of symbols included in the panel schedule.
        status: ``succeeded``, ``failed``, or ``skipped``.
        rows_generated: Output row count when succeeded.
        passed_rows: Count of rows with ``PASS`` status on success.
        failed_status_rows: Count of rows with ``FAIL`` status on success.
        error_type: Exception type name when failed.
        error_message: Exception message when failed.
        error_code: Optional CQROS error code when failed.
    """

    timeframe: Timeframe
    year: int
    symbols: int
    status: str
    rows_generated: int | None = None
    passed_rows: int | None = None
    failed_status_rows: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class FactorValidationGenerationSummary:
    """Immutable aggregate summary for a factor-validation-generation run.

    Attributes:
        manager: Order manager identity used for generation.
        engine: Factor validation engine registry key used for generation.
        symbols: Unique symbols scheduled across all panels.
        rows: Sum of output rows across successes.
        passed_rows: Sum of rows with ``PASS`` status across successes.
        failed_status_rows: Sum of rows with ``FAIL`` status across
            successes.
        successful_tasks: Count of succeeded panel tasks.
        failed_tasks: Count of failed panel tasks.
        skipped_tasks: Count of skipped existing partitions.
        duration_seconds: Wall-clock generation duration.
        output_directory: Factor-validation-tier output directory.
        failed_task_labels: Deterministic failed-task labels for reporting.
    """

    manager: str
    engine: str
    symbols: int
    rows: int
    passed_rows: int
    failed_status_rows: int
    successful_tasks: int
    failed_tasks: int
    skipped_tasks: int
    duration_seconds: float
    output_directory: Path
    failed_task_labels: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    """Create the factor-validation-generation argument parser.

    Returns:
        Configured ``ArgumentParser`` for factor-validation-generation flags.
    """
    parser = argparse.ArgumentParser(
        prog="cqros-generate-factor-validation",
        description=(
            "Generate CQROS factor validation metrics datasets from discovered "
            "Factors partitions and an injected factor validation engine."
        ),
    )
    parser.add_argument(
        "--manager",
        dest="manager",
        required=True,
        metavar="NAME",
        help=("Order manager identity used for discovery and factor validation " "lineage."),
    )
    parser.add_argument(
        "--engine",
        dest="engine",
        default=_DEFAULT_ENGINE,
        metavar="NAME",
        help=f"Factor validation engine registry key (default: {_DEFAULT_ENGINE}).",
    )
    parser.add_argument(
        "--symbols",
        dest="symbols",
        nargs="*",
        default=None,
        metavar="SYMBOL",
        help=(
            "Optional symbol allowlist for panel assembly (0..N values). "
            "Omit to include all discovered symbols."
        ),
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
        help="Regenerate factor validation partitions that already exist.",
    )
    parser.add_argument(
        "--workers",
        dest="workers",
        type=int,
        default=_DEFAULT_WORKER_COUNT,
        metavar="INT",
        help=f"Maximum concurrent panels (default: {_DEFAULT_WORKER_COUNT}).",
    )
    parser.add_argument(
        "--execution-mode",
        dest="execution_mode",
        choices=tuple(mode.value for mode in FactorValidationExecutionMode),
        default=_DEFAULT_EXECUTION_MODE,
        metavar="MODE",
        help=(
            "Panel execution strategy: memory_efficient (default; spill + "
            "factor-identity batches) or full_panel (legacy in-RAM concat)."
        ),
    )
    parser.add_argument(
        "--factor-batch-size",
        dest="factor_batch_size",
        type=int,
        default=_DEFAULT_FACTOR_BATCH_SIZE,
        metavar="INT",
        help=(
            "Factor identities loaded per engine batch in memory_efficient "
            f"mode (default: {_DEFAULT_FACTOR_BATCH_SIZE})."
        ),
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


def build_options(args: argparse.Namespace) -> FactorValidationGenerationOptions:
    """Map parsed CLI arguments onto ``FactorValidationGenerationOptions``.

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

    factor_batch_size = int(args.factor_batch_size)
    if factor_batch_size < 1:
        raise ValidationError(
            "factor_batch_size must be >= 1",
            error_code=_ERROR_FACTOR_BATCH_SIZE,
            details={"parameter": "factor_batch_size", "value": factor_batch_size},
        )

    try:
        execution_mode = FactorValidationExecutionMode(str(args.execution_mode).strip())
    except ValueError as exc:
        raise ValidationError(
            "execution_mode must be memory_efficient or full_panel",
            error_code=_ERROR_EXECUTION_MODE,
            details={"parameter": "execution_mode", "value": args.execution_mode},
        ) from exc

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

    storage_root = (
        Path(args.storage_root) if args.storage_root is not None else Path(DEFAULT_STORAGE_ROOT)
    )

    return FactorValidationGenerationOptions(
        storage_root=storage_root,
        manager=manager,
        engine=engine,
        symbols=_normalize_symbols(args.symbols),
        timeframes=_normalize_timeframes(args.timeframes),
        years=_normalize_years(args.years),
        overwrite=bool(args.overwrite),
        workers=workers,
        execution_mode=execution_mode,
        factor_batch_size=factor_batch_size,
        verbose=bool(args.verbose),
        debug=bool(args.debug),
    )


def build_default_engine() -> FactorValidationEngine:
    """Compose the default production factor validation engine for the CLI.

    Returns:
        ``SimpleFactorValidationEngine`` instance.
    """
    return SimpleFactorValidationEngine()


def build_registry(
    *,
    engines: Mapping[str, FactorValidationEngine] | None = None,
) -> FactorValidationEngineRegistry:
    """Compose a registry with default or injected factor validation engines.

    Args:
        engines: Optional mapping of registry names to engine instances.
            When ``None``, registers ``SimpleFactorValidationEngine`` under
            ``simple``.

    Returns:
        Fully populated ``FactorValidationEngineRegistry``.
    """
    registry = FactorValidationEngineRegistry()
    if engines is None:
        registry.register(_DEFAULT_ENGINE, build_default_engine())
    else:
        for name, engine in engines.items():
            registry.register(name, engine)
    return registry


def build_factor_validation_pipeline(
    options: FactorValidationGenerationOptions,
    *,
    builder: ValidationDatasetBuilder,
    engine_registry: FactorValidationEngineRegistry | None = None,
) -> FactorValidationPipeline:
    """Compose ``FactorValidationPipeline`` from injected collaborators.

    Args:
        options: Immutable generation options providing the engine name.
        builder: Validation dataset builder that joins Factors and Labels.
        engine_registry: Optional engine registry. When ``None``, a default
            registry containing ``SimpleFactorValidationEngine`` is built.

    Returns:
        Fully wired ``FactorValidationPipeline``.
    """
    if engine_registry is None:
        engine_registry = build_registry()
    elif options.engine == _DEFAULT_ENGINE and not engine_registry.exists(options.engine):
        engine_registry.register(options.engine, build_default_engine())
    execution_config = FactorValidationExecutionConfig(
        mode=options.execution_mode,
        factor_batch_size=options.factor_batch_size,
    )
    return FactorValidationPipeline(
        engine_registry,
        builder,
        execution_config=execution_config,
    )


def discover_work(
    factors_repository: FactorsRepository,
    options: FactorValidationGenerationOptions,
) -> tuple[DiscoveredWorkItem, ...]:
    """Discover validation-ready Factors panels matching CLI filters.

    Only Factors partitions that exist are scheduled. Missing Factors
    partitions are never invented. Partitions are grouped into
    ``(manager, timeframe, year)`` panels containing every contributing
    symbol.

    Args:
        factors_repository: Factors repository providing discovery APIs.
        options: CLI filters for manager, symbol, timeframe, and year.

    Returns:
        Deterministically ordered discovered panel work items.
    """
    partitions = factors_repository.discover_partitions(
        managers=(options.manager,),
        symbols=options.symbols,
        timeframes=options.timeframes,
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    return _group_partitions(partitions, year_filter=options.years)


def format_summary(summary: FactorValidationGenerationSummary) -> str:
    """Render a deterministic factor-validation-generation summary report.

    Args:
        summary: Aggregate generation summary.

    Returns:
        Multi-line summary text suitable for stdout.
    """
    lines = [
        "=====================================",
        "CQROS Factor Validation Generation Summary",
        "=====================================",
        "",
        f"Manager: {summary.manager}",
        f"Engine: {summary.engine}",
        "",
        f"Symbols: {summary.symbols}",
        f"Rows: {summary.rows}",
        f"Passed: {summary.passed_rows}",
        f"Failed Status: {summary.failed_status_rows}",
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
    """Run the factor-validation-generation CLI.

    Args:
        argv: Optional argument vector. When ``None``, ``sys.argv[1:]`` is used.

    Returns:
        ``0`` on completion; ``1`` when a fatal CLI error occurs or any task
        failed.
    """
    _configure_stdio_utf8()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        options = build_options(args)
        _configure_logging(verbose=options.verbose, debug=options.debug)
        layout = StorageLayout(options.storage_root)
        datastore = ParquetStore()
        factors_repository = FactorsRepository(layout, datastore)
        label_repository = LabelRepository(layout, datastore)
        factor_validation_repository = FactorValidationRepository(layout, datastore)
        builder = ValidationDatasetBuilder(factors_repository, label_repository)
        pipeline = build_factor_validation_pipeline(options, builder=builder)
        work = discover_work(factors_repository, options)
        summary = await run_generation(
            pipeline=pipeline,
            factors_repository=factors_repository,
            factor_validation_repository=factor_validation_repository,
            options=options,
            work=work,
        )
    except CQROSError as exc:
        _emit_text(str(exc), stream=sys.stderr)
        return _EXIT_FAILURE
    except Exception as exc:
        _emit_text(str(exc), stream=sys.stderr)
        return _EXIT_FAILURE

    _emit_text(format_summary(summary), end="")
    return _EXIT_SUCCESS if summary.failed_tasks == 0 else _EXIT_FAILURE


async def run_generation(
    *,
    pipeline: FactorValidationPipeline,
    factors_repository: FactorsRepository,
    factor_validation_repository: FactorValidationRepository,
    options: FactorValidationGenerationOptions,
    work: Sequence[DiscoveredWorkItem],
) -> FactorValidationGenerationSummary:
    """Execute discovered panels through a bounded worker pool.

    Args:
        pipeline: Injected factor validation pipeline.
        factors_repository: Factors partition repository.
        factor_validation_repository: Factor validation partition repository.
        options: Immutable generation options.
        work: Discovered panel work items.

    Returns:
        Aggregate immutable summary.
    """
    started = time.perf_counter()
    output_directory = options.storage_root / STORAGE_DIR_FACTOR_VALIDATION

    if len(work) == 0:
        return FactorValidationGenerationSummary(
            manager=options.manager,
            engine=options.engine,
            symbols=0,
            rows=0,
            passed_rows=0,
            failed_status_rows=0,
            successful_tasks=0,
            failed_tasks=0,
            skipped_tasks=0,
            duration_seconds=time.perf_counter() - started,
            output_directory=output_directory,
            failed_task_labels=(),
        )

    results = await _run_worker_pool(
        pipeline=pipeline,
        factors_repository=factors_repository,
        factor_validation_repository=factor_validation_repository,
        work=work,
        worker_count=options.workers,
        overwrite=options.overwrite,
        debug=options.debug,
        manager_name=options.manager,
        engine_name=options.engine,
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


def _configure_stdio_utf8() -> None:
    """Prefer UTF-8 stdout/stderr with replacement errors when supported.

    Uses ``TextIO.reconfigure`` when available. Failures are ignored so
    generation can proceed on consoles that reject reconfiguration.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            continue


def _emit_text(
    text: str,
    *,
    stream: TextIO | None = None,
    end: str = "\n",
    flush: bool = False,
) -> None:
    """Write text to a console stream without aborting generation.

    Isolates ``UnicodeEncodeError``, ``OSError``, and ``BrokenPipeError`` so
    progress and summary rendering cannot terminate asyncio workers or the
    CLI process control flow. Failures are logged at DEBUG and swallowed.

    Args:
        text: Payload to write.
        stream: Destination stream. Defaults to ``sys.stdout``.
        end: Suffix appended after ``text`` (default newline).
        flush: When ``True``, flush the stream after a successful write.
    """
    target: TextIO = sys.stdout if stream is None else stream
    try:
        print(text, end=end, file=target, flush=flush)
    except (UnicodeEncodeError, BrokenPipeError, OSError) as exc:
        # BrokenPipeError is an OSError subclass; listed explicitly for clarity.
        _logger.debug(
            "console write failed; continuing generation",
            exc_info=exc,
            extra={
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )


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
    partitions: Sequence[FactorPartitionRef],
    *,
    year_filter: tuple[int, ...] | None,
) -> tuple[DiscoveredWorkItem, ...]:
    """Group Factors year partitions into manager/timeframe/year panels."""
    year_allowlist = set(year_filter) if year_filter is not None else None
    grouped: dict[tuple[str, str, int], list[Symbol]] = {}
    for partition in partitions:
        if year_allowlist is not None and partition.year not in year_allowlist:
            continue
        key = (partition.manager, partition.timeframe, partition.year)
        symbols = grouped.setdefault(key, [])
        if partition.symbol not in symbols:
            symbols.append(partition.symbol)

    items: list[DiscoveredWorkItem] = []
    for (manager, timeframe, year), symbols in grouped.items():
        items.append(
            DiscoveredWorkItem(
                manager=manager,
                timeframe=timeframe,
                year=year,
                symbols=tuple(sorted(symbols)),
            )
        )
    return tuple(
        sorted(
            items,
            key=lambda item: (item.manager, item.timeframe, item.year),
        )
    )


async def _run_worker_pool(
    *,
    pipeline: FactorValidationPipeline,
    factors_repository: FactorsRepository,
    factor_validation_repository: FactorValidationRepository,
    work: Sequence[DiscoveredWorkItem],
    worker_count: int,
    overwrite: bool,
    debug: bool,
    manager_name: str,
    engine_name: str,
) -> tuple[FactorValidationTaskResult, ...]:
    """Drain panels through a bounded asyncio worker pool."""
    if len(work) == 0:
        return ()

    queue: asyncio.Queue[DiscoveredWorkItem | None] = asyncio.Queue()
    for item in work:
        queue.put_nowait(item)
    for _ in range(worker_count):
        queue.put_nowait(None)

    collected: list[FactorValidationTaskResult] = []
    lock = asyncio.Lock()

    async def worker() -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                result = await asyncio.to_thread(
                    _generate_panel,
                    pipeline,
                    factors_repository,
                    factor_validation_repository,
                    item=item,
                    overwrite=overwrite,
                    debug=debug,
                    manager_name=manager_name,
                    engine_name=engine_name,
                )
                # Record the task result before any console rendering so progress
                # failures cannot drop completed computation/persistence outcomes.
                async with lock:
                    collected.append(result)
                _print_progress(result)
            finally:
                queue.task_done()

    worker_tasks = [
        asyncio.create_task(worker(), name=f"generate-factor-validation-worker-{index}")
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
            key=lambda result: (result.timeframe, result.year),
        )
    )


def _generate_panel(
    pipeline: FactorValidationPipeline,
    factors_repository: FactorsRepository,
    factor_validation_repository: FactorValidationRepository,
    *,
    item: DiscoveredWorkItem,
    overwrite: bool,
    debug: bool,
    manager_name: str,
    engine_name: str,
) -> FactorValidationTaskResult:
    """Generate one factor validation timeframe/year panel synchronously."""
    symbol_count = len(item.symbols)
    if not overwrite and factor_validation_repository.exists(
        manager=manager_name,
        exchange=_EXCHANGE,
        market=_MARKET,
        timeframe=item.timeframe,
        year=item.year,
    ):
        return FactorValidationTaskResult(
            timeframe=item.timeframe,
            year=item.year,
            symbols=symbol_count,
            status="skipped",
        )

    try:
        missing_symbols = [
            symbol
            for symbol in item.symbols
            if not factors_repository.exists(
                manager=item.manager,
                exchange=_EXCHANGE,
                market=_MARKET,
                symbol=symbol,
                timeframe=item.timeframe,
                year=item.year,
            )
        ]
        if missing_symbols:
            raise ValidationError(
                (
                    "factors partitions missing for "
                    f"{item.manager}/{item.timeframe}/{item.year}: "
                    f"{','.join(missing_symbols)}"
                ),
                error_code=_ERROR_FACTORS_MISSING,
                details={
                    "manager": item.manager,
                    "timeframe": item.timeframe,
                    "year": item.year,
                    "missing_symbols": tuple(missing_symbols),
                    "missing": "factors",
                },
            )

        output = pipeline.run(
            engine_name,
            manager=item.manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            timeframe=item.timeframe,
            year=item.year,
            symbols=item.symbols,
        )
        factor_validation_repository.save(
            output,
            manager=manager_name,
            exchange=_EXCHANGE,
            market=_MARKET,
            timeframe=item.timeframe,
            year=item.year,
        )
    except Exception as exc:
        # Temporary diagnosis: surface the full traceback before the failure is
        # reduced to a failed-task result (progress/summary omit the stack).
        if debug:
            _emit_debug_traceback()
        _log_partition_failure(
            timeframe=item.timeframe,
            year=item.year,
            exc=exc,
            debug=debug,
        )
        return FactorValidationTaskResult(
            timeframe=item.timeframe,
            year=item.year,
            symbols=symbol_count,
            status="failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
            error_code=exc.error_code if isinstance(exc, CQROSError) else None,
        )

    rows_generated, passed_rows, failed_status_rows = _extract_partition_stats(output)
    return FactorValidationTaskResult(
        timeframe=item.timeframe,
        year=item.year,
        symbols=symbol_count,
        status="succeeded",
        rows_generated=rows_generated,
        passed_rows=passed_rows,
        failed_status_rows=failed_status_rows,
    )


def _extract_partition_stats(frame: pl.DataFrame) -> tuple[int, int, int]:
    """Extract row count and validation-status counts from one metrics frame.

    Args:
        frame: Finalized factor validation output DataFrame.

    Returns:
        A 3-tuple of ``(rows, passed_rows, failed_status_rows)``.
    """
    if frame.height == 0:
        return 0, 0, 0

    passed_rows = int(
        frame.select((pl.col(_COL_STATUS) == FactorValidationStatus.PASS.value).sum()).item()
    )
    failed_status_rows = int(
        frame.select((pl.col(_COL_STATUS) == FactorValidationStatus.FAIL.value).sum()).item()
    )
    return frame.height, passed_rows, failed_status_rows


def _emit_debug_traceback() -> None:
    """Emit the current exception traceback without aborting generation."""
    try:
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
    except (UnicodeEncodeError, OSError, BrokenPipeError) as exc:
        _logger.debug(
            "debug traceback console write failed; continuing generation",
            exc_info=exc,
            extra={
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )


def _print_progress(result: FactorValidationTaskResult) -> None:
    """Print a deterministic one-line progress record for a task result.

    Progress rendering is best-effort. Console failures never propagate so
    asyncio task execution remains independent of stdout availability.
    """
    label = f"{result.timeframe} {result.year} symbols={result.symbols}"
    if result.status == "succeeded":
        rows = result.rows_generated if result.rows_generated is not None else 0
        message = f"OK {label} rows={rows}"
    elif result.status == "skipped":
        message = f"SKIP {label}"
    else:
        error_type = result.error_type if result.error_type is not None else "Exception"
        message = f"FAIL {label} {error_type}"
    _emit_text(message, flush=True)


def _log_partition_failure(
    *,
    timeframe: Timeframe,
    year: int,
    exc: BaseException,
    debug: bool,
) -> None:
    """Log a panel generation failure without aborting the run."""
    log_extra = {
        "timeframe": timeframe,
        "year": year,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "error_code": exc.error_code if isinstance(exc, CQROSError) else None,
    }
    if debug:
        # Traceback already printed to stdout in ``_generate_panel``.
        _logger.error(
            "Failed factor validation generation panel; continuing",
            extra=log_extra,
        )
    else:
        _logger.warning(
            "Failed factor validation generation panel; continuing",
            extra=log_extra,
        )


def _build_summary(
    *,
    options: FactorValidationGenerationOptions,
    work: Sequence[DiscoveredWorkItem],
    results: Sequence[FactorValidationTaskResult],
    duration_seconds: float,
    output_directory: Path,
) -> FactorValidationGenerationSummary:
    """Aggregate task results into a generation report."""
    symbols_discovered = {symbol for item in work for symbol in item.symbols}
    successful_tasks = 0
    failed_tasks = 0
    skipped_tasks = 0
    rows = 0
    passed_rows = 0
    failed_status_rows = 0
    failed_labels: set[str] = set()

    for result in results:
        if result.status == "succeeded":
            successful_tasks += 1
            if result.rows_generated is not None:
                rows += result.rows_generated
            if result.passed_rows is not None:
                passed_rows += result.passed_rows
            if result.failed_status_rows is not None:
                failed_status_rows += result.failed_status_rows
        elif result.status == "skipped":
            skipped_tasks += 1
        else:
            failed_tasks += 1
            failed_labels.add(f"{result.timeframe} {result.year}")

    return FactorValidationGenerationSummary(
        manager=options.manager,
        engine=options.engine,
        symbols=len(symbols_discovered),
        rows=rows,
        passed_rows=passed_rows,
        failed_status_rows=failed_status_rows,
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
