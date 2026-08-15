"""CQROS factor-selection-generation CLI.

Purpose:
    Provide an argparse-based production entry point that discovers canonical
    Factor Validation panels and executes ``FactorSelectionPipeline`` across
    timeframe/year panels with bounded concurrency, persisting factor
    selection metrics through ``FactorSelectionRepository``.

Responsibilities:
    - Parse CLI arguments for factor selection dataset generation
    - Discover available Factor Validation partitions through
      ``FactorValidationRepository``
    - Load matching Factor Validation panels for each discovered partition
    - Resolve ``--engine`` through ``FactorSelectionEngineRegistry``
    - Execute ``FactorSelectionPipeline`` and persist via
      ``FactorSelectionRepository``
    - Honor ``--overwrite``, worker concurrency, and debug logging
    - Print deterministic progress and a final summary

Dependencies:
    ``argparse``, ``asyncio``, ``polars``, ``cqros.config``, ``cqros.core``,
    ``cqros.factor_validation``, ``cqros.factor_selection``, and ``cqros.storage``.

Public API:
    ``build_parser``, ``build_options``, ``build_default_engine``,
    ``build_registry``, ``build_observation_source``,
    ``build_factor_selection_pipeline``,
    ``discover_work``, ``format_summary``, ``run_generation``, and ``main``.

Notes:
    This module is a thin composition root. It does not implement selection
    math, schema validation, or repository filesystem walks beyond calling
    repository discovery and load/save APIs. Metric computation is delegated
    exclusively to ``FactorSelectionPipeline``. Persistence remains in the
    CLI because ``FactorSelectionPipeline`` does not own a repository.
    Optional ``--export-detailed-csv`` writes audit CSVs via
    ``cqros.factor_selection.detailed_export`` without replacing Parquet.
    Partitions are cross-sectional panels keyed by manager/timeframe/year
    (no symbol).
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
    STORAGE_DIR_FACTOR_SELECTION,
    SUPPORTED_TIMEFRAMES,
)
from cqros.core.exceptions import CQROSError, ValidationError
from cqros.core.types import Timeframe
from cqros.factor_selection import (
    FactorEligibilityPolicy,
    FactorObservationSource,
    FactorSelectionEngine,
    FactorSelectionEngineRegistry,
    FactorSelectionExecutionMode,
    FactorSelectionPipeline,
    FactorSelectionRepository,
    FactorSelectionStatus,
    FactorsObservationLoader,
    MemoryEfficientFactorsObservationLoader,
    SimpleFactorSelectionEngine,
    build_detailed_audit_frame,
    combined_detailed_csv_path,
    detailed_csv_path,
    write_combined_detailed_csv,
    write_detailed_csv,
)
from cqros.factor_selection.engine import DEFAULT_TOP_N
from cqros.factor_selection.memory_efficient import DEFAULT_FACTOR_BATCH_SIZE
from cqros.factor_selection.redundancy import (
    DEFAULT_CANDIDATE_N,
    DEFAULT_MAX_FACTOR_CORRELATION,
    DEFAULT_MIN_CORRELATION_OVERLAP,
)
from cqros.factor_validation import FactorValidationPartitionRef, FactorValidationRepository
from cqros.storage import ParquetStore, StorageLayout

__all__ = [
    "DiscoveredWorkItem",
    "FactorSelectionGenerationOptions",
    "FactorSelectionGenerationSummary",
    "FactorSelectionTaskResult",
    "build_default_engine",
    "build_factor_selection_pipeline",
    "build_observation_source",
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
_DEFAULT_EXECUTION_MODE: Final[str] = FactorSelectionExecutionMode.MEMORY_EFFICIENT.value

_EXCHANGE: Final[str] = EXCHANGE_BINANCE
_MARKET: Final[str] = MARKET_USDT_PERPETUAL

_ERROR_WORKERS: Final[str] = "CLI-GENERATE-FACTOR-SELECTION-001"
_ERROR_TIMEFRAME: Final[str] = "CLI-GENERATE-FACTOR-SELECTION-002"
_ERROR_YEAR: Final[str] = "CLI-GENERATE-FACTOR-SELECTION-003"
_ERROR_MANAGER: Final[str] = "CLI-GENERATE-FACTOR-SELECTION-004"
_ERROR_ENGINE: Final[str] = "CLI-GENERATE-FACTOR-SELECTION-005"
_ERROR_FACTOR_VALIDATION_MISSING: Final[str] = "CLI-GENERATE-FACTOR-SELECTION-006"
_ERROR_TOP_N: Final[str] = "CLI-GENERATE-FACTOR-SELECTION-007"
_ERROR_CANDIDATE_N: Final[str] = "CLI-GENERATE-FACTOR-SELECTION-008"
_ERROR_MAX_CORR: Final[str] = "CLI-GENERATE-FACTOR-SELECTION-009"
_ERROR_MIN_OVERLAP: Final[str] = "CLI-GENERATE-FACTOR-SELECTION-010"
_ERROR_EXECUTION_MODE: Final[str] = "CLI-GENERATE-FACTOR-SELECTION-011"
_ERROR_FACTOR_BATCH_SIZE: Final[str] = "CLI-GENERATE-FACTOR-SELECTION-012"

_COL_STATUS: Final[str] = "status"


@dataclass(frozen=True, slots=True)
class FactorSelectionGenerationOptions:
    """Immutable CLI options for factor selection dataset generation.

    Attributes:
        storage_root: Storage root containing ``factor_validation`` and
            ``factor_selection``.
        manager: Order manager identity used for discovery and factor
            selection lineage.
        engine: Registry key of the factor selection engine to execute.
        top_n: Maximum factors selected per timeframe after redundancy filtering.
        candidate_n: Maximum ranked candidates considered by redundancy filtering.
        max_factor_correlation: Absolute Pearson redundancy threshold.
        min_overlap: Minimum pairwise observation overlap for redundancy.
        timeframes: Optional timeframe allowlist. ``None`` discovers all.
        years: Optional year allowlist. ``None`` discovers all.
        overwrite: When ``True``, regenerate existing factor selection
            partitions.
        export_detailed_csv: When ``True``, write per-timeframe and combined
            detailed audit CSV exports alongside canonical Parquet output.
        workers: Maximum concurrent panels.
        execution_mode: ``memory_efficient`` (default) or ``full_panel``.
        factor_batch_size: Factor identities spilled per symbol-scan batch when
            using ``memory_efficient`` mode.
        verbose: When ``True``, enable INFO logging.
        debug: When ``True``, enable DEBUG logging and log full failure
            tracebacks.
    """

    storage_root: Path
    manager: str
    engine: str
    top_n: int
    candidate_n: int
    max_factor_correlation: float
    min_overlap: int
    timeframes: tuple[Timeframe, ...] | None
    years: tuple[int, ...] | None
    overwrite: bool
    export_detailed_csv: bool
    workers: int
    execution_mode: FactorSelectionExecutionMode
    factor_batch_size: int
    verbose: bool
    debug: bool


@dataclass(frozen=True, slots=True)
class DiscoveredWorkItem:
    """One discovered Factor Validation panel group ready for selection generation.

    Attributes:
        manager: Order manager identifier of the source partitions.
        timeframe: Available bar interval.
        years: Calendar years with existing Factor Validation parquet partitions.
    """

    manager: str
    timeframe: Timeframe
    years: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FactorSelectionTaskResult:
    """Immutable result for one timeframe/year panel generation task.

    Attributes:
        timeframe: Bar interval.
        year: Calendar year of the partition.
        status: ``succeeded``, ``failed``, or ``skipped``.
        rows_generated: Output row count when succeeded.
        selected_rows: Count of rows with ``SELECTED`` status on success.
        rejected_status_rows: Count of rows with ``REJECTED`` status on success.
        detailed_audit: Optional detailed audit frame when CSV export is enabled.
        error_type: Exception type name when failed.
        error_message: Exception message when failed.
        error_code: Optional CQROS error code when failed.
    """

    timeframe: Timeframe
    year: int
    status: str
    rows_generated: int | None = None
    selected_rows: int | None = None
    rejected_status_rows: int | None = None
    detailed_audit: pl.DataFrame | None = None
    error_type: str | None = None
    error_message: str | None = None
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class FactorSelectionGenerationSummary:
    """Immutable aggregate summary for a factor-selection-generation run.

    Attributes:
        manager: Order manager identity used for generation.
        engine: Factor selection engine registry key used for generation.
        panels: Unique timeframe/year panels for which generation was attempted.
        rows: Sum of output rows across successes.
        selected_rows: Sum of rows with ``SELECTED`` status across successes.
        rejected_status_rows: Sum of rows with ``REJECTED`` status across successes.
        successful_tasks: Count of succeeded partition tasks.
        failed_tasks: Count of failed partition tasks.
        skipped_tasks: Count of skipped existing partitions.
        duration_seconds: Wall-clock generation duration.
        output_directory: Factor-selection-tier output directory.
        failed_task_labels: Deterministic failed-task labels for reporting.
    """

    manager: str
    engine: str
    panels: int
    rows: int
    selected_rows: int
    rejected_status_rows: int
    successful_tasks: int
    failed_tasks: int
    skipped_tasks: int
    duration_seconds: float
    output_directory: Path
    failed_task_labels: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    """Create the factor-selection-generation argument parser.

    Returns:
        Configured ``ArgumentParser`` for factor-selection-generation flags.
    """
    parser = argparse.ArgumentParser(
        prog="cqros-generate-factor-selection",
        description=(
            "Generate CQROS factor selection datasets from discovered "
            "Factor Validation panels and an injected factor selection engine."
        ),
    )
    parser.add_argument(
        "--manager",
        dest="manager",
        required=True,
        metavar="NAME",
        help="Order manager identity used for discovery and factor selection lineage.",
    )
    parser.add_argument(
        "--engine",
        dest="engine",
        default=_DEFAULT_ENGINE,
        metavar="NAME",
        help=f"Factor selection engine registry key (default: {_DEFAULT_ENGINE}).",
    )
    parser.add_argument(
        "--top-n",
        dest="top_n",
        type=int,
        default=DEFAULT_TOP_N,
        metavar="INT",
        help=("Maximum factors selected per timeframe " f"(default: {DEFAULT_TOP_N})."),
    )
    parser.add_argument(
        "--candidate-n",
        dest="candidate_n",
        type=int,
        default=DEFAULT_CANDIDATE_N,
        metavar="INT",
        help=(
            "Maximum ranked candidates considered by redundancy filtering "
            f"(default: {DEFAULT_CANDIDATE_N})."
        ),
    )
    parser.add_argument(
        "--max-factor-correlation",
        dest="max_factor_correlation",
        type=float,
        default=DEFAULT_MAX_FACTOR_CORRELATION,
        metavar="FLOAT",
        help=(
            "Absolute Pearson correlation threshold for redundancy rejection "
            f"(default: {DEFAULT_MAX_FACTOR_CORRELATION})."
        ),
    )
    parser.add_argument(
        "--min-overlap",
        dest="min_overlap",
        type=int,
        default=DEFAULT_MIN_CORRELATION_OVERLAP,
        metavar="INT",
        help=(
            "Minimum pairwise complete observations required for a "
            f"redundancy decision (default: {DEFAULT_MIN_CORRELATION_OVERLAP})."
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
        help="Regenerate factor selection partitions that already exist.",
    )
    parser.add_argument(
        "--export-detailed-csv",
        dest="export_detailed_csv",
        action="store_true",
        help=(
            "Write detailed audit CSV exports (per-timeframe and combined) "
            "alongside canonical Parquet factor selection datasets."
        ),
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
        choices=tuple(mode.value for mode in FactorSelectionExecutionMode),
        default=_DEFAULT_EXECUTION_MODE,
        metavar="MODE",
        help=(
            "Observation materialization strategy: memory_efficient (default; "
            "per-factor spill + pairwise redundancy) or full_panel (legacy "
            "in-RAM candidate panel)."
        ),
    )
    parser.add_argument(
        "--factor-batch-size",
        dest="factor_batch_size",
        type=int,
        default=DEFAULT_FACTOR_BATCH_SIZE,
        metavar="INT",
        help=(
            "Factor identities spilled per symbol-scan batch in "
            f"memory_efficient mode (default: {DEFAULT_FACTOR_BATCH_SIZE})."
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


def build_options(args: argparse.Namespace) -> FactorSelectionGenerationOptions:
    """Map parsed CLI arguments onto ``FactorSelectionGenerationOptions``.

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

    top_n = int(args.top_n)
    if top_n <= 0:
        raise ValidationError(
            "top_n must be greater than 0",
            error_code=_ERROR_TOP_N,
            details={"parameter": "top_n", "value": top_n},
        )

    candidate_n = int(args.candidate_n)
    if candidate_n <= 0:
        raise ValidationError(
            "candidate_n must be greater than 0",
            error_code=_ERROR_CANDIDATE_N,
            details={"parameter": "candidate_n", "value": candidate_n},
        )
    if candidate_n < top_n:
        raise ValidationError(
            "candidate_n must be greater than or equal to top_n",
            error_code=_ERROR_CANDIDATE_N,
            details={"parameter": "candidate_n", "value": candidate_n, "top_n": top_n},
        )

    max_factor_correlation = float(args.max_factor_correlation)
    if (
        max_factor_correlation <= 0.0
        or max_factor_correlation >= 1.0
        or max_factor_correlation != max_factor_correlation
    ):
        raise ValidationError(
            "max_factor_correlation must be in (0, 1)",
            error_code=_ERROR_MAX_CORR,
            details={
                "parameter": "max_factor_correlation",
                "value": max_factor_correlation,
            },
        )

    min_overlap = int(args.min_overlap)
    if min_overlap <= 0:
        raise ValidationError(
            "min_overlap must be greater than 0",
            error_code=_ERROR_MIN_OVERLAP,
            details={"parameter": "min_overlap", "value": min_overlap},
        )

    factor_batch_size = int(args.factor_batch_size)
    if factor_batch_size < 1:
        raise ValidationError(
            "factor_batch_size must be >= 1",
            error_code=_ERROR_FACTOR_BATCH_SIZE,
            details={"parameter": "factor_batch_size", "value": factor_batch_size},
        )

    try:
        execution_mode = FactorSelectionExecutionMode(str(args.execution_mode).strip())
    except ValueError as exc:
        raise ValidationError(
            "execution_mode must be memory_efficient or full_panel",
            error_code=_ERROR_EXECUTION_MODE,
            details={"parameter": "execution_mode", "value": args.execution_mode},
        ) from exc

    storage_root = (
        Path(args.storage_root) if args.storage_root is not None else Path(DEFAULT_STORAGE_ROOT)
    )

    return FactorSelectionGenerationOptions(
        storage_root=storage_root,
        manager=manager,
        engine=engine,
        top_n=top_n,
        candidate_n=candidate_n,
        max_factor_correlation=max_factor_correlation,
        min_overlap=min_overlap,
        timeframes=_normalize_timeframes(args.timeframes),
        years=_normalize_years(args.years),
        overwrite=bool(args.overwrite),
        export_detailed_csv=bool(args.export_detailed_csv),
        workers=workers,
        execution_mode=execution_mode,
        factor_batch_size=factor_batch_size,
        verbose=bool(args.verbose),
        debug=bool(args.debug),
    )


def build_default_engine(
    *,
    top_n: int = DEFAULT_TOP_N,
    candidate_n: int = DEFAULT_CANDIDATE_N,
    max_factor_correlation: float = DEFAULT_MAX_FACTOR_CORRELATION,
    min_overlap: int = DEFAULT_MIN_CORRELATION_OVERLAP,
    observation_source: FactorObservationSource | None = None,
    eligibility_policy: FactorEligibilityPolicy | None = None,
) -> FactorSelectionEngine:
    """Compose the default production factor selection engine for the CLI.

    The ``eligibility_policy`` defaults to ``FactorEligibilityPolicy()`` so
    that the production path always enforces eligibility gating.  Pass
    ``eligibility_policy=None`` only in tests that explicitly need the
    backward-compatible no-filter mode.
    """
    if eligibility_policy is None:
        eligibility_policy = FactorEligibilityPolicy()
    return SimpleFactorSelectionEngine(
        top_n=top_n,
        candidate_n=candidate_n,
        max_factor_correlation=max_factor_correlation,
        min_overlap=min_overlap,
        observation_source=observation_source,
        eligibility_policy=eligibility_policy,
    )


def build_registry(
    *,
    engines: Mapping[str, FactorSelectionEngine] | None = None,
    top_n: int = DEFAULT_TOP_N,
    candidate_n: int = DEFAULT_CANDIDATE_N,
    max_factor_correlation: float = DEFAULT_MAX_FACTOR_CORRELATION,
    min_overlap: int = DEFAULT_MIN_CORRELATION_OVERLAP,
    observation_source: FactorObservationSource | None = None,
    eligibility_policy: FactorEligibilityPolicy | None = None,
) -> FactorSelectionEngineRegistry:
    """Compose a registry with default or injected factor selection engines."""
    registry = FactorSelectionEngineRegistry()
    if engines is None:
        registry.register(
            _DEFAULT_ENGINE,
            build_default_engine(
                top_n=top_n,
                candidate_n=candidate_n,
                max_factor_correlation=max_factor_correlation,
                min_overlap=min_overlap,
                observation_source=observation_source,
                eligibility_policy=eligibility_policy,
            ),
        )
    else:
        for name, engine in engines.items():
            registry.register(name, engine)
    return registry


def build_factor_selection_pipeline(
    options: FactorSelectionGenerationOptions,
    *,
    engine_registry: FactorSelectionEngineRegistry | None = None,
    observation_source: FactorObservationSource | None = None,
    eligibility_policy: FactorEligibilityPolicy | None = None,
) -> FactorSelectionPipeline:
    """Compose ``FactorSelectionPipeline`` from injected engine registries."""
    if engine_registry is None:
        engine_registry = build_registry(
            top_n=options.top_n,
            candidate_n=options.candidate_n,
            max_factor_correlation=options.max_factor_correlation,
            min_overlap=options.min_overlap,
            observation_source=observation_source,
            eligibility_policy=eligibility_policy,
        )
    elif options.engine == _DEFAULT_ENGINE and not engine_registry.exists(options.engine):
        engine_registry.register(
            options.engine,
            build_default_engine(
                top_n=options.top_n,
                candidate_n=options.candidate_n,
                max_factor_correlation=options.max_factor_correlation,
                min_overlap=options.min_overlap,
                observation_source=observation_source,
                eligibility_policy=eligibility_policy,
            ),
        )
    return FactorSelectionPipeline(engine_registry)


def discover_work(
    factor_validation_repository: FactorValidationRepository,
    options: FactorSelectionGenerationOptions,
) -> tuple[DiscoveredWorkItem, ...]:
    """Discover selection-ready Factor Validation panels matching CLI filters.

    Only Factor Validation partitions that exist are scheduled. Missing Factor
    Validation partitions are never invented.

    Args:
        factor_validation_repository: Factor Validation repository providing discovery APIs.
        options: CLI filters for manager, timeframe, and year.

    Returns:
        Deterministically ordered discovered work items.
    """
    partitions = factor_validation_repository.discover_partitions(
        managers=(options.manager,),
        timeframes=options.timeframes,
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    return _group_partitions(partitions, year_filter=options.years)


def format_summary(summary: FactorSelectionGenerationSummary) -> str:
    """Render a deterministic factor-selection-generation summary report.

    Args:
        summary: Aggregate generation summary.

    Returns:
        Multi-line summary text suitable for stdout.
    """
    lines = [
        "=====================================",
        "CQROS Factor Selection Generation Summary",
        "=====================================",
        "",
        f"Manager: {summary.manager}",
        f"Engine: {summary.engine}",
        "",
        f"Panels: {summary.panels}",
        f"Rows: {summary.rows}",
        f"Selected: {summary.selected_rows}",
        f"Rejected Status: {summary.rejected_status_rows}",
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
    """Run the factor-selection-generation CLI.

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
        factor_validation_repository = FactorValidationRepository(layout, datastore)
        factor_selection_repository = FactorSelectionRepository(layout, datastore)
        work = discover_work(factor_validation_repository, options)
        summary = await run_generation(
            layout=layout,
            factor_validation_repository=factor_validation_repository,
            factor_selection_repository=factor_selection_repository,
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
    layout: StorageLayout,
    factor_validation_repository: FactorValidationRepository,
    factor_selection_repository: FactorSelectionRepository,
    options: FactorSelectionGenerationOptions,
    work: Sequence[DiscoveredWorkItem],
) -> FactorSelectionGenerationSummary:
    """Execute discovered work through a bounded panel worker pool."""
    started = time.perf_counter()
    output_directory = options.storage_root / STORAGE_DIR_FACTOR_SELECTION

    if len(work) == 0:
        return FactorSelectionGenerationSummary(
            manager=options.manager,
            engine=options.engine,
            panels=0,
            rows=0,
            selected_rows=0,
            rejected_status_rows=0,
            successful_tasks=0,
            failed_tasks=0,
            skipped_tasks=0,
            duration_seconds=time.perf_counter() - started,
            output_directory=output_directory,
            failed_task_labels=(),
        )

    work_by_timeframe = _group_work_by_timeframe(work)
    results = await _run_worker_pool(
        layout=layout,
        factor_validation_repository=factor_validation_repository,
        factor_selection_repository=factor_selection_repository,
        work_by_timeframe=work_by_timeframe,
        worker_count=options.workers,
        overwrite=options.overwrite,
        export_detailed_csv=options.export_detailed_csv,
        debug=options.debug,
        manager_name=options.manager,
        engine_name=options.engine,
        top_n=options.top_n,
        candidate_n=options.candidate_n,
        max_factor_correlation=options.max_factor_correlation,
        min_overlap=options.min_overlap,
        execution_mode=options.execution_mode,
        factor_batch_size=options.factor_batch_size,
        storage_root=options.storage_root,
    )
    if options.export_detailed_csv:
        _write_combined_detailed_export(
            results=results,
            storage_root=options.storage_root,
            manager=options.manager,
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
    partitions: Sequence[FactorValidationPartitionRef],
    *,
    year_filter: tuple[int, ...] | None,
) -> tuple[DiscoveredWorkItem, ...]:
    """Group Factor Validation year partitions into manager/timeframe work items."""
    year_allowlist = set(year_filter) if year_filter is not None else None
    grouped: dict[tuple[str, str], list[int]] = {}
    for partition in partitions:
        if year_allowlist is not None and partition.year not in year_allowlist:
            continue
        key = (partition.manager, partition.timeframe)
        grouped.setdefault(key, []).append(partition.year)

    items: list[DiscoveredWorkItem] = []
    for (manager, timeframe), years in grouped.items():
        items.append(
            DiscoveredWorkItem(
                manager=manager,
                timeframe=timeframe,
                years=tuple(sorted(years)),
            )
        )
    return tuple(
        sorted(
            items,
            key=lambda item: (item.manager, item.timeframe),
        )
    )


def _group_work_by_timeframe(
    work: Sequence[DiscoveredWorkItem],
) -> dict[Timeframe, tuple[DiscoveredWorkItem, ...]]:
    """Group discovered work by timeframe while preserving discovery order."""
    grouped: dict[Timeframe, list[DiscoveredWorkItem]] = {}
    for item in work:
        grouped.setdefault(item.timeframe, []).append(item)
    return {timeframe: tuple(items) for timeframe, items in grouped.items()}


async def _run_worker_pool(
    *,
    layout: StorageLayout,
    factor_validation_repository: FactorValidationRepository,
    factor_selection_repository: FactorSelectionRepository,
    work_by_timeframe: Mapping[Timeframe, Sequence[DiscoveredWorkItem]],
    worker_count: int,
    overwrite: bool,
    export_detailed_csv: bool,
    debug: bool,
    manager_name: str,
    engine_name: str,
    top_n: int,
    candidate_n: int,
    max_factor_correlation: float,
    min_overlap: int,
    execution_mode: FactorSelectionExecutionMode,
    factor_batch_size: int,
    storage_root: Path,
) -> tuple[FactorSelectionTaskResult, ...]:
    """Drain timeframes through a bounded asyncio worker pool."""
    timeframes = tuple(work_by_timeframe.keys())
    if len(timeframes) == 0:
        return ()

    queue: asyncio.Queue[Timeframe | None] = asyncio.Queue()
    for timeframe in timeframes:
        queue.put_nowait(timeframe)
    for _ in range(worker_count):
        queue.put_nowait(None)

    collected: list[FactorSelectionTaskResult] = []
    lock = asyncio.Lock()

    async def worker() -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                results = await _generate_timeframe_work(
                    layout=layout,
                    factor_validation_repository=factor_validation_repository,
                    factor_selection_repository=factor_selection_repository,
                    timeframe=item,
                    work_items=work_by_timeframe[item],
                    overwrite=overwrite,
                    export_detailed_csv=export_detailed_csv,
                    debug=debug,
                    manager_name=manager_name,
                    engine_name=engine_name,
                    top_n=top_n,
                    candidate_n=candidate_n,
                    max_factor_correlation=max_factor_correlation,
                    min_overlap=min_overlap,
                    execution_mode=execution_mode,
                    factor_batch_size=factor_batch_size,
                    storage_root=storage_root,
                )
                async with lock:
                    collected.extend(results)
            finally:
                queue.task_done()

    worker_tasks = [
        asyncio.create_task(worker(), name=f"generate-factor-selection-worker-{index}")
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


async def _generate_timeframe_work(
    *,
    layout: StorageLayout,
    factor_validation_repository: FactorValidationRepository,
    factor_selection_repository: FactorSelectionRepository,
    timeframe: Timeframe,
    work_items: Sequence[DiscoveredWorkItem],
    overwrite: bool,
    export_detailed_csv: bool,
    debug: bool,
    manager_name: str,
    engine_name: str,
    top_n: int,
    candidate_n: int,
    max_factor_correlation: float,
    min_overlap: int,
    execution_mode: FactorSelectionExecutionMode,
    factor_batch_size: int,
    storage_root: Path,
) -> tuple[FactorSelectionTaskResult, ...]:
    """Generate factor selection datasets for every discovered year for one timeframe."""
    results: list[FactorSelectionTaskResult] = []
    for item in work_items:
        for year in item.years:
            result = await asyncio.to_thread(
                _generate_partition,
                layout,
                factor_validation_repository,
                factor_selection_repository,
                manager=item.manager,
                timeframe=timeframe,
                year=year,
                overwrite=overwrite,
                export_detailed_csv=export_detailed_csv,
                debug=debug,
                manager_name=manager_name,
                engine_name=engine_name,
                top_n=top_n,
                candidate_n=candidate_n,
                max_factor_correlation=max_factor_correlation,
                min_overlap=min_overlap,
                execution_mode=execution_mode,
                factor_batch_size=factor_batch_size,
                storage_root=storage_root,
            )
            _print_progress(result)
            results.append(result)
    return tuple(results)


def _generate_partition(
    layout: StorageLayout,
    factor_validation_repository: FactorValidationRepository,
    factor_selection_repository: FactorSelectionRepository,
    *,
    manager: str,
    timeframe: Timeframe,
    year: int,
    overwrite: bool,
    export_detailed_csv: bool,
    debug: bool,
    manager_name: str,
    engine_name: str,
    top_n: int,
    candidate_n: int,
    max_factor_correlation: float,
    min_overlap: int,
    execution_mode: FactorSelectionExecutionMode,
    factor_batch_size: int,
    storage_root: Path,
) -> FactorSelectionTaskResult:
    """Generate one factor selection year partition synchronously."""
    if not overwrite and factor_selection_repository.exists(
        manager=manager_name,
        exchange=_EXCHANGE,
        market=_MARKET,
        timeframe=timeframe,
        year=year,
    ):
        return FactorSelectionTaskResult(
            timeframe=timeframe,
            year=year,
            status="skipped",
        )

    try:
        if not factor_validation_repository.exists(
            manager=manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            timeframe=timeframe,
            year=year,
        ):
            raise ValidationError(
                f"factor validation partition missing for {manager}/{timeframe}/{year}",
                error_code=_ERROR_FACTOR_VALIDATION_MISSING,
                details={
                    "manager": manager,
                    "timeframe": timeframe,
                    "year": year,
                    "missing": "factor_validation",
                },
            )

        factor_validation = factor_validation_repository.load(
            manager=manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            timeframe=timeframe,
            year=year,
        )
        observation_source = build_observation_source(
            layout,
            manager=manager_name,
            year=year,
            execution_mode=execution_mode,
            factor_batch_size=factor_batch_size,
            storage_root=storage_root,
        )
        engine = build_default_engine(
            top_n=top_n,
            candidate_n=candidate_n,
            max_factor_correlation=max_factor_correlation,
            min_overlap=min_overlap,
            observation_source=observation_source,
        )
        registry = FactorSelectionEngineRegistry()
        registry.register(engine_name, engine)
        pipeline = FactorSelectionPipeline(registry)
        output = pipeline.run(engine_name, factor_validation)
        redundancy_audit = (
            engine.last_audit if isinstance(engine, SimpleFactorSelectionEngine) else None
        )
        factor_selection_repository.save(
            output,
            manager=manager_name,
            exchange=_EXCHANGE,
            market=_MARKET,
            timeframe=timeframe,
            year=year,
        )
        detailed_audit: pl.DataFrame | None = None
        if export_detailed_csv:
            detailed_audit = build_detailed_audit_frame(
                factor_validation,
                output,
                top_n=top_n,
                manager=manager_name,
                exchange=_EXCHANGE,
                market=_MARKET,
                redundancy_audit=redundancy_audit,
                candidate_n=candidate_n,
                max_factor_correlation=max_factor_correlation,
                min_correlation_overlap=min_overlap,
            )
            write_detailed_csv(
                detailed_audit,
                detailed_csv_path(
                    storage_root,
                    manager=manager_name,
                    exchange=_EXCHANGE,
                    market=_MARKET,
                    timeframe=timeframe,
                    year=year,
                ),
            )
    except Exception as exc:
        _log_partition_failure(
            timeframe=timeframe,
            year=year,
            exc=exc,
            debug=debug,
        )
        return FactorSelectionTaskResult(
            timeframe=timeframe,
            year=year,
            status="failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
            error_code=exc.error_code if isinstance(exc, CQROSError) else None,
        )

    rows_generated, selected_rows, rejected_status_rows = _extract_partition_stats(output)
    return FactorSelectionTaskResult(
        timeframe=timeframe,
        year=year,
        status="succeeded",
        rows_generated=rows_generated,
        selected_rows=selected_rows,
        rejected_status_rows=rejected_status_rows,
        detailed_audit=detailed_audit,
    )


def build_observation_source(
    layout: StorageLayout,
    *,
    manager: str,
    year: int,
    execution_mode: FactorSelectionExecutionMode,
    factor_batch_size: int,
    storage_root: Path,
) -> FactorObservationSource:
    """Compose the observation loader for the configured execution mode."""
    if execution_mode is FactorSelectionExecutionMode.MEMORY_EFFICIENT:
        return MemoryEfficientFactorsObservationLoader(
            layout,
            manager=manager,
            year=year,
            exchange=_EXCHANGE,
            market=_MARKET,
            factor_batch_size=factor_batch_size,
            spill_parent=storage_root / ".cqros_tmp" / "factor_selection_spill",
        )
    return FactorsObservationLoader(
        layout,
        manager=manager,
        year=year,
        exchange=_EXCHANGE,
        market=_MARKET,
    )


def _write_combined_detailed_export(
    *,
    results: Sequence[FactorSelectionTaskResult],
    storage_root: Path,
    manager: str,
) -> None:
    """Write the combined detailed audit CSV from successful partition exports."""
    frames = [
        result.detailed_audit
        for result in results
        if result.status == "succeeded" and result.detailed_audit is not None
    ]
    if len(frames) == 0:
        return
    write_combined_detailed_csv(
        frames,
        combined_detailed_csv_path(
            storage_root,
            manager=manager,
            exchange=_EXCHANGE,
            market=_MARKET,
        ),
    )


def _extract_partition_stats(frame: pl.DataFrame) -> tuple[int, int, int]:
    """Extract row count and selection-status counts from one metrics frame.

    Args:
        frame: Finalized factor selection output DataFrame.

    Returns:
        A 3-tuple of ``(rows, selected_rows, rejected_status_rows)``.
    """
    if frame.height == 0:
        return 0, 0, 0

    selected_rows = int(
        frame.select((pl.col(_COL_STATUS) == FactorSelectionStatus.SELECTED.value).sum()).item()
    )
    rejected_status_rows = int(
        frame.select((pl.col(_COL_STATUS) == FactorSelectionStatus.REJECTED.value).sum()).item()
    )
    return frame.height, selected_rows, rejected_status_rows


def _print_progress(result: FactorSelectionTaskResult) -> None:
    """Print a deterministic one-line progress record for a task result."""
    label = f"{result.timeframe} {result.year}"
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
    timeframe: Timeframe,
    year: int,
    exc: BaseException,
    debug: bool,
) -> None:
    """Log a partition generation failure without aborting the run."""
    log_extra = {
        "timeframe": timeframe,
        "year": year,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "error_code": exc.error_code if isinstance(exc, CQROSError) else None,
    }
    if debug:
        _logger.exception(
            "Failed factor selection generation partition; continuing",
            extra=log_extra,
            exc_info=True,
        )
    else:
        _logger.warning(
            "Failed factor selection generation partition; continuing",
            extra=log_extra,
        )


def _build_summary(
    *,
    options: FactorSelectionGenerationOptions,
    work: Sequence[DiscoveredWorkItem],
    results: Sequence[FactorSelectionTaskResult],
    duration_seconds: float,
    output_directory: Path,
) -> FactorSelectionGenerationSummary:
    """Aggregate task results into a generation report."""
    panels_discovered = sum(len(item.years) for item in work)
    panels_processed: set[tuple[Timeframe, int]] = set()
    successful_tasks = 0
    failed_tasks = 0
    skipped_tasks = 0
    rows = 0
    selected_rows = 0
    rejected_status_rows = 0
    failed_labels: set[str] = set()

    for result in results:
        panels_processed.add((result.timeframe, result.year))
        if result.status == "succeeded":
            successful_tasks += 1
            if result.rows_generated is not None:
                rows += result.rows_generated
            if result.selected_rows is not None:
                selected_rows += result.selected_rows
            if result.rejected_status_rows is not None:
                rejected_status_rows += result.rejected_status_rows
        elif result.status == "skipped":
            skipped_tasks += 1
        else:
            failed_tasks += 1
            failed_labels.add(f"{result.timeframe} {result.year}")

    return FactorSelectionGenerationSummary(
        manager=options.manager,
        engine=options.engine,
        panels=len(panels_processed) if results else panels_discovered,
        rows=rows,
        selected_rows=selected_rows,
        rejected_status_rows=rejected_status_rows,
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
