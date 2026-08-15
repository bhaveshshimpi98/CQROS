"""CQROS purged-CV evaluation CLI.

Purpose:
    Discover existing Purged-CV ledger partitions, reconstruct membership from
    matching Walk-Forward panels, optionally enrich with Labels
    ``future_return_1`` via the Walk-Forward evaluation-input adapter, persist
    a separate evaluation-result artifact, and write CSV reports.

Responsibilities:
    - Parse CLI arguments for purged-CV evaluation
    - Discover Purged-CV ledger partitions (never invent timeframes)
    - Load Walk-Forward ledgers for purge/embargo audit reconstruction
    - Optionally rebuild evaluation input via ``WalkForwardInputBuilder``
    - Execute ``PurgedCVEvaluator`` and persist via
      ``PurgedCVEvaluationRepository``
    - Write evaluation CSV reports under ``reports/purged_cv``
    - Leave the Purged-CV ledger byte-identical
    - Remain free of Alpha, Regime, Predictions, Signals, and ``cqros.ml``

Dependencies:
    ``argparse``, ``asyncio``, ``hashlib``, ``polars``, ``cqros.config``,
    ``cqros.core``, ``cqros.factor_selection``, ``cqros.factors``,
    ``cqros.purged_cv``, ``cqros.walk_forward``,
    ``cqros.reporting.purged_cv_evaluation_report``, and ``cqros.storage``.

Public API:
    ``build_parser``, ``build_options``, ``discover_work``, ``run_evaluation``,
    ``format_summary``, and ``main``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
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
    STORAGE_DIR_PURGED_CV,
    STORAGE_DIR_PURGED_CV_EVALUATION,
    STORAGE_DIR_WALK_FORWARD,
    SUPPORTED_TIMEFRAMES,
)
from cqros.core.exceptions import CQROSError, ValidationError
from cqros.core.types import Timeframe
from cqros.factor_selection import FactorSelectionRepository
from cqros.factors import FactorsRepository
from cqros.purged_cv import PurgedCVPartitionRef, PurgedCVRepository
from cqros.purged_cv.evaluation import PurgedCVEvaluator
from cqros.purged_cv.evaluation_repository import PurgedCVEvaluationRepository
from cqros.reporting.purged_cv_evaluation_report import (
    DEFAULT_OUTPUT_ROOT,
    PurgedCVEvaluationReporter,
)
from cqros.storage import LabelRepository, ParquetStore, StorageLayout
from cqros.walk_forward import WalkForwardRepository
from cqros.walk_forward.evaluation_input import WalkForwardInputBuilder

__all__ = [
    "DiscoveredWorkItem",
    "PurgedCVEvaluationOptions",
    "PurgedCVEvaluationSummary",
    "PurgedCVEvaluationTaskResult",
    "build_options",
    "build_parser",
    "discover_work",
    "format_summary",
    "main",
    "run_evaluation",
]

_logger = logging.getLogger(__name__)

_EXIT_SUCCESS: Final[int] = 0
_EXIT_FAILURE: Final[int] = 1

_DEFAULT_WORKER_COUNT: Final[int] = ResearchConfig().worker_count
_DEFAULT_ENGINE: Final[str] = "simple"

_EXCHANGE: Final[str] = EXCHANGE_BINANCE
_MARKET: Final[str] = MARKET_USDT_PERPETUAL

_ERROR_WORKERS: Final[str] = "CLI-EVALUATE-PURGED-CV-001"
_ERROR_TIMEFRAME: Final[str] = "CLI-EVALUATE-PURGED-CV-002"
_ERROR_YEAR: Final[str] = "CLI-EVALUATE-PURGED-CV-003"
_ERROR_MANAGER: Final[str] = "CLI-EVALUATE-PURGED-CV-004"
_ERROR_ENGINE: Final[str] = "CLI-EVALUATE-PURGED-CV-005"
_ERROR_PURGED_CV_MISSING: Final[str] = "CLI-EVALUATE-PURGED-CV-006"
_ERROR_WALK_FORWARD_MISSING: Final[str] = "CLI-EVALUATE-PURGED-CV-007"


@dataclass(frozen=True, slots=True)
class PurgedCVEvaluationOptions:
    """Immutable CLI options for purged-CV evaluation."""

    storage_root: Path
    manager: str
    engine: str
    timeframes: tuple[Timeframe, ...] | None
    years: tuple[int, ...] | None
    overwrite: bool
    workers: int
    verbose: bool
    debug: bool
    report_output: Path


@dataclass(frozen=True, slots=True)
class DiscoveredWorkItem:
    """One discovered Purged-CV ledger panel ready for evaluation."""

    manager: str
    timeframe: Timeframe
    years: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PurgedCVEvaluationTaskResult:
    """Immutable result for one timeframe/year evaluation task."""

    timeframe: Timeframe
    year: int
    status: str
    rows_generated: int | None = None
    folds: int | None = None
    oos_rows: int | None = None
    oos_return_mean: float | None = None
    unique_selected_factors: int | None = None
    panel_status: str | None = None
    ledger_sha256_before: str | None = None
    ledger_sha256_after: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    error_code: str | None = None
    summary_row: pl.DataFrame | None = None
    fold_metrics: pl.DataFrame | None = None
    factor_metrics: pl.DataFrame | None = None


@dataclass(frozen=True, slots=True)
class PurgedCVEvaluationSummary:
    """Immutable aggregate summary for a purged-CV evaluation run."""

    manager: str
    engine: str
    panels: int
    rows: int
    folds: int
    successful_tasks: int
    failed_tasks: int
    skipped_tasks: int
    duration_seconds: float
    output_directory: Path
    report_directory: Path
    failed_task_labels: tuple[str, ...]
    ledger_hashes_unchanged: bool


def build_parser() -> argparse.ArgumentParser:
    """Create the purged-CV-evaluation argument parser."""
    parser = argparse.ArgumentParser(
        prog="cqros-evaluate-purged-cv",
        description=(
            "Evaluate CQROS Purged-CV ledgers into a separate evaluation-result "
            "artifact and CSV reports. Discovers existing purged_cv partitions."
        ),
    )
    parser.add_argument(
        "--manager",
        dest="manager",
        required=True,
        metavar="NAME",
        help="Order manager identity used for discovery and evaluation lineage.",
    )
    parser.add_argument(
        "--engine",
        dest="engine",
        default=_DEFAULT_ENGINE,
        metavar="NAME",
        help=f"Purged-CV engine label recorded on artifacts (default: {_DEFAULT_ENGINE}).",
    )
    parser.add_argument(
        "--timeframes",
        dest="timeframes",
        nargs="*",
        default=None,
        metavar="TIMEFRAME",
        help="Optional timeframe allowlist. Omit to discover all ledger timeframes.",
    )
    parser.add_argument(
        "--years",
        dest="years",
        nargs="*",
        default=None,
        metavar="YEAR",
        help="Optional calendar-year allowlist. Omit to discover all ledger years.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate evaluation partitions that already exist.",
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
        "--verbose",
        action="store_true",
        help="Enable INFO logging.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging and full failure tracebacks.",
    )
    parser.add_argument(
        "--storage-root",
        dest="storage_root",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"Storage root for dataset tiers (default: {DEFAULT_STORAGE_ROOT}).",
    )
    parser.add_argument(
        "--output",
        dest="output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        metavar="PATH",
        help=f"CSV report directory (default: {DEFAULT_OUTPUT_ROOT.as_posix()}).",
    )
    return parser


def build_options(args: argparse.Namespace) -> PurgedCVEvaluationOptions:
    """Map parsed CLI arguments onto immutable options."""
    workers = int(args.workers)
    if workers <= 0:
        raise ValidationError(
            "workers must be greater than 0",
            error_code=_ERROR_WORKERS,
            details={"parameter": "workers", "value": workers},
        )
    manager = str(args.manager).strip()
    if not manager:
        raise ValidationError(
            "manager must be a non-empty string",
            error_code=_ERROR_MANAGER,
            details={"parameter": "manager", "value": args.manager},
        )
    engine = str(args.engine).strip()
    if not engine:
        raise ValidationError(
            "engine must be a non-empty string",
            error_code=_ERROR_ENGINE,
            details={"parameter": "engine", "value": args.engine},
        )
    storage_root = (
        Path(args.storage_root) if args.storage_root is not None else Path(DEFAULT_STORAGE_ROOT)
    )
    return PurgedCVEvaluationOptions(
        storage_root=storage_root,
        manager=manager,
        engine=engine,
        timeframes=_normalize_timeframes(args.timeframes),
        years=_normalize_years(args.years),
        overwrite=bool(args.overwrite),
        workers=workers,
        verbose=bool(args.verbose),
        debug=bool(args.debug),
        report_output=Path(args.output),
    )


def discover_work(
    purged_cv_repository: PurgedCVRepository,
    options: PurgedCVEvaluationOptions,
) -> tuple[DiscoveredWorkItem, ...]:
    """Discover Purged-CV ledger panels matching CLI filters."""
    partitions = purged_cv_repository.discover_partitions(
        managers=(options.manager,),
        timeframes=options.timeframes,
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    return _group_partitions(partitions, year_filter=options.years)


def format_summary(summary: PurgedCVEvaluationSummary) -> str:
    """Render a deterministic purged-CV-evaluation summary report."""
    lines = [
        "=======================================",
        "CQROS Purged-CV Evaluation Summary",
        "=======================================",
        "",
        f"Manager: {summary.manager}",
        f"Engine: {summary.engine}",
        "",
        f"Panels: {summary.panels}",
        f"Observation Rows: {summary.rows}",
        f"Folds: {summary.folds}",
        "",
        f"Successful: {summary.successful_tasks}",
        f"Failed: {summary.failed_tasks}",
        f"Skipped: {summary.skipped_tasks}",
        f"Ledger Unchanged: {summary.ledger_hashes_unchanged}",
        "",
        f"Duration: {_format_duration(summary.duration_seconds)}",
        "",
        f"Output directory: {summary.output_directory.as_posix()}",
        f"Report directory: {summary.report_directory.as_posix()}",
    ]
    if summary.failed_task_labels:
        lines.extend(["", "Failed Tasks", ""])
        lines.extend(f"- {label}" for label in summary.failed_task_labels)
    return "\n".join(lines) + "\n"


async def main(argv: Sequence[str] | None = None) -> int:
    """Run the purged-CV-evaluation CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        options = build_options(args)
        _configure_logging(verbose=options.verbose, debug=options.debug)
        layout = StorageLayout(options.storage_root)
        datastore = ParquetStore()
        purged_cv_repository = PurgedCVRepository(layout, datastore)
        walk_forward_repository = WalkForwardRepository(layout, datastore)
        evaluation_repository = PurgedCVEvaluationRepository(layout, datastore)
        factor_selection_repository = FactorSelectionRepository(layout, datastore)
        walk_forward_input_builder = WalkForwardInputBuilder(
            FactorsRepository(layout, datastore),
            LabelRepository(layout, datastore),
        )
        work = discover_work(purged_cv_repository, options)
        summary = await run_evaluation(
            purged_cv_repository=purged_cv_repository,
            walk_forward_repository=walk_forward_repository,
            evaluation_repository=evaluation_repository,
            factor_selection_repository=factor_selection_repository,
            walk_forward_input_builder=walk_forward_input_builder,
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


async def run_evaluation(
    *,
    purged_cv_repository: PurgedCVRepository,
    walk_forward_repository: WalkForwardRepository,
    evaluation_repository: PurgedCVEvaluationRepository,
    factor_selection_repository: FactorSelectionRepository,
    walk_forward_input_builder: WalkForwardInputBuilder,
    options: PurgedCVEvaluationOptions,
    work: Sequence[DiscoveredWorkItem],
) -> PurgedCVEvaluationSummary:
    """Execute discovered evaluation work and write CSV reports."""
    started = time.perf_counter()
    output_directory = options.storage_root / STORAGE_DIR_PURGED_CV_EVALUATION
    if len(work) == 0:
        PurgedCVEvaluationReporter(options.report_output).write_reports(
            summaries=pl.DataFrame(),
            fold_metrics=pl.DataFrame(),
            factor_metrics=pl.DataFrame(),
        )
        return PurgedCVEvaluationSummary(
            manager=options.manager,
            engine=options.engine,
            panels=0,
            rows=0,
            folds=0,
            successful_tasks=0,
            failed_tasks=0,
            skipped_tasks=0,
            duration_seconds=time.perf_counter() - started,
            output_directory=output_directory,
            report_directory=options.report_output,
            failed_task_labels=(),
            ledger_hashes_unchanged=True,
        )

    work_by_timeframe = _group_work_by_timeframe(work)
    results = await _run_worker_pool(
        purged_cv_repository=purged_cv_repository,
        walk_forward_repository=walk_forward_repository,
        evaluation_repository=evaluation_repository,
        factor_selection_repository=factor_selection_repository,
        walk_forward_input_builder=walk_forward_input_builder,
        work_by_timeframe=work_by_timeframe,
        options=options,
    )
    _write_reports(results, report_output=options.report_output)
    return _build_summary(
        options=options,
        work=work,
        results=results,
        duration_seconds=time.perf_counter() - started,
        output_directory=output_directory,
    )


def _write_reports(
    results: Sequence[PurgedCVEvaluationTaskResult],
    *,
    report_output: Path,
) -> None:
    """Aggregate successful task artifacts into CSV reports."""
    summaries: list[pl.DataFrame] = []
    folds: list[pl.DataFrame] = []
    factors: list[pl.DataFrame] = []
    for result in results:
        if result.status != "succeeded":
            continue
        if result.summary_row is not None:
            summaries.append(result.summary_row)
        if result.fold_metrics is not None:
            folds.append(result.fold_metrics)
        if result.factor_metrics is not None:
            factors.append(result.factor_metrics)
    reporter = PurgedCVEvaluationReporter(report_output)
    reporter.write_reports(
        summaries=pl.concat(summaries) if summaries else pl.DataFrame(),
        fold_metrics=pl.concat(folds) if folds else pl.DataFrame(),
        factor_metrics=pl.concat(factors) if factors else pl.DataFrame(),
    )


def _configure_logging(*, verbose: bool, debug: bool) -> None:
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
    partitions: Sequence[PurgedCVPartitionRef],
    *,
    year_filter: tuple[int, ...] | None,
) -> tuple[DiscoveredWorkItem, ...]:
    year_allowlist = set(year_filter) if year_filter is not None else None
    grouped: dict[tuple[str, str], list[int]] = {}
    for partition in partitions:
        if year_allowlist is not None and partition.year not in year_allowlist:
            continue
        key = (partition.manager, partition.timeframe)
        grouped.setdefault(key, []).append(partition.year)
    items: list[DiscoveredWorkItem] = []
    for (manager, timeframe), years in sorted(grouped.items()):
        items.append(
            DiscoveredWorkItem(
                manager=manager,
                timeframe=timeframe,
                years=tuple(sorted(set(years))),
            )
        )
    return tuple(items)


def _group_work_by_timeframe(
    work: Sequence[DiscoveredWorkItem],
) -> dict[Timeframe, tuple[DiscoveredWorkItem, ...]]:
    grouped: dict[Timeframe, list[DiscoveredWorkItem]] = {}
    for item in work:
        grouped.setdefault(item.timeframe, []).append(item)
    return {
        timeframe: tuple(items)
        for timeframe, items in sorted(grouped.items(), key=lambda pair: pair[0])
    }


async def _run_worker_pool(
    *,
    purged_cv_repository: PurgedCVRepository,
    walk_forward_repository: WalkForwardRepository,
    evaluation_repository: PurgedCVEvaluationRepository,
    factor_selection_repository: FactorSelectionRepository,
    walk_forward_input_builder: WalkForwardInputBuilder,
    work_by_timeframe: Mapping[Timeframe, Sequence[DiscoveredWorkItem]],
    options: PurgedCVEvaluationOptions,
) -> tuple[PurgedCVEvaluationTaskResult, ...]:
    timeframes = tuple(work_by_timeframe.keys())
    if len(timeframes) == 0:
        return ()

    queue: asyncio.Queue[Timeframe | None] = asyncio.Queue()
    for timeframe in timeframes:
        queue.put_nowait(timeframe)
    for _ in range(options.workers):
        queue.put_nowait(None)

    collected: list[PurgedCVEvaluationTaskResult] = []
    lock = asyncio.Lock()

    async def worker() -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                results = await _evaluate_timeframe_work(
                    purged_cv_repository=purged_cv_repository,
                    walk_forward_repository=walk_forward_repository,
                    evaluation_repository=evaluation_repository,
                    factor_selection_repository=factor_selection_repository,
                    walk_forward_input_builder=walk_forward_input_builder,
                    timeframe=item,
                    work_items=work_by_timeframe[item],
                    options=options,
                )
                async with lock:
                    collected.extend(results)
            finally:
                queue.task_done()

    worker_tasks = [
        asyncio.create_task(worker(), name=f"evaluate-purged-cv-worker-{index}")
        for index in range(options.workers)
    ]
    try:
        await asyncio.gather(*worker_tasks)
    finally:
        for task in worker_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*worker_tasks, return_exceptions=True)

    return tuple(sorted(collected, key=lambda result: (result.timeframe, result.year)))


async def _evaluate_timeframe_work(
    *,
    purged_cv_repository: PurgedCVRepository,
    walk_forward_repository: WalkForwardRepository,
    evaluation_repository: PurgedCVEvaluationRepository,
    factor_selection_repository: FactorSelectionRepository,
    walk_forward_input_builder: WalkForwardInputBuilder,
    timeframe: Timeframe,
    work_items: Sequence[DiscoveredWorkItem],
    options: PurgedCVEvaluationOptions,
) -> tuple[PurgedCVEvaluationTaskResult, ...]:
    results: list[PurgedCVEvaluationTaskResult] = []
    for item in work_items:
        for year in item.years:
            result = await asyncio.to_thread(
                _evaluate_partition,
                purged_cv_repository,
                walk_forward_repository,
                evaluation_repository,
                factor_selection_repository,
                walk_forward_input_builder,
                manager=item.manager,
                timeframe=timeframe,
                year=year,
                options=options,
            )
            _print_progress(result)
            results.append(result)
    return tuple(results)


def _evaluate_partition(
    purged_cv_repository: PurgedCVRepository,
    walk_forward_repository: WalkForwardRepository,
    evaluation_repository: PurgedCVEvaluationRepository,
    factor_selection_repository: FactorSelectionRepository,
    walk_forward_input_builder: WalkForwardInputBuilder,
    *,
    manager: str,
    timeframe: Timeframe,
    year: int,
    options: PurgedCVEvaluationOptions,
) -> PurgedCVEvaluationTaskResult:
    """Evaluate one Purged-CV ledger year partition synchronously."""
    layout = StorageLayout(options.storage_root)
    ledger_path = layout.purged_cv_path(
        options.manager,
        _EXCHANGE,
        _MARKET,
        timeframe,
        year,
    )
    if not overwrite_and_exists_skip(
        evaluation_repository=evaluation_repository,
        manager=options.manager,
        timeframe=timeframe,
        year=year,
        overwrite=options.overwrite,
    ):
        return PurgedCVEvaluationTaskResult(
            timeframe=timeframe,
            year=year,
            status="skipped",
            ledger_sha256_before=_sha256_file(ledger_path),
            ledger_sha256_after=_sha256_file(ledger_path),
        )

    sha_before = _sha256_file(ledger_path)
    try:
        if not purged_cv_repository.exists(
            manager=options.manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            timeframe=timeframe,
            year=year,
        ):
            raise ValidationError(
                f"purged-CV ledger missing for {options.manager}/{timeframe}/{year}",
                error_code=_ERROR_PURGED_CV_MISSING,
                details={
                    "manager": options.manager,
                    "timeframe": timeframe,
                    "year": year,
                    "tier": STORAGE_DIR_PURGED_CV,
                },
            )
        if not walk_forward_repository.exists(
            manager=options.manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            timeframe=timeframe,
            year=year,
        ):
            raise ValidationError(
                f"walk-forward ledger missing for {options.manager}/{timeframe}/{year}",
                error_code=_ERROR_WALK_FORWARD_MISSING,
                details={
                    "manager": options.manager,
                    "timeframe": timeframe,
                    "year": year,
                    "tier": STORAGE_DIR_WALK_FORWARD,
                },
            )

        purged_cv = purged_cv_repository.load(
            manager=options.manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            timeframe=timeframe,
            year=year,
        )
        walk_forward = walk_forward_repository.load(
            manager=options.manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            timeframe=timeframe,
            year=year,
        )
        evaluation_input = _try_build_evaluation_input(
            factor_selection_repository=factor_selection_repository,
            walk_forward_input_builder=walk_forward_input_builder,
            manager=manager,
            timeframe=timeframe,
            year=year,
        )
        artifacts = PurgedCVEvaluator().evaluate(
            purged_cv,
            walk_forward,
            manager=options.manager,
            engine=options.engine,
            exchange=_EXCHANGE,
            market=_MARKET,
            year=year,
            evaluation_input=evaluation_input,
        )
        evaluation_repository.save(
            artifacts.observations,
            manager=options.manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            timeframe=timeframe,
            year=year,
        )
        sha_after = _sha256_file(ledger_path)
        summary = artifacts.summary
        return PurgedCVEvaluationTaskResult(
            timeframe=timeframe,
            year=year,
            status="succeeded",
            rows_generated=artifacts.observations.height,
            folds=int(summary["folds"][0]),
            oos_rows=int(summary["oos_rows"][0]),
            oos_return_mean=_as_optional_float(summary["oos_return_mean"][0]),
            unique_selected_factors=int(summary["unique_selected_factors"][0]),
            panel_status=str(summary["status"][0]),
            ledger_sha256_before=sha_before,
            ledger_sha256_after=sha_after,
            summary_row=summary,
            fold_metrics=artifacts.fold_metrics,
            factor_metrics=artifacts.factor_metrics,
        )
    except Exception as exc:
        if options.debug:
            _logger.exception(
                "Purged-CV evaluation failed",
                extra={"timeframe": timeframe, "year": year},
            )
        error_code = getattr(exc, "error_code", None)
        return PurgedCVEvaluationTaskResult(
            timeframe=timeframe,
            year=year,
            status="failed",
            ledger_sha256_before=sha_before,
            ledger_sha256_after=_sha256_file(ledger_path),
            error_type=type(exc).__name__,
            error_message=str(exc),
            error_code=str(error_code) if error_code is not None else None,
        )


def _try_build_evaluation_input(
    *,
    factor_selection_repository: FactorSelectionRepository,
    walk_forward_input_builder: WalkForwardInputBuilder,
    manager: str,
    timeframe: Timeframe,
    year: int,
) -> pl.DataFrame | None:
    """Build Labels/Factors evaluation input when upstream partitions exist."""
    if not factor_selection_repository.exists(
        manager=manager,
        exchange=_EXCHANGE,
        market=_MARKET,
        timeframe=timeframe,
        year=year,
    ):
        _logger.warning(
            "Factor Selection missing; evaluating purged-CV without Labels enrichment",
            extra={"manager": manager, "timeframe": timeframe, "year": year},
        )
        return None
    try:
        factor_selection = factor_selection_repository.load(
            manager=manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            timeframe=timeframe,
            year=year,
        )
        selected_only = factor_selection.filter(pl.col("selected"))
        return walk_forward_input_builder.build(
            factor_selection=selected_only,
            manager=manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            timeframe=timeframe,
            year=year,
        )
    except Exception as exc:
        _logger.warning(
            "Failed to assemble evaluation input; continuing without Labels metrics: %s",
            exc,
            extra={"manager": manager, "timeframe": timeframe, "year": year},
        )
        return None


def overwrite_and_exists_skip(
    *,
    evaluation_repository: PurgedCVEvaluationRepository,
    manager: str,
    timeframe: Timeframe,
    year: int,
    overwrite: bool,
) -> bool:
    """Return ``True`` when evaluation should run for the partition."""
    if overwrite:
        return True
    return not evaluation_repository.exists(
        manager=manager,
        exchange=_EXCHANGE,
        market=_MARKET,
        timeframe=timeframe,
        year=year,
    )


def _build_summary(
    *,
    options: PurgedCVEvaluationOptions,
    work: Sequence[DiscoveredWorkItem],
    results: Sequence[PurgedCVEvaluationTaskResult],
    duration_seconds: float,
    output_directory: Path,
) -> PurgedCVEvaluationSummary:
    panels = sum(len(item.years) for item in work)
    successful = [result for result in results if result.status == "succeeded"]
    failed = [result for result in results if result.status == "failed"]
    skipped = [result for result in results if result.status == "skipped"]
    hashes_unchanged = all(
        result.ledger_sha256_before == result.ledger_sha256_after
        for result in results
        if result.ledger_sha256_before is not None and result.ledger_sha256_after is not None
    )
    return PurgedCVEvaluationSummary(
        manager=options.manager,
        engine=options.engine,
        panels=panels,
        rows=sum(result.rows_generated or 0 for result in successful),
        folds=sum(result.folds or 0 for result in successful),
        successful_tasks=len(successful),
        failed_tasks=len(failed),
        skipped_tasks=len(skipped),
        duration_seconds=duration_seconds,
        output_directory=output_directory,
        report_directory=options.report_output,
        failed_task_labels=tuple(
            f"{result.timeframe}/{result.year}: {result.error_message}" for result in failed
        ),
        ledger_hashes_unchanged=hashes_unchanged,
    )


def _print_progress(result: PurgedCVEvaluationTaskResult) -> None:
    if result.status == "succeeded":
        print(
            f"[ok] {result.timeframe}/{result.year} "
            f"rows={result.rows_generated} folds={result.folds} "
            f"status={result.panel_status}",
            flush=True,
        )
    elif result.status == "skipped":
        print(f"[skip] {result.timeframe}/{result.year}", flush=True)
    else:
        print(
            f"[fail] {result.timeframe}/{result.year}: {result.error_message}",
            flush=True,
        )


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _as_optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    if value != value:
        return None
    return float(value)


def _format_duration(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60.0)
    remainder = seconds - float(minutes * 60)
    return f"{minutes}m {remainder:.2f}s"


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
