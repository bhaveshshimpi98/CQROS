"""CQROS walk-forward evaluation CLI.

Purpose:
    Discover existing Walk-Forward ledger partitions, rebuild the enriched
    evaluation input (Factors + Labels ``future_return_1`` + Factor Selection),
    persist a separate evaluation-result artifact, and write CSV reports.

Responsibilities:
    - Parse CLI arguments for walk-forward evaluation
    - Discover Walk-Forward ledger partitions (never invent timeframes)
    - Rebuild evaluation input via ``WalkForwardInputBuilder``
    - Execute ``WalkForwardEvaluator`` and persist via
      ``WalkForwardEvaluationRepository``
    - Write evaluation CSV reports under ``reports/walk_forward``
    - Leave the Walk-Forward ledger byte-identical
    - Remain free of Alpha, Regime, Predictions, Signals, and ``cqros.ml``

Dependencies:
    ``argparse``, ``asyncio``, ``hashlib``, ``polars``, ``cqros.config``,
    ``cqros.core``, ``cqros.factor_selection``, ``cqros.factors``,
    ``cqros.walk_forward``, ``cqros.reporting.walk_forward_evaluation_report``,
    and ``cqros.storage``.

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
    STORAGE_DIR_WALK_FORWARD,
    STORAGE_DIR_WALK_FORWARD_EVALUATION,
    SUPPORTED_TIMEFRAMES,
)
from cqros.core.exceptions import CQROSError, ValidationError
from cqros.core.types import Timeframe
from cqros.factor_selection import FactorSelectionRepository
from cqros.factors import FactorsRepository
from cqros.reporting.walk_forward_evaluation_report import (
    DEFAULT_OUTPUT_ROOT,
    WalkForwardEvaluationReporter,
)
from cqros.storage import LabelRepository, ParquetStore, StorageLayout
from cqros.walk_forward import (
    WalkForwardInputBuilder,
    WalkForwardPartitionRef,
    WalkForwardRepository,
)
from cqros.walk_forward.evaluation import WalkForwardEvaluator
from cqros.walk_forward.evaluation_repository import WalkForwardEvaluationRepository

__all__ = [
    "DiscoveredWorkItem",
    "WalkForwardEvaluationOptions",
    "WalkForwardEvaluationSummary",
    "WalkForwardEvaluationTaskResult",
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

_ERROR_WORKERS: Final[str] = "CLI-EVALUATE-WALK-FORWARD-001"
_ERROR_TIMEFRAME: Final[str] = "CLI-EVALUATE-WALK-FORWARD-002"
_ERROR_YEAR: Final[str] = "CLI-EVALUATE-WALK-FORWARD-003"
_ERROR_MANAGER: Final[str] = "CLI-EVALUATE-WALK-FORWARD-004"
_ERROR_ENGINE: Final[str] = "CLI-EVALUATE-WALK-FORWARD-005"
_ERROR_FACTOR_SELECTION_MISSING: Final[str] = "CLI-EVALUATE-WALK-FORWARD-006"
_ERROR_LEDGER_MISSING: Final[str] = "CLI-EVALUATE-WALK-FORWARD-007"


@dataclass(frozen=True, slots=True)
class WalkForwardEvaluationOptions:
    """Immutable CLI options for walk-forward evaluation."""

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
    """One discovered Walk-Forward ledger panel ready for evaluation."""

    manager: str
    timeframe: Timeframe
    years: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class WalkForwardEvaluationTaskResult:
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
class WalkForwardEvaluationSummary:
    """Immutable aggregate summary for a walk-forward evaluation run."""

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
    """Create the walk-forward-evaluation argument parser."""
    parser = argparse.ArgumentParser(
        prog="cqros-evaluate-walk-forward",
        description=(
            "Evaluate CQROS Walk-Forward ledgers into a separate evaluation-result "
            "artifact and CSV reports. Discovers existing walk_forward partitions."
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
        help=f"Walk-forward engine label recorded on artifacts (default: {_DEFAULT_ENGINE}).",
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
        "--report-output",
        dest="report_output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        metavar="PATH",
        help=f"CSV report directory (default: {DEFAULT_OUTPUT_ROOT.as_posix()}).",
    )
    return parser


def build_options(args: argparse.Namespace) -> WalkForwardEvaluationOptions:
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
    return WalkForwardEvaluationOptions(
        storage_root=storage_root,
        manager=manager,
        engine=engine,
        timeframes=_normalize_timeframes(args.timeframes),
        years=_normalize_years(args.years),
        overwrite=bool(args.overwrite),
        workers=workers,
        verbose=bool(args.verbose),
        debug=bool(args.debug),
        report_output=Path(args.report_output),
    )


def discover_work(
    walk_forward_repository: WalkForwardRepository,
    options: WalkForwardEvaluationOptions,
) -> tuple[DiscoveredWorkItem, ...]:
    """Discover Walk-Forward ledger panels matching CLI filters."""
    partitions = walk_forward_repository.discover_partitions(
        managers=(options.manager,),
        timeframes=options.timeframes,
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    return _group_partitions(partitions, year_filter=options.years)


def format_summary(summary: WalkForwardEvaluationSummary) -> str:
    """Render a deterministic walk-forward-evaluation summary report."""
    lines = [
        "=======================================",
        "CQROS Walk-Forward Evaluation Summary",
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
    """Run the walk-forward-evaluation CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        options = build_options(args)
        _configure_logging(verbose=options.verbose, debug=options.debug)
        layout = StorageLayout(options.storage_root)
        datastore = ParquetStore()
        walk_forward_repository = WalkForwardRepository(layout, datastore)
        evaluation_repository = WalkForwardEvaluationRepository(layout, datastore)
        factor_selection_repository = FactorSelectionRepository(layout, datastore)
        walk_forward_input_builder = WalkForwardInputBuilder(
            FactorsRepository(layout, datastore),
            LabelRepository(layout, datastore),
        )
        work = discover_work(walk_forward_repository, options)
        summary = await run_evaluation(
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
    walk_forward_repository: WalkForwardRepository,
    evaluation_repository: WalkForwardEvaluationRepository,
    factor_selection_repository: FactorSelectionRepository,
    walk_forward_input_builder: WalkForwardInputBuilder,
    options: WalkForwardEvaluationOptions,
    work: Sequence[DiscoveredWorkItem],
) -> WalkForwardEvaluationSummary:
    """Execute discovered evaluation work and write CSV reports."""
    started = time.perf_counter()
    output_directory = options.storage_root / STORAGE_DIR_WALK_FORWARD_EVALUATION
    if len(work) == 0:
        WalkForwardEvaluationReporter(options.report_output).write_reports(
            summaries=pl.DataFrame(),
            fold_metrics=pl.DataFrame(),
            factor_metrics=pl.DataFrame(),
        )
        return WalkForwardEvaluationSummary(
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
    results: Sequence[WalkForwardEvaluationTaskResult],
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
    reporter = WalkForwardEvaluationReporter(report_output)
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
    partitions: Sequence[WalkForwardPartitionRef],
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
    walk_forward_repository: WalkForwardRepository,
    evaluation_repository: WalkForwardEvaluationRepository,
    factor_selection_repository: FactorSelectionRepository,
    walk_forward_input_builder: WalkForwardInputBuilder,
    work_by_timeframe: Mapping[Timeframe, Sequence[DiscoveredWorkItem]],
    options: WalkForwardEvaluationOptions,
) -> tuple[WalkForwardEvaluationTaskResult, ...]:
    timeframes = tuple(work_by_timeframe.keys())
    if len(timeframes) == 0:
        return ()

    queue: asyncio.Queue[Timeframe | None] = asyncio.Queue()
    for timeframe in timeframes:
        queue.put_nowait(timeframe)
    for _ in range(options.workers):
        queue.put_nowait(None)

    collected: list[WalkForwardEvaluationTaskResult] = []
    lock = asyncio.Lock()

    async def worker() -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                results = await _evaluate_timeframe_work(
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
        asyncio.create_task(worker(), name=f"evaluate-walk-forward-worker-{index}")
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
    walk_forward_repository: WalkForwardRepository,
    evaluation_repository: WalkForwardEvaluationRepository,
    factor_selection_repository: FactorSelectionRepository,
    walk_forward_input_builder: WalkForwardInputBuilder,
    timeframe: Timeframe,
    work_items: Sequence[DiscoveredWorkItem],
    options: WalkForwardEvaluationOptions,
) -> tuple[WalkForwardEvaluationTaskResult, ...]:
    results: list[WalkForwardEvaluationTaskResult] = []
    for item in work_items:
        for year in item.years:
            result = await asyncio.to_thread(
                _evaluate_partition,
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
    walk_forward_repository: WalkForwardRepository,
    evaluation_repository: WalkForwardEvaluationRepository,
    factor_selection_repository: FactorSelectionRepository,
    walk_forward_input_builder: WalkForwardInputBuilder,
    *,
    manager: str,
    timeframe: Timeframe,
    year: int,
    options: WalkForwardEvaluationOptions,
) -> WalkForwardEvaluationTaskResult:
    """Evaluate one Walk-Forward ledger year partition synchronously."""
    layout = StorageLayout(options.storage_root)
    ledger_path = layout.walk_forward_path(
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
        return WalkForwardEvaluationTaskResult(
            timeframe=timeframe,
            year=year,
            status="skipped",
            ledger_sha256_before=_sha256_file(ledger_path),
            ledger_sha256_after=_sha256_file(ledger_path),
        )

    sha_before = _sha256_file(ledger_path)
    try:
        if not walk_forward_repository.exists(
            manager=options.manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            timeframe=timeframe,
            year=year,
        ):
            raise ValidationError(
                f"walk-forward ledger missing for {options.manager}/{timeframe}/{year}",
                error_code=_ERROR_LEDGER_MISSING,
                details={
                    "manager": options.manager,
                    "timeframe": timeframe,
                    "year": year,
                    "tier": STORAGE_DIR_WALK_FORWARD,
                },
            )
        if not factor_selection_repository.exists(
            manager=manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            timeframe=timeframe,
            year=year,
        ):
            raise ValidationError(
                f"factor selection partition missing for {manager}/{timeframe}/{year}",
                error_code=_ERROR_FACTOR_SELECTION_MISSING,
                details={
                    "manager": manager,
                    "timeframe": timeframe,
                    "year": year,
                    "missing": "factor_selection",
                },
            )

        factor_selection = factor_selection_repository.load(
            manager=manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            timeframe=timeframe,
            year=year,
        )
        # Evaluation OOS metrics use selected factors only. Filtering before the
        # Factors/Labels join avoids loading rejected-candidate panels that OOM
        # large timeframes on constrained hosts.
        selected_only = factor_selection.filter(pl.col("selected"))
        evaluation_input = walk_forward_input_builder.build(
            factor_selection=selected_only,
            manager=manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            timeframe=timeframe,
            year=year,
        )
        artifacts = WalkForwardEvaluator().evaluate(
            evaluation_input,
            manager=options.manager,
            engine=options.engine,
            year=year,
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
        return WalkForwardEvaluationTaskResult(
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
                "Walk-forward evaluation failed",
                extra={"timeframe": timeframe, "year": year},
            )
        error_code = getattr(exc, "error_code", None)
        return WalkForwardEvaluationTaskResult(
            timeframe=timeframe,
            year=year,
            status="failed",
            ledger_sha256_before=sha_before,
            ledger_sha256_after=_sha256_file(ledger_path),
            error_type=type(exc).__name__,
            error_message=str(exc),
            error_code=str(error_code) if error_code is not None else None,
        )


def overwrite_and_exists_skip(
    *,
    evaluation_repository: WalkForwardEvaluationRepository,
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
    options: WalkForwardEvaluationOptions,
    work: Sequence[DiscoveredWorkItem],
    results: Sequence[WalkForwardEvaluationTaskResult],
    duration_seconds: float,
    output_directory: Path,
) -> WalkForwardEvaluationSummary:
    panels = sum(len(item.years) for item in work)
    successful = [result for result in results if result.status == "succeeded"]
    failed = [result for result in results if result.status == "failed"]
    skipped = [result for result in results if result.status == "skipped"]
    hashes_unchanged = all(
        result.ledger_sha256_before == result.ledger_sha256_after
        for result in results
        if result.ledger_sha256_before is not None and result.ledger_sha256_after is not None
    )
    return WalkForwardEvaluationSummary(
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


def _print_progress(result: WalkForwardEvaluationTaskResult) -> None:
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
