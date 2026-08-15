"""CQROS factor timeframe analysis dataset verification CLI.

Purpose:
    Provide an argparse-based production entry point that discovers
    factor timeframe analysis panels and executes
    ``FactorTimeframeAnalysisVerifier.verify_against_selection`` across
    the universe with bounded panel concurrency.

Responsibilities:
    - Parse CLI arguments for factor timeframe analysis dataset verification
    - Discover available FTA partitions through
      ``FactorTimeframeAnalysisRepository``
    - Load FTA frames and matching Factor Selection frames via
      ``load_factor_selection_for_analysis``
    - Verify each FTA frame with ``FactorTimeframeAnalysisVerifier``
    - Aggregate results into a final PASS/FAIL report
    - Print structured per-partition failure diagnostics
    - Print the report and return an exit code

Dependencies:
    ``argparse``, ``asyncio``, ``cqros.config``, ``cqros.core``,
    ``cqros.factor_selection``, ``cqros.factor_timeframe_analysis``,
    ``cqros.processing.verification.report``, and ``cqros.storage``.

Public API:
    ``DiscoveredWorkItem``, ``FactorTimeframeAnalysisTaskResult``,
    ``VerifyFactorTimeframeAnalysisOptions``,
    ``VerifyFactorTimeframeAnalysisSummary``, ``build_options``,
    ``build_parser``, ``discover_work``, ``format_partition_failure``,
    ``format_summary``, ``main``, ``run_verification``.

Notes:
    This module is a thin composition root. It does not implement
    verification logic or repository filesystem walks beyond calling
    repository discovery and load APIs. FTA partitions are cross-sectional
    panels keyed by manager/year (no symbol).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from cqros.config.models import ResearchConfig
from cqros.core.constants import (
    DEFAULT_STORAGE_ROOT,
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
)
from cqros.core.exceptions import CQROSError, ValidationError
from cqros.factor_selection import FactorSelectionRepository
from cqros.factor_timeframe_analysis import (
    FactorTimeframeAnalysisPartitionRef,
    FactorTimeframeAnalysisRepository,
    FactorTimeframeAnalysisVerifier,
    load_factor_selection_for_analysis,
)
from cqros.processing.verification.report import VerificationReport
from cqros.storage import ParquetStore, StorageLayout

__all__ = [
    "DiscoveredWorkItem",
    "FactorTimeframeAnalysisTaskResult",
    "VerifyFactorTimeframeAnalysisOptions",
    "VerifyFactorTimeframeAnalysisSummary",
    "build_options",
    "build_parser",
    "discover_work",
    "format_partition_failure",
    "format_summary",
    "main",
    "run_verification",
]

_logger = logging.getLogger(__name__)

_EXIT_SUCCESS: Final[int] = 0
_EXIT_FAILURE: Final[int] = 1

_DEFAULT_WORKER_COUNT: Final[int] = ResearchConfig().worker_count

_EXCHANGE: Final[str] = EXCHANGE_BINANCE
_MARKET: Final[str] = MARKET_USDT_PERPETUAL

_ERROR_WORKERS: Final[str] = "CLI-VERIFY-FTA-001"
_ERROR_YEAR: Final[str] = "CLI-VERIFY-FTA-002"
_ERROR_MANAGER: Final[str] = "CLI-VERIFY-FTA-003"

_VERIFIER_NAME: Final[str] = "FactorTimeframeAnalysisVerifier"
_DATASET_DISPLAY_NAME: Final[str] = "Factor Timeframe Analysis"


@dataclass(frozen=True, slots=True)
class VerifyFactorTimeframeAnalysisOptions:
    """Immutable CLI options for factor timeframe analysis dataset verification.

    Attributes:
        storage_root: Storage root containing ``factor_timeframe_analysis``
            and ``factor_selection``.
        manager: Optional manager allowlist identity. ``None`` discovers all.
        years: Optional year allowlist. ``None`` discovers all.
        workers: Maximum concurrent panels.
        verbose: When ``True``, enable INFO logging.
        debug: When ``True``, enable DEBUG logging and log full failure
            tracebacks.
    """

    storage_root: Path
    manager: str | None
    years: tuple[int, ...] | None
    workers: int
    verbose: bool
    debug: bool


@dataclass(frozen=True, slots=True)
class DiscoveredWorkItem:
    """One discovered FTA panel ready for verification.

    Attributes:
        manager: Order manager identifier.
        year: Calendar year of the FTA partition.
    """

    manager: str
    year: int


@dataclass(frozen=True, slots=True)
class FactorTimeframeAnalysisTaskResult:
    """Immutable result for one manager/year FTA verification task.

    Attributes:
        manager: Order manager identifier.
        year: Calendar year of the partition.
        status: ``succeeded`` or ``failed``.
        report: Verifier report when succeeded.
        error_type: Exception type name when failed.
        error_message: Exception message when failed.
        error_code: Optional CQROS error code when failed.
    """

    manager: str
    year: int
    status: str
    report: VerificationReport | None = None
    error_type: str | None = None
    error_message: str | None = None
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class VerifyFactorTimeframeAnalysisSummary:
    """Immutable aggregate summary for a FTA verification run.

    Attributes:
        panels_verified: Unique manager/year panels attempted.
        datasets_verified: Unique datasets attempted (always FTA).
        managers_verified: Unique managers attempted.
        successful_tasks: Count of succeeded partition tasks.
        failed_tasks: Count of failed partition tasks.
        rows_checked: Sum of verifier ``rows_checked`` across successes.
        duplicate_timestamps: Sum of duplicate-primary-key counters.
        null_rows: Sum of null-row counters.
        nan_rows: Sum of NaN-row counters.
        invalid_timestamps: Sum of invalid-timestamp counters.
        invalid_numeric_rows: Sum of invalid-numeric-row counters.
        warnings: Sum of warning counts across successes.
        duration_seconds: Wall-clock verification duration.
        repository_passed: Whether the repository status is PASS.
    """

    panels_verified: int
    datasets_verified: int
    managers_verified: int
    successful_tasks: int
    failed_tasks: int
    rows_checked: int
    duplicate_timestamps: int
    null_rows: int
    nan_rows: int
    invalid_timestamps: int
    invalid_numeric_rows: int
    warnings: int
    duration_seconds: float
    repository_passed: bool


def build_parser() -> argparse.ArgumentParser:
    """Create the factor timeframe analysis dataset verification argument parser.

    Returns:
        Configured ``ArgumentParser`` for verification flags.
    """
    parser = argparse.ArgumentParser(
        prog="cqros-verify-factor-timeframe-analysis",
        description=(
            "Verify CQROS factor timeframe analysis datasets across the " "discovered FTA universe."
        ),
    )
    parser.add_argument(
        "--manager",
        dest="manager",
        default=None,
        metavar="NAME",
        help=(
            "Optional order-manager filter applied to partition discovery. "
            "Omit to verify all managers present under the FTA tier."
        ),
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


def build_options(args: argparse.Namespace) -> VerifyFactorTimeframeAnalysisOptions:
    """Map parsed CLI arguments onto ``VerifyFactorTimeframeAnalysisOptions``.

    Args:
        args: Namespace produced by ``build_parser().parse_args(...)``.

    Returns:
        Immutable verification options.

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

    manager = _normalize_optional_identity(
        args.manager,
        parameter="manager",
        error_code=_ERROR_MANAGER,
    )

    storage_root = (
        Path(args.storage_root) if args.storage_root is not None else Path(DEFAULT_STORAGE_ROOT)
    )

    return VerifyFactorTimeframeAnalysisOptions(
        storage_root=storage_root,
        manager=manager,
        years=_normalize_years(args.years),
        workers=workers,
        verbose=bool(args.verbose),
        debug=bool(args.debug),
    )


def discover_work(
    fta_repository: FactorTimeframeAnalysisRepository,
    options: VerifyFactorTimeframeAnalysisOptions,
) -> tuple[DiscoveredWorkItem, ...]:
    """Discover FTA partitions matching the CLI filters.

    Args:
        fta_repository: FTA repository providing discovery APIs.
        options: CLI filters for manager and year.

    Returns:
        Deterministically ordered discovered work items.
    """
    managers = (options.manager,) if options.manager is not None else None
    partitions = fta_repository.discover_partitions(
        managers=managers,
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    return _group_partitions(partitions, year_filter=options.years)


def format_partition_failure(
    *,
    dataset: str,
    manager: str,
    year: int,
    partition: str,
    verifier: str,
    exception_type: str,
    message: str,
    code: str | None = None,
) -> str:
    """Render a structured per-partition verification failure report.

    Args:
        dataset: Human-readable dataset name.
        manager: Order manager identifier.
        year: Calendar year of the panel.
        partition: Partition label (typically ``{year}.parquet``).
        verifier: Verifier class name.
        exception_type: Exception type name.
        message: Exception message.
        code: Optional CQROS error code.

    Returns:
        Multi-line failure report string.
    """
    lines = [
        "FAILED",
        "",
        f"Dataset: {dataset}",
        f"Manager: {manager}",
        f"Year: {year}",
        f"Partition: {partition}",
        f"Verifier: {verifier}",
        f"Exception: {exception_type}",
    ]
    if code is not None:
        lines.append(f"Code: {code}")
    lines.append(f"Message: {message}")
    return "\n".join(lines) + "\n"


def format_summary(summary: VerifyFactorTimeframeAnalysisSummary) -> str:
    """Render a deterministic FTA verification summary report.

    Args:
        summary: Aggregate verification summary.

    Returns:
        Multi-line summary text suitable for stdout.
    """
    status = "PASS" if summary.repository_passed else "FAIL"
    lines = [
        "=====================================",
        "CQROS FTA Verification Summary",
        "=====================================",
        "",
        f"Panels verified: {summary.panels_verified}",
        f"Datasets verified: {summary.datasets_verified}",
        f"Managers verified: {summary.managers_verified}",
        "",
        f"Successful tasks: {summary.successful_tasks}",
        f"Failed tasks: {summary.failed_tasks}",
        "",
        f"Rows checked: {summary.rows_checked}",
        "",
        f"Duplicate timestamps: {summary.duplicate_timestamps}",
        f"NULL rows: {summary.null_rows}",
        f"NaN rows: {summary.nan_rows}",
        f"Invalid timestamps: {summary.invalid_timestamps}",
        f"Invalid numeric rows: {summary.invalid_numeric_rows}",
        "",
        f"Warnings: {summary.warnings}",
        "",
        f"Verification duration: {_format_duration(summary.duration_seconds)}",
        "",
        "Repository status:",
        "",
        status,
        "",
        "=====================================",
    ]
    return "\n".join(lines) + "\n"


async def main(argv: Sequence[str] | None = None) -> int:
    """Run the factor timeframe analysis dataset verification CLI.

    Args:
        argv: Optional argument vector. When ``None``, ``sys.argv[1:]`` is used.

    Returns:
        ``0`` when verification passes; ``1`` on fatal error or FAIL status.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        options = build_options(args)
        _configure_logging(verbose=options.verbose, debug=options.debug)
        layout = StorageLayout(options.storage_root)
        datastore = ParquetStore()
        fta_repository = FactorTimeframeAnalysisRepository(layout, datastore)
        fs_repository = FactorSelectionRepository(layout, datastore)
        work = discover_work(fta_repository, options)
        summary = await run_verification(
            fta_repository=fta_repository,
            fs_repository=fs_repository,
            verifier=FactorTimeframeAnalysisVerifier(),
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
    return _EXIT_SUCCESS if summary.repository_passed else _EXIT_FAILURE


async def run_verification(
    *,
    fta_repository: FactorTimeframeAnalysisRepository,
    fs_repository: FactorSelectionRepository,
    verifier: FactorTimeframeAnalysisVerifier,
    options: VerifyFactorTimeframeAnalysisOptions,
    work: Sequence[DiscoveredWorkItem],
) -> VerifyFactorTimeframeAnalysisSummary:
    """Execute discovered work through a bounded panel worker pool.

    Args:
        fta_repository: FTA partition repository.
        fs_repository: Factor Selection repository for cross-frame loading.
        verifier: FTA verifier instance.
        options: Immutable verification options.
        work: Discovered work items.

    Returns:
        Aggregate immutable summary.
    """
    started = time.perf_counter()
    if len(work) == 0:
        return VerifyFactorTimeframeAnalysisSummary(
            panels_verified=0,
            datasets_verified=0,
            managers_verified=0,
            successful_tasks=0,
            failed_tasks=0,
            rows_checked=0,
            duplicate_timestamps=0,
            null_rows=0,
            nan_rows=0,
            invalid_timestamps=0,
            invalid_numeric_rows=0,
            warnings=0,
            duration_seconds=time.perf_counter() - started,
            repository_passed=True,
        )

    results = await _run_worker_pool(
        fta_repository=fta_repository,
        fs_repository=fs_repository,
        verifier=verifier,
        work=work,
        worker_count=options.workers,
        debug=options.debug,
    )
    return _build_summary(
        results=results,
        duration_seconds=time.perf_counter() - started,
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


def _normalize_optional_identity(
    value: object | None,
    *,
    parameter: str,
    error_code: str,
) -> str | None:
    """Normalize an optional non-empty identity string filter."""
    if value is None:
        return None
    normalized = str(value).strip()
    if normalized == "":
        raise ValidationError(
            f"{parameter} must be a non-empty string when provided",
            error_code=error_code,
            details={"parameter": parameter, "value": value},
        )
    return normalized


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
    partitions: Sequence[FactorTimeframeAnalysisPartitionRef],
    *,
    year_filter: tuple[int, ...] | None,
) -> tuple[DiscoveredWorkItem, ...]:
    """Group FTA year partitions into manager/year work items."""
    year_allowlist = set(year_filter) if year_filter is not None else None
    items: list[DiscoveredWorkItem] = []
    for partition in partitions:
        if year_allowlist is not None and partition.year not in year_allowlist:
            continue
        items.append(DiscoveredWorkItem(manager=partition.manager, year=partition.year))
    return tuple(
        sorted(
            items,
            key=lambda item: (item.manager, item.year),
        )
    )


async def _run_worker_pool(
    *,
    fta_repository: FactorTimeframeAnalysisRepository,
    fs_repository: FactorSelectionRepository,
    verifier: FactorTimeframeAnalysisVerifier,
    work: Sequence[DiscoveredWorkItem],
    worker_count: int,
    debug: bool,
) -> tuple[FactorTimeframeAnalysisTaskResult, ...]:
    """Drain work items through a bounded asyncio worker pool."""
    if len(work) == 0:
        return ()

    queue: asyncio.Queue[DiscoveredWorkItem | None] = asyncio.Queue()
    for item in work:
        queue.put_nowait(item)
    for _ in range(worker_count):
        queue.put_nowait(None)

    collected: list[FactorTimeframeAnalysisTaskResult] = []
    lock = asyncio.Lock()

    async def worker() -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                result = await asyncio.to_thread(
                    _verify_partition,
                    fta_repository,
                    fs_repository,
                    verifier,
                    manager=item.manager,
                    year=item.year,
                    debug=debug,
                )
                if result.status == "failed":
                    _report_task_failure(result)
                async with lock:
                    collected.append(result)
            finally:
                queue.task_done()

    worker_tasks = [
        asyncio.create_task(worker(), name=f"verify-fta-worker-{index}")
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
            key=lambda result: (result.manager, result.year),
        )
    )


def _verify_partition(
    fta_repository: FactorTimeframeAnalysisRepository,
    fs_repository: FactorSelectionRepository,
    verifier: FactorTimeframeAnalysisVerifier,
    *,
    manager: str,
    year: int,
    debug: bool,
) -> FactorTimeframeAnalysisTaskResult:
    """Verify one FTA year partition synchronously."""
    try:
        fta_frame = fta_repository.load(
            manager=manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            year=year,
        )
        selection_frame = load_factor_selection_for_analysis(
            fs_repository,
            manager=manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            year=year,
        )
        report = verifier.verify_against_selection(fta_frame, selection_frame)
    except Exception as exc:
        _log_partition_failure(manager=manager, year=year, exc=exc, debug=debug)
        return FactorTimeframeAnalysisTaskResult(
            manager=manager,
            year=year,
            status="failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
            error_code=exc.error_code if isinstance(exc, CQROSError) else None,
        )

    return FactorTimeframeAnalysisTaskResult(
        manager=manager,
        year=year,
        status="succeeded",
        report=report,
    )


def _report_task_failure(result: FactorTimeframeAnalysisTaskResult) -> None:
    """Print structured diagnostics for a failed verification task."""
    message = result.error_message if result.error_message is not None else ""
    exception_type = result.error_type if result.error_type is not None else "Exception"
    print(
        format_partition_failure(
            dataset=_DATASET_DISPLAY_NAME,
            manager=result.manager,
            year=result.year,
            partition=_partition_label(result.year),
            verifier=_VERIFIER_NAME,
            exception_type=exception_type,
            message=message,
            code=result.error_code,
        ),
        end="",
        flush=True,
    )


def _log_partition_failure(
    *,
    manager: str,
    year: int,
    exc: BaseException,
    debug: bool,
) -> None:
    """Log a partition verification failure without aborting the run."""
    log_extra = {
        "manager": manager,
        "year": year,
        "verifier": _VERIFIER_NAME,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "error_code": exc.error_code if isinstance(exc, CQROSError) else None,
    }
    if debug:
        _logger.exception(
            "Failed FTA partition verification; continuing",
            extra=log_extra,
            exc_info=True,
        )
    else:
        _logger.warning(
            "Failed FTA partition verification; continuing",
            extra=log_extra,
        )


def _build_summary(
    *,
    results: Sequence[FactorTimeframeAnalysisTaskResult],
    duration_seconds: float,
) -> VerifyFactorTimeframeAnalysisSummary:
    """Aggregate task results into a verification report."""
    panels_verified: set[tuple[str, int]] = set()
    managers_verified: set[str] = set()
    successful_tasks = 0
    failed_tasks = 0
    rows_checked = 0
    duplicate_timestamps = 0
    null_rows = 0
    nan_rows = 0
    invalid_timestamps = 0
    invalid_numeric_rows = 0
    warnings = 0

    for result in results:
        panels_verified.add((result.manager, result.year))
        managers_verified.add(result.manager)
        if result.status == "succeeded":
            successful_tasks += 1
            report = result.report
            if report is not None:
                rows_checked += report.rows_checked
                duplicate_timestamps += report.duplicate_timestamp_rows
                null_rows += report.null_rows
                nan_rows += report.nan_rows
                invalid_timestamps += report.invalid_timestamp_rows
                invalid_numeric_rows += report.invalid_numeric_rows
                warnings += len(report.warnings)
        else:
            failed_tasks += 1

    repository_passed = (
        failed_tasks == 0
        and duplicate_timestamps == 0
        and null_rows == 0
        and nan_rows == 0
        and invalid_timestamps == 0
        and invalid_numeric_rows == 0
        and warnings == 0
    )

    datasets_verified = 1 if results else 0
    return VerifyFactorTimeframeAnalysisSummary(
        panels_verified=len(panels_verified),
        datasets_verified=datasets_verified,
        managers_verified=len(managers_verified),
        successful_tasks=successful_tasks,
        failed_tasks=failed_tasks,
        rows_checked=rows_checked,
        duplicate_timestamps=duplicate_timestamps,
        null_rows=null_rows,
        nan_rows=nan_rows,
        invalid_timestamps=invalid_timestamps,
        invalid_numeric_rows=invalid_numeric_rows,
        warnings=warnings,
        duration_seconds=duration_seconds,
        repository_passed=repository_passed,
    )


def _partition_label(year: int) -> str:
    """Return the partition filename identifier for a calendar year."""
    return f"{year}.parquet"


def _format_duration(duration_seconds: float) -> str:
    """Format duration seconds for the summary report."""
    return f"{duration_seconds:.3f}s"


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
