"""CQROS factor selection dataset verification CLI.

Purpose:
    Provide an argparse-based production entry point that discovers
    factor selection panels and executes ``FactorSelectionVerifier``
    across the universe with bounded panel concurrency.

Responsibilities:
    - Parse CLI arguments for factor selection dataset verification
    - Discover available partitions through ``FactorSelectionRepository``
    - Load partitions and verify them with ``FactorSelectionVerifier``
    - Aggregate results into a final PASS/FAIL report
    - Print structured per-partition failure diagnostics
    - Print the report and return an exit code

Dependencies:
    ``argparse``, ``asyncio``, ``cqros.config``, ``cqros.core``,
    ``cqros.factor_selection``, ``cqros.processing.verification.report``,
    and ``cqros.storage``.

Public API:
    ``build_parser``, ``build_options``, ``discover_work``,
    ``format_partition_failure``, ``format_summary``, ``run_verification``,
    and ``main``.

Notes:
    This module is a thin composition root. It does not implement verification
    logic or repository filesystem walks beyond calling repository discovery
    and load APIs. Partitions are cross-sectional panels keyed by
    manager/timeframe/year (no symbol).
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

from cqros.config.models import ResearchConfig
from cqros.core.constants import (
    DEFAULT_STORAGE_ROOT,
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
    SUPPORTED_TIMEFRAMES,
)
from cqros.core.exceptions import CQROSError, ValidationError
from cqros.core.types import Timeframe
from cqros.factor_selection import FactorSelectionRepository, FactorSelectionVerifier
from cqros.factor_selection.engine import DEFAULT_TOP_N
from cqros.factor_selection.redundancy import DEFAULT_CANDIDATE_N
from cqros.factor_selection.repository import FactorSelectionPartitionRef
from cqros.processing.verification.report import VerificationReport
from cqros.storage import ParquetStore, StorageLayout

__all__ = [
    "DiscoveredWorkItem",
    "FactorSelectionTaskResult",
    "VerifyFactorSelectionOptions",
    "VerifyFactorSelectionSummary",
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

_ERROR_WORKERS: Final[str] = "CLI-VERIFY-FACTOR-SELECTION-001"
_ERROR_TIMEFRAME: Final[str] = "CLI-VERIFY-FACTOR-SELECTION-002"
_ERROR_YEAR: Final[str] = "CLI-VERIFY-FACTOR-SELECTION-003"
_ERROR_MANAGER: Final[str] = "CLI-VERIFY-FACTOR-SELECTION-004"
_ERROR_TOP_N: Final[str] = "CLI-VERIFY-FACTOR-SELECTION-005"

_VERIFIER_NAME: Final[str] = "FactorSelectionVerifier"
_DATASET_DISPLAY_NAME: Final[str] = "Factor Selection"

_WARN_EMPTY_STATUS: Final[str] = "Empty status values detected."
_WARN_INVALID_STATUS: Final[str] = "Invalid FactorSelectionStatus values detected."


@dataclass(frozen=True, slots=True)
class VerifyFactorSelectionOptions:
    """Immutable CLI options for factor selection dataset verification.

    Attributes:
        storage_root: Storage root containing ``factor_selection``.
        manager: Optional manager allowlist identity. ``None`` discovers all.
        top_n: Top-N limit expected by the ranking verifier.
        candidate_n: Candidate pool size expected by the ranking verifier.
        timeframes: Optional timeframe allowlist. ``None`` discovers all.
        years: Optional year allowlist. ``None`` discovers all.
        workers: Maximum concurrent panels.
        verbose: When ``True``, enable INFO logging.
        debug: When ``True``, enable DEBUG logging and log full failure
            tracebacks.
    """

    storage_root: Path
    manager: str | None
    top_n: int
    candidate_n: int
    timeframes: tuple[Timeframe, ...] | None
    years: tuple[int, ...] | None
    workers: int
    verbose: bool
    debug: bool


@dataclass(frozen=True, slots=True)
class DiscoveredWorkItem:
    """One discovered factor selection panel group ready for verification.

    Attributes:
        manager: Order manager identifier.
        timeframe: Available bar interval.
        years: Calendar years with existing factor selection parquet partitions.
    """

    manager: str
    timeframe: Timeframe
    years: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FactorSelectionTaskResult:
    """Immutable result for one manager/timeframe/year verification task.

    Attributes:
        manager: Order manager identifier.
        timeframe: Bar interval.
        year: Calendar year of the partition.
        status: ``succeeded`` or ``failed``.
        report: Verifier report when succeeded.
        error_type: Exception type name when failed.
        error_message: Exception message when failed.
        error_code: Optional CQROS error code when failed.
    """

    manager: str
    timeframe: Timeframe
    year: int
    status: str
    report: VerificationReport | None = None
    error_type: str | None = None
    error_message: str | None = None
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class VerifyFactorSelectionSummary:
    """Immutable aggregate summary for a factor selection verification run.

    Attributes:
        panels_verified: Unique manager/timeframe/year panels attempted.
        datasets_verified: Unique datasets attempted (always factor selection).
        timeframes_verified: Unique timeframes attempted.
        successful_tasks: Count of succeeded partition tasks.
        failed_tasks: Count of failed partition tasks.
        rows_checked: Sum of verifier ``rows_checked`` across successes.
        duplicate_timestamps: Sum of duplicate-primary-key counters.
        null_rows: Sum of null-row counters.
        nan_rows: Sum of NaN-row counters.
        invalid_timestamps: Sum of invalid-timestamp counters.
        invalid_status_rows: Sum of attributed invalid-status counters.
        warnings: Sum of warning counts across successes.
        duration_seconds: Wall-clock verification duration.
        repository_passed: Whether the repository status is PASS.
    """

    panels_verified: int
    datasets_verified: int
    timeframes_verified: int
    successful_tasks: int
    failed_tasks: int
    rows_checked: int
    duplicate_timestamps: int
    null_rows: int
    nan_rows: int
    invalid_timestamps: int
    invalid_status_rows: int
    warnings: int
    duration_seconds: float
    repository_passed: bool


def build_parser() -> argparse.ArgumentParser:
    """Create the factor selection dataset verification argument parser.

    Returns:
        Configured ``ArgumentParser`` for verification flags.
    """
    parser = argparse.ArgumentParser(
        prog="cqros-verify-factor-selection",
        description=(
            "Verify CQROS factor selection datasets across the discovered "
            "factor selection universe."
        ),
    )
    parser.add_argument(
        "--manager",
        dest="manager",
        default=None,
        metavar="NAME",
        help=(
            "Optional order-manager filter applied to partition discovery. "
            "Omit to verify all managers present under the factor selection tier."
        ),
    )
    parser.add_argument(
        "--top-n",
        dest="top_n",
        type=int,
        default=DEFAULT_TOP_N,
        metavar="INT",
        help=(
            "Top-N configuration expected by the ranking verifier " f"(default: {DEFAULT_TOP_N})."
        ),
    )
    parser.add_argument(
        "--candidate-n",
        dest="candidate_n",
        type=int,
        default=DEFAULT_CANDIDATE_N,
        metavar="INT",
        help=(
            "Candidate pool size expected by the ranking verifier "
            f"(default: {DEFAULT_CANDIDATE_N})."
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


def build_options(args: argparse.Namespace) -> VerifyFactorSelectionOptions:
    """Map parsed CLI arguments onto ``VerifyFactorSelectionOptions``.

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
            error_code="FSEL_CANDIDATE_N_INVALID",
            details={"parameter": "candidate_n", "value": candidate_n},
        )
    if candidate_n < top_n:
        raise ValidationError(
            "candidate_n must be greater than or equal to top_n",
            error_code="FSEL_CANDIDATE_N_LT_TOP_N",
            details={"parameter": "candidate_n", "value": candidate_n, "top_n": top_n},
        )

    storage_root = (
        Path(args.storage_root) if args.storage_root is not None else Path(DEFAULT_STORAGE_ROOT)
    )

    return VerifyFactorSelectionOptions(
        storage_root=storage_root,
        manager=manager,
        top_n=top_n,
        candidate_n=candidate_n,
        timeframes=_normalize_timeframes(args.timeframes),
        years=_normalize_years(args.years),
        workers=workers,
        verbose=bool(args.verbose),
        debug=bool(args.debug),
    )


def discover_work(
    repository: FactorSelectionRepository,
    options: VerifyFactorSelectionOptions,
) -> tuple[DiscoveredWorkItem, ...]:
    """Discover factor selection partitions matching the CLI filters.

    Args:
        repository: Factor selection repository providing discovery APIs.
        options: CLI filters for manager, timeframe, and year.

    Returns:
        Deterministically ordered discovered work items.
    """
    managers = (options.manager,) if options.manager is not None else None
    partitions = repository.discover(
        managers=managers,
        timeframes=options.timeframes,
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    return _group_partitions(partitions, year_filter=options.years)


def format_partition_failure(
    *,
    dataset: str,
    timeframe: Timeframe,
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
        timeframe: Bar interval.
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
        f"Timeframe: {timeframe}",
        f"Year: {year}",
        f"Partition: {partition}",
        f"Verifier: {verifier}",
        f"Exception: {exception_type}",
    ]
    if code is not None:
        lines.append(f"Code: {code}")
    lines.append(f"Message: {message}")
    return "\n".join(lines) + "\n"


def format_summary(summary: VerifyFactorSelectionSummary) -> str:
    """Render a deterministic verification summary report.

    Args:
        summary: Aggregate verification summary.

    Returns:
        Multi-line summary text suitable for stdout.
    """
    status = "PASS" if summary.repository_passed else "FAIL"
    lines = [
        "=====================================",
        "CQROS Verification Summary",
        "=====================================",
        "",
        f"Panels verified: {summary.panels_verified}",
        f"Datasets verified: {summary.datasets_verified}",
        f"Timeframes verified: {summary.timeframes_verified}",
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
        "",
        f"Invalid status rows: {summary.invalid_status_rows}",
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
    """Run the factor selection dataset verification CLI.

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
        repository = FactorSelectionRepository(layout, ParquetStore())
        work = discover_work(repository, options)
        summary = await run_verification(
            repository=repository,
            verifier=FactorSelectionVerifier(
                top_n=options.top_n,
                candidate_n=options.candidate_n,
            ),
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
    repository: FactorSelectionRepository,
    verifier: FactorSelectionVerifier,
    options: VerifyFactorSelectionOptions,
    work: Sequence[DiscoveredWorkItem],
) -> VerifyFactorSelectionSummary:
    """Execute discovered work through a bounded panel worker pool.

    Args:
        repository: Factor selection partition repository.
        verifier: Factor selection verifier instance.
        options: Immutable verification options.
        work: Discovered work items.

    Returns:
        Aggregate immutable summary.
    """
    started = time.perf_counter()
    if len(work) == 0:
        return VerifyFactorSelectionSummary(
            panels_verified=0,
            datasets_verified=0,
            timeframes_verified=0,
            successful_tasks=0,
            failed_tasks=0,
            rows_checked=0,
            duplicate_timestamps=0,
            null_rows=0,
            nan_rows=0,
            invalid_timestamps=0,
            invalid_status_rows=0,
            warnings=0,
            duration_seconds=time.perf_counter() - started,
            repository_passed=True,
        )

    work_by_timeframe = _group_work_by_timeframe(work)
    results = await _run_worker_pool(
        repository=repository,
        verifier=verifier,
        work_by_timeframe=work_by_timeframe,
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
    partitions: Sequence[FactorSelectionPartitionRef],
    *,
    year_filter: tuple[int, ...] | None,
) -> tuple[DiscoveredWorkItem, ...]:
    """Group factor selection year partitions into work items."""
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
    repository: FactorSelectionRepository,
    verifier: FactorSelectionVerifier,
    work_by_timeframe: Mapping[Timeframe, Sequence[DiscoveredWorkItem]],
    worker_count: int,
    debug: bool,
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
                results = await _verify_timeframe_work(
                    repository=repository,
                    verifier=verifier,
                    timeframe=item,
                    work_items=work_by_timeframe[item],
                    debug=debug,
                )
                async with lock:
                    collected.extend(results)
            finally:
                queue.task_done()

    worker_tasks = [
        asyncio.create_task(worker(), name=f"verify-factor-selection-worker-{index}")
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
            key=lambda result: (
                result.manager,
                result.timeframe,
                result.year,
            ),
        )
    )


async def _verify_timeframe_work(
    *,
    repository: FactorSelectionRepository,
    verifier: FactorSelectionVerifier,
    timeframe: Timeframe,
    work_items: Sequence[DiscoveredWorkItem],
    debug: bool,
) -> tuple[FactorSelectionTaskResult, ...]:
    """Verify every discovered year for one timeframe sequentially."""
    results: list[FactorSelectionTaskResult] = []
    for item in work_items:
        for year in item.years:
            result = await asyncio.to_thread(
                _verify_partition,
                repository,
                verifier,
                manager=item.manager,
                timeframe=timeframe,
                year=year,
                debug=debug,
            )
            if result.status == "failed":
                _report_task_failure(result)
            results.append(result)
    return tuple(results)


def _verify_partition(
    repository: FactorSelectionRepository,
    verifier: FactorSelectionVerifier,
    *,
    manager: str,
    timeframe: Timeframe,
    year: int,
    debug: bool,
) -> FactorSelectionTaskResult:
    """Verify one factor selection year partition synchronously."""
    try:
        frame = repository.load(
            manager=manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            timeframe=timeframe,
            year=year,
        )
        report = verifier.verify(frame)
    except Exception as exc:
        _log_partition_failure(
            manager=manager,
            timeframe=timeframe,
            year=year,
            exc=exc,
            debug=debug,
        )
        return FactorSelectionTaskResult(
            manager=manager,
            timeframe=timeframe,
            year=year,
            status="failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
            error_code=exc.error_code if isinstance(exc, CQROSError) else None,
        )

    return FactorSelectionTaskResult(
        manager=manager,
        timeframe=timeframe,
        year=year,
        status="succeeded",
        report=report,
    )


def _report_task_failure(result: FactorSelectionTaskResult) -> None:
    """Print structured diagnostics for a failed verification task."""
    message = result.error_message if result.error_message is not None else ""
    exception_type = result.error_type if result.error_type is not None else "Exception"
    print(
        format_partition_failure(
            dataset=_DATASET_DISPLAY_NAME,
            timeframe=result.timeframe,
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
    timeframe: Timeframe,
    year: int,
    exc: BaseException,
    debug: bool,
) -> None:
    """Log a partition verification failure without aborting the run."""
    log_extra = {
        "manager": manager,
        "timeframe": timeframe,
        "year": year,
        "verifier": _VERIFIER_NAME,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "error_code": exc.error_code if isinstance(exc, CQROSError) else None,
    }
    if debug:
        _logger.exception(
            "Failed factor selection partition verification; continuing",
            extra=log_extra,
            exc_info=True,
        )
    else:
        _logger.warning(
            "Failed factor selection partition verification; continuing",
            extra=log_extra,
        )


def _attribute_invalid_status_rows(report: VerificationReport) -> int:
    """Attribute combined invalid-numeric rows onto the status summary category.

    Maps ``status`` verifier warnings onto the shared invalid-status summary
    field, consistent with how other dataset verifiers attribute enumeration
    warnings.
    """
    numeric = report.invalid_numeric_rows
    if numeric == 0:
        return 0
    status_warning = (
        _WARN_EMPTY_STATUS in report.warnings or _WARN_INVALID_STATUS in report.warnings
    )
    if status_warning:
        return numeric
    return 0


def _build_summary(
    *,
    results: Sequence[FactorSelectionTaskResult],
    duration_seconds: float,
) -> VerifyFactorSelectionSummary:
    """Aggregate task results into a verification report."""
    panels_verified: set[tuple[str, Timeframe, int]] = set()
    timeframes_verified: set[Timeframe] = set()
    successful_tasks = 0
    failed_tasks = 0
    rows_checked = 0
    duplicate_timestamps = 0
    null_rows = 0
    nan_rows = 0
    invalid_timestamps = 0
    invalid_status_rows = 0
    warnings = 0

    for result in results:
        panels_verified.add((result.manager, result.timeframe, result.year))
        timeframes_verified.add(result.timeframe)
        if result.status == "succeeded":
            successful_tasks += 1
            report = result.report
            if report is not None:
                rows_checked += report.rows_checked
                duplicate_timestamps += report.duplicate_timestamp_rows
                null_rows += report.null_rows
                nan_rows += report.nan_rows
                invalid_timestamps += report.invalid_timestamp_rows
                invalid_status_rows += _attribute_invalid_status_rows(report)
                warnings += len(report.warnings)
        else:
            failed_tasks += 1

    repository_passed = (
        failed_tasks == 0
        and duplicate_timestamps == 0
        and null_rows == 0
        and nan_rows == 0
        and invalid_timestamps == 0
        and invalid_status_rows == 0
        and warnings == 0
    )

    datasets_verified = 1 if results else 0
    return VerifyFactorSelectionSummary(
        panels_verified=len(panels_verified),
        datasets_verified=datasets_verified,
        timeframes_verified=len(timeframes_verified),
        successful_tasks=successful_tasks,
        failed_tasks=failed_tasks,
        rows_checked=rows_checked,
        duplicate_timestamps=duplicate_timestamps,
        null_rows=null_rows,
        nan_rows=nan_rows,
        invalid_timestamps=invalid_timestamps,
        invalid_status_rows=invalid_status_rows,
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
