"""CQROS factor orthogonalization dataset verification CLI.

Purpose:
    Provide an argparse-based production entry point that discovers
    factor orthogonalization panels and executes
    ``FactorOrthogonalizationVerifier.verify`` followed by
    ``FactorOrthogonalizationVerifier.verify_against_combination`` across
    the universe with bounded panel concurrency.

Responsibilities:
    - Parse CLI arguments for factor orthogonalization dataset verification
    - Discover available partitions through ``FactorOrthogonalizationRepository``
    - Load orthogonalization frames and corresponding Combination frames
    - Verify each frame structurally and against Combination lineage
    - Aggregate results into a final PASS/FAIL report
    - Print structured per-partition failure diagnostics
    - Print the report and return an exit code

Dependencies:
    ``argparse``, ``asyncio``, ``cqros.config``, ``cqros.core``,
    ``cqros.factor_combination``, ``cqros.factor_orthogonalization``,
    and ``cqros.storage``.

Public API:
    ``DiscoveredWorkItem``, ``FactorOrthogonalizationTaskResult``,
    ``VerifyFactorOrthogonalizationOptions``,
    ``VerifyFactorOrthogonalizationSummary``, ``build_options``,
    ``build_parser``, ``discover_work``, ``format_partition_failure``,
    ``format_summary``, ``main``, ``run_verification``.
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
from cqros.core.types import Timeframe
from cqros.factor_combination import FactorCombinationRepository
from cqros.factor_orthogonalization import (
    FactorOrthogonalizationRepository,
    FactorOrthogonalizationVerifier,
)
from cqros.factor_orthogonalization.repository import FactorOrthogonalizationPartitionRef
from cqros.storage import ParquetStore, StorageLayout

__all__ = [
    "DiscoveredWorkItem",
    "FactorOrthogonalizationTaskResult",
    "VerifyFactorOrthogonalizationOptions",
    "VerifyFactorOrthogonalizationSummary",
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

_ERROR_WORKERS: Final[str] = "CLI-VERIFY-FORTH-001"
_ERROR_YEAR: Final[str] = "CLI-VERIFY-FORTH-002"
_ERROR_MANAGER: Final[str] = "CLI-VERIFY-FORTH-003"

_VERIFIER_NAME: Final[str] = "FactorOrthogonalizationVerifier"
_DATASET_DISPLAY_NAME: Final[str] = "Factor Orthogonalization"


@dataclass(frozen=True, slots=True)
class VerifyFactorOrthogonalizationOptions:
    """Immutable CLI options for factor orthogonalization dataset verification."""

    storage_root: Path
    manager: str | None
    years: tuple[int, ...] | None
    workers: int
    verbose: bool
    debug: bool


@dataclass(frozen=True, slots=True)
class DiscoveredWorkItem:
    """One discovered orthogonalization partition ready for verification."""

    manager: str
    timeframe: Timeframe
    year: int


@dataclass(frozen=True, slots=True)
class FactorOrthogonalizationTaskResult:
    """Immutable result for one manager/timeframe/year verification task."""

    manager: str
    timeframe: Timeframe
    year: int
    status: str
    rows_checked: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class VerifyFactorOrthogonalizationSummary:
    """Immutable aggregate summary for an orthogonalization verification run."""

    panels_verified: int
    datasets_verified: int
    timeframes_verified: int
    successful_tasks: int
    failed_tasks: int
    rows_checked: int
    duration_seconds: float
    repository_passed: bool


def build_parser() -> argparse.ArgumentParser:
    """Create the factor orthogonalization verification argument parser."""
    parser = argparse.ArgumentParser(
        prog="cqros-verify-factor-orthogonalization",
        description=(
            "Verify CQROS factor orthogonalization datasets across the discovered "
            "orthogonalization universe."
        ),
    )
    parser.add_argument(
        "--manager",
        dest="manager",
        default=None,
        metavar="NAME",
        help=(
            "Optional order-manager filter applied to partition discovery. "
            "Omit to verify all managers present under the factor orthogonalization tier."
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


def build_options(args: argparse.Namespace) -> VerifyFactorOrthogonalizationOptions:
    """Map parsed CLI arguments onto verification options."""
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

    return VerifyFactorOrthogonalizationOptions(
        storage_root=storage_root,
        manager=manager,
        years=_normalize_years(args.years),
        workers=workers,
        verbose=bool(args.verbose),
        debug=bool(args.debug),
    )


def discover_work(
    repository: FactorOrthogonalizationRepository,
    options: VerifyFactorOrthogonalizationOptions,
) -> tuple[DiscoveredWorkItem, ...]:
    """Discover orthogonalization partitions matching the CLI filters."""
    managers = (options.manager,) if options.manager is not None else None
    partitions = repository.discover_partitions(
        managers=managers,
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    return _group_partitions(partitions, year_filter=options.years)


def format_partition_failure(
    *,
    dataset: str,
    manager: str,
    timeframe: Timeframe,
    year: int,
    partition: str,
    verifier: str,
    exception_type: str,
    message: str,
    code: str | None = None,
) -> str:
    """Render a structured per-partition verification failure report."""
    lines = [
        "FAILED",
        "",
        f"Dataset: {dataset}",
        f"Manager: {manager}",
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


def format_summary(summary: VerifyFactorOrthogonalizationSummary) -> str:
    """Render a deterministic orthogonalization verification summary report."""
    status = "PASS" if summary.repository_passed else "FAIL"
    lines = [
        "=====================================",
        "CQROS Factor Orthogonalization Verification Summary",
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
    """Run the factor orthogonalization dataset verification CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        options = build_options(args)
        _configure_logging(verbose=options.verbose, debug=options.debug)
        layout = StorageLayout(options.storage_root)
        datastore = ParquetStore()
        orthogonalization_repository = FactorOrthogonalizationRepository(layout, datastore)
        combination_repository = FactorCombinationRepository(layout, datastore)
        work = discover_work(orthogonalization_repository, options)
        summary = await run_verification(
            orthogonalization_repository=orthogonalization_repository,
            combination_repository=combination_repository,
            verifier=FactorOrthogonalizationVerifier(),
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
    orthogonalization_repository: FactorOrthogonalizationRepository,
    combination_repository: FactorCombinationRepository,
    verifier: FactorOrthogonalizationVerifier,
    options: VerifyFactorOrthogonalizationOptions,
    work: Sequence[DiscoveredWorkItem],
) -> VerifyFactorOrthogonalizationSummary:
    """Execute discovered work through a bounded panel worker pool."""
    started = time.perf_counter()
    if len(work) == 0:
        return VerifyFactorOrthogonalizationSummary(
            panels_verified=0,
            datasets_verified=0,
            timeframes_verified=0,
            successful_tasks=0,
            failed_tasks=0,
            rows_checked=0,
            duration_seconds=time.perf_counter() - started,
            repository_passed=True,
        )

    results = await _run_worker_pool(
        orthogonalization_repository=orthogonalization_repository,
        combination_repository=combination_repository,
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
    partitions: Sequence[FactorOrthogonalizationPartitionRef],
    *,
    year_filter: tuple[int, ...] | None,
) -> tuple[DiscoveredWorkItem, ...]:
    """Group orthogonalization partitions into discovered work items."""
    year_allowlist = set(year_filter) if year_filter is not None else None
    items: list[DiscoveredWorkItem] = []
    for partition in partitions:
        if year_allowlist is not None and partition.year not in year_allowlist:
            continue
        items.append(
            DiscoveredWorkItem(
                manager=partition.manager,
                timeframe=partition.timeframe,
                year=partition.year,
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
    orthogonalization_repository: FactorOrthogonalizationRepository,
    combination_repository: FactorCombinationRepository,
    verifier: FactorOrthogonalizationVerifier,
    work: Sequence[DiscoveredWorkItem],
    worker_count: int,
    debug: bool,
) -> tuple[FactorOrthogonalizationTaskResult, ...]:
    """Drain work items through a bounded asyncio worker pool."""
    if len(work) == 0:
        return ()

    queue: asyncio.Queue[DiscoveredWorkItem | None] = asyncio.Queue()
    for item in work:
        queue.put_nowait(item)
    for _ in range(worker_count):
        queue.put_nowait(None)

    collected: list[FactorOrthogonalizationTaskResult] = []
    lock = asyncio.Lock()

    async def worker() -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                result = await asyncio.to_thread(
                    _verify_partition,
                    orthogonalization_repository,
                    combination_repository,
                    verifier,
                    manager=item.manager,
                    timeframe=item.timeframe,
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
        asyncio.create_task(worker(), name=f"verify-orthogonalization-worker-{index}")
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
            key=lambda result: (result.manager, result.timeframe, result.year),
        )
    )


def _verify_partition(
    orthogonalization_repository: FactorOrthogonalizationRepository,
    combination_repository: FactorCombinationRepository,
    verifier: FactorOrthogonalizationVerifier,
    *,
    manager: str,
    timeframe: Timeframe,
    year: int,
    debug: bool,
) -> FactorOrthogonalizationTaskResult:
    """Verify one orthogonalization year partition synchronously."""
    try:
        orthogonalization_frame = orthogonalization_repository.load(
            manager=manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            timeframe=timeframe,
            year=year,
        )
        combination_frame = combination_repository.load(
            manager=manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            timeframe=timeframe,
            year=year,
        )
        verifier.verify(orthogonalization_frame)
        verifier.verify_against_combination(orthogonalization_frame, combination_frame)
        rows_checked = orthogonalization_frame.height
    except Exception as exc:
        _log_partition_failure(
            manager=manager,
            timeframe=timeframe,
            year=year,
            exc=exc,
            debug=debug,
        )
        return FactorOrthogonalizationTaskResult(
            manager=manager,
            timeframe=timeframe,
            year=year,
            status="failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
            error_code=exc.error_code if isinstance(exc, CQROSError) else None,
        )

    return FactorOrthogonalizationTaskResult(
        manager=manager,
        timeframe=timeframe,
        year=year,
        status="succeeded",
        rows_checked=rows_checked,
    )


def _report_task_failure(result: FactorOrthogonalizationTaskResult) -> None:
    """Print structured diagnostics for a failed verification task."""
    message = result.error_message if result.error_message is not None else ""
    exception_type = result.error_type if result.error_type is not None else "Exception"
    print(
        format_partition_failure(
            dataset=_DATASET_DISPLAY_NAME,
            manager=result.manager,
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
            "Failed orthogonalization partition verification; continuing",
            extra=log_extra,
            exc_info=True,
        )
    else:
        _logger.warning(
            "Failed orthogonalization partition verification; continuing",
            extra=log_extra,
        )


def _build_summary(
    *,
    results: Sequence[FactorOrthogonalizationTaskResult],
    duration_seconds: float,
) -> VerifyFactorOrthogonalizationSummary:
    """Aggregate task results into a verification report."""
    panels_verified: set[tuple[str, Timeframe, int]] = set()
    timeframes_verified: set[Timeframe] = set()
    successful_tasks = 0
    failed_tasks = 0
    rows_checked = 0

    for result in results:
        panels_verified.add((result.manager, result.timeframe, result.year))
        timeframes_verified.add(result.timeframe)
        if result.status == "succeeded":
            successful_tasks += 1
            if result.rows_checked is not None:
                rows_checked += result.rows_checked
        else:
            failed_tasks += 1

    repository_passed = failed_tasks == 0
    datasets_verified = 1 if results else 0

    return VerifyFactorOrthogonalizationSummary(
        panels_verified=len(panels_verified),
        datasets_verified=datasets_verified,
        timeframes_verified=len(timeframes_verified),
        successful_tasks=successful_tasks,
        failed_tasks=failed_tasks,
        rows_checked=rows_checked,
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
