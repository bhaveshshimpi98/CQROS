"""CQROS Research Model Ledger generation CLI.

Purpose:
    Provide an argparse-based production entry point that discovers canonical
    Regime partitions and executes Research Model Ledger generation across
    manager/timeframe/year/symbol work items with bounded concurrency.

Responsibilities:
    - Parse CLI arguments for Models dataset generation
    - Discover Regime partitions through ``RegimeRepository``
    - Generate ledger rows via ``SimpleModelEngine`` / ``ModelRegistry``
    - Persist non-empty partitions through ``ModelRepository`` (same save
      contract used by ``ModelPipeline``)
    - Honor ``--overwrite``, ``--symbols``, ``--years``, worker concurrency,
      and debug logging
    - Report generated, skipped, empty, and failed scopes
    - Print deterministic progress and a final summary

Dependencies:
    ``argparse``, ``asyncio``, ``cqros.config``, ``cqros.core``,
    ``cqros.models``, ``cqros.regime``, and ``cqros.storage``.

Public API:
    ``DiscoveredWorkItem``, ``ModelGenerationOptions``,
    ``ModelGenerationSummary``, ``ModelTaskResult``, ``build_options``,
    ``build_parser``, ``discover_work``, ``format_summary``, ``main``,
    ``run_generation``.

Notes:
    This module is a thin composition root. Model semantics remain exclusively
    in ``SimpleModelEngine``. Persistence layout remains
    ``models/{manager}/{exchange}/{market}/{symbol}/{timeframe}/{year}.parquet``.
    This CLI does not train supervised ML models and does not import
    ``cqros.ml``.
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
    STORAGE_DIR_MODELS,
)
from cqros.core.exceptions import CQROSError, ValidationError
from cqros.core.types import Symbol, Timeframe
from cqros.models import ModelRegistry, ModelRepository, SimpleModelEngine
from cqros.models.exceptions import ModelError
from cqros.regime import RegimePartitionRef, RegimeRepository
from cqros.storage import ParquetStore, StorageLayout

__all__ = [
    "DiscoveredWorkItem",
    "ModelGenerationOptions",
    "ModelGenerationSummary",
    "ModelTaskResult",
    "build_options",
    "build_parser",
    "discover_work",
    "format_summary",
    "main",
    "run_generation",
]

_logger = logging.getLogger(__name__)

_EXIT_SUCCESS: Final[int] = 0
_EXIT_FAILURE: Final[int] = 1

_DEFAULT_WORKER_COUNT: Final[int] = ResearchConfig().worker_count

_EXCHANGE: Final[str] = EXCHANGE_BINANCE
_MARKET: Final[str] = MARKET_USDT_PERPETUAL

_ERROR_WORKERS: Final[str] = "CLI-GENERATE-MODELS-001"
_ERROR_YEAR: Final[str] = "CLI-GENERATE-MODELS-002"
_ERROR_MANAGER: Final[str] = "CLI-GENERATE-MODELS-003"
_ERROR_SYMBOL: Final[str] = "CLI-GENERATE-MODELS-004"

_ERROR_NO_REGIMES: Final[str] = "MODEL_NO_REGIMES"
_ERROR_FRAME_EMPTY: Final[str] = "MODEL_FRAME_EMPTY"


@dataclass(frozen=True, slots=True)
class ModelGenerationOptions:
    """Immutable CLI options for Research Model Ledger generation."""

    storage_root: Path
    manager: str
    years: tuple[int, ...] | None
    symbols: tuple[Symbol, ...] | None
    overwrite: bool
    workers: int
    verbose: bool
    debug: bool


@dataclass(frozen=True, slots=True)
class DiscoveredWorkItem:
    """One discovered manager/timeframe/year/symbol models generation scope."""

    manager: str
    timeframe: Timeframe
    year: int
    symbol: Symbol


@dataclass(frozen=True, slots=True)
class ModelTaskResult:
    """Immutable result for one Research Model Ledger partition task."""

    year: int
    timeframe: Timeframe
    symbol: Symbol
    status: str
    rows_generated: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class ModelGenerationSummary:
    """Immutable aggregate summary for a models-generation run."""

    manager: str
    panels: int
    rows: int
    successful_tasks: int
    failed_tasks: int
    skipped_tasks: int
    empty_tasks: int
    duration_seconds: float
    output_directory: Path
    failed_task_labels: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    """Create the Research Model Ledger generation argument parser."""
    parser = argparse.ArgumentParser(
        prog="cqros-generate-models",
        description=(
            "Generate CQROS Research Model Ledger datasets from discovered " "Regime partitions."
        ),
    )
    parser.add_argument(
        "--manager",
        dest="manager",
        required=True,
        metavar="NAME",
        help="Order manager identity used for discovery and model lineage.",
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
        "--symbols",
        dest="symbols",
        nargs="*",
        default=None,
        metavar="SYMBOL",
        help="Optional symbol allowlist (0..N values). Omit to discover all.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate Models partitions that already exist.",
    )
    parser.add_argument(
        "--workers",
        dest="workers",
        type=int,
        default=_DEFAULT_WORKER_COUNT,
        metavar="INT",
        help=f"Maximum concurrent Models partitions (default: {_DEFAULT_WORKER_COUNT}).",
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


def build_options(args: argparse.Namespace) -> ModelGenerationOptions:
    """Map parsed CLI arguments onto generation options."""
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

    storage_root = (
        Path(args.storage_root) if args.storage_root is not None else Path(DEFAULT_STORAGE_ROOT)
    )

    return ModelGenerationOptions(
        storage_root=storage_root,
        manager=manager,
        years=_normalize_years(args.years),
        symbols=_normalize_symbols(args.symbols),
        overwrite=bool(args.overwrite),
        workers=workers,
        verbose=bool(args.verbose),
        debug=bool(args.debug),
    )


def discover_work(
    regime_repository: RegimeRepository,
    options: ModelGenerationOptions,
) -> tuple[DiscoveredWorkItem, ...]:
    """Discover Models-ready Regime partitions for generation."""
    partitions = regime_repository.discover_partitions(
        managers=(options.manager,),
        symbols=options.symbols,
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    return _expand_work_items(partitions, year_filter=options.years)


def format_summary(summary: ModelGenerationSummary) -> str:
    """Render a deterministic Models-generation summary report."""
    lines = [
        "=====================================",
        "CQROS Research Model Ledger Summary",
        "=====================================",
        "",
        f"Manager: {summary.manager}",
        "",
        f"Panels: {summary.panels}",
        f"Rows: {summary.rows}",
        "",
        f"Successful: {summary.successful_tasks}",
        f"Failed: {summary.failed_tasks}",
        f"Skipped: {summary.skipped_tasks}",
        f"Empty: {summary.empty_tasks}",
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
    """Run the Research Model Ledger generation CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        options = build_options(args)
        _configure_logging(verbose=options.verbose, debug=options.debug)
        layout = StorageLayout(options.storage_root)
        datastore = ParquetStore()
        regime_repository = RegimeRepository(layout, datastore)
        model_repository = ModelRepository(layout, datastore)
        work = discover_work(regime_repository, options)
        summary = await run_generation(
            regime_repository=regime_repository,
            model_repository=model_repository,
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
    regime_repository: RegimeRepository,
    model_repository: ModelRepository,
    options: ModelGenerationOptions,
    work: Sequence[DiscoveredWorkItem],
) -> ModelGenerationSummary:
    """Execute discovered Models generation work items."""
    started = time.perf_counter()
    output_directory = options.storage_root / STORAGE_DIR_MODELS / options.manager
    if len(work) == 0:
        return ModelGenerationSummary(
            manager=options.manager,
            panels=0,
            rows=0,
            successful_tasks=0,
            failed_tasks=0,
            skipped_tasks=0,
            empty_tasks=0,
            duration_seconds=time.perf_counter() - started,
            output_directory=output_directory,
            failed_task_labels=(),
        )

    results = await _run_worker_pool(
        regime_repository=regime_repository,
        model_repository=model_repository,
        work=work,
        options=options,
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


def _normalize_symbols(values: Sequence[str] | None) -> tuple[Symbol, ...] | None:
    """Validate and freeze optional symbol filters."""
    if values is None:
        return None
    normalized: list[Symbol] = []
    for symbol in values:
        stripped = str(symbol).strip()
        if stripped == "":
            continue
        if stripped not in normalized:
            normalized.append(stripped)
    if len(values) > 0 and len(normalized) == 0:
        raise ValidationError(
            "symbols must contain at least one non-empty symbol",
            error_code=_ERROR_SYMBOL,
            details={"parameter": "symbols", "value": values},
        )
    return tuple(normalized) if normalized else None


def _expand_work_items(
    partitions: Sequence[RegimePartitionRef],
    *,
    year_filter: tuple[int, ...] | None,
) -> tuple[DiscoveredWorkItem, ...]:
    """Convert Regime partition refs into discovered work items."""
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
                symbol=partition.symbol,
            )
        )
    return tuple(
        sorted(
            items,
            key=lambda item: (item.manager, item.timeframe, item.year, item.symbol),
        )
    )


async def _run_worker_pool(
    *,
    regime_repository: RegimeRepository,
    model_repository: ModelRepository,
    work: Sequence[DiscoveredWorkItem],
    options: ModelGenerationOptions,
) -> tuple[ModelTaskResult, ...]:
    """Drain work items through a bounded asyncio worker pool."""
    queue: asyncio.Queue[DiscoveredWorkItem | None] = asyncio.Queue()
    for item in work:
        queue.put_nowait(item)
    for _ in range(options.workers):
        queue.put_nowait(None)

    collected: list[ModelTaskResult] = []
    lock = asyncio.Lock()

    async def worker() -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                result = await asyncio.to_thread(
                    _generate_partition,
                    regime_repository,
                    model_repository,
                    manager=item.manager,
                    timeframe=item.timeframe,
                    year=item.year,
                    symbol=item.symbol,
                    options=options,
                )
                _print_progress(result)
                async with lock:
                    collected.append(result)
            finally:
                queue.task_done()

    worker_tasks = [
        asyncio.create_task(worker(), name=f"generate-models-worker-{index}")
        for index in range(options.workers)
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
            key=lambda result: (result.year, result.timeframe, result.symbol),
        )
    )


def _generate_partition(
    regime_repository: RegimeRepository,
    model_repository: ModelRepository,
    *,
    manager: str,
    timeframe: Timeframe,
    year: int,
    symbol: Symbol,
    options: ModelGenerationOptions,
) -> ModelTaskResult:
    """Generate one Research Model Ledger partition synchronously."""
    if not options.overwrite and model_repository.exists(
        manager=manager,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=symbol,
        timeframe=timeframe,
        year=year,
    ):
        return ModelTaskResult(
            year=year,
            timeframe=timeframe,
            symbol=symbol,
            status="skipped",
        )

    if not regime_repository.exists(
        manager=manager,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=symbol,
        timeframe=timeframe,
        year=year,
    ):
        return ModelTaskResult(
            year=year,
            timeframe=timeframe,
            symbol=symbol,
            status="empty",
            rows_generated=0,
        )

    try:
        regime_frame = regime_repository.load(
            manager=manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        )
        if regime_frame.height == 0:
            return ModelTaskResult(
                year=year,
                timeframe=timeframe,
                symbol=symbol,
                status="empty",
                rows_generated=0,
            )

        registry = ModelRegistry(engine=SimpleModelEngine())
        try:
            created = registry.build(regime_frame)
        except ModelError as exc:
            if exc.error_code in {_ERROR_NO_REGIMES, _ERROR_FRAME_EMPTY}:
                return ModelTaskResult(
                    year=year,
                    timeframe=timeframe,
                    symbol=symbol,
                    status="empty",
                    rows_generated=0,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    error_code=exc.error_code,
                )
            raise

        if created.height == 0:
            return ModelTaskResult(
                year=year,
                timeframe=timeframe,
                symbol=symbol,
                status="empty",
                rows_generated=0,
            )

        model_repository.save(
            created,
            manager=manager,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        )
        persisted = created
    except Exception as exc:
        _log_partition_failure(
            manager=manager,
            year=year,
            timeframe=timeframe,
            symbol=symbol,
            exc=exc,
            debug=options.debug,
        )
        return ModelTaskResult(
            year=year,
            timeframe=timeframe,
            symbol=symbol,
            status="failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
            error_code=exc.error_code if isinstance(exc, CQROSError) else None,
        )

    return ModelTaskResult(
        year=year,
        timeframe=timeframe,
        symbol=symbol,
        status="succeeded",
        rows_generated=persisted.height,
    )


def _print_progress(result: ModelTaskResult) -> None:
    """Print a deterministic one-line progress record for a task result."""
    label = f"{result.year}/{result.timeframe}/{result.symbol}"
    if result.status == "succeeded":
        rows = result.rows_generated if result.rows_generated is not None else 0
        message = f"OK {label} rows={rows}"
    elif result.status == "skipped":
        message = f"SKIP {label}"
    elif result.status == "empty":
        message = f"EMPTY {label}"
    else:
        error_type = result.error_type if result.error_type is not None else "Exception"
        message = f"FAIL {label} {error_type}"
    print(message, flush=True)


def _log_partition_failure(
    *,
    manager: str,
    year: int,
    timeframe: Timeframe,
    symbol: Symbol,
    exc: BaseException,
    debug: bool,
) -> None:
    """Log a partition-level Models failure without aborting the run."""
    log_extra = {
        "manager": manager,
        "year": year,
        "timeframe": timeframe,
        "symbol": symbol,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "error_code": exc.error_code if isinstance(exc, CQROSError) else None,
    }
    if debug:
        _logger.exception(
            "Failed Models generation partition; continuing",
            extra=log_extra,
            exc_info=True,
        )
    else:
        _logger.warning(
            "Failed Models generation partition; continuing",
            extra=log_extra,
        )


def _build_summary(
    *,
    options: ModelGenerationOptions,
    work: Sequence[DiscoveredWorkItem],
    results: Sequence[ModelTaskResult],
    duration_seconds: float,
    output_directory: Path,
) -> ModelGenerationSummary:
    """Aggregate task results into a generation report."""
    successful_tasks = 0
    failed_tasks = 0
    skipped_tasks = 0
    empty_tasks = 0
    rows = 0
    failed_labels: list[str] = []
    for result in results:
        if result.status == "succeeded":
            successful_tasks += 1
            rows += result.rows_generated if result.rows_generated is not None else 0
        elif result.status == "skipped":
            skipped_tasks += 1
        elif result.status == "empty":
            empty_tasks += 1
        else:
            failed_tasks += 1
            failed_labels.append(f"{result.year}/{result.timeframe}/{result.symbol}")
    return ModelGenerationSummary(
        manager=options.manager,
        panels=len(work),
        rows=rows,
        successful_tasks=successful_tasks,
        failed_tasks=failed_tasks,
        skipped_tasks=skipped_tasks,
        empty_tasks=empty_tasks,
        duration_seconds=duration_seconds,
        output_directory=output_directory,
        failed_task_labels=tuple(failed_labels),
    )


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds for summary output."""
    return f"{seconds:.2f}s"


def _format_output_directory(path: Path) -> str:
    """Format the output directory path for summary output."""
    return str(path)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
