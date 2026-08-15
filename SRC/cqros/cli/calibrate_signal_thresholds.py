"""CQROS regression signal threshold calibration CLI.

Purpose:
    Provide an argparse-based research entry point that discovers prediction
    partitions, analyzes prediction distributions, and prints recommended
    BUY/SELL thresholds for regression signal policies.

Responsibilities:
    - Parse CLI arguments for threshold calibration
    - Discover available prediction partitions through ``PredictionRepository``
    - Load partitions read-only and calibrate with ``SignalThresholdCalibrator``
    - Print per-symbol/timeframe and global recommendation summaries
    - Remain free of signal generation, policy mutation, and repository writes

Dependencies:
    ``argparse``, ``asyncio``, ``polars``, ``cqros.config``, ``cqros.core``,
    ``cqros.research.signal_threshold_calibrator``, and ``cqros.storage``.

Public API:
    ``build_parser``, ``build_options``, ``discover_work``,
    ``format_group_report``, ``format_summary``, ``run_calibration``, and
    ``main``.

Notes:
    This module is a thin composition root. It does not implement percentile
    math, mutate policies, persist signals, or update repositories.
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
    STORAGE_DIR_PREDICTIONS,
    SUPPORTED_TIMEFRAMES,
)
from cqros.core.exceptions import CQROSError, ResearchError, ValidationError
from cqros.core.types import Symbol, Timeframe
from cqros.research.signal_threshold_calibrator import (
    SignalThresholdCalibrator,
    SymbolTimeframeCalibration,
    ThresholdCalibrationResult,
    ThresholdRecommendation,
)
from cqros.storage import (
    ParquetStore,
    PredictionPartitionRef,
    PredictionRepository,
    StorageLayout,
)

__all__ = [
    "CalibrationOptions",
    "CalibrationSummary",
    "DiscoveredWorkItem",
    "build_options",
    "build_parser",
    "discover_work",
    "format_group_report",
    "format_summary",
    "main",
    "run_calibration",
]

_logger = logging.getLogger(__name__)

_EXIT_SUCCESS: Final[int] = 0
_EXIT_FAILURE: Final[int] = 1

_DEFAULT_WORKER_COUNT: Final[int] = ResearchConfig().worker_count

_EXCHANGE: Final[str] = EXCHANGE_BINANCE
_MARKET: Final[str] = MARKET_USDT_PERPETUAL

_ERROR_WORKERS: Final[str] = "CLI-CALIBRATE-SIGNAL-THRESHOLDS-001"
_ERROR_TIMEFRAME: Final[str] = "CLI-CALIBRATE-SIGNAL-THRESHOLDS-002"
_ERROR_YEAR: Final[str] = "CLI-CALIBRATE-SIGNAL-THRESHOLDS-003"
_ERROR_MODEL: Final[str] = "CLI-CALIBRATE-SIGNAL-THRESHOLDS-004"
_ERROR_VERSION: Final[str] = "CLI-CALIBRATE-SIGNAL-THRESHOLDS-005"


@dataclass(frozen=True, slots=True)
class CalibrationOptions:
    """Immutable CLI options for regression threshold calibration.

    Attributes:
        storage_root: Storage root containing ``predictions``.
        model: Stable model identifier.
        version: Model version identifier.
        symbols: Optional symbol allowlist. ``None`` discovers all.
        timeframes: Optional timeframe allowlist. ``None`` discovers all.
        years: Optional year allowlist. ``None`` discovers all.
        workers: Maximum concurrent symbols.
        verbose: When ``True``, enable INFO logging.
        debug: When ``True``, enable DEBUG logging and log full failure
            tracebacks.
    """

    storage_root: Path
    model: str
    version: str
    symbols: tuple[Symbol, ...] | None
    timeframes: tuple[Timeframe, ...] | None
    years: tuple[int, ...] | None
    workers: int
    verbose: bool
    debug: bool


@dataclass(frozen=True, slots=True)
class DiscoveredWorkItem:
    """One discovered prediction partition group ready for calibration.

    Attributes:
        framework: Machine-learning framework identifier.
        model_name: Stable model identifier.
        model_version: Model version identifier.
        symbol: Prediction symbol.
        timeframe: Available bar interval.
        years: Calendar years with existing prediction parquet partitions.
    """

    framework: str
    model_name: str
    model_version: str
    symbol: Symbol
    timeframe: Timeframe
    years: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CalibrationSummary:
    """Immutable aggregate summary for a threshold calibration run.

    Attributes:
        model: Stable model identifier analyzed.
        version: Model version analyzed.
        result: Aggregate calibration result when analysis succeeded.
        duration_seconds: Wall-clock calibration duration.
        failed_groups: Count of symbol/timeframe groups that failed.
        failed_group_labels: Deterministic failed-group labels.
    """

    model: str
    version: str
    result: ThresholdCalibrationResult | None
    duration_seconds: float
    failed_groups: int
    failed_group_labels: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    """Create the regression threshold calibration argument parser.

    Returns:
        Configured ``ArgumentParser`` for calibration flags.
    """
    parser = argparse.ArgumentParser(
        prog="cqros-calibrate-signal-thresholds",
        description=(
            "Analyze CQROS prediction distributions and recommend "
            "regression BUY/SELL thresholds from empirical percentiles."
        ),
    )
    parser.add_argument(
        "--model",
        dest="model",
        required=True,
        metavar="NAME",
        help="Stable model identifier whose predictions are analyzed.",
    )
    parser.add_argument(
        "--version",
        dest="version",
        required=True,
        metavar="VERSION",
        help="Model version identifier whose predictions are analyzed.",
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


def build_options(args: argparse.Namespace) -> CalibrationOptions:
    """Map parsed CLI arguments onto ``CalibrationOptions``.

    Args:
        args: Namespace produced by ``build_parser().parse_args(...)``.

    Returns:
        Immutable calibration options.

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

    model = str(args.model).strip()
    if model == "":
        raise ValidationError(
            "model must be a non-empty string",
            error_code=_ERROR_MODEL,
            details={"parameter": "model", "value": args.model},
        )
    version = str(args.version).strip()
    if version == "":
        raise ValidationError(
            "version must be a non-empty string",
            error_code=_ERROR_VERSION,
            details={"parameter": "version", "value": args.version},
        )

    return CalibrationOptions(
        storage_root=Path(DEFAULT_STORAGE_ROOT),
        model=model,
        version=version,
        symbols=_normalize_symbols(args.symbols),
        timeframes=_normalize_timeframes(args.timeframes),
        years=_normalize_years(args.years),
        workers=workers,
        verbose=bool(args.verbose),
        debug=bool(args.debug),
    )


def discover_work(
    repository: PredictionRepository,
    options: CalibrationOptions,
) -> tuple[DiscoveredWorkItem, ...]:
    """Discover prediction partitions matching the CLI filters.

    Args:
        repository: Prediction repository providing discovery APIs.
        options: CLI filters for model, version, symbol, timeframe, and year.

    Returns:
        Deterministically ordered discovered work items.
    """
    frameworks = _discover_frameworks(options.storage_root)
    partitions: list[tuple[str, PredictionPartitionRef]] = []
    for framework in frameworks:
        discovered = repository.discover_partitions(
            framework=framework,
            model_names=(options.model,),
            versions=(options.version,),
            symbols=options.symbols,
            timeframes=options.timeframes,
            exchange=_EXCHANGE,
            market=_MARKET,
        )
        partitions.extend((framework, partition) for partition in discovered)
    return _group_partitions(partitions, year_filter=options.years)


def format_group_report(calibration: SymbolTimeframeCalibration) -> str:
    """Render a per-symbol/timeframe calibration report.

    Args:
        calibration: Immutable per-group calibration result.

    Returns:
        Multi-line text suitable for stdout.
    """
    stats = calibration.statistics
    lines = [
        "-------------------------------------",
        f"Symbol: {calibration.symbol}",
        f"Timeframe: {calibration.timeframe}",
        "-------------------------------------",
        "",
        "Prediction summary",
        "",
        f"Rows: {stats.count}",
        f"Min: {_format_float(stats.minimum)}",
        f"Max: {_format_float(stats.maximum)}",
        f"Mean: {_format_float(stats.mean)}",
        f"Std: {_format_float(stats.std)}",
        f"Median: {_format_float(stats.median)}",
        f"1%: {_format_float(stats.percentile_01)}",
        f"2.5%: {_format_float(stats.percentile_025)}",
        f"5%: {_format_float(stats.percentile_05)}",
        f"10%: {_format_float(stats.percentile_10)}",
        f"90%: {_format_float(stats.percentile_90)}",
        f"95%: {_format_float(stats.percentile_95)}",
        f"97.5%: {_format_float(stats.percentile_975)}",
        f"99%: {_format_float(stats.percentile_99)}",
        f"Positive ratio: {_format_ratio(stats.positive_ratio)}",
        f"Negative ratio: {_format_ratio(stats.negative_ratio)}",
        "",
        "Recommended thresholds",
        "",
    ]
    lines.extend(_format_recommendation_block(calibration.recommendations))
    return "\n".join(lines) + "\n"


def format_summary(summary: CalibrationSummary) -> str:
    """Render a deterministic global threshold-calibration summary.

    Args:
        summary: Aggregate calibration summary.

    Returns:
        Multi-line summary text suitable for stdout.
    """
    lines = [
        "=====================================",
        "CQROS Regression Threshold Analysis",
        "=====================================",
        "",
        f"Model: {summary.model}",
        f"Version: {summary.version}",
    ]

    result = summary.result
    if result is None:
        lines.extend(
            [
                "",
                "Symbols analyzed: 0",
                "Datasets: 0",
                "Rows analyzed: 0",
                "",
                f"Duration: {_format_duration(summary.duration_seconds)}",
            ]
        )
        if summary.failed_group_labels:
            lines.extend(["", "Failed Groups", ""])
            lines.extend(f"- {label}" for label in summary.failed_group_labels)
        lines.extend(["", "====================================="])
        return "\n".join(lines) + "\n"

    stats = result.global_statistics
    symbols_display = ", ".join(result.symbols_analyzed) if result.symbols_analyzed else "(none)"
    lines.extend(
        [
            "",
            f"Symbols analyzed: {symbols_display}",
            f"Datasets: {result.datasets_analyzed}",
            f"Rows analyzed: {result.rows_analyzed}",
            "",
            "Prediction range:",
            f"Min: {_format_float(stats.minimum)}",
            f"Max: {_format_float(stats.maximum)}",
            "",
            "Global percentiles:",
            f"1%: {_format_float(stats.percentile_01)}",
            f"2.5%: {_format_float(stats.percentile_025)}",
            f"5%: {_format_float(stats.percentile_05)}",
            f"10%: {_format_float(stats.percentile_10)}",
            f"50%: {_format_float(stats.median)}",
            f"90%: {_format_float(stats.percentile_90)}",
            f"95%: {_format_float(stats.percentile_95)}",
            f"97.5%: {_format_float(stats.percentile_975)}",
            f"99%: {_format_float(stats.percentile_99)}",
            "",
            "Recommended thresholds",
            "",
        ]
    )
    lines.extend(_format_recommendation_block(result.recommendations))
    lines.extend(
        [
            "",
            f"Duration: {_format_duration(summary.duration_seconds)}",
        ]
    )
    if summary.failed_group_labels:
        lines.extend(["", "Failed Groups", ""])
        lines.extend(f"- {label}" for label in summary.failed_group_labels)
    lines.extend(["", "====================================="])
    return "\n".join(lines) + "\n"


async def main(argv: Sequence[str] | None = None) -> int:
    """Run the regression threshold calibration CLI.

    Args:
        argv: Optional argument vector. When ``None``, ``sys.argv[1:]`` is used.

    Returns:
        ``0`` on successful calibration with no failed groups; ``1`` otherwise.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        options = build_options(args)
        _configure_logging(verbose=options.verbose, debug=options.debug)
        layout = StorageLayout(options.storage_root)
        repository = PredictionRepository(layout, ParquetStore())
        work = discover_work(repository, options)
        summary = await run_calibration(
            repository=repository,
            calibrator=SignalThresholdCalibrator(logger=_logger),
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
    if summary.result is None or summary.failed_groups > 0:
        return _EXIT_FAILURE
    return _EXIT_SUCCESS


async def run_calibration(
    *,
    repository: PredictionRepository,
    calibrator: SignalThresholdCalibrator,
    options: CalibrationOptions,
    work: Sequence[DiscoveredWorkItem],
) -> CalibrationSummary:
    """Execute discovered work through a bounded symbol worker pool.

    Args:
        repository: Injected prediction repository (read-only loads).
        calibrator: Injected threshold calibrator.
        options: Immutable calibration options.
        work: Discovered work items.

    Returns:
        Aggregate immutable summary.
    """
    started = time.perf_counter()
    if len(work) == 0:
        return CalibrationSummary(
            model=options.model,
            version=options.version,
            result=None,
            duration_seconds=time.perf_counter() - started,
            failed_groups=0,
            failed_group_labels=(),
        )

    work_by_symbol = _group_work_by_symbol(work)
    loaded = await _run_worker_pool(
        repository=repository,
        work_by_symbol=work_by_symbol,
        worker_count=options.workers,
        debug=options.debug,
    )

    group_frames: dict[tuple[Symbol, Timeframe], pl.DataFrame] = {}
    failed_labels: list[str] = []
    for item in loaded:
        if item.error_message is not None:
            failed_labels.append(f"{item.symbol}/{item.timeframe}")
            continue
        assert item.frame is not None
        try:
            group_calibration = calibrator.calibrate_group(
                item.frame,
                symbol=item.symbol,
                timeframe=item.timeframe,
            )
        except ResearchError as exc:
            _log_group_failure(
                symbol=item.symbol,
                timeframe=item.timeframe,
                exc=exc,
                debug=options.debug,
            )
            failed_labels.append(f"{item.symbol}/{item.timeframe}")
            continue
        group_frames[(item.symbol, item.timeframe)] = item.frame
        print(format_group_report(group_calibration), end="", flush=True)

    result: ThresholdCalibrationResult | None = None
    if group_frames:
        result = calibrator.calibrate(group_frames)

    return CalibrationSummary(
        model=options.model,
        version=options.version,
        result=result,
        duration_seconds=time.perf_counter() - started,
        failed_groups=len(failed_labels),
        failed_group_labels=tuple(sorted(set(failed_labels))),
    )


@dataclass(frozen=True, slots=True)
class _LoadedGroup:
    """Internal load outcome for one symbol/timeframe group."""

    symbol: Symbol
    timeframe: Timeframe
    frame: pl.DataFrame | None
    error_message: str | None


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


def _discover_frameworks(storage_root: Path) -> tuple[str, ...]:
    """Return sorted framework directories under the predictions tier."""
    base = storage_root / STORAGE_DIR_PREDICTIONS
    if not base.is_dir():
        return ()
    return tuple(sorted(path.name for path in base.iterdir() if path.is_dir()))


def _group_partitions(
    partitions: Sequence[tuple[str, PredictionPartitionRef]],
    *,
    year_filter: tuple[int, ...] | None,
) -> tuple[DiscoveredWorkItem, ...]:
    """Group year partitions into framework/model/symbol/timeframe work items."""
    year_allowlist = set(year_filter) if year_filter is not None else None
    grouped: dict[tuple[str, str, str, str, str], list[int]] = {}
    for framework, partition in partitions:
        if year_allowlist is not None and partition.year not in year_allowlist:
            continue
        key = (
            framework,
            partition.model_name,
            partition.model_version,
            partition.symbol,
            partition.timeframe,
        )
        grouped.setdefault(key, []).append(partition.year)

    items: list[DiscoveredWorkItem] = []
    for (framework, model_name, model_version, symbol, timeframe), years in grouped.items():
        items.append(
            DiscoveredWorkItem(
                framework=framework,
                model_name=model_name,
                model_version=model_version,
                symbol=symbol,
                timeframe=timeframe,
                years=tuple(sorted(years)),
            )
        )
    return tuple(
        sorted(
            items,
            key=lambda item: (
                item.framework,
                item.model_name,
                item.model_version,
                item.symbol,
                item.timeframe,
            ),
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
    repository: PredictionRepository,
    work_by_symbol: Mapping[Symbol, Sequence[DiscoveredWorkItem]],
    worker_count: int,
    debug: bool,
) -> tuple[_LoadedGroup, ...]:
    """Drain symbols through a bounded asyncio worker pool."""
    symbols = tuple(work_by_symbol.keys())
    if len(symbols) == 0:
        return ()

    queue: asyncio.Queue[Symbol | None] = asyncio.Queue()
    for symbol in symbols:
        queue.put_nowait(symbol)
    for _ in range(worker_count):
        queue.put_nowait(None)

    collected: list[_LoadedGroup] = []
    lock = asyncio.Lock()

    async def worker() -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                results = await _load_symbol_work(
                    repository=repository,
                    symbol=item,
                    work_items=work_by_symbol[item],
                    debug=debug,
                )
                async with lock:
                    collected.extend(results)
            finally:
                queue.task_done()

    worker_tasks = [
        asyncio.create_task(worker(), name=f"calibrate-signal-thresholds-worker-{index}")
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
            key=lambda result: (result.symbol, result.timeframe),
        )
    )


async def _load_symbol_work(
    *,
    repository: PredictionRepository,
    symbol: Symbol,
    work_items: Sequence[DiscoveredWorkItem],
    debug: bool,
) -> tuple[_LoadedGroup, ...]:
    """Load and concatenate year partitions for one symbol."""
    results: list[_LoadedGroup] = []
    for item in work_items:
        result = await asyncio.to_thread(
            _load_group,
            repository,
            framework=item.framework,
            model_name=item.model_name,
            model_version=item.model_version,
            symbol=symbol,
            timeframe=item.timeframe,
            years=item.years,
            debug=debug,
        )
        results.append(result)
    return tuple(results)


def _load_group(
    repository: PredictionRepository,
    *,
    framework: str,
    model_name: str,
    model_version: str,
    symbol: Symbol,
    timeframe: Timeframe,
    years: Sequence[int],
    debug: bool,
) -> _LoadedGroup:
    """Load and concatenate prediction year partitions for one group."""
    try:
        frames: list[pl.DataFrame] = []
        for year in years:
            frame = repository.load(
                framework=framework,
                model_name=model_name,
                model_version=model_version,
                exchange=_EXCHANGE,
                market=_MARKET,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
            )
            frames.append(frame)
        concatenated = pl.concat(frames, how="vertical") if len(frames) > 1 else frames[0]
    except Exception as exc:
        _log_group_failure(
            symbol=symbol,
            timeframe=timeframe,
            exc=exc,
            debug=debug,
        )
        return _LoadedGroup(
            symbol=symbol,
            timeframe=timeframe,
            frame=None,
            error_message=str(exc),
        )

    return _LoadedGroup(
        symbol=symbol,
        timeframe=timeframe,
        frame=concatenated,
        error_message=None,
    )


def _log_group_failure(
    *,
    symbol: Symbol,
    timeframe: Timeframe,
    exc: BaseException,
    debug: bool,
) -> None:
    """Log a symbol/timeframe load failure without aborting the run."""
    log_extra = {
        "symbol": symbol,
        "timeframe": timeframe,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "error_code": exc.error_code if isinstance(exc, CQROSError) else None,
    }
    if debug:
        _logger.exception(
            "Failed prediction group load for threshold calibration; continuing",
            extra=log_extra,
            exc_info=True,
        )
    else:
        _logger.warning(
            "Failed prediction group load for threshold calibration; continuing",
            extra=log_extra,
        )


def _format_recommendation_block(
    recommendations: Sequence[ThresholdRecommendation],
) -> list[str]:
    """Render recommended thresholds and expected signal rates."""
    lines: list[str] = []
    by_profile = {item.profile: item for item in recommendations}
    for profile in ("Conservative", "Balanced", "Active"):
        item = by_profile[profile]
        lines.extend(
            [
                f"{profile}:",
                f"BUY >= {_format_float(item.buy_threshold)}",
                f"SELL <= {_format_float(item.sell_threshold)}",
                "",
            ]
        )

    balanced = by_profile["Balanced"]
    lines.extend(
        [
            f"Expected BUY: {_format_ratio(balanced.expected_buy_ratio)}",
            f"Expected SELL: {_format_ratio(balanced.expected_sell_ratio)}",
            f"Expected HOLD: {_format_ratio(balanced.expected_hold_ratio)}",
        ]
    )
    return lines


def _format_float(value: float) -> str:
    """Format a floating-point statistic for report output."""
    return f"{value:.8g}"


def _format_ratio(value: float) -> str:
    """Format a ratio as a percentage string."""
    return f"{value * 100.0:.4f}%"


def _format_duration(duration_seconds: float) -> str:
    """Format duration seconds for the summary report."""
    return f"{duration_seconds:.3f}s"


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
