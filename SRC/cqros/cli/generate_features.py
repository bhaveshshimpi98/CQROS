"""CQROS feature-generation CLI.

Purpose:
    Provide an argparse-based production entry point that discovers processed
    market partitions and executes ``FeaturePipeline`` across the universe
    with bounded symbol concurrency.

Responsibilities:
    - Parse CLI arguments for feature generation
    - Discover available processed OHLCV partitions
    - Load and join companion processed series into a feature input frame
    - Align the joined frame to the first row with complete companion inputs
    - Execute ``FeaturePipeline`` and persist via ``FeatureRepository``
    - Honor ``--overwrite``, worker concurrency, and debug logging
    - Print deterministic progress and a final summary

Dependencies:
    ``argparse``, ``asyncio``, ``polars``, ``cqros.config``, ``cqros.core``,
    ``cqros.features``, and ``cqros.storage``.

Public API:
    ``align_feature_input_frame``, ``build_parser``, ``build_options``,
    ``build_feature_pipeline``, ``build_default_registry``, ``discover_work``,
    ``format_summary``, ``load_feature_input_frame``, ``run_generation``, and
    ``main``.

Notes:
    This module is a thin composition root. It does not implement feature
    formulas, verification, or repository filesystem walks beyond calling
    repository discovery and load APIs. Companion alignment drops leading
    OHLCV rows that lack complete as-of-joined companion values; it never
    fills missing inputs.
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
    STORAGE_DIR_FEATURES,
    SUPPORTED_TIMEFRAMES,
)
from cqros.core.exceptions import CQROSError, ValidationError
from cqros.core.types import Symbol, Timeframe
from cqros.features import (
    FEATURE_NAMES,
    ATRFeature,
    BuyPressureFeature,
    BuySellRatioFeature,
    CrowdingScoreFeature,
    DeltaVolumeFeature,
    DollarVolumeFeature,
    FeaturePipeline,
    FeatureRegistry,
    FlowImbalanceFeature,
    FundingChangeFeature,
    FundingMomentumFeature,
    FundingRollingMeanFeature,
    FundingZScoreFeature,
    LogReturnsFeature,
    OIChangeFeature,
    OIMomentumFeature,
    OIPercentChangeFeature,
    OIRollingMeanFeature,
    OIZScoreFeature,
    RatioChangeFeature,
    RatioMomentumFeature,
    RatioZScoreFeature,
    ReturnsFeature,
    RollingMaxFeature,
    RollingMeanFeature,
    RollingMinFeature,
    RollingStdFeature,
    SellPressureFeature,
)
from cqros.ingestion import DEFAULT_FUNDING_TIMEFRAME
from cqros.storage import (
    DatasetNotFoundError,
    FeatureRepository,
    ParquetStore,
    ProcessedMarketDataRepository,
    ProcessedPartitionRef,
    StorageLayout,
)

__all__ = [
    "DiscoveredWorkItem",
    "FeatureGenerationOptions",
    "FeatureGenerationSummary",
    "FeatureTaskResult",
    "align_feature_input_frame",
    "build_default_registry",
    "build_feature_pipeline",
    "build_options",
    "build_parser",
    "discover_work",
    "format_summary",
    "load_feature_input_frame",
    "main",
    "run_generation",
]

_logger = logging.getLogger(__name__)

_EXIT_SUCCESS: Final[int] = 0
_EXIT_FAILURE: Final[int] = 1

_DEFAULT_WORKER_COUNT: Final[int] = ResearchConfig().worker_count

_EXCHANGE: Final[str] = EXCHANGE_BINANCE
_MARKET: Final[str] = MARKET_USDT_PERPETUAL

_ERROR_WORKERS: Final[str] = "CLI-GENERATE-FEATURES-001"
_ERROR_TIMEFRAME: Final[str] = "CLI-GENERATE-FEATURES-002"
_ERROR_YEAR: Final[str] = "CLI-GENERATE-FEATURES-003"
_ERROR_OHLCV_COLUMNS: Final[str] = "CLI-GENERATE-FEATURES-004"
_ERROR_COMPANION_COLUMNS: Final[str] = "CLI-GENERATE-FEATURES-005"
_ERROR_COMPANION_ALIGNMENT: Final[str] = "CLI-GENERATE-FEATURES-006"

_COL_OPEN_TIME: Final[str] = "open_time"
_COL_SYMBOL: Final[str] = "symbol"
_COL_TIMEFRAME: Final[str] = "timeframe"
_COL_CLOSE: Final[str] = "close"
_COL_HIGH: Final[str] = "high"
_COL_LOW: Final[str] = "low"
_COL_VOLUME: Final[str] = "volume"
_COL_FUNDING_TIME: Final[str] = "funding_time"
_COL_FUNDING_RATE: Final[str] = "funding_rate"
_COL_TIMESTAMP: Final[str] = "timestamp"
_COL_OPEN_INTEREST: Final[str] = "open_interest"
_COL_BUY_VOLUME: Final[str] = "buy_volume"
_COL_SELL_VOLUME: Final[str] = "sell_volume"
_COL_LONG_SHORT_RATIO: Final[str] = "long_short_ratio"

# Processed funding partitions are stored under the native settlement cadence
# (``8h``), not under the OHLCV bar timeframe used for feature generation.
_FUNDING_STORAGE_TIMEFRAME: Final[Timeframe] = DEFAULT_FUNDING_TIMEFRAME

# Input columns required by the full Feature Engine catalog before pipeline run.
_REQUIRED_FEATURE_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    _COL_SYMBOL,
    _COL_TIMEFRAME,
    _COL_OPEN_TIME,
    _COL_CLOSE,
    _COL_HIGH,
    _COL_LOW,
    _COL_VOLUME,
    _COL_FUNDING_RATE,
    _COL_OPEN_INTEREST,
    _COL_BUY_VOLUME,
    _COL_SELL_VOLUME,
    _COL_LONG_SHORT_RATIO,
)

# Companion value columns that must be non-null before feature execution.
_REQUIRED_COMPANION_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    _COL_FUNDING_RATE,
    _COL_OPEN_INTEREST,
    _COL_BUY_VOLUME,
    _COL_SELL_VOLUME,
    _COL_LONG_SHORT_RATIO,
)


@dataclass(frozen=True, slots=True)
class FeatureGenerationOptions:
    """Immutable CLI options for feature generation.

    Attributes:
        storage_root: Storage root containing ``processed`` and ``features``.
        symbols: Optional symbol allowlist. ``None`` discovers all.
        timeframes: Optional timeframe allowlist. ``None`` discovers all.
        years: Optional year allowlist. ``None`` discovers all.
        overwrite: When ``True``, regenerate existing feature partitions.
        workers: Maximum concurrent symbols.
        verbose: When ``True``, enable INFO logging.
        debug: When ``True``, enable DEBUG logging and log full failure
            tracebacks.
    """

    storage_root: Path
    symbols: tuple[Symbol, ...] | None
    timeframes: tuple[Timeframe, ...] | None
    years: tuple[int, ...] | None
    overwrite: bool
    workers: int
    verbose: bool
    debug: bool


@dataclass(frozen=True, slots=True)
class DiscoveredWorkItem:
    """One discovered processed partition group ready for feature generation.

    Attributes:
        symbol: Processed symbol.
        timeframe: Available bar interval.
        years: Calendar years with existing processed OHLCV parquet partitions.
    """

    symbol: Symbol
    timeframe: Timeframe
    years: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FeatureTaskResult:
    """Immutable result for one symbol/timeframe/year generation task.

    Attributes:
        symbol: Processed symbol.
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
class FeatureGenerationSummary:
    """Immutable aggregate summary for a feature generation run.

    Attributes:
        symbols_discovered: Unique symbols discovered from processed storage.
        symbols_processed: Unique symbols for which generation was attempted.
        timeframes_processed: Unique timeframes attempted.
        successful_tasks: Count of succeeded partition tasks.
        failed_tasks: Count of failed partition tasks.
        skipped_tasks: Count of skipped existing partitions.
        rows_generated: Sum of output rows across successes.
        duration_seconds: Wall-clock generation duration.
        output_directory: Features-tier output directory.
        failed_task_labels: Deterministic failed-task labels for reporting.
    """

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
    """Create the feature-generation argument parser.

    Returns:
        Configured ``ArgumentParser`` for feature generation flags.
    """
    parser = argparse.ArgumentParser(
        prog="cqros-generate-features",
        description=(
            "Generate CQROS merged feature datasets from discovered processed " "market partitions."
        ),
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
        help="Regenerate feature partitions that already exist.",
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


def build_options(args: argparse.Namespace) -> FeatureGenerationOptions:
    """Map parsed CLI arguments onto ``FeatureGenerationOptions``.

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

    return FeatureGenerationOptions(
        storage_root=Path(DEFAULT_STORAGE_ROOT),
        symbols=_normalize_symbols(args.symbols),
        timeframes=_normalize_timeframes(args.timeframes),
        years=_normalize_years(args.years),
        overwrite=bool(args.overwrite),
        workers=workers,
        verbose=bool(args.verbose),
        debug=bool(args.debug),
    )


def build_default_registry() -> FeatureRegistry:
    """Return a registry containing every currently implemented feature.

    Returns:
        ``FeatureRegistry`` populated with the Feature Engine catalog.
    """
    registry = FeatureRegistry()
    registry.register_many(
        (
            ReturnsFeature(),
            LogReturnsFeature(),
            RollingMeanFeature(),
            RollingStdFeature(),
            RollingMaxFeature(),
            RollingMinFeature(),
            ATRFeature(),
            DollarVolumeFeature(),
            FundingChangeFeature(),
            FundingRollingMeanFeature(),
            FundingZScoreFeature(),
            FundingMomentumFeature(),
            OIChangeFeature(),
            OIPercentChangeFeature(),
            OIRollingMeanFeature(),
            OIZScoreFeature(),
            OIMomentumFeature(),
            BuyPressureFeature(),
            SellPressureFeature(),
            BuySellRatioFeature(),
            DeltaVolumeFeature(),
            FlowImbalanceFeature(),
            RatioChangeFeature(),
            RatioMomentumFeature(),
            RatioZScoreFeature(),
            CrowdingScoreFeature(),
        )
    )
    return registry


def build_feature_pipeline(
    options: FeatureGenerationOptions,
    *,
    logger: logging.Logger | None = None,
) -> FeaturePipeline:
    """Compose ``FeaturePipeline`` from shared storage dependencies.

    Args:
        options: Immutable generation options providing the storage root.
        logger: Optional logger forwarded to the pipeline.

    Returns:
        Fully wired ``FeaturePipeline``.
    """
    layout = StorageLayout(options.storage_root)
    datastore = ParquetStore()
    feature_repository = FeatureRepository(layout, datastore)
    return FeaturePipeline(
        build_default_registry(),
        feature_repository,
        logger=logger if logger is not None else _logger,
    )


def discover_work(
    repository: ProcessedMarketDataRepository,
    options: FeatureGenerationOptions,
) -> tuple[DiscoveredWorkItem, ...]:
    """Discover feature-ready processed partitions matching the CLI filters.

    OHLCV year partitions are included only when every companion dataset
    required by the full Feature Engine catalog also exists for that
    symbol/timeframe/year (funding under the native ``8h`` settlement
    timeframe; open interest, taker volume, and long/short under the OHLCV
    bar timeframe). Incomplete years are excluded rather than scheduled as
    mid-pipeline failures.

    Args:
        repository: Processed repository providing discovery APIs.
        options: CLI filters for symbol, timeframe, and year.

    Returns:
        Deterministically ordered discovered work items.
    """
    ohlcv_partitions = repository.discover_partitions(
        datasets=("ohlcv",),
        symbols=options.symbols,
        timeframes=options.timeframes,
    )
    ready = _filter_partitions_with_required_companions(
        repository,
        ohlcv_partitions,
        symbols=options.symbols,
        timeframes=options.timeframes,
    )
    return _group_partitions(ready, year_filter=options.years)


def format_summary(summary: FeatureGenerationSummary) -> str:
    """Render a deterministic feature-generation summary report.

    Args:
        summary: Aggregate generation summary.

    Returns:
        Multi-line summary text suitable for stdout.
    """
    lines = [
        "=====================================",
        "CQROS Feature Generation Summary",
        "=====================================",
        "",
        f"Symbols discovered: {summary.symbols_discovered}",
        f"Symbols processed: {summary.symbols_processed}",
        f"Timeframes processed: {summary.timeframes_processed}",
        f"Successful tasks: {summary.successful_tasks}",
        f"Failed tasks: {summary.failed_tasks}",
        f"Skipped tasks: {summary.skipped_tasks}",
        f"Rows generated: {summary.rows_generated}",
        f"Generation duration: {_format_duration(summary.duration_seconds)}",
        f"Output directory: {_format_output_directory(summary.output_directory)}",
    ]
    if summary.failed_task_labels:
        lines.extend(["", "Failed Tasks", ""])
        lines.extend(f"- {label}" for label in summary.failed_task_labels)
    return "\n".join(lines) + "\n"


def load_feature_input_frame(
    repository: ProcessedMarketDataRepository,
    *,
    symbol: Symbol,
    timeframe: Timeframe,
    year: int,
    exchange: str = _EXCHANGE,
    market: str = _MARKET,
) -> pl.DataFrame:
    """Load processed partitions and assemble a FeaturePipeline input frame.

    OHLCV forms the bar timeline. Companion series required by the full Feature
    Engine catalog are loaded and backward as-of joined on ``open_time``.
    Funding is loaded from the native settlement timeframe (``8h``), not
    ``timeframe``, because processed funding partitions are not duplicated per
    bar interval. Missing companion partitions raise ``DatasetNotFoundError``;
    they are never silently omitted.

    After joins, leading rows that still have null companion inputs are dropped
    so the returned frame starts at the first bar where every required companion
    column is non-null. Pipeline warm-up trimming remains the responsibility of
    ``FeaturePipeline``.

    Args:
        repository: Processed market-data repository.
        symbol: Tradeable symbol.
        timeframe: Bar interval.
        year: Calendar year partition.
        exchange: Exchange identifier.
        market: Market segment.

    Returns:
        Eager DataFrame sorted by ``open_time`` containing feature inputs with
        complete companion coverage from the first row.

    Raises:
        DatasetNotFoundError: If the OHLCV or any required companion partition
            does not exist, or if no row has complete companion coverage.
        ValidationError: If required OHLCV or companion columns are missing.
    """
    ohlcv = repository.load_ohlcv(
        exchange=exchange,
        market=market,
        symbol=symbol,
        timeframe=timeframe,
        year=year,
    )
    frame = _prepare_ohlcv_frame(ohlcv, symbol=symbol, timeframe=timeframe)

    frame = _join_companion(
        frame,
        repository.load_funding(
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=_FUNDING_STORAGE_TIMEFRAME,
            year=year,
        ),
        dataset="funding",
        time_column=_COL_FUNDING_TIME,
        value_columns=(_COL_FUNDING_RATE,),
    )
    frame = _join_companion(
        frame,
        repository.load_open_interest(
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        ),
        dataset="open_interest",
        time_column=_COL_TIMESTAMP,
        value_columns=(_COL_OPEN_INTEREST,),
    )
    frame = _join_companion(
        frame,
        repository.load_taker_volume(
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        ),
        dataset="taker_volume",
        time_column=_COL_TIMESTAMP,
        value_columns=(_COL_BUY_VOLUME, _COL_SELL_VOLUME),
    )
    frame = _join_companion(
        frame,
        repository.load_global_long_short_account_ratio(
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        ),
        dataset="global_long_short_account_ratio",
        time_column=_COL_TIMESTAMP,
        value_columns=(_COL_LONG_SHORT_RATIO,),
    )
    validated = _require_feature_input_columns(frame.sort(_COL_OPEN_TIME))
    return align_feature_input_frame(validated)


def align_feature_input_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Drop leading rows that lack complete companion feature inputs.

    After companion as-of joins, OHLCV bars that precede every required
    companion observation still carry null companion columns. Those leading
    rows are removed so ``FeaturePipeline`` only receives bars where every
    present required companion input is non-null. Values are never filled.

    Args:
        frame: Joined feature-input frame sorted by ``open_time``.

    Returns:
        Frame sliced from the first fully companion-complete row inclusive.

    Raises:
        DatasetNotFoundError: If no row has all existing required companion
            columns non-null.
    """
    companion_columns = tuple(
        name for name in _REQUIRED_COMPANION_INPUT_COLUMNS if name in frame.columns
    )
    if not companion_columns:
        raise DatasetNotFoundError(
            "feature input frame has no required companion columns to align",
            error_code=_ERROR_COMPANION_ALIGNMENT,
            details={
                "required_companion_columns": _REQUIRED_COMPANION_INPUT_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
            recovery_suggestion=(
                "Ensure companion datasets were joined before feature generation."
            ),
        )

    complete_mask = pl.all_horizontal(*(pl.col(name).is_not_null() for name in companion_columns))
    indexed = frame.with_row_index("_cqros_align_idx")
    first_complete = indexed.filter(complete_mask).select("_cqros_align_idx").head(1)
    if first_complete.height == 0:
        raise DatasetNotFoundError(
            "no feature-input rows have complete companion coverage",
            error_code=_ERROR_COMPANION_ALIGNMENT,
            details={
                "required_companion_columns": companion_columns,
                "row_count": frame.height,
            },
            recovery_suggestion=(
                "Extend companion history so it overlaps the OHLCV timeline, "
                "or choose a year/timeframe with overlapping companion data."
            ),
        )
    start_index = int(first_complete.item())
    if start_index == 0:
        return frame
    return frame.slice(start_index)


async def main(argv: Sequence[str] | None = None) -> int:
    """Run the feature-generation CLI.

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
        processed_repository = ProcessedMarketDataRepository(layout, datastore)
        feature_repository = FeatureRepository(layout, datastore)
        pipeline = build_feature_pipeline(options)
        work = discover_work(processed_repository, options)
        summary = await run_generation(
            pipeline=pipeline,
            processed_repository=processed_repository,
            feature_repository=feature_repository,
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
    pipeline: FeaturePipeline,
    processed_repository: ProcessedMarketDataRepository,
    feature_repository: FeatureRepository,
    options: FeatureGenerationOptions,
    work: Sequence[DiscoveredWorkItem],
) -> FeatureGenerationSummary:
    """Execute discovered work through a bounded symbol worker pool.

    Args:
        pipeline: Injected feature pipeline.
        processed_repository: Processed market-data repository.
        feature_repository: Feature partition repository.
        options: Immutable generation options.
        work: Discovered work items.

    Returns:
        Aggregate immutable summary.
    """
    started = time.perf_counter()
    output_directory = options.storage_root / STORAGE_DIR_FEATURES

    if len(work) == 0:
        return FeatureGenerationSummary(
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
        processed_repository=processed_repository,
        feature_repository=feature_repository,
        work_by_symbol=work_by_symbol,
        worker_count=options.workers,
        overwrite=options.overwrite,
        debug=options.debug,
    )
    return _build_summary(
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
    partitions: Sequence[ProcessedPartitionRef],
    *,
    year_filter: tuple[int, ...] | None,
) -> tuple[DiscoveredWorkItem, ...]:
    """Group year partitions into symbol/timeframe work items."""
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


def _filter_partitions_with_required_companions(
    repository: ProcessedMarketDataRepository,
    ohlcv_partitions: Sequence[ProcessedPartitionRef],
    *,
    symbols: tuple[Symbol, ...] | None,
    timeframes: tuple[Timeframe, ...] | None,
) -> tuple[ProcessedPartitionRef, ...]:
    """Keep OHLCV partitions that have every catalog-required companion year."""
    if len(ohlcv_partitions) == 0:
        return ()

    funding_keys = {
        (partition.symbol, partition.year)
        for partition in repository.discover_partitions(
            datasets=("funding",),
            symbols=symbols,
            timeframes=(_FUNDING_STORAGE_TIMEFRAME,),
        )
    }
    open_interest_keys = {
        (partition.symbol, partition.timeframe, partition.year)
        for partition in repository.discover_partitions(
            datasets=("open_interest",),
            symbols=symbols,
            timeframes=timeframes,
        )
    }
    taker_keys = {
        (partition.symbol, partition.timeframe, partition.year)
        for partition in repository.discover_partitions(
            datasets=("taker_volume",),
            symbols=symbols,
            timeframes=timeframes,
        )
    }
    long_short_keys = {
        (partition.symbol, partition.timeframe, partition.year)
        for partition in repository.discover_partitions(
            datasets=("global_long_short_account_ratio",),
            symbols=symbols,
            timeframes=timeframes,
        )
    }

    ready: list[ProcessedPartitionRef] = []
    for partition in ohlcv_partitions:
        bar_key = (partition.symbol, partition.timeframe, partition.year)
        funding_key = (partition.symbol, partition.year)
        if funding_key not in funding_keys:
            continue
        if bar_key not in open_interest_keys:
            continue
        if bar_key not in taker_keys:
            continue
        if bar_key not in long_short_keys:
            continue
        ready.append(partition)
    return tuple(ready)


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
    pipeline: FeaturePipeline,
    processed_repository: ProcessedMarketDataRepository,
    feature_repository: FeatureRepository,
    work_by_symbol: Mapping[Symbol, Sequence[DiscoveredWorkItem]],
    worker_count: int,
    overwrite: bool,
    debug: bool,
) -> tuple[FeatureTaskResult, ...]:
    """Drain symbols through a bounded asyncio worker pool."""
    symbols = tuple(work_by_symbol.keys())
    if len(symbols) == 0:
        return ()

    queue: asyncio.Queue[Symbol | None] = asyncio.Queue()
    for symbol in symbols:
        queue.put_nowait(symbol)
    for _ in range(worker_count):
        queue.put_nowait(None)

    collected: list[FeatureTaskResult] = []
    lock = asyncio.Lock()

    async def worker() -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                results = await _generate_symbol_work(
                    pipeline=pipeline,
                    processed_repository=processed_repository,
                    feature_repository=feature_repository,
                    symbol=item,
                    work_items=work_by_symbol[item],
                    overwrite=overwrite,
                    debug=debug,
                )
                async with lock:
                    collected.extend(results)
            finally:
                queue.task_done()

    worker_tasks = [
        asyncio.create_task(worker(), name=f"generate-features-worker-{index}")
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
    pipeline: FeaturePipeline,
    processed_repository: ProcessedMarketDataRepository,
    feature_repository: FeatureRepository,
    symbol: Symbol,
    work_items: Sequence[DiscoveredWorkItem],
    overwrite: bool,
    debug: bool,
) -> tuple[FeatureTaskResult, ...]:
    """Generate features for every discovered year for one symbol."""
    results: list[FeatureTaskResult] = []
    for item in work_items:
        for year in item.years:
            result = await asyncio.to_thread(
                _generate_partition,
                pipeline,
                processed_repository,
                feature_repository,
                symbol=symbol,
                timeframe=item.timeframe,
                year=year,
                overwrite=overwrite,
                debug=debug,
            )
            _print_progress(result)
            results.append(result)
    return tuple(results)


def _generate_partition(
    pipeline: FeaturePipeline,
    processed_repository: ProcessedMarketDataRepository,
    feature_repository: FeatureRepository,
    *,
    symbol: Symbol,
    timeframe: Timeframe,
    year: int,
    overwrite: bool,
    debug: bool,
) -> FeatureTaskResult:
    """Generate one feature year partition synchronously."""
    if not overwrite and feature_repository.exists(
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=symbol,
        timeframe=timeframe,
        year=year,
    ):
        return FeatureTaskResult(
            symbol=symbol,
            timeframe=timeframe,
            year=year,
            status="skipped",
        )

    try:
        frame = load_feature_input_frame(
            processed_repository,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        )
        output = pipeline.run(
            frame,
            FEATURE_NAMES,
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
        return FeatureTaskResult(
            symbol=symbol,
            timeframe=timeframe,
            year=year,
            status="failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
            error_code=exc.error_code if isinstance(exc, CQROSError) else None,
        )

    return FeatureTaskResult(
        symbol=symbol,
        timeframe=timeframe,
        year=year,
        status="succeeded",
        rows_generated=output.height,
    )


def _print_progress(result: FeatureTaskResult) -> None:
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
            "Failed feature generation partition; continuing",
            extra=log_extra,
            exc_info=True,
        )
    else:
        _logger.warning(
            "Failed feature generation partition; continuing",
            extra=log_extra,
        )


def _build_summary(
    *,
    work: Sequence[DiscoveredWorkItem],
    results: Sequence[FeatureTaskResult],
    duration_seconds: float,
    output_directory: Path,
) -> FeatureGenerationSummary:
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

    return FeatureGenerationSummary(
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


def _prepare_ohlcv_frame(
    frame: pl.DataFrame,
    *,
    symbol: Symbol,
    timeframe: Timeframe,
) -> pl.DataFrame:
    """Normalize OHLCV columns required by the Feature Engine catalog."""
    required = (
        _COL_OPEN_TIME,
        _COL_CLOSE,
        _COL_HIGH,
        _COL_LOW,
        _COL_VOLUME,
    )
    missing = tuple(name for name in required if name not in frame.columns)
    if missing:
        raise ValidationError(
            f"processed OHLCV missing required columns: {list(missing)}",
            error_code=_ERROR_OHLCV_COLUMNS,
            details={
                "missing_columns": missing,
                "available_columns": tuple(frame.columns),
            },
        )

    working = frame
    if _COL_SYMBOL not in working.columns:
        working = working.with_columns(pl.lit(symbol).alias(_COL_SYMBOL))
    if _COL_TIMEFRAME not in working.columns:
        working = working.with_columns(pl.lit(timeframe).alias(_COL_TIMEFRAME))
    return working.select(
        [
            _COL_SYMBOL,
            _COL_TIMEFRAME,
            _COL_OPEN_TIME,
            _COL_CLOSE,
            _COL_HIGH,
            _COL_LOW,
            _COL_VOLUME,
        ]
    )


def _require_feature_input_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Fail fast when the merged input frame lacks catalog-required columns."""
    missing = tuple(name for name in _REQUIRED_FEATURE_INPUT_COLUMNS if name not in frame.columns)
    if missing:
        raise ValidationError(
            f"feature input frame missing required columns: {list(missing)}",
            error_code=_ERROR_COMPANION_COLUMNS,
            details={
                "missing_columns": missing,
                "required_columns": _REQUIRED_FEATURE_INPUT_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )
    return frame


def _join_companion(
    base: pl.DataFrame,
    companion: pl.DataFrame,
    *,
    dataset: str,
    time_column: str,
    value_columns: Sequence[str],
) -> pl.DataFrame:
    """Backward as-of join companion value columns onto ``open_time``."""
    missing = tuple(name for name in (time_column, *value_columns) if name not in companion.columns)
    if missing:
        raise ValidationError(
            f"processed {dataset} missing required columns: {list(missing)}",
            error_code=_ERROR_COMPANION_COLUMNS,
            details={
                "dataset": dataset,
                "missing_columns": missing,
                "available_columns": tuple(companion.columns),
            },
        )
    right = (
        companion.select([time_column, *value_columns])
        .rename({time_column: _COL_OPEN_TIME})
        .sort(_COL_OPEN_TIME)
    )
    left = base.sort(_COL_OPEN_TIME)
    return left.join_asof(right, on=_COL_OPEN_TIME, strategy="backward")


def _format_duration(duration_seconds: float) -> str:
    """Format duration seconds for the summary report."""
    return f"{duration_seconds:.3f}s"


def _format_output_directory(path: Path) -> str:
    """Format the output directory using POSIX separators."""
    return path.as_posix()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
