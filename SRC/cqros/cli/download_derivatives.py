"""CQROS derivatives universe download CLI.

Purpose:
    Provide an argparse-based command-line entry point that discovers the
    Binance Futures universe and downloads every remaining derivatives dataset
    through existing CQROS downloaders.

Responsibilities:
    - Parse CLI arguments for derivatives universe download
    - Discover symbols via ``HistoricalBootstrap.discover_symbols`` (same path
      as ``bootstrap_universe.py``)
    - Compose ``FundingDownloader``, ``OpenInterestDownloader``,
      ``TakerVolumeDownloader``, and ``LongShortDownloader``
    - Run dataset phases in order with a bounded symbol worker pool
    - Print dataset progress and a deterministic final summary

Dependencies:
    ``argparse``, ``asyncio``, ``cqros.bootstrap``, ``cqros.config``,
    ``cqros.ingestion``, and ``cqros.storage``.

Public API:
    ``DEFAULT_DERIVATIVES_PERIODS``, ``build_parser``, ``build_options``,
    ``build_downloaders``, ``format_summary``, ``run_derivatives_download``,
    ``effective_futures_data_history_days``, and ``main``.

Notes:
    This module is a thin composition root. It does not implement exchange
    fetching, planning, persistence, or downloader pagination. Dataset-specific
    lookbacks come from ``DownloadConfig``; Futures Data windows are clamped to
    ``futures_data_history_days - futures_data_safety_margin_days`` before
    downloaders are invoked. Failures for one dataset or symbol do not abort
    the remaining work.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import threading
import time
import traceback
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import polars as pl

from cqros.bootstrap import BootstrapOptions, HistoricalBootstrap
from cqros.bootstrap.universe import DEFAULT_UNIVERSE_WORKER_COUNT
from cqros.config import (
    AppConfig,
    DownloadConfig,
    ExchangeConfig,
    ResearchConfig,
    StorageConfig,
)
from cqros.core.constants import DEFAULT_STORAGE_ROOT, MILLISECONDS_PER_DAY
from cqros.core.exceptions import CQROSError, ExchangeError, ValidationError
from cqros.core.types import FilePath, Symbol, Timeframe, UnixTimestampMs
from cqros.ingestion import (
    LONG_SHORT_PERIODS,
    OPEN_INTEREST_PERIODS,
    TAKER_VOLUME_PERIODS,
    BinanceClient,
    DownloadResult,
    FundingDownloader,
    FundingDownloadPlanner,
    LongShortDownloader,
    LongShortDownloadPlanner,
    LongShortRatioKind,
    OpenInterestDownloader,
    OpenInterestDownloadPlanner,
    TakerVolumeDownloader,
    TakerVolumeDownloadPlanner,
)
from cqros.ingestion.client import (
    DEFAULT_BINANCE_FUTURES_REST_BASE_URL,
    DEFAULT_BINANCE_FUTURES_TESTNET_REST_BASE_URL,
)
from cqros.storage import MarketDataRepository, ParquetStore, StorageLayout
from cqros.storage.interfaces import IDataStore

__all__ = [
    "DATASET_FUNDING",
    "DATASET_GLOBAL_RATIO",
    "DATASET_OPEN_INTEREST",
    "DATASET_TAKER_VOLUME",
    "DATASET_TOP_ACCOUNT_RATIO",
    "DATASET_TOP_POSITION_RATIO",
    "DEFAULT_DERIVATIVES_PERIODS",
    "DEFAULT_DERIVATIVES_WORKER_COUNT",
    "DERIVATIVES_DATASETS",
    "DerivativesDownloadOptions",
    "DerivativesDownloadSummary",
    "DerivativesDownloaders",
    "ResolvedDownloadRange",
    "build_downloaders",
    "build_options",
    "build_parser",
    "effective_futures_data_history_days",
    "format_summary",
    "main",
    "resolve_dataset_download_range",
    "run_derivatives_download",
    "validate_download_window",
]

_logger = logging.getLogger(__name__)

_EXIT_SUCCESS: Final[int] = 0
_EXIT_FAILURE: Final[int] = 1

DEFAULT_DERIVATIVES_WORKER_COUNT: Final[int] = DEFAULT_UNIVERSE_WORKER_COUNT

# Research-oriented defaults for period-based derivatives endpoints.
DEFAULT_DERIVATIVES_PERIODS: Final[tuple[Timeframe, ...]] = ("1h", "4h", "1d")

_ERROR_WORKERS: Final[str] = "CLI-DOWNLOAD-DERIVATIVES-001"
_ERROR_PERIOD: Final[str] = "CLI-DOWNLOAD-DERIVATIVES-003"
_ERROR_SYMBOL: Final[str] = "CLI-DOWNLOAD-DERIVATIVES-004"
_ERROR_HISTORY_DAYS: Final[str] = "CLI-DOWNLOAD-DERIVATIVES-005"
_ERROR_TIME_WINDOW: Final[str] = "CLI-DOWNLOAD-DERIVATIVES-006"
_ERROR_FUTURES_DATA_MARGIN: Final[str] = "CLI-DOWNLOAD-DERIVATIVES-007"

DATASET_FUNDING: Final[str] = "Funding"
DATASET_OPEN_INTEREST: Final[str] = "Open Interest"
DATASET_TAKER_VOLUME: Final[str] = "Taker Volume"
DATASET_GLOBAL_RATIO: Final[str] = "Global Ratio"
DATASET_TOP_ACCOUNT_RATIO: Final[str] = "Top Account Ratio"
DATASET_TOP_POSITION_RATIO: Final[str] = "Top Position Ratio"

DERIVATIVES_DATASETS: Final[tuple[str, ...]] = (
    DATASET_FUNDING,
    DATASET_OPEN_INTEREST,
    DATASET_TAKER_VOLUME,
    DATASET_GLOBAL_RATIO,
    DATASET_TOP_ACCOUNT_RATIO,
    DATASET_TOP_POSITION_RATIO,
)

_SUPPORTED_DERIVATIVES_PERIODS: Final[frozenset[str]] = (
    OPEN_INTEREST_PERIODS & TAKER_VOLUME_PERIODS & LONG_SHORT_PERIODS
)

type _SymbolTask = Callable[[Symbol], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class DerivativesDownloadOptions:
    """Immutable CLI options for derivatives universe download.

    Attributes:
        storage_root: Root directory for market-data artifacts.
        start_time: Inclusive historical range start as UTC Unix milliseconds.
        end_time: Inclusive historical range end as UTC Unix milliseconds.
        history_days: Optional CLI lookback override applied to every dataset
            family before retention clamping.
        symbols: Optional symbol allowlist. Empty means full discovery.
        periods: Aggregation periods for non-funding derivatives datasets.
        max_symbols: Optional upper bound on discovered contracts.
        workers: Maximum concurrent symbols.
        testnet: Whether to target Binance Futures testnet endpoints.
        verbose: When ``True``, enable INFO logging.
        debug: When ``True``, enable DEBUG logging and failure stack traces.
        app: Application identity settings used for composition defaults.
        storage: Storage path settings used for composition defaults.
        exchange: Exchange connectivity settings used for composition defaults.
        research: Research defaults retained for composition compatibility.
        download: Dataset-specific download retention defaults.
    """

    storage_root: Path
    start_time: UnixTimestampMs | None
    end_time: UnixTimestampMs | None
    history_days: int | None
    symbols: tuple[Symbol, ...]
    periods: tuple[Timeframe, ...]
    max_symbols: int | None
    workers: int
    testnet: bool
    verbose: bool
    debug: bool = False
    app: AppConfig = field(default_factory=AppConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)


@dataclass(frozen=True, slots=True)
class ResolvedDownloadRange:
    """Inclusive download window after defaults and retention clamping.

    Attributes:
        start_time: Inclusive coverage window start (Unix ms, UTC).
        end_time: Inclusive coverage window end (Unix ms, UTC).
        history_days_applied: Effective lookback days used before clamping.
        clamped: Whether the start was clamped to configured retention.
    """

    start_time: UnixTimestampMs
    end_time: UnixTimestampMs
    history_days_applied: int
    clamped: bool


@dataclass(frozen=True, slots=True)
class DerivativesDownloaders:
    """Injected downloader collaborators for one derivatives run.

    Attributes:
        funding: Funding-rate downloader.
        open_interest: Open-interest history downloader.
        taker_volume: Taker buy/sell volume downloader.
        long_short: Long/short ratio downloader (all three kinds).
    """

    funding: FundingDownloader
    open_interest: OpenInterestDownloader
    taker_volume: TakerVolumeDownloader
    long_short: LongShortDownloader


@dataclass(frozen=True, slots=True)
class DerivativesDownloadSummary:
    """Immutable aggregate summary for a derivatives universe download.

    Attributes:
        symbols_discovered: Count of symbols discovered from the exchange.
        symbols_processed: Count of symbols for which download was attempted.
        funding_tasks: Count of funding download tasks attempted.
        open_interest_tasks: Count of open-interest download tasks attempted.
        taker_volume_tasks: Count of taker-volume download tasks attempted.
        global_ratio_tasks: Count of global long/short tasks attempted.
        top_account_ratio_tasks: Count of top-trader account tasks attempted.
        top_position_ratio_tasks: Count of top-trader position tasks attempted.
        successful_tasks: Count of succeeded download tasks.
        failed_tasks: Count of failed download tasks.
        rows_downloaded: Sum of persisted partition row counts.
        duration_seconds: Wall-clock download duration.
        output_directory: Storage root used for persisted artifacts.
    """

    symbols_discovered: int
    symbols_processed: int
    funding_tasks: int
    open_interest_tasks: int
    taker_volume_tasks: int
    global_ratio_tasks: int
    top_account_ratio_tasks: int
    top_position_ratio_tasks: int
    successful_tasks: int
    failed_tasks: int
    rows_downloaded: int
    duration_seconds: float
    output_directory: Path


@dataclass(slots=True)
class _RunCounters:
    """Mutable counters shared by symbol workers."""

    funding_tasks: int = 0
    open_interest_tasks: int = 0
    taker_volume_tasks: int = 0
    global_ratio_tasks: int = 0
    top_account_ratio_tasks: int = 0
    top_position_ratio_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    rows_downloaded: int = 0


class _CountingDataStore:
    """``IDataStore`` wrapper that counts persisted rows.

    Args:
        inner: Concrete datastore receiving all I/O.
        counters: Shared mutable counters updated on successful writes.
        lock: Thread lock protecting ``counters.rows_downloaded``.
    """

    __slots__ = ("_inner", "_counters", "_lock")

    _inner: IDataStore
    _counters: _RunCounters
    _lock: threading.Lock

    def __init__(
        self,
        inner: IDataStore,
        counters: _RunCounters,
        lock: threading.Lock,
    ) -> None:
        self._inner = inner
        self._counters = counters
        self._lock = lock

    def write(self, path: FilePath, dataframe: pl.DataFrame) -> None:
        """Persist ``dataframe`` and increment the shared row counter."""
        with self._lock:
            self._counters.rows_downloaded += dataframe.height
        self._inner.write(path, dataframe)

    def read(self, path: FilePath) -> pl.DataFrame:
        """Load a DataFrame from ``path``."""
        return self._inner.read(path)

    def scan(self, path: FilePath) -> pl.LazyFrame:
        """Open a lazy scan over the dataset at ``path``."""
        return self._inner.scan(path)

    def exists(self, path: FilePath) -> bool:
        """Return whether a dataset file exists at ``path``."""
        return self._inner.exists(path)

    def delete(self, path: FilePath) -> None:
        """Delete the dataset file at ``path``."""
        self._inner.delete(path)

    def schema(self, path: FilePath) -> pl.Schema:
        """Return the stored schema for the dataset at ``path``."""
        return self._inner.schema(path)

    def row_count(self, path: FilePath) -> int:
        """Return the number of rows stored at ``path``."""
        return self._inner.row_count(path)


def build_parser() -> argparse.ArgumentParser:
    """Create the derivatives download argument parser.

    Returns:
        Configured ``ArgumentParser`` for derivatives universe download flags.
    """
    parser = argparse.ArgumentParser(
        prog="cqros-download-derivatives",
        description=(
            "Download CQROS Binance Futures derivatives datasets for the full "
            "discovered universe (funding, open interest, taker volume, and "
            "long/short ratios)."
        ),
    )

    time_range = parser.add_mutually_exclusive_group()
    time_range.add_argument(
        "--history-days",
        dest="history_days",
        type=int,
        default=None,
        metavar="INT",
        help="Lookback window in days used when start_time is not set.",
    )
    time_range.add_argument(
        "--start-time",
        dest="start_time",
        type=int,
        default=None,
        metavar="UNIX_MS",
        help="Inclusive historical range start as UTC Unix milliseconds.",
    )

    parser.add_argument(
        "--end-time",
        dest="end_time",
        type=int,
        default=None,
        metavar="UNIX_MS",
        help="Inclusive historical range end as UTC Unix milliseconds (default: current UTC).",
    )
    parser.add_argument(
        "--storage-root",
        dest="storage_root",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"Root directory for market-data artifacts (default: {DEFAULT_STORAGE_ROOT}).",
    )
    parser.add_argument(
        "--symbols",
        dest="symbols",
        nargs="*",
        default=None,
        metavar="SYMBOL",
        help="Optional symbol allowlist (0..N values). Omit to discover the full universe.",
    )
    parser.add_argument(
        "--periods",
        dest="periods",
        nargs="*",
        default=None,
        metavar="PERIOD",
        help=(
            "Aggregation periods for open interest, taker volume, and long/short "
            f"datasets (0..N values). Defaults to {', '.join(DEFAULT_DERIVATIVES_PERIODS)}."
        ),
    )
    parser.add_argument(
        "--max-symbols",
        dest="max_symbols",
        type=int,
        default=None,
        metavar="INT",
        help="Optional upper bound on discovered contracts (useful for smoke tests).",
    )
    parser.add_argument(
        "--workers",
        dest="workers",
        type=int,
        default=DEFAULT_DERIVATIVES_WORKER_COUNT,
        metavar="INT",
        help=("Maximum concurrent symbols " f"(default: {DEFAULT_DERIVATIVES_WORKER_COUNT})."),
    )
    parser.add_argument(
        "--testnet",
        action="store_true",
        help="Target Binance Futures testnet endpoints.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO logging.",
    )
    return parser


def build_options(
    args: argparse.Namespace,
    *,
    app: AppConfig | None = None,
    storage: StorageConfig | None = None,
    exchange: ExchangeConfig | None = None,
    research: ResearchConfig | None = None,
    download: DownloadConfig | None = None,
) -> DerivativesDownloadOptions:
    """Map parsed CLI arguments onto ``DerivativesDownloadOptions``.

    Args:
        args: Namespace produced by ``build_parser().parse_args(...)``.
        app: Optional application config defaults.
        storage: Optional storage config defaults.
        exchange: Optional exchange config defaults.
        research: Optional research config defaults.
        download: Optional download retention config defaults.

    Returns:
        Immutable derivatives download options.

    Raises:
        ValidationError: If workers, symbols, periods, history days, or
            bootstrap options derived from these flags fail validation.
    """
    app_config = app if app is not None else AppConfig()
    storage_config = storage if storage is not None else StorageConfig()
    exchange_config = exchange if exchange is not None else ExchangeConfig()
    research_config = research if research is not None else ResearchConfig()
    download_config = download if download is not None else DownloadConfig()

    workers = int(args.workers)
    if workers <= 0:
        raise ValidationError(
            "workers must be greater than 0",
            error_code=_ERROR_WORKERS,
            details={"parameter": "workers", "value": workers},
        )

    storage_root = (
        args.storage_root
        if args.storage_root is not None
        else Path(storage_config.root if storage_config.root else DEFAULT_STORAGE_ROOT)
    )

    history_days = args.history_days
    if history_days is not None and history_days <= 0:
        raise ValidationError(
            "history_days must be greater than 0 when provided",
            error_code=_ERROR_HISTORY_DAYS,
            details={"parameter": "history_days", "value": history_days},
        )

    start_time = args.start_time
    end_time = args.end_time
    if start_time is not None and end_time is not None:
        validate_download_window(start_time=start_time, end_time=end_time)

    testnet = bool(args.testnet) or bool(exchange_config.testnet)
    symbols = _normalize_symbols(args.symbols)
    periods = _normalize_periods(args.periods)
    debug = bool(app_config.debug)

    # Validate shared bootstrap invariants (max_symbols, timestamps, etc.).
    BootstrapOptions(
        storage_root=storage_root,
        symbols=symbols,
        start_time=start_time,
        end_time=end_time,
        history_days=history_days,
        max_symbols=args.max_symbols,
        testnet=testnet,
    )

    return DerivativesDownloadOptions(
        storage_root=Path(storage_root),
        start_time=start_time,
        end_time=end_time,
        history_days=history_days,
        symbols=symbols,
        periods=periods,
        max_symbols=args.max_symbols,
        workers=workers,
        testnet=testnet,
        verbose=bool(args.verbose),
        debug=debug,
        app=app_config,
        storage=storage_config,
        exchange=exchange_config,
        research=research_config,
        download=download_config,
    )


def build_downloaders(
    *,
    client: BinanceClient,
    repository: MarketDataRepository,
    logger: logging.Logger | None = None,
) -> DerivativesDownloaders:
    """Compose derivatives downloaders from shared client and repository.

    Args:
        client: Open ``BinanceClient`` for exchange requests.
        repository: Market-data repository used for persistence.
        logger: Optional logger forwarded to downloaders.

    Returns:
        Fully wired ``DerivativesDownloaders`` bundle.
    """
    active_logger = logger if logger is not None else _logger
    return DerivativesDownloaders(
        funding=FundingDownloader(
            client,
            repository,
            FundingDownloadPlanner(),
            logger=active_logger,
        ),
        open_interest=OpenInterestDownloader(
            client,
            repository,
            OpenInterestDownloadPlanner(),
            logger=active_logger,
        ),
        taker_volume=TakerVolumeDownloader(
            client,
            repository,
            TakerVolumeDownloadPlanner(),
            logger=active_logger,
        ),
        long_short=LongShortDownloader(
            client,
            repository,
            LongShortDownloadPlanner(),
            logger=active_logger,
        ),
    )


def format_summary(summary: DerivativesDownloadSummary) -> str:
    """Render a deterministic derivatives download summary report.

    Args:
        summary: Aggregate universe download summary.

    Returns:
        Multi-line summary text suitable for stdout.
    """
    lines = [
        "========================================",
        "Final Summary",
        "========================================",
        "",
        "CQROS Derivatives Download Summary",
        "",
        f"Symbols discovered: {summary.symbols_discovered}",
        f"Symbols processed: {summary.symbols_processed}",
        f"Funding tasks: {summary.funding_tasks}",
        f"Open Interest tasks: {summary.open_interest_tasks}",
        f"Taker Volume tasks: {summary.taker_volume_tasks}",
        f"Global Ratio tasks: {summary.global_ratio_tasks}",
        f"Top Account Ratio tasks: {summary.top_account_ratio_tasks}",
        f"Top Position Ratio tasks: {summary.top_position_ratio_tasks}",
        f"Successful tasks: {summary.successful_tasks}",
        f"Failed tasks: {summary.failed_tasks}",
        f"Rows downloaded: {summary.rows_downloaded}",
        f"Elapsed time: {_format_duration(summary.duration_seconds)}",
        f"Output directory: {_format_output_directory(summary.output_directory)}",
        "========================================",
    ]
    return "\n".join(lines) + "\n"


async def main(argv: Sequence[str] | None = None) -> int:
    """Run the derivatives universe download CLI.

    Args:
        argv: Optional argument vector. When ``None``, ``sys.argv[1:]`` is used.

    Returns:
        ``0`` on completion; ``1`` when a fatal CLI error occurs.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        options = build_options(args)
        _configure_logging(verbose=options.verbose, debug=options.debug)
        bootstrap_options = _to_bootstrap_options(options)
        historical_bootstrap = HistoricalBootstrap(bootstrap_options)
        client = BinanceClient(
            base_url=_resolve_base_url(bootstrap_options),
            timeout=bootstrap_options.timeout,
            max_retries=bootstrap_options.max_retries,
            backoff_factor=bootstrap_options.backoff_factor,
        )
        async with client:
            # Same discovery entry point used by bootstrap_universe / UniverseBootstrap.
            contracts = await historical_bootstrap.discover_symbols()
            symbols = tuple(contract.symbol for contract in contracts)
            funding_range = resolve_dataset_download_range(
                start_time=options.start_time,
                end_time=options.end_time,
                history_days_override=options.history_days,
                default_history_days=options.download.funding_history_days,
            )
            futures_data_history_days = effective_futures_data_history_days(options.download)
            futures_data_range = resolve_dataset_download_range(
                start_time=options.start_time,
                end_time=options.end_time,
                history_days_override=options.history_days,
                default_history_days=futures_data_history_days,
                max_history_days=futures_data_history_days,
            )
            if futures_data_range.clamped:
                _logger.info(
                    "Clamped Futures Data download window to configured retention",
                    extra={
                        "start_time": futures_data_range.start_time,
                        "end_time": futures_data_range.end_time,
                        "max_history_days": futures_data_history_days,
                        "futures_data_history_days": (options.download.futures_data_history_days),
                        "futures_data_safety_margin_days": (
                            options.download.futures_data_safety_margin_days
                        ),
                        "history_days_applied": futures_data_range.history_days_applied,
                    },
                )
            counters = _RunCounters()
            row_lock = threading.Lock()
            repository = MarketDataRepository(
                StorageLayout(options.storage_root),
                _CountingDataStore(ParquetStore(), counters, row_lock),
            )
            downloaders = build_downloaders(client=client, repository=repository)
            summary = await run_derivatives_download(
                downloaders=downloaders,
                options=options,
                symbols=symbols,
                funding_start_time=funding_range.start_time,
                funding_end_time=funding_range.end_time,
                futures_data_start_time=futures_data_range.start_time,
                futures_data_end_time=futures_data_range.end_time,
                counters=counters,
            )
    except CQROSError as exc:
        print(str(exc), file=sys.stderr)
        return _EXIT_FAILURE
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return _EXIT_FAILURE

    print(format_summary(summary), end="")
    return _EXIT_SUCCESS


async def run_derivatives_download(
    *,
    downloaders: DerivativesDownloaders,
    options: DerivativesDownloadOptions,
    symbols: Sequence[Symbol],
    funding_start_time: UnixTimestampMs,
    funding_end_time: UnixTimestampMs,
    futures_data_start_time: UnixTimestampMs,
    futures_data_end_time: UnixTimestampMs,
    counters: _RunCounters | None = None,
) -> DerivativesDownloadSummary:
    """Download derivatives datasets across symbols with bounded concurrency.

    Datasets execute in the required global order. Within each dataset phase,
    at most ``options.workers`` symbols run concurrently. Periods for
    non-funding datasets run sequentially inside each symbol worker.

    Funding uses the funding coverage window. Open interest, taker volume, and
    long/short ratio datasets use the Futures Data coverage window.

    Args:
        downloaders: Injected downloader collaborators.
        options: Immutable download options.
        symbols: Discovered symbols in discovery order.
        funding_start_time: Inclusive funding coverage window start (Unix ms).
        funding_end_time: Inclusive funding coverage window end (Unix ms).
        futures_data_start_time: Inclusive Futures Data window start (Unix ms).
        futures_data_end_time: Inclusive Futures Data window end (Unix ms).
        counters: Optional shared counters (created when omitted).

    Returns:
        Aggregate immutable summary.

    Raises:
        ValidationError: If either coverage window is invalid.
    """
    validate_download_window(start_time=funding_start_time, end_time=funding_end_time)
    validate_download_window(
        start_time=futures_data_start_time,
        end_time=futures_data_end_time,
    )
    _normalize_periods(options.periods)

    started = time.perf_counter()
    active_counters = counters if counters is not None else _RunCounters()
    symbols_tuple = tuple(symbols)
    periods = options.periods

    if len(symbols_tuple) == 0:
        return _build_summary(
            symbols_discovered=0,
            symbols_processed=0,
            counters=active_counters,
            duration_seconds=time.perf_counter() - started,
            output_directory=options.storage_root,
        )

    counter_lock = asyncio.Lock()
    await _run_dataset_phase(
        title=DATASET_FUNDING,
        symbols=symbols_tuple,
        worker_count=options.workers,
        process_symbol=_make_funding_processor(
            downloaders=downloaders,
            start_time=funding_start_time,
            end_time=funding_end_time,
            counters=active_counters,
            lock=counter_lock,
            debug=options.debug,
        ),
    )
    await _run_dataset_phase(
        title=DATASET_OPEN_INTEREST,
        symbols=symbols_tuple,
        worker_count=options.workers,
        process_symbol=_make_period_processor(
            downloaders=downloaders,
            dataset="open_interest",
            periods=periods,
            start_time=futures_data_start_time,
            end_time=futures_data_end_time,
            counters=active_counters,
            lock=counter_lock,
            debug=options.debug,
        ),
    )
    await _run_dataset_phase(
        title=DATASET_TAKER_VOLUME,
        symbols=symbols_tuple,
        worker_count=options.workers,
        process_symbol=_make_period_processor(
            downloaders=downloaders,
            dataset="taker_volume",
            periods=periods,
            start_time=futures_data_start_time,
            end_time=futures_data_end_time,
            counters=active_counters,
            lock=counter_lock,
            debug=options.debug,
        ),
    )
    await _run_dataset_phase(
        title=DATASET_GLOBAL_RATIO,
        symbols=symbols_tuple,
        worker_count=options.workers,
        process_symbol=_make_period_processor(
            downloaders=downloaders,
            dataset="global_ratio",
            periods=periods,
            start_time=futures_data_start_time,
            end_time=futures_data_end_time,
            counters=active_counters,
            lock=counter_lock,
            debug=options.debug,
            kind=LongShortRatioKind.GLOBAL_ACCOUNT,
        ),
    )
    await _run_dataset_phase(
        title=DATASET_TOP_ACCOUNT_RATIO,
        symbols=symbols_tuple,
        worker_count=options.workers,
        process_symbol=_make_period_processor(
            downloaders=downloaders,
            dataset="top_account_ratio",
            periods=periods,
            start_time=futures_data_start_time,
            end_time=futures_data_end_time,
            counters=active_counters,
            lock=counter_lock,
            debug=options.debug,
            kind=LongShortRatioKind.TOP_TRADER_ACCOUNT,
        ),
    )
    await _run_dataset_phase(
        title=DATASET_TOP_POSITION_RATIO,
        symbols=symbols_tuple,
        worker_count=options.workers,
        process_symbol=_make_period_processor(
            downloaders=downloaders,
            dataset="top_position_ratio",
            periods=periods,
            start_time=futures_data_start_time,
            end_time=futures_data_end_time,
            counters=active_counters,
            lock=counter_lock,
            debug=options.debug,
            kind=LongShortRatioKind.TOP_TRADER_POSITION,
        ),
    )

    return _build_summary(
        symbols_discovered=len(symbols_tuple),
        symbols_processed=len(symbols_tuple),
        counters=active_counters,
        duration_seconds=time.perf_counter() - started,
        output_directory=options.storage_root,
    )


def _configure_logging(*, verbose: bool, debug: bool = False) -> None:
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


def validate_download_window(
    *,
    start_time: object,
    end_time: object,
) -> None:
    """Reject invalid inclusive download windows.

    Args:
        start_time: Inclusive coverage window start (Unix ms, UTC).
        end_time: Inclusive coverage window end (Unix ms, UTC).

    Raises:
        ValidationError: If timestamps are not integers, are negative, or
            ``end_time`` is not strictly greater than ``start_time``.
    """
    if not isinstance(start_time, int) or isinstance(start_time, bool) or start_time < 0:
        raise ValidationError(
            "start_time must be a non-negative Unix millisecond timestamp",
            error_code=_ERROR_TIME_WINDOW,
            details={"parameter": "start_time", "value": start_time},
        )
    if not isinstance(end_time, int) or isinstance(end_time, bool) or end_time < 0:
        raise ValidationError(
            "end_time must be a non-negative Unix millisecond timestamp",
            error_code=_ERROR_TIME_WINDOW,
            details={"parameter": "end_time", "value": end_time},
        )
    if end_time <= start_time:
        raise ValidationError(
            "end_time must be greater than start_time",
            error_code=_ERROR_TIME_WINDOW,
            details={"start_time": start_time, "end_time": end_time},
            recovery_suggestion=("Provide a valid inclusive window where end_time > start_time."),
        )


def effective_futures_data_history_days(download: DownloadConfig) -> int:
    """Return Futures Data lookback after applying the configured safety margin.

    Args:
        download: Dataset retention configuration.

    Returns:
        Effective lookback days used when computing Futures Data start times.

    Raises:
        ValidationError: If the effective lookback is not greater than 0.
    """
    effective_history = (
        download.futures_data_history_days - download.futures_data_safety_margin_days
    )
    if effective_history <= 0:
        raise ValidationError(
            (
                "futures_data_history_days minus futures_data_safety_margin_days "
                "must be greater than 0"
            ),
            error_code=_ERROR_FUTURES_DATA_MARGIN,
            details={
                "futures_data_history_days": download.futures_data_history_days,
                "futures_data_safety_margin_days": (download.futures_data_safety_margin_days),
                "effective_history_days": effective_history,
            },
            recovery_suggestion=(
                "Increase futures_data_history_days or reduce "
                "futures_data_safety_margin_days so the effective lookback "
                "remains positive."
            ),
        )
    return effective_history


def resolve_dataset_download_range(
    *,
    start_time: UnixTimestampMs | None,
    end_time: UnixTimestampMs | None,
    history_days_override: int | None,
    default_history_days: int,
    max_history_days: int | None = None,
    now_ms: UnixTimestampMs | None = None,
) -> ResolvedDownloadRange:
    """Resolve an inclusive download window with optional retention clamping.

    Args:
        start_time: Optional explicit inclusive start (Unix ms, UTC).
        end_time: Optional inclusive end (Unix ms, UTC). Defaults to now.
        history_days_override: Optional CLI lookback override.
        default_history_days: Dataset-family default lookback from
            ``DownloadConfig``.
        max_history_days: Optional retention ceiling. When set, ``start_time``
            is clamped so the window never exceeds this many days.
        now_ms: Optional clock override for deterministic tests.

    Returns:
        Resolved inclusive window after defaults and clamping.

    Raises:
        ValidationError: If history days or the resulting window are invalid.
    """
    if default_history_days <= 0:
        raise ValidationError(
            "default_history_days must be greater than 0",
            error_code=_ERROR_HISTORY_DAYS,
            details={"parameter": "default_history_days", "value": default_history_days},
        )
    if history_days_override is not None and history_days_override <= 0:
        raise ValidationError(
            "history_days must be greater than 0 when provided",
            error_code=_ERROR_HISTORY_DAYS,
            details={"parameter": "history_days", "value": history_days_override},
        )
    if max_history_days is not None and max_history_days <= 0:
        raise ValidationError(
            "max_history_days must be greater than 0 when provided",
            error_code=_ERROR_HISTORY_DAYS,
            details={"parameter": "max_history_days", "value": max_history_days},
        )

    resolved_end = (
        end_time
        if end_time is not None
        else (now_ms if now_ms is not None else int(datetime.now(UTC).timestamp() * 1000))
    )

    if start_time is not None:
        resolved_start = start_time
        applied_days = max(
            1,
            (resolved_end - resolved_start + MILLISECONDS_PER_DAY - 1) // MILLISECONDS_PER_DAY,
        )
    elif history_days_override is not None:
        applied_days = history_days_override
        resolved_start = max(0, resolved_end - (applied_days * MILLISECONDS_PER_DAY))
    else:
        applied_days = default_history_days
        resolved_start = max(0, resolved_end - (applied_days * MILLISECONDS_PER_DAY))

    clamped = False
    if max_history_days is not None:
        earliest_allowed = max(
            0,
            resolved_end - (max_history_days * MILLISECONDS_PER_DAY),
        )
        if resolved_start < earliest_allowed:
            resolved_start = earliest_allowed
            clamped = True

    validate_download_window(start_time=resolved_start, end_time=resolved_end)
    return ResolvedDownloadRange(
        start_time=resolved_start,
        end_time=resolved_end,
        history_days_applied=applied_days,
        clamped=clamped,
    )


def _normalize_symbols(values: Sequence[str] | None) -> tuple[Symbol, ...]:
    """Validate and freeze an optional symbol allowlist."""
    if values is None:
        return ()
    normalized: list[Symbol] = []
    for index, symbol in enumerate(values):
        cleaned = symbol.strip()
        if cleaned == "":
            raise ValidationError(
                "symbols entries must be non-empty strings",
                error_code=_ERROR_SYMBOL,
                details={"parameter": "symbols", "index": index, "value": symbol},
            )
        if cleaned not in normalized:
            normalized.append(cleaned)
    return tuple(normalized)


def _normalize_periods(values: Sequence[str] | None) -> tuple[Timeframe, ...]:
    """Validate and freeze aggregation periods for non-funding datasets."""
    selected = tuple(values) if values is not None else DEFAULT_DERIVATIVES_PERIODS
    if not selected:
        raise ValidationError(
            "periods must contain at least one period",
            error_code=_ERROR_PERIOD,
            details={"parameter": "periods"},
        )
    normalized: list[Timeframe] = []
    for period in selected:
        if period not in _SUPPORTED_DERIVATIVES_PERIODS:
            raise ValidationError(
                f"unsupported derivatives period: {period}",
                error_code=_ERROR_PERIOD,
                details={
                    "parameter": "periods",
                    "value": period,
                    "supported": tuple(sorted(_SUPPORTED_DERIVATIVES_PERIODS)),
                },
            )
        if period not in normalized:
            normalized.append(period)
    return tuple(normalized)


def _to_bootstrap_options(options: DerivativesDownloadOptions) -> BootstrapOptions:
    """Map derivatives options onto ``BootstrapOptions`` for discovery."""
    return BootstrapOptions(
        storage_root=options.storage_root,
        symbols=options.symbols,
        start_time=options.start_time,
        end_time=options.end_time,
        history_days=options.history_days,
        max_symbols=options.max_symbols,
        testnet=options.testnet,
    )


def _resolve_base_url(options: BootstrapOptions) -> str:
    """Resolve the REST base URL from bootstrap options."""
    if options.base_url is not None:
        return options.base_url.strip().rstrip("/")
    if options.testnet:
        return DEFAULT_BINANCE_FUTURES_TESTNET_REST_BASE_URL
    return DEFAULT_BINANCE_FUTURES_REST_BASE_URL


async def _run_dataset_phase(
    *,
    title: str,
    symbols: Sequence[Symbol],
    worker_count: int,
    process_symbol: _SymbolTask,
) -> None:
    """Run one dataset phase across symbols with a bounded worker pool."""
    _print_section(title)
    await _run_worker_pool(
        symbols=symbols,
        worker_count=worker_count,
        process_symbol=process_symbol,
    )
    print(f"{title} complete.", flush=True)
    print(flush=True)


async def _run_worker_pool(
    *,
    symbols: Sequence[Symbol],
    worker_count: int,
    process_symbol: _SymbolTask,
) -> None:
    """Drain symbols through a bounded asyncio worker pool."""
    queue: asyncio.Queue[Symbol | None] = asyncio.Queue()
    for symbol in symbols:
        queue.put_nowait(symbol)
    for _ in range(worker_count):
        queue.put_nowait(None)

    async def worker() -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                await process_symbol(item)
            finally:
                queue.task_done()

    worker_tasks = [
        asyncio.create_task(worker(), name=f"download-derivatives-worker-{index}")
        for index in range(worker_count)
    ]
    try:
        await asyncio.gather(*worker_tasks)
    finally:
        for task in worker_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*worker_tasks, return_exceptions=True)


def _make_funding_processor(
    *,
    downloaders: DerivativesDownloaders,
    start_time: UnixTimestampMs,
    end_time: UnixTimestampMs,
    counters: _RunCounters,
    lock: asyncio.Lock,
    debug: bool = False,
) -> _SymbolTask:
    """Build the per-symbol funding download coroutine factory."""

    async def process(symbol: Symbol) -> None:
        print(f"{symbol} {DATASET_FUNDING}", flush=True)
        result, failure = await _run_task(
            downloaders.funding.download_symbol(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
            ),
            dataset=DATASET_FUNDING,
            symbol=symbol,
            debug=debug,
        )
        _print_download_result(result, failure=failure)
        async with lock:
            counters.funding_tasks += 1
            if result is not None:
                counters.successful_tasks += 1
            else:
                counters.failed_tasks += 1

    return process


def _make_period_processor(
    *,
    downloaders: DerivativesDownloaders,
    dataset: str,
    periods: Sequence[Timeframe],
    start_time: UnixTimestampMs,
    end_time: UnixTimestampMs,
    counters: _RunCounters,
    lock: asyncio.Lock,
    debug: bool = False,
    kind: LongShortRatioKind | None = None,
) -> _SymbolTask:
    """Build a per-symbol processor that downloads every configured period."""

    async def process(symbol: Symbol) -> None:
        for period in periods:
            label = _dataset_progress_label(dataset=dataset, period=period)
            print(f"{symbol} {label}", flush=True)
            if dataset == "open_interest":
                result, failure = await _run_task(
                    downloaders.open_interest.download_symbol(
                        symbol=symbol,
                        start_time=start_time,
                        end_time=end_time,
                        period=period,
                    ),
                    dataset=DATASET_OPEN_INTEREST,
                    symbol=symbol,
                    period=period,
                    debug=debug,
                )
                _print_download_result(result, failure=failure)
                async with lock:
                    counters.open_interest_tasks += 1
                    if result is not None:
                        counters.successful_tasks += 1
                    else:
                        counters.failed_tasks += 1
            elif dataset == "taker_volume":
                result, failure = await _run_task(
                    downloaders.taker_volume.download_symbol(
                        symbol=symbol,
                        start_time=start_time,
                        end_time=end_time,
                        period=period,
                    ),
                    dataset=DATASET_TAKER_VOLUME,
                    symbol=symbol,
                    period=period,
                    debug=debug,
                )
                _print_download_result(result, failure=failure)
                async with lock:
                    counters.taker_volume_tasks += 1
                    if result is not None:
                        counters.successful_tasks += 1
                    else:
                        counters.failed_tasks += 1
            elif dataset == "global_ratio":
                assert kind is not None
                result, failure = await _run_task(
                    downloaders.long_short.download_symbol(
                        symbol=symbol,
                        kind=kind,
                        start_time=start_time,
                        end_time=end_time,
                        period=period,
                    ),
                    dataset=DATASET_GLOBAL_RATIO,
                    symbol=symbol,
                    period=period,
                    debug=debug,
                )
                _print_download_result(result, failure=failure)
                async with lock:
                    counters.global_ratio_tasks += 1
                    if result is not None:
                        counters.successful_tasks += 1
                    else:
                        counters.failed_tasks += 1
            elif dataset == "top_account_ratio":
                assert kind is not None
                result, failure = await _run_task(
                    downloaders.long_short.download_symbol(
                        symbol=symbol,
                        kind=kind,
                        start_time=start_time,
                        end_time=end_time,
                        period=period,
                    ),
                    dataset=DATASET_TOP_ACCOUNT_RATIO,
                    symbol=symbol,
                    period=period,
                    debug=debug,
                )
                _print_download_result(result, failure=failure)
                async with lock:
                    counters.top_account_ratio_tasks += 1
                    if result is not None:
                        counters.successful_tasks += 1
                    else:
                        counters.failed_tasks += 1
            else:
                assert kind is not None
                result, failure = await _run_task(
                    downloaders.long_short.download_symbol(
                        symbol=symbol,
                        kind=kind,
                        start_time=start_time,
                        end_time=end_time,
                        period=period,
                    ),
                    dataset=DATASET_TOP_POSITION_RATIO,
                    symbol=symbol,
                    period=period,
                    debug=debug,
                )
                _print_download_result(result, failure=failure)
                async with lock:
                    counters.top_position_ratio_tasks += 1
                    if result is not None:
                        counters.successful_tasks += 1
                    else:
                        counters.failed_tasks += 1

    return process


@dataclass(frozen=True, slots=True)
class _TaskFailure:
    """Structured diagnostics for a failed derivatives download task."""

    dataset: str
    symbol: Symbol
    period: Timeframe | None
    message: str
    http_status: object | None
    binance_code: object | None
    traceback_text: str | None


async def _run_task(
    coro: Awaitable[DownloadResult],
    *,
    dataset: str,
    symbol: Symbol,
    period: Timeframe | None = None,
    debug: bool = False,
) -> tuple[DownloadResult | None, _TaskFailure | None]:
    """Await one downloader call, logging and continuing on failure."""
    try:
        return await coro, None
    except Exception as exc:
        http_status, binance_code = _exchange_error_fields(exc)
        traceback_text = "".join(traceback.format_exception(exc)) if debug else None
        failure = _TaskFailure(
            dataset=dataset,
            symbol=symbol,
            period=period,
            message=str(exc),
            http_status=http_status,
            binance_code=binance_code,
            traceback_text=traceback_text,
        )
        _logger.warning(
            "Derivatives download task failed; continuing",
            extra={
                "dataset": dataset,
                "symbol": symbol,
                "period": period,
                "http_status": http_status,
                "binance_code": binance_code,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
            exc_info=debug,
        )
        return None, failure


def _exchange_error_fields(exc: BaseException) -> tuple[object | None, object | None]:
    """Extract HTTP status and Binance error code from an exception when present."""
    if isinstance(exc, ExchangeError):
        return exc.details.get("status_code"), exc.details.get("binance_code")
    if isinstance(exc, CQROSError):
        return exc.details.get("status_code"), exc.details.get("binance_code")
    return None, None


def _dataset_progress_label(*, dataset: str, period: Timeframe) -> str:
    """Build a human-readable progress label for a period-based dataset."""
    titles = {
        "open_interest": DATASET_OPEN_INTEREST,
        "taker_volume": DATASET_TAKER_VOLUME,
        "global_ratio": DATASET_GLOBAL_RATIO,
        "top_account_ratio": DATASET_TOP_ACCOUNT_RATIO,
        "top_position_ratio": DATASET_TOP_POSITION_RATIO,
    }
    return f"{titles[dataset]} {period}"


def _print_download_result(
    result: DownloadResult | None,
    *,
    failure: _TaskFailure | None = None,
) -> None:
    """Print resumable download progress for one completed or failed task."""
    if result is not None:
        print(result.format_progress(), flush=True)
        return
    if failure is None:
        print("FAILED", flush=True)
        return
    lines = [
        "FAILED",
        f"Dataset: {failure.dataset}",
        f"Symbol: {failure.symbol}",
        f"Period: {failure.period if failure.period is not None else '-'}",
        f"HTTP status: {failure.http_status if failure.http_status is not None else '-'}",
        (
            "Binance error code: "
            f"{failure.binance_code if failure.binance_code is not None else '-'}"
        ),
        f"Message: {failure.message}",
    ]
    if failure.traceback_text:
        lines.append("Stack trace:")
        lines.append(failure.traceback_text.rstrip())
    print("\n".join(lines), flush=True)


def _print_section(title: str) -> None:
    """Print a dataset section banner."""
    print("========================================", flush=True)
    print(title, flush=True)
    print("========================================", flush=True)
    print(flush=True)


def _build_summary(
    *,
    symbols_discovered: int,
    symbols_processed: int,
    counters: _RunCounters,
    duration_seconds: float,
    output_directory: Path,
) -> DerivativesDownloadSummary:
    """Assemble the immutable final summary from shared counters."""
    return DerivativesDownloadSummary(
        symbols_discovered=symbols_discovered,
        symbols_processed=symbols_processed,
        funding_tasks=counters.funding_tasks,
        open_interest_tasks=counters.open_interest_tasks,
        taker_volume_tasks=counters.taker_volume_tasks,
        global_ratio_tasks=counters.global_ratio_tasks,
        top_account_ratio_tasks=counters.top_account_ratio_tasks,
        top_position_ratio_tasks=counters.top_position_ratio_tasks,
        successful_tasks=counters.successful_tasks,
        failed_tasks=counters.failed_tasks,
        rows_downloaded=counters.rows_downloaded,
        duration_seconds=duration_seconds,
        output_directory=output_directory,
    )


def _format_duration(duration_seconds: float) -> str:
    """Format duration seconds for summary output."""
    return f"{duration_seconds:.3f}s"


def _format_output_directory(path: Path) -> str:
    """Format the output directory using POSIX separators."""
    return path.as_posix()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
