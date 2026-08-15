"""CQROS historical universe bootstrap CLI.

Purpose:
    Provide an argparse-based command-line entry point that maps user flags
    onto ``BootstrapOptions`` and runs resumable ``UniverseBootstrap``.

Responsibilities:
    - Parse CLI arguments for universe historical bootstrap
    - Construct ``BootstrapOptions`` from parsed flags
    - Wire ``HistoricalBootstrap``, storage, validation, update, and repair
      services into ``UniverseBootstrap``
    - Await ``UniverseBootstrap.run()`` and return process exit codes

Dependencies:
    ``argparse``, ``asyncio``, ``cqros.bootstrap``, ``cqros.ingestion``, and
    ``cqros.storage``.

Public API:
    ``build_parser``, ``build_options``, and ``main``.

Notes:
    This module is a thin CLI composition root. It does not discover symbols,
    download market data, construct storage paths beyond injecting storage
    services, validate beyond argparse and ``BootstrapOptions``, or implement
    logging, retries, or progress reporting.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from cqros.bootstrap import (
    BootstrapOptions,
    HistoricalBootstrap,
    UniverseBootstrap,
)
from cqros.core.constants import DEFAULT_STORAGE_ROOT, DEFAULT_TIMEFRAMES
from cqros.core.exceptions import CQROSError
from cqros.ingestion import (
    DEFAULT_DOWNLOAD_BATCH_SIZE,
    DEFAULT_DOWNLOAD_WORKERS,
    BinanceClient,
    DatasetRepairEngine,
    DownloadPlanner,
    HistoricalDownloader,
    IncrementalUpdater,
    ManifestRepository,
    MarketDataValidator,
)
from cqros.ingestion.client import (
    DEFAULT_BINANCE_FUTURES_REST_BASE_URL,
    DEFAULT_BINANCE_FUTURES_TESTNET_REST_BASE_URL,
)
from cqros.storage import MarketDataRepository, ParquetStore, StorageLayout

__all__ = [
    "build_parser",
    "build_options",
    "build_universe_bootstrap",
    "main",
]

_EXIT_SUCCESS: Final[int] = 0
_EXIT_FAILURE: Final[int] = 1


def build_parser() -> argparse.ArgumentParser:
    """Create the universe bootstrap argument parser.

    Returns:
        Configured ``ArgumentParser`` for universe historical bootstrap flags.
    """
    parser = argparse.ArgumentParser(
        prog="cqros-bootstrap-universe",
        description=(
            "Bootstrap CQROS historical market data for the full discovered "
            "Binance Futures universe."
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
        "--timeframe",
        dest="timeframes",
        action="append",
        default=None,
        metavar="TIMEFRAME",
        help=(
            "Candle interval to bootstrap. Repeat for multiple intervals. "
            f"Defaults to {', '.join(DEFAULT_TIMEFRAMES)}."
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
        default=DEFAULT_DOWNLOAD_WORKERS,
        metavar="INT",
        help=("Maximum concurrent symbols " f"(default: {DEFAULT_DOWNLOAD_WORKERS})."),
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=DEFAULT_DOWNLOAD_BATCH_SIZE,
        metavar="INT",
        help=("Download batch size " f"(default: {DEFAULT_DOWNLOAD_BATCH_SIZE})."),
    )
    parser.add_argument(
        "--testnet",
        action="store_true",
        help="Target Binance Futures testnet endpoints.",
    )
    return parser


def build_options(args: argparse.Namespace) -> BootstrapOptions:
    """Map parsed CLI arguments onto ``BootstrapOptions``.

    Args:
        args: Namespace produced by ``build_parser().parse_args(...)``.

    Returns:
        Immutable bootstrap options for ``HistoricalBootstrap``.

    Raises:
        ValidationError: Propagated from ``BootstrapOptions`` validation.
    """
    return BootstrapOptions(
        storage_root=(
            args.storage_root if args.storage_root is not None else Path(DEFAULT_STORAGE_ROOT)
        ),
        timeframes=(tuple(args.timeframes) if args.timeframes is not None else DEFAULT_TIMEFRAMES),
        start_time=args.start_time,
        end_time=args.end_time,
        history_days=args.history_days,
        max_symbols=args.max_symbols,
        testnet=bool(args.testnet),
        workers=args.workers,
        batch_size=args.batch_size,
    )


def _resolve_base_url(options: BootstrapOptions) -> str:
    """Resolve the REST base URL from bootstrap options.

    Args:
        options: Immutable bootstrap configuration.

    Returns:
        Explicit ``base_url`` when provided; otherwise the production or
        testnet default depending on ``testnet``.
    """
    if options.base_url is not None:
        return options.base_url.strip().rstrip("/")
    if options.testnet:
        return DEFAULT_BINANCE_FUTURES_TESTNET_REST_BASE_URL
    return DEFAULT_BINANCE_FUTURES_REST_BASE_URL


def build_universe_bootstrap(
    options: BootstrapOptions,
    *,
    client: BinanceClient,
) -> UniverseBootstrap:
    """Compose ``UniverseBootstrap`` from options and an openable client.

    Args:
        options: Immutable bootstrap configuration.
        client: Shared Binance client for incremental update and repair paths.

    Returns:
        Fully wired ``UniverseBootstrap`` orchestrator.
    """
    historical_bootstrap = HistoricalBootstrap(options)
    repository = MarketDataRepository(
        StorageLayout(options.storage_root),
        ParquetStore(),
    )
    validator = MarketDataValidator()
    downloader = HistoricalDownloader(
        client,
        repository,
        DownloadPlanner(),
        workers=options.workers,
        batch_size=options.batch_size,
    )
    updater = IncrementalUpdater(
        client,
        repository,
        validator,
        downloader,
    )
    repair_engine = DatasetRepairEngine(
        repository,
        downloader,
        validator,
        ManifestRepository(options.storage_root),
    )
    return UniverseBootstrap(
        historical_bootstrap,
        repository,
        validator,
        updater,
        repair_engine,
        worker_count=options.workers,
    )


async def main(argv: Sequence[str] | None = None) -> int:
    """Run the universe historical bootstrap CLI.

    Args:
        argv: Optional argument vector. When ``None``, ``sys.argv[1:]`` is used.

    Returns:
        ``0`` on success; ``1`` when an exception occurs after argument parsing.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        options = build_options(args)
        client = BinanceClient(
            base_url=_resolve_base_url(options),
            timeout=options.timeout,
            max_retries=options.max_retries,
            backoff_factor=options.backoff_factor,
        )
        universe_bootstrap = build_universe_bootstrap(options, client=client)
        async with client:
            await universe_bootstrap.run()
    except CQROSError as exc:
        print(str(exc), file=sys.stderr)
        return _EXIT_FAILURE
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return _EXIT_FAILURE

    return _EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
