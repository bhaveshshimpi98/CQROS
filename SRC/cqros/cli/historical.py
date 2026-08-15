"""CQROS historical market-data bootstrap CLI.

Purpose:
    Provide an argparse-based command-line entry point that maps user flags
    onto ``BootstrapOptions`` and runs ``HistoricalBootstrap``.

Responsibilities:
    - Parse CLI arguments for historical bootstrap
    - Construct ``BootstrapOptions`` from parsed flags
    - Instantiate ``HistoricalBootstrap`` and await ``run()``
    - Return process exit codes without calling ``sys.exit()``

Dependencies:
    ``argparse``, ``asyncio``, and ``cqros.bootstrap``.

Public API:
    ``build_parser``, ``build_options``, and ``main``.

Notes:
    This module is a thin CLI adapter. It does not discover symbols, download
    market data, validate options beyond argparse parsing, or implement
    logging or progress reporting.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from cqros.bootstrap import BootstrapOptions, HistoricalBootstrap
from cqros.core.constants import DEFAULT_STORAGE_ROOT, DEFAULT_TIMEFRAMES
from cqros.core.exceptions import CQROSError

__all__ = [
    "build_parser",
    "build_options",
    "main",
]

_EXIT_SUCCESS: Final[int] = 0
_EXIT_FAILURE: Final[int] = 1


def build_parser() -> argparse.ArgumentParser:
    """Create the historical bootstrap argument parser.

    Returns:
        Configured ``ArgumentParser`` for historical bootstrap flags.
    """
    parser = argparse.ArgumentParser(
        prog="cqros-historical",
        description="Bootstrap CQROS historical market data from Binance Futures.",
    )
    parser.add_argument(
        "--symbol",
        dest="symbols",
        nargs="*",
        default=None,
        metavar="SYMBOL",
        help=(
            "Optional symbol allowlist (0..N values). Omit to discover the full supported universe."
        ),
    )
    parser.add_argument(
        "--timeframe",
        dest="timeframes",
        nargs="*",
        default=None,
        metavar="TIMEFRAME",
        help=(
            "Candle intervals to bootstrap (0..N values). "
            f"Defaults to {', '.join(DEFAULT_TIMEFRAMES)}."
        ),
    )
    parser.add_argument(
        "--history-days",
        dest="history_days",
        type=int,
        default=None,
        metavar="DAYS",
        help="Lookback window in days used when start_time is not set.",
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
        symbols=tuple(args.symbols) if args.symbols is not None else (),
        history_days=args.history_days,
        testnet=bool(args.testnet),
    )


async def main(argv: Sequence[str] | None = None) -> int:
    """Run the historical bootstrap CLI.

    Args:
        argv: Optional argument vector. When ``None``, ``sys.argv[1:]`` is used.

    Returns:
        ``0`` on success; ``1`` when bootstrap raises a ``CQROSError``.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    options = build_options(args)
    bootstrap = HistoricalBootstrap(options)

    try:
        await bootstrap.run()
    except CQROSError as exc:
        print(str(exc), file=sys.stderr)
        return _EXIT_FAILURE

    return _EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
