"""Unit tests for CQROS historical universe bootstrap CLI."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cqros.bootstrap import BootstrapOptions, HistoricalBootstrap, UniverseBootstrap
from cqros.cli.bootstrap_universe import build_options, build_parser, main
from cqros.core.constants import DEFAULT_STORAGE_ROOT, DEFAULT_TIMEFRAMES
from cqros.core.exceptions import ValidationError
from cqros.ingestion import (
    DEFAULT_DOWNLOAD_BATCH_SIZE,
    DEFAULT_DOWNLOAD_WORKERS,
    BinanceClient,
    HistoricalDownloader,
)


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def test_build_options_default_timeframes() -> None:
    """Omitted --timeframe maps to DEFAULT_TIMEFRAMES."""
    args = build_parser().parse_args([])
    options = build_options(args)
    assert options.timeframes == DEFAULT_TIMEFRAMES
    assert options.storage_root == Path(DEFAULT_STORAGE_ROOT)
    assert options.history_days is None
    assert options.start_time is None
    assert options.end_time is None
    assert options.max_symbols is None
    assert options.testnet is False
    assert options.workers == DEFAULT_DOWNLOAD_WORKERS
    assert options.batch_size == DEFAULT_DOWNLOAD_BATCH_SIZE


def test_build_options_single_timeframe() -> None:
    """A single --timeframe is preserved as a one-element tuple."""
    args = build_parser().parse_args(["--timeframe", "1h"])
    options = build_options(args)
    assert options.timeframes == ("1h",)


def test_build_options_multiple_timeframes_preserve_order() -> None:
    """Repeated --timeframe flags preserve user order."""
    args = build_parser().parse_args(
        [
            "--timeframe",
            "1h",
            "--timeframe",
            "4h",
            "--timeframe",
            "1d",
        ]
    )
    options = build_options(args)
    assert options.timeframes == ("1h", "4h", "1d")


def test_build_options_history_days() -> None:
    """--history-days is forwarded onto BootstrapOptions."""
    args = build_parser().parse_args(["--history-days", "30"])
    options = build_options(args)
    assert options.history_days == 30
    assert options.start_time is None


def test_build_options_start_time() -> None:
    """--start-time is forwarded onto BootstrapOptions."""
    args = build_parser().parse_args(["--start-time", "1700000000000"])
    options = build_options(args)
    assert options.start_time == 1_700_000_000_000
    assert options.history_days is None


def test_build_options_end_time_storage_root_and_max_symbols() -> None:
    """Optional end-time, storage-root, and max-symbols map correctly."""
    args = build_parser().parse_args(
        [
            "--end-time",
            "1700000100000",
            "--storage-root",
            "tmp-universe",
            "--max-symbols",
            "5",
            "--history-days",
            "7",
        ]
    )
    options = build_options(args)
    assert options.end_time == 1_700_000_100_000
    assert options.storage_root == Path("tmp-universe")
    assert options.max_symbols == 5
    assert options.history_days == 7


def test_build_options_testnet() -> None:
    """--testnet sets BootstrapOptions.testnet."""
    args = build_parser().parse_args(["--testnet"])
    options = build_options(args)
    assert options.testnet is True


def test_build_options_workers_and_batch_size() -> None:
    """--workers and --batch-size map onto BootstrapOptions."""
    args = build_parser().parse_args(["--workers", "4", "--batch-size", "25"])
    options = build_options(args)
    assert options.workers == 4
    assert options.batch_size == 25


def test_build_options_rejects_non_positive_workers() -> None:
    """Non-positive --workers fails BootstrapOptions validation."""
    args = build_parser().parse_args(["--workers", "0"])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "BOOTSTRAP-HISTORICAL-017"


def test_build_options_rejects_non_positive_batch_size() -> None:
    """Non-positive --batch-size fails BootstrapOptions validation."""
    args = build_parser().parse_args(["--batch-size", "0"])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "BOOTSTRAP-HISTORICAL-018"


def test_parser_rejects_history_days_with_start_time() -> None:
    """--history-days and --start-time are mutually exclusive."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--history-days", "7", "--start-time", "1000"])


def test_build_options_rejects_unsupported_timeframe() -> None:
    """Unsupported --timeframe values fail BootstrapOptions validation."""
    args = build_parser().parse_args(["--timeframe", "2x"])
    with pytest.raises(ValidationError):
        build_options(args)


def test_build_options_rejects_non_positive_max_symbols() -> None:
    """Non-positive --max-symbols fails BootstrapOptions validation."""
    args = build_parser().parse_args(["--max-symbols", "0"])
    with pytest.raises(ValidationError):
        build_options(args)


def test_main_wires_universe_bootstrap_dependencies() -> None:
    """main wires storage/ingestion services into UniverseBootstrap then awaits run."""
    universe = MagicMock(spec=UniverseBootstrap)
    universe.run = AsyncMock(return_value=None)
    client = MagicMock(spec=BinanceClient)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)

    with (
        patch(
            "cqros.cli.bootstrap_universe.BinanceClient",
            return_value=client,
        ) as client_cls,
        patch(
            "cqros.cli.bootstrap_universe.build_universe_bootstrap",
            return_value=universe,
        ) as build_universe,
    ):
        exit_code = _run(
            main(
                [
                    "--history-days",
                    "1",
                    "--timeframe",
                    "1h",
                    "--max-symbols",
                    "2",
                    "--testnet",
                ]
            )
        )

    assert exit_code == 0
    client_cls.assert_called_once()
    build_universe.assert_called_once()
    options = build_universe.call_args.args[0]
    assert isinstance(options, BootstrapOptions)
    assert options.history_days == 1
    assert options.timeframes == ("1h",)
    assert options.max_symbols == 2
    assert options.testnet is True
    assert build_universe.call_args.kwargs["client"] is client
    universe.run.assert_awaited_once_with()


def test_main_returns_failure_on_cqros_error() -> None:
    """CQROSError during universe run maps to exit code 1."""
    universe = MagicMock(spec=UniverseBootstrap)
    universe.run = AsyncMock(
        side_effect=ValidationError(
            "Requested bootstrap symbols were not discovered",
            error_code="BOOTSTRAP-HISTORICAL-014",
        )
    )
    client = MagicMock(spec=BinanceClient)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)

    with (
        patch(
            "cqros.cli.bootstrap_universe.BinanceClient",
            return_value=client,
        ),
        patch(
            "cqros.cli.bootstrap_universe.build_universe_bootstrap",
            return_value=universe,
        ),
    ):
        exit_code = _run(main(["--history-days", "1"]))

    assert exit_code == 1


def test_build_universe_bootstrap_injects_historical_bootstrap() -> None:
    """Composition root constructs UniverseBootstrap with HistoricalBootstrap."""
    from cqros.cli.bootstrap_universe import build_universe_bootstrap

    options = BootstrapOptions(
        history_days=1,
        timeframes=("1h",),
        workers=3,
        batch_size=10,
    )
    client = MagicMock(spec=BinanceClient)

    with patch(
        "cqros.cli.bootstrap_universe.HistoricalDownloader",
        wraps=HistoricalDownloader,
    ) as downloader_cls:
        universe = build_universe_bootstrap(options, client=client)

    assert isinstance(universe, UniverseBootstrap)
    assert isinstance(universe.historical_bootstrap, HistoricalBootstrap)
    assert universe.historical_bootstrap.options == options
    assert universe.worker_count == options.workers
    assert downloader_cls.call_args.kwargs["workers"] == options.workers
    assert downloader_cls.call_args.kwargs["batch_size"] == options.batch_size
    assert universe.repository is not None
    assert universe.validator is not None
    assert universe.updater is not None
    assert universe.repair_engine is not None
