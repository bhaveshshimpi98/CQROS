"""Unit tests for CQROS historical bootstrap CLI."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from cqros.bootstrap import BootstrapOptions, HistoricalBootstrap
from cqros.cli import main as package_main
from cqros.cli.historical import build_options, build_parser, main
from cqros.core.constants import DEFAULT_STORAGE_ROOT, DEFAULT_TIMEFRAMES
from cqros.core.exceptions import ValidationError


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def test_package_exports_main() -> None:
    """Package public API exposes the historical CLI main entry point."""
    assert package_main is main


def test_build_parser_accepts_all_flags() -> None:
    """Parser accepts every documented historical bootstrap flag."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "--symbol",
            "BTCUSDT",
            "ETHUSDT",
            "--timeframe",
            "1m",
            "1h",
            "--history-days",
            "30",
            "--storage-root",
            "custom-data",
            "--testnet",
        ]
    )
    assert args.symbols == ["BTCUSDT", "ETHUSDT"]
    assert args.timeframes == ["1m", "1h"]
    assert args.history_days == 30
    assert args.storage_root == Path("custom-data")
    assert args.testnet is True


def test_build_parser_defaults_when_flags_omitted() -> None:
    """Omitted optional flags remain unset for BootstrapOptions defaults."""
    parser = build_parser()
    args = parser.parse_args([])
    assert args.symbols is None
    assert args.timeframes is None
    assert args.history_days is None
    assert args.storage_root is None
    assert args.testnet is False


def test_build_options_uses_bootstrap_defaults() -> None:
    """Empty argv maps to BootstrapOptions defaults without inventing values."""
    args = build_parser().parse_args([])
    options = build_options(args)
    assert options.storage_root == Path(DEFAULT_STORAGE_ROOT)
    assert options.timeframes == DEFAULT_TIMEFRAMES
    assert options.symbols == ()
    assert options.history_days is None
    assert options.testnet is False


def test_build_options_maps_explicit_flags() -> None:
    """Explicit CLI flags are forwarded onto BootstrapOptions."""
    args = build_parser().parse_args(
        [
            "--symbol",
            "BTCUSDT",
            "--timeframe",
            "5m",
            "--history-days",
            "7",
            "--storage-root",
            "tmp-data",
            "--testnet",
        ]
    )
    options = build_options(args)
    assert options.symbols == ("BTCUSDT",)
    assert options.timeframes == ("5m",)
    assert options.history_days == 7
    assert options.storage_root == Path("tmp-data")
    assert options.testnet is True


def test_build_options_rejects_empty_timeframes() -> None:
    """Explicit empty --timeframe is rejected by BootstrapOptions validation."""
    args = build_parser().parse_args(["--timeframe"])
    with pytest.raises(ValidationError):
        build_options(args)


def test_main_runs_bootstrap_and_returns_success() -> None:
    """Successful bootstrap returns exit code 0."""
    with patch.object(HistoricalBootstrap, "run", new=AsyncMock()) as run_mock:
        exit_code = _run(main(["--symbol", "BTCUSDT", "--history-days", "1"]))
    assert exit_code == 0
    run_mock.assert_awaited_once()


def test_main_constructs_options_from_argv() -> None:
    """main builds HistoricalBootstrap with options derived from argv."""
    captured: list[BootstrapOptions] = []

    async def _run_stub(self: HistoricalBootstrap) -> None:
        captured.append(self.options)

    with patch.object(HistoricalBootstrap, "run", new=_run_stub):
        exit_code = _run(
            main(
                [
                    "--symbol",
                    "ETHUSDT",
                    "--timeframe",
                    "1h",
                    "--history-days",
                    "14",
                    "--storage-root",
                    "cli-data",
                    "--testnet",
                ]
            )
        )

    assert exit_code == 0
    assert len(captured) == 1
    options = captured[0]
    assert options.symbols == ("ETHUSDT",)
    assert options.timeframes == ("1h",)
    assert options.history_days == 14
    assert options.storage_root == Path("cli-data")
    assert options.testnet is True


def test_main_returns_failure_on_cqros_error() -> None:
    """CQROSError during bootstrap maps to exit code 1."""

    async def _fail(_self: HistoricalBootstrap) -> None:
        raise ValidationError(
            "download range requires start_time or history_days",
            error_code="BOOTSTRAP-HISTORICAL-015",
        )

    with patch.object(HistoricalBootstrap, "run", new=_fail):
        exit_code = _run(main(["--symbol", "BTCUSDT"]))
    assert exit_code == 1
