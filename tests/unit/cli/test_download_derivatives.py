"""Unit tests for CQROS derivatives universe download CLI."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cqros.bootstrap import BootstrapOptions, HistoricalBootstrap
from cqros.cli.download_derivatives import (
    DATASET_FUNDING,
    DATASET_GLOBAL_RATIO,
    DATASET_OPEN_INTEREST,
    DATASET_TAKER_VOLUME,
    DATASET_TOP_ACCOUNT_RATIO,
    DATASET_TOP_POSITION_RATIO,
    DEFAULT_DERIVATIVES_PERIODS,
    DEFAULT_DERIVATIVES_WORKER_COUNT,
    DERIVATIVES_DATASETS,
    DerivativesDownloaders,
    DerivativesDownloadOptions,
    DerivativesDownloadSummary,
    build_downloaders,
    build_options,
    build_parser,
    effective_futures_data_history_days,
    format_summary,
    main,
    resolve_dataset_download_range,
    run_derivatives_download,
    validate_download_window,
)
from cqros.config import (
    AppConfig,
    DownloadConfig,
    ExchangeConfig,
    StorageConfig,
)
from cqros.core.constants import DEFAULT_STORAGE_ROOT, MILLISECONDS_PER_DAY
from cqros.core.exceptions import ExchangeValidationError, ValidationError
from cqros.data.contracts import Contract
from cqros.ingestion import (
    BinanceClient,
    DownloadResult,
    DownloadStatus,
    FundingDownloader,
    LongShortDownloader,
    LongShortRatioKind,
    OpenInterestDownloader,
    TakerVolumeDownloader,
)
from cqros.storage import MarketDataRepository

_END = 1_700_000_100_000
_START = 1_700_000_000_000


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def _options(
    *,
    storage_root: Path | None = None,
    history_days: int | None = 7,
    start_time: int | None = None,
    end_time: int | None = None,
    symbols: tuple[str, ...] = (),
    periods: tuple[str, ...] = DEFAULT_DERIVATIVES_PERIODS,
    max_symbols: int | None = None,
    workers: int = DEFAULT_DERIVATIVES_WORKER_COUNT,
    testnet: bool = False,
    verbose: bool = False,
    debug: bool = False,
    download: DownloadConfig | None = None,
) -> DerivativesDownloadOptions:
    """Build derivatives options for unit tests."""
    return DerivativesDownloadOptions(
        storage_root=storage_root if storage_root is not None else Path(DEFAULT_STORAGE_ROOT),
        start_time=start_time,
        end_time=end_time,
        history_days=history_days,
        symbols=symbols,
        periods=periods,
        max_symbols=max_symbols,
        workers=workers,
        testnet=testnet,
        verbose=verbose,
        debug=debug,
        download=download if download is not None else DownloadConfig(),
    )


def _contract(symbol: str) -> MagicMock:
    """Create a minimal contract-like fixture exposing ``symbol``."""
    contract = MagicMock(spec=Contract)
    contract.symbol = symbol
    return contract


def _downloaders() -> tuple[DerivativesDownloaders, MagicMock, MagicMock, MagicMock, MagicMock]:
    """Compose mocked downloaders returning a DerivativesDownloaders bundle."""
    success = DownloadResult(status=DownloadStatus.FULL, rows_downloaded=1)
    funding = MagicMock(spec=FundingDownloader)
    funding.download_symbol = AsyncMock(return_value=success)
    open_interest = MagicMock(spec=OpenInterestDownloader)
    open_interest.download_symbol = AsyncMock(return_value=success)
    taker_volume = MagicMock(spec=TakerVolumeDownloader)
    taker_volume.download_symbol = AsyncMock(return_value=success)
    long_short = MagicMock(spec=LongShortDownloader)
    long_short.download_symbol = AsyncMock(return_value=success)
    bundle = DerivativesDownloaders(
        funding=funding,
        open_interest=open_interest,
        taker_volume=taker_volume,
        long_short=long_short,
    )
    return bundle, funding, open_interest, taker_volume, long_short


def _run_download(
    *,
    downloaders: DerivativesDownloaders,
    options: DerivativesDownloadOptions,
    symbols: tuple[str, ...],
    funding_start: int = _START,
    funding_end: int = _END,
    futures_start: int = _START,
    futures_end: int = _END,
) -> DerivativesDownloadSummary:
    """Run derivatives download with shared test windows."""
    return _run(
        run_derivatives_download(
            downloaders=downloaders,
            options=options,
            symbols=symbols,
            funding_start_time=funding_start,
            funding_end_time=funding_end,
            futures_data_start_time=futures_start,
            futures_data_end_time=futures_end,
        )
    )


def test_derivatives_datasets_order() -> None:
    """Dataset labels follow the required download order."""
    assert DERIVATIVES_DATASETS == (
        DATASET_FUNDING,
        DATASET_OPEN_INTEREST,
        DATASET_TAKER_VOLUME,
        DATASET_GLOBAL_RATIO,
        DATASET_TOP_ACCOUNT_RATIO,
        DATASET_TOP_POSITION_RATIO,
    )
    assert DEFAULT_DERIVATIVES_PERIODS == ("1h", "4h", "1d")


def test_build_parser_defaults() -> None:
    """Omitted optional flags keep discovery defaults."""
    args = build_parser().parse_args([])
    assert args.history_days is None
    assert args.start_time is None
    assert args.end_time is None
    assert args.storage_root is None
    assert args.symbols is None
    assert args.periods is None
    assert args.max_symbols is None
    assert args.workers == DEFAULT_DERIVATIVES_WORKER_COUNT
    assert args.testnet is False
    assert args.verbose is False


def test_build_parser_accepts_all_flags() -> None:
    """Parser accepts every documented derivatives download flag."""
    args = build_parser().parse_args(
        [
            "--history-days",
            "30",
            "--end-time",
            "1700000100000",
            "--storage-root",
            "tmp-derivatives",
            "--symbols",
            "BTCUSDT",
            "ETHUSDT",
            "--periods",
            "1h",
            "4h",
            "--max-symbols",
            "5",
            "--workers",
            "2",
            "--testnet",
            "--verbose",
        ]
    )
    assert args.history_days == 30
    assert args.end_time == 1_700_000_100_000
    assert args.storage_root == Path("tmp-derivatives")
    assert args.symbols == ["BTCUSDT", "ETHUSDT"]
    assert args.periods == ["1h", "4h"]
    assert args.max_symbols == 5
    assert args.workers == 2
    assert args.testnet is True
    assert args.verbose is True


def test_parser_rejects_history_days_with_start_time() -> None:
    """--history-days and --start-time are mutually exclusive."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--history-days", "7", "--start-time", "1000"])


def test_build_options_uses_dataset_specific_download_defaults() -> None:
    """Omitted time range keeps history_days unset; DownloadConfig supplies defaults."""
    download = DownloadConfig(
        ohlcv_history_days=100,
        funding_history_days=200,
        futures_data_history_days=15,
    )
    options = build_options(build_parser().parse_args([]), download=download)
    assert options.history_days is None
    assert options.download == download
    assert options.periods == DEFAULT_DERIVATIVES_PERIODS
    assert options.symbols == ()
    assert options.storage_root == Path(StorageConfig().root)
    assert options.workers == DEFAULT_DERIVATIVES_WORKER_COUNT
    assert options.app == AppConfig()


def test_build_options_explicit_history_override() -> None:
    """--history-days overrides dataset defaults before retention clamping."""
    options = build_options(
        build_parser().parse_args(["--history-days", "12"]),
        download=DownloadConfig(funding_history_days=3650, futures_data_history_days=30),
    )
    assert options.history_days == 12


def test_build_options_maps_symbols_and_periods() -> None:
    """--symbols and --periods map onto immutable option tuples."""
    options = build_options(
        build_parser().parse_args(
            [
                "--start-time",
                "1700000000000",
                "--symbols",
                "BTCUSDT",
                "ETHUSDT",
                "--periods",
                "1h",
                "1d",
                "--workers",
                "8",
            ]
        )
    )
    assert options.symbols == ("BTCUSDT", "ETHUSDT")
    assert options.periods == ("1h", "1d")
    assert options.workers == 8
    assert options.history_days is None


def test_build_options_uses_exchange_config_testnet() -> None:
    """ExchangeConfig.testnet enables testnet when CLI flag is omitted."""
    options = build_options(
        build_parser().parse_args(["--history-days", "1"]),
        exchange=ExchangeConfig(testnet=True),
    )
    assert options.testnet is True


def test_build_options_rejects_non_positive_workers() -> None:
    """Non-positive --workers fails validation."""
    args = build_parser().parse_args(["--workers", "0", "--history-days", "1"])
    with pytest.raises(ValidationError, match="workers must be greater than 0"):
        build_options(args)


def test_build_options_rejects_unsupported_period() -> None:
    """Unsupported --periods values fail validation."""
    args = build_parser().parse_args(["--periods", "3m", "--history-days", "1"])
    with pytest.raises(ValidationError, match="unsupported derivatives period"):
        build_options(args)


def test_build_options_rejects_empty_symbol() -> None:
    """Blank --symbols entries fail validation."""
    args = build_parser().parse_args(["--symbols", " ", "--history-days", "1"])
    with pytest.raises(ValidationError, match="symbols entries must be non-empty"):
        build_options(args)


def test_build_options_rejects_invalid_time_window() -> None:
    """end_time <= start_time fails validation before download."""
    args = build_parser().parse_args(
        ["--start-time", "2000", "--end-time", "1000"],
    )
    with pytest.raises(ValidationError, match="end_time must be greater than start_time"):
        build_options(args)


def test_validate_download_window_rejects_equal_bounds() -> None:
    """Equal inclusive bounds are rejected as impossible windows."""
    with pytest.raises(ValidationError, match="end_time must be greater than start_time"):
        validate_download_window(start_time=1000, end_time=1000)


def test_resolve_dataset_download_range_uses_family_defaults() -> None:
    """Funding and Futures Data families apply distinct configured defaults."""
    end_time = 2_000_000_000_000
    funding = resolve_dataset_download_range(
        start_time=None,
        end_time=end_time,
        history_days_override=None,
        default_history_days=3650,
    )
    futures = resolve_dataset_download_range(
        start_time=None,
        end_time=end_time,
        history_days_override=None,
        default_history_days=30,
        max_history_days=30,
    )
    assert funding.start_time == end_time - (3650 * MILLISECONDS_PER_DAY)
    assert futures.start_time == end_time - (30 * MILLISECONDS_PER_DAY)
    assert funding.clamped is False
    assert futures.clamped is False


def test_futures_data_safety_margin_produces_29_day_window() -> None:
    """Default 30-day retention minus 1-day margin yields a 29-day request window."""
    download = DownloadConfig(
        futures_data_history_days=30,
        futures_data_safety_margin_days=1,
    )
    effective_history = effective_futures_data_history_days(download)
    assert effective_history == 29

    end_time = 2_000_000_000_000
    resolved = resolve_dataset_download_range(
        start_time=None,
        end_time=end_time,
        history_days_override=None,
        default_history_days=effective_history,
        max_history_days=effective_history,
    )
    assert resolved.start_time == end_time - (29 * MILLISECONDS_PER_DAY)
    assert resolved.end_time == end_time
    assert (resolved.end_time - resolved.start_time) / MILLISECONDS_PER_DAY == 29.0


def test_effective_futures_data_history_days_rejects_non_positive() -> None:
    """Zero or negative effective Futures Data history raises ValidationError."""
    with pytest.raises(ValidationError, match="must be greater than 0"):
        effective_futures_data_history_days(
            DownloadConfig(
                futures_data_history_days=30,
                futures_data_safety_margin_days=30,
            )
        )


def test_resolve_dataset_download_range_clamps_futures_data() -> None:
    """Requested Futures Data history is clamped to configured retention."""
    end_time = 2_000_000_000_000
    resolved = resolve_dataset_download_range(
        start_time=None,
        end_time=end_time,
        history_days_override=3650,
        default_history_days=30,
        max_history_days=30,
    )
    assert resolved.start_time == end_time - (30 * MILLISECONDS_PER_DAY)
    assert resolved.end_time == end_time
    assert resolved.clamped is True
    assert resolved.history_days_applied == 3650


def test_resolve_dataset_download_range_honors_explicit_override_within_limit() -> None:
    """Explicit history shorter than retention is not clamped."""
    end_time = 2_000_000_000_000
    resolved = resolve_dataset_download_range(
        start_time=None,
        end_time=end_time,
        history_days_override=7,
        default_history_days=30,
        max_history_days=30,
    )
    assert resolved.start_time == end_time - (7 * MILLISECONDS_PER_DAY)
    assert resolved.clamped is False


def test_build_downloaders_wires_existing_classes() -> None:
    """Composition root constructs the four existing downloader types."""
    client = MagicMock(spec=BinanceClient)
    repository = MagicMock(spec=MarketDataRepository)

    with (
        patch(
            "cqros.cli.download_derivatives.FundingDownloader",
            wraps=FundingDownloader,
        ) as funding_cls,
        patch(
            "cqros.cli.download_derivatives.OpenInterestDownloader",
            wraps=OpenInterestDownloader,
        ) as oi_cls,
        patch(
            "cqros.cli.download_derivatives.TakerVolumeDownloader",
            wraps=TakerVolumeDownloader,
        ) as taker_cls,
        patch(
            "cqros.cli.download_derivatives.LongShortDownloader",
            wraps=LongShortDownloader,
        ) as ls_cls,
        patch("cqros.cli.download_derivatives.FundingDownloadPlanner") as funding_planner,
        patch("cqros.cli.download_derivatives.OpenInterestDownloadPlanner") as oi_planner,
        patch("cqros.cli.download_derivatives.TakerVolumeDownloadPlanner") as taker_planner,
        patch("cqros.cli.download_derivatives.LongShortDownloadPlanner") as ls_planner,
    ):
        funding_cls.return_value = MagicMock(spec=FundingDownloader)
        oi_cls.return_value = MagicMock(spec=OpenInterestDownloader)
        taker_cls.return_value = MagicMock(spec=TakerVolumeDownloader)
        ls_cls.return_value = MagicMock(spec=LongShortDownloader)
        downloaders = build_downloaders(client=client, repository=repository)

    funding_cls.assert_called_once()
    oi_cls.assert_called_once()
    taker_cls.assert_called_once()
    ls_cls.assert_called_once()
    funding_planner.assert_called_once_with()
    oi_planner.assert_called_once_with()
    taker_planner.assert_called_once_with()
    ls_planner.assert_called_once_with()
    assert downloaders.funding is funding_cls.return_value
    assert downloaders.open_interest is oi_cls.return_value
    assert downloaders.taker_volume is taker_cls.return_value
    assert downloaders.long_short is ls_cls.return_value


def test_format_summary_contains_required_fields() -> None:
    """Summary report includes every required derivatives summary line."""
    summary = DerivativesDownloadSummary(
        symbols_discovered=10,
        symbols_processed=9,
        funding_tasks=9,
        open_interest_tasks=27,
        taker_volume_tasks=27,
        global_ratio_tasks=27,
        top_account_ratio_tasks=27,
        top_position_ratio_tasks=27,
        successful_tasks=140,
        failed_tasks=4,
        rows_downloaded=12345,
        duration_seconds=12.345,
        output_directory=Path("data"),
    )
    text = format_summary(summary)
    assert "Final Summary" in text
    assert "CQROS Derivatives Download Summary" in text
    assert "Symbols discovered: 10" in text
    assert "Symbols processed: 9" in text
    assert "Funding tasks: 9" in text
    assert "Open Interest tasks: 27" in text
    assert "Taker Volume tasks: 27" in text
    assert "Global Ratio tasks: 27" in text
    assert "Top Account Ratio tasks: 27" in text
    assert "Top Position Ratio tasks: 27" in text
    assert "Successful tasks: 140" in text
    assert "Failed tasks: 4" in text
    assert "Rows downloaded: 12345" in text
    assert "Elapsed time: 12.345s" in text
    assert "Output directory: data" in text


def test_run_derivatives_download_empty_universe(tmp_path: Path) -> None:
    """Empty discovery produces a zeroed summary without downloader calls."""
    bundle, funding, open_interest, taker_volume, long_short = _downloaders()
    summary = _run_download(
        downloaders=bundle,
        options=_options(storage_root=tmp_path, workers=2),
        symbols=(),
    )
    assert summary.symbols_discovered == 0
    assert summary.symbols_processed == 0
    assert summary.successful_tasks == 0
    assert summary.failed_tasks == 0
    funding.download_symbol.assert_not_awaited()
    open_interest.download_symbol.assert_not_awaited()
    taker_volume.download_symbol.assert_not_awaited()
    long_short.download_symbol.assert_not_awaited()


def test_run_derivatives_download_phases_and_default_periods(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Datasets run in phases using the default research periods."""
    bundle, funding, open_interest, taker_volume, long_short = _downloaders()
    phase_order: list[str] = []

    async def funding_side_effect(**_kwargs: Any) -> DownloadResult:
        phase_order.append("funding")
        return DownloadResult(status=DownloadStatus.FULL, rows_downloaded=1)

    async def oi_side_effect(**kwargs: Any) -> DownloadResult:
        phase_order.append(f"oi:{kwargs['period']}")
        return DownloadResult(status=DownloadStatus.FULL, rows_downloaded=1)

    async def taker_side_effect(**kwargs: Any) -> DownloadResult:
        phase_order.append(f"taker:{kwargs['period']}")
        return DownloadResult(status=DownloadStatus.FULL, rows_downloaded=1)

    async def ls_side_effect(**kwargs: Any) -> DownloadResult:
        phase_order.append(f"ls:{kwargs['kind'].value}:{kwargs['period']}")
        return DownloadResult(status=DownloadStatus.FULL, rows_downloaded=1)

    funding.download_symbol = AsyncMock(side_effect=funding_side_effect)
    open_interest.download_symbol = AsyncMock(side_effect=oi_side_effect)
    taker_volume.download_symbol = AsyncMock(side_effect=taker_side_effect)
    long_short.download_symbol = AsyncMock(side_effect=ls_side_effect)

    summary = _run_download(
        downloaders=bundle,
        options=_options(storage_root=tmp_path, workers=1),
        symbols=("BTCUSDT",),
        funding_start=_START,
        funding_end=_END,
        futures_start=_START + 1_000,
        futures_end=_END,
    )

    assert phase_order[0] == "funding"
    assert phase_order[1:4] == ["oi:1h", "oi:4h", "oi:1d"]
    assert phase_order[4:7] == ["taker:1h", "taker:4h", "taker:1d"]
    assert phase_order[7:10] == [
        "ls:global_long_short_account_ratio:1h",
        "ls:global_long_short_account_ratio:4h",
        "ls:global_long_short_account_ratio:1d",
    ]

    funding.download_symbol.assert_awaited_once_with(
        symbol="BTCUSDT",
        start_time=_START,
        end_time=_END,
    )
    open_interest.download_symbol.assert_awaited()
    assert open_interest.download_symbol.await_args.kwargs["start_time"] == _START + 1_000
    assert open_interest.download_symbol.await_args.kwargs["end_time"] == _END
    assert open_interest.download_symbol.await_count == 3
    assert taker_volume.download_symbol.await_count == 3
    assert long_short.download_symbol.await_count == 9

    kinds = [c.kwargs["kind"] for c in long_short.download_symbol.await_args_list]
    assert kinds.count(LongShortRatioKind.GLOBAL_ACCOUNT) == 3
    assert kinds.count(LongShortRatioKind.TOP_TRADER_ACCOUNT) == 3
    assert kinds.count(LongShortRatioKind.TOP_TRADER_POSITION) == 3

    assert summary.funding_tasks == 1
    assert summary.open_interest_tasks == 3
    assert summary.taker_volume_tasks == 3
    assert summary.global_ratio_tasks == 3
    assert summary.top_account_ratio_tasks == 3
    assert summary.top_position_ratio_tasks == 3
    assert summary.successful_tasks == 16
    assert summary.failed_tasks == 0

    captured = capsys.readouterr().out
    assert "========================================\nFunding\n" in captured
    assert "Funding complete." in captured
    assert "Open Interest complete." in captured
    assert "Final Summary" not in captured


def test_run_respects_period_override(tmp_path: Path) -> None:
    """--periods override limits every non-funding dataset."""
    bundle, _, open_interest, taker_volume, long_short = _downloaders()
    summary = _run_download(
        downloaders=bundle,
        options=_options(storage_root=tmp_path, workers=1, periods=("1h",)),
        symbols=("BTCUSDT", "ETHUSDT"),
    )
    assert open_interest.download_symbol.await_count == 2
    assert taker_volume.download_symbol.await_count == 2
    assert long_short.download_symbol.await_count == 6
    assert summary.open_interest_tasks == 2
    assert summary.taker_volume_tasks == 2
    assert summary.global_ratio_tasks == 2


def test_run_continues_after_dataset_and_symbol_failures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Dataset and symbol failures are counted without aborting remaining work."""
    bundle, funding, open_interest, taker_volume, long_short = _downloaders()

    async def funding_side_effect(*, symbol: str, **_kwargs: Any) -> DownloadResult:
        if symbol == "BTCUSDT":
            raise ExchangeValidationError(
                "startTime is outside retention",
                error_code="INGESTION-CLIENT-VALIDATION",
                details={
                    "status_code": 400,
                    "binance_code": -1130,
                },
            )
        return DownloadResult(status=DownloadStatus.FULL, rows_downloaded=1)

    async def oi_side_effect(*, symbol: str, period: str, **_kwargs: Any) -> DownloadResult:
        if symbol == "ETHUSDT" and period == "1h":
            raise RuntimeError("oi failed")
        return DownloadResult(status=DownloadStatus.FULL, rows_downloaded=1)

    funding.download_symbol = AsyncMock(side_effect=funding_side_effect)
    open_interest.download_symbol = AsyncMock(side_effect=oi_side_effect)

    summary = _run_download(
        downloaders=bundle,
        options=_options(storage_root=tmp_path, workers=2, periods=("1h", "4h")),
        symbols=("BTCUSDT", "ETHUSDT"),
    )

    assert funding.download_symbol.await_count == 2
    assert open_interest.download_symbol.await_count == 4
    assert taker_volume.download_symbol.await_count == 4
    assert long_short.download_symbol.await_count == 12
    assert summary.symbols_processed == 2
    assert summary.failed_tasks == 2
    assert summary.successful_tasks == (
        summary.funding_tasks
        + summary.open_interest_tasks
        + summary.taker_volume_tasks
        + summary.global_ratio_tasks
        + summary.top_account_ratio_tasks
        + summary.top_position_ratio_tasks
        - summary.failed_tasks
    )

    captured = capsys.readouterr().out
    assert "FAILED" in captured
    assert "Dataset: Funding" in captured
    assert "Symbol: BTCUSDT" in captured
    assert "HTTP status: 400" in captured
    assert "Binance error code: -1130" in captured
    assert "Message:" in captured


def test_run_logs_stack_trace_in_debug_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    """Debug mode includes stack traces for failed tasks."""
    bundle, funding, _, _, _ = _downloaders()
    funding.download_symbol = AsyncMock(side_effect=RuntimeError("boom"))

    with caplog.at_level(logging.WARNING):
        _run_download(
            downloaders=bundle,
            options=_options(
                storage_root=tmp_path,
                workers=1,
                periods=("1h",),
                debug=True,
            ),
            symbols=("BTCUSDT",),
        )

    captured = capsys.readouterr().out
    assert "Stack trace:" in captured
    assert "RuntimeError: boom" in captured


def test_run_respects_worker_count_bound(tmp_path: Path) -> None:
    """At most worker_count symbols run concurrently within a dataset phase."""
    bundle, funding, open_interest, taker_volume, long_short = _downloaders()
    active = 0
    max_active = 0
    lock = asyncio.Lock()

    async def funding_side_effect(**_kwargs: Any) -> DownloadResult:
        nonlocal active, max_active
        async with lock:
            active += 1
            max_active = max(max_active, active)
        await asyncio.sleep(0.05)
        async with lock:
            active -= 1
        return DownloadResult(status=DownloadStatus.FULL, rows_downloaded=1)

    funding.download_symbol = AsyncMock(side_effect=funding_side_effect)

    _run_download(
        downloaders=bundle,
        options=_options(storage_root=tmp_path, workers=2, periods=("1h",)),
        symbols=("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT"),
    )

    assert max_active <= 2
    assert funding.download_symbol.await_count == 4
    assert open_interest.download_symbol.await_count == 4
    assert taker_volume.download_symbol.await_count == 4
    assert long_short.download_symbol.await_count == 12


def test_main_wires_discovery_and_dataset_specific_ranges(tmp_path: Path) -> None:
    """main discovers symbols and resolves funding vs Futures Data windows."""
    client = MagicMock(spec=BinanceClient)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    historical = MagicMock(spec=HistoricalBootstrap)
    historical.discover_symbols = AsyncMock(
        return_value=(_contract("BTCUSDT"), _contract("ETHUSDT"))
    )
    bundle, _, _, _, _ = _downloaders()
    summary = DerivativesDownloadSummary(
        symbols_discovered=2,
        symbols_processed=2,
        funding_tasks=2,
        open_interest_tasks=6,
        taker_volume_tasks=6,
        global_ratio_tasks=6,
        top_account_ratio_tasks=6,
        top_position_ratio_tasks=6,
        successful_tasks=32,
        failed_tasks=0,
        rows_downloaded=0,
        duration_seconds=0.001,
        output_directory=tmp_path,
    )
    download = DownloadConfig(
        funding_history_days=100,
        futures_data_history_days=30,
    )
    end_time = 2_000_000_000_000

    with (
        patch(
            "cqros.cli.download_derivatives.BinanceClient",
            return_value=client,
        ) as client_cls,
        patch(
            "cqros.cli.download_derivatives.HistoricalBootstrap",
            return_value=historical,
        ) as bootstrap_cls,
        patch(
            "cqros.cli.download_derivatives.build_downloaders",
            return_value=bundle,
        ) as build_dl,
        patch(
            "cqros.cli.download_derivatives.run_derivatives_download",
            new=AsyncMock(return_value=summary),
        ) as run_dl,
        patch("cqros.cli.download_derivatives.print") as printed,
        patch(
            "cqros.cli.download_derivatives.build_options",
            wraps=build_options,
        ) as build_opts,
    ):

        def _build_options_with_download(args: Any, **kwargs: Any) -> DerivativesDownloadOptions:
            kwargs.setdefault("download", download)
            return build_options(args, **kwargs)

        build_opts.side_effect = _build_options_with_download
        exit_code = _run(
            main(
                [
                    "--end-time",
                    str(end_time),
                    "--storage-root",
                    str(tmp_path),
                    "--symbols",
                    "BTCUSDT",
                    "ETHUSDT",
                    "--periods",
                    "1h",
                    "4h",
                    "--workers",
                    "1",
                    "--testnet",
                ]
            )
        )

    assert exit_code == 0
    client_cls.assert_called_once()
    bootstrap_cls.assert_called_once()
    options_arg = bootstrap_cls.call_args.args[0]
    assert isinstance(options_arg, BootstrapOptions)
    assert options_arg.history_days is None
    assert options_arg.symbols == ("BTCUSDT", "ETHUSDT")
    assert options_arg.testnet is True
    historical.discover_symbols.assert_awaited_once_with()
    build_dl.assert_called_once()
    run_dl.assert_awaited_once()
    assert run_dl.await_args is not None
    run_kwargs = run_dl.await_args.kwargs
    assert run_kwargs["symbols"] == ("BTCUSDT", "ETHUSDT")
    assert run_kwargs["options"].periods == ("1h", "4h")
    assert run_kwargs["downloaders"] is bundle
    assert run_kwargs["funding_start_time"] == end_time - (100 * MILLISECONDS_PER_DAY)
    assert run_kwargs["funding_end_time"] == end_time
    assert run_kwargs["futures_data_start_time"] == end_time - (29 * MILLISECONDS_PER_DAY)
    assert run_kwargs["futures_data_end_time"] == end_time
    printed.assert_called()
    assert printed.call_args is not None
    summary_text = printed.call_args.args[0]
    assert "CQROS Derivatives Download Summary" in summary_text


def test_main_clamps_explicit_history_for_futures_data(tmp_path: Path) -> None:
    """Explicit --history-days is clamped for Futures Data before download."""
    client = MagicMock(spec=BinanceClient)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    historical = MagicMock(spec=HistoricalBootstrap)
    historical.discover_symbols = AsyncMock(return_value=(_contract("BTCUSDT"),))
    bundle, _, _, _, _ = _downloaders()
    summary = DerivativesDownloadSummary(
        symbols_discovered=1,
        symbols_processed=1,
        funding_tasks=1,
        open_interest_tasks=1,
        taker_volume_tasks=1,
        global_ratio_tasks=1,
        top_account_ratio_tasks=1,
        top_position_ratio_tasks=1,
        successful_tasks=5,
        failed_tasks=0,
        rows_downloaded=0,
        duration_seconds=0.001,
        output_directory=tmp_path,
    )
    download = DownloadConfig(funding_history_days=3650, futures_data_history_days=30)
    end_time = 2_000_000_000_000

    with (
        patch("cqros.cli.download_derivatives.BinanceClient", return_value=client),
        patch(
            "cqros.cli.download_derivatives.HistoricalBootstrap",
            return_value=historical,
        ),
        patch("cqros.cli.download_derivatives.build_downloaders", return_value=bundle),
        patch(
            "cqros.cli.download_derivatives.run_derivatives_download",
            new=AsyncMock(return_value=summary),
        ) as run_dl,
        patch("cqros.cli.download_derivatives.print"),
        patch(
            "cqros.cli.download_derivatives.build_options",
            wraps=build_options,
        ) as build_opts,
    ):

        def _build_options_with_download(args: Any, **kwargs: Any) -> DerivativesDownloadOptions:
            kwargs.setdefault("download", download)
            return build_options(args, **kwargs)

        build_opts.side_effect = _build_options_with_download
        exit_code = _run(
            main(
                [
                    "--history-days",
                    "3650",
                    "--end-time",
                    str(end_time),
                    "--storage-root",
                    str(tmp_path),
                    "--symbols",
                    "BTCUSDT",
                    "--periods",
                    "1h",
                    "--workers",
                    "1",
                ]
            )
        )

    assert exit_code == 0
    assert run_dl.await_args is not None
    run_kwargs = run_dl.await_args.kwargs
    assert run_kwargs["funding_start_time"] == end_time - (3650 * MILLISECONDS_PER_DAY)
    assert run_kwargs["futures_data_start_time"] == end_time - (29 * MILLISECONDS_PER_DAY)


def test_main_returns_failure_on_cqros_error(tmp_path: Path) -> None:
    """CQROSError during discovery maps to exit code 1."""
    client = MagicMock(spec=BinanceClient)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    historical = MagicMock(spec=HistoricalBootstrap)
    historical.discover_symbols = AsyncMock(
        side_effect=ValidationError(
            "Requested bootstrap symbols were not discovered",
            error_code="BOOTSTRAP-HISTORICAL-014",
        )
    )

    with (
        patch("cqros.cli.download_derivatives.BinanceClient", return_value=client),
        patch(
            "cqros.cli.download_derivatives.HistoricalBootstrap",
            return_value=historical,
        ),
    ):
        exit_code = _run(main(["--history-days", "1", "--storage-root", str(tmp_path)]))

    assert exit_code == 1


def test_resume_uses_clamped_futures_window(tmp_path: Path) -> None:
    """Clamped Futures Data windows are forwarded to period downloaders."""
    bundle, funding, open_interest, taker_volume, long_short = _downloaders()
    futures_start = _END - (30 * MILLISECONDS_PER_DAY)
    funding_start = _END - (3650 * MILLISECONDS_PER_DAY)

    _run_download(
        downloaders=bundle,
        options=_options(storage_root=tmp_path, workers=1, periods=("1h",)),
        symbols=("BTCUSDT",),
        funding_start=funding_start,
        funding_end=_END,
        futures_start=futures_start,
        futures_end=_END,
    )

    funding.download_symbol.assert_awaited_once_with(
        symbol="BTCUSDT",
        start_time=funding_start,
        end_time=_END,
    )
    assert open_interest.download_symbol.await_args.kwargs["start_time"] == futures_start
    assert taker_volume.download_symbol.await_args.kwargs["start_time"] == futures_start
    assert long_short.download_symbol.await_args.kwargs["start_time"] == futures_start
