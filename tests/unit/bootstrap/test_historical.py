"""Unit tests for CQROS historical market-data bootstrap."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cqros.bootstrap import BootstrapOptions, HistoricalBootstrap
from cqros.bootstrap.historical import (
    BootstrapOptions as BootstrapOptionsDirect,
)
from cqros.bootstrap.historical import (
    HistoricalBootstrap as HistoricalBootstrapDirect,
)
from cqros.core.constants import (
    DEFAULT_STORAGE_ROOT,
    DEFAULT_TIMEFRAMES,
    MILLISECONDS_PER_DAY,
)
from cqros.core.exceptions import ValidationError
from cqros.data.contracts import (
    Contract,
    ContractStatus,
    ContractType,
    PriceFilter,
    QuantityFilter,
)
from cqros.ingestion.client import (
    DEFAULT_BACKOFF_FACTOR_SECONDS,
    DEFAULT_BINANCE_FUTURES_REST_BASE_URL,
    DEFAULT_BINANCE_FUTURES_TESTNET_REST_BASE_URL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT_SECONDS,
    BinanceClient,
)
from cqros.ingestion.downloader import (
    DEFAULT_DOWNLOAD_BATCH_SIZE,
    DEFAULT_DOWNLOAD_WORKERS,
    HistoricalDownloader,
)
from cqros.ingestion.resume import DownloadResult, DownloadStatus
from cqros.storage.layout import StorageLayout
from cqros.storage.parquet import ParquetStore
from cqros.storage.repository import MarketDataRepository


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def _contract(symbol: str) -> Contract:
    """Build a minimal immutable contract for selection tests."""
    return Contract(
        symbol=symbol,
        exchange="binance",
        base_asset=symbol.removesuffix("USDT"),
        quote_asset="USDT",
        contract_type=ContractType.PERPETUAL,
        status=ContractStatus.TRADING,
        price_filter=PriceFilter(tick_size=0.1),
        quantity_filter=QuantityFilter(step_size=0.001, min_quantity=0.001),
    )


def test_exports_match_module_symbols() -> None:
    """Package exports match the historical module classes."""
    assert BootstrapOptions is BootstrapOptionsDirect
    assert HistoricalBootstrap is HistoricalBootstrapDirect


def test_bootstrap_options_defaults() -> None:
    """Default options match CQROS storage and client defaults."""
    options = BootstrapOptions()
    assert is_dataclass(options)
    assert options.storage_root == Path(DEFAULT_STORAGE_ROOT)
    assert options.timeframes == DEFAULT_TIMEFRAMES
    assert options.symbols == ()
    assert options.start_time is None
    assert options.end_time is None
    assert options.history_days is None
    assert options.max_symbols is None
    assert options.testnet is False
    assert options.base_url is None
    assert options.timeout == DEFAULT_TIMEOUT_SECONDS
    assert options.max_retries == DEFAULT_MAX_RETRIES
    assert options.backoff_factor == DEFAULT_BACKOFF_FACTOR_SECONDS
    assert options.workers == DEFAULT_DOWNLOAD_WORKERS
    assert options.batch_size == DEFAULT_DOWNLOAD_BATCH_SIZE


def test_bootstrap_options_normalizes_storage_root_string() -> None:
    """String storage roots are normalized to pathlib.Path."""
    options = BootstrapOptions(storage_root="custom-data")
    assert options.storage_root == Path("custom-data")


def test_bootstrap_options_is_immutable() -> None:
    """BootstrapOptions instances reject attribute assignment."""
    options = BootstrapOptions()
    with pytest.raises(FrozenInstanceError):
        options.testnet = True  # type: ignore[misc]


def test_bootstrap_options_rejects_invalid_values() -> None:
    """Invalid option values raise ValidationError with stable codes."""
    with pytest.raises(ValidationError, match="storage_root") as storage_error:
        BootstrapOptions(storage_root="")
    assert storage_error.value.error_code == "BOOTSTRAP-HISTORICAL-001"

    with pytest.raises(ValidationError, match="timeframes") as empty_tf:
        BootstrapOptions(timeframes=())
    assert empty_tf.value.error_code == "BOOTSTRAP-HISTORICAL-002"

    with pytest.raises(ValidationError, match="unsupported") as bad_tf:
        BootstrapOptions(timeframes=("1m", "2x"))
    assert bad_tf.value.error_code == "BOOTSTRAP-HISTORICAL-003"

    with pytest.raises(ValidationError, match="symbols") as bad_symbol:
        BootstrapOptions(symbols=("BTCUSDT", "  "))
    assert bad_symbol.value.error_code == "BOOTSTRAP-HISTORICAL-004"

    with pytest.raises(ValidationError, match="max_symbols") as max_symbols:
        BootstrapOptions(max_symbols=0)
    assert max_symbols.value.error_code == "BOOTSTRAP-HISTORICAL-005"

    with pytest.raises(ValidationError, match="history_days") as history_days:
        BootstrapOptions(history_days=-1)
    assert history_days.value.error_code == "BOOTSTRAP-HISTORICAL-006"

    with pytest.raises(ValidationError, match="start_time") as start_time:
        BootstrapOptions(start_time=-1)
    assert start_time.value.error_code == "BOOTSTRAP-HISTORICAL-007"

    with pytest.raises(ValidationError, match="end_time") as end_time:
        BootstrapOptions(end_time=-5)
    assert end_time.value.error_code == "BOOTSTRAP-HISTORICAL-008"

    with pytest.raises(ValidationError, match="start_time") as time_range:
        BootstrapOptions(start_time=200, end_time=100)
    assert time_range.value.error_code == "BOOTSTRAP-HISTORICAL-009"

    with pytest.raises(ValidationError, match="timeout") as timeout:
        BootstrapOptions(timeout=0)
    assert timeout.value.error_code == "BOOTSTRAP-HISTORICAL-010"

    with pytest.raises(ValidationError, match="max_retries") as retries:
        BootstrapOptions(max_retries=-1)
    assert retries.value.error_code == "BOOTSTRAP-HISTORICAL-011"

    with pytest.raises(ValidationError, match="backoff_factor") as backoff:
        BootstrapOptions(backoff_factor=-0.1)
    assert backoff.value.error_code == "BOOTSTRAP-HISTORICAL-012"

    with pytest.raises(ValidationError, match="base_url") as base_url:
        BootstrapOptions(base_url="   ")
    assert base_url.value.error_code == "BOOTSTRAP-HISTORICAL-013"

    with pytest.raises(ValidationError, match="workers") as workers:
        BootstrapOptions(workers=0)
    assert workers.value.error_code == "BOOTSTRAP-HISTORICAL-017"

    with pytest.raises(ValidationError, match="batch_size") as batch_size:
        BootstrapOptions(batch_size=0)
    assert batch_size.value.error_code == "BOOTSTRAP-HISTORICAL-018"


def test_create_client_uses_production_defaults() -> None:
    """Client factory targets production Binance Futures by default."""
    bootstrap = HistoricalBootstrap(BootstrapOptions())
    client = bootstrap._create_client()  # pyright: ignore[reportPrivateUsage]
    assert isinstance(client, BinanceClient)
    assert client.base_url == DEFAULT_BINANCE_FUTURES_REST_BASE_URL
    assert client.timeout == DEFAULT_TIMEOUT_SECONDS
    assert client.max_retries == DEFAULT_MAX_RETRIES
    assert client.backoff_factor == DEFAULT_BACKOFF_FACTOR_SECONDS


def test_create_client_uses_testnet_and_custom_settings() -> None:
    """Client factory honors testnet flag and transport overrides."""
    bootstrap = HistoricalBootstrap(
        BootstrapOptions(
            testnet=True,
            timeout=12.5,
            max_retries=7,
            backoff_factor=1.25,
        )
    )
    client = bootstrap._create_client()  # pyright: ignore[reportPrivateUsage]
    assert client.base_url == DEFAULT_BINANCE_FUTURES_TESTNET_REST_BASE_URL
    assert client.timeout == 12.5
    assert client.max_retries == 7
    assert client.backoff_factor == 1.25


def test_create_client_prefers_explicit_base_url() -> None:
    """Explicit base_url overrides testnet URL selection."""
    bootstrap = HistoricalBootstrap(
        BootstrapOptions(
            testnet=True,
            base_url="https://example.test/fapi/",
        )
    )
    client = bootstrap._create_client()  # pyright: ignore[reportPrivateUsage]
    assert client.base_url == "https://example.test/fapi"


def test_discover_symbols_returns_discovered_contracts() -> None:
    """Discovery returns contracts from SymbolDiscovery unchanged by default."""
    contracts = (_contract("BTCUSDT"), _contract("ETHUSDT"))
    client = MagicMock(spec=BinanceClient)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)

    bootstrap = HistoricalBootstrap(BootstrapOptions())
    with (
        patch.object(HistoricalBootstrap, "_create_client", return_value=client),
        patch(
            "cqros.bootstrap.historical.SymbolDiscovery.discover",
            new=AsyncMock(return_value=contracts),
        ),
    ):
        result = _run(bootstrap.discover_symbols())

    assert result == contracts


def test_discover_symbols_applies_allowlist_and_max_symbols() -> None:
    """Allowlist ordering is preserved and max_symbols truncates the result."""
    contracts = (
        _contract("BTCUSDT"),
        _contract("ETHUSDT"),
        _contract("SOLUSDT"),
    )
    client = MagicMock(spec=BinanceClient)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)

    bootstrap = HistoricalBootstrap(
        BootstrapOptions(
            symbols=("ETHUSDT", "BTCUSDT", "SOLUSDT"),
            max_symbols=2,
        )
    )
    with (
        patch.object(HistoricalBootstrap, "_create_client", return_value=client),
        patch(
            "cqros.bootstrap.historical.SymbolDiscovery.discover",
            new=AsyncMock(return_value=contracts),
        ),
    ):
        result = _run(bootstrap.discover_symbols())

    assert tuple(contract.symbol for contract in result) == ("ETHUSDT", "BTCUSDT")


def test_discover_symbols_rejects_missing_allowlist_symbols() -> None:
    """Explicit allowlist symbols that are not discovered raise ValidationError."""
    contracts = (_contract("BTCUSDT"),)
    client = MagicMock(spec=BinanceClient)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)

    bootstrap = HistoricalBootstrap(BootstrapOptions(symbols=("BTCUSDT", "ETHUSDT")))
    with (
        patch.object(HistoricalBootstrap, "_create_client", return_value=client),
        patch(
            "cqros.bootstrap.historical.SymbolDiscovery.discover",
            new=AsyncMock(return_value=contracts),
        ),
        pytest.raises(ValidationError, match="not discovered") as missing,
    ):
        _run(bootstrap.discover_symbols())

    assert missing.value.error_code == "BOOTSTRAP-HISTORICAL-014"
    assert missing.value.details["missing_symbols"] == ("ETHUSDT",)


def test_options_property_returns_configured_options() -> None:
    """HistoricalBootstrap exposes the injected options immutably."""
    options = BootstrapOptions(max_symbols=10, testnet=True)
    bootstrap = HistoricalBootstrap(options)
    assert bootstrap.options is options


def test_build_downloader_composes_existing_services(tmp_path: Path) -> None:
    """Downloader factory wires repository, planner, and HistoricalDownloader."""
    client = MagicMock(spec=BinanceClient)
    bootstrap = HistoricalBootstrap(BootstrapOptions(storage_root=tmp_path))
    downloader = bootstrap._build_downloader(client)  # pyright: ignore[reportPrivateUsage]

    assert isinstance(downloader, HistoricalDownloader)
    assert downloader._client is client  # pyright: ignore[reportPrivateUsage]
    assert isinstance(
        downloader._repository,  # pyright: ignore[reportPrivateUsage]
        MarketDataRepository,
    )
    assert isinstance(
        downloader._repository._layout,  # pyright: ignore[reportPrivateUsage]
        StorageLayout,
    )
    assert downloader._repository._layout.root == tmp_path  # pyright: ignore[reportPrivateUsage]
    assert isinstance(
        downloader._repository._datastore,  # pyright: ignore[reportPrivateUsage]
        ParquetStore,
    )
    assert downloader.workers == DEFAULT_DOWNLOAD_WORKERS
    assert downloader.batch_size == DEFAULT_DOWNLOAD_BATCH_SIZE


def test_build_downloader_propagates_execution_options(tmp_path: Path) -> None:
    """Downloader factory forwards workers and batch_size from options."""
    client = MagicMock(spec=BinanceClient)
    bootstrap = HistoricalBootstrap(
        BootstrapOptions(storage_root=tmp_path, workers=3, batch_size=12)
    )
    downloader = bootstrap._build_downloader(client)  # pyright: ignore[reportPrivateUsage]

    assert downloader.workers == 3
    assert downloader.batch_size == 12


def test_resolve_download_range_uses_explicit_bounds() -> None:
    """Explicit start_time and end_time are returned unchanged."""
    bootstrap = HistoricalBootstrap(
        BootstrapOptions(start_time=1_000, end_time=2_000, history_days=30)
    )
    resolved = bootstrap._resolve_download_range()  # pyright: ignore[reportPrivateUsage]
    assert resolved == (1_000, 2_000)


def test_resolve_download_range_derives_start_from_history_days() -> None:
    """Missing start_time is derived from end_time and history_days."""
    end_time = 400 * MILLISECONDS_PER_DAY
    history_days = 3
    bootstrap = HistoricalBootstrap(BootstrapOptions(end_time=end_time, history_days=history_days))
    resolved = bootstrap._resolve_download_range()  # pyright: ignore[reportPrivateUsage]
    assert resolved == (end_time - (history_days * MILLISECONDS_PER_DAY), end_time)


def test_resolve_download_range_clamps_negative_start_to_zero() -> None:
    """Derived start timestamps earlier than Unix epoch are clamped to zero."""
    bootstrap = HistoricalBootstrap(BootstrapOptions(end_time=10_000_000, history_days=3))
    resolved = bootstrap._resolve_download_range()  # pyright: ignore[reportPrivateUsage]
    assert resolved == (0, 10_000_000)


def test_resolve_download_range_rejects_unresolved_start() -> None:
    """Missing start_time and history_days raise a stable ValidationError."""
    bootstrap = HistoricalBootstrap(BootstrapOptions(end_time=5_000))
    with pytest.raises(ValidationError, match="start_time or history_days") as exc_info:
        bootstrap._resolve_download_range()  # pyright: ignore[reportPrivateUsage]
    assert exc_info.value.error_code == "BOOTSTRAP-HISTORICAL-015"


def test_download_symbol_rejects_empty_symbol() -> None:
    """Empty download symbols fail before any client session is opened."""
    bootstrap = HistoricalBootstrap(BootstrapOptions(history_days=1))
    with pytest.raises(ValidationError, match="symbol") as exc_info:
        _run(bootstrap.download_symbol(symbol=""))
    assert exc_info.value.error_code == "BOOTSTRAP-HISTORICAL-016"


def test_download_symbol_delegates_to_historical_downloader() -> None:
    """Symbol download opens a client and delegates each configured timeframe."""
    client = MagicMock(spec=BinanceClient)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)

    downloader = MagicMock(spec=HistoricalDownloader)
    downloader.download_symbol = AsyncMock(
        return_value=DownloadResult(status=DownloadStatus.FULL, rows_downloaded=1)
    )

    bootstrap = HistoricalBootstrap(
        BootstrapOptions(
            timeframes=("1m", "5m"),
            start_time=1_000,
            end_time=2_000,
        )
    )
    with (
        patch.object(HistoricalBootstrap, "_create_client", return_value=client),
        patch.object(HistoricalBootstrap, "_build_downloader", return_value=downloader),
    ):
        _run(bootstrap.download_symbol(symbol="BTCUSDT"))

    assert downloader.download_symbol.await_count == 2
    downloader.download_symbol.assert_any_await(
        symbol="BTCUSDT",
        timeframe="1m",
        start_time=1_000,
        end_time=2_000,
    )
    downloader.download_symbol.assert_any_await(
        symbol="BTCUSDT",
        timeframe="5m",
        start_time=1_000,
        end_time=2_000,
    )
    client.__aenter__.assert_awaited_once()
    client.__aexit__.assert_awaited_once()


def test_run_discovers_and_downloads_each_selected_symbol() -> None:
    """run() discovers contracts then reuses download_symbol for each."""
    contracts = (_contract("BTCUSDT"), _contract("ETHUSDT"))
    bootstrap = HistoricalBootstrap(BootstrapOptions(history_days=1))
    download_symbol = AsyncMock(return_value=None)

    with (
        patch.object(
            HistoricalBootstrap,
            "discover_symbols",
            new=AsyncMock(return_value=contracts),
        ) as discover_symbols,
        patch.object(HistoricalBootstrap, "download_symbol", new=download_symbol),
    ):
        _run(bootstrap.run())

    discover_symbols.assert_awaited_once_with()
    assert download_symbol.await_count == 2
    download_symbol.assert_any_await(symbol="BTCUSDT")
    download_symbol.assert_any_await(symbol="ETHUSDT")


def test_run_completes_when_discovery_returns_empty() -> None:
    """run() is a no-op download loop when no contracts are selected."""
    bootstrap = HistoricalBootstrap(BootstrapOptions(history_days=1))
    download_symbol = AsyncMock(return_value=None)

    with (
        patch.object(
            HistoricalBootstrap,
            "discover_symbols",
            new=AsyncMock(return_value=()),
        ),
        patch.object(HistoricalBootstrap, "download_symbol", new=download_symbol),
    ):
        _run(bootstrap.run())

    download_symbol.assert_not_awaited()
