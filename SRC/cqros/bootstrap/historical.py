"""CQROS historical market-data bootstrap orchestration.

Purpose:
    Compose existing ingestion components into a reusable bootstrap entry
    point for seeding CQROS historical market data.

Responsibilities:
    - Represent immutable ``BootstrapOptions`` for bootstrap runs
    - Construct configured ``BinanceClient`` instances
    - Discover tradeable USDT perpetual contracts via ``SymbolDiscovery``
    - Apply optional symbol allowlists and size limits
    - Compose ``HistoricalDownloader`` for symbol historical seeding
    - Download and persist configured timeframes for a single symbol
    - Orchestrate full bootstrap runs across selected symbols

Dependencies:
    ``cqros.core``, ``cqros.data.contracts``, ``cqros.ingestion``, and
    ``cqros.storage``.

Public API:
    ``BootstrapOptions`` and ``HistoricalBootstrap``.

Notes:
    This module is a composition root for historical bootstrap. It wires
    existing CQROS services and never calls exchange endpoints, constructs
    storage paths, or validates market data itself. No CLI, argparse, or
    logging is provided.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from cqros.core.constants import (
    DEFAULT_STORAGE_ROOT,
    DEFAULT_TIMEFRAMES,
    MILLISECONDS_PER_DAY,
    SUPPORTED_TIMEFRAMES,
)
from cqros.core.exceptions import ValidationError
from cqros.core.types import FilePath, Symbol, Timeframe, UnixTimestampMs
from cqros.data.contracts import Contract
from cqros.ingestion.client import (
    DEFAULT_BACKOFF_FACTOR_SECONDS,
    DEFAULT_BINANCE_FUTURES_REST_BASE_URL,
    DEFAULT_BINANCE_FUTURES_TESTNET_REST_BASE_URL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT_SECONDS,
    BinanceClient,
)
from cqros.ingestion.discovery import SymbolDiscovery
from cqros.ingestion.downloader import (
    DEFAULT_DOWNLOAD_BATCH_SIZE,
    DEFAULT_DOWNLOAD_WORKERS,
    DownloadPlanner,
    HistoricalDownloader,
)
from cqros.storage.layout import StorageLayout
from cqros.storage.parquet import ParquetStore
from cqros.storage.repository import MarketDataRepository

__all__ = [
    "BootstrapOptions",
    "HistoricalBootstrap",
]

_ERROR_STORAGE_ROOT: Final[str] = "BOOTSTRAP-HISTORICAL-001"
_ERROR_TIMEFRAMES_EMPTY: Final[str] = "BOOTSTRAP-HISTORICAL-002"
_ERROR_TIMEFRAME_UNSUPPORTED: Final[str] = "BOOTSTRAP-HISTORICAL-003"
_ERROR_SYMBOL_EMPTY: Final[str] = "BOOTSTRAP-HISTORICAL-004"
_ERROR_MAX_SYMBOLS: Final[str] = "BOOTSTRAP-HISTORICAL-005"
_ERROR_HISTORY_DAYS: Final[str] = "BOOTSTRAP-HISTORICAL-006"
_ERROR_START_TIME: Final[str] = "BOOTSTRAP-HISTORICAL-007"
_ERROR_END_TIME: Final[str] = "BOOTSTRAP-HISTORICAL-008"
_ERROR_TIME_RANGE: Final[str] = "BOOTSTRAP-HISTORICAL-009"
_ERROR_TIMEOUT: Final[str] = "BOOTSTRAP-HISTORICAL-010"
_ERROR_MAX_RETRIES: Final[str] = "BOOTSTRAP-HISTORICAL-011"
_ERROR_BACKOFF: Final[str] = "BOOTSTRAP-HISTORICAL-012"
_ERROR_BASE_URL: Final[str] = "BOOTSTRAP-HISTORICAL-013"
_ERROR_MISSING_SYMBOLS: Final[str] = "BOOTSTRAP-HISTORICAL-014"
_ERROR_DOWNLOAD_RANGE: Final[str] = "BOOTSTRAP-HISTORICAL-015"
_ERROR_DOWNLOAD_SYMBOL: Final[str] = "BOOTSTRAP-HISTORICAL-016"
_ERROR_WORKERS: Final[str] = "BOOTSTRAP-HISTORICAL-017"
_ERROR_BATCH_SIZE: Final[str] = "BOOTSTRAP-HISTORICAL-018"


@dataclass(frozen=True, slots=True)
class BootstrapOptions:
    """Immutable configuration for a historical market-data bootstrap run.

    Attributes:
        storage_root: Root directory for CQROS market-data artifacts.
        timeframes: Candle intervals to bootstrap in later download stages.
        symbols: Optional allowlist of symbols. An empty tuple means discover
            the full supported universe.
        start_time: Inclusive historical range start as UTC Unix milliseconds.
        end_time: Inclusive historical range end as UTC Unix milliseconds.
        history_days: Optional lookback window in days used to derive
            ``start_time`` when it is omitted for download.
        max_symbols: Optional upper bound on discovered or selected symbols.
        testnet: Whether to target Binance Futures testnet endpoints.
        base_url: Optional explicit REST base URL. When set, overrides
            ``testnet`` URL selection.
        timeout: Per-request HTTP timeout in seconds.
        max_retries: Maximum retry attempts after the initial request.
        backoff_factor: Base delay in seconds for exponential backoff.
        workers: Maximum concurrent symbol workers for bootstrap execution.
        batch_size: Maximum batch size for bootstrap download execution.
    """

    storage_root: FilePath = field(default_factory=lambda: Path(DEFAULT_STORAGE_ROOT))
    timeframes: tuple[Timeframe, ...] = DEFAULT_TIMEFRAMES
    symbols: tuple[Symbol, ...] = ()
    start_time: UnixTimestampMs | None = None
    end_time: UnixTimestampMs | None = None
    history_days: int | None = None
    max_symbols: int | None = None
    testnet: bool = False
    base_url: str | None = None
    timeout: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR_SECONDS
    workers: int = DEFAULT_DOWNLOAD_WORKERS
    batch_size: int = DEFAULT_DOWNLOAD_BATCH_SIZE

    def __post_init__(self) -> None:
        """Normalize path inputs and validate option invariants.

        Raises:
            ValidationError: If any option value is invalid.
        """
        if isinstance(self.storage_root, str) and self.storage_root.strip() == "":
            raise ValidationError(
                "storage_root must be a non-empty path",
                error_code=_ERROR_STORAGE_ROOT,
                details={"parameter": "storage_root", "value": self.storage_root},
            )
        object.__setattr__(self, "storage_root", Path(self.storage_root))

        if not self.timeframes:
            raise ValidationError(
                "timeframes must contain at least one timeframe",
                error_code=_ERROR_TIMEFRAMES_EMPTY,
                details={"parameter": "timeframes"},
            )
        unsupported = tuple(
            timeframe for timeframe in self.timeframes if timeframe not in SUPPORTED_TIMEFRAMES
        )
        if unsupported:
            raise ValidationError(
                "timeframes contains unsupported values",
                error_code=_ERROR_TIMEFRAME_UNSUPPORTED,
                details={
                    "parameter": "timeframes",
                    "unsupported": unsupported,
                    "supported": tuple(sorted(SUPPORTED_TIMEFRAMES)),
                },
            )

        for index, symbol in enumerate(self.symbols):
            if symbol.strip() == "":
                raise ValidationError(
                    "symbols entries must be non-empty strings",
                    error_code=_ERROR_SYMBOL_EMPTY,
                    details={"parameter": "symbols", "index": index, "value": symbol},
                )

        if self.max_symbols is not None and self.max_symbols <= 0:
            raise ValidationError(
                "max_symbols must be greater than 0 when provided",
                error_code=_ERROR_MAX_SYMBOLS,
                details={"parameter": "max_symbols", "value": self.max_symbols},
            )

        if self.history_days is not None and self.history_days <= 0:
            raise ValidationError(
                "history_days must be greater than 0 when provided",
                error_code=_ERROR_HISTORY_DAYS,
                details={"parameter": "history_days", "value": self.history_days},
            )

        if self.start_time is not None:
            _require_unix_ms(self.start_time, parameter="start_time", error_code=_ERROR_START_TIME)
        if self.end_time is not None:
            _require_unix_ms(self.end_time, parameter="end_time", error_code=_ERROR_END_TIME)
        if (
            self.start_time is not None
            and self.end_time is not None
            and self.start_time > self.end_time
        ):
            raise ValidationError(
                "start_time must be less than or equal to end_time",
                error_code=_ERROR_TIME_RANGE,
                details={
                    "parameter": "start_time",
                    "start_time": self.start_time,
                    "end_time": self.end_time,
                },
            )

        if self.timeout <= 0:
            raise ValidationError(
                "timeout must be greater than 0",
                error_code=_ERROR_TIMEOUT,
                details={"parameter": "timeout", "value": self.timeout},
            )
        if self.max_retries < 0:
            raise ValidationError(
                "max_retries must be greater than or equal to 0",
                error_code=_ERROR_MAX_RETRIES,
                details={"parameter": "max_retries", "value": self.max_retries},
            )
        if self.backoff_factor < 0:
            raise ValidationError(
                "backoff_factor must be greater than or equal to 0",
                error_code=_ERROR_BACKOFF,
                details={"parameter": "backoff_factor", "value": self.backoff_factor},
            )

        if self.base_url is not None and self.base_url.strip() == "":
            raise ValidationError(
                "base_url must be a non-empty string when provided",
                error_code=_ERROR_BASE_URL,
                details={"parameter": "base_url"},
            )

        if self.workers < 1:
            raise ValidationError(
                "workers must be greater than or equal to 1",
                error_code=_ERROR_WORKERS,
                details={"parameter": "workers", "value": self.workers},
            )
        if self.batch_size < 1:
            raise ValidationError(
                "batch_size must be greater than or equal to 1",
                error_code=_ERROR_BATCH_SIZE,
                details={"parameter": "batch_size", "value": self.batch_size},
            )


class HistoricalBootstrap:
    """Orchestrate historical market-data bootstrap using CQROS components.

    Discovery selects tradeable contracts. Download composition wires
    ``HistoricalDownloader`` with repository and planner services so a single
    symbol can be seeded for every configured timeframe. ``run`` sequences
    discovery and per-symbol download without duplicating either path.

    Args:
        options: Immutable bootstrap configuration.
    """

    __slots__ = ("_options",)

    _options: BootstrapOptions

    def __init__(self, options: BootstrapOptions) -> None:
        """Initialize bootstrap orchestration with validated options.

        Args:
            options: Immutable bootstrap configuration.
        """
        self._options = options

    @property
    def options(self) -> BootstrapOptions:
        """Return the immutable bootstrap options."""
        return self._options

    def _create_client(self) -> BinanceClient:
        """Create a ``BinanceClient`` configured from bootstrap options.

        Returns:
            A new ``BinanceClient`` that has not yet opened an HTTP session.
        """
        base_url = self._resolve_base_url()
        return BinanceClient(
            base_url=base_url,
            timeout=self._options.timeout,
            max_retries=self._options.max_retries,
            backoff_factor=self._options.backoff_factor,
        )

    def _build_downloader(self, client: BinanceClient) -> HistoricalDownloader:
        """Compose a ``HistoricalDownloader`` from bootstrap storage options.

        Args:
            client: Open ``BinanceClient`` used for kline requests.

        Returns:
            A downloader wired to ``MarketDataRepository``, ``ParquetStore``,
            and ``DownloadPlanner``. Filesystem paths are owned by storage
            services and are never constructed here.
        """
        repository = MarketDataRepository(
            StorageLayout(self._options.storage_root),
            ParquetStore(),
        )
        return HistoricalDownloader(
            client,
            repository,
            DownloadPlanner(),
            workers=self._options.workers,
            batch_size=self._options.batch_size,
        )

    async def discover_symbols(self) -> tuple[Contract, ...]:
        """Discover tradeable contracts and apply bootstrap selection rules.

        Opens a short-lived Binance client session, discovers USDT perpetual
        contracts through ``SymbolDiscovery``, then applies the optional
        symbol allowlist and ``max_symbols`` limit from options.

        Returns:
            Immutable tuple of selected ``Contract`` instances.

        Raises:
            ValidationError: If an explicit symbol allowlist requests symbols
                that were not discovered.
            ExchangeError: Propagated from transport failures during discovery.
        """
        client = self._create_client()
        async with client:
            contracts = await SymbolDiscovery(client).discover()
        return self._select_contracts(contracts)

    async def download_symbol(
        self,
        *,
        symbol: Symbol,
        timeframe: Timeframe | None = None,
    ) -> None:
        """Download and persist historical OHLCV for one symbol.

        Resolves the download window from bootstrap options, opens a short-lived
        Binance client session, composes ``HistoricalDownloader``, and downloads
        either ``timeframe`` or every configured timeframe for ``symbol``.
        Exchange I/O and persistence are delegated entirely to
        ``HistoricalDownloader``.

        Args:
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Optional single bar interval to download. When omitted,
                every timeframe in bootstrap options is downloaded.

        Raises:
            ValidationError: If ``symbol`` is empty, ``timeframe`` is unsupported,
                or the download window cannot be resolved from options.
            ExchangeError: Propagated from transport failures during download.
        """
        if symbol.strip() == "":
            raise ValidationError(
                "symbol must be a non-empty string",
                error_code=_ERROR_DOWNLOAD_SYMBOL,
                details={"parameter": "symbol", "value": symbol},
            )

        timeframes: tuple[Timeframe, ...]
        if timeframe is None:
            timeframes = self._options.timeframes
        elif timeframe not in SUPPORTED_TIMEFRAMES:
            raise ValidationError(
                "timeframe is not supported",
                error_code=_ERROR_TIMEFRAME_UNSUPPORTED,
                details={
                    "parameter": "timeframe",
                    "value": timeframe,
                    "supported": tuple(sorted(SUPPORTED_TIMEFRAMES)),
                },
            )
        else:
            timeframes = (timeframe,)

        start_time, end_time = self._resolve_download_range()
        client = self._create_client()
        async with client:
            downloader = self._build_downloader(client)
            for selected in timeframes:
                print(f"{symbol} OHLCV {selected}", flush=True)
                result = await downloader.download_symbol(
                    symbol=symbol,
                    timeframe=selected,
                    start_time=start_time,
                    end_time=end_time,
                )
                print(result.format_progress(), flush=True)

    async def run(self) -> None:
        """Discover selected contracts and download historical data for each.

        Calls ``discover_symbols`` to obtain the filtered contract universe,
        then reuses ``download_symbol`` for every selected symbol. Selection
        rules and download mechanics remain owned by those methods.

        Raises:
            ValidationError: Propagated from discovery selection or download
                range resolution.
            ExchangeError: Propagated from transport failures during discovery
                or download.
        """
        contracts = await self.discover_symbols()
        for contract in contracts:
            await self.download_symbol(symbol=contract.symbol)

    def _resolve_base_url(self) -> str:
        """Resolve the REST base URL from options.

        Returns:
            Explicit ``base_url`` when provided; otherwise the production or
            testnet default depending on ``testnet``.
        """
        if self._options.base_url is not None:
            return self._options.base_url.strip().rstrip("/")
        if self._options.testnet:
            return DEFAULT_BINANCE_FUTURES_TESTNET_REST_BASE_URL
        return DEFAULT_BINANCE_FUTURES_REST_BASE_URL

    def _resolve_download_range(self) -> tuple[UnixTimestampMs, UnixTimestampMs]:
        """Resolve inclusive download start and end timestamps from options.

        Returns:
            ``(start_time, end_time)`` as UTC Unix milliseconds.

        Raises:
            ValidationError: If ``start_time`` cannot be derived because both
                ``start_time`` and ``history_days`` are omitted.
        """
        end_time = (
            self._options.end_time
            if self._options.end_time is not None
            else int(datetime.now(UTC).timestamp() * 1000)
        )

        if self._options.start_time is not None:
            start_time = self._options.start_time
        elif self._options.history_days is not None:
            start_time = max(
                0,
                end_time - (self._options.history_days * MILLISECONDS_PER_DAY),
            )
        else:
            raise ValidationError(
                "download range requires start_time or history_days",
                error_code=_ERROR_DOWNLOAD_RANGE,
                details={
                    "start_time": self._options.start_time,
                    "end_time": self._options.end_time,
                    "history_days": self._options.history_days,
                },
                recovery_suggestion=(
                    "Set BootstrapOptions.start_time or BootstrapOptions.history_days "
                    "before calling download_symbol."
                ),
            )

        return start_time, end_time

    def _select_contracts(self, contracts: tuple[Contract, ...]) -> tuple[Contract, ...]:
        """Apply allowlist and size limits to discovered contracts.

        Args:
            contracts: Contracts returned by ``SymbolDiscovery``.

        Returns:
            Filtered and optionally truncated contract tuple.

        Raises:
            ValidationError: If requested allowlist symbols are missing.
        """
        selected: tuple[Contract, ...]
        if self._options.symbols:
            by_symbol = {contract.symbol: contract for contract in contracts}
            missing = tuple(symbol for symbol in self._options.symbols if symbol not in by_symbol)
            if missing:
                raise ValidationError(
                    "Requested bootstrap symbols were not discovered",
                    error_code=_ERROR_MISSING_SYMBOLS,
                    details={
                        "missing_symbols": missing,
                        "requested_symbols": self._options.symbols,
                    },
                    recovery_suggestion=(
                        "Remove unavailable symbols from BootstrapOptions.symbols "
                        "or wait until they are listed as trading USDT perpetuals."
                    ),
                )
            selected = tuple(by_symbol[symbol] for symbol in self._options.symbols)
        else:
            selected = contracts

        if self._options.max_symbols is not None:
            selected = selected[: self._options.max_symbols]
        return selected


def _require_unix_ms(value: object, *, parameter: str, error_code: str) -> None:
    """Validate a Unix millisecond timestamp.

    Args:
        value: Candidate timestamp value.
        parameter: Parameter name for error context.
        error_code: Stable machine-readable error code.

    Raises:
        ValidationError: If ``value`` is not a non-negative integer.
    """
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValidationError(
            f"{parameter} must be a non-negative Unix millisecond timestamp",
            error_code=error_code,
            details={"parameter": parameter, "value": value},
        )
