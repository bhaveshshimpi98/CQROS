"""CQROS incremental market-data updater.

Purpose:
    Extend stored Binance OHLCV datasets with only the candles that are missing
    after the latest persisted open time, without redownloading complete
    history.

Responsibilities:
    - Resolve the latest stored ``open_time`` for a symbol/timeframe
    - Fetch only the missing forward range through ``HistoricalDownloader``
    - Validate downloaded frames before merge
    - Merge with existing year partitions, deduplicate, and sort
    - Rewrite solely the year partitions touched by new rows

Dependencies:
    ``polars``, ``cqros.core``, ``cqros.ingestion.client``,
    ``cqros.ingestion.downloader``, ``cqros.ingestion.validator``, and
    ``cqros.storage``.

Public API:
    ``IncrementalUpdater``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Final

import polars as pl

from cqros.core.constants import (
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
)
from cqros.core.exceptions import (
    DataValidationError,
    MissingDataError,
    ValidationError,
)
from cqros.core.types import (
    Exchange,
    Market,
    Symbol,
    Timeframe,
    UnixTimestampMs,
)
from cqros.ingestion.client import BinanceClient
from cqros.ingestion.downloader import HistoricalDownloader
from cqros.ingestion.validator import MarketDataValidator, ValidationReport
from cqros.storage.exceptions import DatasetNotFoundError
from cqros.storage.repository import MarketDataRepository

__all__ = [
    "IncrementalUpdater",
]

_EXCHANGE: Final[Exchange] = EXCHANGE_BINANCE
_MARKET: Final[Market] = MARKET_USDT_PERPETUAL

_OHLCV_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "symbol": pl.String,
        "timeframe": pl.String,
        "open_time": pl.Int64,
        "close_time": pl.Int64,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
        "quote_volume": pl.Float64,
        "trade_count": pl.Int64,
    }
)

_logger = logging.getLogger(__name__)


class IncrementalUpdater:
    """Incrementally update stored Binance OHLCV datasets.

    Reads the latest persisted candle open time, downloads only missing forward
    data, validates the download, merges it into existing year partitions, and
    rewrites solely the partitions that change. Complete historical ranges are
    never redownloaded; callers must seed storage with
    ``HistoricalDownloader`` first.

    Args:
        client: Binance REST client; the session is opened before each fetch.
        repository: Market-data repository used to load and rewrite partitions.
        validator: Validator applied to downloaded OHLCV frames before merge.
        downloader: Historical downloader used to fetch missing ranges without
            persisting them directly (merge rewrite is owned by this class).
        logger: Optional logger instance. Defaults to the module logger.
    """

    __slots__ = (
        "_client",
        "_repository",
        "_validator",
        "_downloader",
        "_logger",
    )

    _client: BinanceClient
    _repository: MarketDataRepository
    _validator: MarketDataValidator
    _downloader: HistoricalDownloader
    _logger: logging.Logger

    def __init__(
        self,
        client: BinanceClient,
        repository: MarketDataRepository,
        validator: MarketDataValidator,
        downloader: HistoricalDownloader,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the updater with injected collaborators.

        Args:
            client: Binance USDⓈ-M Futures REST client.
            repository: Repository used to read and rewrite OHLCV partitions.
            validator: Market-data validator for downloaded frames.
            downloader: Downloader used to fetch missing kline ranges.
            logger: Optional logger instance.
        """
        self._client = client
        self._repository = repository
        self._validator = validator
        self._downloader = downloader
        self._logger = logger if logger is not None else _logger

    async def update_symbol(
        self,
        *,
        symbol: Symbol,
        timeframe: Timeframe,
        end_time: UnixTimestampMs | None = None,
    ) -> None:
        """Incrementally update OHLCV storage for a single symbol.

        Args:
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Bar interval identifier (for example ``1m``).
            end_time: Inclusive update end as UTC Unix milliseconds. When
                omitted, the current UTC wall-clock time is used.

        Raises:
            MissingDataError: If no stored OHLCV partitions exist for the
                symbol and timeframe.
            ValidationError: If ``end_time`` is not a valid Unix millisecond
                timestamp.
            DataValidationError: If the downloaded frame fails validation.
            ExchangeError: Propagated from transport failures during fetch.
        """
        resolved_end = _resolve_end_time(end_time)
        latest = self._latest_open_time(symbol=symbol, timeframe=timeframe)
        if latest is None:
            raise MissingDataError(
                "No stored OHLCV data found for incremental update",
                error_code="INGESTION-UPDATER-001",
                details={
                    "exchange": _EXCHANGE,
                    "market": _MARKET,
                    "symbol": symbol,
                    "timeframe": timeframe,
                },
                recovery_suggestion=(
                    "Run HistoricalDownloader.download_symbol to seed storage "
                    "before calling IncrementalUpdater."
                ),
            )

        start_time = latest + 1
        if start_time > resolved_end:
            self._logger.info(
                "Symbol already up to date",
                extra={
                    "exchange": _EXCHANGE,
                    "market": _MARKET,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "latest_open_time": latest,
                    "end_time": resolved_end,
                },
            )
            return

        self._logger.info(
            "Starting symbol incremental update",
            extra={
                "exchange": _EXCHANGE,
                "market": _MARKET,
                "symbol": symbol,
                "timeframe": timeframe,
                "latest_open_time": latest,
                "start_time": start_time,
                "end_time": resolved_end,
            },
        )

        await self._client.open()
        downloaded = await self._downloader.fetch_symbol(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=resolved_end,
        )
        if downloaded.height == 0:
            self._logger.info(
                "Incremental update fetched no new rows",
                extra={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "start_time": start_time,
                    "end_time": resolved_end,
                },
            )
            return

        report = self._validator.validate(downloaded, timeframe)
        _require_valid_download(report, symbol=symbol, timeframe=timeframe)

        self._merge_and_rewrite(
            downloaded,
            symbol=symbol,
            timeframe=timeframe,
        )
        self._logger.info(
            "Completed symbol incremental update",
            extra={
                "symbol": symbol,
                "timeframe": timeframe,
                "start_time": start_time,
                "end_time": resolved_end,
                "downloaded_rows": downloaded.height,
            },
        )

    async def update_universe(
        self,
        symbols: Sequence[Symbol],
        *,
        timeframe: Timeframe,
        end_time: UnixTimestampMs | None = None,
    ) -> None:
        """Incrementally update OHLCV storage for multiple symbols.

        Symbols are processed sequentially so rate-limit pressure remains
        predictable. Each symbol is updated independently.

        Args:
            symbols: Ordered symbols to update.
            timeframe: Bar interval identifier (for example ``1m``).
            end_time: Inclusive update end as UTC Unix milliseconds. When
                omitted, the current UTC wall-clock time is used.

        Raises:
            MissingDataError: If any symbol has no stored OHLCV partitions.
            ValidationError: If ``end_time`` is not a valid Unix millisecond
                timestamp.
            DataValidationError: If a downloaded frame fails validation.
            ExchangeError: Propagated from transport failures during fetch.
        """
        resolved_end = _resolve_end_time(end_time)
        self._logger.info(
            "Starting universe incremental update",
            extra={
                "exchange": _EXCHANGE,
                "market": _MARKET,
                "symbol_count": len(symbols),
                "timeframe": timeframe,
                "end_time": resolved_end,
            },
        )
        for symbol in symbols:
            await self.update_symbol(
                symbol=symbol,
                timeframe=timeframe,
                end_time=resolved_end,
            )
        self._logger.info(
            "Completed universe incremental update",
            extra={
                "symbol_count": len(symbols),
                "timeframe": timeframe,
                "end_time": resolved_end,
            },
        )

    def _latest_open_time(
        self,
        *,
        symbol: Symbol,
        timeframe: Timeframe,
    ) -> UnixTimestampMs | None:
        """Return the maximum stored ``open_time``, if any partition exists.

        Args:
            symbol: Tradeable symbol.
            timeframe: Bar interval identifier.

        Returns:
            Latest ``open_time`` in UTC milliseconds, or ``None`` when no
            readable year partition exists.
        """
        return self._repository.get_latest_ohlcv_timestamp(
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=symbol,
            timeframe=timeframe,
        )

    def _merge_and_rewrite(
        self,
        downloaded: pl.DataFrame,
        *,
        symbol: Symbol,
        timeframe: Timeframe,
    ) -> None:
        """Merge downloaded rows into existing partitions and rewrite them.

        Args:
            downloaded: Validated OHLCV rows to merge.
            symbol: Tradeable symbol.
            timeframe: Bar interval identifier.
        """
        year_expr = pl.from_epoch(pl.col("open_time"), time_unit="ms").dt.year()
        # Polars DataFrame stubs expose Unknown aliases under pyright strict.
        year_frame = downloaded.with_columns(  # pyright: ignore[reportUnknownMemberType]
            year_expr.alias("_year")
        )
        years = year_frame.get_column("_year").unique().sort().to_list()

        for year in years:
            year_int = int(year)
            new_partition = year_frame.filter(  # pyright: ignore[reportUnknownMemberType]
                pl.col("_year") == year
            ).drop("_year")
            existing = self._load_partition_or_empty(
                symbol=symbol,
                timeframe=timeframe,
                year=year_int,
            )
            merged = _merge_ohlcv(existing, new_partition)
            self._repository.save_ohlcv(
                merged,
                exchange=_EXCHANGE,
                market=_MARKET,
                symbol=symbol,
                timeframe=timeframe,
                year=year_int,
            )
            self._logger.debug(
                "Rewrote OHLCV year partition after incremental merge",
                extra={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "year": year_int,
                    "rows": merged.height,
                },
            )

    def _load_partition_or_empty(
        self,
        *,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> pl.DataFrame:
        """Load a year partition or return an empty OHLCV frame.

        Args:
            symbol: Tradeable symbol.
            timeframe: Bar interval identifier.
            year: Calendar year of the partition.

        Returns:
            Existing partition rows, or an empty frame with the canonical
            OHLCV schema when the partition is absent.
        """
        try:
            return self._repository.load_ohlcv(
                exchange=_EXCHANGE,
                market=_MARKET,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
            )
        except DatasetNotFoundError:
            return pl.DataFrame(schema=_OHLCV_SCHEMA)


def _resolve_end_time(end_time: UnixTimestampMs | None) -> UnixTimestampMs:
    """Resolve the inclusive update end timestamp.

    Args:
        end_time: Caller-supplied end time, or ``None`` for current UTC time.

    Returns:
        Inclusive end timestamp in UTC Unix milliseconds.

    Raises:
        ValidationError: If ``end_time`` is not an ``int``.
    """
    if end_time is None:
        return int(datetime.now(UTC).timestamp() * 1000)
    return _require_unix_ms(end_time, parameter="end_time")


def _require_unix_ms(value: object, *, parameter: str) -> UnixTimestampMs:
    """Validate a Unix-millisecond timestamp parameter.

    Args:
        value: Candidate timestamp value.
        parameter: Parameter name for error context.

    Returns:
        The validated integer timestamp.

    Raises:
        ValidationError: If ``value`` is not an ``int``.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(
            f"{parameter} must be an int Unix timestamp in milliseconds",
            error_code="INGESTION-UPDATER-002",
            details={"parameter": parameter, "type": type(value).__name__},
        )
    return value


def _require_valid_download(
    report: ValidationReport,
    *,
    symbol: Symbol,
    timeframe: Timeframe,
) -> None:
    """Raise when a downloaded frame failed validation.

    Args:
        report: Validation outcome for the downloaded frame.
        symbol: Tradeable symbol for error context.
        timeframe: Bar interval for error context.

    Raises:
        DataValidationError: If the report contains error-severity issues.
    """
    if report.is_valid:
        return

    errors = report.errors()
    raise DataValidationError(
        "Downloaded OHLCV failed validation before incremental merge",
        error_code="INGESTION-UPDATER-003",
        details={
            "symbol": symbol,
            "timeframe": timeframe,
            "row_count": report.row_count,
            "error_count": len(errors),
            "checks": [issue.check for issue in errors],
        },
        recovery_suggestion=(
            "Inspect validation errors, correct the exchange payload or "
            "timeframe contract, and retry the incremental update."
        ),
    )


def _merge_ohlcv(existing: pl.DataFrame, new_rows: pl.DataFrame) -> pl.DataFrame:
    """Concatenate, deduplicate by ``open_time``, and sort chronologically.

    When both frames contain the same ``open_time``, the newer row wins so
    exchange corrections in the download replace previously stored values.

    Args:
        existing: Rows already stored for the affected year partition.
        new_rows: Newly downloaded rows for the same year.

    Returns:
        Deduplicated frame sorted by ascending ``open_time``.
    """
    if existing.height == 0:
        base = new_rows
    elif new_rows.height == 0:
        base = existing
    else:
        base = pl.concat([existing, new_rows], how="vertical")

    return base.unique(
        subset=["open_time"], keep="last"
    ).sort(  # pyright: ignore[reportUnknownMemberType]
        "open_time"
    )
