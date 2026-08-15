"""CQROS market-data dataset repository.

Purpose:
    Provide a path-free facade for persisting and retrieving raw market
    datasets by composing canonical locations through ``StorageLayout`` and
    delegating I/O to an injected ``IDataStore``.

Responsibilities:
    - Resolve storage locations for OHLCV, funding, open interest, taker
      volume, long/short ratios, and liquidation year partitions
    - Report OHLCV partition existence without reading stored frames
    - Expose latest-timestamp helpers for resumable downloads without
      loading full partitions
    - Merge year partitions on save while preserving timestamp uniqueness
    - Ignore unreadable newest partitions when resolving latest timestamps
    - Delegate all read and write operations to ``IDataStore``
    - Keep filesystem paths out of the public API
    - Remain free of validation, downloader, and domain transforms

Dependencies:
    ``polars``, ``cqros.core``, and ``cqros.storage`` layout/interfaces.

Public API:
    ``MarketDataRepository``
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import polars as pl
from polars.exceptions import PolarsError

from cqros.core.constants import EXCHANGE_BINANCE, MARKET_USDT_PERPETUAL
from cqros.core.types import Exchange, Market, Symbol, Timeframe, UnixTimestampMs
from cqros.storage.exceptions import CorruptedDatasetError, StorageError
from cqros.storage.interfaces import IDataStore
from cqros.storage.layout import StorageLayout

__all__ = [
    "MarketDataRepository",
]

_logger = logging.getLogger(__name__)

_EXCHANGE: Final[Exchange] = EXCHANGE_BINANCE
_MARKET: Final[Market] = MARKET_USDT_PERPETUAL

# Binance USDⓈ-M Futures launched in 2019; partitions cannot precede this year.
_EARLIEST_PARTITION_YEAR: Final[int] = 2019

_TS_OPEN_TIME: Final[str] = "open_time"
_TS_FUNDING_TIME: Final[str] = "funding_time"
_TS_TIMESTAMP: Final[str] = "timestamp"

type _YearPartitionPathBuilder = Callable[
    [Exchange, Market, Symbol, Timeframe, int],
    Path,
]


class MarketDataRepository:
    """Repository facade for raw market datasets.

    Callers identify datasets by exchange, market, symbol, timeframe, and
    year. Paths are composed privately via ``StorageLayout`` and never
    returned. Persistence is delegated entirely to the injected
    ``IDataStore`` so alternate backends can be substituted without changing
    this API.

    Args:
        layout: Canonical path composer for the data lake.
        datastore: Storage backend implementing ``IDataStore``.
        logger: Optional logger instance. Defaults to the module logger.
    """

    __slots__ = ("_datastore", "_layout", "_logger")

    _layout: StorageLayout
    _datastore: IDataStore
    _logger: logging.Logger

    def __init__(
        self,
        layout: StorageLayout,
        datastore: IDataStore,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the repository with injected layout and datastore.

        Args:
            layout: Canonical path composer for the data lake.
            datastore: Storage backend used for all persistence operations.
            logger: Optional logger instance.
        """
        self._layout = layout
        self._datastore = datastore
        self._logger = logger if logger is not None else _logger

    def has_ohlcv(self, symbol: Symbol, timeframe: Timeframe) -> bool:
        """Return whether any OHLCV year partition exists for a symbol.

        Existence is determined solely through ``IDataStore.exists`` on paths
        composed by ``StorageLayout``. Parquet contents are never read,
        scanned, or validated.

        Args:
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Bar interval (for example ``1m``).

        Returns:
            ``True`` when at least one yearly OHLCV partition exists for the
            Binance USDⓈ-M perpetual dataset; otherwise ``False``.
        """
        current_year = datetime.now(UTC).year
        for year in range(_EARLIEST_PARTITION_YEAR, current_year + 1):
            path = self._layout.raw_ohlcv_path(
                _EXCHANGE,
                _MARKET,
                symbol,
                timeframe,
                year,
            )
            if self._datastore.exists(path):
                self._logger.debug(
                    "OHLCV dataset exists",
                    extra={
                        "dataset": "ohlcv",
                        "exchange": _EXCHANGE,
                        "market": _MARKET,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "year": year,
                    },
                )
                return True

        self._logger.debug(
            "OHLCV dataset does not exist",
            extra={
                "dataset": "ohlcv",
                "exchange": _EXCHANGE,
                "market": _MARKET,
                "symbol": symbol,
                "timeframe": timeframe,
            },
        )
        return False

    def save_ohlcv(
        self,
        dataframe: pl.DataFrame,
        *,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> None:
        """Persist an OHLCV year partition.

        Args:
            dataframe: OHLCV frame to store.
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Bar interval (for example ``1m``).
            year: Calendar year of the partition.
        """
        self._save_year_partition(
            self._layout.raw_ohlcv_path,
            dataframe,
            dataset="ohlcv",
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
            unique_column=_TS_OPEN_TIME,
        )

    def load_ohlcv(
        self,
        *,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> pl.DataFrame:
        """Load an OHLCV year partition.

        Args:
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Bar interval (for example ``1m``).
            year: Calendar year of the partition.

        Returns:
            Eagerly loaded OHLCV DataFrame.
        """
        return self._load_year_partition(
            self._layout.raw_ohlcv_path,
            dataset="ohlcv",
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        )

    def save_funding(
        self,
        dataframe: pl.DataFrame,
        *,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> None:
        """Persist a funding-rate year partition.

        Args:
            dataframe: Funding frame to store.
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Sampling interval used for the stored series.
            year: Calendar year of the partition.
        """
        self._save_year_partition(
            self._layout.raw_funding_path,
            dataframe,
            dataset="funding",
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
            unique_column=_TS_FUNDING_TIME,
        )

    def load_funding(
        self,
        *,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> pl.DataFrame:
        """Load a funding-rate year partition.

        Args:
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Sampling interval used for the stored series.
            year: Calendar year of the partition.

        Returns:
            Eagerly loaded funding DataFrame.
        """
        return self._load_year_partition(
            self._layout.raw_funding_path,
            dataset="funding",
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        )

    def save_open_interest(
        self,
        dataframe: pl.DataFrame,
        *,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> None:
        """Persist an open-interest year partition.

        Args:
            dataframe: Open-interest frame to store.
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Sampling interval used for the stored series.
            year: Calendar year of the partition.
        """
        self._save_year_partition(
            self._layout.raw_open_interest_path,
            dataframe,
            dataset="open_interest",
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
            unique_column=_TS_TIMESTAMP,
        )

    def load_open_interest(
        self,
        *,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> pl.DataFrame:
        """Load an open-interest year partition.

        Args:
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Sampling interval used for the stored series.
            year: Calendar year of the partition.

        Returns:
            Eagerly loaded open-interest DataFrame.
        """
        return self._load_year_partition(
            self._layout.raw_open_interest_path,
            dataset="open_interest",
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        )

    def save_taker_volume(
        self,
        dataframe: pl.DataFrame,
        *,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> None:
        """Persist a taker buy/sell volume year partition.

        Args:
            dataframe: Taker-volume frame to store.
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Sampling interval used for the stored series.
            year: Calendar year of the partition.
        """
        self._save_year_partition(
            self._layout.raw_taker_volume_path,
            dataframe,
            dataset="taker_volume",
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
            unique_column=_TS_TIMESTAMP,
        )

    def load_taker_volume(
        self,
        *,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> pl.DataFrame:
        """Load a taker buy/sell volume year partition.

        Args:
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Sampling interval used for the stored series.
            year: Calendar year of the partition.

        Returns:
            Eagerly loaded taker-volume DataFrame.
        """
        return self._load_year_partition(
            self._layout.raw_taker_volume_path,
            dataset="taker_volume",
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        )

    def save_global_long_short_account_ratio(
        self,
        dataframe: pl.DataFrame,
        *,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> None:
        """Persist a global long/short account-ratio year partition.

        Args:
            dataframe: Long/short ratio frame to store.
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Sampling interval used for the stored series.
            year: Calendar year of the partition.
        """
        self._save_year_partition(
            self._layout.raw_global_long_short_account_ratio_path,
            dataframe,
            dataset="global_long_short_account_ratio",
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
            unique_column=_TS_TIMESTAMP,
        )

    def load_global_long_short_account_ratio(
        self,
        *,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> pl.DataFrame:
        """Load a global long/short account-ratio year partition.

        Args:
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Sampling interval used for the stored series.
            year: Calendar year of the partition.

        Returns:
            Eagerly loaded global long/short account-ratio DataFrame.
        """
        return self._load_year_partition(
            self._layout.raw_global_long_short_account_ratio_path,
            dataset="global_long_short_account_ratio",
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        )

    def save_top_long_short_account_ratio(
        self,
        dataframe: pl.DataFrame,
        *,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> None:
        """Persist a top-trader long/short account-ratio year partition.

        Args:
            dataframe: Long/short ratio frame to store.
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Sampling interval used for the stored series.
            year: Calendar year of the partition.
        """
        self._save_year_partition(
            self._layout.raw_top_long_short_account_ratio_path,
            dataframe,
            dataset="top_long_short_account_ratio",
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
            unique_column=_TS_TIMESTAMP,
        )

    def load_top_long_short_account_ratio(
        self,
        *,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> pl.DataFrame:
        """Load a top-trader long/short account-ratio year partition.

        Args:
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Sampling interval used for the stored series.
            year: Calendar year of the partition.

        Returns:
            Eagerly loaded top-trader long/short account-ratio DataFrame.
        """
        return self._load_year_partition(
            self._layout.raw_top_long_short_account_ratio_path,
            dataset="top_long_short_account_ratio",
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        )

    def save_top_long_short_position_ratio(
        self,
        dataframe: pl.DataFrame,
        *,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> None:
        """Persist a top-trader long/short position-ratio year partition.

        Args:
            dataframe: Long/short ratio frame to store.
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Sampling interval used for the stored series.
            year: Calendar year of the partition.
        """
        self._save_year_partition(
            self._layout.raw_top_long_short_position_ratio_path,
            dataframe,
            dataset="top_long_short_position_ratio",
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
            unique_column=_TS_TIMESTAMP,
        )

    def load_top_long_short_position_ratio(
        self,
        *,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> pl.DataFrame:
        """Load a top-trader long/short position-ratio year partition.

        Args:
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Sampling interval used for the stored series.
            year: Calendar year of the partition.

        Returns:
            Eagerly loaded top-trader long/short position-ratio DataFrame.
        """
        return self._load_year_partition(
            self._layout.raw_top_long_short_position_ratio_path,
            dataset="top_long_short_position_ratio",
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        )

    def save_liquidations(
        self,
        dataframe: pl.DataFrame,
        *,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> None:
        """Persist a liquidation year partition.

        Args:
            dataframe: Liquidation frame to store.
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Sampling interval used for the stored series.
            year: Calendar year of the partition.
        """
        self._save_year_partition(
            self._layout.raw_liquidation_path,
            dataframe,
            dataset="liquidations",
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
            unique_column=_TS_TIMESTAMP,
        )

    def load_liquidations(
        self,
        *,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> pl.DataFrame:
        """Load a liquidation year partition.

        Args:
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Sampling interval used for the stored series.
            year: Calendar year of the partition.

        Returns:
            Eagerly loaded liquidation DataFrame.
        """
        return self._load_year_partition(
            self._layout.raw_liquidation_path,
            dataset="liquidations",
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        )

    def get_latest_ohlcv_timestamp(
        self,
        *,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
    ) -> UnixTimestampMs | None:
        """Return the latest persisted OHLCV ``open_time``, if any.

        Only the newest readable year partition is inspected, reading solely
        the timestamp column. Corrupt newest partitions are skipped.
        """
        return self._get_latest_timestamp(
            self._layout.raw_ohlcv_path,
            dataset="ohlcv",
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            timestamp_column=_TS_OPEN_TIME,
        )

    def get_latest_funding_timestamp(
        self,
        *,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
    ) -> UnixTimestampMs | None:
        """Return the latest persisted funding ``funding_time``, if any."""
        return self._get_latest_timestamp(
            self._layout.raw_funding_path,
            dataset="funding",
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            timestamp_column=_TS_FUNDING_TIME,
        )

    def get_latest_open_interest_timestamp(
        self,
        *,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
    ) -> UnixTimestampMs | None:
        """Return the latest persisted open-interest ``timestamp``, if any."""
        return self._get_latest_timestamp(
            self._layout.raw_open_interest_path,
            dataset="open_interest",
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            timestamp_column=_TS_TIMESTAMP,
        )

    def get_latest_taker_volume_timestamp(
        self,
        *,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
    ) -> UnixTimestampMs | None:
        """Return the latest persisted taker-volume ``timestamp``, if any."""
        return self._get_latest_timestamp(
            self._layout.raw_taker_volume_path,
            dataset="taker_volume",
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            timestamp_column=_TS_TIMESTAMP,
        )

    def get_latest_long_short_timestamp(
        self,
        *,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        dataset: str,
    ) -> UnixTimestampMs | None:
        """Return the latest persisted long/short ``timestamp``, if any.

        Args:
            exchange: Exchange identifier.
            market: Market segment.
            symbol: Tradeable symbol.
            timeframe: Aggregation period used as the partition key.
            dataset: Storage dataset name selecting the long/short namespace
                (``global_long_short_account_ratio``,
                ``top_long_short_account_ratio``, or
                ``top_long_short_position_ratio``).

        Returns:
            Latest timestamp in UTC milliseconds, or ``None`` when no readable
            partition exists.

        Raises:
            ValueError: If ``dataset`` is not a supported long/short namespace.
        """
        path_builder = _long_short_path_builder(self._layout, dataset=dataset)
        return self._get_latest_timestamp(
            path_builder,
            dataset=dataset,
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            timestamp_column=_TS_TIMESTAMP,
        )

    def _get_latest_timestamp(
        self,
        path_builder: _YearPartitionPathBuilder,
        *,
        dataset: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        timestamp_column: str,
    ) -> UnixTimestampMs | None:
        """Scan year partitions newest-first for the max timestamp column."""
        current_year = datetime.now(UTC).year
        for year in range(current_year, _EARLIEST_PARTITION_YEAR - 1, -1):
            path = path_builder(exchange, market, symbol, timeframe, year)
            if not self._datastore.exists(path):
                continue
            try:
                latest = (
                    self._datastore.scan(path)
                    .select(pl.col(timestamp_column).max())
                    .collect()
                    .item()
                )
            except (CorruptedDatasetError, StorageError, PolarsError, OSError) as exc:
                self._logger.warning(
                    "Ignoring unreadable partition while resolving latest timestamp",
                    extra={
                        "dataset": dataset,
                        "exchange": exchange,
                        "market": market,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "year": year,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                )
                continue

            if latest is None:
                continue
            return int(latest)

        return None

    def _save_year_partition(
        self,
        path_builder: _YearPartitionPathBuilder,
        dataframe: pl.DataFrame,
        *,
        dataset: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
        unique_column: str,
    ) -> None:
        """Compose a year-partition path and write through the datastore.

        When a partition already exists, rows are merged and deduplicated on
        ``unique_column`` (keeping the last occurrence) before rewrite. Unreadable
        existing partitions are replaced by the incoming frame.
        """
        path = path_builder(exchange, market, symbol, timeframe, year)
        to_write = self._merge_with_existing(
            path,
            dataframe,
            dataset=dataset,
            unique_column=unique_column,
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        )
        self._logger.debug(
            "Saving market dataset",
            extra=_dataset_log_extra(
                dataset=dataset,
                exchange=exchange,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
                rows=to_write.height,
                columns=to_write.width,
            ),
        )
        self._datastore.write(path, to_write)
        self._logger.info(
            "Saved market dataset",
            extra=_dataset_log_extra(
                dataset=dataset,
                exchange=exchange,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
                rows=to_write.height,
                columns=to_write.width,
            ),
        )

    def _merge_with_existing(
        self,
        path: Path,
        dataframe: pl.DataFrame,
        *,
        dataset: str,
        unique_column: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> pl.DataFrame:
        """Merge ``dataframe`` with an existing partition, preserving uniqueness."""
        if not self._datastore.exists(path):
            return dataframe
        try:
            existing = self._datastore.read(path)
        except (CorruptedDatasetError, StorageError, OSError) as exc:
            self._logger.warning(
                "Replacing unreadable partition during merge save",
                extra={
                    "dataset": dataset,
                    "exchange": exchange,
                    "market": market,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "year": year,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            return dataframe

        if existing.height == 0:
            return dataframe
        if dataframe.height == 0:
            return existing

        merged = pl.concat([existing, dataframe], how="vertical")
        return merged.unique(subset=[unique_column], keep="last").sort(unique_column)

    def _load_year_partition(
        self,
        path_builder: _YearPartitionPathBuilder,
        *,
        dataset: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> pl.DataFrame:
        """Compose a year-partition path and read through the datastore.

        Args:
            path_builder: Layout method that returns the source path.
            dataset: Logical dataset name for logging.
            exchange: Exchange identifier.
            market: Market segment.
            symbol: Tradeable symbol.
            timeframe: Bar or sampling interval.
            year: Calendar year of the partition.

        Returns:
            Eagerly loaded DataFrame from the datastore.
        """
        path = path_builder(exchange, market, symbol, timeframe, year)
        self._logger.debug(
            "Loading market dataset",
            extra=_dataset_log_extra(
                dataset=dataset,
                exchange=exchange,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
            ),
        )
        frame = self._datastore.read(path)
        self._logger.info(
            "Loaded market dataset",
            extra=_dataset_log_extra(
                dataset=dataset,
                exchange=exchange,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
                rows=frame.height,
                columns=frame.width,
            ),
        )
        return frame


def _long_short_path_builder(
    layout: StorageLayout,
    *,
    dataset: str,
) -> _YearPartitionPathBuilder:
    """Resolve the layout path builder for a long/short storage dataset."""
    mapping: dict[str, _YearPartitionPathBuilder] = {
        "global_long_short_account_ratio": layout.raw_global_long_short_account_ratio_path,
        "top_long_short_account_ratio": layout.raw_top_long_short_account_ratio_path,
        "top_long_short_position_ratio": layout.raw_top_long_short_position_ratio_path,
    }
    try:
        return mapping[dataset]
    except KeyError as exc:
        raise ValueError(
            "dataset must be a supported long/short storage namespace",
        ) from exc


def _dataset_log_extra(
    *,
    dataset: str,
    exchange: Exchange,
    market: Market,
    symbol: Symbol,
    timeframe: Timeframe,
    year: int,
    rows: int | None = None,
    columns: int | None = None,
) -> dict[str, object]:
    """Build structured log fields for a dataset operation.

    Args:
        dataset: Logical dataset name.
        exchange: Exchange identifier.
        market: Market segment.
        symbol: Tradeable symbol.
        timeframe: Bar or sampling interval.
        year: Calendar year of the partition.
        rows: Optional row count.
        columns: Optional column count.

    Returns:
        Mapping suitable for ``logging.Logger`` ``extra``.
    """
    payload: dict[str, object] = {
        "dataset": dataset,
        "exchange": exchange,
        "market": market,
        "symbol": symbol,
        "timeframe": timeframe,
        "year": year,
    }
    if rows is not None:
        payload["rows"] = rows
    if columns is not None:
        payload["columns"] = columns
    return payload
