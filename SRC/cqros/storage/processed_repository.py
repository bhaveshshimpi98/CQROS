"""CQROS processed market-data dataset repository.

Purpose:
    Provide a path-free facade for persisting and retrieving processed market
    datasets produced by the Processing Framework.

Responsibilities:
    - Resolve storage locations for processed OHLCV, funding, open interest,
      taker volume, and long/short ratio year partitions
    - Report processed OHLCV partition existence without reading stored frames
    - Discover processed datasets, symbols, timeframes, and year partitions
    - Delegate all read and write operations to an injected ``IDataStore``
      (typically ``ParquetStore`` for atomic Parquet writes)
    - Keep filesystem paths out of the public API
    - Remain free of processing, feature, factor, research, and validation
      logic beyond storage integrity

Dependencies:
    ``polars``, ``cqros.core``, and ``cqros.storage`` layout/interfaces.

Public API:
    ``PROCESSED_DATASETS``, ``ProcessedPartitionRef``,
    ``ProcessedMarketDataRepository``
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import polars as pl

from cqros.core.constants import (
    EXCHANGE_BINANCE,
    FILE_EXTENSION_PARQUET,
    MARKET_USDT_PERPETUAL,
    STORAGE_DIR_PROCESSED,
)
from cqros.core.types import Exchange, Market, Symbol, Timeframe
from cqros.storage.interfaces import IDataStore
from cqros.storage.layout import StorageLayout

__all__ = [
    "PROCESSED_DATASETS",
    "ProcessedMarketDataRepository",
    "ProcessedPartitionRef",
]

_logger = logging.getLogger(__name__)

_EXCHANGE: Final[Exchange] = EXCHANGE_BINANCE
_MARKET: Final[Market] = MARKET_USDT_PERPETUAL

# Binance USDⓈ-M Futures launched in 2019; partitions cannot precede this year.
_EARLIEST_PARTITION_YEAR: Final[int] = 2019

PROCESSED_DATASETS: Final[tuple[str, ...]] = (
    "ohlcv",
    "funding",
    "open_interest",
    "taker_volume",
    "global_long_short_account_ratio",
    "top_long_short_account_ratio",
    "top_long_short_position_ratio",
)

type _YearPartitionPathBuilder = Callable[
    [Exchange, Market, Symbol, Timeframe, int],
    Path,
]


@dataclass(frozen=True, slots=True)
class ProcessedPartitionRef:
    """Identity of one discovered processed year partition.

    Attributes:
        dataset: Storage dataset name under the processed tier.
        symbol: Tradeable symbol.
        timeframe: Bar or sampling interval.
        year: Calendar year of the partition.
    """

    dataset: str
    symbol: Symbol
    timeframe: Timeframe
    year: int


class ProcessedMarketDataRepository:
    """Repository facade for processed market datasets.

    Callers identify datasets by exchange, market, symbol, timeframe, and
    year. Paths are composed privately via ``StorageLayout`` and never
    returned. Persistence is delegated entirely to the injected
    ``IDataStore`` (commonly ``ParquetStore``) so atomic writes and alternate
    backends can be substituted without changing this API.

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

    def discover_datasets(
        self,
        *,
        exchange: Exchange = _EXCHANGE,
        market: Market = _MARKET,
    ) -> tuple[str, ...]:
        """Return sorted processed dataset names that contain partitions.

        Args:
            exchange: Exchange identifier.
            market: Market segment.

        Returns:
            Deterministically ordered dataset names under the processed tier.
        """
        discovered: list[str] = []
        for dataset in PROCESSED_DATASETS:
            if self.discover_partitions(
                datasets=(dataset,),
                exchange=exchange,
                market=market,
            ):
                discovered.append(dataset)
        return tuple(discovered)

    def discover_symbols(
        self,
        *,
        dataset: str,
        exchange: Exchange = _EXCHANGE,
        market: Market = _MARKET,
    ) -> tuple[Symbol, ...]:
        """Return sorted symbols with at least one partition for ``dataset``.

        Args:
            dataset: Storage dataset name under the processed tier.
            exchange: Exchange identifier.
            market: Market segment.

        Returns:
            Deterministically ordered symbols discovered on disk.
        """
        base = self._dataset_root(dataset, exchange=exchange, market=market)
        if not base.is_dir():
            return ()
        return tuple(sorted(path.name for path in base.iterdir() if path.is_dir()))

    def discover_timeframes(
        self,
        *,
        dataset: str,
        symbol: Symbol,
        exchange: Exchange = _EXCHANGE,
        market: Market = _MARKET,
    ) -> tuple[Timeframe, ...]:
        """Return sorted timeframes with at least one partition for ``symbol``.

        Args:
            dataset: Storage dataset name under the processed tier.
            symbol: Tradeable symbol.
            exchange: Exchange identifier.
            market: Market segment.

        Returns:
            Deterministically ordered timeframes discovered on disk.
        """
        base = self._dataset_root(dataset, exchange=exchange, market=market) / symbol
        if not base.is_dir():
            return ()
        return tuple(sorted(path.name for path in base.iterdir() if path.is_dir()))

    def discover_partitions(
        self,
        *,
        datasets: Sequence[str] | None = None,
        symbols: Sequence[Symbol] | None = None,
        timeframes: Sequence[Timeframe] | None = None,
        exchange: Exchange = _EXCHANGE,
        market: Market = _MARKET,
    ) -> tuple[ProcessedPartitionRef, ...]:
        """Discover processed year partitions matching optional filters.

        Missing dataset trees are skipped. Only year partitions that exist as
        ``{year}.parquet`` files are included. Paths are never returned.

        Args:
            datasets: Optional dataset allowlist. ``None`` discovers every
                known processed dataset.
            symbols: Optional symbol allowlist. ``None`` discovers every
                symbol present for each dataset.
            timeframes: Optional timeframe allowlist. ``None`` discovers every
                timeframe present for each symbol.
            exchange: Exchange identifier.
            market: Market segment.

        Returns:
            Deterministically ordered partition references.
        """
        dataset_list = tuple(datasets) if datasets is not None else PROCESSED_DATASETS
        symbol_filter = set(symbols) if symbols is not None else None
        timeframe_filter = set(timeframes) if timeframes is not None else None

        items: list[ProcessedPartitionRef] = []
        for dataset in dataset_list:
            for symbol in self.discover_symbols(
                dataset=dataset,
                exchange=exchange,
                market=market,
            ):
                if symbol_filter is not None and symbol not in symbol_filter:
                    continue
                for timeframe in self.discover_timeframes(
                    dataset=dataset,
                    symbol=symbol,
                    exchange=exchange,
                    market=market,
                ):
                    if timeframe_filter is not None and timeframe not in timeframe_filter:
                        continue
                    for year in self._discover_years(
                        dataset=dataset,
                        symbol=symbol,
                        timeframe=timeframe,
                        exchange=exchange,
                        market=market,
                    ):
                        items.append(
                            ProcessedPartitionRef(
                                dataset=dataset,
                                symbol=symbol,
                                timeframe=timeframe,
                                year=year,
                            )
                        )

        return tuple(
            sorted(
                items,
                key=lambda item: (
                    item.dataset,
                    item.symbol,
                    item.timeframe,
                    item.year,
                ),
            )
        )

    def has_ohlcv(self, symbol: Symbol, timeframe: Timeframe) -> bool:
        """Return whether any processed OHLCV year partition exists.

        Existence is determined solely through ``IDataStore.exists`` on paths
        composed by ``StorageLayout``. Parquet contents are never read,
        scanned, or validated.

        Args:
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Bar interval (for example ``1m``).

        Returns:
            ``True`` when at least one yearly processed OHLCV partition exists
            for the Binance USDⓈ-M perpetual dataset; otherwise ``False``.
        """
        current_year = datetime.now(UTC).year
        for year in range(_EARLIEST_PARTITION_YEAR, current_year + 1):
            path = self._layout.processed_ohlcv_path(
                _EXCHANGE,
                _MARKET,
                symbol,
                timeframe,
                year,
            )
            if self._datastore.exists(path):
                self._logger.debug(
                    "Processed OHLCV dataset exists",
                    extra={
                        "dataset": "ohlcv",
                        "tier": "processed",
                        "exchange": _EXCHANGE,
                        "market": _MARKET,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "year": year,
                    },
                )
                return True

        self._logger.debug(
            "Processed OHLCV dataset does not exist",
            extra={
                "dataset": "ohlcv",
                "tier": "processed",
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
        """Persist a processed OHLCV year partition.

        Args:
            dataframe: Processed OHLCV frame to store.
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Bar interval (for example ``1m``).
            year: Calendar year of the partition.
        """
        self._save_year_partition(
            self._layout.processed_ohlcv_path,
            dataframe,
            dataset="ohlcv",
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
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
        """Load a processed OHLCV year partition.

        Args:
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Bar interval (for example ``1m``).
            year: Calendar year of the partition.

        Returns:
            Eagerly loaded processed OHLCV DataFrame.
        """
        return self._load_year_partition(
            self._layout.processed_ohlcv_path,
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
        """Persist a processed funding-rate year partition.

        Args:
            dataframe: Processed funding frame to store.
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Sampling interval used for the stored series.
            year: Calendar year of the partition.
        """
        self._save_year_partition(
            self._layout.processed_funding_path,
            dataframe,
            dataset="funding",
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
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
        """Load a processed funding-rate year partition.

        Args:
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Sampling interval used for the stored series.
            year: Calendar year of the partition.

        Returns:
            Eagerly loaded processed funding DataFrame.
        """
        return self._load_year_partition(
            self._layout.processed_funding_path,
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
        """Persist a processed open-interest year partition.

        Args:
            dataframe: Processed open-interest frame to store.
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Sampling interval used for the stored series.
            year: Calendar year of the partition.
        """
        self._save_year_partition(
            self._layout.processed_open_interest_path,
            dataframe,
            dataset="open_interest",
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
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
        """Load a processed open-interest year partition.

        Args:
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Sampling interval used for the stored series.
            year: Calendar year of the partition.

        Returns:
            Eagerly loaded processed open-interest DataFrame.
        """
        return self._load_year_partition(
            self._layout.processed_open_interest_path,
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
        """Persist a processed taker buy/sell volume year partition.

        Args:
            dataframe: Processed taker-volume frame to store.
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Sampling interval used for the stored series.
            year: Calendar year of the partition.
        """
        self._save_year_partition(
            self._layout.processed_taker_volume_path,
            dataframe,
            dataset="taker_volume",
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
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
        """Load a processed taker buy/sell volume year partition.

        Args:
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Sampling interval used for the stored series.
            year: Calendar year of the partition.

        Returns:
            Eagerly loaded processed taker-volume DataFrame.
        """
        return self._load_year_partition(
            self._layout.processed_taker_volume_path,
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
        """Persist a processed global long/short account-ratio year partition.

        Args:
            dataframe: Processed long/short ratio frame to store.
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Sampling interval used for the stored series.
            year: Calendar year of the partition.
        """
        self._save_year_partition(
            self._layout.processed_global_long_short_account_ratio_path,
            dataframe,
            dataset="global_long_short_account_ratio",
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
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
        """Load a processed global long/short account-ratio year partition.

        Args:
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Sampling interval used for the stored series.
            year: Calendar year of the partition.

        Returns:
            Eagerly loaded processed global long/short account-ratio DataFrame.
        """
        return self._load_year_partition(
            self._layout.processed_global_long_short_account_ratio_path,
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
        """Persist a processed top-trader long/short account-ratio year partition.

        Args:
            dataframe: Processed long/short ratio frame to store.
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Sampling interval used for the stored series.
            year: Calendar year of the partition.
        """
        self._save_year_partition(
            self._layout.processed_top_long_short_account_ratio_path,
            dataframe,
            dataset="top_long_short_account_ratio",
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
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
        """Load a processed top-trader long/short account-ratio year partition.

        Args:
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Sampling interval used for the stored series.
            year: Calendar year of the partition.

        Returns:
            Eagerly loaded processed top-trader long/short account-ratio
            DataFrame.
        """
        return self._load_year_partition(
            self._layout.processed_top_long_short_account_ratio_path,
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
        """Persist a processed top-trader long/short position-ratio year partition.

        Args:
            dataframe: Processed long/short ratio frame to store.
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Sampling interval used for the stored series.
            year: Calendar year of the partition.
        """
        self._save_year_partition(
            self._layout.processed_top_long_short_position_ratio_path,
            dataframe,
            dataset="top_long_short_position_ratio",
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
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
        """Load a processed top-trader long/short position-ratio year partition.

        Args:
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Sampling interval used for the stored series.
            year: Calendar year of the partition.

        Returns:
            Eagerly loaded processed top-trader long/short position-ratio
            DataFrame.
        """
        return self._load_year_partition(
            self._layout.processed_top_long_short_position_ratio_path,
            dataset="top_long_short_position_ratio",
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        )

    def _dataset_root(
        self,
        dataset: str,
        *,
        exchange: Exchange,
        market: Market,
    ) -> Path:
        """Return the processed dataset directory for exchange/market."""
        return self._layout.root / STORAGE_DIR_PROCESSED / dataset / exchange / market

    def _discover_years(
        self,
        *,
        dataset: str,
        symbol: Symbol,
        timeframe: Timeframe,
        exchange: Exchange,
        market: Market,
    ) -> tuple[int, ...]:
        """Return sorted calendar years present as parquet partitions."""
        base = self._dataset_root(dataset, exchange=exchange, market=market) / symbol / timeframe
        if not base.is_dir():
            return ()
        years: list[int] = []
        for path in sorted(base.iterdir()):
            if not path.is_file():
                continue
            if path.suffix != FILE_EXTENSION_PARQUET:
                continue
            stem = path.stem
            if not stem.isdigit():
                continue
            year = int(stem)
            if year >= 1:
                years.append(year)
        return tuple(years)

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
    ) -> None:
        """Compose a year-partition path and write through the datastore."""
        path = path_builder(exchange, market, symbol, timeframe, year)
        self._logger.debug(
            "Saving processed market dataset",
            extra=_dataset_log_extra(
                dataset=dataset,
                exchange=exchange,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
                rows=dataframe.height,
                columns=dataframe.width,
            ),
        )
        self._datastore.write(path, dataframe)
        self._logger.info(
            "Saved processed market dataset",
            extra=_dataset_log_extra(
                dataset=dataset,
                exchange=exchange,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
                rows=dataframe.height,
                columns=dataframe.width,
            ),
        )

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
        """Compose a year-partition path and read through the datastore."""
        path = path_builder(exchange, market, symbol, timeframe, year)
        self._logger.debug(
            "Loading processed market dataset",
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
            "Loaded processed market dataset",
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
    """Build structured log fields for a processed dataset operation."""
    payload: dict[str, object] = {
        "dataset": dataset,
        "tier": "processed",
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
