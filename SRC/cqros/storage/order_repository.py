"""CQROS OMS order dataset repository.

Purpose:
    Provide a path-free facade for persisting and retrieving canonical OMS
    order year partitions produced by CQROS order management.

Responsibilities:
    - Resolve storage locations for order year partitions via
      ``StorageLayout.order_path``
    - Persist, load, check existence, and delete order partitions
    - Discover managers, symbols, timeframes, and year partitions under the
      orders tier
    - Delegate all read and write operations to an injected ``IDataStore``
      (typically ``ParquetStore`` for atomic Parquet writes)
    - Keep filesystem paths out of the public API
    - Remain free of order generation, schema validation, and trading logic

Dependencies:
    ``polars``, ``cqros.core``, and ``cqros.storage`` layout/interfaces.

Public API:
    ``OrderPartitionRef``, ``OrderRepository``

Notes:
    Frames are persisted and loaded exactly as provided. Canonical Order
    schema validation belongs to ``OrderPipeline`` and future OMS
    verification. This module must not import ``cqros.oms``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import polars as pl

from cqros.core.constants import (
    EXCHANGE_BINANCE,
    FILE_EXTENSION_PARQUET,
    MARKET_USDT_PERPETUAL,
    STORAGE_DIR_ORDERS,
)
from cqros.core.types import Exchange, Market, Symbol, Timeframe
from cqros.storage.interfaces import IDataStore
from cqros.storage.layout import StorageLayout

__all__ = [
    "OrderPartitionRef",
    "OrderRepository",
]

_logger = logging.getLogger(__name__)

_EXCHANGE: Final[Exchange] = EXCHANGE_BINANCE
_MARKET: Final[Market] = MARKET_USDT_PERPETUAL


@dataclass(frozen=True, slots=True)
class OrderPartitionRef:
    """Identity of one discovered OMS order year partition.

    Attributes:
        manager: Order manager identifier.
        exchange: Exchange identifier.
        market: Market segment.
        symbol: Tradeable symbol.
        timeframe: Order bar interval.
        year: Calendar year of the partition.
    """

    manager: str
    exchange: Exchange
    market: Market
    symbol: Symbol
    timeframe: Timeframe
    year: int


class OrderRepository:
    """Repository facade for canonical OMS order datasets.

    Callers identify partitions by manager, exchange, market, symbol,
    timeframe, and year. Paths are composed privately via
    ``StorageLayout.order_path`` and never returned. Persistence is delegated
    entirely to the injected ``IDataStore`` (commonly ``ParquetStore``) so
    atomic writes and alternate backends can be substituted without changing
    this API. Frames are stored exactly as provided; schema validation is not
    performed here.

    Partition layout::

        orders/{manager}/{exchange}/{market}/{symbol}/{timeframe}/{year}.parquet

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

    def discover_managers(self) -> tuple[str, ...]:
        """Return sorted manager identifiers with at least one order partition.

        Returns:
            Deterministically ordered manager identifiers discovered on disk.
        """
        base = self._orders_root()
        if not base.is_dir():
            return ()
        return tuple(sorted(path.name for path in base.iterdir() if path.is_dir()))

    def discover_symbols(
        self,
        *,
        manager: str,
        exchange: Exchange = _EXCHANGE,
        market: Market = _MARKET,
    ) -> tuple[Symbol, ...]:
        """Return sorted symbols with at least one order partition for ``manager``.

        Args:
            manager: Order manager identifier.
            exchange: Exchange identifier.
            market: Market segment.

        Returns:
            Deterministically ordered symbols discovered on disk.
        """
        base = self._manager_root(manager=manager, exchange=exchange, market=market)
        if not base.is_dir():
            return ()
        return tuple(sorted(path.name for path in base.iterdir() if path.is_dir()))

    def discover_timeframes(
        self,
        *,
        manager: str,
        symbol: Symbol,
        exchange: Exchange = _EXCHANGE,
        market: Market = _MARKET,
    ) -> tuple[Timeframe, ...]:
        """Return sorted timeframes with at least one partition for ``symbol``.

        Args:
            manager: Order manager identifier.
            symbol: Tradeable symbol.
            exchange: Exchange identifier.
            market: Market segment.

        Returns:
            Deterministically ordered timeframes discovered on disk.
        """
        base = self._manager_root(manager=manager, exchange=exchange, market=market) / symbol
        if not base.is_dir():
            return ()
        return tuple(sorted(path.name for path in base.iterdir() if path.is_dir()))

    def list_years(
        self,
        *,
        manager: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
    ) -> tuple[int, ...]:
        """Return sorted calendar years present as order parquet partitions.

        Args:
            manager: Order manager identifier.
            exchange: Exchange identifier.
            market: Market segment.
            symbol: Tradeable symbol.
            timeframe: Order bar interval.

        Returns:
            Deterministically ordered year values for existing partitions.
        """
        return self._discover_years(
            manager=manager,
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
        )

    def discover_partitions(
        self,
        *,
        managers: Sequence[str] | None = None,
        symbols: Sequence[Symbol] | None = None,
        timeframes: Sequence[Timeframe] | None = None,
        exchange: Exchange = _EXCHANGE,
        market: Market = _MARKET,
    ) -> tuple[OrderPartitionRef, ...]:
        """Discover order year partitions matching optional filters.

        Missing order trees are skipped. Only year partitions that exist as
        ``{year}.parquet`` files are included. Paths are never returned.

        Args:
            managers: Optional manager allowlist. ``None`` discovers every
                manager present under the orders tier.
            symbols: Optional symbol allowlist. ``None`` discovers every
                symbol present under each manager.
            timeframes: Optional timeframe allowlist. ``None`` discovers every
                timeframe present for each symbol.
            exchange: Exchange identifier.
            market: Market segment.

        Returns:
            Deterministically ordered partition references.
        """
        manager_filter = set(managers) if managers is not None else None
        symbol_filter = set(symbols) if symbols is not None else None
        timeframe_filter = set(timeframes) if timeframes is not None else None

        items: list[OrderPartitionRef] = []
        for manager in self.discover_managers():
            if manager_filter is not None and manager not in manager_filter:
                continue
            for symbol in self.discover_symbols(
                manager=manager,
                exchange=exchange,
                market=market,
            ):
                if symbol_filter is not None and symbol not in symbol_filter:
                    continue
                for timeframe in self.discover_timeframes(
                    manager=manager,
                    symbol=symbol,
                    exchange=exchange,
                    market=market,
                ):
                    if timeframe_filter is not None and timeframe not in timeframe_filter:
                        continue
                    for year in self._discover_years(
                        manager=manager,
                        exchange=exchange,
                        market=market,
                        symbol=symbol,
                        timeframe=timeframe,
                    ):
                        items.append(
                            OrderPartitionRef(
                                manager=manager,
                                exchange=exchange,
                                market=market,
                                symbol=symbol,
                                timeframe=timeframe,
                                year=year,
                            )
                        )

        return tuple(
            sorted(
                items,
                key=lambda item: (
                    item.manager,
                    item.exchange,
                    item.market,
                    item.symbol,
                    item.timeframe,
                    item.year,
                ),
            )
        )

    def exists(
        self,
        *,
        manager: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> bool:
        """Return whether an order year partition exists.

        Existence is determined solely through ``IDataStore.exists`` on the
        path composed by ``StorageLayout.order_path``. Parquet contents are
        never read, scanned, or validated.

        Args:
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Order bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            ``True`` when the yearly order partition exists; otherwise
            ``False``.
        """
        path = self._layout.order_path(
            manager,
            exchange,
            market,
            symbol,
            timeframe,
            year,
        )
        present = self._datastore.exists(path)
        self._logger.debug(
            "Order dataset exists" if present else "Order dataset does not exist",
            extra=_order_log_extra(
                manager=manager,
                exchange=exchange,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
            ),
        )
        return present

    def save(
        self,
        dataframe: pl.DataFrame,
        *,
        manager: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> None:
        """Persist an order year partition.

        Args:
            dataframe: Order frame to store.
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Order bar interval (for example ``1h``).
            year: Calendar year of the partition.
        """
        path = self._layout.order_path(
            manager,
            exchange,
            market,
            symbol,
            timeframe,
            year,
        )
        self._logger.debug(
            "Saving order dataset",
            extra=_order_log_extra(
                manager=manager,
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
            "Saved order dataset",
            extra=_order_log_extra(
                manager=manager,
                exchange=exchange,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
                rows=dataframe.height,
                columns=dataframe.width,
            ),
        )

    def load(
        self,
        *,
        manager: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> pl.DataFrame:
        """Load an order year partition.

        Args:
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Order bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            Eagerly loaded order DataFrame.
        """
        path = self._layout.order_path(
            manager,
            exchange,
            market,
            symbol,
            timeframe,
            year,
        )
        self._logger.debug(
            "Loading order dataset",
            extra=_order_log_extra(
                manager=manager,
                exchange=exchange,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
            ),
        )
        frame = self._datastore.read(path)
        self._logger.info(
            "Loaded order dataset",
            extra=_order_log_extra(
                manager=manager,
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

    def delete(
        self,
        *,
        manager: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> None:
        """Delete an order year partition.

        Args:
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Order bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Raises:
            DatasetNotFoundError: If the partition does not exist (propagated
                from the injected datastore).
        """
        path = self._layout.order_path(
            manager,
            exchange,
            market,
            symbol,
            timeframe,
            year,
        )
        self._logger.debug(
            "Deleting order dataset",
            extra=_order_log_extra(
                manager=manager,
                exchange=exchange,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
            ),
        )
        self._datastore.delete(path)
        self._logger.info(
            "Deleted order dataset",
            extra=_order_log_extra(
                manager=manager,
                exchange=exchange,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
            ),
        )

    def _orders_root(self) -> Path:
        """Return the orders directory under the storage root."""
        return self._layout.root / STORAGE_DIR_ORDERS

    def _manager_root(
        self,
        *,
        manager: str,
        exchange: Exchange,
        market: Market,
    ) -> Path:
        """Return the orders directory for manager/exchange/market."""
        return self._orders_root() / manager / exchange / market

    def _discover_years(
        self,
        *,
        manager: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
    ) -> tuple[int, ...]:
        """Return sorted calendar years present as parquet partitions."""
        base = (
            self._manager_root(manager=manager, exchange=exchange, market=market)
            / symbol
            / timeframe
        )
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


def _order_log_extra(
    *,
    manager: str,
    exchange: Exchange,
    market: Market,
    symbol: Symbol,
    timeframe: Timeframe,
    year: int,
    rows: int | None = None,
    columns: int | None = None,
) -> dict[str, object]:
    """Build structured log fields for an order dataset operation."""
    payload: dict[str, object] = {
        "tier": "orders",
        "manager": manager,
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
