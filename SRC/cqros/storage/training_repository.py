"""CQROS training dataset repository.

Purpose:
    Provide a path-free facade for persisting and retrieving merged training
    year partitions produced by the Training Dataset pipeline.

Responsibilities:
    - Resolve storage locations for training year partitions via
      ``StorageLayout.training_path``
    - Persist, load, check existence, and delete training partitions
    - Discover symbols, timeframes, and year partitions under the training
      tier
    - Delegate all read and write operations to an injected ``IDataStore``
      (typically ``ParquetStore`` for atomic Parquet writes)
    - Keep filesystem paths out of the public API
    - Remain free of training dataset generation, joins, validation, and
      pipeline logic

Dependencies:
    ``polars``, ``cqros.core``, and ``cqros.storage`` layout/interfaces.

Public API:
    ``TrainingPartitionRef``, ``TrainingRepository``
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
    STORAGE_DIR_TRAINING,
)
from cqros.core.types import Exchange, Market, Symbol, Timeframe
from cqros.storage.interfaces import IDataStore
from cqros.storage.layout import StorageLayout

__all__ = [
    "TrainingPartitionRef",
    "TrainingRepository",
]

_logger = logging.getLogger(__name__)

_EXCHANGE: Final[Exchange] = EXCHANGE_BINANCE
_MARKET: Final[Market] = MARKET_USDT_PERPETUAL


@dataclass(frozen=True, slots=True)
class TrainingPartitionRef:
    """Identity of one discovered training year partition.

    Attributes:
        symbol: Tradeable symbol.
        timeframe: Training bar interval.
        year: Calendar year of the partition.
    """

    symbol: Symbol
    timeframe: Timeframe
    year: int


class TrainingRepository:
    """Repository facade for merged training datasets.

    Callers identify partitions by exchange, market, symbol, timeframe, and
    year. Paths are composed privately via ``StorageLayout.training_path`` and
    never returned. Persistence is delegated entirely to the injected
    ``IDataStore`` (commonly ``ParquetStore``) so atomic writes and alternate
    backends can be substituted without changing this API.

    Partition layout::

        training/{exchange}/{market}/{symbol}/{timeframe}/{year}.parquet

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

    def discover_symbols(
        self,
        *,
        exchange: Exchange = _EXCHANGE,
        market: Market = _MARKET,
    ) -> tuple[Symbol, ...]:
        """Return sorted symbols with at least one training partition.

        Args:
            exchange: Exchange identifier.
            market: Market segment.

        Returns:
            Deterministically ordered symbols discovered on disk.
        """
        base = self._training_root(exchange=exchange, market=market)
        if not base.is_dir():
            return ()
        return tuple(sorted(path.name for path in base.iterdir() if path.is_dir()))

    def discover_timeframes(
        self,
        *,
        symbol: Symbol,
        exchange: Exchange = _EXCHANGE,
        market: Market = _MARKET,
    ) -> tuple[Timeframe, ...]:
        """Return sorted timeframes with at least one partition for ``symbol``.

        Args:
            symbol: Tradeable symbol.
            exchange: Exchange identifier.
            market: Market segment.

        Returns:
            Deterministically ordered timeframes discovered on disk.
        """
        base = self._training_root(exchange=exchange, market=market) / symbol
        if not base.is_dir():
            return ()
        return tuple(sorted(path.name for path in base.iterdir() if path.is_dir()))

    def list_years(
        self,
        *,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
    ) -> tuple[int, ...]:
        """Return sorted calendar years present as training parquet partitions.

        Args:
            exchange: Exchange identifier.
            market: Market segment.
            symbol: Tradeable symbol.
            timeframe: Training bar interval.

        Returns:
            Deterministically ordered year values for existing partitions.
        """
        return self._discover_years(
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
        )

    def discover_partitions(
        self,
        *,
        symbols: Sequence[Symbol] | None = None,
        timeframes: Sequence[Timeframe] | None = None,
        exchange: Exchange = _EXCHANGE,
        market: Market = _MARKET,
    ) -> tuple[TrainingPartitionRef, ...]:
        """Discover training year partitions matching optional filters.

        Missing training trees are skipped. Only year partitions that exist as
        ``{year}.parquet`` files are included. Paths are never returned.

        Args:
            symbols: Optional symbol allowlist. ``None`` discovers every
                symbol present under the training tier.
            timeframes: Optional timeframe allowlist. ``None`` discovers every
                timeframe present for each symbol.
            exchange: Exchange identifier.
            market: Market segment.

        Returns:
            Deterministically ordered partition references.
        """
        symbol_filter = set(symbols) if symbols is not None else None
        timeframe_filter = set(timeframes) if timeframes is not None else None

        items: list[TrainingPartitionRef] = []
        for symbol in self.discover_symbols(exchange=exchange, market=market):
            if symbol_filter is not None and symbol not in symbol_filter:
                continue
            for timeframe in self.discover_timeframes(
                symbol=symbol,
                exchange=exchange,
                market=market,
            ):
                if timeframe_filter is not None and timeframe not in timeframe_filter:
                    continue
                for year in self._discover_years(
                    exchange=exchange,
                    market=market,
                    symbol=symbol,
                    timeframe=timeframe,
                ):
                    items.append(
                        TrainingPartitionRef(
                            symbol=symbol,
                            timeframe=timeframe,
                            year=year,
                        )
                    )

        return tuple(
            sorted(
                items,
                key=lambda item: (item.symbol, item.timeframe, item.year),
            )
        )

    def exists(
        self,
        *,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> bool:
        """Return whether a training year partition exists.

        Existence is determined solely through ``IDataStore.exists`` on the
        path composed by ``StorageLayout.training_path``. Parquet contents are
        never read, scanned, or validated.

        Args:
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Training bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            ``True`` when the yearly training partition exists; otherwise
            ``False``.
        """
        path = self._layout.training_path(exchange, market, symbol, timeframe, year)
        present = self._datastore.exists(path)
        self._logger.debug(
            "Training dataset exists" if present else "Training dataset does not exist",
            extra=_training_log_extra(
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
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> None:
        """Persist a training year partition.

        Args:
            dataframe: Training frame to store.
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Training bar interval (for example ``1h``).
            year: Calendar year of the partition.
        """
        path = self._layout.training_path(exchange, market, symbol, timeframe, year)
        self._logger.debug(
            "Saving training dataset",
            extra=_training_log_extra(
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
            "Saved training dataset",
            extra=_training_log_extra(
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
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> pl.DataFrame:
        """Load a training year partition.

        Args:
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Training bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            Eagerly loaded training DataFrame.
        """
        path = self._layout.training_path(exchange, market, symbol, timeframe, year)
        self._logger.debug(
            "Loading training dataset",
            extra=_training_log_extra(
                exchange=exchange,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
            ),
        )
        frame = self._datastore.read(path)
        self._logger.info(
            "Loaded training dataset",
            extra=_training_log_extra(
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
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> None:
        """Delete a training year partition.

        Args:
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Training bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Raises:
            DatasetNotFoundError: If the partition does not exist (propagated
                from the injected datastore).
        """
        path = self._layout.training_path(exchange, market, symbol, timeframe, year)
        self._logger.debug(
            "Deleting training dataset",
            extra=_training_log_extra(
                exchange=exchange,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
            ),
        )
        self._datastore.delete(path)
        self._logger.info(
            "Deleted training dataset",
            extra=_training_log_extra(
                exchange=exchange,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
            ),
        )

    def _training_root(
        self,
        *,
        exchange: Exchange,
        market: Market,
    ) -> Path:
        """Return the training directory for exchange/market."""
        return self._layout.root / STORAGE_DIR_TRAINING / exchange / market

    def _discover_years(
        self,
        *,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
    ) -> tuple[int, ...]:
        """Return sorted calendar years present as parquet partitions."""
        base = self._training_root(exchange=exchange, market=market) / symbol / timeframe
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


def _training_log_extra(
    *,
    exchange: Exchange,
    market: Market,
    symbol: Symbol,
    timeframe: Timeframe,
    year: int,
    rows: int | None = None,
    columns: int | None = None,
) -> dict[str, object]:
    """Build structured log fields for a training dataset operation."""
    payload: dict[str, object] = {
        "tier": "training",
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
