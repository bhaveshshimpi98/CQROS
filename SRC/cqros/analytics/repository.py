"""CQROS analytics metrics dataset repository.

Purpose:
    Provide a path-free facade for persisting and retrieving canonical
    analytics year partitions produced by the CQROS Analytics Engine.

Responsibilities:
    - Resolve storage locations for analytics year partitions via
      ``StorageLayout.analytics_path``
    - Persist, load, check existence, and discover analytics partitions
    - Validate frames against ``ANALYTICS_SCHEMA`` before save
    - Cast loaded frames to ``ANALYTICS_SCHEMA`` after read
    - Discover managers, symbols, timeframes, and year partitions under the
      analytics tier
    - Delegate all read and write operations to an injected ``IDataStore``
      (typically ``ParquetStore`` for atomic Parquet writes)
    - Keep filesystem paths out of the public API
    - Remain free of analytics math and trading logic

Dependencies:
    ``polars``, ``cqros.core``, ``cqros.analytics.exceptions``,
    ``cqros.analytics.schema``, and ``cqros.storage`` layout/interfaces.

Public API:
    ``AnalyticsPartitionRef``, ``AnalyticsRepository``

Notes:
    Schema validation and casting keep the on-disk contract aligned with
    ``ANALYTICS_SCHEMA``. Metric computation belongs to the engine and
    pipeline. This module must not import analytics engines for execution.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import polars as pl

from cqros.analytics.exceptions import AnalyticsValidationError
from cqros.analytics.schema import (
    ANALYTICS_COLUMNS,
    ANALYTICS_SCHEMA,
    REQUIRED_COLUMNS,
)
from cqros.core.constants import (
    EXCHANGE_BINANCE,
    FILE_EXTENSION_PARQUET,
    MARKET_USDT_PERPETUAL,
    STORAGE_DIR_ANALYTICS,
)
from cqros.core.types import Exchange, Market, Symbol, Timeframe
from cqros.storage.interfaces import IDataStore
from cqros.storage.layout import StorageLayout

__all__ = [
    "AnalyticsPartitionRef",
    "AnalyticsRepository",
]

_logger = logging.getLogger(__name__)

_EXCHANGE: Final[Exchange] = EXCHANGE_BINANCE
_MARKET: Final[Market] = MARKET_USDT_PERPETUAL

_ERROR_FRAME_TYPE: Final[str] = "ANA_REPO_FRAME_TYPE"
_ERROR_MISSING_COLUMNS: Final[str] = "ANA_REPO_MISSING_COLUMNS"
_ERROR_SCHEMA_CAST: Final[str] = "ANA_REPO_SCHEMA_CAST"


@dataclass(frozen=True, slots=True)
class AnalyticsPartitionRef:
    """Identity of one discovered analytics year partition.

    Attributes:
        manager: Order manager identifier.
        exchange: Exchange identifier.
        market: Market segment.
        symbol: Tradeable symbol.
        timeframe: Trade bar interval.
        year: Calendar year of the partition.
    """

    manager: str
    exchange: Exchange
    market: Market
    symbol: Symbol
    timeframe: Timeframe
    year: int


class AnalyticsRepository:
    """Repository facade for canonical analytics datasets.

    Callers identify partitions by manager, exchange, market, symbol,
    timeframe, and year. Paths are composed privately via
    ``StorageLayout.analytics_path`` and never returned. Persistence is
    delegated entirely to the injected ``IDataStore`` (commonly
    ``ParquetStore``) so atomic writes and alternate backends can be
    substituted without changing this API. Frames are validated and cast to
    ``ANALYTICS_SCHEMA`` on save and load.

    Partition layout::

        analytics/{manager}/{exchange}/{market}/{symbol}/{timeframe}/{year}.parquet

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
        """Return sorted manager identifiers with at least one analytics partition.

        Returns:
            Deterministically ordered manager identifiers discovered on disk.
        """
        base = self._analytics_root()
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
        """Return sorted symbols with at least one analytics partition for ``manager``.

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
        """Return sorted calendar years present as analytics parquet partitions.

        Args:
            manager: Order manager identifier.
            exchange: Exchange identifier.
            market: Market segment.
            symbol: Tradeable symbol.
            timeframe: Trade bar interval.

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

    def discover(
        self,
        *,
        managers: Sequence[str] | None = None,
        symbols: Sequence[Symbol] | None = None,
        timeframes: Sequence[Timeframe] | None = None,
        exchange: Exchange = _EXCHANGE,
        market: Market = _MARKET,
    ) -> tuple[AnalyticsPartitionRef, ...]:
        """Discover analytics year partitions matching optional filters.

        Missing analytics trees are skipped. Only year partitions that exist
        as ``{year}.parquet`` files are included. Paths are never returned.

        Args:
            managers: Optional manager allowlist. ``None`` discovers every
                manager present under the analytics tier.
            symbols: Optional symbol allowlist. ``None`` discovers every
                symbol present under each manager.
            timeframes: Optional timeframe allowlist. ``None`` discovers every
                timeframe present for each symbol.
            exchange: Exchange identifier.
            market: Market segment.

        Returns:
            Deterministically ordered partition references.
        """
        return self.discover_partitions(
            managers=managers,
            symbols=symbols,
            timeframes=timeframes,
            exchange=exchange,
            market=market,
        )

    def discover_partitions(
        self,
        *,
        managers: Sequence[str] | None = None,
        symbols: Sequence[Symbol] | None = None,
        timeframes: Sequence[Timeframe] | None = None,
        exchange: Exchange = _EXCHANGE,
        market: Market = _MARKET,
    ) -> tuple[AnalyticsPartitionRef, ...]:
        """Discover analytics year partitions matching optional filters.

        Args:
            managers: Optional manager allowlist. ``None`` discovers every
                manager present under the analytics tier.
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

        items: list[AnalyticsPartitionRef] = []
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
                            AnalyticsPartitionRef(
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
        """Return whether an analytics year partition exists.

        Existence is determined solely through ``IDataStore.exists`` on the
        path composed by ``StorageLayout.analytics_path``. Parquet contents
        are never read, scanned, or validated.

        Args:
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            ``True`` when the yearly analytics partition exists; otherwise
            ``False``.
        """
        path = self._layout.analytics_path(
            manager,
            exchange,
            market,
            symbol,
            timeframe,
            year,
        )
        present = self._datastore.exists(path)
        self._logger.debug(
            "Analytics dataset exists" if present else "Analytics dataset does not exist",
            extra=_analytics_log_extra(
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
        """Persist an analytics year partition after schema validation.

        Args:
            dataframe: Analytics frame to store. Must conform to
                ``ANALYTICS_SCHEMA``.
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Raises:
            AnalyticsValidationError: If ``dataframe`` is not a DataFrame,
                required columns are missing, or casting to
                ``ANALYTICS_SCHEMA`` fails.
        """
        validated = _require_analytics_schema(dataframe)
        path = self._layout.analytics_path(
            manager,
            exchange,
            market,
            symbol,
            timeframe,
            year,
        )
        self._logger.debug(
            "Saving analytics dataset",
            extra=_analytics_log_extra(
                manager=manager,
                exchange=exchange,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
                rows=validated.height,
                columns=validated.width,
            ),
        )
        self._datastore.write(path, validated)
        self._logger.info(
            "Saved analytics dataset",
            extra=_analytics_log_extra(
                manager=manager,
                exchange=exchange,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
                rows=validated.height,
                columns=validated.width,
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
        """Load an analytics year partition cast to ``ANALYTICS_SCHEMA``.

        Args:
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            Eagerly loaded analytics DataFrame cast to ``ANALYTICS_SCHEMA``.

        Raises:
            AnalyticsValidationError: If the loaded frame fails schema
                validation or casting.
        """
        path = self._layout.analytics_path(
            manager,
            exchange,
            market,
            symbol,
            timeframe,
            year,
        )
        self._logger.debug(
            "Loading analytics dataset",
            extra=_analytics_log_extra(
                manager=manager,
                exchange=exchange,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
            ),
        )
        frame = self._datastore.read(path)
        validated = _require_analytics_schema(frame)
        self._logger.info(
            "Loaded analytics dataset",
            extra=_analytics_log_extra(
                manager=manager,
                exchange=exchange,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
                rows=validated.height,
                columns=validated.width,
            ),
        )
        return validated

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
        """Delete an analytics year partition.

        Args:
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Raises:
            DatasetNotFoundError: If the partition does not exist (propagated
                from the injected datastore).
        """
        path = self._layout.analytics_path(
            manager,
            exchange,
            market,
            symbol,
            timeframe,
            year,
        )
        self._logger.debug(
            "Deleting analytics dataset",
            extra=_analytics_log_extra(
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
            "Deleted analytics dataset",
            extra=_analytics_log_extra(
                manager=manager,
                exchange=exchange,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
            ),
        )

    def _analytics_root(self) -> Path:
        """Return the analytics directory under the storage root."""
        return self._layout.root / STORAGE_DIR_ANALYTICS

    def _manager_root(
        self,
        *,
        manager: str,
        exchange: Exchange,
        market: Market,
    ) -> Path:
        """Return the analytics directory for manager/exchange/market."""
        return self._analytics_root() / manager / exchange / market

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


def _require_analytics_schema(frame: object) -> pl.DataFrame:
    """Validate and cast ``frame`` to ``ANALYTICS_SCHEMA``."""
    if not isinstance(frame, pl.DataFrame):
        raise AnalyticsValidationError(
            "analytics frame must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={"actual_type": type(frame).__name__},
        )
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise AnalyticsValidationError(
            "analytics schema is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "missing_columns": tuple(missing),
                "required_columns": REQUIRED_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )
    try:
        return frame.select(list(ANALYTICS_COLUMNS)).cast(ANALYTICS_SCHEMA)
    except (pl.exceptions.PolarsError, TypeError, ValueError) as exc:
        raise AnalyticsValidationError(
            "analytics frame failed ANALYTICS_SCHEMA cast",
            error_code=_ERROR_SCHEMA_CAST,
            details={"reason": str(exc)},
        ) from exc


def _analytics_log_extra(
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
    """Build structured log fields for an analytics dataset operation."""
    payload: dict[str, object] = {
        "tier": "analytics",
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
