"""CQROS regime dataset repository.

Purpose:
    Provide a path-free facade for persisting and retrieving canonical
    regime year partitions produced by the CQROS Regime Engine.

Responsibilities:
    - Resolve storage locations for regime year partitions via
      ``StorageLayout.regime_path``
    - Persist, load, check existence, and discover regime partitions
    - Validate frames against ``REGIME_SCHEMA`` before save
    - Cast loaded frames to ``REGIME_SCHEMA`` after read
    - Discover managers, symbols, timeframes, and year partitions under the
      regime tier
    - Delegate all read and write operations to an injected ``IDataStore``
      (typically ``ParquetStore`` for atomic Parquet writes)
    - Keep filesystem paths out of the public API
    - Remain free of regime math and trading logic

Dependencies:
    ``polars``, ``cqros.core``, ``cqros.regime.exceptions``,
    ``cqros.regime.schema``, and ``cqros.storage`` layout/interfaces.

Public API:
    ``RegimePartitionRef``, ``RegimeRepository``

Notes:
    Schema validation and casting keep the on-disk contract aligned with
    ``REGIME_SCHEMA``. Regime generation belongs to the engine and pipeline.
    This module must not import regime engines for execution.
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
    STORAGE_DIR_REGIME,
)
from cqros.core.types import Exchange, Market, Symbol, Timeframe
from cqros.regime.exceptions import RegimeError
from cqros.regime.schema import REGIME_COLUMNS, REGIME_SCHEMA, REQUIRED_COLUMNS
from cqros.storage.interfaces import IDataStore
from cqros.storage.layout import StorageLayout

__all__ = [
    "RegimePartitionRef",
    "RegimeRepository",
]

_logger = logging.getLogger(__name__)

_EXCHANGE: Final[Exchange] = EXCHANGE_BINANCE
_MARKET: Final[Market] = MARKET_USDT_PERPETUAL

_ERROR_FRAME_TYPE: Final[str] = "REGIME_REPO_FRAME_TYPE"
_ERROR_MISSING_COLUMNS: Final[str] = "REGIME_REPO_MISSING_COLUMNS"
_ERROR_SCHEMA_CAST: Final[str] = "REGIME_REPO_SCHEMA_CAST"


@dataclass(frozen=True, slots=True)
class RegimePartitionRef:
    """Identity of one discovered regime year partition.

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


class RegimeRepository:
    """Repository facade for canonical regime datasets.

    Callers identify partitions by manager, exchange, market, symbol,
    timeframe, and year. Paths are composed privately via
    ``StorageLayout.regime_path`` and never returned. Persistence is
    delegated entirely to the injected ``IDataStore`` (commonly
    ``ParquetStore``) so atomic writes and alternate backends can be
    substituted without changing this API. Frames are validated and cast to
    ``REGIME_SCHEMA`` on save and load.

    Partition layout::

        regime/{manager}/{exchange}/{market}/{symbol}/{timeframe}/{year}.parquet

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
        """Return sorted manager identifiers with at least one partition.

        Returns:
            Deterministically ordered manager identifiers discovered on disk.
        """
        base = self._regime_root()
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
        """Return sorted symbols with at least one partition for ``manager``.

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
        """Return sorted calendar years present as regime partitions.

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
    ) -> tuple[RegimePartitionRef, ...]:
        """Discover regime year partitions matching optional filters.

        Missing regime trees are skipped. Only year partitions that exist as
        ``{year}.parquet`` files are included. Paths are never returned.

        Args:
            managers: Optional manager allowlist. ``None`` discovers every
                manager present under the regime tier.
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
    ) -> tuple[RegimePartitionRef, ...]:
        """Discover regime year partitions matching optional filters.

        Args:
            managers: Optional manager allowlist. ``None`` discovers every
                manager present under the regime tier.
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

        items: list[RegimePartitionRef] = []
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
                            RegimePartitionRef(
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
        """Return whether a regime year partition exists.

        Existence is determined solely through ``IDataStore.exists`` on the
        path composed by ``StorageLayout.regime_path``. Parquet contents are
        never read, scanned, or validated.

        Args:
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            ``True`` when the yearly regime partition exists; otherwise
            ``False``.
        """
        path = self._layout.regime_path(
            manager,
            exchange,
            market,
            symbol,
            timeframe,
            year,
        )
        present = self._datastore.exists(path)
        self._logger.debug(
            ("Regime dataset exists" if present else "Regime dataset does not exist"),
            extra=_regime_log_extra(
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
        """Persist a regime year partition after schema validation.

        Args:
            dataframe: Regime frame to store. Must conform to ``REGIME_SCHEMA``.
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Raises:
            RegimeError: If ``dataframe`` is not a DataFrame, required columns
                are missing, or casting to ``REGIME_SCHEMA`` fails.
        """
        validated = _require_regime_schema(dataframe)
        path = self._layout.regime_path(
            manager,
            exchange,
            market,
            symbol,
            timeframe,
            year,
        )
        self._logger.debug(
            "Saving regime dataset",
            extra=_regime_log_extra(
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
            "Saved regime dataset",
            extra=_regime_log_extra(
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
        """Load a regime partition cast to the canonical schema.

        When the partition does not exist, returns an empty DataFrame with
        ``REGIME_SCHEMA``. Never returns ``None``.

        Args:
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            Eagerly loaded regime DataFrame cast to ``REGIME_SCHEMA``, or an
            empty schema-conformant frame when the partition is absent.

        Raises:
            RegimeError: If the loaded frame fails schema validation or
                casting.
        """
        path = self._layout.regime_path(
            manager,
            exchange,
            market,
            symbol,
            timeframe,
            year,
        )
        self._logger.debug(
            "Loading regime dataset",
            extra=_regime_log_extra(
                manager=manager,
                exchange=exchange,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
            ),
        )
        if not self._datastore.exists(path):
            empty = pl.DataFrame(schema=REGIME_SCHEMA)
            self._logger.info(
                "Regime dataset missing; returning empty frame",
                extra=_regime_log_extra(
                    manager=manager,
                    exchange=exchange,
                    market=market,
                    symbol=symbol,
                    timeframe=timeframe,
                    year=year,
                    rows=empty.height,
                    columns=empty.width,
                ),
            )
            return empty
        frame = self._datastore.read(path)
        validated = _require_regime_schema(frame)
        self._logger.info(
            "Loaded regime dataset",
            extra=_regime_log_extra(
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
        """Delete a regime year partition.

        Silently succeeds when the partition does not exist.

        Args:
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.
        """
        path = self._layout.regime_path(
            manager,
            exchange,
            market,
            symbol,
            timeframe,
            year,
        )
        self._logger.debug(
            "Deleting regime dataset",
            extra=_regime_log_extra(
                manager=manager,
                exchange=exchange,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
            ),
        )
        if not self._datastore.exists(path):
            self._logger.debug(
                "Regime dataset already absent",
                extra=_regime_log_extra(
                    manager=manager,
                    exchange=exchange,
                    market=market,
                    symbol=symbol,
                    timeframe=timeframe,
                    year=year,
                ),
            )
            return
        self._datastore.delete(path)
        self._logger.info(
            "Deleted regime dataset",
            extra=_regime_log_extra(
                manager=manager,
                exchange=exchange,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
            ),
        )

    def _regime_root(self) -> Path:
        """Return the regime directory under the storage root."""
        return self._layout.root / STORAGE_DIR_REGIME

    def _manager_root(
        self,
        *,
        manager: str,
        exchange: Exchange,
        market: Market,
    ) -> Path:
        """Return the regime directory for manager/exchange/market."""
        return self._regime_root() / manager / exchange / market

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


def _require_regime_schema(frame: object) -> pl.DataFrame:
    """Validate and cast ``frame`` to ``REGIME_SCHEMA``."""
    if not isinstance(frame, pl.DataFrame):
        raise RegimeError(
            "regime frame must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={"actual_type": type(frame).__name__},
        )
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise RegimeError(
            "regime schema is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "missing_columns": tuple(missing),
                "required_columns": REQUIRED_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )
    try:
        return frame.select(list(REGIME_COLUMNS)).cast(REGIME_SCHEMA)
    except (pl.exceptions.PolarsError, TypeError, ValueError) as exc:
        raise RegimeError(
            "regime frame failed REGIME_SCHEMA cast",
            error_code=_ERROR_SCHEMA_CAST,
            details={"reason": str(exc)},
        ) from exc


def _regime_log_extra(
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
    """Build structured log fields for a regime dataset operation."""
    payload: dict[str, object] = {
        "tier": "regime",
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
