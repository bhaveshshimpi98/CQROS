"""CQROS risk-decision dataset repository.

Purpose:
    Provide a path-free facade for persisting and retrieving canonical risk
    decision year partitions produced by CQROS risk evaluation.

Responsibilities:
    - Resolve storage locations for risk year partitions via
      ``StorageLayout.risk_path``
    - Persist, load, check existence, and delete risk partitions
    - Discover policies, symbols, timeframes, and year partitions under the
      risks tier
    - Delegate all read and write operations to an injected ``IDataStore``
      (typically ``ParquetStore`` for atomic Parquet writes)
    - Keep filesystem paths out of the public API
    - Remain free of risk calculations, schema validation, and trading logic

Dependencies:
    ``polars``, ``cqros.core``, and ``cqros.storage`` layout/interfaces.

Public API:
    ``RiskPartitionRef``, ``RiskRepository``

Notes:
    Frames are persisted and loaded exactly as provided. Canonical Risk
    Decision schema validation belongs to ``RiskPipeline`` and future risk
    verification. This module must not import ``cqros.risk``.
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
    STORAGE_DIR_RISKS,
)
from cqros.core.types import Exchange, Market, Symbol, Timeframe
from cqros.storage.interfaces import IDataStore
from cqros.storage.layout import StorageLayout

__all__ = [
    "RiskPartitionRef",
    "RiskRepository",
]

_logger = logging.getLogger(__name__)

_EXCHANGE: Final[Exchange] = EXCHANGE_BINANCE
_MARKET: Final[Market] = MARKET_USDT_PERPETUAL


@dataclass(frozen=True, slots=True)
class RiskPartitionRef:
    """Identity of one discovered risk-decision year partition.

    Attributes:
        policy: Risk policy identifier.
        exchange: Exchange identifier.
        market: Market segment.
        symbol: Tradeable symbol.
        timeframe: Risk-decision bar interval.
        year: Calendar year of the partition.
    """

    policy: str
    exchange: Exchange
    market: Market
    symbol: Symbol
    timeframe: Timeframe
    year: int


class RiskRepository:
    """Repository facade for canonical risk-decision datasets.

    Callers identify partitions by policy, exchange, market, symbol,
    timeframe, and year. Paths are composed privately via
    ``StorageLayout.risk_path`` and never returned. Persistence is delegated
    entirely to the injected ``IDataStore`` (commonly ``ParquetStore``) so
    atomic writes and alternate backends can be substituted without changing
    this API. Frames are stored exactly as provided; schema validation is not
    performed here.

    Partition layout::

        risks/{policy}/{exchange}/{market}/{symbol}/{timeframe}/{year}.parquet

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

    def discover_policies(self) -> tuple[str, ...]:
        """Return sorted policy identifiers with at least one risk partition.

        Returns:
            Deterministically ordered policy identifiers discovered on disk.
        """
        base = self._risks_root()
        if not base.is_dir():
            return ()
        return tuple(sorted(path.name for path in base.iterdir() if path.is_dir()))

    def discover_symbols(
        self,
        *,
        policy: str,
        exchange: Exchange = _EXCHANGE,
        market: Market = _MARKET,
    ) -> tuple[Symbol, ...]:
        """Return sorted symbols with at least one risk partition for ``policy``.

        Args:
            policy: Risk policy identifier.
            exchange: Exchange identifier.
            market: Market segment.

        Returns:
            Deterministically ordered symbols discovered on disk.
        """
        base = self._policy_root(policy=policy, exchange=exchange, market=market)
        if not base.is_dir():
            return ()
        return tuple(sorted(path.name for path in base.iterdir() if path.is_dir()))

    def discover_timeframes(
        self,
        *,
        policy: str,
        symbol: Symbol,
        exchange: Exchange = _EXCHANGE,
        market: Market = _MARKET,
    ) -> tuple[Timeframe, ...]:
        """Return sorted timeframes with at least one partition for ``symbol``.

        Args:
            policy: Risk policy identifier.
            symbol: Tradeable symbol.
            exchange: Exchange identifier.
            market: Market segment.

        Returns:
            Deterministically ordered timeframes discovered on disk.
        """
        base = self._policy_root(policy=policy, exchange=exchange, market=market) / symbol
        if not base.is_dir():
            return ()
        return tuple(sorted(path.name for path in base.iterdir() if path.is_dir()))

    def list_years(
        self,
        *,
        policy: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
    ) -> tuple[int, ...]:
        """Return sorted calendar years present as risk parquet partitions.

        Args:
            policy: Risk policy identifier.
            exchange: Exchange identifier.
            market: Market segment.
            symbol: Tradeable symbol.
            timeframe: Risk-decision bar interval.

        Returns:
            Deterministically ordered year values for existing partitions.
        """
        return self._discover_years(
            policy=policy,
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
        )

    def discover_partitions(
        self,
        *,
        policies: Sequence[str] | None = None,
        symbols: Sequence[Symbol] | None = None,
        timeframes: Sequence[Timeframe] | None = None,
        exchange: Exchange = _EXCHANGE,
        market: Market = _MARKET,
    ) -> tuple[RiskPartitionRef, ...]:
        """Discover risk year partitions matching optional filters.

        Missing risk trees are skipped. Only year partitions that exist as
        ``{year}.parquet`` files are included. Paths are never returned.

        Args:
            policies: Optional policy allowlist. ``None`` discovers every
                policy present under the risks tier.
            symbols: Optional symbol allowlist. ``None`` discovers every
                symbol present under each policy.
            timeframes: Optional timeframe allowlist. ``None`` discovers every
                timeframe present for each symbol.
            exchange: Exchange identifier.
            market: Market segment.

        Returns:
            Deterministically ordered partition references.
        """
        policy_filter = set(policies) if policies is not None else None
        symbol_filter = set(symbols) if symbols is not None else None
        timeframe_filter = set(timeframes) if timeframes is not None else None

        items: list[RiskPartitionRef] = []
        for policy in self.discover_policies():
            if policy_filter is not None and policy not in policy_filter:
                continue
            for symbol in self.discover_symbols(
                policy=policy,
                exchange=exchange,
                market=market,
            ):
                if symbol_filter is not None and symbol not in symbol_filter:
                    continue
                for timeframe in self.discover_timeframes(
                    policy=policy,
                    symbol=symbol,
                    exchange=exchange,
                    market=market,
                ):
                    if timeframe_filter is not None and timeframe not in timeframe_filter:
                        continue
                    for year in self._discover_years(
                        policy=policy,
                        exchange=exchange,
                        market=market,
                        symbol=symbol,
                        timeframe=timeframe,
                    ):
                        items.append(
                            RiskPartitionRef(
                                policy=policy,
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
                    item.policy,
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
        policy: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> bool:
        """Return whether a risk year partition exists.

        Existence is determined solely through ``IDataStore.exists`` on the
        path composed by ``StorageLayout.risk_path``. Parquet contents are
        never read, scanned, or validated.

        Args:
            policy: Risk policy identifier (for example ``fixed_risk``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Risk-decision bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            ``True`` when the yearly risk partition exists; otherwise
            ``False``.
        """
        path = self._layout.risk_path(
            policy,
            exchange,
            market,
            symbol,
            timeframe,
            year,
        )
        present = self._datastore.exists(path)
        self._logger.debug(
            "Risk dataset exists" if present else "Risk dataset does not exist",
            extra=_risk_log_extra(
                policy=policy,
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
        policy: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> None:
        """Persist a risk year partition.

        Args:
            dataframe: Risk-decision frame to store.
            policy: Risk policy identifier (for example ``fixed_risk``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Risk-decision bar interval (for example ``1h``).
            year: Calendar year of the partition.
        """
        path = self._layout.risk_path(
            policy,
            exchange,
            market,
            symbol,
            timeframe,
            year,
        )
        self._logger.debug(
            "Saving risk dataset",
            extra=_risk_log_extra(
                policy=policy,
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
            "Saved risk dataset",
            extra=_risk_log_extra(
                policy=policy,
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
        policy: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> pl.DataFrame:
        """Load a risk year partition.

        Args:
            policy: Risk policy identifier (for example ``fixed_risk``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Risk-decision bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            Eagerly loaded risk-decision DataFrame.
        """
        path = self._layout.risk_path(
            policy,
            exchange,
            market,
            symbol,
            timeframe,
            year,
        )
        self._logger.debug(
            "Loading risk dataset",
            extra=_risk_log_extra(
                policy=policy,
                exchange=exchange,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
            ),
        )
        frame = self._datastore.read(path)
        self._logger.info(
            "Loaded risk dataset",
            extra=_risk_log_extra(
                policy=policy,
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
        policy: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> None:
        """Delete a risk year partition.

        Args:
            policy: Risk policy identifier (for example ``fixed_risk``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Risk-decision bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Raises:
            DatasetNotFoundError: If the partition does not exist (propagated
                from the injected datastore).
        """
        path = self._layout.risk_path(
            policy,
            exchange,
            market,
            symbol,
            timeframe,
            year,
        )
        self._logger.debug(
            "Deleting risk dataset",
            extra=_risk_log_extra(
                policy=policy,
                exchange=exchange,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
            ),
        )
        self._datastore.delete(path)
        self._logger.info(
            "Deleted risk dataset",
            extra=_risk_log_extra(
                policy=policy,
                exchange=exchange,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
            ),
        )

    def _risks_root(self) -> Path:
        """Return the risks directory under the storage root."""
        return self._layout.root / STORAGE_DIR_RISKS

    def _policy_root(
        self,
        *,
        policy: str,
        exchange: Exchange,
        market: Market,
    ) -> Path:
        """Return the risks directory for policy/exchange/market."""
        return self._risks_root() / policy / exchange / market

    def _discover_years(
        self,
        *,
        policy: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
    ) -> tuple[int, ...]:
        """Return sorted calendar years present as parquet partitions."""
        base = (
            self._policy_root(policy=policy, exchange=exchange, market=market) / symbol / timeframe
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


def _risk_log_extra(
    *,
    policy: str,
    exchange: Exchange,
    market: Market,
    symbol: Symbol,
    timeframe: Timeframe,
    year: int,
    rows: int | None = None,
    columns: int | None = None,
) -> dict[str, object]:
    """Build structured log fields for a risk dataset operation."""
    payload: dict[str, object] = {
        "tier": "risks",
        "policy": policy,
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
