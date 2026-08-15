"""CQROS purged-CV metrics dataset repository.

Purpose:
    Provide a path-free facade for persisting and retrieving canonical
    purged-CV year partitions produced by the CQROS Purged-CV Engine.

Responsibilities:
    - Resolve storage locations for purged-CV year partitions via
      ``StorageLayout.purged_cv_path``
    - Persist, load, check existence, and discover purged-CV
      partitions
    - Validate frames against ``PURGED_CV_SCHEMA`` before save
    - Cast loaded frames to ``PURGED_CV_SCHEMA`` after read
    - Discover managers, timeframes, and year partitions under the
      purged-CV tier
    - Delegate all read and write operations to an injected ``IDataStore``
      (typically ``ParquetStore`` for atomic Parquet writes)
    - Keep filesystem paths out of the public API
    - Remain free of purged-CV math and trading logic

Dependencies:
    ``polars``, ``cqros.core``, ``cqros.purged_cv.exceptions``,
    ``cqros.purged_cv.schema``, and ``cqros.storage``
    layout/interfaces.

Public API:
    ``PurgedCVPartitionRef``, ``PurgedCVRepository``

Notes:
    Schema validation and casting keep the on-disk contract aligned with
    ``PURGED_CV_SCHEMA``. Purged-CV computation belongs to the engine
    and pipeline. This module must not import purged-CV engines for
    execution.

    Partition identity is cross-sectional::

        purged_cv/{manager}/{exchange}/{market}/{timeframe}/{year}.parquet
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
    STORAGE_DIR_PURGED_CV,
)
from cqros.core.types import Exchange, Market, Timeframe
from cqros.purged_cv.exceptions import PurgedCVError
from cqros.purged_cv.schema import (
    PURGED_CV_COLUMNS,
    PURGED_CV_SCHEMA,
    REQUIRED_COLUMNS,
)
from cqros.storage.interfaces import IDataStore
from cqros.storage.layout import StorageLayout

__all__ = [
    "PurgedCVPartitionRef",
    "PurgedCVRepository",
]

_logger = logging.getLogger(__name__)

_EXCHANGE: Final[Exchange] = EXCHANGE_BINANCE
_MARKET: Final[Market] = MARKET_USDT_PERPETUAL

_ERROR_FRAME_TYPE: Final[str] = "PCV_REPO_FRAME_TYPE"
_ERROR_MISSING_COLUMNS: Final[str] = "PCV_REPO_MISSING_COLUMNS"
_ERROR_SCHEMA_CAST: Final[str] = "PCV_REPO_SCHEMA_CAST"


@dataclass(frozen=True, slots=True)
class PurgedCVPartitionRef:
    """Identity of one discovered purged-CV year partition.

    Attributes:
        manager: Order manager identifier.
        exchange: Exchange identifier.
        market: Market segment.
        timeframe: Trade bar interval.
        year: Calendar year of the partition.
    """

    manager: str
    exchange: Exchange
    market: Market
    timeframe: Timeframe
    year: int


class PurgedCVRepository:
    """Repository facade for canonical purged-CV datasets.

    Callers identify partitions by manager, exchange, market, timeframe, and
    year. Paths are composed privately via
    ``StorageLayout.purged_cv_path`` and never returned. Persistence
    is delegated entirely to the injected ``IDataStore`` (commonly
    ``ParquetStore``) so atomic writes and alternate backends can be
    substituted without changing this API. Frames are validated and cast to
    ``PURGED_CV_SCHEMA`` on save and load.

    Partition layout::

        purged_cv/{manager}/{exchange}/{market}/{timeframe}/{year}.parquet

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
        base = self._purged_cv_root()
        if not base.is_dir():
            return ()
        return tuple(sorted(path.name for path in base.iterdir() if path.is_dir()))

    def discover_timeframes(
        self,
        *,
        manager: str,
        exchange: Exchange = _EXCHANGE,
        market: Market = _MARKET,
    ) -> tuple[Timeframe, ...]:
        """Return sorted timeframes with at least one partition for ``manager``.

        Args:
            manager: Order manager identifier.
            exchange: Exchange identifier.
            market: Market segment.

        Returns:
            Deterministically ordered timeframes discovered on disk.
        """
        base = self._manager_root(manager=manager, exchange=exchange, market=market)
        if not base.is_dir():
            return ()
        return tuple(sorted(path.name for path in base.iterdir() if path.is_dir()))

    def list_years(
        self,
        *,
        manager: str,
        exchange: Exchange,
        market: Market,
        timeframe: Timeframe,
    ) -> tuple[int, ...]:
        """Return sorted calendar years present as purged-CV partitions.

        Args:
            manager: Order manager identifier.
            exchange: Exchange identifier.
            market: Market segment.
            timeframe: Trade bar interval.

        Returns:
            Deterministically ordered year values for existing partitions.
        """
        return self._discover_years(
            manager=manager,
            exchange=exchange,
            market=market,
            timeframe=timeframe,
        )

    def discover(
        self,
        *,
        managers: Sequence[str] | None = None,
        timeframes: Sequence[Timeframe] | None = None,
        exchange: Exchange = _EXCHANGE,
        market: Market = _MARKET,
    ) -> tuple[PurgedCVPartitionRef, ...]:
        """Discover purged-CV year partitions matching optional filters.

        Missing purged-CV trees are skipped. Only year partitions that
        exist as ``{year}.parquet`` files are included. Paths are never
        returned.

        Args:
            managers: Optional manager allowlist. ``None`` discovers every
                manager present under the purged-CV tier.
            timeframes: Optional timeframe allowlist. ``None`` discovers every
                timeframe present for each manager.
            exchange: Exchange identifier.
            market: Market segment.

        Returns:
            Deterministically ordered partition references.
        """
        return self.discover_partitions(
            managers=managers,
            timeframes=timeframes,
            exchange=exchange,
            market=market,
        )

    def discover_partitions(
        self,
        *,
        managers: Sequence[str] | None = None,
        timeframes: Sequence[Timeframe] | None = None,
        exchange: Exchange = _EXCHANGE,
        market: Market = _MARKET,
    ) -> tuple[PurgedCVPartitionRef, ...]:
        """Discover purged-CV year partitions matching optional filters.

        Args:
            managers: Optional manager allowlist. ``None`` discovers every
                manager present under the purged-CV tier.
            timeframes: Optional timeframe allowlist. ``None`` discovers every
                timeframe present for each manager.
            exchange: Exchange identifier.
            market: Market segment.

        Returns:
            Deterministically ordered partition references.
        """
        manager_filter = set(managers) if managers is not None else None
        timeframe_filter = set(timeframes) if timeframes is not None else None

        items: list[PurgedCVPartitionRef] = []
        for manager in self.discover_managers():
            if manager_filter is not None and manager not in manager_filter:
                continue
            for timeframe in self.discover_timeframes(
                manager=manager,
                exchange=exchange,
                market=market,
            ):
                if timeframe_filter is not None and timeframe not in timeframe_filter:
                    continue
                for year in self._discover_years(
                    manager=manager,
                    exchange=exchange,
                    market=market,
                    timeframe=timeframe,
                ):
                    items.append(
                        PurgedCVPartitionRef(
                            manager=manager,
                            exchange=exchange,
                            market=market,
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
        timeframe: Timeframe,
        year: int,
    ) -> bool:
        """Return whether a purged-CV year partition exists.

        Existence is determined solely through ``IDataStore.exists`` on the
        path composed by ``StorageLayout.purged_cv_path``. Parquet
        contents are never read, scanned, or validated.

        Args:
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            ``True`` when the yearly purged-CV partition exists;
            otherwise ``False``.
        """
        path = self._layout.purged_cv_path(
            manager,
            exchange,
            market,
            timeframe,
            year,
        )
        present = self._datastore.exists(path)
        self._logger.debug(
            ("Purged-CV dataset exists" if present else "Purged-CV dataset does not exist"),
            extra=_purged_cv_log_extra(
                manager=manager,
                exchange=exchange,
                market=market,
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
        timeframe: Timeframe,
        year: int,
    ) -> None:
        """Persist a purged-CV year partition after schema validation.

        Args:
            dataframe: Purged-CV frame to store. Must conform to
                ``PURGED_CV_SCHEMA``.
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Raises:
            PurgedCVError: If ``dataframe`` is not a DataFrame,
                required columns are missing, or casting to
                ``PURGED_CV_SCHEMA`` fails.
        """
        validated = _require_purged_cv_schema(dataframe)
        path = self._layout.purged_cv_path(
            manager,
            exchange,
            market,
            timeframe,
            year,
        )
        self._logger.debug(
            "Saving purged-CV dataset",
            extra=_purged_cv_log_extra(
                manager=manager,
                exchange=exchange,
                market=market,
                timeframe=timeframe,
                year=year,
                rows=validated.height,
                columns=validated.width,
            ),
        )
        self._datastore.write(path, validated)
        self._logger.info(
            "Saved purged-CV dataset",
            extra=_purged_cv_log_extra(
                manager=manager,
                exchange=exchange,
                market=market,
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
        timeframe: Timeframe,
        year: int,
    ) -> pl.DataFrame:
        """Load a purged-CV partition cast to ``PURGED_CV_SCHEMA``.

        Args:
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            Eagerly loaded purged-CV DataFrame cast to
            ``PURGED_CV_SCHEMA``.

        Raises:
            PurgedCVError: If the loaded frame fails schema
                validation or casting.
        """
        path = self._layout.purged_cv_path(
            manager,
            exchange,
            market,
            timeframe,
            year,
        )
        self._logger.debug(
            "Loading purged-CV dataset",
            extra=_purged_cv_log_extra(
                manager=manager,
                exchange=exchange,
                market=market,
                timeframe=timeframe,
                year=year,
            ),
        )
        frame = self._datastore.read(path)
        validated = _require_purged_cv_schema(frame)
        self._logger.info(
            "Loaded purged-CV dataset",
            extra=_purged_cv_log_extra(
                manager=manager,
                exchange=exchange,
                market=market,
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
        timeframe: Timeframe,
        year: int,
    ) -> None:
        """Delete a purged-CV year partition.

        Args:
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Raises:
            DatasetNotFoundError: If the partition does not exist (propagated
                from the injected datastore).
        """
        path = self._layout.purged_cv_path(
            manager,
            exchange,
            market,
            timeframe,
            year,
        )
        self._logger.debug(
            "Deleting purged-CV dataset",
            extra=_purged_cv_log_extra(
                manager=manager,
                exchange=exchange,
                market=market,
                timeframe=timeframe,
                year=year,
            ),
        )
        self._datastore.delete(path)
        self._logger.info(
            "Deleted purged-CV dataset",
            extra=_purged_cv_log_extra(
                manager=manager,
                exchange=exchange,
                market=market,
                timeframe=timeframe,
                year=year,
            ),
        )

    def _purged_cv_root(self) -> Path:
        """Return the purged-CV directory under the storage root."""
        return self._layout.root / STORAGE_DIR_PURGED_CV

    def _manager_root(
        self,
        *,
        manager: str,
        exchange: Exchange,
        market: Market,
    ) -> Path:
        """Return the purged-CV directory for manager/exchange/market."""
        return self._purged_cv_root() / manager / exchange / market

    def _discover_years(
        self,
        *,
        manager: str,
        exchange: Exchange,
        market: Market,
        timeframe: Timeframe,
    ) -> tuple[int, ...]:
        """Return sorted calendar years present as parquet partitions."""
        base = self._manager_root(manager=manager, exchange=exchange, market=market) / timeframe
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


def _require_purged_cv_schema(frame: object) -> pl.DataFrame:
    """Validate and cast ``frame`` to ``PURGED_CV_SCHEMA``."""
    if not isinstance(frame, pl.DataFrame):
        raise PurgedCVError(
            "purged-CV frame must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={"actual_type": type(frame).__name__},
        )
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise PurgedCVError(
            "purged-CV schema is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "missing_columns": tuple(missing),
                "required_columns": REQUIRED_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )
    try:
        return frame.select(list(PURGED_CV_COLUMNS)).cast(PURGED_CV_SCHEMA)
    except (pl.exceptions.PolarsError, TypeError, ValueError) as exc:
        raise PurgedCVError(
            "purged-CV frame failed PURGED_CV_SCHEMA cast",
            error_code=_ERROR_SCHEMA_CAST,
            details={"reason": str(exc)},
        ) from exc


def _purged_cv_log_extra(
    *,
    manager: str,
    exchange: Exchange,
    market: Market,
    timeframe: Timeframe,
    year: int,
    rows: int | None = None,
    columns: int | None = None,
) -> dict[str, object]:
    """Build structured log fields for a purged-CV dataset operation."""
    payload: dict[str, object] = {
        "tier": "purged_cv",
        "manager": manager,
        "exchange": exchange,
        "market": market,
        "timeframe": timeframe,
        "year": year,
    }
    if rows is not None:
        payload["rows"] = rows
    if columns is not None:
        payload["columns"] = columns
    return payload
