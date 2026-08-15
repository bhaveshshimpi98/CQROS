"""CQROS factor selection metrics dataset repository.

Purpose:
    Provide a path-free facade for persisting and retrieving canonical
    factor selection year partitions produced by the CQROS Factor
    Selection Engine.

Responsibilities:
    - Resolve storage locations for factor selection year partitions via
      ``StorageLayout.factor_selection_path``
    - Persist, load, check existence, and discover factor selection
      partitions
    - Validate frames against ``FACTOR_SELECTION_SCHEMA`` before save
    - Cast loaded frames to ``FACTOR_SELECTION_SCHEMA`` after read
    - Discover managers, timeframes, and year partitions under the
      factor selection tier
    - Delegate all read and write operations to an injected ``IDataStore``
      (typically ``ParquetStore`` for atomic Parquet writes)
    - Keep filesystem paths out of the public API
    - Remain free of selection math and trading logic

Dependencies:
    ``polars``, ``cqros.core``, ``cqros.factor_selection.exceptions``,
    ``cqros.factor_selection.schema``, and ``cqros.storage``
    layout/interfaces.

Public API:
    ``FactorSelectionPartitionRef``, ``FactorSelectionRepository``

Notes:
    Schema validation and casting keep the on-disk contract aligned with
    ``FACTOR_SELECTION_SCHEMA``. Selection computation belongs to the engine
    and pipeline. This module must not import factor selection engines for
    execution.

    Partition identity is cross-sectional::

        factor_selection/{manager}/{exchange}/{market}/{timeframe}/{year}.parquet
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
    STORAGE_DIR_FACTOR_SELECTION,
)
from cqros.core.types import Exchange, Market, Timeframe
from cqros.factor_selection.exceptions import FactorSelectionError
from cqros.factor_selection.schema import (
    ELIGIBILITY_COLUMN_DTYPES,
    ELIGIBILITY_COLUMNS,
    FACTOR_SELECTION_COLUMNS,
    FACTOR_SELECTION_SCHEMA,
    REQUIRED_COLUMNS,
)
from cqros.storage.interfaces import IDataStore
from cqros.storage.layout import StorageLayout

__all__ = [
    "FactorSelectionPartitionRef",
    "FactorSelectionRepository",
]

_logger = logging.getLogger(__name__)

_EXCHANGE: Final[Exchange] = EXCHANGE_BINANCE
_MARKET: Final[Market] = MARKET_USDT_PERPETUAL

_ERROR_FRAME_TYPE: Final[str] = "FSEL_REPO_FRAME_TYPE"
_ERROR_MISSING_COLUMNS: Final[str] = "FSEL_REPO_MISSING_COLUMNS"
_ERROR_SCHEMA_CAST: Final[str] = "FSEL_REPO_SCHEMA_CAST"


@dataclass(frozen=True, slots=True)
class FactorSelectionPartitionRef:
    """Identity of one discovered factor selection year partition.

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


class FactorSelectionRepository:
    """Repository facade for canonical factor selection datasets.

    Callers identify partitions by manager, exchange, market, timeframe, and
    year. Paths are composed privately via
    ``StorageLayout.factor_selection_path`` and never returned. Persistence
    is delegated entirely to the injected ``IDataStore`` (commonly
    ``ParquetStore``) so atomic writes and alternate backends can be
    substituted without changing this API. Frames are validated and cast to
    ``FACTOR_SELECTION_SCHEMA`` on save and load.

    Partition layout::

        factor_selection/{manager}/{exchange}/{market}/{timeframe}/{year}.parquet

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
        base = self._factor_selection_root()
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
        """Return sorted calendar years present as factor selection partitions.

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
    ) -> tuple[FactorSelectionPartitionRef, ...]:
        """Discover factor selection year partitions matching optional filters.

        Missing factor selection trees are skipped. Only year partitions that
        exist as ``{year}.parquet`` files are included. Paths are never
        returned.

        Args:
            managers: Optional manager allowlist. ``None`` discovers every
                manager present under the factor selection tier.
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
    ) -> tuple[FactorSelectionPartitionRef, ...]:
        """Discover factor selection year partitions matching optional filters.

        Args:
            managers: Optional manager allowlist. ``None`` discovers every
                manager present under the factor selection tier.
            timeframes: Optional timeframe allowlist. ``None`` discovers every
                timeframe present for each manager.
            exchange: Exchange identifier.
            market: Market segment.

        Returns:
            Deterministically ordered partition references.
        """
        manager_filter = set(managers) if managers is not None else None
        timeframe_filter = set(timeframes) if timeframes is not None else None

        items: list[FactorSelectionPartitionRef] = []
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
                        FactorSelectionPartitionRef(
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
        """Return whether a factor selection year partition exists.

        Existence is determined solely through ``IDataStore.exists`` on the
        path composed by ``StorageLayout.factor_selection_path``. Parquet
        contents are never read, scanned, or validated.

        Args:
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            ``True`` when the yearly factor selection partition exists;
            otherwise ``False``.
        """
        path = self._layout.factor_selection_path(
            manager,
            exchange,
            market,
            timeframe,
            year,
        )
        present = self._datastore.exists(path)
        self._logger.debug(
            (
                "Factor selection dataset exists"
                if present
                else "Factor selection dataset does not exist"
            ),
            extra=_factor_selection_log_extra(
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
        """Persist a factor selection year partition after schema validation.

        Args:
            dataframe: Factor selection frame to store. Must conform to
                ``FACTOR_SELECTION_SCHEMA``.
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Raises:
            FactorSelectionError: If ``dataframe`` is not a DataFrame,
                required columns are missing, or casting to
                ``FACTOR_SELECTION_SCHEMA`` fails.
        """
        validated = _require_factor_selection_schema(dataframe)
        path = self._layout.factor_selection_path(
            manager,
            exchange,
            market,
            timeframe,
            year,
        )
        self._logger.debug(
            "Saving factor selection dataset",
            extra=_factor_selection_log_extra(
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
            "Saved factor selection dataset",
            extra=_factor_selection_log_extra(
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
        """Load a factor selection partition cast to ``FACTOR_SELECTION_SCHEMA``.

        Args:
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            Eagerly loaded factor selection DataFrame cast to
            ``FACTOR_SELECTION_SCHEMA``.

        Raises:
            FactorSelectionError: If the loaded frame fails schema
                validation or casting.
        """
        path = self._layout.factor_selection_path(
            manager,
            exchange,
            market,
            timeframe,
            year,
        )
        self._logger.debug(
            "Loading factor selection dataset",
            extra=_factor_selection_log_extra(
                manager=manager,
                exchange=exchange,
                market=market,
                timeframe=timeframe,
                year=year,
            ),
        )
        frame = self._datastore.read(path)
        validated = _require_factor_selection_schema(frame)
        self._logger.info(
            "Loaded factor selection dataset",
            extra=_factor_selection_log_extra(
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
        """Delete a factor selection year partition.

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
        path = self._layout.factor_selection_path(
            manager,
            exchange,
            market,
            timeframe,
            year,
        )
        self._logger.debug(
            "Deleting factor selection dataset",
            extra=_factor_selection_log_extra(
                manager=manager,
                exchange=exchange,
                market=market,
                timeframe=timeframe,
                year=year,
            ),
        )
        self._datastore.delete(path)
        self._logger.info(
            "Deleted factor selection dataset",
            extra=_factor_selection_log_extra(
                manager=manager,
                exchange=exchange,
                market=market,
                timeframe=timeframe,
                year=year,
            ),
        )

    def _factor_selection_root(self) -> Path:
        """Return the factor selection directory under the storage root."""
        return self._layout.root / STORAGE_DIR_FACTOR_SELECTION

    def _manager_root(
        self,
        *,
        manager: str,
        exchange: Exchange,
        market: Market,
    ) -> Path:
        """Return the factor selection directory for manager/exchange/market."""
        return self._factor_selection_root() / manager / exchange / market

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


def _require_factor_selection_schema(frame: object) -> pl.DataFrame:
    """Validate and cast ``frame`` to ``FACTOR_SELECTION_SCHEMA``.

    Eligibility metadata columns (``ELIGIBILITY_COLUMNS``) are preserved and
    cast when present.  Legacy frames without eligibility columns are accepted
    without error; downstream code that requires eligibility metadata must call
    ``require_eligibility_metadata`` explicitly.
    """
    if not isinstance(frame, pl.DataFrame):
        raise FactorSelectionError(
            "factor selection frame must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={"actual_type": type(frame).__name__},
        )
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise FactorSelectionError(
            "factor selection schema is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "missing_columns": tuple(missing),
                "required_columns": REQUIRED_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )
    present_eligibility = [col for col in ELIGIBILITY_COLUMNS if col in frame.columns]
    select_columns = list(FACTOR_SELECTION_COLUMNS) + present_eligibility
    cast_schema: pl.Schema = pl.Schema(
        {
            **dict(FACTOR_SELECTION_SCHEMA),
            **{col: ELIGIBILITY_COLUMN_DTYPES[col] for col in present_eligibility},
        }
    )
    try:
        return frame.select(select_columns).cast(cast_schema)
    except (pl.exceptions.PolarsError, TypeError, ValueError) as exc:
        raise FactorSelectionError(
            "factor selection frame failed FACTOR_SELECTION_SCHEMA cast",
            error_code=_ERROR_SCHEMA_CAST,
            details={"reason": str(exc)},
        ) from exc


def _factor_selection_log_extra(
    *,
    manager: str,
    exchange: Exchange,
    market: Market,
    timeframe: Timeframe,
    year: int,
    rows: int | None = None,
    columns: int | None = None,
) -> dict[str, object]:
    """Build structured log fields for a factor selection dataset operation."""
    payload: dict[str, object] = {
        "tier": "factor_selection",
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
