"""CQROS factor orthogonalization metrics dataset repository.

Purpose:
    Provide a path-free facade for persisting and retrieving canonical
    factor orthogonalization year partitions produced by the CQROS Factor
    Orthogonalization Engine.

Responsibilities:
    - Resolve storage locations for factor orthogonalization year partitions
      via ``StorageLayout.factor_orthogonalization_path``
    - Persist, load, check existence, and discover factor orthogonalization
      partitions
    - Validate frames against ``FACTOR_ORTHOGONALIZATION_SCHEMA`` before save
    - Cast loaded frames to ``FACTOR_ORTHOGONALIZATION_SCHEMA`` after read
    - Discover managers, timeframes, and year partitions under the
      factor orthogonalization tier
    - Delegate all read and write operations to an injected ``IDataStore``
    - Keep filesystem paths out of the public API
    - Remain free of orthogonalization math and trading logic

Dependencies:
    ``polars``, ``cqros.core``, ``cqros.factor_orthogonalization.exceptions``,
    ``cqros.factor_orthogonalization.schema``, and ``cqros.storage``
    layout/interfaces.

Public API:
    ``FactorOrthogonalizationPartitionRef``,
    ``FactorOrthogonalizationRepository``

Notes:
    Partition identity is cross-sectional::

        factor_orthogonalization/{manager}/{exchange}/{market}/{timeframe}/{year}.parquet
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
    STORAGE_DIR_FACTOR_ORTHOGONALIZATION,
)
from cqros.core.types import Exchange, Market, Timeframe
from cqros.factor_orthogonalization.exceptions import FactorOrthogonalizationError
from cqros.factor_orthogonalization.schema import (
    FACTOR_ORTHOGONALIZATION_COLUMNS,
    FACTOR_ORTHOGONALIZATION_SCHEMA,
    REQUIRED_COLUMNS,
)
from cqros.storage.interfaces import IDataStore
from cqros.storage.layout import StorageLayout

__all__ = [
    "FactorOrthogonalizationPartitionRef",
    "FactorOrthogonalizationRepository",
]

_logger = logging.getLogger(__name__)

_EXCHANGE: Final[Exchange] = EXCHANGE_BINANCE
_MARKET: Final[Market] = MARKET_USDT_PERPETUAL

_ERROR_FRAME_TYPE: Final[str] = "FORTH_REPO_FRAME_TYPE"
_ERROR_MISSING_COLUMNS: Final[str] = "FORTH_REPO_MISSING_COLUMNS"
_ERROR_SCHEMA_CAST: Final[str] = "FORTH_REPO_SCHEMA_CAST"


@dataclass(frozen=True, slots=True)
class FactorOrthogonalizationPartitionRef:
    """Identity of one discovered factor orthogonalization year partition.

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


class FactorOrthogonalizationRepository:
    """Repository facade for canonical factor orthogonalization datasets.

    Callers identify partitions by manager, exchange, market, timeframe, and
    year. Paths are composed privately via
    ``StorageLayout.factor_orthogonalization_path`` and never returned.

    Partition layout::

        factor_orthogonalization/{manager}/{exchange}/{market}/{timeframe}/{year}.parquet
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
        """Initialize the repository with injected layout and datastore."""
        self._layout = layout
        self._datastore = datastore
        self._logger = logger if logger is not None else _logger

    def discover_managers(self) -> tuple[str, ...]:
        """Return sorted manager identifiers with at least one partition."""
        base = self._factor_orthogonalization_root()
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
        """Return sorted timeframes with at least one partition for ``manager``."""
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
        """Return sorted calendar years present as orthogonalization partitions."""
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
    ) -> tuple[FactorOrthogonalizationPartitionRef, ...]:
        """Discover orthogonalization year partitions matching optional filters."""
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
    ) -> tuple[FactorOrthogonalizationPartitionRef, ...]:
        """Discover orthogonalization year partitions matching optional filters."""
        manager_filter = set(managers) if managers is not None else None
        timeframe_filter = set(timeframes) if timeframes is not None else None

        items: list[FactorOrthogonalizationPartitionRef] = []
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
                        FactorOrthogonalizationPartitionRef(
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
        """Return whether a factor orthogonalization year partition exists."""
        path = self._layout.factor_orthogonalization_path(
            manager,
            exchange,
            market,
            timeframe,
            year,
        )
        present = self._datastore.exists(path)
        self._logger.debug(
            (
                "Factor orthogonalization dataset exists"
                if present
                else "Factor orthogonalization dataset does not exist"
            ),
            extra=_factor_orthogonalization_log_extra(
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
        """Persist an orthogonalization year partition after schema validation."""
        validated = _require_factor_orthogonalization_schema(dataframe)
        path = self._layout.factor_orthogonalization_path(
            manager,
            exchange,
            market,
            timeframe,
            year,
        )
        self._logger.debug(
            "Saving factor orthogonalization dataset",
            extra=_factor_orthogonalization_log_extra(
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
            "Saved factor orthogonalization dataset",
            extra=_factor_orthogonalization_log_extra(
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
        """Load an orthogonalization partition cast to the canonical schema."""
        path = self._layout.factor_orthogonalization_path(
            manager,
            exchange,
            market,
            timeframe,
            year,
        )
        self._logger.debug(
            "Loading factor orthogonalization dataset",
            extra=_factor_orthogonalization_log_extra(
                manager=manager,
                exchange=exchange,
                market=market,
                timeframe=timeframe,
                year=year,
            ),
        )
        if not self._datastore.exists(path):
            empty = pl.DataFrame(schema=FACTOR_ORTHOGONALIZATION_SCHEMA)
            self._logger.info(
                "Factor orthogonalization dataset missing; returning empty frame",
                extra=_factor_orthogonalization_log_extra(
                    manager=manager,
                    exchange=exchange,
                    market=market,
                    timeframe=timeframe,
                    year=year,
                    rows=empty.height,
                    columns=empty.width,
                ),
            )
            return empty
        frame = self._datastore.read(path)
        validated = _require_factor_orthogonalization_schema(frame)
        self._logger.info(
            "Loaded factor orthogonalization dataset",
            extra=_factor_orthogonalization_log_extra(
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
        """Delete a factor orthogonalization year partition."""
        path = self._layout.factor_orthogonalization_path(
            manager,
            exchange,
            market,
            timeframe,
            year,
        )
        self._logger.debug(
            "Deleting factor orthogonalization dataset",
            extra=_factor_orthogonalization_log_extra(
                manager=manager,
                exchange=exchange,
                market=market,
                timeframe=timeframe,
                year=year,
            ),
        )
        if not self._datastore.exists(path):
            self._logger.debug(
                "Factor orthogonalization dataset already absent",
                extra=_factor_orthogonalization_log_extra(
                    manager=manager,
                    exchange=exchange,
                    market=market,
                    timeframe=timeframe,
                    year=year,
                ),
            )
            return
        self._datastore.delete(path)
        self._logger.info(
            "Deleted factor orthogonalization dataset",
            extra=_factor_orthogonalization_log_extra(
                manager=manager,
                exchange=exchange,
                market=market,
                timeframe=timeframe,
                year=year,
            ),
        )

    def _factor_orthogonalization_root(self) -> Path:
        """Return the factor orthogonalization directory under the storage root."""
        return self._layout.root / STORAGE_DIR_FACTOR_ORTHOGONALIZATION

    def _manager_root(
        self,
        *,
        manager: str,
        exchange: Exchange,
        market: Market,
    ) -> Path:
        """Return the orthogonalization directory for manager/exchange/market."""
        return self._factor_orthogonalization_root() / manager / exchange / market

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


def _require_factor_orthogonalization_schema(frame: object) -> pl.DataFrame:
    """Validate and cast ``frame`` to ``FACTOR_ORTHOGONALIZATION_SCHEMA``."""
    if not isinstance(frame, pl.DataFrame):
        raise FactorOrthogonalizationError(
            "factor orthogonalization frame must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={"actual_type": type(frame).__name__},
        )
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise FactorOrthogonalizationError(
            "factor orthogonalization schema is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "missing_columns": tuple(missing),
                "required_columns": REQUIRED_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )
    try:
        return frame.select(list(FACTOR_ORTHOGONALIZATION_COLUMNS)).cast(
            FACTOR_ORTHOGONALIZATION_SCHEMA
        )
    except (pl.exceptions.PolarsError, TypeError, ValueError) as exc:
        raise FactorOrthogonalizationError(
            "factor orthogonalization frame failed " "FACTOR_ORTHOGONALIZATION_SCHEMA cast",
            error_code=_ERROR_SCHEMA_CAST,
            details={"reason": str(exc)},
        ) from exc


def _factor_orthogonalization_log_extra(
    *,
    manager: str,
    exchange: Exchange,
    market: Market,
    timeframe: Timeframe,
    year: int,
    rows: int | None = None,
    columns: int | None = None,
) -> dict[str, object]:
    """Build structured log fields for an orthogonalization dataset operation."""
    payload: dict[str, object] = {
        "tier": "factor_orthogonalization",
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
