"""CQROS factor timeframe analysis metrics dataset repository.

Purpose:
    Provide a path-free facade for persisting and retrieving canonical
    factor timeframe analysis year partitions produced by the CQROS Factor
    Timeframe Analysis Engine.

Responsibilities:
    - Resolve storage locations for factor timeframe analysis year
      partitions via ``StorageLayout.factor_timeframe_analysis_path``
    - Persist, load, check existence, and discover timeframe analysis
      partitions
    - Validate frames against ``TIMEFRAME_ANALYSIS_SCHEMA`` before save
    - Cast loaded frames to ``TIMEFRAME_ANALYSIS_SCHEMA`` after read
    - Discover managers and year partitions under the factor timeframe
      analysis tier
    - Delegate all read and write operations to an injected ``IDataStore``
      (typically ``ParquetStore`` for atomic Parquet writes)
    - Keep filesystem paths out of the public API
    - Remain free of analysis math and trading logic

Dependencies:
    ``polars``, ``cqros.core``, ``cqros.factor_timeframe_analysis.exceptions``,
    ``cqros.factor_timeframe_analysis.schema``, and ``cqros.storage``
    layout/interfaces.

Public API:
    ``FactorTimeframeAnalysisPartitionRef``, ``FactorTimeframeAnalysisRepository``

Notes:
    Schema validation and casting keep the on-disk contract aligned with
    ``TIMEFRAME_ANALYSIS_SCHEMA``. Analysis computation belongs to the engine
    and pipeline. This module must not import timeframe analysis engines for
    execution.

    FTA panels are cross-sectional: symbol and source timeframe are resolved
    by the engine and stored as FTA schema columns, not as partition path
    segments. Partition layout::

        factor_timeframe_analysis/{manager}/{exchange}/{market}/{year}.parquet
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
    STORAGE_DIR_FACTOR_TIMEFRAME_ANALYSIS,
)
from cqros.core.types import Exchange, Market
from cqros.factor_timeframe_analysis.exceptions import FactorTimeframeAnalysisError
from cqros.factor_timeframe_analysis.schema import (
    FACTOR_TIMEFRAME_ANALYSIS_COLUMNS,
    REQUIRED_COLUMNS,
    TIMEFRAME_ANALYSIS_SCHEMA,
)
from cqros.storage.interfaces import IDataStore
from cqros.storage.layout import StorageLayout

__all__ = [
    "FactorTimeframeAnalysisPartitionRef",
    "FactorTimeframeAnalysisRepository",
]

_logger = logging.getLogger(__name__)

_EXCHANGE: Final[Exchange] = EXCHANGE_BINANCE
_MARKET: Final[Market] = MARKET_USDT_PERPETUAL

_ERROR_FRAME_TYPE: Final[str] = "FTA_REPO_FRAME_TYPE"
_ERROR_MISSING_COLUMNS: Final[str] = "FTA_REPO_MISSING_COLUMNS"
_ERROR_SCHEMA_CAST: Final[str] = "FTA_REPO_SCHEMA_CAST"


@dataclass(frozen=True, slots=True)
class FactorTimeframeAnalysisPartitionRef:
    """Identity of one discovered factor timeframe analysis year partition.

    FTA panels are cross-sectional: symbol and source timeframe are encoded
    inside FTA schema columns (``best_timeframe``, ``source_selection_version``),
    not in the partition path.

    Attributes:
        manager: Order manager identifier.
        exchange: Exchange identifier.
        market: Market segment.
        year: Calendar year of the partition.
    """

    manager: str
    exchange: Exchange
    market: Market
    year: int


class FactorTimeframeAnalysisRepository:
    """Repository facade for canonical factor timeframe analysis datasets.

    Callers identify partitions by manager, exchange, market, and year.
    Paths are composed privately via
    ``StorageLayout.factor_timeframe_analysis_path`` and never returned.
    Persistence is delegated entirely to the injected ``IDataStore``
    (commonly ``ParquetStore``) so atomic writes and alternate backends can
    be substituted without changing this API. Frames are validated and cast
    to ``TIMEFRAME_ANALYSIS_SCHEMA`` on save and load.

    Partition layout::

        factor_timeframe_analysis/{manager}/{exchange}/{market}/{year}.parquet

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
        base = self._factor_timeframe_analysis_root()
        if not base.is_dir():
            return ()
        return tuple(sorted(path.name for path in base.iterdir() if path.is_dir()))

    def list_years(
        self,
        *,
        manager: str,
        exchange: Exchange = _EXCHANGE,
        market: Market = _MARKET,
    ) -> tuple[int, ...]:
        """Return sorted calendar years present as timeframe analysis partitions.

        Args:
            manager: Order manager identifier.
            exchange: Exchange identifier.
            market: Market segment.

        Returns:
            Deterministically ordered year values for existing partitions.
        """
        return self._discover_years(
            manager=manager,
            exchange=exchange,
            market=market,
        )

    def discover(
        self,
        *,
        managers: Sequence[str] | None = None,
        exchange: Exchange = _EXCHANGE,
        market: Market = _MARKET,
    ) -> tuple[FactorTimeframeAnalysisPartitionRef, ...]:
        """Discover timeframe analysis year partitions matching optional filters.

        Missing timeframe analysis trees are skipped. Only year partitions
        that exist as ``{year}.parquet`` files are included. Paths are never
        returned.

        Args:
            managers: Optional manager allowlist. ``None`` discovers every
                manager present under the timeframe analysis tier.
            exchange: Exchange identifier.
            market: Market segment.

        Returns:
            Deterministically ordered partition references.
        """
        return self.discover_partitions(
            managers=managers,
            exchange=exchange,
            market=market,
        )

    def discover_partitions(
        self,
        *,
        managers: Sequence[str] | None = None,
        exchange: Exchange = _EXCHANGE,
        market: Market = _MARKET,
    ) -> tuple[FactorTimeframeAnalysisPartitionRef, ...]:
        """Discover timeframe analysis year partitions matching optional filters.

        Args:
            managers: Optional manager allowlist. ``None`` discovers every
                manager present under the timeframe analysis tier.
            exchange: Exchange identifier.
            market: Market segment.

        Returns:
            Deterministically ordered partition references.
        """
        manager_filter = set(managers) if managers is not None else None

        items: list[FactorTimeframeAnalysisPartitionRef] = []
        for manager in self.discover_managers():
            if manager_filter is not None and manager not in manager_filter:
                continue
            for year in self._discover_years(
                manager=manager,
                exchange=exchange,
                market=market,
            ):
                items.append(
                    FactorTimeframeAnalysisPartitionRef(
                        manager=manager,
                        exchange=exchange,
                        market=market,
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
        year: int,
    ) -> bool:
        """Return whether a timeframe analysis year partition exists.

        Existence is determined solely through ``IDataStore.exists`` on the
        path composed by ``StorageLayout.factor_timeframe_analysis_path``.
        Parquet contents are never read, scanned, or validated.

        Args:
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            year: Calendar year of the partition.

        Returns:
            ``True`` when the yearly timeframe analysis partition exists;
            otherwise ``False``.
        """
        path = self._layout.factor_timeframe_analysis_path(manager, exchange, market, year)
        present = self._datastore.exists(path)
        self._logger.debug(
            (
                "Factor timeframe analysis dataset exists"
                if present
                else "Factor timeframe analysis dataset does not exist"
            ),
            extra=_fta_log_extra(
                manager=manager,
                exchange=exchange,
                market=market,
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
        year: int,
    ) -> None:
        """Persist a timeframe analysis year partition after schema validation.

        Args:
            dataframe: Timeframe analysis frame to store. Must conform to
                ``TIMEFRAME_ANALYSIS_SCHEMA``.
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            year: Calendar year of the partition.

        Raises:
            FactorTimeframeAnalysisError: If ``dataframe`` is not a DataFrame,
                required columns are missing, or casting to
                ``TIMEFRAME_ANALYSIS_SCHEMA`` fails.
        """
        validated = _require_timeframe_analysis_schema(dataframe)
        path = self._layout.factor_timeframe_analysis_path(manager, exchange, market, year)
        self._logger.debug(
            "Saving factor timeframe analysis dataset",
            extra=_fta_log_extra(
                manager=manager,
                exchange=exchange,
                market=market,
                year=year,
                rows=validated.height,
                columns=validated.width,
            ),
        )
        self._datastore.write(path, validated)
        self._logger.info(
            "Saved factor timeframe analysis dataset",
            extra=_fta_log_extra(
                manager=manager,
                exchange=exchange,
                market=market,
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
        year: int,
    ) -> pl.DataFrame:
        """Load a timeframe analysis partition cast to ``TIMEFRAME_ANALYSIS_SCHEMA``.

        Args:
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            year: Calendar year of the partition.

        Returns:
            Eagerly loaded timeframe analysis DataFrame cast to
            ``TIMEFRAME_ANALYSIS_SCHEMA``.

        Raises:
            FactorTimeframeAnalysisError: If the loaded frame fails schema
                validation or casting.
        """
        path = self._layout.factor_timeframe_analysis_path(manager, exchange, market, year)
        self._logger.debug(
            "Loading factor timeframe analysis dataset",
            extra=_fta_log_extra(
                manager=manager,
                exchange=exchange,
                market=market,
                year=year,
            ),
        )
        frame = self._datastore.read(path)
        validated = _require_timeframe_analysis_schema(frame)
        self._logger.info(
            "Loaded factor timeframe analysis dataset",
            extra=_fta_log_extra(
                manager=manager,
                exchange=exchange,
                market=market,
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
        year: int,
    ) -> None:
        """Delete a timeframe analysis year partition.

        Args:
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            year: Calendar year of the partition.

        Raises:
            DatasetNotFoundError: If the partition does not exist (propagated
                from the injected datastore).
        """
        path = self._layout.factor_timeframe_analysis_path(manager, exchange, market, year)
        self._logger.debug(
            "Deleting factor timeframe analysis dataset",
            extra=_fta_log_extra(
                manager=manager,
                exchange=exchange,
                market=market,
                year=year,
            ),
        )
        self._datastore.delete(path)
        self._logger.info(
            "Deleted factor timeframe analysis dataset",
            extra=_fta_log_extra(
                manager=manager,
                exchange=exchange,
                market=market,
                year=year,
            ),
        )

    def _factor_timeframe_analysis_root(self) -> Path:
        """Return the factor timeframe analysis directory under the storage root."""
        return self._layout.root / STORAGE_DIR_FACTOR_TIMEFRAME_ANALYSIS

    def _manager_root(
        self,
        *,
        manager: str,
        exchange: Exchange,
        market: Market,
    ) -> Path:
        """Return the timeframe analysis directory for manager/exchange/market."""
        return self._factor_timeframe_analysis_root() / manager / exchange / market

    def _discover_years(
        self,
        *,
        manager: str,
        exchange: Exchange,
        market: Market,
    ) -> tuple[int, ...]:
        """Return sorted calendar years present as parquet partitions."""
        base = self._manager_root(manager=manager, exchange=exchange, market=market)
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


def _require_timeframe_analysis_schema(frame: object) -> pl.DataFrame:
    """Validate and cast ``frame`` to ``TIMEFRAME_ANALYSIS_SCHEMA``."""
    if not isinstance(frame, pl.DataFrame):
        raise FactorTimeframeAnalysisError(
            "factor timeframe analysis frame must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={"actual_type": type(frame).__name__},
        )
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise FactorTimeframeAnalysisError(
            "factor timeframe analysis schema is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "missing_columns": tuple(missing),
                "required_columns": REQUIRED_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )
    try:
        return frame.select(list(FACTOR_TIMEFRAME_ANALYSIS_COLUMNS)).cast(TIMEFRAME_ANALYSIS_SCHEMA)
    except (pl.exceptions.PolarsError, TypeError, ValueError) as exc:
        raise FactorTimeframeAnalysisError(
            "factor timeframe analysis frame failed TIMEFRAME_ANALYSIS_SCHEMA cast",
            error_code=_ERROR_SCHEMA_CAST,
            details={"reason": str(exc)},
        ) from exc


def _fta_log_extra(
    *,
    manager: str,
    exchange: Exchange,
    market: Market,
    year: int,
    rows: int | None = None,
    columns: int | None = None,
) -> dict[str, object]:
    """Build structured log fields for a timeframe analysis dataset operation."""
    payload: dict[str, object] = {
        "tier": "factor_timeframe_analysis",
        "manager": manager,
        "exchange": exchange,
        "market": market,
        "year": year,
    }
    if rows is not None:
        payload["rows"] = rows
    if columns is not None:
        payload["columns"] = columns
    return payload
