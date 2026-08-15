"""CQROS prediction dataset repository.

Purpose:
    Provide a path-free facade for persisting and retrieving model prediction
    year partitions produced by CQROS inference prior to signal generation.

Responsibilities:
    - Resolve storage locations for prediction year partitions via
      ``StorageLayout.prediction_path``
    - Persist, load, check existence, and delete prediction partitions
    - Discover models, versions, and year partitions under the predictions
      tier
    - Delegate all read and write operations to an injected ``IDataStore``
      (typically ``ParquetStore`` for atomic Parquet writes)
    - Keep filesystem paths out of the public API
    - Remain free of inference, validation, signal generation, and trading
      logic

Dependencies:
    ``polars``, ``cqros.core``, and ``cqros.storage`` layout/interfaces.

Public API:
    ``PredictionPartitionRef``, ``PredictionRepository``
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
    STORAGE_DIR_PREDICTIONS,
)
from cqros.core.types import Exchange, Market, Symbol, Timeframe
from cqros.storage.interfaces import IDataStore
from cqros.storage.layout import StorageLayout

__all__ = [
    "PredictionPartitionRef",
    "PredictionRepository",
]

_logger = logging.getLogger(__name__)

_EXCHANGE: Final[Exchange] = EXCHANGE_BINANCE
_MARKET: Final[Market] = MARKET_USDT_PERPETUAL


@dataclass(frozen=True, slots=True)
class PredictionPartitionRef:
    """Identity of one discovered prediction year partition.

    Attributes:
        exchange: Exchange identifier.
        market: Market segment.
        symbol: Tradeable symbol.
        timeframe: Prediction bar interval.
        year: Calendar year of the partition.
        model_name: Stable model identifier.
        model_version: Model version identifier.
    """

    exchange: Exchange
    market: Market
    symbol: Symbol
    timeframe: Timeframe
    year: int
    model_name: str
    model_version: str


class PredictionRepository:
    """Repository facade for model prediction datasets.

    Callers identify partitions by framework, model name, model version,
    exchange, market, symbol, timeframe, and year. Paths are composed
    privately via ``StorageLayout.prediction_path`` and never returned.
    Persistence is delegated entirely to the injected ``IDataStore``
    (commonly ``ParquetStore``) so atomic writes and alternate backends can
    be substituted without changing this API.

    Partition layout::

        predictions/{framework}/{model_name}/{version}/{exchange}/{market}/
            {symbol}/{timeframe}/{year}.parquet

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

    def discover_models(self, *, framework: str) -> tuple[str, ...]:
        """Return sorted model names with at least one version for ``framework``.

        Args:
            framework: Machine-learning framework identifier.

        Returns:
            Deterministically ordered model names discovered on disk.
        """
        base = self._predictions_root() / framework
        if not base.is_dir():
            return ()
        return tuple(sorted(path.name for path in base.iterdir() if path.is_dir()))

    def list_versions(self, *, framework: str, model_name: str) -> tuple[str, ...]:
        """Return sorted versions present for ``framework`` / ``model_name``.

        Args:
            framework: Machine-learning framework identifier.
            model_name: Stable model identifier.

        Returns:
            Deterministically ordered version identifiers for existing
            prediction trees.
        """
        return self._discover_versions(framework=framework, model_name=model_name)

    def discover_partitions(
        self,
        *,
        framework: str,
        model_names: Sequence[str] | None = None,
        versions: Sequence[str] | None = None,
        symbols: Sequence[Symbol] | None = None,
        timeframes: Sequence[Timeframe] | None = None,
        exchange: Exchange = _EXCHANGE,
        market: Market = _MARKET,
    ) -> tuple[PredictionPartitionRef, ...]:
        """Discover prediction year partitions matching optional filters.

        Missing prediction trees are skipped. Only year partitions that exist
        as ``{year}.parquet`` files are included. Paths are never returned.

        Args:
            framework: Machine-learning framework identifier.
            model_names: Optional model-name allowlist. ``None`` discovers
                every model present under ``framework``.
            versions: Optional version allowlist. ``None`` discovers every
                version present for each model.
            symbols: Optional symbol allowlist. ``None`` discovers every
                symbol present under each version.
            timeframes: Optional timeframe allowlist. ``None`` discovers every
                timeframe present for each symbol.
            exchange: Exchange identifier.
            market: Market segment.

        Returns:
            Deterministically ordered partition references.
        """
        model_filter = set(model_names) if model_names is not None else None
        version_filter = set(versions) if versions is not None else None
        symbol_filter = set(symbols) if symbols is not None else None
        timeframe_filter = set(timeframes) if timeframes is not None else None

        items: list[PredictionPartitionRef] = []
        for model_name in self.discover_models(framework=framework):
            if model_filter is not None and model_name not in model_filter:
                continue
            for model_version in self._discover_versions(
                framework=framework,
                model_name=model_name,
            ):
                if version_filter is not None and model_version not in version_filter:
                    continue
                for symbol in self._discover_symbols(
                    framework=framework,
                    model_name=model_name,
                    model_version=model_version,
                    exchange=exchange,
                    market=market,
                ):
                    if symbol_filter is not None and symbol not in symbol_filter:
                        continue
                    for timeframe in self._discover_timeframes(
                        framework=framework,
                        model_name=model_name,
                        model_version=model_version,
                        exchange=exchange,
                        market=market,
                        symbol=symbol,
                    ):
                        if timeframe_filter is not None and timeframe not in timeframe_filter:
                            continue
                        for year in self._discover_years(
                            framework=framework,
                            model_name=model_name,
                            model_version=model_version,
                            exchange=exchange,
                            market=market,
                            symbol=symbol,
                            timeframe=timeframe,
                        ):
                            items.append(
                                PredictionPartitionRef(
                                    exchange=exchange,
                                    market=market,
                                    symbol=symbol,
                                    timeframe=timeframe,
                                    year=year,
                                    model_name=model_name,
                                    model_version=model_version,
                                )
                            )

        return tuple(
            sorted(
                items,
                key=lambda item: (
                    item.model_name,
                    item.model_version,
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
        framework: str,
        model_name: str,
        model_version: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> bool:
        """Return whether a prediction year partition exists.

        Existence is determined solely through ``IDataStore.exists`` on the
        path composed by ``StorageLayout.prediction_path``. Parquet contents
        are never read, scanned, or validated.

        Args:
            framework: Machine-learning framework identifier.
            model_name: Stable model identifier.
            model_version: Model version identifier.
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Prediction bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            ``True`` when the yearly prediction partition exists; otherwise
            ``False``.
        """
        path = self._layout.prediction_path(
            framework,
            model_name,
            model_version,
            exchange,
            market,
            symbol,
            timeframe,
            year,
        )
        present = self._datastore.exists(path)
        self._logger.debug(
            "Prediction dataset exists" if present else "Prediction dataset does not exist",
            extra=_prediction_log_extra(
                framework=framework,
                model_name=model_name,
                model_version=model_version,
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
        framework: str,
        model_name: str,
        model_version: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> None:
        """Persist a prediction year partition.

        Args:
            dataframe: Prediction frame to store.
            framework: Machine-learning framework identifier.
            model_name: Stable model identifier.
            model_version: Model version identifier.
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Prediction bar interval (for example ``1h``).
            year: Calendar year of the partition.
        """
        path = self._layout.prediction_path(
            framework,
            model_name,
            model_version,
            exchange,
            market,
            symbol,
            timeframe,
            year,
        )
        self._logger.debug(
            "Saving prediction dataset",
            extra=_prediction_log_extra(
                framework=framework,
                model_name=model_name,
                model_version=model_version,
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
            "Saved prediction dataset",
            extra=_prediction_log_extra(
                framework=framework,
                model_name=model_name,
                model_version=model_version,
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
        framework: str,
        model_name: str,
        model_version: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> pl.DataFrame:
        """Load a prediction year partition.

        Args:
            framework: Machine-learning framework identifier.
            model_name: Stable model identifier.
            model_version: Model version identifier.
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Prediction bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            Eagerly loaded prediction DataFrame.
        """
        path = self._layout.prediction_path(
            framework,
            model_name,
            model_version,
            exchange,
            market,
            symbol,
            timeframe,
            year,
        )
        self._logger.debug(
            "Loading prediction dataset",
            extra=_prediction_log_extra(
                framework=framework,
                model_name=model_name,
                model_version=model_version,
                exchange=exchange,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
            ),
        )
        frame = self._datastore.read(path)
        self._logger.info(
            "Loaded prediction dataset",
            extra=_prediction_log_extra(
                framework=framework,
                model_name=model_name,
                model_version=model_version,
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
        framework: str,
        model_name: str,
        model_version: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> None:
        """Delete a prediction year partition.

        Args:
            framework: Machine-learning framework identifier.
            model_name: Stable model identifier.
            model_version: Model version identifier.
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Prediction bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Raises:
            DatasetNotFoundError: If the partition does not exist (propagated
                from the injected datastore).
        """
        path = self._layout.prediction_path(
            framework,
            model_name,
            model_version,
            exchange,
            market,
            symbol,
            timeframe,
            year,
        )
        self._logger.debug(
            "Deleting prediction dataset",
            extra=_prediction_log_extra(
                framework=framework,
                model_name=model_name,
                model_version=model_version,
                exchange=exchange,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
            ),
        )
        self._datastore.delete(path)
        self._logger.info(
            "Deleted prediction dataset",
            extra=_prediction_log_extra(
                framework=framework,
                model_name=model_name,
                model_version=model_version,
                exchange=exchange,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
            ),
        )

    def _predictions_root(self) -> Path:
        """Return the predictions tier root directory."""
        return self._layout.root / STORAGE_DIR_PREDICTIONS

    def _discover_versions(
        self,
        *,
        framework: str,
        model_name: str,
    ) -> tuple[str, ...]:
        """Return sorted version directories under a model."""
        base = self._predictions_root() / framework / model_name
        if not base.is_dir():
            return ()
        return tuple(sorted(path.name for path in base.iterdir() if path.is_dir()))

    def _discover_symbols(
        self,
        *,
        framework: str,
        model_name: str,
        model_version: str,
        exchange: Exchange,
        market: Market,
    ) -> tuple[Symbol, ...]:
        """Return sorted symbols under a model version tree."""
        base = self._predictions_root() / framework / model_name / model_version / exchange / market
        if not base.is_dir():
            return ()
        return tuple(sorted(path.name for path in base.iterdir() if path.is_dir()))

    def _discover_timeframes(
        self,
        *,
        framework: str,
        model_name: str,
        model_version: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
    ) -> tuple[Timeframe, ...]:
        """Return sorted timeframes under a symbol tree."""
        base = (
            self._predictions_root()
            / framework
            / model_name
            / model_version
            / exchange
            / market
            / symbol
        )
        if not base.is_dir():
            return ()
        return tuple(sorted(path.name for path in base.iterdir() if path.is_dir()))

    def _discover_years(
        self,
        *,
        framework: str,
        model_name: str,
        model_version: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
    ) -> tuple[int, ...]:
        """Return sorted calendar years present as parquet partitions."""
        base = (
            self._predictions_root()
            / framework
            / model_name
            / model_version
            / exchange
            / market
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


def _prediction_log_extra(
    *,
    framework: str,
    model_name: str,
    model_version: str,
    exchange: Exchange,
    market: Market,
    symbol: Symbol,
    timeframe: Timeframe,
    year: int,
    rows: int | None = None,
    columns: int | None = None,
) -> dict[str, object]:
    """Build structured log fields for a prediction dataset operation."""
    payload: dict[str, object] = {
        "tier": "predictions",
        "framework": framework,
        "model_name": model_name,
        "model_version": model_version,
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
