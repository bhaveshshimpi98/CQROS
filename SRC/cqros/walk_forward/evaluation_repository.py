"""CQROS walk-forward evaluation-result repository.

Purpose:
    Provide a path-free facade for persisting and retrieving Walk-Forward
    evaluation observation partitions under ``walk_forward_evaluation``.

Responsibilities:
    - Resolve storage locations via ``StorageLayout.walk_forward_evaluation_path``
    - Persist, load, check existence, and discover evaluation partitions
    - Validate frames against ``EVALUATION_OBSERVATION_SCHEMA``
    - Keep filesystem paths out of the public API
    - Remain free of fold math and trading logic

Dependencies:
    ``polars``, ``cqros.core``, ``cqros.storage``, ``cqros.walk_forward.exceptions``,
    and ``cqros.walk_forward.evaluation_schema``.

Public API:
    ``WalkForwardEvaluationPartitionRef``, ``WalkForwardEvaluationRepository``
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
    STORAGE_DIR_WALK_FORWARD_EVALUATION,
)
from cqros.core.types import Exchange, Market, Timeframe
from cqros.storage.interfaces import IDataStore
from cqros.storage.layout import StorageLayout
from cqros.walk_forward.evaluation_schema import (
    EVALUATION_OBSERVATION_COLUMNS,
    EVALUATION_OBSERVATION_SCHEMA,
)
from cqros.walk_forward.exceptions import WalkForwardError

__all__ = [
    "WalkForwardEvaluationPartitionRef",
    "WalkForwardEvaluationRepository",
]

_logger = logging.getLogger(__name__)

_EXCHANGE: Final[Exchange] = EXCHANGE_BINANCE
_MARKET: Final[Market] = MARKET_USDT_PERPETUAL

_ERROR_FRAME_TYPE: Final[str] = "WF_EVAL_REPO_FRAME_TYPE"
_ERROR_MISSING_COLUMNS: Final[str] = "WF_EVAL_REPO_MISSING_COLUMNS"
_ERROR_SCHEMA_CAST: Final[str] = "WF_EVAL_REPO_SCHEMA_CAST"


@dataclass(frozen=True, slots=True)
class WalkForwardEvaluationPartitionRef:
    """Identity of one discovered walk-forward evaluation year partition."""

    manager: str
    exchange: Exchange
    market: Market
    timeframe: Timeframe
    year: int


class WalkForwardEvaluationRepository:
    """Repository facade for walk-forward evaluation observation datasets.

    Partition layout::

        walk_forward_evaluation/{manager}/{exchange}/{market}/{timeframe}/{year}.parquet
    """

    __slots__ = ("_datastore", "_layout", "_logger")

    def __init__(
        self,
        layout: StorageLayout,
        datastore: IDataStore,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize with injected layout and datastore."""
        self._layout = layout
        self._datastore = datastore
        self._logger = logger if logger is not None else _logger

    def discover_managers(self) -> tuple[str, ...]:
        """Return sorted manager identifiers with at least one partition."""
        base = self._evaluation_root()
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

    def discover_partitions(
        self,
        *,
        managers: Sequence[str] | None = None,
        timeframes: Sequence[Timeframe] | None = None,
        exchange: Exchange = _EXCHANGE,
        market: Market = _MARKET,
    ) -> tuple[WalkForwardEvaluationPartitionRef, ...]:
        """Discover evaluation year partitions matching optional filters."""
        manager_filter = set(managers) if managers is not None else None
        timeframe_filter = set(timeframes) if timeframes is not None else None
        items: list[WalkForwardEvaluationPartitionRef] = []
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
                        WalkForwardEvaluationPartitionRef(
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
        """Return whether an evaluation year partition exists."""
        path = self._layout.walk_forward_evaluation_path(
            manager,
            exchange,
            market,
            timeframe,
            year,
        )
        return self._datastore.exists(path)

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
        """Persist an evaluation observation year partition."""
        validated = _require_observation_schema(dataframe)
        path = self._layout.walk_forward_evaluation_path(
            manager,
            exchange,
            market,
            timeframe,
            year,
        )
        self._logger.info(
            "Saving walk-forward evaluation dataset",
            extra={
                "tier": STORAGE_DIR_WALK_FORWARD_EVALUATION,
                "manager": manager,
                "exchange": exchange,
                "market": market,
                "timeframe": timeframe,
                "year": year,
                "rows": validated.height,
            },
        )
        self._datastore.write(path, validated)

    def load(
        self,
        *,
        manager: str,
        exchange: Exchange,
        market: Market,
        timeframe: Timeframe,
        year: int,
    ) -> pl.DataFrame:
        """Load an evaluation partition cast to the observation schema."""
        path = self._layout.walk_forward_evaluation_path(
            manager,
            exchange,
            market,
            timeframe,
            year,
        )
        frame = self._datastore.read(path)
        return _require_observation_schema(frame)

    def _evaluation_root(self) -> Path:
        return self._layout.root / STORAGE_DIR_WALK_FORWARD_EVALUATION

    def _manager_root(
        self,
        *,
        manager: str,
        exchange: Exchange,
        market: Market,
    ) -> Path:
        return self._evaluation_root() / manager / exchange / market

    def _discover_years(
        self,
        *,
        manager: str,
        exchange: Exchange,
        market: Market,
        timeframe: Timeframe,
    ) -> tuple[int, ...]:
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


def _require_observation_schema(frame: object) -> pl.DataFrame:
    """Validate and cast ``frame`` to the observation schema."""
    if not isinstance(frame, pl.DataFrame):
        raise WalkForwardError(
            "walk-forward evaluation frame must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={"actual_type": type(frame).__name__},
        )
    missing = [column for column in EVALUATION_OBSERVATION_COLUMNS if column not in frame.columns]
    if missing:
        raise WalkForwardError(
            "walk-forward evaluation schema is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "missing_columns": tuple(missing),
                "required_columns": EVALUATION_OBSERVATION_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )
    try:
        return frame.select(list(EVALUATION_OBSERVATION_COLUMNS)).cast(
            EVALUATION_OBSERVATION_SCHEMA
        )
    except (pl.exceptions.PolarsError, TypeError, ValueError) as exc:
        raise WalkForwardError(
            "walk-forward evaluation frame failed schema cast",
            error_code=_ERROR_SCHEMA_CAST,
            details={"reason": str(exc)},
        ) from exc
