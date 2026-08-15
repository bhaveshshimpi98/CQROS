"""CQROS ML DatasetLoader.

Purpose:
    Load canonical training datasets from ``TrainingRepository`` into a single
    schema-consistent Polars DataFrame for ML consumers.

Responsibilities:
    - Discover matching training partitions through ``TrainingRepository``
    - Load and concatenate matching partitions
    - Preserve canonical column order, dtypes, and primary-key ordering
    - Validate optional filter parameter types
    - Remain free of splitting, scaling, training, evaluation, and CLI logic

Dependencies:
    ``polars``, ``cqros.core.constants``, ``cqros.ml.dataset.exceptions``,
    ``cqros.ml.dataset.schema``, and ``cqros.storage.training_repository``.

Public API:
    ``DatasetLoader``
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Final, cast

import polars as pl

from cqros.core.constants import EXCHANGE_BINANCE, MARKET_USDT_PERPETUAL
from cqros.core.types import Exchange, Market, Symbol, Timeframe
from cqros.ml.dataset.exceptions import DatasetLoaderError
from cqros.ml.dataset.schema import (
    CANONICAL_COLUMN_ORDER,
    MERGED_TRAINING_SCHEMA,
    PRIMARY_KEY_COLUMNS,
)
from cqros.storage.training_repository import TrainingPartitionRef, TrainingRepository

__all__ = ["DatasetLoader"]

_logger = logging.getLogger(__name__)

_EXCHANGE: Final[Exchange] = EXCHANGE_BINANCE
_MARKET: Final[Market] = MARKET_USDT_PERPETUAL

_ERROR_SYMBOLS_TYPE: Final[str] = "ML-DATASET-LOAD-001"
_ERROR_TIMEFRAMES_TYPE: Final[str] = "ML-DATASET-LOAD-002"
_ERROR_YEARS_TYPE: Final[str] = "ML-DATASET-LOAD-003"
_ERROR_SYMBOL_ENTRY: Final[str] = "ML-DATASET-LOAD-004"
_ERROR_TIMEFRAME_ENTRY: Final[str] = "ML-DATASET-LOAD-005"
_ERROR_YEAR_ENTRY: Final[str] = "ML-DATASET-LOAD-006"

_SORT_KEYS: Final[list[str]] = list(PRIMARY_KEY_COLUMNS)
_COLUMN_ORDER: Final[list[str]] = list(CANONICAL_COLUMN_ORDER)


class DatasetLoader:
    """Assemble canonical ML datasets from ``TrainingRepository`` partitions.

    The loader discovers partitions through the injected repository, optionally
    filters by symbol, timeframe, and year, loads every matching partition,
    concatenates them, and returns a new DataFrame ordered by the canonical
    primary key with canonical column order and dtypes. Repository frames are
    never mutated. Paths and storage backends are never accessed directly.

    Args:
        repository: Training repository used for discovery and partition loads.
        logger: Optional logger instance. Defaults to the module logger.
    """

    __slots__ = ("_logger", "_repository")

    _repository: TrainingRepository
    _logger: logging.Logger

    def __init__(
        self,
        repository: TrainingRepository,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the loader with a training repository.

        Args:
            repository: Repository used to discover and load partitions.
            logger: Optional logger instance.
        """
        self._repository = repository
        self._logger = logger if logger is not None else _logger

    def load(
        self,
        *,
        symbols: Sequence[Symbol] | None = None,
        timeframes: Sequence[Timeframe] | None = None,
        years: Sequence[int] | None = None,
    ) -> pl.DataFrame:
        """Load matching training partitions into one canonical DataFrame.

        All filters are optional. When no filters are supplied, every training
        partition discovered by the repository is loaded. When no partitions
        match, an empty DataFrame with ``MERGED_TRAINING_SCHEMA`` is returned.

        Args:
            symbols: Optional symbol allowlist.
            timeframes: Optional timeframe allowlist.
            years: Optional calendar-year allowlist.

        Returns:
            A new Polars DataFrame in canonical column order, sorted by
            ``symbol``, ``timeframe``, ``open_time``.

        Raises:
            DatasetLoaderError: If any filter argument has an invalid type.
        """
        symbol_filter = _normalize_string_sequence(
            symbols,
            parameter="symbols",
            type_error_code=_ERROR_SYMBOLS_TYPE,
            entry_error_code=_ERROR_SYMBOL_ENTRY,
        )
        timeframe_filter = _normalize_string_sequence(
            timeframes,
            parameter="timeframes",
            type_error_code=_ERROR_TIMEFRAMES_TYPE,
            entry_error_code=_ERROR_TIMEFRAME_ENTRY,
        )
        year_filter = _normalize_year_sequence(years)

        self._logger.debug(
            "Loading ML dataset",
            extra={
                "symbols": symbol_filter,
                "timeframes": timeframe_filter,
                "years": year_filter,
            },
        )

        partitions = self._repository.discover_partitions(
            symbols=symbol_filter,
            timeframes=timeframe_filter,
            exchange=_EXCHANGE,
            market=_MARKET,
        )
        if year_filter is not None:
            year_set = set(year_filter)
            partitions = tuple(partition for partition in partitions if partition.year in year_set)

        if not partitions:
            self._logger.info(
                "Loaded empty ML dataset",
                extra={"partition_count": 0, "rows": 0},
            )
            return pl.DataFrame(schema=MERGED_TRAINING_SCHEMA)

        frames = [_load_partition(self._repository, partition) for partition in partitions]
        combined = pl.concat(frames, how="vertical")
        result = combined.select(_COLUMN_ORDER).cast(MERGED_TRAINING_SCHEMA).sort(_SORT_KEYS)

        self._logger.info(
            "Loaded ML dataset",
            extra={
                "partition_count": len(partitions),
                "rows": result.height,
                "columns": result.width,
            },
        )
        return result


def _load_partition(
    repository: TrainingRepository,
    partition: TrainingPartitionRef,
) -> pl.DataFrame:
    """Load one training partition through the repository."""
    return repository.load(
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=partition.symbol,
        timeframe=partition.timeframe,
        year=partition.year,
    )


def _normalize_string_sequence(
    value: object,
    *,
    parameter: str,
    type_error_code: str,
    entry_error_code: str,
) -> tuple[str, ...] | None:
    """Validate an optional string sequence filter.

    Args:
        value: Candidate filter sequence, or ``None``.
        parameter: Parameter name used in error messages.
        type_error_code: Error code when ``value`` is not a string sequence.
        entry_error_code: Error code when an entry is not a string.

    Returns:
        An immutable tuple copy of the filter, or ``None``.

    Raises:
        DatasetLoaderError: If ``value`` is not a sequence of strings.
    """
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DatasetLoaderError(
            f"{parameter} must be a sequence of strings",
            error_code=type_error_code,
            details={"parameter": parameter, "value_type": type(value).__name__},
        )

    sequence = cast(Sequence[object], value)
    frozen: list[str] = []
    for index, entry in enumerate(sequence):
        if not isinstance(entry, str):
            raise DatasetLoaderError(
                f"{parameter} entries must be strings",
                error_code=entry_error_code,
                details={
                    "parameter": parameter,
                    "index": index,
                    "value": entry,
                    "value_type": type(entry).__name__,
                },
            )
        frozen.append(entry)
    return tuple(frozen)


def _normalize_year_sequence(value: object) -> tuple[int, ...] | None:
    """Validate an optional integer year sequence filter.

    Args:
        value: Candidate year filter sequence, or ``None``.

    Returns:
        An immutable tuple copy of the year filter, or ``None``.

    Raises:
        DatasetLoaderError: If ``value`` is not a sequence of integers.
    """
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DatasetLoaderError(
            "years must be a sequence of integers",
            error_code=_ERROR_YEARS_TYPE,
            details={"parameter": "years", "value_type": type(value).__name__},
        )

    sequence = cast(Sequence[object], value)
    frozen: list[int] = []
    for index, entry in enumerate(sequence):
        if not isinstance(entry, int) or isinstance(entry, bool):
            raise DatasetLoaderError(
                "years entries must be integers",
                error_code=_ERROR_YEAR_ENTRY,
                details={
                    "parameter": "years",
                    "index": index,
                    "value": entry,
                    "value_type": type(entry).__name__,
                },
            )
        frozen.append(entry)
    return tuple(frozen)
