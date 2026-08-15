"""CQROS Training package pipeline.

Purpose:
    Orchestrate deterministic assembly of the canonical machine-learning
    dataset by joining verified feature and label partitions, including
    primary-key validation, inner-join alignment, merged-schema
    finalization, and persistence through ``TrainingRepository``.

Responsibilities:
    - Validate required primary-key columns on feature and label inputs
    - Reject duplicate join keys on either input
    - Inner-join features and labels on the shared primary key
    - Finalize outputs against the canonical merged training schema
    - Persist the merged partition through an injected ``TrainingRepository``
    - Preserve input DataFrame immutability
    - Remain free of feature computation, label computation, verification,
      CLI, scaling, and model-training logic

Dependencies:
    ``polars``, ``cqros.core.types``, ``cqros.training.exceptions``,
    ``cqros.training.schema``, and ``cqros.storage.training_repository``.

Public API:
    ``TrainingPipeline``
"""

from __future__ import annotations

import logging
from typing import Final

import polars as pl

from cqros.core.types import Exchange, Market, Symbol, Timeframe
from cqros.storage.training_repository import TrainingRepository
from cqros.training.exceptions import TrainingValidationError
from cqros.training.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
)

__all__ = ["TrainingPipeline"]

_ERROR_MISSING_PRIMARY_KEY: Final[str] = "TRAINING-PIPE-001"
_ERROR_DUPLICATE_KEYS: Final[str] = "TRAINING-PIPE-002"
_ERROR_MISSING_COLUMNS: Final[str] = "TRAINING-PIPE-003"

_JOIN_KEYS: Final[list[str]] = list(PRIMARY_KEY_COLUMNS)

_logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Deterministic orchestrator for merged training dataset assembly.

    The pipeline validates primary keys on feature and label inputs, rejects
    duplicate join keys, performs an inner join on the shared primary key,
    finalizes the result to the canonical merged training schema, and
    persists the partition through ``TrainingRepository``. Caller-supplied
    input frames are never mutated.

    Args:
        repository: Persistence facade for merged training partitions.
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
        """Initialize the pipeline with a training repository.

        Args:
            repository: Repository used to persist finalized partitions.
            logger: Optional logger instance.
        """
        self._repository = repository
        self._logger = logger if logger is not None else _logger

    def run(
        self,
        features: pl.DataFrame,
        labels: pl.DataFrame,
        *,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> pl.DataFrame:
        """Join, finalize, and persist merged training data.

        Both inputs must contain the shared primary key
        (``symbol``, ``timeframe``, ``open_time``) with unique join keys.
        Features and labels are inner-joined on that key. The result is
        checked against ``REQUIRED_COLUMNS``, reordered to
        ``CANONICAL_COLUMN_ORDER``, cast to ``COLUMN_DTYPES``, and saved
        through ``TrainingRepository``. The original ``features`` and
        ``labels`` frames are never mutated.

        Args:
            features: Feature DataFrame for the partition.
            labels: Label DataFrame for the partition.
            exchange: Exchange identifier for the persisted partition.
            market: Market segment for the persisted partition.
            symbol: Tradeable symbol for the persisted partition.
            timeframe: Training bar interval for the persisted partition.
            year: Calendar year of the persisted partition.

        Returns:
            A new DataFrame containing the finalized merged training matrix.

        Raises:
            TrainingValidationError: If primary keys are missing, join keys
                are duplicated, or required merged-schema columns are absent.
        """
        _require_primary_key_columns(features, side="features")
        _require_primary_key_columns(labels, side="labels")
        _require_unique_join_keys(features, side="features")
        _require_unique_join_keys(labels, side="labels")

        joined = _inner_join(features, labels)
        finalized = _finalize(joined)

        self._logger.debug(
            "Persisting merged training partition",
            extra={
                "exchange": exchange,
                "market": market,
                "symbol": symbol,
                "timeframe": timeframe,
                "year": year,
                "rows": finalized.height,
                "columns": finalized.width,
                "feature_rows": features.height,
                "label_rows": labels.height,
            },
        )
        self._repository.save(
            finalized,
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        )
        self._logger.info(
            "Persisted merged training partition",
            extra={
                "exchange": exchange,
                "market": market,
                "symbol": symbol,
                "timeframe": timeframe,
                "year": year,
                "rows": finalized.height,
                "columns": finalized.width,
                "feature_rows": features.height,
                "label_rows": labels.height,
            },
        )
        return finalized


def _inner_join(features: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    """Inner-join features and labels on the shared primary key.

    Args:
        features: Validated feature DataFrame.
        labels: Validated label DataFrame.

    Returns:
        A new DataFrame containing only rows present in both inputs.
    """
    return features.join(labels, on=_JOIN_KEYS, how="inner")


def _finalize(frame: pl.DataFrame) -> pl.DataFrame:
    """Apply merged-schema checks, ordering, and casting.

    Args:
        frame: Frame produced by the feature/label inner join.

    Returns:
        Finalized merged training DataFrame.

    Raises:
        TrainingValidationError: If any required schema column is missing.
    """
    _require_schema_columns(frame)
    ordered = frame.select(list(CANONICAL_COLUMN_ORDER))
    return ordered.cast(COLUMN_DTYPES)


def _require_primary_key_columns(frame: pl.DataFrame, *, side: str) -> None:
    """Raise when any primary-key column is missing from ``frame``.

    Args:
        frame: Feature or label input DataFrame.
        side: Human-readable input identity (``features`` or ``labels``).

    Raises:
        TrainingValidationError: If one or more primary-key columns are absent.
    """
    missing = [column for column in PRIMARY_KEY_COLUMNS if column not in frame.columns]
    if missing:
        raise TrainingValidationError(
            f"{side} input is missing required primary-key columns",
            error_code=_ERROR_MISSING_PRIMARY_KEY,
            details={
                "side": side,
                "missing_columns": tuple(missing),
                "required_columns": PRIMARY_KEY_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )


def _require_unique_join_keys(frame: pl.DataFrame, *, side: str) -> None:
    """Raise when primary-key combinations are duplicated in ``frame``.

    Args:
        frame: Feature or label input DataFrame with primary-key columns.
        side: Human-readable input identity (``features`` or ``labels``).

    Raises:
        TrainingValidationError: If any primary-key combination appears more
            than once.
    """
    unique_keys = frame.select(_JOIN_KEYS).n_unique()
    if unique_keys != frame.height:
        raise TrainingValidationError(
            f"{side} input contains duplicate join keys",
            error_code=_ERROR_DUPLICATE_KEYS,
            details={
                "side": side,
                "join_keys": PRIMARY_KEY_COLUMNS,
                "row_count": frame.height,
                "unique_key_count": unique_keys,
            },
        )


def _require_schema_columns(frame: pl.DataFrame) -> None:
    """Raise when any required merged-schema column is missing.

    Raises:
        TrainingValidationError: If one or more ``REQUIRED_COLUMNS`` are absent.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise TrainingValidationError(
            "merged training schema is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "missing_columns": tuple(missing),
                "required_columns": REQUIRED_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )
