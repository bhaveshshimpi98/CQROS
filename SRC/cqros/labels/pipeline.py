"""CQROS Label Engine pipeline.

Purpose:
    Orchestrate deterministic generation of merged regression and
    classification labels from processed OHLCV data, including horizon
    trimming, merged-schema finalization, and persistence through
    ``LabelRepository``.

Responsibilities:
    - Validate required processed-OHLCV input columns
    - Compute every canonical regression and direction label
    - Trim trailing rows covered by the maximum prediction horizon
    - Finalize outputs against the canonical merged label schema
    - Persist the merged partition through an injected ``LabelRepository``
    - Preserve input DataFrame immutability
    - Remain free of verification, CLI, and model-training logic

Dependencies:
    ``polars``, ``cqros.core.types``, ``cqros.labels.exceptions``,
    ``cqros.labels.schema``, and ``cqros.storage.label_repository``.

Public API:
    ``LabelPipeline``
"""

from __future__ import annotations

import logging
from typing import Final

import polars as pl

from cqros.core.types import Exchange, Market, Symbol, Timeframe
from cqros.labels.exceptions import LabelValidationError
from cqros.labels.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    REGRESSION_LABEL_COLUMNS,
)
from cqros.storage.label_repository import LabelRepository

__all__ = ["LabelPipeline"]

_ERROR_MISSING_INPUT_COLUMNS: Final[str] = "LABEL-PIPE-001"

_REQUIRED_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
    "close",
)

_PREDICTION_HORIZONS: Final[tuple[int, ...]] = tuple(
    int(column.removeprefix("future_return_")) for column in REGRESSION_LABEL_COLUMNS
)
_MAX_PREDICTION_HORIZON: Final[int] = max(_PREDICTION_HORIZONS)

_logger = logging.getLogger(__name__)


class LabelPipeline:
    """Deterministic orchestrator for merged label generation.

    The pipeline validates processed OHLCV inputs, computes forward-return and
    direction labels, trims the trailing maximum prediction horizon, finalizes
    the result to the canonical merged label schema, and persists the
    partition through ``LabelRepository``. The caller-supplied input frame is
    never mutated.

    Args:
        repository: Persistence facade for merged label partitions.
        logger: Optional logger instance. Defaults to the module logger.
    """

    __slots__ = ("_logger", "_repository")

    _repository: LabelRepository
    _logger: logging.Logger

    def __init__(
        self,
        repository: LabelRepository,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the pipeline with a label repository.

        Args:
            repository: Repository used to persist finalized partitions.
            logger: Optional logger instance.
        """
        self._repository = repository
        self._logger = logger if logger is not None else _logger

    def run(
        self,
        frame: pl.DataFrame,
        *,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> pl.DataFrame:
        """Generate, finalize, trim, and persist merged labels.

        Required input columns are validated first. Regression and direction
        labels are then computed for every configured horizon. Trailing rows
        covered by the maximum prediction horizon are removed so finalized
        labels contain no incomplete forward windows. The frame is reordered
        to ``CANONICAL_COLUMN_ORDER``, cast to ``COLUMN_DTYPES``, and saved
        through ``LabelRepository``. The original ``frame`` is never mutated.

        Args:
            frame: Processed OHLCV DataFrame.
            exchange: Exchange identifier for the persisted partition.
            market: Market segment for the persisted partition.
            symbol: Tradeable symbol for the persisted partition.
            timeframe: Label bar interval for the persisted partition.
            year: Calendar year of the persisted partition.

        Returns:
            A new DataFrame containing the finalized merged label matrix.

        Raises:
            LabelValidationError: If any required input column is missing.
        """
        _require_input_columns(frame)
        labeled = _compute_labels(frame)
        finalized = _finalize(labeled)
        self._logger.debug(
            "Persisting merged label partition",
            extra={
                "exchange": exchange,
                "market": market,
                "symbol": symbol,
                "timeframe": timeframe,
                "year": year,
                "rows": finalized.height,
                "columns": finalized.width,
                "max_horizon": _MAX_PREDICTION_HORIZON,
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
            "Persisted merged label partition",
            extra={
                "exchange": exchange,
                "market": market,
                "symbol": symbol,
                "timeframe": timeframe,
                "year": year,
                "rows": finalized.height,
                "columns": finalized.width,
                "max_horizon": _MAX_PREDICTION_HORIZON,
            },
        )
        return finalized


def _compute_labels(frame: pl.DataFrame) -> pl.DataFrame:
    """Compute regression and direction labels without mutating ``frame``.

    Args:
        frame: Validated processed OHLCV DataFrame.

    Returns:
        A new DataFrame containing the original columns plus label columns.
    """
    return_expressions = [
        ((pl.col("close").shift(-horizon) - pl.col("close")) / pl.col("close")).alias(
            f"future_return_{horizon}"
        )
        for horizon in _PREDICTION_HORIZONS
    ]
    with_returns = frame.with_columns(return_expressions)
    direction_expressions = [
        (pl.col(f"future_return_{horizon}") > 0).cast(pl.Int8).alias(f"direction_{horizon}")
        for horizon in _PREDICTION_HORIZONS
    ]
    return with_returns.with_columns(direction_expressions)


def _finalize(frame: pl.DataFrame) -> pl.DataFrame:
    """Trim the prediction horizon and apply merged-schema ordering/casting.

    Args:
        frame: Frame produced by label computation.

    Returns:
        Finalized merged label DataFrame.
    """
    trimmed = _trim_prediction_horizon(frame)
    ordered = trimmed.select(list(CANONICAL_COLUMN_ORDER))
    return ordered.cast(COLUMN_DTYPES)


def _trim_prediction_horizon(frame: pl.DataFrame) -> pl.DataFrame:
    """Remove trailing rows covered by the maximum prediction horizon.

    Args:
        frame: Labeled DataFrame prior to schema finalization.

    Returns:
        Frame with the final ``_MAX_PREDICTION_HORIZON`` rows removed.
    """
    remaining = max(0, frame.height - _MAX_PREDICTION_HORIZON)
    return frame.head(remaining)


def _require_input_columns(frame: pl.DataFrame) -> None:
    """Raise when any required processed-OHLCV input column is missing.

    Raises:
        LabelValidationError: If one or more required input columns are absent.
    """
    missing = [column for column in _REQUIRED_INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise LabelValidationError(
            "processed OHLCV input is missing required columns",
            error_code=_ERROR_MISSING_INPUT_COLUMNS,
            details={
                "missing_columns": tuple(missing),
                "required_columns": _REQUIRED_INPUT_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )
