"""CQROS Predictions package pipeline.

Purpose:
    Orchestrate deterministic assembly of canonical prediction datasets from
    model inference outputs, including prediction-result validation, merged-
    schema finalization, and persistence through ``PredictionRepository``.

Responsibilities:
    - Delegate inference exclusively to an injected ``InferencePipeline``
    - Validate ``PredictionResult`` structural integrity
    - Validate required primary-key columns on the feature frame
    - Reject duplicate primary keys
    - Construct the canonical prediction dataset from primary keys, model
      identity, and inference predictions
    - Finalize outputs against the canonical merged prediction schema
    - Persist the partition through an injected ``PredictionRepository``
    - Preserve feature-frame and ``PredictionResult`` immutability
    - Remain free of model training, signal generation, feature engineering,
      joins, verification, and CLI logic

Dependencies:
    ``polars``, ``cqros.ml.inference.result``, ``cqros.predictions.exceptions``,
    ``cqros.predictions.interfaces``, ``cqros.predictions.schema``, and
    ``cqros.storage.prediction_repository``.

Public API:
    ``PredictionPipeline``
"""

from __future__ import annotations

import logging
from typing import Final

import polars as pl

from cqros.ml.inference.result import PredictionResult
from cqros.predictions.exceptions import PredictionValidationError
from cqros.predictions.interfaces import InferencePipeline
from cqros.predictions.schema import (
    CANONICAL_COLUMN_ORDER,
    MERGED_PREDICTION_SCHEMA,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
)
from cqros.storage.prediction_repository import (
    PredictionPartitionRef,
    PredictionRepository,
)

__all__ = ["PredictionPipeline"]

_ERROR_INVALID_PREDICTION_RESULT: Final[str] = "PRED-PIPE-001"
_ERROR_MISSING_PRIMARY_KEY: Final[str] = "PRED-PIPE-002"
_ERROR_DUPLICATE_KEYS: Final[str] = "PRED-PIPE-003"
_ERROR_MISSING_COLUMNS: Final[str] = "PRED-PIPE-004"
_ERROR_LENGTH_MISMATCH: Final[str] = "PRED-PIPE-005"

_PRIMARY_KEY_LIST: Final[list[str]] = list(PRIMARY_KEY_COLUMNS)

_logger = logging.getLogger(__name__)


class PredictionPipeline:
    """Deterministic orchestrator for canonical prediction dataset assembly.

    The pipeline delegates inference to an injected ``InferencePipeline``,
    validates the resulting ``PredictionResult``, constructs the canonical
    prediction frame from feature primary keys and model identity, finalizes
    against the merged prediction schema, and persists the partition through
    ``PredictionRepository``. Caller-supplied feature frames are never
    mutated. Inference semantics remain exclusively in the injected pipeline.

    Args:
        inference_pipeline: Inference facade implementing ``InferencePipeline``.
        repository: Persistence facade for prediction partitions.
        logger: Optional logger instance. Defaults to the module logger.
    """

    __slots__ = ("_inference_pipeline", "_logger", "_repository")

    _inference_pipeline: InferencePipeline
    _repository: PredictionRepository
    _logger: logging.Logger

    def __init__(
        self,
        inference_pipeline: InferencePipeline,
        repository: PredictionRepository,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the pipeline with inference and repository dependencies.

        Args:
            inference_pipeline: Pipeline used to produce ``PredictionResult``.
            repository: Repository used to persist finalized partitions.
            logger: Optional logger instance.
        """
        self._inference_pipeline = inference_pipeline
        self._repository = repository
        self._logger = logger if logger is not None else _logger

    def run(
        self,
        model_name: str,
        model_version: str,
        feature_frame: pl.DataFrame,
        partition_ref: PredictionPartitionRef,
    ) -> pl.DataFrame:
        """Generate, finalize, and persist a canonical prediction partition.

        Primary keys on ``feature_frame`` are validated first. Inference is
        then delegated to ``InferencePipeline.predict``. The returned
        ``PredictionResult`` is validated, combined with primary-key columns
        and model identity into the prediction schema, checked against
        ``REQUIRED_COLUMNS``, rejected when primary keys are duplicated,
        reordered to ``CANONICAL_COLUMN_ORDER``, cast to
        ``MERGED_PREDICTION_SCHEMA``, and saved through
        ``PredictionRepository``. The original ``feature_frame`` is never
        mutated.

        Args:
            model_name: Registry key / stable model identifier.
            model_version: Model version written into the prediction dataset.
            feature_frame: Feature DataFrame containing primary-key columns.
            partition_ref: Partition identity used for persistence.

        Returns:
            A new DataFrame containing the finalized merged prediction matrix.

        Raises:
            PredictionValidationError: If ``PredictionResult`` is invalid,
                required primary-key columns are missing, primary keys are
                duplicated, or required schema columns are absent.
        """
        _require_primary_key_columns(feature_frame)
        _require_unique_primary_keys(feature_frame)

        prediction_result = self._inference_pipeline.predict(
            model_name,
            feature_frame,
        )
        _require_valid_prediction_result(prediction_result)
        _require_prediction_length(prediction_result, feature_frame)

        assembled = _assemble(
            feature_frame,
            model_name=model_name,
            model_version=model_version,
            predictions=prediction_result.predictions,
        )
        finalized = _finalize(assembled)
        framework = str(prediction_result.model_metadata.framework)

        self._logger.debug(
            "Persisting prediction partition",
            extra={
                "framework": framework,
                "model_name": model_name,
                "model_version": model_version,
                "exchange": partition_ref.exchange,
                "market": partition_ref.market,
                "symbol": partition_ref.symbol,
                "timeframe": partition_ref.timeframe,
                "year": partition_ref.year,
                "rows": finalized.height,
                "columns": finalized.width,
                "prediction_count": prediction_result.prediction_count,
            },
        )
        self._repository.save(
            finalized,
            framework=framework,
            model_name=model_name,
            model_version=model_version,
            exchange=partition_ref.exchange,
            market=partition_ref.market,
            symbol=partition_ref.symbol,
            timeframe=partition_ref.timeframe,
            year=partition_ref.year,
        )
        self._logger.info(
            "Persisted prediction partition",
            extra={
                "framework": framework,
                "model_name": model_name,
                "model_version": model_version,
                "exchange": partition_ref.exchange,
                "market": partition_ref.market,
                "symbol": partition_ref.symbol,
                "timeframe": partition_ref.timeframe,
                "year": partition_ref.year,
                "rows": finalized.height,
                "columns": finalized.width,
                "prediction_count": prediction_result.prediction_count,
            },
        )
        return finalized


def _assemble(
    feature_frame: pl.DataFrame,
    *,
    model_name: str,
    model_version: str,
    predictions: pl.Series,
) -> pl.DataFrame:
    """Build the canonical prediction frame without mutating inputs.

    Args:
        feature_frame: Feature frame providing primary-key columns.
        model_name: Stable model identifier.
        model_version: Model version identifier.
        predictions: Inference prediction series aligned to feature rows.

    Returns:
        A new DataFrame containing primary keys, model metadata, and
        predictions.
    """
    return feature_frame.select(_PRIMARY_KEY_LIST).with_columns(
        pl.lit(model_name).alias("model_name"),
        pl.lit(model_version).alias("model_version"),
        predictions.alias("prediction"),
    )


def _finalize(frame: pl.DataFrame) -> pl.DataFrame:
    """Apply schema checks, uniqueness checks, ordering, and casting.

    Args:
        frame: Assembled prediction DataFrame.

    Returns:
        Finalized merged prediction DataFrame.

    Raises:
        PredictionValidationError: If required columns are missing or primary
            keys are duplicated.
    """
    _require_schema_columns(frame)
    _require_unique_primary_keys(frame)
    ordered = frame.select(list(CANONICAL_COLUMN_ORDER))
    return ordered.cast(MERGED_PREDICTION_SCHEMA)


def _require_valid_prediction_result(prediction_result: object) -> None:
    """Raise when ``prediction_result`` fails structural integrity checks.

    Args:
        prediction_result: Candidate inference outcome.

    Raises:
        PredictionValidationError: If the value is not a valid
            ``PredictionResult``.
    """
    if not isinstance(prediction_result, PredictionResult):
        raise PredictionValidationError(
            "prediction_result must be a PredictionResult instance",
            error_code=_ERROR_INVALID_PREDICTION_RESULT,
            details={
                "actual_type": type(prediction_result).__name__,
            },
        )
    if prediction_result.prediction_count < 0:
        raise PredictionValidationError(
            "prediction_result.prediction_count must be non-negative",
            error_code=_ERROR_INVALID_PREDICTION_RESULT,
            details={
                "prediction_count": prediction_result.prediction_count,
            },
        )
    if prediction_result.prediction_time < 0.0:
        raise PredictionValidationError(
            "prediction_result.prediction_time must be non-negative",
            error_code=_ERROR_INVALID_PREDICTION_RESULT,
            details={
                "prediction_time": prediction_result.prediction_time,
            },
        )
    series_length = prediction_result.predictions.len()
    if prediction_result.prediction_count != series_length:
        raise PredictionValidationError(
            "prediction_result.prediction_count does not match predictions length",
            error_code=_ERROR_INVALID_PREDICTION_RESULT,
            details={
                "prediction_count": prediction_result.prediction_count,
                "predictions_length": series_length,
            },
        )


def _require_prediction_length(
    prediction_result: PredictionResult,
    feature_frame: pl.DataFrame,
) -> None:
    """Raise when prediction row count does not match the feature frame.

    Args:
        prediction_result: Validated inference outcome.
        feature_frame: Feature frame used for inference.

    Raises:
        PredictionValidationError: If lengths differ.
    """
    if prediction_result.prediction_count != feature_frame.height:
        raise PredictionValidationError(
            "prediction_result length does not match feature_frame row count",
            error_code=_ERROR_LENGTH_MISMATCH,
            details={
                "prediction_count": prediction_result.prediction_count,
                "feature_rows": feature_frame.height,
            },
        )


def _require_primary_key_columns(frame: pl.DataFrame) -> None:
    """Raise when any primary-key column is missing from ``frame``.

    Args:
        frame: Feature input DataFrame.

    Raises:
        PredictionValidationError: If one or more primary-key columns are
            absent.
    """
    missing = [column for column in PRIMARY_KEY_COLUMNS if column not in frame.columns]
    if missing:
        raise PredictionValidationError(
            "feature_frame is missing required primary-key columns",
            error_code=_ERROR_MISSING_PRIMARY_KEY,
            details={
                "missing_columns": tuple(missing),
                "required_columns": PRIMARY_KEY_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )


def _require_unique_primary_keys(frame: pl.DataFrame) -> None:
    """Raise when primary-key combinations are duplicated in ``frame``.

    Args:
        frame: DataFrame containing primary-key columns.

    Raises:
        PredictionValidationError: If any primary-key combination appears more
            than once.
    """
    unique_keys = frame.select(_PRIMARY_KEY_LIST).n_unique()
    if unique_keys != frame.height:
        raise PredictionValidationError(
            "prediction frame contains duplicate primary keys",
            error_code=_ERROR_DUPLICATE_KEYS,
            details={
                "primary_key_columns": PRIMARY_KEY_COLUMNS,
                "row_count": frame.height,
                "unique_key_count": unique_keys,
            },
        )


def _require_schema_columns(frame: pl.DataFrame) -> None:
    """Raise when any required merged-schema column is missing.

    Raises:
        PredictionValidationError: If one or more ``REQUIRED_COLUMNS`` are
            absent.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise PredictionValidationError(
            "merged prediction schema is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "missing_columns": tuple(missing),
                "required_columns": REQUIRED_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )
