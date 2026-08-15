"""CQROS ML PredictionPipeline orchestration.

Purpose:
    Orchestrate framework-independent inference using registered CQROS models
    without coupling to training, evaluation, or framework-specific internals.

Responsibilities:
    - Resolve models from an injected ``ModelRegistry``
    - Validate inference frames against model feature-column contracts
    - Call ``model.predict`` and measure wall-clock duration
    - Return an immutable ``PredictionResult``
    - Remain free of training, evaluation, HPO, persistence, and signal logic

Dependencies:
    ``polars``, ``cqros.ml.models.interfaces.Model``,
    ``cqros.ml.models.registry.ModelRegistry``,
    ``cqros.ml.inference.exceptions``, and ``cqros.ml.inference.result``.

Public API:
    ``PredictionPipeline``, ``PredictionResult``
"""

from __future__ import annotations

import logging
import time
from typing import Final, cast

import polars as pl

from cqros.ml.inference.exceptions import ModelValidationError
from cqros.ml.inference.result import PredictionResult
from cqros.ml.models.interfaces import Model
from cqros.ml.models.registry import ModelRegistry

__all__ = [
    "PredictionPipeline",
    "PredictionResult",
]

_logger = logging.getLogger(__name__)

_ERROR_REGISTRY_TYPE: Final[str] = "ML-INFER-001"
_ERROR_FRAME_TYPE: Final[str] = "ML-INFER-002"
_ERROR_FRAME_EMPTY: Final[str] = "ML-INFER-003"
_ERROR_MISSING_FEATURES: Final[str] = "ML-INFER-004"


class PredictionPipeline:
    """Framework-independent orchestrator for registered-model inference.

    The pipeline retrieves a model by name from ``ModelRegistry``, validates
    the inference frame against the model's feature-column contract, invokes
    ``predict``, and returns a ``PredictionResult``. Fitted-state validation is
    enforced by the shared ``Model.predict`` contract. The pipeline never
    trains, evaluates, tunes, or saves models.

    Args:
        model_registry: Catalog of registered model implementations.
        logger: Optional logger instance. Defaults to the module logger.
    """

    __slots__ = ("_logger", "_registry")

    _registry: ModelRegistry
    _logger: logging.Logger

    def __init__(
        self,
        model_registry: ModelRegistry,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the pipeline with an injected model registry.

        Args:
            model_registry: Registry used to resolve models by name.
            logger: Optional logger instance.

        Raises:
            ModelValidationError: If ``model_registry`` is not a
                ``ModelRegistry``.
        """
        if not isinstance(cast(object, model_registry), ModelRegistry):
            raise ModelValidationError(
                "model_registry must be a ModelRegistry instance",
                error_code=_ERROR_REGISTRY_TYPE,
                details={
                    "parameter": "model_registry",
                    "value_type": type(model_registry).__name__,
                },
            )
        self._registry = model_registry
        self._logger = logger if logger is not None else _logger

    def predict(
        self,
        model_name: str,
        frame: pl.DataFrame,
    ) -> PredictionResult:
        """Generate predictions for the registered model named ``model_name``.

        Args:
            model_name: Registry key of the fitted model to use.
            frame: Feature dataset. Must not be mutated.

        Returns:
            Immutable ``PredictionResult`` describing the completed inference.

        Raises:
            ModelValidationError: If the model is unknown or unfitted, the
                frame is empty, or required feature columns are missing.
        """
        model = self._registry.get(model_name)
        metadata = model.metadata()
        validated_frame = _require_inference_frame(frame, parameter="frame")
        _require_feature_columns(validated_frame, model, parameter="frame")

        self._logger.info(
            "Starting model inference",
            extra={
                "model_name": model_name,
                "prediction_rows": validated_frame.height,
            },
        )

        started = time.perf_counter()
        predictions = model.predict(validated_frame)
        duration = time.perf_counter() - started

        result = PredictionResult(
            model_metadata=metadata,
            prediction_count=predictions.len(),
            prediction_time=duration,
            predictions=predictions,
        )

        self._logger.info(
            "Completed model inference",
            extra={
                "model_name": model_name,
                "prediction_count": result.prediction_count,
                "prediction_time": result.prediction_time,
            },
        )
        return result


def _require_inference_frame(frame: object, *, parameter: str) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty Polars DataFrame."""
    if not isinstance(frame, pl.DataFrame):
        raise ModelValidationError(
            f"{parameter} must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={"parameter": parameter, "value_type": type(frame).__name__},
        )
    if frame.height == 0:
        raise ModelValidationError(
            f"{parameter} must contain at least one row",
            error_code=_ERROR_FRAME_EMPTY,
            details={"parameter": parameter, "rows": frame.height},
        )
    return frame


def _require_feature_columns(frame: pl.DataFrame, model: Model, *, parameter: str) -> None:
    """Validate that ``frame`` contains the model's feature columns."""
    metadata = model.metadata()
    missing = [column for column in metadata.feature_columns if column not in frame.columns]
    if missing:
        raise ModelValidationError(
            f"{parameter} is missing required feature columns",
            error_code=_ERROR_MISSING_FEATURES,
            details={
                "parameter": parameter,
                "missing_columns": tuple(missing),
                "required_feature_columns": metadata.feature_columns,
                "available_columns": tuple(frame.columns),
            },
        )
