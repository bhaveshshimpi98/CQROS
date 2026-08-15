"""CQROS ML ModelEvaluator orchestration.

Purpose:
    Orchestrate end-to-end evaluation of fitted CQROS models without coupling
    to framework-specific training or inference details.

Responsibilities:
    - Validate fitted models and evaluation frames
    - Generate predictions through the shared ``Model`` surface
    - Compare predictions against the model label column
    - Compute task-appropriate metrics
    - Return an immutable ``EvaluationResult``
    - Remain free of training, cross-validation, HPO, and persistence

Dependencies:
    ``polars``, ``cqros.ml.models.interfaces.Model``,
    ``cqros.ml.models.metadata``, ``cqros.ml.evaluation.exceptions``,
    ``cqros.ml.evaluation.interfaces``, and ``cqros.ml.evaluation.metrics``.

Public API:
    ``ModelEvaluator``, ``EvaluationResult``
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Any, Final, cast

import numpy as np
import polars as pl

from cqros.ml.evaluation.exceptions import ModelValidationError
from cqros.ml.evaluation.interfaces import EvaluationResult
from cqros.ml.evaluation.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
)
from cqros.ml.models.interfaces import Model
from cqros.ml.models.metadata import ModelTaskType

__all__ = [
    "EvaluationResult",
    "ModelEvaluator",
]

_logger = logging.getLogger(__name__)

_ERROR_MODEL_TYPE: Final[str] = "ML-EVAL-001"
_ERROR_FRAME_TYPE: Final[str] = "ML-EVAL-002"
_ERROR_FRAME_EMPTY: Final[str] = "ML-EVAL-003"
_ERROR_MISSING_COLUMNS: Final[str] = "ML-EVAL-004"
_ERROR_TASK_TYPE: Final[str] = "ML-EVAL-005"

_SUPPORTED_TASK_TYPES: Final[frozenset[ModelTaskType]] = frozenset(
    {
        ModelTaskType.REGRESSION,
        ModelTaskType.CLASSIFICATION,
    }
)


class ModelEvaluator:
    """Framework-independent orchestrator for fitted-model evaluation.

    The evaluator validates a fitted ``Model``, validates the evaluation
    frame against the model's metadata column contract, invokes
    ``predict``, computes task-appropriate metrics, and returns an
    ``EvaluationResult``. It never trains, cross-validates, tunes, or
    saves models.

    Args:
        logger: Optional logger instance. Defaults to the module logger.
    """

    __slots__ = ("_logger",)

    _logger: logging.Logger

    def __init__(self, *, logger: logging.Logger | None = None) -> None:
        """Initialize the evaluator.

        Args:
            logger: Optional logger instance.
        """
        self._logger = logger if logger is not None else _logger

    def evaluate(
        self,
        model: Model,
        frame: pl.DataFrame,
    ) -> EvaluationResult:
        """Evaluate ``model`` on ``frame``.

        Args:
            model: Fitted model implementing the ``Model`` contract.
            frame: Evaluation dataset. Must not be mutated.

        Returns:
            Immutable ``EvaluationResult`` describing the completed evaluation.

        Raises:
            ModelValidationError: If the model is unfitted, the frame is empty,
                required columns are missing, or the task type is unsupported.
        """
        validated_model = _require_model(model)
        metadata = validated_model.metadata()
        task_type = metadata.task_type
        _require_supported_task_type(task_type)

        validated_frame = _require_evaluation_frame(frame, parameter="frame")
        _require_model_columns(validated_frame, validated_model, parameter="frame")

        self._logger.info(
            "Starting model evaluation",
            extra={
                "model_name": metadata.name,
                "task_type": str(task_type),
                "dataset_rows": validated_frame.height,
            },
        )

        started = time.perf_counter()
        predictions = validated_model.predict(validated_frame)
        labels = validated_frame.get_column(metadata.label_column)
        metrics = _compute_metrics(
            task_type=task_type,
            y_true=labels,
            y_pred=predictions,
        )
        duration = time.perf_counter() - started

        result = EvaluationResult(
            model_metadata=metadata,
            task_type=task_type,
            dataset_rows=validated_frame.height,
            metrics=metrics,
            evaluation_duration=duration,
        )

        self._logger.info(
            "Completed model evaluation",
            extra={
                "model_name": metadata.name,
                "task_type": str(task_type),
                "dataset_rows": result.dataset_rows,
                "evaluation_duration": result.evaluation_duration,
            },
        )
        return result


def _require_model(model: object) -> Model:
    """Validate that ``model`` implements the ``Model`` contract."""
    if not isinstance(model, Model):
        raise ModelValidationError(
            "model must implement the Model protocol",
            error_code=_ERROR_MODEL_TYPE,
            details={
                "parameter": "model",
                "value_type": type(model).__name__,
            },
        )
    return model


def _require_supported_task_type(task_type: ModelTaskType) -> None:
    """Validate that ``task_type`` is supported by the evaluator."""
    if task_type not in _SUPPORTED_TASK_TYPES:
        raise ModelValidationError(
            "task_type must be REGRESSION or CLASSIFICATION",
            error_code=_ERROR_TASK_TYPE,
            details={"task_type": str(task_type)},
        )


def _require_evaluation_frame(frame: object, *, parameter: str) -> pl.DataFrame:
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


def _require_model_columns(frame: pl.DataFrame, model: Model, *, parameter: str) -> None:
    """Validate that ``frame`` contains the model's feature and label columns."""
    metadata = model.metadata()
    required = (*metadata.feature_columns, metadata.label_column)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ModelValidationError(
            f"{parameter} is missing required model columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "parameter": parameter,
                "missing_columns": tuple(missing),
                "required_feature_columns": metadata.feature_columns,
                "required_label_column": metadata.label_column,
                "available_columns": tuple(frame.columns),
            },
        )


def _compute_metrics(
    *,
    task_type: ModelTaskType,
    y_true: pl.Series,
    y_pred: pl.Series,
) -> Mapping[str, float]:
    """Dispatch metric computation by ``task_type``."""
    true_values = cast(
        np.ndarray[Any, np.dtype[Any]],
        y_true.to_numpy(),
    )
    pred_values = cast(
        np.ndarray[Any, np.dtype[Any]],
        y_pred.to_numpy(),
    )
    if task_type is ModelTaskType.REGRESSION:
        return compute_regression_metrics(true_values, pred_values)
    return compute_classification_metrics(true_values, pred_values)
