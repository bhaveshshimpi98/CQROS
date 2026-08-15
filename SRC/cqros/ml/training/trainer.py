"""CQROS ML ModelTrainer orchestration.

Purpose:
    Orchestrate end-to-end model fitting using registered CQROS model
    implementations without coupling to framework-specific details.

Responsibilities:
    - Resolve models from an injected ``ModelRegistry``
    - Validate training inputs against model metadata contracts
    - Call ``model.fit`` and measure wall-clock duration
    - Return an immutable ``TrainerResult``
    - Remain free of splitting, scaling, evaluation, HPO, and persistence

Dependencies:
    ``polars``, ``cqros.ml.models.interfaces.Model``,
    ``cqros.ml.models.registry.ModelRegistry``,
    ``cqros.ml.training.exceptions``, and ``cqros.ml.training.interfaces``.

Public API:
    ``ModelTrainer``, ``TrainerResult``
"""

from __future__ import annotations

import logging
import time
from typing import Any, Final, cast

import polars as pl

from cqros.ml.models.interfaces import Model
from cqros.ml.models.registry import ModelRegistry
from cqros.ml.training.exceptions import ModelValidationError
from cqros.ml.training.interfaces import TrainerResult

__all__ = [
    "ModelTrainer",
    "TrainerResult",
]

_logger = logging.getLogger(__name__)

_ERROR_REGISTRY_TYPE: Final[str] = "ML-TRAINER-001"
_ERROR_FRAME_TYPE: Final[str] = "ML-TRAINER-002"
_ERROR_FRAME_EMPTY: Final[str] = "ML-TRAINER-003"
_ERROR_MISSING_COLUMNS: Final[str] = "ML-TRAINER-004"


class ModelTrainer:
    """Framework-independent orchestrator for registered model training.

    The trainer retrieves a model by name from ``ModelRegistry``, validates
    the training frame against the model's metadata column contract, invokes
    ``fit``, and returns a ``TrainerResult``. It never splits, scales,
    evaluates, tunes, or saves models.

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
        """Initialize the trainer with an injected model registry.

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

    def train(
        self,
        model_name: str,
        train_frame: pl.DataFrame,
        validation_frame: pl.DataFrame | None = None,
    ) -> TrainerResult:
        """Train the registered model named ``model_name``.

        Args:
            model_name: Registry key of the model to train.
            train_frame: Training dataset. Must not be mutated.
            validation_frame: Optional validation dataset. Must not be mutated.

        Returns:
            Immutable ``TrainerResult`` describing the completed fit.

        Raises:
            ModelValidationError: If the model is unknown, frames are empty,
                or required columns are missing.
        """
        model = self._registry.get(model_name)
        metadata = model.metadata()

        validated_train = _require_training_frame(train_frame, parameter="train_frame")
        _require_model_columns(validated_train, model, parameter="train_frame")

        validated_validation: pl.DataFrame | None = None
        if validation_frame is not None:
            validated_validation = _require_training_frame(
                validation_frame,
                parameter="validation_frame",
            )
            _require_model_columns(
                validated_validation,
                model,
                parameter="validation_frame",
            )

        self._logger.info(
            "Starting model training",
            extra={
                "model_name": model_name,
                "train_rows": validated_train.height,
                "validation_rows": (
                    validated_validation.height if validated_validation is not None else 0
                ),
            },
        )

        started = time.perf_counter()
        fitted_model = _fit_model(
            model,
            validated_train,
            validation_frame=validated_validation,
        )
        duration = time.perf_counter() - started

        result = TrainerResult(
            model_metadata=fitted_model.metadata(),
            train_rows=validated_train.height,
            validation_rows=(
                validated_validation.height if validated_validation is not None else 0
            ),
            test_rows=0,
            feature_count=len(metadata.feature_columns),
            label_column=metadata.label_column,
            training_duration=duration,
            fitted_model=fitted_model,
        )

        self._logger.info(
            "Completed model training",
            extra={
                "model_name": model_name,
                "train_rows": result.train_rows,
                "validation_rows": result.validation_rows,
                "training_duration": result.training_duration,
            },
        )
        return result


def _require_training_frame(frame: object, *, parameter: str) -> pl.DataFrame:
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


def _fit_model(
    model: Model,
    train_frame: pl.DataFrame,
    *,
    validation_frame: pl.DataFrame | None,
) -> Model:
    """Fit ``model`` through the shared Model surface.

    Concrete CQROS models accept an optional ``validation_frame`` keyword.
    When absent, only the positional training frame is passed.
    """
    if validation_frame is None:
        return model.fit(train_frame)
    return cast(
        Model,
        cast(Any, model).fit(train_frame, validation_frame=validation_frame),
    )
