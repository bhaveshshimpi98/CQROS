"""CQROS ML TimeSeriesCrossValidator orchestration.

Purpose:
    Orchestrate chronological walk-forward cross-validation using injected
    ``ModelRegistry``, ``ModelTrainer``, and ``ModelEvaluator`` dependencies
    without coupling to framework-specific model internals.

Responsibilities:
    - Build expanding training windows and chronological validation segments
    - Train each fold through ``ModelTrainer``
    - Evaluate each fold through ``ModelEvaluator``
    - Aggregate mean and standard-deviation metrics across folds
    - Return an immutable ``CrossValidationResult``
    - Remain free of shuffling, HPO, experiment tracking, and persistence

Dependencies:
    ``polars``, ``cqros.ml.models.registry.ModelRegistry``,
    ``cqros.ml.training.trainer.ModelTrainer``,
    ``cqros.ml.evaluation.evaluator.ModelEvaluator``,
    ``cqros.ml.evaluation.exceptions``, and ``cqros.ml.evaluation.interfaces``.

Public API:
    ``TimeSeriesCrossValidator``, ``CrossValidationFold``,
    ``CrossValidationResult``
"""

from __future__ import annotations

import logging
import statistics
import time
from collections.abc import Mapping, Sequence
from typing import Final, cast

import polars as pl

from cqros.ml.evaluation.evaluator import ModelEvaluator
from cqros.ml.evaluation.exceptions import ModelValidationError
from cqros.ml.evaluation.interfaces import (
    CrossValidationFold,
    CrossValidationResult,
)
from cqros.ml.models.registry import ModelRegistry
from cqros.ml.training.trainer import ModelTrainer

__all__ = [
    "CrossValidationFold",
    "CrossValidationResult",
    "TimeSeriesCrossValidator",
]

_logger = logging.getLogger(__name__)

_ERROR_REGISTRY_TYPE: Final[str] = "ML-CV-001"
_ERROR_TRAINER_TYPE: Final[str] = "ML-CV-002"
_ERROR_EVALUATOR_TYPE: Final[str] = "ML-CV-003"
_ERROR_FRAME_TYPE: Final[str] = "ML-CV-004"
_ERROR_FRAME_EMPTY: Final[str] = "ML-CV-005"
_ERROR_FOLDS: Final[str] = "ML-CV-006"
_ERROR_INSUFFICIENT_ROWS: Final[str] = "ML-CV-007"
_ERROR_MODEL_UNKNOWN: Final[str] = "ML-CV-008"

_MINIMUM_FOLDS: Final[int] = 2


class TimeSeriesCrossValidator:
    """Framework-independent chronological walk-forward cross-validator.

    For each fold the training window expands forward in row order and the
    next chronological segment is held out for validation. Training uses
    ``ModelTrainer`` on the training window only. Evaluation uses
    ``ModelEvaluator`` on the validation segment. Rows are never shuffled or
    reordered.

    Args:
        model_registry: Catalog used to resolve models by name.
        model_trainer: Trainer used to fit each fold.
        model_evaluator: Evaluator used to score each validation segment.
        logger: Optional logger instance. Defaults to the module logger.
    """

    __slots__ = ("_evaluator", "_logger", "_registry", "_trainer")

    _registry: ModelRegistry
    _trainer: ModelTrainer
    _evaluator: ModelEvaluator
    _logger: logging.Logger

    def __init__(
        self,
        model_registry: ModelRegistry,
        model_trainer: ModelTrainer,
        model_evaluator: ModelEvaluator,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the cross-validator with injected dependencies.

        Args:
            model_registry: Registry used to resolve models by name.
            model_trainer: Trainer used for each fold fit.
            model_evaluator: Evaluator used for each fold score.
            logger: Optional logger instance.

        Raises:
            ModelValidationError: If any dependency has an invalid type.
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
        if not isinstance(cast(object, model_trainer), ModelTrainer):
            raise ModelValidationError(
                "model_trainer must be a ModelTrainer instance",
                error_code=_ERROR_TRAINER_TYPE,
                details={
                    "parameter": "model_trainer",
                    "value_type": type(model_trainer).__name__,
                },
            )
        if not isinstance(cast(object, model_evaluator), ModelEvaluator):
            raise ModelValidationError(
                "model_evaluator must be a ModelEvaluator instance",
                error_code=_ERROR_EVALUATOR_TYPE,
                details={
                    "parameter": "model_evaluator",
                    "value_type": type(model_evaluator).__name__,
                },
            )
        self._registry = model_registry
        self._trainer = model_trainer
        self._evaluator = model_evaluator
        self._logger = logger if logger is not None else _logger

    def evaluate(
        self,
        model_name: str,
        frame: pl.DataFrame,
        folds: int,
    ) -> CrossValidationResult:
        """Run expanding-window walk-forward cross-validation.

        Args:
            model_name: Registry key of the model to train and evaluate.
            frame: Chronologically ordered dataset. Must not be mutated.
            folds: Number of walk-forward folds. Must be at least ``2``.

        Returns:
            Immutable ``CrossValidationResult`` summarizing all folds.

        Raises:
            ModelValidationError: If ``folds`` is invalid, ``frame`` is empty,
                rows are insufficient for the requested folds, or the model is
                unknown.
        """
        validated_frame = _require_frame(frame, parameter="frame")
        validated_folds = _require_folds(folds)
        _require_registered_model(self._registry, model_name)
        fold_windows = _build_expanding_windows(
            row_count=validated_frame.height,
            folds=validated_folds,
        )

        self._logger.info(
            "Starting time-series cross-validation",
            extra={
                "model_name": model_name,
                "folds": validated_folds,
                "total_rows": validated_frame.height,
            },
        )

        started = time.perf_counter()
        fold_results: list[CrossValidationFold] = []
        for fold_number, train_end, validation_start, validation_end in fold_windows:
            train_frame = validated_frame.slice(0, train_end)
            validation_frame = validated_frame.slice(
                validation_start,
                validation_end - validation_start,
            )

            trainer_result = self._trainer.train(model_name, train_frame)
            evaluation_result = self._evaluator.evaluate(
                trainer_result.fitted_model,
                validation_frame,
            )
            fold_results.append(
                CrossValidationFold(
                    fold_number=fold_number,
                    train_rows=train_frame.height,
                    validation_rows=validation_frame.height,
                    evaluation_result=evaluation_result,
                )
            )

        duration = time.perf_counter() - started
        mean_metrics, std_metrics = _aggregate_metrics(fold_results)
        result = CrossValidationResult(
            folds=tuple(fold_results),
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            fold_count=len(fold_results),
            total_rows=validated_frame.height,
            duration=duration,
        )

        self._logger.info(
            "Completed time-series cross-validation",
            extra={
                "model_name": model_name,
                "fold_count": result.fold_count,
                "total_rows": result.total_rows,
                "duration": result.duration,
            },
        )
        return result


def _require_frame(frame: object, *, parameter: str) -> pl.DataFrame:
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


def _require_folds(folds: object) -> int:
    """Validate that ``folds`` is an integer of at least ``2``."""
    if not isinstance(folds, int) or isinstance(folds, bool) or folds < _MINIMUM_FOLDS:
        raise ModelValidationError(
            f"folds must be an integer >= {_MINIMUM_FOLDS}",
            error_code=_ERROR_FOLDS,
            details={"parameter": "folds", "value": folds, "minimum": _MINIMUM_FOLDS},
        )
    return folds


def _require_registered_model(registry: ModelRegistry, model_name: str) -> None:
    """Validate that ``model_name`` exists in ``registry``."""
    if not registry.exists(model_name):
        raise ModelValidationError(
            f"model not registered: {model_name}",
            error_code=_ERROR_MODEL_UNKNOWN,
            details={"name": model_name},
        )


def _build_expanding_windows(
    *,
    row_count: int,
    folds: int,
) -> tuple[tuple[int, int, int, int], ...]:
    """Build expanding train/validation windows for walk-forward CV.

    Follows the sklearn ``TimeSeriesSplit`` expanding-window layout:
    validation segments of equal length move forward in time while the
    training window always starts at row ``0`` and ends at the validation
    start.

    Args:
        row_count: Total rows available in chronological order.
        folds: Number of walk-forward folds.

    Returns:
        Tuples of ``(fold_number, train_end, validation_start, validation_end)``.

    Raises:
        ModelValidationError: If ``row_count`` cannot support ``folds``.
    """
    segment_count = folds + 1
    validation_size = row_count // segment_count
    if validation_size < 1:
        raise ModelValidationError(
            "frame has insufficient rows for the requested folds",
            error_code=_ERROR_INSUFFICIENT_ROWS,
            details={
                "rows": row_count,
                "folds": folds,
                "minimum_rows": segment_count,
            },
        )

    windows: list[tuple[int, int, int, int]] = []
    first_validation_start = row_count - folds * validation_size
    for fold_index in range(folds):
        validation_start = first_validation_start + fold_index * validation_size
        validation_end = validation_start + validation_size
        train_end = validation_start
        windows.append(
            (
                fold_index + 1,
                train_end,
                validation_start,
                validation_end,
            )
        )
    return tuple(windows)


def _aggregate_metrics(
    folds: Sequence[CrossValidationFold],
) -> tuple[Mapping[str, float], Mapping[str, float]]:
    """Compute mean and population standard deviation for each metric."""
    if len(folds) == 0:
        return {}, {}

    metric_names = tuple(folds[0].evaluation_result.metrics.keys())
    mean_metrics: dict[str, float] = {}
    std_metrics: dict[str, float] = {}
    for name in metric_names:
        values = [float(fold.evaluation_result.metrics[name]) for fold in folds]
        mean_metrics[name] = float(statistics.fmean(values))
        std_metrics[name] = float(statistics.pstdev(values))
    return mean_metrics, std_metrics
