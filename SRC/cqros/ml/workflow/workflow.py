"""CQROS ML TrainingWorkflow orchestration.

Purpose:
    Orchestrate the complete CQROS ML training lifecycle using injected
    dataset, model, evaluation, optimization, and experiment components
    without coupling to framework-specific model internals.

Responsibilities:
    - Load, split, and scale datasets through injected dataset components
    - Train and evaluate through injected trainer and evaluator components
    - Optionally run walk-forward cross-validation and hyperparameter search
    - Record an immutable ``ExperimentRecord`` through ``ExperimentTracker``
    - Return an immutable ``WorkflowResult``
    - Remain free of metric computation, direct training, scaling, evaluation,
      filesystem I/O, and CLI logic

Dependencies:
    ``polars``, ``cqros.ml.dataset``, ``cqros.ml.evaluation``,
    ``cqros.ml.experiments``, ``cqros.ml.models``, ``cqros.ml.optimization``,
    ``cqros.ml.training``, and ``cqros.ml.workflow`` result/exception types.

Public API:
    ``TrainingWorkflow``, ``WorkflowResult``
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Final, cast

import polars as pl

from cqros.ml.dataset.loader import DatasetLoader
from cqros.ml.dataset.scaler import DatasetScaler
from cqros.ml.dataset.splitter import DatasetSplitter
from cqros.ml.evaluation.cross_validation import TimeSeriesCrossValidator
from cqros.ml.evaluation.evaluator import ModelEvaluator
from cqros.ml.evaluation.interfaces import (
    CrossValidationResult,
    EvaluationResult,
)
from cqros.ml.experiments.schema import ExperimentRecord
from cqros.ml.experiments.tracker import ExperimentTracker
from cqros.ml.models.metadata import ModelTaskType
from cqros.ml.models.registry import ModelRegistry
from cqros.ml.optimization.interfaces import OptimizationResult
from cqros.ml.optimization.optimizer import HyperparameterOptimizer
from cqros.ml.training.trainer import ModelTrainer
from cqros.ml.workflow.exceptions import ModelValidationError
from cqros.ml.workflow.result import WorkflowResult

__all__ = [
    "TrainingWorkflow",
    "WorkflowResult",
]

_logger = logging.getLogger(__name__)

_ERROR_LOADER_TYPE: Final[str] = "ML-WORKFLOW-001"
_ERROR_SPLITTER_TYPE: Final[str] = "ML-WORKFLOW-002"
_ERROR_SCALER_TYPE: Final[str] = "ML-WORKFLOW-003"
_ERROR_REGISTRY_TYPE: Final[str] = "ML-WORKFLOW-004"
_ERROR_TRAINER_TYPE: Final[str] = "ML-WORKFLOW-005"
_ERROR_EVALUATOR_TYPE: Final[str] = "ML-WORKFLOW-006"
_ERROR_CROSS_VALIDATOR_TYPE: Final[str] = "ML-WORKFLOW-007"
_ERROR_OPTIMIZER_TYPE: Final[str] = "ML-WORKFLOW-008"
_ERROR_TRACKER_TYPE: Final[str] = "ML-WORKFLOW-009"
_ERROR_MODEL_UNKNOWN: Final[str] = "ML-WORKFLOW-010"
_ERROR_DATASET_EMPTY: Final[str] = "ML-WORKFLOW-011"
_ERROR_EXPERIMENT_ID: Final[str] = "ML-WORKFLOW-012"
_ERROR_CV_FOLDS: Final[str] = "ML-WORKFLOW-013"
_ERROR_OPTIMIZATION_CONFIG: Final[str] = "ML-WORKFLOW-014"
_ERROR_OPTIMIZATION_FOLDS: Final[str] = "ML-WORKFLOW-015"
_ERROR_PRIMARY_METRIC: Final[str] = "ML-WORKFLOW-016"

_MINIMUM_FOLDS: Final[int] = 2
_REGRESSION_PRIMARY_METRIC: Final[str] = "mae"
_CLASSIFICATION_PRIMARY_METRIC: Final[str] = "accuracy"


class TrainingWorkflow:
    """Framework-independent orchestrator for the CQROS ML training lifecycle.

    The workflow delegates every step to injected CQROS components. It never
    computes metrics, trains models, scales features, or evaluates predictions
    directly. Optional cross-validation and hyperparameter optimization are
    executed only when explicitly requested through ``run`` arguments.

    Args:
        dataset_loader: Loader used to assemble the canonical ML dataset.
        dataset_splitter: Chronological train/validation/test splitter.
        dataset_scaler: Feature scaler fit on train and applied to all splits.
        model_registry: Catalog used to resolve models by name.
        model_trainer: Trainer used for the primary fit step.
        model_evaluator: Evaluator used for validation and test scoring.
        cross_validator: Walk-forward cross-validator for optional CV.
        hyperparameter_optimizer: Optimizer for optional grid search.
        experiment_tracker: Tracker used to record experiment metadata.
        logger: Optional logger instance. Defaults to the module logger.
    """

    __slots__ = (
        "_cross_validator",
        "_evaluator",
        "_loader",
        "_logger",
        "_optimizer",
        "_registry",
        "_scaler",
        "_splitter",
        "_tracker",
        "_trainer",
    )

    _loader: DatasetLoader
    _splitter: DatasetSplitter
    _scaler: DatasetScaler
    _registry: ModelRegistry
    _trainer: ModelTrainer
    _evaluator: ModelEvaluator
    _cross_validator: TimeSeriesCrossValidator
    _optimizer: HyperparameterOptimizer
    _tracker: ExperimentTracker
    _logger: logging.Logger

    def __init__(
        self,
        dataset_loader: DatasetLoader,
        dataset_splitter: DatasetSplitter,
        dataset_scaler: DatasetScaler,
        model_registry: ModelRegistry,
        model_trainer: ModelTrainer,
        model_evaluator: ModelEvaluator,
        cross_validator: TimeSeriesCrossValidator,
        hyperparameter_optimizer: HyperparameterOptimizer,
        experiment_tracker: ExperimentTracker,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the workflow with injected CQROS dependencies.

        Args:
            dataset_loader: Loader used to assemble the canonical ML dataset.
            dataset_splitter: Chronological train/validation/test splitter.
            dataset_scaler: Feature scaler fit on train and applied to splits.
            model_registry: Registry used to resolve models by name.
            model_trainer: Trainer used for the primary fit step.
            model_evaluator: Evaluator used for validation and test scoring.
            cross_validator: Walk-forward cross-validator for optional CV.
            hyperparameter_optimizer: Optimizer for optional grid search.
            experiment_tracker: Tracker used to record experiment metadata.
            logger: Optional logger instance.

        Raises:
            ModelValidationError: If any dependency has an invalid type.
        """
        if not isinstance(cast(object, dataset_loader), DatasetLoader):
            raise ModelValidationError(
                "dataset_loader must be a DatasetLoader instance",
                error_code=_ERROR_LOADER_TYPE,
                details={
                    "parameter": "dataset_loader",
                    "value_type": type(dataset_loader).__name__,
                },
            )
        if not isinstance(cast(object, dataset_splitter), DatasetSplitter):
            raise ModelValidationError(
                "dataset_splitter must be a DatasetSplitter instance",
                error_code=_ERROR_SPLITTER_TYPE,
                details={
                    "parameter": "dataset_splitter",
                    "value_type": type(dataset_splitter).__name__,
                },
            )
        if not isinstance(cast(object, dataset_scaler), DatasetScaler):
            raise ModelValidationError(
                "dataset_scaler must be a DatasetScaler instance",
                error_code=_ERROR_SCALER_TYPE,
                details={
                    "parameter": "dataset_scaler",
                    "value_type": type(dataset_scaler).__name__,
                },
            )
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
        if not isinstance(cast(object, cross_validator), TimeSeriesCrossValidator):
            raise ModelValidationError(
                "cross_validator must be a TimeSeriesCrossValidator instance",
                error_code=_ERROR_CROSS_VALIDATOR_TYPE,
                details={
                    "parameter": "cross_validator",
                    "value_type": type(cross_validator).__name__,
                },
            )
        if not isinstance(cast(object, hyperparameter_optimizer), HyperparameterOptimizer):
            raise ModelValidationError(
                "hyperparameter_optimizer must be a HyperparameterOptimizer instance",
                error_code=_ERROR_OPTIMIZER_TYPE,
                details={
                    "parameter": "hyperparameter_optimizer",
                    "value_type": type(hyperparameter_optimizer).__name__,
                },
            )
        if not isinstance(cast(object, experiment_tracker), ExperimentTracker):
            raise ModelValidationError(
                "experiment_tracker must be an ExperimentTracker instance",
                error_code=_ERROR_TRACKER_TYPE,
                details={
                    "parameter": "experiment_tracker",
                    "value_type": type(experiment_tracker).__name__,
                },
            )

        self._loader = dataset_loader
        self._splitter = dataset_splitter
        self._scaler = dataset_scaler
        self._registry = model_registry
        self._trainer = model_trainer
        self._evaluator = model_evaluator
        self._cross_validator = cross_validator
        self._optimizer = hyperparameter_optimizer
        self._tracker = experiment_tracker
        self._logger = logger if logger is not None else _logger

    def run(
        self,
        *,
        model_name: str,
        experiment_id: str,
        train_ratio: float,
        validation_ratio: float,
        test_ratio: float,
        symbols: Sequence[str] | None = None,
        timeframes: Sequence[str] | None = None,
        years: Sequence[int] | None = None,
        cross_validation_folds: int | None = None,
        parameter_grid: Mapping[str, Sequence[object]] | None = None,
        optimization_metric: str | None = None,
        optimization_folds: int | None = None,
        artifact_path: str = "",
        notes: str = "",
    ) -> WorkflowResult:
        """Orchestrate one end-to-end training workflow.

        Args:
            model_name: Registry key of the model to train.
            experiment_id: Unique identifier for the recorded experiment.
            train_ratio: Fraction of rows assigned to the train split.
            validation_ratio: Fraction of rows assigned to the validation split.
            test_ratio: Fraction of rows assigned to the test split.
            symbols: Optional symbol allowlist passed to ``DatasetLoader``.
            timeframes: Optional timeframe allowlist passed to ``DatasetLoader``.
            years: Optional year allowlist passed to ``DatasetLoader``.
            cross_validation_folds: Optional walk-forward fold count. When
                provided, ``TimeSeriesCrossValidator`` is executed.
            parameter_grid: Optional hyperparameter grid. When provided with
                ``optimization_metric`` and ``optimization_folds``,
                ``HyperparameterOptimizer`` is executed.
            optimization_metric: Metric name used to rank optimization trials.
            optimization_folds: Walk-forward fold count for optimization.
            artifact_path: Optional artifact location recorded on the experiment.
            notes: Optional free-form notes recorded on the experiment.

        Returns:
            Immutable ``WorkflowResult`` describing the completed workflow.

        Raises:
            ModelValidationError: If the model is unknown, the dataset is empty,
                or the workflow configuration is invalid.
        """
        validated_experiment_id = _require_experiment_id(experiment_id)
        _require_registered_model(self._registry, model_name)
        validated_cv_folds = _require_optional_folds(
            cross_validation_folds,
            parameter="cross_validation_folds",
            error_code=_ERROR_CV_FOLDS,
        )
        run_optimization = _require_optimization_configuration(
            parameter_grid=parameter_grid,
            optimization_metric=optimization_metric,
            optimization_folds=optimization_folds,
        )

        self._logger.info(
            "Starting training workflow",
            extra={
                "model_name": model_name,
                "experiment_id": validated_experiment_id,
                "cross_validation_folds": validated_cv_folds,
                "run_optimization": run_optimization,
            },
        )

        started = time.perf_counter()

        dataset = self._loader.load(
            symbols=symbols,
            timeframes=timeframes,
            years=years,
        )
        _require_non_empty_dataset(dataset)

        train_frame, validation_frame, test_frame = self._splitter.split(
            dataset,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
        )

        scaled_train = self._scaler.fit_transform(train_frame)
        scaled_validation = self._scaler.transform(validation_frame)
        scaled_test = self._scaler.transform(test_frame)

        train_result = self._trainer.train(
            model_name,
            scaled_train,
            validation_frame=scaled_validation,
        )
        fitted_model = train_result.fitted_model
        model_metadata = train_result.model_metadata

        validation_result = self._evaluator.evaluate(fitted_model, scaled_validation)
        test_result = self._evaluator.evaluate(fitted_model, scaled_test)

        scoring_frame = pl.concat(
            [scaled_train, scaled_validation],
            how="vertical",
        )

        cross_validation_result: CrossValidationResult | None = None
        if validated_cv_folds is not None:
            cross_validation_result = self._cross_validator.evaluate(
                model_name,
                scoring_frame,
                validated_cv_folds,
            )

        optimization_result: OptimizationResult | None = None
        if run_optimization:
            optimization_result = self._optimizer.optimize(
                model_name,
                scoring_frame,
                cast(Mapping[str, Sequence[object]], parameter_grid),
                cast(str, optimization_metric),
                cast(int, optimization_folds),
            )

        best_metric = _resolve_best_metric(
            task_type=model_metadata.task_type,
            validation_result=validation_result,
            cross_validation_result=cross_validation_result,
            optimization_result=optimization_result,
        )
        parameters = _resolve_parameters(
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            optimization_result=optimization_result,
        )
        cv_metrics = (
            dict(cross_validation_result.mean_metrics)
            if cross_validation_result is not None
            else {}
        )

        self._tracker.record(
            ExperimentRecord(
                experiment_id=validated_experiment_id,
                timestamp=datetime.now(UTC),
                model_name=model_metadata.name,
                framework=model_metadata.framework,
                task_type=model_metadata.task_type,
                label_column=model_metadata.label_column,
                feature_count=len(model_metadata.feature_columns),
                train_rows=train_result.train_rows,
                validation_rows=train_result.validation_rows,
                test_rows=scaled_test.height,
                parameters=parameters,
                cross_validation_metrics=cv_metrics,
                best_metric=best_metric,
                artifact_path=artifact_path,
                notes=notes,
            )
        )

        duration = time.perf_counter() - started
        result = WorkflowResult(
            experiment_id=validated_experiment_id,
            model_metadata=model_metadata,
            scaler=self._scaler,
            train_result=train_result,
            validation_result=validation_result,
            test_result=test_result,
            cross_validation_result=cross_validation_result,
            optimization_result=optimization_result,
            duration=duration,
        )

        self._logger.info(
            "Completed training workflow",
            extra={
                "model_name": model_name,
                "experiment_id": validated_experiment_id,
                "duration": duration,
                "train_rows": train_result.train_rows,
                "validation_rows": train_result.validation_rows,
                "test_rows": scaled_test.height,
            },
        )
        return result


def _require_experiment_id(experiment_id: object) -> str:
    """Validate that ``experiment_id`` is a non-empty string."""
    if not isinstance(experiment_id, str) or experiment_id.strip() == "":
        raise ModelValidationError(
            "experiment_id must be a non-empty string",
            error_code=_ERROR_EXPERIMENT_ID,
            details={"parameter": "experiment_id", "value": experiment_id},
        )
    return experiment_id


def _require_registered_model(registry: ModelRegistry, model_name: str) -> None:
    """Validate that ``model_name`` is registered."""
    if not registry.exists(model_name):
        raise ModelValidationError(
            f"model not registered: {model_name}",
            error_code=_ERROR_MODEL_UNKNOWN,
            details={"model_name": model_name},
        )


def _require_non_empty_dataset(frame: pl.DataFrame) -> None:
    """Reject empty loaded datasets before splitting."""
    if frame.height == 0:
        raise ModelValidationError(
            "dataset must contain at least one row",
            error_code=_ERROR_DATASET_EMPTY,
            details={"rows": frame.height},
        )


def _require_optional_folds(
    folds: object,
    *,
    parameter: str,
    error_code: str,
) -> int | None:
    """Validate an optional fold count of at least ``_MINIMUM_FOLDS``."""
    if folds is None:
        return None
    if not isinstance(folds, int) or isinstance(folds, bool) or folds < _MINIMUM_FOLDS:
        raise ModelValidationError(
            f"{parameter} must be an integer greater than or equal to {_MINIMUM_FOLDS}",
            error_code=error_code,
            details={"parameter": parameter, "value": folds, "minimum": _MINIMUM_FOLDS},
        )
    return folds


def _require_optimization_configuration(
    *,
    parameter_grid: Mapping[str, Sequence[object]] | None,
    optimization_metric: str | None,
    optimization_folds: int | None,
) -> bool:
    """Validate optional HPO arguments and return whether optimization runs."""
    provided = (
        parameter_grid is not None,
        optimization_metric is not None,
        optimization_folds is not None,
    )
    if not any(provided):
        return False
    if not all(provided):
        raise ModelValidationError(
            "parameter_grid, optimization_metric, and optimization_folds "
            "must all be provided together",
            error_code=_ERROR_OPTIMIZATION_CONFIG,
            details={
                "parameter_grid_provided": parameter_grid is not None,
                "optimization_metric_provided": optimization_metric is not None,
                "optimization_folds_provided": optimization_folds is not None,
            },
        )
    if not isinstance(optimization_metric, str) or optimization_metric.strip() == "":
        raise ModelValidationError(
            "optimization_metric must be a non-empty string",
            error_code=_ERROR_OPTIMIZATION_CONFIG,
            details={
                "parameter": "optimization_metric",
                "value": optimization_metric,
            },
        )
    _require_optional_folds(
        optimization_folds,
        parameter="optimization_folds",
        error_code=_ERROR_OPTIMIZATION_FOLDS,
    )
    return True


def _resolve_parameters(
    *,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    optimization_result: OptimizationResult | None,
) -> Mapping[str, object]:
    """Build experiment parameters from ratios or optimization results."""
    if optimization_result is not None:
        return dict(optimization_result.best_parameters)
    return {
        "train_ratio": train_ratio,
        "validation_ratio": validation_ratio,
        "test_ratio": test_ratio,
    }


def _resolve_best_metric(
    *,
    task_type: ModelTaskType,
    validation_result: EvaluationResult,
    cross_validation_result: CrossValidationResult | None,
    optimization_result: OptimizationResult | None,
) -> float:
    """Select the primary scalar metric retained on the experiment record."""
    if optimization_result is not None:
        return float(optimization_result.best_score)
    if cross_validation_result is not None:
        return _extract_primary_metric(
            cross_validation_result.mean_metrics,
            task_type=task_type,
            source="cross_validation_result.mean_metrics",
        )
    return _extract_primary_metric(
        validation_result.metrics,
        task_type=task_type,
        source="validation_result.metrics",
    )


def _extract_primary_metric(
    metrics: Mapping[str, float],
    *,
    task_type: ModelTaskType,
    source: str,
) -> float:
    """Extract the task-appropriate primary metric from ``metrics``."""
    metric_name = (
        _CLASSIFICATION_PRIMARY_METRIC
        if task_type is ModelTaskType.CLASSIFICATION
        else _REGRESSION_PRIMARY_METRIC
    )
    if metric_name not in metrics:
        raise ModelValidationError(
            f"primary metric {metric_name!r} missing from {source}",
            error_code=_ERROR_PRIMARY_METRIC,
            details={
                "metric_name": metric_name,
                "available_metrics": tuple(metrics.keys()),
                "source": source,
            },
        )
    return float(metrics[metric_name])
