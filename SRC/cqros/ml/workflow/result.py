"""CQROS ML Workflow result models.

Purpose:
    Provide immutable value objects that describe training-workflow outcomes
    without coupling to dataset loading, training, evaluation, or framework-
    specific model internals.

Responsibilities:
    - Define ``WorkflowResult`` as the training-workflow outcome contract
    - Remain free of orchestration execution and dependency wiring

Dependencies:
    ``cqros.ml.dataset.scaler.DatasetScaler``,
    ``cqros.ml.evaluation.interfaces``,
    ``cqros.ml.models.metadata.ModelMetadata``,
    ``cqros.ml.optimization.interfaces.OptimizationResult``, and
    ``cqros.ml.training.interfaces.TrainerResult``.

Public API:
    ``WorkflowResult``
"""

from __future__ import annotations

from dataclasses import dataclass

from cqros.ml.dataset.scaler import DatasetScaler
from cqros.ml.evaluation.interfaces import (
    CrossValidationResult,
    EvaluationResult,
)
from cqros.ml.models.metadata import ModelMetadata
from cqros.ml.optimization.interfaces import OptimizationResult
from cqros.ml.training.interfaces import TrainerResult

__all__ = [
    "WorkflowResult",
]


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    """Immutable outcome of one ``TrainingWorkflow.run`` orchestration.

    Attributes:
        experiment_id: Identifier recorded for the completed experiment.
        model_metadata: Metadata for the trained model.
        scaler: Fitted scaler used for train, validation, and test transforms.
        train_result: Outcome of the primary training step.
        validation_result: Evaluation metrics on the validation split.
        test_result: Evaluation metrics on the test split.
        cross_validation_result: Optional walk-forward cross-validation outcome.
        optimization_result: Optional hyperparameter-optimization outcome.
        duration: Wall-clock duration of the full workflow run in seconds.
    """

    experiment_id: str
    model_metadata: ModelMetadata
    scaler: DatasetScaler
    train_result: TrainerResult
    validation_result: EvaluationResult
    test_result: EvaluationResult
    cross_validation_result: CrossValidationResult | None
    optimization_result: OptimizationResult | None
    duration: float
