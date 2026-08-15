"""CQROS ML Evaluation public interfaces.

Purpose:
    Define structural contracts for model-evaluation orchestration so every
    evaluator and cross-validator implementation shares one public surface.

Responsibilities:
    - Expose ``ModelEvaluator`` as the shared evaluation-orchestration contract
    - Expose ``EvaluationResult`` as the immutable evaluation outcome contract
    - Expose ``TimeSeriesCrossValidator`` and cross-validation result contracts
    - Remain free of prediction, metric computation, and concrete evaluate logic

Dependencies:
    ``polars``, ``cqros.ml.models.interfaces.Model``, and
    ``cqros.ml.models.metadata``.

Public API:
    ``ModelEvaluator``, ``EvaluationResult``, ``CrossValidationFold``,
    ``CrossValidationResult``, ``TimeSeriesCrossValidator``
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import polars as pl

from cqros.ml.models.interfaces import Model
from cqros.ml.models.metadata import ModelMetadata, ModelTaskType

__all__ = [
    "CrossValidationFold",
    "CrossValidationResult",
    "EvaluationResult",
    "ModelEvaluator",
    "TimeSeriesCrossValidator",
]


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Immutable outcome of one ``ModelEvaluator.evaluate`` orchestration.

    Attributes:
        model_metadata: Metadata for the evaluated model.
        task_type: Supervised learning task type used for metric selection.
        dataset_rows: Number of rows in the evaluation frame.
        metrics: Computed metric name to float value mapping.
        evaluation_duration: Wall-clock evaluation duration in seconds.
    """

    model_metadata: ModelMetadata
    task_type: ModelTaskType
    dataset_rows: int
    metrics: Mapping[str, float]
    evaluation_duration: float


@dataclass(frozen=True, slots=True)
class CrossValidationFold:
    """Immutable outcome of one walk-forward cross-validation fold.

    Attributes:
        fold_number: One-based fold index within the walk-forward run.
        train_rows: Number of rows in the expanding training window.
        validation_rows: Number of rows in the chronological validation segment.
        evaluation_result: Metrics produced by ``ModelEvaluator`` on validation.
    """

    fold_number: int
    train_rows: int
    validation_rows: int
    evaluation_result: EvaluationResult


@dataclass(frozen=True, slots=True)
class CrossValidationResult:
    """Immutable outcome of one ``TimeSeriesCrossValidator.evaluate`` run.

    Attributes:
        folds: Per-fold walk-forward results in chronological order.
        mean_metrics: Mean of each metric across folds.
        std_metrics: Population standard deviation of each metric across folds.
        fold_count: Number of folds executed.
        total_rows: Number of rows in the full evaluation frame.
        duration: Wall-clock duration of the full cross-validation run.
    """

    folds: tuple[CrossValidationFold, ...]
    mean_metrics: Mapping[str, float]
    std_metrics: Mapping[str, float]
    fold_count: int
    total_rows: int
    duration: float


@runtime_checkable
class ModelEvaluator(Protocol):
    """Structural contract for framework-independent model evaluation.

    Implementations validate fitted models and evaluation frames, generate
    predictions through the ``Model`` surface, compute task-appropriate
    metrics, and return an immutable ``EvaluationResult``. They must not
    train, cross-validate, tune, or persist models.
    """

    def evaluate(
        self,
        model: Model,
        frame: pl.DataFrame,
    ) -> EvaluationResult:
        """Evaluate ``model`` on ``frame``.

        Args:
            model: Fitted model implementing the ``Model`` contract.
            frame: Evaluation dataset containing feature and label columns.
                Must not be mutated.

        Returns:
            Immutable ``EvaluationResult`` describing the completed evaluation.
        """
        ...


@runtime_checkable
class TimeSeriesCrossValidator(Protocol):
    """Structural contract for chronological walk-forward cross-validation.

    Implementations expand training windows forward in time, validate on the
    next chronological segment, and orchestrate training and evaluation through
    injected trainer and evaluator dependencies. They must not shuffle rows,
    reorder timestamps, or inspect framework-specific model internals.
    """

    def evaluate(
        self,
        model_name: str,
        frame: pl.DataFrame,
        folds: int,
    ) -> CrossValidationResult:
        """Run walk-forward cross-validation for ``model_name`` on ``frame``.

        Args:
            model_name: Registry key of the model to train and evaluate.
            frame: Chronologically ordered dataset. Must not be mutated.
            folds: Number of walk-forward folds. Must be at least ``2``.

        Returns:
            Immutable ``CrossValidationResult`` summarizing all folds.
        """
        ...
