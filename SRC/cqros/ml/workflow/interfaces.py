"""CQROS ML Workflow public interfaces.

Purpose:
    Define structural contracts for training-workflow orchestration so every
    workflow implementation shares one public surface.

Responsibilities:
    - Expose ``TrainingWorkflow`` as the shared workflow-orchestration contract
    - Remain free of loading, scaling, training, evaluation, and recording logic

Dependencies:
    ``polars`` collections types and ``cqros.ml.workflow.result.WorkflowResult``.

Public API:
    ``TrainingWorkflow``
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from cqros.ml.workflow.result import WorkflowResult

__all__ = [
    "TrainingWorkflow",
]


@runtime_checkable
class TrainingWorkflow(Protocol):
    """Structural contract for framework-independent ML training orchestration.

    Implementations load and split datasets, scale features, train and evaluate
    models, optionally run cross-validation and hyperparameter optimization,
    record experiment metadata, and return an immutable ``WorkflowResult``.
    They must not compute metrics, train, scale, or evaluate directly—every
    step is delegated to injected CQROS components.
    """

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
        """
        ...
