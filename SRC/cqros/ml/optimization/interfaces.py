"""CQROS ML Optimization public interfaces.

Purpose:
    Define structural contracts for hyperparameter-optimization orchestration
    so every optimizer implementation shares one public surface.

Responsibilities:
    - Expose ``HyperparameterOptimizer`` as the shared optimization contract
    - Expose ``OptimizationTrial`` and ``OptimizationResult`` outcome contracts
    - Expose ``OptimizationDirection`` for metric ranking
    - Remain free of grid expansion, registry mutation, and concrete search logic

Dependencies:
    ``polars``, ``cqros.ml.evaluation.interfaces.CrossValidationResult``.

Public API:
    ``OptimizationDirection``, ``OptimizationTrial``, ``OptimizationResult``,
    ``HyperparameterOptimizer``
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

import polars as pl

from cqros.ml.evaluation.interfaces import CrossValidationResult

__all__ = [
    "HyperparameterOptimizer",
    "OptimizationDirection",
    "OptimizationResult",
    "OptimizationTrial",
]


class OptimizationDirection(StrEnum):
    """Direction used when ranking hyperparameter trial scores."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass(frozen=True, slots=True)
class OptimizationTrial:
    """Immutable outcome of one hyperparameter grid-search trial.

    Attributes:
        trial_number: One-based trial index within the search run.
        parameters: Parameter combination evaluated for this trial.
        score: Selected cross-validation mean metric for this trial.
        cross_validation_result: Full walk-forward result for the trial.
    """

    trial_number: int
    parameters: Mapping[str, object]
    score: float
    cross_validation_result: CrossValidationResult


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Immutable outcome of one ``HyperparameterOptimizer.optimize`` run.

    Attributes:
        model_name: Registry key of the optimized model.
        metric_name: Metric used to rank trials.
        optimization_direction: Whether lower or higher scores are better.
        best_parameters: Parameter combination from the best trial.
        best_score: Score from the best trial.
        best_trial: Full best-trial record.
        trials: All trials in evaluation order.
        duration: Wall-clock duration of the full optimization run.
    """

    model_name: str
    metric_name: str
    optimization_direction: OptimizationDirection
    best_parameters: Mapping[str, object]
    best_score: float
    best_trial: OptimizationTrial
    trials: tuple[OptimizationTrial, ...]
    duration: float


@runtime_checkable
class HyperparameterOptimizer(Protocol):
    """Structural contract for framework-independent hyperparameter search.

    Implementations expand a parameter grid, evaluate each candidate through
    ``TimeSeriesCrossValidator``, and return an immutable
    ``OptimizationResult``. They must not train models or compute metrics
    directly.
    """

    def optimize(
        self,
        model_name: str,
        frame: pl.DataFrame,
        parameter_grid: Mapping[str, Sequence[object]],
        metric: str,
        folds: int,
    ) -> OptimizationResult:
        """Search ``parameter_grid`` for ``model_name`` on ``frame``.

        Args:
            model_name: Registry key of the model to optimize.
            frame: Chronologically ordered dataset. Must not be mutated.
            parameter_grid: Mapping of parameter name to candidate values.
            metric: Cross-validation mean metric used for ranking.
            folds: Number of walk-forward folds passed to the cross-validator.

        Returns:
            Immutable ``OptimizationResult`` summarizing the search.
        """
        ...
