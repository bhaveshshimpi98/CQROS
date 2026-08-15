"""CQROS ML HyperparameterOptimizer orchestration.

Purpose:
    Orchestrate exhaustive grid-search hyperparameter optimization using an
    injected ``TimeSeriesCrossValidator`` without coupling to framework-specific
    model internals.

Responsibilities:
    - Expand parameter grids into independent candidate combinations
    - Materialize a fresh candidate model for each combination
    - Evaluate candidates only through ``TimeSeriesCrossValidator.evaluate``
    - Rank trials by the selected cross-validation mean metric
    - Return an immutable ``OptimizationResult``
    - Remain free of direct training, metric computation, and HPO frameworks

Dependencies:
    ``polars``, ``dataclasses``, ``cqros.ml.evaluation.cross_validation``,
    ``cqros.ml.models.interfaces.Model``, ``cqros.ml.models.registry``,
    ``cqros.ml.optimization.exceptions``, ``cqros.ml.optimization.interfaces``,
    and ``cqros.ml.optimization.search``.

Public API:
    ``HyperparameterOptimizer``, ``OptimizationResult``, ``OptimizationTrial``
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from typing import Final, cast

import polars as pl

from cqros.ml.evaluation.cross_validation import TimeSeriesCrossValidator
from cqros.ml.evaluation.interfaces import CrossValidationResult
from cqros.ml.models.interfaces import Model
from cqros.ml.models.registry import ModelRegistry
from cqros.ml.optimization.exceptions import ModelValidationError
from cqros.ml.optimization.interfaces import (
    OptimizationResult,
    OptimizationTrial,
)
from cqros.ml.optimization.search import (
    expand_parameter_grid,
    is_better_score,
    resolve_optimization_direction,
)

__all__ = [
    "HyperparameterOptimizer",
    "OptimizationResult",
    "OptimizationTrial",
]

_logger = logging.getLogger(__name__)

_ERROR_CROSS_VALIDATOR_TYPE: Final[str] = "ML-HPO-001"
_ERROR_REGISTRY_MISSING: Final[str] = "ML-HPO-002"
_ERROR_FRAME_TYPE: Final[str] = "ML-HPO-003"
_ERROR_FRAME_EMPTY: Final[str] = "ML-HPO-004"
_ERROR_FOLDS: Final[str] = "ML-HPO-005"
_ERROR_MODEL_UNKNOWN: Final[str] = "ML-HPO-006"
_ERROR_MODEL_TYPE: Final[str] = "ML-HPO-007"
_ERROR_PARAMETER_NAME: Final[str] = "ML-HPO-008"
_ERROR_METRIC_MISSING: Final[str] = "ML-HPO-009"

_MINIMUM_FOLDS: Final[int] = 2


class HyperparameterOptimizer:
    """Framework-independent exhaustive grid-search hyperparameter optimizer.

    Each parameter combination is applied to a fresh model instance derived
    from the registered template. Candidates are assessed only through the
    injected ``TimeSeriesCrossValidator``. The optimizer never trains models
    or computes metrics directly.

    Args:
        cross_validator: Walk-forward cross-validator used for every trial.
        logger: Optional logger instance. Defaults to the module logger.
    """

    __slots__ = ("_cross_validator", "_logger", "_registry")

    _cross_validator: TimeSeriesCrossValidator
    _registry: ModelRegistry
    _logger: logging.Logger

    def __init__(
        self,
        cross_validator: TimeSeriesCrossValidator,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the optimizer with an injected cross-validator.

        Args:
            cross_validator: Cross-validator used to score each candidate.
            logger: Optional logger instance.

        Raises:
            ModelValidationError: If ``cross_validator`` is invalid or does not
                expose a ``ModelRegistry``.
        """
        if not isinstance(cast(object, cross_validator), TimeSeriesCrossValidator):
            raise ModelValidationError(
                "cross_validator must be a TimeSeriesCrossValidator instance",
                error_code=_ERROR_CROSS_VALIDATOR_TYPE,
                details={
                    "parameter": "cross_validator",
                    "value_type": type(cross_validator).__name__,
                },
            )
        registry = getattr(cross_validator, "_registry", None)
        if not isinstance(registry, ModelRegistry):
            raise ModelValidationError(
                "cross_validator must expose a ModelRegistry",
                error_code=_ERROR_REGISTRY_MISSING,
                details={"parameter": "cross_validator"},
            )
        self._cross_validator = cross_validator
        self._registry = registry
        self._logger = logger if logger is not None else _logger

    def optimize(
        self,
        model_name: str,
        frame: pl.DataFrame,
        parameter_grid: Mapping[str, Sequence[object]],
        metric: str,
        folds: int,
    ) -> OptimizationResult:
        """Search ``parameter_grid`` using walk-forward cross-validation.

        Args:
            model_name: Registry key of the model to optimize.
            frame: Chronologically ordered dataset. Must not be mutated.
            parameter_grid: Mapping of parameter name to candidate values.
            metric: Cross-validation mean metric used for ranking.
            folds: Number of walk-forward folds. Must be at least ``2``.

        Returns:
            Immutable ``OptimizationResult`` summarizing the search.

        Raises:
            ModelValidationError: If inputs are invalid, the model is unknown,
                or a trial metric is missing from the cross-validation result.
        """
        validated_frame = _require_frame(frame, parameter="frame")
        validated_folds = _require_folds(folds)
        direction = resolve_optimization_direction(metric)
        combinations = expand_parameter_grid(parameter_grid)
        template = _require_registered_model(self._registry, model_name)

        self._logger.info(
            "Starting hyperparameter optimization",
            extra={
                "model_name": model_name,
                "metric": metric,
                "folds": validated_folds,
                "trial_count": len(combinations),
            },
        )

        started = time.perf_counter()
        trials: list[OptimizationTrial] = []
        best_trial: OptimizationTrial | None = None

        for trial_index, parameters in enumerate(combinations, start=1):
            candidate = _build_candidate_model(template, parameters)
            cv_result = _evaluate_candidate(
                registry=self._registry,
                cross_validator=self._cross_validator,
                model_name=model_name,
                template=template,
                candidate=candidate,
                frame=validated_frame,
                folds=validated_folds,
            )
            score = _extract_score(cv_result, metric=metric)
            trial = OptimizationTrial(
                trial_number=trial_index,
                parameters=dict(parameters),
                score=score,
                cross_validation_result=cv_result,
            )
            trials.append(trial)
            if best_trial is None or is_better_score(
                trial.score,
                best_trial.score,
                direction=direction,
            ):
                best_trial = trial

        if best_trial is None:
            raise ModelValidationError(
                "optimization produced no trials",
                error_code=_ERROR_METRIC_MISSING,
                details={"model_name": model_name, "trial_count": 0},
            )

        duration = time.perf_counter() - started

        result = OptimizationResult(
            model_name=model_name,
            metric_name=metric,
            optimization_direction=direction,
            best_parameters=best_trial.parameters,
            best_score=best_trial.score,
            best_trial=best_trial,
            trials=tuple(trials),
            duration=duration,
        )

        self._logger.info(
            "Completed hyperparameter optimization",
            extra={
                "model_name": model_name,
                "metric": metric,
                "best_score": result.best_score,
                "trial_count": len(result.trials),
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


def _require_registered_model(registry: ModelRegistry, model_name: str) -> Model:
    """Return the registered template model or raise when missing."""
    if not registry.exists(model_name):
        raise ModelValidationError(
            f"model not registered: {model_name}",
            error_code=_ERROR_MODEL_UNKNOWN,
            details={"name": model_name},
        )
    return registry.get(model_name)


def _build_candidate_model(
    template: Model,
    parameters: Mapping[str, object],
) -> Model:
    """Build a fresh unfitted model with ``parameters`` applied.

    Candidates are constructed through the template's public dataclass
    constructor fields only. Framework-specific internals are never inspected.
    """
    model_type = type(template)
    if not is_dataclass(model_type):
        raise ModelValidationError(
            "registered model must be a dataclass to apply parameters",
            error_code=_ERROR_MODEL_TYPE,
            details={"model_type": model_type.__name__},
        )

    init_fields = {
        field_info.name
        for field_info in fields(model_type)
        if field_info.init and field_info.name != "model_metadata"
    }
    unknown = sorted(name for name in parameters if name not in init_fields)
    if unknown:
        raise ModelValidationError(
            "parameter_grid contains unknown model parameters",
            error_code=_ERROR_PARAMETER_NAME,
            details={
                "unknown_parameters": tuple(unknown),
                "allowed_parameters": tuple(sorted(init_fields)),
            },
        )

    init_kwargs: dict[str, object] = {"model_metadata": template.metadata()}
    for name in init_fields:
        init_kwargs[name] = getattr(template, name)
    init_kwargs.update(dict(parameters))
    return cast(Model, model_type(**init_kwargs))


def _evaluate_candidate(
    *,
    registry: ModelRegistry,
    cross_validator: TimeSeriesCrossValidator,
    model_name: str,
    template: Model,
    candidate: Model,
    frame: pl.DataFrame,
    folds: int,
) -> CrossValidationResult:
    """Swap ``candidate`` into the registry, evaluate, then restore ``template``."""
    registry.remove(model_name)
    registry.register(candidate)
    try:
        return cross_validator.evaluate(model_name, frame, folds)
    finally:
        registry.remove(model_name)
        registry.register(template)


def _extract_score(result: CrossValidationResult, *, metric: str) -> float:
    """Extract the selected mean metric from a cross-validation result."""
    if metric not in result.mean_metrics:
        raise ModelValidationError(
            f"metric not present in cross-validation result: {metric}",
            error_code=_ERROR_METRIC_MISSING,
            details={
                "metric": metric,
                "available_metrics": tuple(result.mean_metrics.keys()),
            },
        )
    return float(result.mean_metrics[metric])
