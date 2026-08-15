"""CQROS ML Optimization grid-search helpers.

Purpose:
    Isolate exhaustive parameter-grid expansion and metric-direction helpers
    used by ``HyperparameterOptimizer``.

Responsibilities:
    - Expand parameter grids into independent combinations
    - Resolve optimization direction for supported metrics
    - Rank scores according to optimization direction
    - Remain free of training, evaluation, and registry mutation

Dependencies:
    Python standard library and ``cqros.ml.optimization.interfaces``.

Public API:
    ``expand_parameter_grid``, ``resolve_optimization_direction``,
    ``is_better_score``, ``SUPPORTED_METRICS``
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product
from typing import Final, cast

from cqros.ml.optimization.exceptions import ModelValidationError
from cqros.ml.optimization.interfaces import OptimizationDirection

__all__ = [
    "SUPPORTED_METRICS",
    "expand_parameter_grid",
    "is_better_score",
    "resolve_optimization_direction",
]

_ERROR_GRID_TYPE: Final[str] = "ML-HPO-SEARCH-001"
_ERROR_GRID_EMPTY: Final[str] = "ML-HPO-SEARCH-002"
_ERROR_GRID_VALUES: Final[str] = "ML-HPO-SEARCH-003"
_ERROR_METRIC: Final[str] = "ML-HPO-SEARCH-004"

_MINIMIZE_METRICS: Final[frozenset[str]] = frozenset({"mae", "mse", "rmse"})
_MAXIMIZE_METRICS: Final[frozenset[str]] = frozenset(
    {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "r2",
    }
)
SUPPORTED_METRICS: Final[frozenset[str]] = _MINIMIZE_METRICS | _MAXIMIZE_METRICS


def expand_parameter_grid(
    parameter_grid: Mapping[str, Sequence[object]],
) -> tuple[Mapping[str, object], ...]:
    """Expand ``parameter_grid`` into every parameter combination.

    Args:
        parameter_grid: Mapping of parameter name to a non-empty sequence of
            candidate values.

    Returns:
        Tuple of immutable parameter mappings in deterministic Cartesian order.

    Raises:
        ModelValidationError: If the grid is empty or contains empty value
            sequences.
    """
    grid_object = cast(object, parameter_grid)
    if not isinstance(grid_object, Mapping):
        raise ModelValidationError(
            "parameter_grid must be a mapping of parameter names to values",
            error_code=_ERROR_GRID_TYPE,
            details={"value_type": type(parameter_grid).__name__},
        )
    if len(parameter_grid) == 0:
        raise ModelValidationError(
            "parameter_grid must not be empty",
            error_code=_ERROR_GRID_EMPTY,
            details={"parameter": "parameter_grid"},
        )

    names: list[str] = []
    value_lists: list[tuple[object, ...]] = []
    for name_object, values_object in cast(
        Mapping[object, object],
        parameter_grid,
    ).items():
        if not isinstance(name_object, str) or name_object.strip() == "":
            raise ModelValidationError(
                "parameter_grid keys must be non-empty strings",
                error_code=_ERROR_GRID_VALUES,
                details={"parameter": "parameter_grid", "key": name_object},
            )
        if isinstance(values_object, (str, bytes)) or not isinstance(
            values_object,
            Sequence,
        ):
            raise ModelValidationError(
                "parameter_grid values must be non-empty sequences",
                error_code=_ERROR_GRID_VALUES,
                details={
                    "parameter": name_object,
                    "value_type": type(values_object).__name__,
                },
            )
        sequence = tuple(cast(Sequence[object], values_object))
        if len(sequence) == 0:
            raise ModelValidationError(
                "parameter_grid value sequences must not be empty",
                error_code=_ERROR_GRID_EMPTY,
                details={"parameter": name_object},
            )
        names.append(name_object)
        value_lists.append(sequence)

    combinations: list[Mapping[str, object]] = []
    for combo in product(*value_lists):
        combinations.append(dict(zip(names, combo, strict=True)))
    return tuple(combinations)


def resolve_optimization_direction(metric: object) -> OptimizationDirection:
    """Return the ranking direction for a supported optimization metric.

    Args:
        metric: Metric name produced by model evaluation / cross-validation.

    Returns:
        ``OptimizationDirection.MINIMIZE`` or ``MAXIMIZE``.

    Raises:
        ModelValidationError: If ``metric`` is unsupported.
    """
    if not isinstance(metric, str) or metric.strip() == "":
        raise ModelValidationError(
            "metric must be a non-empty string",
            error_code=_ERROR_METRIC,
            details={"metric": metric},
        )
    if metric in _MINIMIZE_METRICS:
        return OptimizationDirection.MINIMIZE
    if metric in _MAXIMIZE_METRICS:
        return OptimizationDirection.MAXIMIZE
    raise ModelValidationError(
        f"unsupported metric: {metric}",
        error_code=_ERROR_METRIC,
        details={
            "metric": metric,
            "supported_metrics": tuple(sorted(SUPPORTED_METRICS)),
        },
    )


def is_better_score(
    candidate: float,
    incumbent: float,
    *,
    direction: OptimizationDirection,
) -> bool:
    """Return whether ``candidate`` improves on ``incumbent`` for ``direction``."""
    if direction is OptimizationDirection.MINIMIZE:
        return candidate < incumbent
    return candidate > incumbent
