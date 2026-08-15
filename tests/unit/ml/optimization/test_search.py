"""Unit tests for CQROS optimization grid-search helpers."""

from __future__ import annotations

import pytest

from cqros.ml.optimization.exceptions import ModelValidationError
from cqros.ml.optimization.interfaces import OptimizationDirection
from cqros.ml.optimization.search import (
    SUPPORTED_METRICS,
    expand_parameter_grid,
    is_better_score,
    resolve_optimization_direction,
)


def test_expand_parameter_grid_cartesian_product() -> None:
    """Grid expansion yields the full Cartesian product in deterministic order."""
    combinations = expand_parameter_grid(
        {
            "num_boost_round": [10, 20],
            "learning": [0.1, 0.2],
        }
    )

    assert combinations == (
        {"num_boost_round": 10, "learning": 0.1},
        {"num_boost_round": 10, "learning": 0.2},
        {"num_boost_round": 20, "learning": 0.1},
        {"num_boost_round": 20, "learning": 0.2},
    )


def test_expand_parameter_grid_single_parameter() -> None:
    """Single-parameter grids expand to one mapping per value."""
    combinations = expand_parameter_grid({"num_boost_round": [5, 10, 15]})
    assert combinations == (
        {"num_boost_round": 5},
        {"num_boost_round": 10},
        {"num_boost_round": 15},
    )


def test_expand_parameter_grid_rejects_empty_grid() -> None:
    """Empty parameter grids raise ModelValidationError."""
    with pytest.raises(ModelValidationError, match="must not be empty"):
        expand_parameter_grid({})


def test_expand_parameter_grid_rejects_empty_value_sequence() -> None:
    """Empty value sequences raise ModelValidationError."""
    with pytest.raises(ModelValidationError, match="must not be empty"):
        expand_parameter_grid({"num_boost_round": []})


def test_resolve_optimization_direction_minimize_and_maximize() -> None:
    """Supported metrics resolve to the expected ranking direction."""
    assert resolve_optimization_direction("mae") is OptimizationDirection.MINIMIZE
    assert resolve_optimization_direction("mse") is OptimizationDirection.MINIMIZE
    assert resolve_optimization_direction("rmse") is OptimizationDirection.MINIMIZE
    assert resolve_optimization_direction("r2") is OptimizationDirection.MAXIMIZE
    assert resolve_optimization_direction("accuracy") is OptimizationDirection.MAXIMIZE
    assert resolve_optimization_direction("f1") is OptimizationDirection.MAXIMIZE


def test_resolve_optimization_direction_rejects_unsupported_metric() -> None:
    """Unsupported metrics raise ModelValidationError."""
    with pytest.raises(ModelValidationError, match="unsupported metric"):
        resolve_optimization_direction("roc_auc")
    assert "mae" in SUPPORTED_METRICS


def test_is_better_score_respects_direction() -> None:
    """Score comparison follows minimize and maximize semantics."""
    assert is_better_score(1.0, 2.0, direction=OptimizationDirection.MINIMIZE)
    assert not is_better_score(2.0, 1.0, direction=OptimizationDirection.MINIMIZE)
    assert is_better_score(0.9, 0.8, direction=OptimizationDirection.MAXIMIZE)
    assert not is_better_score(0.7, 0.8, direction=OptimizationDirection.MAXIMIZE)
