"""Unit tests for CQROS evaluation metric helpers."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)

from cqros.ml.evaluation.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
)


def test_regression_metrics_match_sklearn() -> None:
    """Regression helpers match scikit-learn reference values."""
    y_true = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=float)
    y_pred = np.asarray([1.5, 2.5, 2.5, 3.5], dtype=float)

    metrics = compute_regression_metrics(y_true, y_pred)

    assert metrics["mae"] == mean_absolute_error(y_true, y_pred)
    assert metrics["mse"] == mean_squared_error(y_true, y_pred)
    assert metrics["rmse"] == root_mean_squared_error(y_true, y_pred)
    assert metrics["r2"] == r2_score(y_true, y_pred)
    assert math.isclose(metrics["rmse"], math.sqrt(metrics["mse"]))


def test_regression_perfect_predictions() -> None:
    """Perfect regression predictions yield zero error and unit R²."""
    values = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=float)
    metrics = compute_regression_metrics(values, values)

    assert metrics["mae"] == 0.0
    assert metrics["mse"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0


def test_classification_metrics_match_sklearn() -> None:
    """Classification helpers match scikit-learn macro-averaged values."""
    y_true = np.asarray([0, 1, 0, 1, 1, 0], dtype=int)
    y_pred = np.asarray([0, 1, 1, 1, 0, 0], dtype=int)

    metrics = compute_classification_metrics(y_true, y_pred)

    assert metrics["accuracy"] == accuracy_score(y_true, y_pred)
    assert metrics["precision"] == precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0.0,
    )
    assert metrics["recall"] == recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0.0,
    )
    assert metrics["f1"] == f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0.0,
    )


def test_classification_perfect_predictions() -> None:
    """Perfect classification predictions yield unit scores."""
    labels = np.asarray([0, 1, 0, 1], dtype=int)
    metrics = compute_classification_metrics(labels, labels)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
