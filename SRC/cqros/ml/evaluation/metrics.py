"""CQROS ML Evaluation metric helpers.

Purpose:
    Isolate supervised learning metric calculations used by
    ``ModelEvaluator`` behind a small, framework-independent surface.

Responsibilities:
    - Compute regression metrics via scikit-learn
    - Compute classification metrics via scikit-learn
    - Remain free of model fitting, prediction, and orchestration logic

Dependencies:
    ``numpy`` and ``scikit-learn``.

Public API:
    ``compute_regression_metrics``, ``compute_classification_metrics``
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, cast

import numpy as np
from sklearn import metrics as sk_metrics  # pyright: ignore[reportMissingTypeStubs]

__all__ = [
    "compute_classification_metrics",
    "compute_regression_metrics",
]

_ZERO_DIVISION: Final[Any] = 0
_MACRO_AVERAGE: Final[str] = "macro"


def compute_regression_metrics(
    y_true: np.ndarray[Any, np.dtype[Any]],
    y_pred: np.ndarray[Any, np.dtype[Any]],
) -> Mapping[str, float]:
    """Compute standard regression metrics for ``y_true`` vs ``y_pred``.

    Args:
        y_true: Ground-truth continuous labels.
        y_pred: Predicted continuous scores.

    Returns:
        Mapping with ``mae``, ``mse``, ``rmse``, and ``r2``.
    """
    true_values = np.asarray(y_true, dtype=float)
    pred_values = np.asarray(y_pred, dtype=float)
    return {
        "mae": _as_float(
            sk_metrics.mean_absolute_error(  # pyright: ignore[reportUnknownMemberType]
                true_values,
                pred_values,
            )
        ),
        "mse": _as_float(
            sk_metrics.mean_squared_error(  # pyright: ignore[reportUnknownMemberType]
                true_values,
                pred_values,
            )
        ),
        "rmse": _as_float(
            sk_metrics.root_mean_squared_error(  # pyright: ignore[reportUnknownMemberType]
                true_values,
                pred_values,
            )
        ),
        "r2": _as_float(
            sk_metrics.r2_score(  # pyright: ignore[reportUnknownMemberType]
                true_values,
                pred_values,
            )
        ),
    }


def compute_classification_metrics(
    y_true: np.ndarray[Any, np.dtype[Any]],
    y_pred: np.ndarray[Any, np.dtype[Any]],
) -> Mapping[str, float]:
    """Compute standard classification metrics for ``y_true`` vs ``y_pred``.

    Precision, recall, and F1 use macro averaging.

    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels.

    Returns:
        Mapping with ``accuracy``, ``precision``, ``recall``, and ``f1``.
    """
    true_values = np.asarray(y_true)
    pred_values = np.asarray(y_pred)
    return {
        "accuracy": _as_float(
            sk_metrics.accuracy_score(  # pyright: ignore[reportUnknownMemberType]
                true_values,
                pred_values,
            )
        ),
        "precision": _as_float(
            sk_metrics.precision_score(  # pyright: ignore[reportUnknownMemberType]
                true_values,
                pred_values,
                average=_MACRO_AVERAGE,
                zero_division=_ZERO_DIVISION,
            )
        ),
        "recall": _as_float(
            sk_metrics.recall_score(  # pyright: ignore[reportUnknownMemberType]
                true_values,
                pred_values,
                average=_MACRO_AVERAGE,
                zero_division=_ZERO_DIVISION,
            )
        ),
        "f1": _as_float(
            sk_metrics.f1_score(  # pyright: ignore[reportUnknownMemberType]
                true_values,
                pred_values,
                average=_MACRO_AVERAGE,
                zero_division=_ZERO_DIVISION,
            )
        ),
    }


def _as_float(value: Any) -> float:
    """Convert a scikit-learn metric result to ``float``."""
    return float(cast(float, value))
