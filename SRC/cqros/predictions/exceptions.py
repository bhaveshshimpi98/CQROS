"""CQROS Predictions exception hierarchy.

Purpose:
    Provide prediction-specific exception types used by the prediction
    pipeline and related Predictions package workflows.

Responsibilities:
    - Expose specialized prediction failures for input and contract validation
    - Remain free of logging, validation, and business logic

Dependencies:
    ``cqros.core.exceptions.DatasetError``.

Public API:
    The exception types listed in ``__all__``.
"""

from __future__ import annotations

from cqros.core.exceptions import DatasetError

__all__ = [
    "DatasetError",
    "PredictionValidationError",
]


class PredictionValidationError(DatasetError):
    """Raised when prediction inputs, outputs, or contracts fail validation."""

    __slots__ = ()
