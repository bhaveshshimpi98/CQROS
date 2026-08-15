"""CQROS Training package exception hierarchy.

Purpose:
    Provide training-specific exception types used by the training pipeline
    and related Training package workflows.

Responsibilities:
    - Expose specialized training failures for input and contract validation
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
    "TrainingValidationError",
]


class TrainingValidationError(DatasetError):
    """Raised when training inputs, joins, or contracts fail validation."""

    __slots__ = ()
