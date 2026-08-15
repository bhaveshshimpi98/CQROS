"""CQROS execution exception hierarchy.

Purpose:
    Provide execution-specific exception types used by simulators, pipelines,
    repositories, and verification workflows.

Responsibilities:
    - Define the package ``ExecutionException`` root under ``ExecutionError``
    - Expose ``ExecutionValidationError`` for input and contract validation
    - Remain free of logging, validation, and business logic

Dependencies:
    ``cqros.core.exceptions.ExecutionError``.

Public API:
    The exception types listed in ``__all__``.
"""

from __future__ import annotations

from cqros.core.exceptions import ExecutionError

__all__ = [
    "ExecutionException",
    "ExecutionValidationError",
]


class ExecutionException(ExecutionError):  # noqa: N818
    """Raised when execution workflows or trade artifacts fail."""

    __slots__ = ()


class ExecutionValidationError(ExecutionException):
    """Raised when execution inputs, outputs, or contracts fail validation."""

    __slots__ = ()
