"""CQROS Data Processing Framework exception hierarchy.

Purpose:
    Provide processing-specific exception types used by registry, pipeline,
    validation, and step execution workflows.

Responsibilities:
    - Define the package ``ProcessingError`` root under data-layer failures
    - Expose specialized processing failures for registration, lookup,
      validation, and execution
    - Remain free of logging, validation, and business logic

Dependencies:
    ``cqros.core.exceptions.DataError``.

Public API:
    The exception types listed in ``__all__``.
"""

from __future__ import annotations

from cqros.core.exceptions import DataError

__all__ = [
    "ProcessingError",
    "ProcessingValidationError",
    "ProcessingRegistrationError",
    "UnknownProcessingStepError",
    "DuplicateProcessingStepError",
    "ProcessingExecutionError",
]


class ProcessingError(DataError):
    """Raised when data processing or processing metadata handling fails."""

    __slots__ = ()


class ProcessingValidationError(ProcessingError):
    """Raised when processing inputs, outputs, or contracts fail validation."""

    __slots__ = ()


class ProcessingRegistrationError(ProcessingError):
    """Raised when registering a processing step fails."""

    __slots__ = ()


class UnknownProcessingStepError(ProcessingError):
    """Raised when a requested processing step cannot be found."""

    __slots__ = ()


class DuplicateProcessingStepError(ProcessingError):
    """Raised when a processing step name is already registered."""

    __slots__ = ()


class ProcessingExecutionError(ProcessingError):
    """Raised when processing step execution fails."""

    __slots__ = ()
