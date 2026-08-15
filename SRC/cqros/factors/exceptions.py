"""CQROS Factor Research Engine exception hierarchy.

Purpose:
    Provide factor-specific exception types used by registry, pipeline,
    validation, and research workflows.

Responsibilities:
    - Define the package ``FactorError`` root under research failures
    - Expose specialized factor failures for registration, lookup, and
      dataset validation
    - Remain free of logging, validation, and business logic

Dependencies:
    ``cqros.core.exceptions.ResearchError``.

Public API:
    The exception types listed in ``__all__``.
"""

from __future__ import annotations

from cqros.core.exceptions import ResearchError

__all__ = [
    "FactorError",
    "FactorExecutionError",
    "FactorRegistrationError",
    "FactorValidationError",
    "UnknownFactorError",
]


class FactorError(ResearchError):
    """Raised when factor research or factor metadata handling fails."""

    __slots__ = ()


class FactorRegistrationError(FactorError):
    """Raised when registering a factor with the Factor Research Engine fails."""

    __slots__ = ()


class FactorValidationError(FactorError):
    """Raised when factor inputs, outputs, or contracts fail validation."""

    __slots__ = ()


class FactorExecutionError(FactorError):
    """Raised when factor compute execution fails inside a pipeline."""

    __slots__ = ()


class UnknownFactorError(FactorError):
    """Raised when a requested factor cannot be found."""

    __slots__ = ()
