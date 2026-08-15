"""CQROS Exit Engine exception hierarchy.

Purpose:
    Provide exit-engine-specific exception types used by engines, pipelines,
    repositories, and verification workflows.

Responsibilities:
    - Define the package ``ExitEngineException`` root under ``ResearchError``
    - Expose ``ExitEngineValidationError`` for input and contract validation
    - Remain free of logging, validation, and business logic

Dependencies:
    ``cqros.core.exceptions.ResearchError``.

Public API:
    The exception types listed in ``__all__``.
"""

from __future__ import annotations

from cqros.core.exceptions import ResearchError

__all__ = [
    "ExitEngineException",
    "ExitEngineValidationError",
]


class ExitEngineException(ResearchError):  # noqa: N818
    """Raised when exit-engine workflows or artifacts fail."""

    __slots__ = ()


class ExitEngineValidationError(ExitEngineException):
    """Raised when exit-engine inputs, outputs, or contracts fail validation."""

    __slots__ = ()
