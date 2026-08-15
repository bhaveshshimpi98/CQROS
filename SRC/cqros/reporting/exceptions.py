"""CQROS Reporting Engine exception hierarchy.

Purpose:
    Provide reporting-specific exception types used by engines, pipelines,
    repositories, and verification workflows.

Responsibilities:
    - Define the package ``ReportingException`` root under ``ResearchError``
    - Expose ``ReportingValidationError`` for input and contract validation
    - Remain free of logging, validation, and business logic

Dependencies:
    ``cqros.core.exceptions.ResearchError``.

Public API:
    The exception types listed in ``__all__``.

Notes:
    ``CQROSError`` already supports ``error_code``, ``details``, and
    ``recovery_suggestion`` on every subclass. Callers should pass those
    keyword arguments when raising for stable programmatic handling.
"""

from __future__ import annotations

from cqros.core.exceptions import ResearchError

__all__ = [
    "ReportingException",
    "ReportingValidationError",
]


class ReportingException(ResearchError):  # noqa: N818
    """Raised when reporting workflows or reporting artifacts fail."""

    __slots__ = ()


class ReportingValidationError(ReportingException):
    """Raised when reporting inputs, outputs, or contracts fail validation."""

    __slots__ = ()
