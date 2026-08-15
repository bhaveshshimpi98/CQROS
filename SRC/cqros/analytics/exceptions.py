"""CQROS Analytics Engine exception hierarchy.

Purpose:
    Provide analytics-specific exception types used by engines, pipelines,
    repositories, and verification workflows.

Responsibilities:
    - Define the package ``AnalyticsException`` root under ``ResearchError``
    - Expose ``AnalyticsValidationError`` for input and contract validation
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
    "AnalyticsException",
    "AnalyticsValidationError",
]


class AnalyticsException(ResearchError):  # noqa: N818
    """Raised when analytics workflows or analytics artifacts fail."""

    __slots__ = ()


class AnalyticsValidationError(AnalyticsException):
    """Raised when analytics inputs, outputs, or contracts fail validation."""

    __slots__ = ()
