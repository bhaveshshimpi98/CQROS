"""CQROS Monitoring Engine exception hierarchy.

Purpose:
    Provide monitoring-specific exception types used by engines, pipelines,
    repositories, and verification workflows.

Responsibilities:
    - Define the package ``MonitoringException`` root under ``ResearchError``
    - Expose ``MonitoringValidationError`` for input and contract validation
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
    "MonitoringException",
    "MonitoringValidationError",
]


class MonitoringException(ResearchError):  # noqa: N818
    """Raised when monitoring workflows or monitoring artifacts fail."""

    __slots__ = ()


class MonitoringValidationError(MonitoringException):
    """Raised when monitoring inputs, outputs, or contracts fail validation."""

    __slots__ = ()
