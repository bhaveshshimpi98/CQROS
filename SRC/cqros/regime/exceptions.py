"""CQROS Regime Engine exception hierarchy.

Purpose:
    Provide regime-specific exception types used by engines, pipelines,
    repositories, and verification workflows.

Responsibilities:
    - Define the package ``RegimeException`` root under ``ResearchError``
    - Expose ``RegimeError`` for input and contract validation
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
    "RegimeError",
    "RegimeException",
]


class RegimeException(ResearchError):  # noqa: N818
    """Raised when regime workflows or artifacts fail."""

    __slots__ = ()


class RegimeError(RegimeException):
    """Raised when regime inputs, outputs, or contracts fail."""

    __slots__ = ()
