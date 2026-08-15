"""CQROS Factor Orthogonalization Engine exception hierarchy.

Purpose:
    Provide factor-orthogonalization-specific exception types used by
    engines, pipelines, repositories, and verification workflows.

Responsibilities:
    - Define the package ``FactorOrthogonalizationException`` root under
      ``ResearchError``
    - Expose ``FactorOrthogonalizationError`` for input and contract
      validation
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
    "FactorOrthogonalizationError",
    "FactorOrthogonalizationException",
]


class FactorOrthogonalizationException(ResearchError):  # noqa: N818
    """Raised when factor-orthogonalization workflows or artifacts fail."""

    __slots__ = ()


class FactorOrthogonalizationError(FactorOrthogonalizationException):
    """Raised when orthogonalization inputs, outputs, or contracts fail."""

    __slots__ = ()
