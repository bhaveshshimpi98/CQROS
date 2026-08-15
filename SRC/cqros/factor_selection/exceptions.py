"""CQROS Factor Selection Engine exception hierarchy.

Purpose:
    Provide factor-selection-specific exception types used by engines,
    pipelines, repositories, and verification workflows.

Responsibilities:
    - Define the package ``FactorSelectionException`` root under
      ``ResearchError``
    - Expose ``FactorSelectionError`` for input and contract validation
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
    "FactorSelectionError",
    "FactorSelectionException",
]


class FactorSelectionException(ResearchError):  # noqa: N818
    """Raised when factor-selection workflows or artifacts fail."""

    __slots__ = ()


class FactorSelectionError(FactorSelectionException):
    """Raised when factor-selection inputs, outputs, or contracts fail."""

    __slots__ = ()
