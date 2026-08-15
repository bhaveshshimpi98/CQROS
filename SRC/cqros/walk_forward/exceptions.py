"""CQROS Walk-Forward Engine exception hierarchy.

Purpose:
    Provide walk-forward-specific exception types used by engines,
    pipelines, repositories, and verification workflows.

Responsibilities:
    - Define the package ``WalkForwardException`` root under
      ``ResearchError``
    - Expose ``WalkForwardError`` for input and contract validation
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
    "WalkForwardError",
    "WalkForwardException",
]


class WalkForwardException(ResearchError):  # noqa: N818
    """Raised when walk-forward workflows or artifacts fail."""

    __slots__ = ()


class WalkForwardError(WalkForwardException):
    """Raised when walk-forward inputs, outputs, or contracts fail."""

    __slots__ = ()
