"""CQROS Purged Cross Validation Engine exception hierarchy.

Purpose:
    Provide purged-CV-specific exception types used by engines,
    pipelines, repositories, and verification workflows.

Responsibilities:
    - Define the package ``PurgedCVException`` root under
      ``ResearchError``
    - Expose ``PurgedCVError`` for input and contract validation
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
    "PurgedCVError",
    "PurgedCVException",
]


class PurgedCVException(ResearchError):  # noqa: N818
    """Raised when purged cross-validation workflows or artifacts fail."""

    __slots__ = ()


class PurgedCVError(PurgedCVException):
    """Raised when purged-CV inputs, outputs, or contracts fail."""

    __slots__ = ()
