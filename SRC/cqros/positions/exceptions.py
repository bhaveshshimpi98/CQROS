"""CQROS Position Engine exception hierarchy.

Purpose:
    Provide position-specific exception types used by engines, pipelines,
    repositories, and verification workflows.

Responsibilities:
    - Define the package ``PositionException`` root under ``ResearchError``
    - Expose ``PositionValidationError`` for input and contract validation
    - Remain free of logging, validation, and business logic

Dependencies:
    ``cqros.core.exceptions.ResearchError``.

Public API:
    The exception types listed in ``__all__``.
"""

from __future__ import annotations

from cqros.core.exceptions import ResearchError

__all__ = [
    "PositionException",
    "PositionValidationError",
]


class PositionException(ResearchError):  # noqa: N818
    """Raised when position workflows or position artifacts fail."""

    __slots__ = ()


class PositionValidationError(PositionException):
    """Raised when position inputs, outputs, or contracts fail validation."""

    __slots__ = ()
