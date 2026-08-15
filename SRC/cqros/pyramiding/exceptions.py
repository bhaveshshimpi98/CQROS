"""CQROS Pyramiding Engine exception hierarchy.

Purpose:
    Provide pyramiding-specific exception types used by engines, pipelines,
    repositories, and verification workflows.

Responsibilities:
    - Define the package ``PyramidingException`` root under ``ResearchError``
    - Expose ``PyramidingValidationError`` for input and contract validation
    - Remain free of logging, validation, and business logic

Dependencies:
    ``cqros.core.exceptions.ResearchError``.

Public API:
    The exception types listed in ``__all__``.
"""

from __future__ import annotations

from cqros.core.exceptions import ResearchError

__all__ = [
    "PyramidingException",
    "PyramidingValidationError",
]


class PyramidingException(ResearchError):  # noqa: N818
    """Raised when pyramiding workflows or artifacts fail."""

    __slots__ = ()


class PyramidingValidationError(PyramidingException):
    """Raised when pyramiding inputs, outputs, or contracts fail validation."""

    __slots__ = ()
