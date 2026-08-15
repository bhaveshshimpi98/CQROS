"""CQROS OMS exception hierarchy.

Purpose:
    Provide OMS-specific exception types used by order managers and related
    Order Management System package workflows.

Responsibilities:
    - Define the package ``OMSException`` root under research failures
    - Expose ``OMSValidationError`` for input and contract validation
    - Remain free of logging, validation, and business logic

Dependencies:
    ``cqros.core.exceptions.ResearchError``.

Public API:
    The exception types listed in ``__all__``.
"""

from __future__ import annotations

from cqros.core.exceptions import ResearchError

__all__ = [
    "OMSException",
    "OMSValidationError",
]


class OMSException(ResearchError):
    """Raised when order management or OMS workflows fail."""

    __slots__ = ()


class OMSValidationError(OMSException):
    """Raised when OMS inputs, outputs, or contracts fail validation."""

    __slots__ = ()
