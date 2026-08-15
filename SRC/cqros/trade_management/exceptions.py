"""CQROS Trade Management Engine exception hierarchy.

Purpose:
    Provide trade-management-specific exception types used by managers,
    pipelines, repositories, and verification workflows.

Responsibilities:
    - Define the package ``TradeManagementException`` root under ``ResearchError``
    - Expose ``TradeManagementValidationError`` for input and contract validation
    - Remain free of logging, validation, and business logic

Dependencies:
    ``cqros.core.exceptions.ResearchError``.

Public API:
    The exception types listed in ``__all__``.
"""

from __future__ import annotations

from cqros.core.exceptions import ResearchError

__all__ = [
    "TradeManagementException",
    "TradeManagementValidationError",
]


class TradeManagementException(ResearchError):  # noqa: N818
    """Raised when trade-management workflows or artifacts fail."""

    __slots__ = ()


class TradeManagementValidationError(TradeManagementException):
    """Raised when trade-management inputs, outputs, or contracts fail validation."""

    __slots__ = ()
