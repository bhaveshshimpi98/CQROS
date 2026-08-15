"""CQROS Portfolio Accounting Engine exception hierarchy.

Purpose:
    Provide accounting-specific exception types used by engines, pipelines,
    repositories, and verification workflows.

Responsibilities:
    - Define the package ``AccountingException`` root under ``ResearchError``
    - Expose ``AccountingValidationError`` for input and contract validation
    - Remain free of logging, validation, and business logic

Dependencies:
    ``cqros.core.exceptions.ResearchError``.

Public API:
    The exception types listed in ``__all__``.
"""

from __future__ import annotations

from cqros.core.exceptions import ResearchError

__all__ = [
    "AccountingException",
    "AccountingValidationError",
]


class AccountingException(ResearchError):  # noqa: N818
    """Raised when accounting workflows or accounting artifacts fail."""

    __slots__ = ()


class AccountingValidationError(AccountingException):
    """Raised when accounting inputs, outputs, or contracts fail validation."""

    __slots__ = ()
