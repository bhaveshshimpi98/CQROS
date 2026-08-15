"""CQROS Portfolio exception hierarchy.

Purpose:
    Provide portfolio-specific exception types used by portfolio optimizers
    and related Portfolio package workflows.

Responsibilities:
    - Define the package ``PortfolioError`` root under research failures
    - Expose ``PortfolioValidationError`` for input and contract validation
    - Remain free of logging, validation, and business logic

Dependencies:
    ``cqros.core.exceptions.ResearchError``.

Public API:
    The exception types listed in ``__all__``.
"""

from __future__ import annotations

from cqros.core.exceptions import ResearchError

__all__ = [
    "PortfolioError",
    "PortfolioValidationError",
]


class PortfolioError(ResearchError):
    """Raised when portfolio optimization or portfolio workflows fail."""

    __slots__ = ()


class PortfolioValidationError(PortfolioError):
    """Raised when portfolio inputs, outputs, or contracts fail validation."""

    __slots__ = ()
