"""CQROS Portfolio Risk Manager exception hierarchy.

Purpose:
    Provide portfolio-risk-specific exception types used by managers,
    pipelines, repositories, and verification workflows.

Responsibilities:
    - Define the package ``PortfolioRiskException`` root under ``ResearchError``
    - Expose ``PortfolioRiskValidationError`` for input and contract validation
    - Remain free of logging, validation, and business logic

Dependencies:
    ``cqros.core.exceptions.ResearchError``.

Public API:
    The exception types listed in ``__all__``.
"""

from __future__ import annotations

from cqros.core.exceptions import ResearchError

__all__ = [
    "PortfolioRiskException",
    "PortfolioRiskValidationError",
]


class PortfolioRiskException(ResearchError):  # noqa: N818
    """Raised when portfolio-risk workflows or portfolio-risk artifacts fail."""

    __slots__ = ()


class PortfolioRiskValidationError(PortfolioRiskException):
    """Raised when portfolio-risk inputs, outputs, or contracts fail validation."""

    __slots__ = ()
