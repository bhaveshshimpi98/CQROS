"""CQROS Risk exception hierarchy.

Purpose:
    Provide risk-specific exception types used by risk managers and related
    Risk Management package workflows.

Responsibilities:
    - Define the package ``RiskError`` root under research failures
    - Expose ``RiskValidationError`` for input and contract validation
    - Remain free of logging, validation, and business logic

Dependencies:
    ``cqros.core.exceptions.ResearchError``.

Public API:
    The exception types listed in ``__all__``.
"""

from __future__ import annotations

from cqros.core.exceptions import ResearchError

__all__ = [
    "RiskError",
    "RiskValidationError",
]


class RiskError(ResearchError):
    """Raised when risk evaluation or risk workflows fail."""

    __slots__ = ()


class RiskValidationError(RiskError):
    """Raised when risk inputs, outputs, or contracts fail validation."""

    __slots__ = ()
