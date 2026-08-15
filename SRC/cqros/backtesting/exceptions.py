"""CQROS Backtesting Engine exception hierarchy.

Purpose:
    Provide backtesting-specific exception types used by engines, pipelines,
    repositories, and verification workflows.

Responsibilities:
    - Define the package ``BacktestingException`` root under ``ResearchError``
    - Expose ``BacktestingValidationError`` for input and contract validation
    - Remain free of logging, validation, and business logic

Dependencies:
    ``cqros.core.exceptions.ResearchError``.

Public API:
    The exception types listed in ``__all__``.
"""

from __future__ import annotations

from cqros.core.exceptions import ResearchError

__all__ = [
    "BacktestingException",
    "BacktestingValidationError",
]


class BacktestingException(ResearchError):  # noqa: N818
    """Raised when backtesting workflows or backtesting artifacts fail."""

    __slots__ = ()


class BacktestingValidationError(BacktestingException):
    """Raised when backtesting inputs, outputs, or contracts fail validation."""

    __slots__ = ()
