"""CQROS Signals exception hierarchy.

Purpose:
    Provide signal-specific exception types used by the signal pipeline and
    related Signals package workflows.

Responsibilities:
    - Expose specialized signal failures for input and contract validation
    - Remain free of logging, validation, and business logic

Dependencies:
    ``cqros.core.exceptions.DatasetError``.

Public API:
    The exception types listed in ``__all__``.
"""

from __future__ import annotations

from cqros.core.exceptions import DatasetError

__all__ = [
    "DatasetError",
    "SignalValidationError",
]


class SignalValidationError(DatasetError):
    """Raised when signal inputs, outputs, or contracts fail validation."""

    __slots__ = ()
