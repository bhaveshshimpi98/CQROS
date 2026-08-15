"""CQROS Label Engine exception hierarchy.

Purpose:
    Provide label-specific exception types used by the label pipeline and
    related Label Engine workflows.

Responsibilities:
    - Expose specialized label failures for input and contract validation
    - Remain free of logging, validation, and business logic

Dependencies:
    ``cqros.core.exceptions.TargetError``.

Public API:
    The exception types listed in ``__all__``.
"""

from __future__ import annotations

from cqros.core.exceptions import TargetError

__all__ = [
    "LabelValidationError",
    "TargetError",
]


class LabelValidationError(TargetError):
    """Raised when label inputs, outputs, or contracts fail validation."""

    __slots__ = ()
