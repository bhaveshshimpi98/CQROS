"""CQROS ML Model exception hierarchy.

Purpose:
    Provide model-specific exception types used by the ML Model abstraction
    layer and future model implementations.

Responsibilities:
    - Re-export the shared ``ModelError`` and ``ModelValidationError`` roots
      from the core taxonomy
    - Remain free of logging, training, inference, and business logic

Dependencies:
    ``cqros.core.exceptions``.

Public API:
    The exception types listed in ``__all__``.
"""

from __future__ import annotations

from cqros.core.exceptions import ModelError, ModelValidationError

__all__ = [
    "ModelError",
    "ModelValidationError",
]
