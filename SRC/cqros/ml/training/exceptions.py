"""CQROS ML Training exception hierarchy.

Purpose:
    Provide training-orchestration exception types used by ``ModelTrainer``
    and related ML training workflows.

Responsibilities:
    - Re-export ``ModelError`` and ``ModelValidationError`` for trainer
      validation failures
    - Remain free of logging, orchestration, and business logic

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
