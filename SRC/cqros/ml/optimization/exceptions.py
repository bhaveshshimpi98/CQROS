"""CQROS ML Optimization exception hierarchy.

Purpose:
    Provide optimization-orchestration exception types used by
    ``HyperparameterOptimizer`` and related ML optimization workflows.

Responsibilities:
    - Re-export ``ModelError`` and ``ModelValidationError`` for optimizer
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
