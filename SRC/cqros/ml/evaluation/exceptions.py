"""CQROS ML Evaluation exception hierarchy.

Purpose:
    Provide evaluation-orchestration exception types used by
    ``ModelEvaluator``, ``TimeSeriesCrossValidator``, and related ML
    evaluation workflows.

Responsibilities:
    - Re-export ``ModelError`` and ``ModelValidationError`` for evaluator and
      cross-validator validation failures
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
