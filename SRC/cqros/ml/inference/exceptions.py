"""CQROS ML Inference exception hierarchy.

Purpose:
    Provide inference-orchestration exception types used by
    ``PredictionPipeline`` and related ML prediction workflows.

Responsibilities:
    - Re-export ``ModelError`` and ``ModelValidationError`` for pipeline
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
