"""CQROS ML Workflow exception hierarchy.

Purpose:
    Provide workflow-orchestration exception types used by
    ``TrainingWorkflow`` and related ML pipeline orchestration.

Responsibilities:
    - Re-export ``ModelError`` and ``ModelValidationError`` for workflow
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
