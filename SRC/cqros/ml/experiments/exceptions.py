"""CQROS ML Experiment exception hierarchy.

Purpose:
    Provide experiment-tracking exception types used by
    ``ExperimentTracker`` and related ML experiment workflows.

Responsibilities:
    - Re-export ``ModelError`` and ``ModelValidationError`` for tracker
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
