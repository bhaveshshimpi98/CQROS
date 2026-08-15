"""CQROS training verification exception surface and error codes.

Purpose:
    Expose training-verification error codes and re-export
    ``TrainingValidationError`` for schema and dtype failures raised by
    ``TrainingVerifier``.

Responsibilities:
    - Define stable machine-readable error codes for training verification
    - Re-export ``TrainingValidationError`` for callers
    - Remain free of verification logic

Dependencies:
    ``cqros.training.exceptions``.

Public API:
    ``TrainingValidationError`` and the training-verification error-code
    constants.
"""

from __future__ import annotations

from typing import Final

from cqros.training.exceptions import TrainingValidationError

__all__ = [
    "ERROR_REQUIRED_COLUMNS",
    "ERROR_SCHEMA_MISMATCH",
    "TrainingValidationError",
]

ERROR_REQUIRED_COLUMNS: Final[str] = "TRAINING-VERIFICATION-001"
ERROR_SCHEMA_MISMATCH: Final[str] = "TRAINING-VERIFICATION-002"
