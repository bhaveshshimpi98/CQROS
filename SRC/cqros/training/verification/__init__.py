"""CQROS training verification package public API.

Purpose:
    Expose the merged-training dataset verifier and its report/exception
    surface.

Responsibilities:
    - Re-export ``TrainingVerifier``, ``VerificationReport``, and schema
      mismatch exception symbols
    - Remain free of unverified verifier implementations

Public API:
    ``TrainingVerifier``, ``VerificationReport``, ``TrainingValidationError``,
    ``ERROR_REQUIRED_COLUMNS``, ``ERROR_SCHEMA_MISMATCH``
"""

from cqros.processing.verification.report import VerificationReport
from cqros.training.verification.exceptions import (
    ERROR_REQUIRED_COLUMNS,
    ERROR_SCHEMA_MISMATCH,
    TrainingValidationError,
)
from cqros.training.verification.verifier import TrainingVerifier

__all__ = [
    "ERROR_REQUIRED_COLUMNS",
    "ERROR_SCHEMA_MISMATCH",
    "TrainingValidationError",
    "TrainingVerifier",
    "VerificationReport",
]
