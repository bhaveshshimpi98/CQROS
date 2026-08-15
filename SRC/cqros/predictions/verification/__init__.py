"""CQROS prediction verification package public API.

Purpose:
    Expose the merged-prediction dataset verifier and its report/exception
    surface.

Responsibilities:
    - Re-export ``PredictionVerifier``, ``VerificationReport``, and schema
      mismatch exception symbols
    - Remain free of unverified verifier implementations

Public API:
    ``PredictionVerifier``, ``VerificationReport``,
    ``PredictionValidationError``, ``ERROR_REQUIRED_COLUMNS``,
    ``ERROR_SCHEMA_MISMATCH``
"""

from cqros.predictions.verification.exceptions import (
    ERROR_REQUIRED_COLUMNS,
    ERROR_SCHEMA_MISMATCH,
    PredictionValidationError,
)
from cqros.predictions.verification.verifier import PredictionVerifier
from cqros.processing.verification.report import VerificationReport

__all__ = [
    "ERROR_REQUIRED_COLUMNS",
    "ERROR_SCHEMA_MISMATCH",
    "PredictionValidationError",
    "PredictionVerifier",
    "VerificationReport",
]
