"""CQROS feature verification package public API.

Purpose:
    Expose the merged-feature dataset verifier and its report/exception
    surface.

Responsibilities:
    - Re-export ``FeatureVerifier``, ``VerificationReport``, and schema
      mismatch exception symbols
    - Remain free of unverified verifier implementations

Public API:
    ``FeatureVerifier``, ``VerificationReport``, ``FeatureValidationError``,
    ``ERROR_SCHEMA_MISMATCH``
"""

from cqros.features.verification.exceptions import (
    ERROR_SCHEMA_MISMATCH,
    FeatureValidationError,
)
from cqros.features.verification.verifier import FeatureVerifier
from cqros.processing.verification.report import VerificationReport

__all__ = [
    "ERROR_SCHEMA_MISMATCH",
    "FeatureValidationError",
    "FeatureVerifier",
    "VerificationReport",
]
