"""CQROS label verification package public API.

Purpose:
    Expose the merged-label dataset verifier and its report/exception
    surface.

Responsibilities:
    - Re-export ``LabelVerifier``, ``VerificationReport``, and schema
      mismatch exception symbols
    - Remain free of unverified verifier implementations

Public API:
    ``LabelVerifier``, ``VerificationReport``, ``LabelValidationError``,
    ``ERROR_REQUIRED_COLUMNS``, ``ERROR_SCHEMA_MISMATCH``
"""

from cqros.labels.verification.exceptions import (
    ERROR_REQUIRED_COLUMNS,
    ERROR_SCHEMA_MISMATCH,
    LabelValidationError,
)
from cqros.labels.verification.verifier import LabelVerifier
from cqros.processing.verification.report import VerificationReport

__all__ = [
    "ERROR_REQUIRED_COLUMNS",
    "ERROR_SCHEMA_MISMATCH",
    "LabelValidationError",
    "LabelVerifier",
    "VerificationReport",
]
