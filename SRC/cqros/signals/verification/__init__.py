"""CQROS signal verification package public API.

Purpose:
    Expose the canonical signal dataset verifier and its report/exception
    surface.

Responsibilities:
    - Re-export ``SignalVerifier``, ``VerificationReport``, and schema
      mismatch exception symbols
    - Remain free of unverified verifier implementations

Public API:
    ``SignalVerifier``, ``VerificationReport``, ``SignalValidationError``,
    ``ERROR_REQUIRED_COLUMNS``, ``ERROR_SCHEMA_MISMATCH``
"""

from cqros.processing.verification.report import VerificationReport
from cqros.signals.verification.exceptions import (
    ERROR_REQUIRED_COLUMNS,
    ERROR_SCHEMA_MISMATCH,
    SignalValidationError,
)
from cqros.signals.verification.verifier import SignalVerifier

__all__ = [
    "ERROR_REQUIRED_COLUMNS",
    "ERROR_SCHEMA_MISMATCH",
    "SignalValidationError",
    "SignalVerifier",
    "VerificationReport",
]
