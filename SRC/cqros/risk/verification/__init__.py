"""CQROS risk verification package public API.

Purpose:
    Expose the canonical risk-decision dataset verifier and its
    report/exception surface.

Responsibilities:
    - Re-export ``RiskVerifier``, ``VerificationReport``, and schema mismatch
      exception symbols
    - Remain free of unverified verifier implementations

Public API:
    ``RiskVerifier``, ``VerificationReport``, ``RiskValidationError``,
    ``ERROR_REQUIRED_COLUMNS``, ``ERROR_SCHEMA_MISMATCH``
"""

from cqros.processing.verification.report import VerificationReport
from cqros.risk.verification.exceptions import (
    ERROR_REQUIRED_COLUMNS,
    ERROR_SCHEMA_MISMATCH,
    RiskValidationError,
)
from cqros.risk.verification.verifier import RiskVerifier

__all__ = [
    "ERROR_REQUIRED_COLUMNS",
    "ERROR_SCHEMA_MISMATCH",
    "RiskValidationError",
    "RiskVerifier",
    "VerificationReport",
]
