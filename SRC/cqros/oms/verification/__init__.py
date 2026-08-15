"""CQROS OMS order verification package public API.

Purpose:
    Expose the canonical OMS order dataset verifier and its report/exception
    surface.

Responsibilities:
    - Re-export ``OrderVerifier``, ``VerificationReport``, and schema mismatch
      exception symbols
    - Remain free of unverified verifier implementations

Public API:
    ``OrderVerifier``, ``VerificationReport``, ``OMSValidationError``,
    ``ERROR_REQUIRED_COLUMNS``, ``ERROR_SCHEMA_MISMATCH``
"""

from cqros.oms.verification.exceptions import (
    ERROR_REQUIRED_COLUMNS,
    ERROR_SCHEMA_MISMATCH,
    OMSValidationError,
)
from cqros.oms.verification.verifier import OrderVerifier
from cqros.processing.verification.report import VerificationReport

__all__ = [
    "ERROR_REQUIRED_COLUMNS",
    "ERROR_SCHEMA_MISMATCH",
    "OMSValidationError",
    "OrderVerifier",
    "VerificationReport",
]
