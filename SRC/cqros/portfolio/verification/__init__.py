"""CQROS portfolio verification package public API.

Purpose:
    Expose the canonical portfolio dataset verifier and its report/exception
    surface.

Responsibilities:
    - Re-export ``PortfolioVerifier``, ``VerificationReport``, and schema
      mismatch exception symbols
    - Remain free of unverified verifier implementations

Public API:
    ``PortfolioVerifier``, ``VerificationReport``, ``PortfolioValidationError``,
    ``ERROR_REQUIRED_COLUMNS``, ``ERROR_SCHEMA_MISMATCH``
"""

from cqros.portfolio.verification.exceptions import (
    ERROR_REQUIRED_COLUMNS,
    ERROR_SCHEMA_MISMATCH,
    PortfolioValidationError,
)
from cqros.portfolio.verification.verifier import PortfolioVerifier
from cqros.processing.verification.report import VerificationReport

__all__ = [
    "ERROR_REQUIRED_COLUMNS",
    "ERROR_SCHEMA_MISMATCH",
    "PortfolioValidationError",
    "PortfolioVerifier",
    "VerificationReport",
]
