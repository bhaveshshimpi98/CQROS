"""CQROS processing verification exception surface and error codes.

Purpose:
    Re-export ``ProcessingValidationError`` for verification callers and
    define verification-specific machine-readable error codes used by
    ``BaseVerifier`` and ``VerificationReport`` construction.

Responsibilities:
    - Re-export the shared processing validation exception
    - Expose stable error-code constants for schema and report validation
    - Remain free of verification logic and a separate exception hierarchy

Dependencies:
    ``cqros.processing.exceptions``.

Public API:
    ``ProcessingValidationError`` and the verification error-code constants.
"""

from __future__ import annotations

from typing import Final

from cqros.processing.exceptions import ProcessingValidationError

__all__ = [
    "ERROR_COUNTER_INVALID",
    "ERROR_REQUIRED_COLUMNS",
    "ERROR_ROWS_CHECKED_INVALID",
    "ERROR_WARNINGS_INVALID",
    "ProcessingValidationError",
]

ERROR_REQUIRED_COLUMNS: Final[str] = "PROCESSING-VERIFICATION-001"
ERROR_ROWS_CHECKED_INVALID: Final[str] = "PROCESSING-VERIFICATION-002"
ERROR_COUNTER_INVALID: Final[str] = "PROCESSING-VERIFICATION-003"
ERROR_WARNINGS_INVALID: Final[str] = "PROCESSING-VERIFICATION-004"
