"""CQROS factor verification exception surface and error codes.

Purpose:
    Expose factor-verification error codes and re-export
    ``FactorValidationError`` for schema and dtype failures raised by
    ``FactorVerifier``.

Responsibilities:
    - Define stable machine-readable error codes for factor verification
    - Re-export ``FactorValidationError`` for callers
    - Remain free of verification logic

Dependencies:
    ``cqros.factors.exceptions``.

Public API:
    ``FactorValidationError`` and the factor-verification error-code
    constants.
"""

from __future__ import annotations

from typing import Final

from cqros.factors.exceptions import FactorValidationError

__all__ = [
    "ERROR_REQUIRED_COLUMNS",
    "ERROR_SCHEMA_MISMATCH",
    "FactorValidationError",
]

ERROR_REQUIRED_COLUMNS: Final[str] = "FACTOR-VERIFICATION-001"
ERROR_SCHEMA_MISMATCH: Final[str] = "FACTOR-VERIFICATION-002"
