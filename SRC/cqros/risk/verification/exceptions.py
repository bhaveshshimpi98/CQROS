"""CQROS risk verification exception surface and error codes.

Purpose:
    Expose risk-verification error codes and re-export
    ``RiskValidationError`` for schema and dtype failures raised by
    ``RiskVerifier``.

Responsibilities:
    - Define stable machine-readable error codes for risk verification
    - Re-export ``RiskValidationError`` for callers
    - Remain free of verification logic

Dependencies:
    ``cqros.risk.exceptions``.

Public API:
    ``RiskValidationError`` and the risk-verification error-code constants.
"""

from __future__ import annotations

from typing import Final

from cqros.risk.exceptions import RiskValidationError

__all__ = [
    "ERROR_REQUIRED_COLUMNS",
    "ERROR_SCHEMA_MISMATCH",
    "RiskValidationError",
]

ERROR_REQUIRED_COLUMNS: Final[str] = "RISK-VERIFICATION-001"
ERROR_SCHEMA_MISMATCH: Final[str] = "RISK-VERIFICATION-002"
