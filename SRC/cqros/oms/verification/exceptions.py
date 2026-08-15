"""CQROS OMS order verification exception surface and error codes.

Purpose:
    Expose order-verification error codes and re-export
    ``OMSValidationError`` for schema and dtype failures raised by
    ``OrderVerifier``.

Responsibilities:
    - Define stable machine-readable error codes for order verification
    - Re-export ``OMSValidationError`` for callers
    - Remain free of verification logic

Dependencies:
    ``cqros.oms.exceptions``.

Public API:
    ``OMSValidationError`` and the order-verification error-code constants.
"""

from __future__ import annotations

from typing import Final

from cqros.oms.exceptions import OMSValidationError

__all__ = [
    "ERROR_REQUIRED_COLUMNS",
    "ERROR_SCHEMA_MISMATCH",
    "OMSValidationError",
]

ERROR_REQUIRED_COLUMNS: Final[str] = "OMS-VERIFICATION-001"
ERROR_SCHEMA_MISMATCH: Final[str] = "OMS-VERIFICATION-002"
