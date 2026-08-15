"""CQROS signal verification exception surface and error codes.

Purpose:
    Expose signal-verification error codes and re-export
    ``SignalValidationError`` for schema and dtype failures raised by
    ``SignalVerifier``.

Responsibilities:
    - Define stable machine-readable error codes for signal verification
    - Re-export ``SignalValidationError`` for callers
    - Remain free of verification logic

Dependencies:
    ``cqros.signals.exceptions``.

Public API:
    ``SignalValidationError`` and the signal-verification error-code
    constants.
"""

from __future__ import annotations

from typing import Final

from cqros.signals.exceptions import SignalValidationError

__all__ = [
    "ERROR_REQUIRED_COLUMNS",
    "ERROR_SCHEMA_MISMATCH",
    "SignalValidationError",
]

ERROR_REQUIRED_COLUMNS: Final[str] = "SIGNAL-VERIFICATION-001"
ERROR_SCHEMA_MISMATCH: Final[str] = "SIGNAL-VERIFICATION-002"
