"""CQROS label verification exception surface and error codes.

Purpose:
    Expose label-verification error codes and re-export
    ``LabelValidationError`` for schema and dtype failures raised by
    ``LabelVerifier``.

Responsibilities:
    - Define stable machine-readable error codes for label verification
    - Re-export ``LabelValidationError`` for callers
    - Remain free of verification logic

Dependencies:
    ``cqros.labels.exceptions``.

Public API:
    ``LabelValidationError`` and the label-verification error-code
    constants.
"""

from __future__ import annotations

from typing import Final

from cqros.labels.exceptions import LabelValidationError

__all__ = [
    "ERROR_REQUIRED_COLUMNS",
    "ERROR_SCHEMA_MISMATCH",
    "LabelValidationError",
]

ERROR_REQUIRED_COLUMNS: Final[str] = "LABEL-VERIFICATION-001"
ERROR_SCHEMA_MISMATCH: Final[str] = "LABEL-VERIFICATION-002"
