"""CQROS prediction verification exception surface and error codes.

Purpose:
    Expose prediction-verification error codes and re-export
    ``PredictionValidationError`` for schema and dtype failures raised by
    ``PredictionVerifier``.

Responsibilities:
    - Define stable machine-readable error codes for prediction verification
    - Re-export ``PredictionValidationError`` for callers
    - Remain free of verification logic

Dependencies:
    ``cqros.predictions.exceptions``.

Public API:
    ``PredictionValidationError`` and the prediction-verification error-code
    constants.
"""

from __future__ import annotations

from typing import Final

from cqros.predictions.exceptions import PredictionValidationError

__all__ = [
    "ERROR_REQUIRED_COLUMNS",
    "ERROR_SCHEMA_MISMATCH",
    "PredictionValidationError",
]

ERROR_REQUIRED_COLUMNS: Final[str] = "PREDICTION-VERIFICATION-001"
ERROR_SCHEMA_MISMATCH: Final[str] = "PREDICTION-VERIFICATION-002"
