"""CQROS feature verification exception surface and error codes.

Purpose:
    Expose feature-verification error codes and re-export
    ``FeatureValidationError`` for schema and dtype failures raised by
    ``FeatureVerifier``.

Responsibilities:
    - Define stable machine-readable error codes for feature verification
    - Re-export ``FeatureValidationError`` for callers
    - Remain free of verification logic

Dependencies:
    ``cqros.features.exceptions``.

Public API:
    ``FeatureValidationError`` and the feature-verification error-code
    constants.
"""

from __future__ import annotations

from typing import Final

from cqros.features.exceptions import FeatureValidationError

__all__ = [
    "ERROR_SCHEMA_MISMATCH",
    "FeatureValidationError",
]

ERROR_SCHEMA_MISMATCH: Final[str] = "FEATURE-VERIFICATION-001"
