"""CQROS portfolio verification exception surface and error codes.

Purpose:
    Expose portfolio-verification error codes and re-export
    ``PortfolioValidationError`` for schema and dtype failures raised by
    ``PortfolioVerifier``.

Responsibilities:
    - Define stable machine-readable error codes for portfolio verification
    - Re-export ``PortfolioValidationError`` for callers
    - Remain free of verification logic

Dependencies:
    ``cqros.portfolio.exceptions``.

Public API:
    ``PortfolioValidationError`` and the portfolio-verification error-code
    constants.
"""

from __future__ import annotations

from typing import Final

from cqros.portfolio.exceptions import PortfolioValidationError

__all__ = [
    "ERROR_REQUIRED_COLUMNS",
    "ERROR_SCHEMA_MISMATCH",
    "PortfolioValidationError",
]

ERROR_REQUIRED_COLUMNS: Final[str] = "PORTFOLIO-VERIFICATION-001"
ERROR_SCHEMA_MISMATCH: Final[str] = "PORTFOLIO-VERIFICATION-002"
