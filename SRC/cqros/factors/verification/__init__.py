"""CQROS factor verification package public API.

Purpose:
    Expose the factor-dataset verifier and its report/exception surface.

Responsibilities:
    - Re-export ``FactorVerifier``, report/diagnostic types, and schema
      mismatch exception symbols
    - Remain free of unverified verifier implementations

Public API:
    ``FactorVerifier``, ``FactorVerificationReport``,
    ``FactorVerificationDiagnostics``, diagnostic value objects,
    ``VerificationReport``, ``FactorValidationError``,
    ``ERROR_REQUIRED_COLUMNS``, ``ERROR_SCHEMA_MISMATCH``,
    ``format_factor_diagnostics``, ``format_global_failure_report``,
    ``collect_global_failure_findings``, ``GlobalFailureFinding``
"""

from cqros.factors.verification.diagnostics import (
    FactorInvalidNumericDiagnostic,
    FactorNullDiagnostic,
    FactorVerificationDiagnostics,
    FactorVerificationReport,
    FactorWarningDiagnostic,
    GlobalFailureFinding,
    InvalidNumericKind,
    NullClassification,
    collect_global_failure_findings,
    format_factor_diagnostics,
    format_global_failure_report,
)
from cqros.factors.verification.exceptions import (
    ERROR_REQUIRED_COLUMNS,
    ERROR_SCHEMA_MISMATCH,
    FactorValidationError,
)
from cqros.factors.verification.verifier import FactorVerifier
from cqros.processing.verification.report import VerificationReport

__all__ = [
    "ERROR_REQUIRED_COLUMNS",
    "ERROR_SCHEMA_MISMATCH",
    "FactorInvalidNumericDiagnostic",
    "FactorNullDiagnostic",
    "FactorValidationError",
    "FactorVerificationDiagnostics",
    "FactorVerificationReport",
    "FactorVerifier",
    "FactorWarningDiagnostic",
    "GlobalFailureFinding",
    "InvalidNumericKind",
    "NullClassification",
    "VerificationReport",
    "collect_global_failure_findings",
    "format_factor_diagnostics",
    "format_global_failure_report",
]
