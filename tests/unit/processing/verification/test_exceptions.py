"""Unit tests for CQROS processing verification exception surface."""

from __future__ import annotations

from cqros.processing.exceptions import ProcessingValidationError as RootProcessingValidationError
from cqros.processing.verification import exceptions as verification_exceptions
from cqros.processing.verification.exceptions import (
    ERROR_COUNTER_INVALID,
    ERROR_REQUIRED_COLUMNS,
    ERROR_ROWS_CHECKED_INVALID,
    ERROR_WARNINGS_INVALID,
    ProcessingValidationError,
)


def test_reuses_processing_validation_error() -> None:
    """Verification exceptions reuse the shared ProcessingValidationError."""
    assert ProcessingValidationError is RootProcessingValidationError


def test_error_code_constants_are_exported() -> None:
    """Verification-specific error codes are stable and public."""
    assert ERROR_REQUIRED_COLUMNS == "PROCESSING-VERIFICATION-001"
    assert ERROR_ROWS_CHECKED_INVALID == "PROCESSING-VERIFICATION-002"
    assert ERROR_COUNTER_INVALID == "PROCESSING-VERIFICATION-003"
    assert ERROR_WARNINGS_INVALID == "PROCESSING-VERIFICATION-004"
    for name in (
        "ERROR_REQUIRED_COLUMNS",
        "ERROR_ROWS_CHECKED_INVALID",
        "ERROR_COUNTER_INVALID",
        "ERROR_WARNINGS_INVALID",
        "ProcessingValidationError",
    ):
        assert name in verification_exceptions.__all__


def test_processing_validation_error_construction() -> None:
    """Verification callers can raise ProcessingValidationError with codes."""
    error = ProcessingValidationError(
        "invalid report",
        error_code=ERROR_COUNTER_INVALID,
        details={"parameter": "null_rows", "value": -1},
    )
    assert error.error_code == ERROR_COUNTER_INVALID
    assert dict(error.details) == {"parameter": "null_rows", "value": -1}
