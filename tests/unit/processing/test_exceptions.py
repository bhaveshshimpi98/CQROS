"""Unit tests for CQROS Data Processing Framework exception hierarchy."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from cqros.core.exceptions import CQROSError, DataError
from cqros.processing import exceptions as exceptions_module
from cqros.processing.exceptions import (
    DuplicateProcessingStepError,
    ProcessingError,
    ProcessingExecutionError,
    ProcessingRegistrationError,
    ProcessingValidationError,
    UnknownProcessingStepError,
)

_SPECIALIZED_TYPES: tuple[type[ProcessingError], ...] = (
    ProcessingValidationError,
    ProcessingRegistrationError,
    UnknownProcessingStepError,
    DuplicateProcessingStepError,
    ProcessingExecutionError,
)


def test_processing_error_inherits_from_data_error() -> None:
    """ProcessingError is a DataError specialization."""
    assert issubclass(ProcessingError, DataError)
    assert issubclass(ProcessingError, CQROSError)


def test_exception_types_are_exported() -> None:
    """Each exception type is listed in the module public API."""
    assert ProcessingError.__name__ in exceptions_module.__all__
    for error_type in _SPECIALIZED_TYPES:
        assert error_type.__name__ in exceptions_module.__all__
        assert getattr(exceptions_module, error_type.__name__) is error_type


def test_processing_error_construction_stores_structured_fields() -> None:
    """ProcessingError stores message, code, details, and recovery guidance."""
    error = ProcessingError(
        "processing failed",
        error_code="PROCESSING-001",
        details={"step": "dedupe", "version": "1.0.0"},
        recovery_suggestion="Inspect step metadata and retry.",
    )
    assert error.message == "processing failed"
    assert error.error_code == "PROCESSING-001"
    assert dict(error.details) == {"step": "dedupe", "version": "1.0.0"}
    assert error.recovery_suggestion == "Inspect step metadata and retry."
    assert error.args == ("processing failed",)


def test_processing_error_defaults_details_to_empty_mapping() -> None:
    """Details default to an empty read-only mapping when omitted."""
    error = ProcessingError("simple failure")
    assert error.error_code is None
    assert error.recovery_suggestion is None
    assert error.details == MappingProxyType({})
    assert isinstance(error.details, MappingProxyType)


def test_processing_error_details_are_immutable() -> None:
    """Details are copied into a read-only mapping."""
    payload = {"step": "sort"}
    error = ProcessingError("invalid processing", details=payload)
    payload["step"] = "filter"
    assert dict(error.details) == {"step": "sort"}
    with pytest.raises(TypeError):
        error.details["step"] = "filter"  # type: ignore[index]


@pytest.mark.parametrize("error_type", _SPECIALIZED_TYPES)
def test_specialized_exceptions_inherit_from_processing_error(
    error_type: type[ProcessingError],
) -> None:
    """Specialized processing exceptions remain under ProcessingError."""
    error = error_type(
        "failure",
        error_code="PROCESSING-X",
        details={"component": "processing"},
        recovery_suggestion="Retry after correction.",
    )
    assert isinstance(error, ProcessingError)
    assert isinstance(error, DataError)
    assert isinstance(error, CQROSError)
    assert error.message == "failure"
    assert error.error_code == "PROCESSING-X"
    assert dict(error.details) == {"component": "processing"}
    assert error.recovery_suggestion == "Retry after correction."


@pytest.mark.parametrize("error_type", _SPECIALIZED_TYPES)
def test_specialized_exception_details_are_immutable(
    error_type: type[ProcessingError],
) -> None:
    """Specialized exceptions expose immutable details mappings."""
    payload = {"name": "dedupe"}
    error = error_type("failure", details=payload)
    payload["name"] = "mutated"
    assert dict(error.details) == {"name": "dedupe"}
    with pytest.raises(TypeError):
        error.details["name"] = "blocked"  # type: ignore[index]


def test_processing_error_str_includes_code_details_and_recovery() -> None:
    """String form includes code, details, and recovery guidance."""
    error = ProcessingValidationError(
        "schema mismatch",
        error_code="PROCESSING-VAL-001",
        details={"step": "align", "missing": "timestamp"},
        recovery_suggestion="Provide required columns.",
    )
    text = str(error)
    assert text.startswith("[PROCESSING-VAL-001] schema mismatch")
    assert "step='align'" in text
    assert "missing='timestamp'" in text
    assert "recovery: Provide required columns." in text


def test_processing_error_str_without_optional_fields() -> None:
    """String form is the message alone when optional fields are absent."""
    assert str(ProcessingExecutionError("boom")) == "boom"


@pytest.mark.parametrize("error_type", (ProcessingError, *_SPECIALIZED_TYPES))
def test_processing_exception_repr_is_unambiguous(
    error_type: type[ProcessingError],
) -> None:
    """Repr includes type name and structured fields."""
    error = error_type("boom", error_code="X", details={"a": 1})
    assert repr(error) == (
        f"{error_type.__name__}(message='boom', error_code='X', "
        "details={'a': 1}, recovery_suggestion=None)"
    )


def test_nested_hierarchy_relationships() -> None:
    """Specialized exceptions preserve ProcessingError lineage."""
    assert issubclass(ProcessingValidationError, ProcessingError)
    assert issubclass(ProcessingRegistrationError, ProcessingError)
    assert issubclass(UnknownProcessingStepError, ProcessingError)
    assert issubclass(DuplicateProcessingStepError, ProcessingError)
    assert issubclass(ProcessingExecutionError, ProcessingError)


def test_catching_processing_error_captures_subclass() -> None:
    """Callers can catch by ProcessingError without knowing the subclass."""
    with pytest.raises(ProcessingError) as exc_info:
        raise UnknownProcessingStepError(
            "processing step not registered",
            error_code="PROCESSING-UNK-001",
            details={"name": "dedupe"},
        )
    assert exc_info.value.error_code == "PROCESSING-UNK-001"
    assert dict(exc_info.value.details) == {"name": "dedupe"}


def test_exception_equality_uses_identity() -> None:
    """Exceptions compare by identity, matching the CQROS base behavior."""
    left = DuplicateProcessingStepError("duplicate", error_code="PROCESSING-DUP-001")
    right = DuplicateProcessingStepError("duplicate", error_code="PROCESSING-DUP-001")
    assert left == left
    assert left is not right
    assert left != right


def test_package_exports_exception_types() -> None:
    """The processing package re-exports the exception hierarchy."""
    import cqros.processing as processing_package

    for name in (
        "ProcessingError",
        "ProcessingValidationError",
        "ProcessingRegistrationError",
        "UnknownProcessingStepError",
        "DuplicateProcessingStepError",
        "ProcessingExecutionError",
    ):
        assert name in processing_package.__all__
        assert getattr(processing_package, name).__name__ == name
