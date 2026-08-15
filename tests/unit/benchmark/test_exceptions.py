"""Unit tests for CQROS Benchmark Engine exception hierarchy."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from cqros.benchmark import exceptions as exceptions_module
from cqros.benchmark.exceptions import (
    BENCHMARK_BUILD_FAILED,
    BENCHMARK_CONFIGURATION_ERROR,
    BENCHMARK_DUPLICATE_PROVIDER,
    BENCHMARK_INVALID_INPUT,
    BENCHMARK_INVALID_SCHEMA,
    BENCHMARK_INVALID_TYPE,
    BENCHMARK_NOT_FOUND,
    BENCHMARK_STORAGE_ERROR,
    BENCHMARK_UNKNOWN_PROVIDER,
    BenchmarkError,
    BenchmarkException,
)
from cqros.core.exceptions import CQROSError, ResearchError

_ERROR_CODE_CONSTANTS: tuple[tuple[str, str], ...] = (
    ("BENCHMARK_INVALID_INPUT", BENCHMARK_INVALID_INPUT),
    ("BENCHMARK_INVALID_SCHEMA", BENCHMARK_INVALID_SCHEMA),
    ("BENCHMARK_UNKNOWN_PROVIDER", BENCHMARK_UNKNOWN_PROVIDER),
    ("BENCHMARK_DUPLICATE_PROVIDER", BENCHMARK_DUPLICATE_PROVIDER),
    ("BENCHMARK_NOT_FOUND", BENCHMARK_NOT_FOUND),
    ("BENCHMARK_STORAGE_ERROR", BENCHMARK_STORAGE_ERROR),
    ("BENCHMARK_CONFIGURATION_ERROR", BENCHMARK_CONFIGURATION_ERROR),
    ("BENCHMARK_INVALID_TYPE", BENCHMARK_INVALID_TYPE),
    ("BENCHMARK_BUILD_FAILED", BENCHMARK_BUILD_FAILED),
)

_EXCEPTION_TYPES: tuple[type[BenchmarkException], ...] = (
    BenchmarkException,
    BenchmarkError,
)


def test_benchmark_exception_inherits_from_research_error() -> None:
    """BenchmarkException is a ResearchError specialization."""
    assert issubclass(BenchmarkException, ResearchError)
    assert issubclass(BenchmarkException, CQROSError)


def test_benchmark_error_inherits_from_benchmark_exception() -> None:
    """BenchmarkError remains under BenchmarkException."""
    assert issubclass(BenchmarkError, BenchmarkException)
    assert issubclass(BenchmarkError, ResearchError)
    assert issubclass(BenchmarkError, CQROSError)


def test_exception_types_are_exported() -> None:
    """Each exception type is listed in the module public API."""
    for error_type in _EXCEPTION_TYPES:
        assert error_type.__name__ in exceptions_module.__all__
        assert getattr(exceptions_module, error_type.__name__) is error_type


def test_error_code_constants_are_exported() -> None:
    """Canonical benchmark error codes are stable and public."""
    for name, value in _ERROR_CODE_CONSTANTS:
        assert name in exceptions_module.__all__
        assert getattr(exceptions_module, name) == value
        assert value == name


def test_error_code_constants_are_unique() -> None:
    """Canonical benchmark error codes contain no duplicates."""
    values = tuple(value for _, value in _ERROR_CODE_CONSTANTS)
    assert len(values) == len(set(values))
    assert len(values) == 9


def test_benchmark_exception_construction_stores_structured_fields() -> None:
    """BenchmarkException stores message, code, details, and recovery guidance."""
    error = BenchmarkException(
        "benchmark failed",
        error_code=BENCHMARK_BUILD_FAILED,
        details={"benchmark": "btc", "version": "1.0.0"},
        recovery_suggestion="Inspect provider configuration and retry.",
    )
    assert error.message == "benchmark failed"
    assert error.error_code == BENCHMARK_BUILD_FAILED
    assert dict(error.details) == {"benchmark": "btc", "version": "1.0.0"}
    assert error.recovery_suggestion == "Inspect provider configuration and retry."
    assert error.args == ("benchmark failed",)


def test_benchmark_error_construction_stores_structured_fields() -> None:
    """BenchmarkError stores message, code, details, and recovery guidance."""
    error = BenchmarkError(
        "invalid schema",
        error_code=BENCHMARK_INVALID_SCHEMA,
        details={"missing": "benchmark_value"},
        recovery_suggestion="Provide required columns.",
    )
    assert error.message == "invalid schema"
    assert error.error_code == BENCHMARK_INVALID_SCHEMA
    assert dict(error.details) == {"missing": "benchmark_value"}
    assert error.recovery_suggestion == "Provide required columns."
    assert error.args == ("invalid schema",)


def test_benchmark_exception_defaults_details_to_empty_mapping() -> None:
    """Details default to an empty read-only mapping when omitted."""
    error = BenchmarkException("simple failure")
    assert error.error_code is None
    assert error.recovery_suggestion is None
    assert error.details == MappingProxyType({})
    assert isinstance(error.details, MappingProxyType)


def test_benchmark_error_defaults_details_to_empty_mapping() -> None:
    """BenchmarkError defaults optional structured fields when omitted."""
    error = BenchmarkError("simple failure")
    assert error.error_code is None
    assert error.recovery_suggestion is None
    assert error.details == MappingProxyType({})
    assert isinstance(error.details, MappingProxyType)


def test_benchmark_exception_details_are_immutable() -> None:
    """Details are copied into a read-only mapping."""
    payload = {"provider": "btc"}
    error = BenchmarkException("invalid benchmark", details=payload)
    payload["provider"] = "eth"
    assert dict(error.details) == {"provider": "btc"}
    with pytest.raises(TypeError):
        error.details["provider"] = "eth"  # type: ignore[index]


@pytest.mark.parametrize("error_type", _EXCEPTION_TYPES)
def test_exception_types_preserve_message_and_code(
    error_type: type[BenchmarkException],
) -> None:
    """Benchmark exceptions preserve message and error_code."""
    error = error_type(
        "failure",
        error_code=BENCHMARK_INVALID_INPUT,
        details={"component": "benchmark"},
        recovery_suggestion="Retry after correction.",
    )
    assert isinstance(error, BenchmarkException)
    assert isinstance(error, ResearchError)
    assert isinstance(error, CQROSError)
    assert error.message == "failure"
    assert error.error_code == BENCHMARK_INVALID_INPUT
    assert dict(error.details) == {"component": "benchmark"}
    assert error.recovery_suggestion == "Retry after correction."


def test_benchmark_error_str_includes_code_details_and_recovery() -> None:
    """String form includes code, details, and recovery guidance."""
    error = BenchmarkError(
        "unknown provider",
        error_code=BENCHMARK_UNKNOWN_PROVIDER,
        details={"provider": "custom_index"},
        recovery_suggestion="Register the provider before use.",
    )
    text = str(error)
    assert text.startswith("[BENCHMARK_UNKNOWN_PROVIDER] unknown provider")
    assert "provider='custom_index'" in text
    assert "recovery: Register the provider before use." in text


def test_benchmark_error_str_without_optional_fields() -> None:
    """String form is the message alone when optional fields are absent."""
    assert str(BenchmarkError("boom")) == "boom"


@pytest.mark.parametrize("error_type", _EXCEPTION_TYPES)
def test_benchmark_exception_repr_is_unambiguous(
    error_type: type[BenchmarkException],
) -> None:
    """Repr includes type name and structured fields."""
    error = error_type("boom", error_code="X", details={"a": 1})
    assert repr(error) == (
        f"{error_type.__name__}(message='boom', error_code='X', "
        "details={'a': 1}, recovery_suggestion=None)"
    )


def test_catching_benchmark_exception_captures_subclass() -> None:
    """Callers can catch by BenchmarkException without knowing the subclass."""
    with pytest.raises(BenchmarkException) as exc_info:
        raise BenchmarkError(
            "provider not registered",
            error_code=BENCHMARK_UNKNOWN_PROVIDER,
            details={"name": "equal_weight"},
        )
    assert exc_info.value.error_code == BENCHMARK_UNKNOWN_PROVIDER
    assert dict(exc_info.value.details) == {"name": "equal_weight"}


def test_exception_equality_uses_identity() -> None:
    """Exceptions compare by identity, matching the CQROS base behavior."""
    left = BenchmarkError("duplicate", error_code=BENCHMARK_DUPLICATE_PROVIDER)
    right = BenchmarkError("duplicate", error_code=BENCHMARK_DUPLICATE_PROVIDER)
    assert left == left
    assert left is not right
    assert left != right


def test_canonical_error_codes_can_be_attached() -> None:
    """Each canonical error code can be attached to BenchmarkError."""
    for _, code in _ERROR_CODE_CONSTANTS:
        error = BenchmarkError("failure", error_code=code)
        assert error.error_code == code
        assert str(error).startswith(f"[{code}] failure")


def test_package_exports_exception_types() -> None:
    """The benchmark package re-exports the exception hierarchy."""
    import cqros.benchmark as benchmark_package

    for name in ("BenchmarkException", "BenchmarkError"):
        assert name in benchmark_package.__all__
        assert getattr(benchmark_package, name).__name__ == name
