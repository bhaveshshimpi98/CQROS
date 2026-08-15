"""Unit tests for the CQROS exception hierarchy."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from cqros.core.exceptions import (
    ConfigurationError,
    ConfigValidationError,
    CQROSError,
    DataError,
    ExchangeError,
    ExchangeRateLimitError,
    ExecutionError,
    InfrastructureError,
    LeakageError,
    ModelError,
    ResearchError,
    RiskError,
    TimeoutError,
    TrainingError,
    ValidationError,
)


def test_cqros_error_stores_message_code_and_details() -> None:
    """CQROSError stores message, optional code, and details."""
    error = CQROSError(
        "operation failed",
        error_code="CORE-001",
        details={"component": "core", "retryable": True},
    )

    assert error.message == "operation failed"
    assert error.error_code == "CORE-001"
    assert dict(error.details) == {"component": "core", "retryable": True}
    assert error.recovery_suggestion is None
    assert error.args == ("operation failed",)


def test_cqros_error_defaults_details_to_empty_mapping() -> None:
    """Details default to an empty read-only mapping when omitted."""
    error = CQROSError("simple failure")

    assert error.error_code is None
    assert error.details == MappingProxyType({})
    assert isinstance(error.details, MappingProxyType)


def test_cqros_error_details_are_immutable() -> None:
    """Details are copied into a read-only mapping."""
    payload = {"symbol": "BTCUSDT"}
    error = CQROSError("missing bar", details=payload)
    payload["symbol"] = "ETHUSDT"

    assert dict(error.details) == {"symbol": "BTCUSDT"}
    with pytest.raises(TypeError):
        error.details["symbol"] = "ETHUSDT"  # type: ignore[index]


def test_cqros_error_str_includes_code_details_and_recovery() -> None:
    """String form includes code, details, and recovery guidance."""
    error = CQROSError(
        "rate limited",
        error_code="EX-429",
        details={"exchange": "binance", "retry_after_ms": 1000},
        recovery_suggestion="Retry after the reported delay.",
    )

    text = str(error)

    assert text.startswith("[EX-429] rate limited")
    assert "exchange='binance'" in text
    assert "retry_after_ms=1000" in text
    assert "recovery: Retry after the reported delay." in text


def test_cqros_error_str_without_optional_fields() -> None:
    """String form is the message alone when optional fields are absent."""
    assert str(CQROSError("boom")) == "boom"


def test_cqros_error_repr_is_unambiguous() -> None:
    """Repr includes type name and structured fields."""
    error = CQROSError("boom", error_code="X", details={"a": 1})

    assert repr(error) == (
        "CQROSError(message='boom', error_code='X', details={'a': 1}, " "recovery_suggestion=None)"
    )


def test_domain_exceptions_inherit_from_cqros_error() -> None:
    """Domain category exceptions all inherit from CQROSError."""
    domain_types = (
        InfrastructureError,
        ConfigurationError,
        ExchangeError,
        DataError,
        ResearchError,
        ModelError,
        RiskError,
        ExecutionError,
        ValidationError,
    )

    for domain_type in domain_types:
        error = domain_type("domain failure", error_code="D-1", details={"k": "v"})
        assert isinstance(error, CQROSError)
        assert error.message == "domain failure"
        assert error.error_code == "D-1"
        assert dict(error.details) == {"k": "v"}


def test_specialized_exceptions_remain_lightweight() -> None:
    """Specialized subclasses reuse the base constructor unchanged."""
    error = ExchangeRateLimitError(
        "too many requests",
        error_code="EX-RATE",
        details={"limit": 1200},
        recovery_suggestion="Back off and retry.",
    )

    assert isinstance(error, ExchangeError)
    assert isinstance(error, CQROSError)
    assert str(error).startswith("[EX-RATE] too many requests")


def test_nested_hierarchy_relationships() -> None:
    """Nested exceptions preserve their domain lineage."""
    assert issubclass(TimeoutError, InfrastructureError)
    assert issubclass(ConfigValidationError, ConfigurationError)
    assert issubclass(LeakageError, ResearchError)
    assert issubclass(TrainingError, ModelError)
    assert issubclass(ExchangeRateLimitError, ExchangeError)


def test_catching_base_type_captures_subclass() -> None:
    """Callers can catch by domain base without knowing the subclass."""
    with pytest.raises(ResearchError) as exc_info:
        raise LeakageError(
            "future timestamp detected",
            error_code="RES-LEAK",
            details={"column": "target_1h"},
        )

    assert exc_info.value.error_code == "RES-LEAK"
    assert dict(exc_info.value.details) == {"column": "target_1h"}
