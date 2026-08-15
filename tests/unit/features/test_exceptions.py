"""Unit tests for CQROS Feature Engine exception hierarchy."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from cqros.core.exceptions import CQROSError, ResearchError
from cqros.core.exceptions import FeatureError as CoreFeatureError
from cqros.features import exceptions as exceptions_module
from cqros.features.exceptions import (
    DuplicateFeatureError,
    FeatureConfigurationError,
    FeatureDependencyError,
    FeatureError,
    FeatureExecutionError,
    FeatureMetadataError,
    FeatureRegistrationError,
    FeatureStoreError,
    FeatureValidationError,
    UnknownFeatureError,
)

_SPECIALIZED_TYPES: tuple[type[FeatureError], ...] = (
    FeatureRegistrationError,
    DuplicateFeatureError,
    UnknownFeatureError,
    FeatureDependencyError,
    FeatureValidationError,
    FeatureExecutionError,
    FeatureMetadataError,
    FeatureConfigurationError,
    FeatureStoreError,
)


def test_feature_error_is_core_feature_error() -> None:
    """Feature Engine FeatureError is the shared core FeatureError type."""
    assert FeatureError is CoreFeatureError
    assert issubclass(FeatureError, ResearchError)
    assert issubclass(FeatureError, CQROSError)


def test_exception_types_are_exported() -> None:
    """Each exception type is listed in the module public API."""
    assert FeatureError.__name__ in exceptions_module.__all__
    for error_type in _SPECIALIZED_TYPES:
        assert error_type.__name__ in exceptions_module.__all__
        assert getattr(exceptions_module, error_type.__name__) is error_type


def test_feature_error_construction_stores_structured_fields() -> None:
    """FeatureError stores message, code, details, and recovery guidance."""
    error = FeatureError(
        "feature failed",
        error_code="FEATURE-001",
        details={"feature": "returns", "version": "1.0.0"},
        recovery_suggestion="Inspect feature metadata and retry.",
    )
    assert error.message == "feature failed"
    assert error.error_code == "FEATURE-001"
    assert dict(error.details) == {"feature": "returns", "version": "1.0.0"}
    assert error.recovery_suggestion == "Inspect feature metadata and retry."
    assert error.args == ("feature failed",)


def test_feature_error_defaults_details_to_empty_mapping() -> None:
    """Details default to an empty read-only mapping when omitted."""
    error = FeatureError("simple failure")
    assert error.error_code is None
    assert error.recovery_suggestion is None
    assert error.details == MappingProxyType({})
    assert isinstance(error.details, MappingProxyType)


def test_feature_error_details_are_immutable() -> None:
    """Details are copied into a read-only mapping."""
    payload = {"feature": "rsi"}
    error = FeatureError("invalid feature", details=payload)
    payload["feature"] = "ema"
    assert dict(error.details) == {"feature": "rsi"}
    with pytest.raises(TypeError):
        error.details["feature"] = "ema"  # type: ignore[index]


@pytest.mark.parametrize("error_type", _SPECIALIZED_TYPES)
def test_specialized_exceptions_inherit_from_feature_error(
    error_type: type[FeatureError],
) -> None:
    """Specialized feature exceptions remain under FeatureError."""
    error = error_type(
        "failure",
        error_code="FEATURE-X",
        details={"component": "features"},
        recovery_suggestion="Retry after correction.",
    )
    assert isinstance(error, FeatureError)
    assert isinstance(error, ResearchError)
    assert isinstance(error, CQROSError)
    assert error.message == "failure"
    assert error.error_code == "FEATURE-X"
    assert dict(error.details) == {"component": "features"}
    assert error.recovery_suggestion == "Retry after correction."


@pytest.mark.parametrize("error_type", _SPECIALIZED_TYPES)
def test_specialized_exception_details_are_immutable(
    error_type: type[FeatureError],
) -> None:
    """Specialized exceptions expose immutable details mappings."""
    payload = {"name": "returns"}
    error = error_type("failure", details=payload)
    payload["name"] = "mutated"
    assert dict(error.details) == {"name": "returns"}
    with pytest.raises(TypeError):
        error.details["name"] = "blocked"  # type: ignore[index]


def test_feature_error_str_includes_code_details_and_recovery() -> None:
    """String form includes code, details, and recovery guidance."""
    error = FeatureValidationError(
        "schema mismatch",
        error_code="FEATURE-VAL-001",
        details={"feature": "ema", "missing": "close"},
        recovery_suggestion="Provide required columns.",
    )
    text = str(error)
    assert text.startswith("[FEATURE-VAL-001] schema mismatch")
    assert "feature='ema'" in text
    assert "missing='close'" in text
    assert "recovery: Provide required columns." in text


def test_feature_error_str_without_optional_fields() -> None:
    """String form is the message alone when optional fields are absent."""
    assert str(FeatureExecutionError("boom")) == "boom"


@pytest.mark.parametrize("error_type", (FeatureError, *_SPECIALIZED_TYPES))
def test_feature_exception_repr_is_unambiguous(error_type: type[FeatureError]) -> None:
    """Repr includes type name and structured fields."""
    error = error_type("boom", error_code="X", details={"a": 1})
    assert repr(error) == (
        f"{error_type.__name__}(message='boom', error_code='X', "
        "details={'a': 1}, recovery_suggestion=None)"
    )


def test_nested_hierarchy_relationships() -> None:
    """Specialized exceptions preserve FeatureError lineage."""
    assert issubclass(FeatureRegistrationError, FeatureError)
    assert issubclass(DuplicateFeatureError, FeatureError)
    assert issubclass(UnknownFeatureError, FeatureError)
    assert issubclass(FeatureDependencyError, FeatureError)
    assert issubclass(FeatureValidationError, FeatureError)
    assert issubclass(FeatureExecutionError, FeatureError)
    assert issubclass(FeatureMetadataError, FeatureError)
    assert issubclass(FeatureConfigurationError, FeatureError)
    assert issubclass(FeatureStoreError, FeatureError)


def test_catching_feature_error_captures_subclass() -> None:
    """Callers can catch by FeatureError without knowing the subclass."""
    with pytest.raises(FeatureError) as exc_info:
        raise UnknownFeatureError(
            "feature not registered",
            error_code="FEATURE-UNK-001",
            details={"name": "macd"},
        )
    assert exc_info.value.error_code == "FEATURE-UNK-001"
    assert dict(exc_info.value.details) == {"name": "macd"}


def test_exception_equality_uses_identity() -> None:
    """Exceptions compare by identity, matching the CQROS base behavior."""
    left = DuplicateFeatureError("duplicate", error_code="FEATURE-DUP-001")
    right = DuplicateFeatureError("duplicate", error_code="FEATURE-DUP-001")
    assert left == left
    assert left is not right
    assert left != right


def test_package_exports_exception_types() -> None:
    """The features package re-exports the exception hierarchy."""
    import cqros.features as features_package

    for name in (
        "FeatureError",
        "FeatureRegistrationError",
        "DuplicateFeatureError",
        "UnknownFeatureError",
        "FeatureDependencyError",
        "FeatureValidationError",
        "FeatureExecutionError",
        "FeatureMetadataError",
        "FeatureConfigurationError",
        "FeatureStoreError",
    ):
        assert name in features_package.__all__
        assert getattr(features_package, name).__name__ == name
