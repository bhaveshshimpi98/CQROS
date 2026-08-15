"""Unit tests for CQROS ``SignalPolicyRegistry``."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.signals import (
    ClassificationSignalPolicy,
    RegressionSignalPolicy,
    SignalPolicy,
    SignalPolicyRegistry,
    SignalValidationError,
)
from cqros.signals.registry import SignalPolicyRegistry as SignalPolicyRegistryDirect

_POLICY_CLASSIFICATION = "classification"
_POLICY_REGRESSION = "regression"
_BUY_THRESHOLD = 0.5
_SELL_THRESHOLD = -0.5


class _StubPolicy:
    """Minimal SignalPolicy-shaped stub for registry unit tests."""

    def generate(self, predictions: pl.DataFrame) -> pl.DataFrame:
        return predictions


class _IncompletePolicy:
    """Stub missing ``generate`` so protocol checks must fail."""

    def evaluate(self, predictions: pl.DataFrame) -> pl.DataFrame:
        return predictions


def _regression_policy() -> RegressionSignalPolicy:
    """Return a regression policy with shared test thresholds."""
    return RegressionSignalPolicy(
        buy_threshold=_BUY_THRESHOLD,
        sell_threshold=_SELL_THRESHOLD,
    )


def test_signal_policy_registry_is_exported_from_package() -> None:
    """Package export matches the registry module by identity."""
    assert SignalPolicyRegistry is SignalPolicyRegistryDirect


def test_empty_registry() -> None:
    """A new registry contains no policies."""
    registry = SignalPolicyRegistry()
    assert registry.list() == ()
    assert registry.exists(_POLICY_CLASSIFICATION) is False


def test_register_and_get() -> None:
    """register stores a policy that get can retrieve by name."""
    registry = SignalPolicyRegistry()
    policy = ClassificationSignalPolicy()
    registry.register(_POLICY_CLASSIFICATION, policy)
    assert registry.get(_POLICY_CLASSIFICATION) is policy


def test_register_rejects_duplicates() -> None:
    """Duplicate policy names raise SignalValidationError."""
    registry = SignalPolicyRegistry()
    first = ClassificationSignalPolicy()
    registry.register(_POLICY_CLASSIFICATION, first)
    with pytest.raises(SignalValidationError, match="already registered") as exc_info:
        registry.register(_POLICY_CLASSIFICATION, ClassificationSignalPolicy())
    assert exc_info.value.error_code == "SIGNAL_REG_DUPLICATE"
    assert registry.get(_POLICY_CLASSIFICATION) is first
    assert registry.list() == (_POLICY_CLASSIFICATION,)


def test_register_rejects_blank_names() -> None:
    """Empty or whitespace-only names raise SignalValidationError."""
    registry = SignalPolicyRegistry()
    policy = ClassificationSignalPolicy()
    for invalid_name in ("", "   ", "\t"):
        with pytest.raises(SignalValidationError, match="non-blank") as exc_info:
            registry.register(invalid_name, policy)
        assert exc_info.value.error_code == "SIGNAL_REG_NAME_BLANK"
    assert registry.list() == ()


def test_register_rejects_non_string_names() -> None:
    """Non-string names raise SignalValidationError."""
    registry = SignalPolicyRegistry()
    with pytest.raises(SignalValidationError, match="non-blank") as exc_info:
        registry.register(123, ClassificationSignalPolicy())  # type: ignore[arg-type]
    assert exc_info.value.error_code == "SIGNAL_REG_NAME_BLANK"
    assert registry.list() == ()


def test_register_rejects_invalid_policy() -> None:
    """Objects that do not implement SignalPolicy are rejected."""
    registry = SignalPolicyRegistry()
    for invalid in (None, "classification", 123, object(), _IncompletePolicy()):
        with pytest.raises(
            SignalValidationError,
            match="SignalPolicy protocol",
        ) as exc_info:
            registry.register(_POLICY_CLASSIFICATION, invalid)  # type: ignore[arg-type]
        assert exc_info.value.error_code == "SIGNAL_REG_NOT_POLICY"
    assert registry.list() == ()


def test_register_many_registers_all() -> None:
    """register_many stores every provided policy."""
    registry = SignalPolicyRegistry()
    classification = ClassificationSignalPolicy()
    regression = _regression_policy()
    stub = _StubPolicy()
    registry.register_many(
        {
            _POLICY_CLASSIFICATION: classification,
            _POLICY_REGRESSION: regression,
            "stub": stub,
        }
    )
    assert registry.get(_POLICY_CLASSIFICATION) is classification
    assert registry.get(_POLICY_REGRESSION) is regression
    assert registry.get("stub") is stub
    assert isinstance(registry.get("stub"), SignalPolicy)
    assert registry.list() == (_POLICY_CLASSIFICATION, _POLICY_REGRESSION, "stub")


def test_register_many_is_atomic_on_duplicate_existing() -> None:
    """register_many leaves the registry unchanged when a name already exists."""
    registry = SignalPolicyRegistry()
    registry.register(_POLICY_CLASSIFICATION, ClassificationSignalPolicy())
    with pytest.raises(SignalValidationError, match="already registered"):
        registry.register_many(
            {
                _POLICY_REGRESSION: _regression_policy(),
                _POLICY_CLASSIFICATION: ClassificationSignalPolicy(),
            }
        )
    assert registry.list() == (_POLICY_CLASSIFICATION,)
    assert registry.exists(_POLICY_REGRESSION) is False


def test_register_many_is_atomic_on_invalid_name() -> None:
    """register_many leaves the registry unchanged when a name is invalid."""
    registry = SignalPolicyRegistry()
    with pytest.raises(SignalValidationError, match="non-blank"):
        registry.register_many(
            {
                _POLICY_CLASSIFICATION: ClassificationSignalPolicy(),
                "": ClassificationSignalPolicy(),
            }
        )
    assert registry.list() == ()
    assert registry.exists(_POLICY_CLASSIFICATION) is False


def test_register_many_is_atomic_on_invalid_policy() -> None:
    """register_many leaves the registry unchanged when an invalid object appears."""
    registry = SignalPolicyRegistry()
    with pytest.raises(
        SignalValidationError,
        match="SignalPolicy protocol",
    ):
        registry.register_many(
            {
                _POLICY_CLASSIFICATION: ClassificationSignalPolicy(),
                "bad": object(),  # type: ignore[dict-item]
            }
        )
    assert registry.list() == ()


def test_get_unknown_raises() -> None:
    """get raises SignalValidationError for missing names."""
    registry = SignalPolicyRegistry()
    with pytest.raises(SignalValidationError, match="not registered") as exc_info:
        registry.get("missing")
    assert exc_info.value.error_code == "SIGNAL_REG_UNKNOWN"


def test_exists() -> None:
    """exists reports registration presence without raising."""
    registry = SignalPolicyRegistry()
    assert registry.exists(_POLICY_CLASSIFICATION) is False
    registry.register(_POLICY_CLASSIFICATION, ClassificationSignalPolicy())
    assert registry.exists(_POLICY_CLASSIFICATION) is True
    assert registry.exists(_POLICY_REGRESSION) is False


def test_clear() -> None:
    """clear removes all registered policies."""
    registry = SignalPolicyRegistry()
    registry.register_many(
        {
            _POLICY_CLASSIFICATION: ClassificationSignalPolicy(),
            _POLICY_REGRESSION: _regression_policy(),
        }
    )
    registry.clear()
    assert registry.list() == ()
    assert registry.exists(_POLICY_CLASSIFICATION) is False


def test_list_preserves_insertion_order() -> None:
    """list returns names in registration insertion order."""
    registry = SignalPolicyRegistry()
    registry.register("zeta", _StubPolicy())
    registry.register("alpha", ClassificationSignalPolicy())
    registry.register("mu", _regression_policy())
    assert registry.list() == ("zeta", "alpha", "mu")


def test_list_returns_immutable_snapshot() -> None:
    """list returns a new tuple unaffected by later registry mutation."""
    registry = SignalPolicyRegistry()
    registry.register_many(
        {
            _POLICY_CLASSIFICATION: ClassificationSignalPolicy(),
            _POLICY_REGRESSION: _regression_policy(),
        }
    )
    names = registry.list()
    assert isinstance(names, tuple)
    with pytest.raises(TypeError):
        names[0] = "changed"  # type: ignore[index]
    registry.clear()
    assert names == (_POLICY_CLASSIFICATION, _POLICY_REGRESSION)
    assert registry.list() == ()


def test_list_returns_independent_snapshots() -> None:
    """Repeated list calls return independent tuple objects."""
    registry = SignalPolicyRegistry()
    registry.register(_POLICY_CLASSIFICATION, ClassificationSignalPolicy())
    first = registry.list()
    second = registry.list()
    assert first == second
    assert first is not second


def test_register_does_not_mutate_policy() -> None:
    """Registry stores the policy reference without altering it."""
    registry = SignalPolicyRegistry()
    policy = ClassificationSignalPolicy()
    registry.register(_POLICY_CLASSIFICATION, policy)
    assert registry.get(_POLICY_CLASSIFICATION) is policy
    assert isinstance(policy, SignalPolicy)


def test_register_many_preserves_mapping_insertion_order() -> None:
    """register_many preserves mapping insertion order in list()."""
    registry = SignalPolicyRegistry()
    registry.register_many(
        {
            _POLICY_REGRESSION: _regression_policy(),
            _POLICY_CLASSIFICATION: ClassificationSignalPolicy(),
            "stub": _StubPolicy(),
        }
    )
    assert registry.list() == (_POLICY_REGRESSION, _POLICY_CLASSIFICATION, "stub")
