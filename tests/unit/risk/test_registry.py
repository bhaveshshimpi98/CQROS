"""Unit tests for CQROS ``RiskPolicyRegistry``."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.risk import (
    FixedRiskPolicy,
    RiskManager,
    RiskPolicyRegistry,
    RiskValidationError,
)
from cqros.risk.registry import RiskPolicyRegistry as RiskPolicyRegistryDirect


class _StubManager:
    """Minimal RiskManager-shaped stub for registry unit tests."""

    def evaluate(self, portfolios: pl.DataFrame) -> pl.DataFrame:
        return portfolios


class _IncompleteManager:
    """Stub missing ``evaluate`` so protocol checks must fail."""

    def optimize(self, portfolios: pl.DataFrame) -> pl.DataFrame:
        return portfolios


def test_risk_policy_registry_is_exported_from_package() -> None:
    """Package export matches the registry module by identity."""
    assert RiskPolicyRegistry is RiskPolicyRegistryDirect


def test_empty_registry() -> None:
    """A new registry contains no policies."""
    registry = RiskPolicyRegistry()
    assert registry.list() == ()
    assert registry.exists("fixed_risk") is False


def test_register_and_get() -> None:
    """register stores a policy that get can retrieve by name."""
    registry = RiskPolicyRegistry()
    policy = FixedRiskPolicy()
    registry.register("fixed_risk", policy)
    assert registry.get("fixed_risk") is policy


def test_register_rejects_duplicates() -> None:
    """Duplicate policy names raise RiskValidationError."""
    registry = RiskPolicyRegistry()
    first = FixedRiskPolicy()
    registry.register("fixed_risk", first)
    with pytest.raises(RiskValidationError, match="already registered") as exc_info:
        registry.register("fixed_risk", FixedRiskPolicy())
    assert exc_info.value.error_code == "RISK_REG_DUPLICATE"
    assert registry.get("fixed_risk") is first
    assert registry.list() == ("fixed_risk",)


def test_register_rejects_blank_names() -> None:
    """Empty or whitespace-only names raise RiskValidationError."""
    registry = RiskPolicyRegistry()
    policy = FixedRiskPolicy()
    for invalid_name in ("", "   ", "\t"):
        with pytest.raises(RiskValidationError, match="non-blank") as exc_info:
            registry.register(invalid_name, policy)
        assert exc_info.value.error_code == "RISK_REG_NAME_BLANK"
    assert registry.list() == ()


def test_register_rejects_non_string_names() -> None:
    """Non-string names raise RiskValidationError."""
    registry = RiskPolicyRegistry()
    with pytest.raises(RiskValidationError, match="non-blank") as exc_info:
        registry.register(123, FixedRiskPolicy())  # type: ignore[arg-type]
    assert exc_info.value.error_code == "RISK_REG_NAME_BLANK"
    assert registry.list() == ()


def test_register_rejects_invalid_policy() -> None:
    """Objects that do not implement RiskManager are rejected."""
    registry = RiskPolicyRegistry()
    for invalid in (None, "fixed_risk", 123, object(), _IncompleteManager()):
        with pytest.raises(
            RiskValidationError,
            match="RiskManager protocol",
        ) as exc_info:
            registry.register("fixed_risk", invalid)  # type: ignore[arg-type]
        assert exc_info.value.error_code == "RISK_REG_NOT_MANAGER"
    assert registry.list() == ()


def test_register_many_registers_all() -> None:
    """register_many stores every provided policy."""
    registry = RiskPolicyRegistry()
    fixed_risk = FixedRiskPolicy()
    stub = _StubManager()
    registry.register_many(
        {
            "fixed_risk": fixed_risk,
            "stub": stub,
        }
    )
    assert registry.get("fixed_risk") is fixed_risk
    assert registry.get("stub") is stub
    assert isinstance(registry.get("stub"), RiskManager)
    assert registry.list() == ("fixed_risk", "stub")


def test_register_many_is_atomic_on_duplicate_existing() -> None:
    """register_many leaves the registry unchanged when a name already exists."""
    registry = RiskPolicyRegistry()
    registry.register("fixed_risk", FixedRiskPolicy())
    with pytest.raises(RiskValidationError, match="already registered"):
        registry.register_many(
            {
                "volatility_target": _StubManager(),
                "fixed_risk": FixedRiskPolicy(),
            }
        )
    assert registry.list() == ("fixed_risk",)
    assert registry.exists("volatility_target") is False


def test_register_many_is_atomic_on_invalid_name() -> None:
    """register_many leaves the registry unchanged when a name is invalid."""
    registry = RiskPolicyRegistry()
    with pytest.raises(RiskValidationError, match="non-blank"):
        registry.register_many(
            {
                "fixed_risk": FixedRiskPolicy(),
                "": FixedRiskPolicy(),
            }
        )
    assert registry.list() == ()
    assert registry.exists("fixed_risk") is False


def test_register_many_is_atomic_on_invalid_policy() -> None:
    """register_many leaves the registry unchanged when an invalid object appears."""
    registry = RiskPolicyRegistry()
    with pytest.raises(
        RiskValidationError,
        match="RiskManager protocol",
    ):
        registry.register_many(
            {
                "fixed_risk": FixedRiskPolicy(),
                "bad": object(),  # type: ignore[dict-item]
            }
        )
    assert registry.list() == ()


def test_get_unknown_raises() -> None:
    """get raises RiskValidationError for missing names."""
    registry = RiskPolicyRegistry()
    with pytest.raises(RiskValidationError, match="not registered") as exc_info:
        registry.get("missing")
    assert exc_info.value.error_code == "RISK_REG_UNKNOWN"


def test_exists() -> None:
    """exists reports registration presence without raising."""
    registry = RiskPolicyRegistry()
    assert registry.exists("fixed_risk") is False
    registry.register("fixed_risk", FixedRiskPolicy())
    assert registry.exists("fixed_risk") is True
    assert registry.exists("volatility_target") is False


def test_clear() -> None:
    """clear removes all registered policies."""
    registry = RiskPolicyRegistry()
    registry.register_many(
        {
            "fixed_risk": FixedRiskPolicy(),
            "stub": _StubManager(),
        }
    )
    registry.clear()
    assert registry.list() == ()
    assert registry.exists("fixed_risk") is False


def test_list_preserves_insertion_order() -> None:
    """list returns names in registration insertion order."""
    registry = RiskPolicyRegistry()
    registry.register("zeta", _StubManager())
    registry.register("alpha", FixedRiskPolicy())
    registry.register("mu", _StubManager())
    assert registry.list() == ("zeta", "alpha", "mu")


def test_list_returns_immutable_snapshot() -> None:
    """list returns a new tuple unaffected by later registry mutation."""
    registry = RiskPolicyRegistry()
    registry.register_many(
        {
            "fixed_risk": FixedRiskPolicy(),
            "stub": _StubManager(),
        }
    )
    names = registry.list()
    assert isinstance(names, tuple)
    with pytest.raises(TypeError):
        names[0] = "changed"  # type: ignore[index]
    registry.clear()
    assert names == ("fixed_risk", "stub")
    assert registry.list() == ()


def test_list_returns_independent_snapshots() -> None:
    """Repeated list calls return independent tuple objects."""
    registry = RiskPolicyRegistry()
    registry.register("fixed_risk", FixedRiskPolicy())
    first = registry.list()
    second = registry.list()
    assert first == second
    assert first is not second


def test_register_does_not_mutate_policy() -> None:
    """Registry stores the policy reference without altering it."""
    registry = RiskPolicyRegistry()
    policy = FixedRiskPolicy()
    registry.register("fixed_risk", policy)
    assert registry.get("fixed_risk") is policy
    assert isinstance(policy, RiskManager)


def test_register_many_preserves_mapping_insertion_order() -> None:
    """register_many preserves mapping insertion order in list()."""
    registry = RiskPolicyRegistry()
    registry.register_many(
        {
            "risk_parity": _StubManager(),
            "fixed_risk": FixedRiskPolicy(),
            "kelly": _StubManager(),
        }
    )
    assert registry.list() == ("risk_parity", "fixed_risk", "kelly")
