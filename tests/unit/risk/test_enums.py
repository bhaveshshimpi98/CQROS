"""Unit tests for the CQROS Risk Management enumerations."""

from __future__ import annotations

from enum import Enum

import pytest

from cqros.risk import (
    RiskDecision,
    RiskPolicy,
    decision_values,
    decisions,
    policies,
    policy_values,
)
from cqros.risk.enums import RiskDecision as RiskDecisionDirect
from cqros.risk.enums import RiskPolicy as RiskPolicyDirect
from cqros.risk.enums import decision_values as decision_values_direct
from cqros.risk.enums import decisions as decisions_direct
from cqros.risk.enums import policies as policies_direct
from cqros.risk.enums import policy_values as policy_values_direct


def test_enums_are_exported_from_package() -> None:
    """Package exports match the enums module by identity."""
    assert RiskDecision is RiskDecisionDirect
    assert RiskPolicy is RiskPolicyDirect
    assert decisions is decisions_direct
    assert policies is policies_direct
    assert decision_values is decision_values_direct
    assert policy_values is policy_values_direct


def test_risk_decision_approve_member() -> None:
    """APPROVE member name and value are stable."""
    assert RiskDecision.APPROVE.name == "APPROVE"
    assert RiskDecision.APPROVE.value == "APPROVE"
    assert RiskDecision.APPROVE == "APPROVE"


def test_risk_decision_resize_member() -> None:
    """RESIZE member name and value are stable."""
    assert RiskDecision.RESIZE.name == "RESIZE"
    assert RiskDecision.RESIZE.value == "RESIZE"
    assert RiskDecision.RESIZE == "RESIZE"


def test_risk_decision_reject_member() -> None:
    """REJECT member name and value are stable."""
    assert RiskDecision.REJECT.name == "REJECT"
    assert RiskDecision.REJECT.value == "REJECT"
    assert RiskDecision.REJECT == "REJECT"


def test_risk_decision_enum_names() -> None:
    """RiskDecision names remain the canonical uppercase identifiers."""
    assert tuple(member.name for member in RiskDecision) == (
        "APPROVE",
        "RESIZE",
        "REJECT",
    )


def test_risk_decision_enum_values() -> None:
    """RiskDecision values remain the canonical uppercase strings."""
    assert tuple(member.value for member in RiskDecision) == (
        "APPROVE",
        "RESIZE",
        "REJECT",
    )


def test_risk_policy_members() -> None:
    """RiskPolicy member names and values are stable."""
    assert RiskPolicy.FIXED_RISK.name == "FIXED_RISK"
    assert RiskPolicy.FIXED_RISK.value == "fixed_risk"
    assert RiskPolicy.VOLATILITY_TARGET.name == "VOLATILITY_TARGET"
    assert RiskPolicy.VOLATILITY_TARGET.value == "volatility_target"
    assert RiskPolicy.RISK_PARITY.name == "RISK_PARITY"
    assert RiskPolicy.RISK_PARITY.value == "risk_parity"
    assert RiskPolicy.KELLY.name == "KELLY"
    assert RiskPolicy.KELLY.value == "kelly"
    assert RiskPolicy.REGIME_AWARE.name == "REGIME_AWARE"
    assert RiskPolicy.REGIME_AWARE.value == "regime_aware"


def test_risk_policy_enum_names() -> None:
    """RiskPolicy names remain the canonical uppercase identifiers."""
    assert tuple(member.name for member in RiskPolicy) == (
        "FIXED_RISK",
        "VOLATILITY_TARGET",
        "RISK_PARITY",
        "KELLY",
        "REGIME_AWARE",
    )


def test_risk_policy_enum_values() -> None:
    """RiskPolicy values remain the reserved snake_case strings."""
    assert tuple(member.value for member in RiskPolicy) == (
        "fixed_risk",
        "volatility_target",
        "risk_parity",
        "kelly",
        "regime_aware",
    )


def test_enums_subclass_str_and_enum() -> None:
    """Both enumerations subclass str and Enum for natural serialization."""
    for enum_cls in (RiskDecision, RiskPolicy):
        assert issubclass(enum_cls, str)
        assert issubclass(enum_cls, Enum)
        for member in enum_cls:
            assert isinstance(member, str)
            assert isinstance(member, enum_cls)
            assert member == member.value


def test_decisions_helper_output() -> None:
    """decisions returns every member in declaration order."""
    assert decisions() == (
        RiskDecision.APPROVE,
        RiskDecision.RESIZE,
        RiskDecision.REJECT,
    )


def test_policies_helper_output() -> None:
    """policies returns every member in declaration order."""
    assert policies() == (
        RiskPolicy.FIXED_RISK,
        RiskPolicy.VOLATILITY_TARGET,
        RiskPolicy.RISK_PARITY,
        RiskPolicy.KELLY,
        RiskPolicy.REGIME_AWARE,
    )


def test_decision_values_helper_output() -> None:
    """decision_values returns every string value in order."""
    assert decision_values() == ("APPROVE", "RESIZE", "REJECT")


def test_policy_values_helper_output() -> None:
    """policy_values returns every string value in order."""
    assert policy_values() == (
        "fixed_risk",
        "volatility_target",
        "risk_parity",
        "kelly",
        "regime_aware",
    )


def test_helper_outputs_are_immutable_tuples() -> None:
    """Helpers return immutable tuples."""
    decision_members = decisions()
    policy_members = policies()
    decision_string_values = decision_values()
    policy_string_values = policy_values()

    assert isinstance(decision_members, tuple)
    assert isinstance(policy_members, tuple)
    assert isinstance(decision_string_values, tuple)
    assert isinstance(policy_string_values, tuple)

    with pytest.raises(TypeError):
        decision_members[0] = RiskDecision.REJECT  # type: ignore[index]

    with pytest.raises(TypeError):
        policy_members[0] = RiskPolicy.KELLY  # type: ignore[index]

    with pytest.raises(TypeError):
        decision_string_values[0] = "REJECT"  # type: ignore[index]

    with pytest.raises(TypeError):
        policy_string_values[0] = "kelly"  # type: ignore[index]


def test_helper_independence() -> None:
    """Helpers return independent copies, not shared mutable state."""
    first_decisions = decisions()
    second_decisions = decisions()
    first_policies = policies()
    second_policies = policies()
    first_decision_values = decision_values()
    second_decision_values = decision_values()
    first_policy_values = policy_values()
    second_policy_values = policy_values()

    assert first_decisions == second_decisions
    assert first_decisions is not second_decisions
    assert first_policies == second_policies
    assert first_policies is not second_policies
    assert first_decision_values == second_decision_values
    assert first_decision_values is not second_decision_values
    assert first_policy_values == second_policy_values
    assert first_policy_values is not second_policy_values


def test_enum_members_and_values_are_unique() -> None:
    """Enum names and values contain no duplicates."""
    for enum_cls, members_helper, values_helper in (
        (RiskDecision, decisions, decision_values),
        (RiskPolicy, policies, policy_values),
    ):
        names = tuple(member.name for member in enum_cls)
        member_values = tuple(member.value for member in enum_cls)

        assert len(names) == len(set(names))
        assert len(member_values) == len(set(member_values))
        assert len(members_helper()) == len(set(members_helper()))
        assert len(values_helper()) == len(set(values_helper()))


def test_enum_round_trips_from_value() -> None:
    """Enum members can be reconstructed from their string values."""
    for enum_cls in (RiskDecision, RiskPolicy):
        for member in enum_cls:
            assert enum_cls(member.value) is member


def test_invalid_value_raises_value_error() -> None:
    """Unknown serialized values raise ValueError."""
    with pytest.raises(ValueError):
        RiskDecision("not_a_valid_decision")

    with pytest.raises(ValueError):
        RiskPolicy("not_a_valid_policy")
