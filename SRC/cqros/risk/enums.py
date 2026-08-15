"""CQROS Risk Management enumerations.

Purpose:
    Define the canonical risk-decision and risk-policy vocabulary used
    throughout CQROS.

Responsibilities:
    - Enumerate every supported risk decision as a string-backed
      enumeration
    - Enumerate every reserved risk policy as a string-backed enumeration
    - Expose helper accessors that return immutable member and value tuples
    - Remain free of policy logic, calculation, validation, and persistence

Dependencies:
    Python standard library only (``enum.Enum``).

Public API:
    ``RiskDecision``, ``RiskPolicy``, ``decisions``, ``policies``,
    ``decision_values``, ``policy_values``

Notes:
    Both enumerations subclass ``str`` and ``Enum`` so members serialize
    naturally into Polars DataFrames without conversion. Enum members
    reserve the public API; no policy or calculation logic lives here.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "RiskDecision",
    "RiskPolicy",
    "decision_values",
    "decisions",
    "policies",
    "policy_values",
]


# Prefer str+Enum over StrEnum so enum values embed directly into Polars.
class RiskDecision(str, Enum):  # noqa: UP042
    """Canonical risk-management approval decision.

    Attributes:
        APPROVE: Accept the proposed target weight unchanged.
        RESIZE: Accept a risk-adjusted approved weight.
        REJECT: Reject the proposed allocation.
    """

    APPROVE = "APPROVE"
    RESIZE = "RESIZE"
    REJECT = "REJECT"


class RiskPolicy(str, Enum):  # noqa: UP042
    """Canonical risk-policy strategy identifiers.

    Attributes:
        FIXED_RISK: Reserved for fixed-risk sizing.
        VOLATILITY_TARGET: Reserved for volatility-target sizing.
        RISK_PARITY: Reserved for risk-parity sizing.
        KELLY: Reserved for Kelly-criterion sizing.
        REGIME_AWARE: Reserved for regime-aware risk control.
    """

    FIXED_RISK = "fixed_risk"
    VOLATILITY_TARGET = "volatility_target"
    RISK_PARITY = "risk_parity"
    KELLY = "kelly"
    REGIME_AWARE = "regime_aware"


def decisions() -> tuple[RiskDecision, ...]:
    """Return an immutable copy of every ``RiskDecision`` member.

    Returns:
        All risk-decision members in declaration order.
    """
    return (
        RiskDecision.APPROVE,
        RiskDecision.RESIZE,
        RiskDecision.REJECT,
    )


def policies() -> tuple[RiskPolicy, ...]:
    """Return an immutable copy of every ``RiskPolicy`` member.

    Returns:
        All risk-policy members in declaration order.
    """
    return (
        RiskPolicy.FIXED_RISK,
        RiskPolicy.VOLATILITY_TARGET,
        RiskPolicy.RISK_PARITY,
        RiskPolicy.KELLY,
        RiskPolicy.REGIME_AWARE,
    )


def decision_values() -> tuple[str, ...]:
    """Return an immutable copy of every ``RiskDecision`` string value.

    Returns:
        All risk-decision values in declaration order.
    """
    return (
        RiskDecision.APPROVE.value,
        RiskDecision.RESIZE.value,
        RiskDecision.REJECT.value,
    )


def policy_values() -> tuple[str, ...]:
    """Return an immutable copy of every ``RiskPolicy`` string value.

    Returns:
        All risk-policy values in declaration order.
    """
    return (
        RiskPolicy.FIXED_RISK.value,
        RiskPolicy.VOLATILITY_TARGET.value,
        RiskPolicy.RISK_PARITY.value,
        RiskPolicy.KELLY.value,
        RiskPolicy.REGIME_AWARE.value,
    )
