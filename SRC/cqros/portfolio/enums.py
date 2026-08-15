"""CQROS Portfolio enumerations.

Purpose:
    Define the canonical portfolio-direction and optimizer-strategy
    vocabulary used throughout CQROS.

Responsibilities:
    - Enumerate every supported portfolio direction as a string-backed
      enumeration
    - Enumerate every reserved optimizer strategy as a string-backed
      enumeration
    - Expose helper accessors that return immutable member and value tuples
    - Remain free of optimization, allocation, validation, and persistence
      logic

Dependencies:
    Python standard library only (``enum.Enum``).

Public API:
    ``PortfolioDirection``, ``OptimizerStrategy``, ``portfolio_directions``,
    ``optimizer_strategies``, ``portfolio_direction_values``,
    ``optimizer_strategy_values``

Notes:
    Both enumerations subclass ``str`` and ``Enum`` so members serialize
    naturally into Polars DataFrames without conversion. Only
    ``EQUAL_WEIGHT`` and ``FIXED_WEIGHT`` are implemented initially; the
    remaining ``OptimizerStrategy`` members reserve the public API for
    future optimizers.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "OptimizerStrategy",
    "PortfolioDirection",
    "optimizer_strategies",
    "optimizer_strategy_values",
    "portfolio_direction_values",
    "portfolio_directions",
]


# Prefer str+Enum over StrEnum so enum values embed directly into Polars.
class PortfolioDirection(str, Enum):  # noqa: UP042
    """Canonical portfolio allocation direction.

    Attributes:
        LONG: Positive target exposure.
        SHORT: Negative target exposure.
        FLAT: Zero target exposure.
    """

    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class OptimizerStrategy(str, Enum):  # noqa: UP042
    """Canonical portfolio optimizer strategy identifiers.

    Attributes:
        EQUAL_WEIGHT: Allocate equal weight across eligible positions.
        FIXED_WEIGHT: Allocate configured fixed weights.
        RISK_PARITY: Reserved for risk-parity optimization.
        MEAN_VARIANCE: Reserved for mean-variance optimization.
        HIERARCHICAL_RISK_PARITY: Reserved for hierarchical risk parity.
        KELLY: Reserved for Kelly-criterion allocation.
    """

    EQUAL_WEIGHT = "equal_weight"
    FIXED_WEIGHT = "fixed_weight"
    RISK_PARITY = "risk_parity"
    MEAN_VARIANCE = "mean_variance"
    HIERARCHICAL_RISK_PARITY = "hierarchical_risk_parity"
    KELLY = "kelly"


def portfolio_directions() -> tuple[PortfolioDirection, ...]:
    """Return an immutable copy of every ``PortfolioDirection`` member.

    Returns:
        All portfolio-direction members in declaration order.
    """
    return (
        PortfolioDirection.LONG,
        PortfolioDirection.SHORT,
        PortfolioDirection.FLAT,
    )


def optimizer_strategies() -> tuple[OptimizerStrategy, ...]:
    """Return an immutable copy of every ``OptimizerStrategy`` member.

    Returns:
        All optimizer-strategy members in declaration order.
    """
    return (
        OptimizerStrategy.EQUAL_WEIGHT,
        OptimizerStrategy.FIXED_WEIGHT,
        OptimizerStrategy.RISK_PARITY,
        OptimizerStrategy.MEAN_VARIANCE,
        OptimizerStrategy.HIERARCHICAL_RISK_PARITY,
        OptimizerStrategy.KELLY,
    )


def portfolio_direction_values() -> tuple[str, ...]:
    """Return an immutable copy of every ``PortfolioDirection`` string value.

    Returns:
        All portfolio-direction values in declaration order.
    """
    return (
        PortfolioDirection.LONG.value,
        PortfolioDirection.SHORT.value,
        PortfolioDirection.FLAT.value,
    )


def optimizer_strategy_values() -> tuple[str, ...]:
    """Return an immutable copy of every ``OptimizerStrategy`` string value.

    Returns:
        All optimizer-strategy values in declaration order.
    """
    return (
        OptimizerStrategy.EQUAL_WEIGHT.value,
        OptimizerStrategy.FIXED_WEIGHT.value,
        OptimizerStrategy.RISK_PARITY.value,
        OptimizerStrategy.MEAN_VARIANCE.value,
        OptimizerStrategy.HIERARCHICAL_RISK_PARITY.value,
        OptimizerStrategy.KELLY.value,
    )
