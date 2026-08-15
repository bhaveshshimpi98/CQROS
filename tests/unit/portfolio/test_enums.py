"""Unit tests for the CQROS Portfolio enumerations."""

from __future__ import annotations

from enum import Enum

import pytest

from cqros.portfolio import (
    OptimizerStrategy,
    PortfolioDirection,
    optimizer_strategies,
    optimizer_strategy_values,
    portfolio_direction_values,
    portfolio_directions,
)
from cqros.portfolio.enums import OptimizerStrategy as OptimizerStrategyDirect
from cqros.portfolio.enums import PortfolioDirection as PortfolioDirectionDirect
from cqros.portfolio.enums import (
    optimizer_strategies as optimizer_strategies_direct,
)
from cqros.portfolio.enums import (
    optimizer_strategy_values as optimizer_strategy_values_direct,
)
from cqros.portfolio.enums import (
    portfolio_direction_values as portfolio_direction_values_direct,
)
from cqros.portfolio.enums import (
    portfolio_directions as portfolio_directions_direct,
)


def test_enums_are_exported_from_package() -> None:
    """Package exports match the enums module by identity."""
    assert PortfolioDirection is PortfolioDirectionDirect
    assert OptimizerStrategy is OptimizerStrategyDirect
    assert portfolio_directions is portfolio_directions_direct
    assert optimizer_strategies is optimizer_strategies_direct
    assert portfolio_direction_values is portfolio_direction_values_direct
    assert optimizer_strategy_values is optimizer_strategy_values_direct


def test_portfolio_direction_long_member() -> None:
    """LONG member name and value are stable."""
    assert PortfolioDirection.LONG.name == "LONG"
    assert PortfolioDirection.LONG.value == "LONG"
    assert PortfolioDirection.LONG == "LONG"


def test_portfolio_direction_short_member() -> None:
    """SHORT member name and value are stable."""
    assert PortfolioDirection.SHORT.name == "SHORT"
    assert PortfolioDirection.SHORT.value == "SHORT"
    assert PortfolioDirection.SHORT == "SHORT"


def test_portfolio_direction_flat_member() -> None:
    """FLAT member name and value are stable."""
    assert PortfolioDirection.FLAT.name == "FLAT"
    assert PortfolioDirection.FLAT.value == "FLAT"
    assert PortfolioDirection.FLAT == "FLAT"


def test_portfolio_direction_enum_names() -> None:
    """PortfolioDirection names remain the canonical uppercase identifiers."""
    assert tuple(member.name for member in PortfolioDirection) == (
        "LONG",
        "SHORT",
        "FLAT",
    )


def test_portfolio_direction_enum_values() -> None:
    """PortfolioDirection values remain the canonical uppercase strings."""
    assert tuple(member.value for member in PortfolioDirection) == (
        "LONG",
        "SHORT",
        "FLAT",
    )


def test_optimizer_strategy_members() -> None:
    """OptimizerStrategy member names and values are stable."""
    assert OptimizerStrategy.EQUAL_WEIGHT.name == "EQUAL_WEIGHT"
    assert OptimizerStrategy.EQUAL_WEIGHT.value == "equal_weight"
    assert OptimizerStrategy.FIXED_WEIGHT.name == "FIXED_WEIGHT"
    assert OptimizerStrategy.FIXED_WEIGHT.value == "fixed_weight"
    assert OptimizerStrategy.RISK_PARITY.name == "RISK_PARITY"
    assert OptimizerStrategy.RISK_PARITY.value == "risk_parity"
    assert OptimizerStrategy.MEAN_VARIANCE.name == "MEAN_VARIANCE"
    assert OptimizerStrategy.MEAN_VARIANCE.value == "mean_variance"
    assert OptimizerStrategy.HIERARCHICAL_RISK_PARITY.name == "HIERARCHICAL_RISK_PARITY"
    assert OptimizerStrategy.HIERARCHICAL_RISK_PARITY.value == "hierarchical_risk_parity"
    assert OptimizerStrategy.KELLY.name == "KELLY"
    assert OptimizerStrategy.KELLY.value == "kelly"


def test_optimizer_strategy_enum_names() -> None:
    """OptimizerStrategy names remain the canonical uppercase identifiers."""
    assert tuple(member.name for member in OptimizerStrategy) == (
        "EQUAL_WEIGHT",
        "FIXED_WEIGHT",
        "RISK_PARITY",
        "MEAN_VARIANCE",
        "HIERARCHICAL_RISK_PARITY",
        "KELLY",
    )


def test_optimizer_strategy_enum_values() -> None:
    """OptimizerStrategy values remain the reserved snake_case strings."""
    assert tuple(member.value for member in OptimizerStrategy) == (
        "equal_weight",
        "fixed_weight",
        "risk_parity",
        "mean_variance",
        "hierarchical_risk_parity",
        "kelly",
    )


def test_enums_subclass_str_and_enum() -> None:
    """Both enumerations subclass str and Enum for natural serialization."""
    for enum_cls in (PortfolioDirection, OptimizerStrategy):
        assert issubclass(enum_cls, str)
        assert issubclass(enum_cls, Enum)
        for member in enum_cls:
            assert isinstance(member, str)
            assert isinstance(member, enum_cls)
            assert member == member.value


def test_portfolio_directions_helper_output() -> None:
    """portfolio_directions returns every member in declaration order."""
    assert portfolio_directions() == (
        PortfolioDirection.LONG,
        PortfolioDirection.SHORT,
        PortfolioDirection.FLAT,
    )


def test_optimizer_strategies_helper_output() -> None:
    """optimizer_strategies returns every member in declaration order."""
    assert optimizer_strategies() == (
        OptimizerStrategy.EQUAL_WEIGHT,
        OptimizerStrategy.FIXED_WEIGHT,
        OptimizerStrategy.RISK_PARITY,
        OptimizerStrategy.MEAN_VARIANCE,
        OptimizerStrategy.HIERARCHICAL_RISK_PARITY,
        OptimizerStrategy.KELLY,
    )


def test_portfolio_direction_values_helper_output() -> None:
    """portfolio_direction_values returns every string value in order."""
    assert portfolio_direction_values() == ("LONG", "SHORT", "FLAT")


def test_optimizer_strategy_values_helper_output() -> None:
    """optimizer_strategy_values returns every string value in order."""
    assert optimizer_strategy_values() == (
        "equal_weight",
        "fixed_weight",
        "risk_parity",
        "mean_variance",
        "hierarchical_risk_parity",
        "kelly",
    )


def test_helper_outputs_are_immutable_tuples() -> None:
    """Helpers return immutable tuples."""
    direction_members = portfolio_directions()
    strategy_members = optimizer_strategies()
    direction_values = portfolio_direction_values()
    strategy_values = optimizer_strategy_values()

    assert isinstance(direction_members, tuple)
    assert isinstance(strategy_members, tuple)
    assert isinstance(direction_values, tuple)
    assert isinstance(strategy_values, tuple)

    with pytest.raises(TypeError):
        direction_members[0] = PortfolioDirection.FLAT  # type: ignore[index]

    with pytest.raises(TypeError):
        strategy_members[0] = OptimizerStrategy.KELLY  # type: ignore[index]

    with pytest.raises(TypeError):
        direction_values[0] = "FLAT"  # type: ignore[index]

    with pytest.raises(TypeError):
        strategy_values[0] = "kelly"  # type: ignore[index]


def test_helper_independence() -> None:
    """Helpers return independent copies, not shared mutable state."""
    first_directions = portfolio_directions()
    second_directions = portfolio_directions()
    first_strategies = optimizer_strategies()
    second_strategies = optimizer_strategies()
    first_direction_values = portfolio_direction_values()
    second_direction_values = portfolio_direction_values()
    first_strategy_values = optimizer_strategy_values()
    second_strategy_values = optimizer_strategy_values()

    assert first_directions == second_directions
    assert first_directions is not second_directions
    assert first_strategies == second_strategies
    assert first_strategies is not second_strategies
    assert first_direction_values == second_direction_values
    assert first_direction_values is not second_direction_values
    assert first_strategy_values == second_strategy_values
    assert first_strategy_values is not second_strategy_values


def test_enum_members_and_values_are_unique() -> None:
    """Enum names and values contain no duplicates."""
    for enum_cls, members_helper, values_helper in (
        (PortfolioDirection, portfolio_directions, portfolio_direction_values),
        (OptimizerStrategy, optimizer_strategies, optimizer_strategy_values),
    ):
        names = tuple(member.name for member in enum_cls)
        member_values = tuple(member.value for member in enum_cls)

        assert len(names) == len(set(names))
        assert len(member_values) == len(set(member_values))
        assert len(members_helper()) == len(set(members_helper()))
        assert len(values_helper()) == len(set(values_helper()))


def test_enum_round_trips_from_value() -> None:
    """Enum members can be reconstructed from their string values."""
    for enum_cls in (PortfolioDirection, OptimizerStrategy):
        for member in enum_cls:
            assert enum_cls(member.value) is member


def test_invalid_value_raises_value_error() -> None:
    """Unknown serialized values raise ValueError."""
    with pytest.raises(ValueError):
        PortfolioDirection("not_a_valid_direction")

    with pytest.raises(ValueError):
        OptimizerStrategy("not_a_valid_strategy")
