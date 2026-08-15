"""Unit tests for the CQROS OMS enumerations."""

from __future__ import annotations

from enum import Enum

import pytest

from cqros.oms import (
    OrderManagerType,
    OrderSide,
    OrderStatus,
    OrderType,
    order_manager_types,
    order_sides,
    order_statuses,
    order_types,
    values,
)
from cqros.oms.enums import OrderManagerType as OrderManagerTypeDirect
from cqros.oms.enums import OrderSide as OrderSideDirect
from cqros.oms.enums import OrderStatus as OrderStatusDirect
from cqros.oms.enums import OrderType as OrderTypeDirect
from cqros.oms.enums import order_manager_types as order_manager_types_direct
from cqros.oms.enums import order_sides as order_sides_direct
from cqros.oms.enums import order_statuses as order_statuses_direct
from cqros.oms.enums import order_types as order_types_direct
from cqros.oms.enums import values as values_direct


def test_enums_are_exported_from_package() -> None:
    """Package exports match the enums module by identity."""
    assert OrderSide is OrderSideDirect
    assert OrderType is OrderTypeDirect
    assert OrderStatus is OrderStatusDirect
    assert OrderManagerType is OrderManagerTypeDirect
    assert order_sides is order_sides_direct
    assert order_types is order_types_direct
    assert order_statuses is order_statuses_direct
    assert order_manager_types is order_manager_types_direct
    assert values is values_direct


def test_order_side_buy_member() -> None:
    """BUY member name and value are stable."""
    assert OrderSide.BUY.name == "BUY"
    assert OrderSide.BUY.value == "BUY"
    assert OrderSide.BUY == "BUY"


def test_order_side_sell_member() -> None:
    """SELL member name and value are stable."""
    assert OrderSide.SELL.name == "SELL"
    assert OrderSide.SELL.value == "SELL"
    assert OrderSide.SELL == "SELL"


def test_order_side_enum_names() -> None:
    """OrderSide names remain the canonical uppercase identifiers."""
    assert tuple(member.name for member in OrderSide) == ("BUY", "SELL")


def test_order_side_enum_values() -> None:
    """OrderSide values remain the canonical uppercase strings."""
    assert tuple(member.value for member in OrderSide) == ("BUY", "SELL")


def test_order_type_members() -> None:
    """OrderType member names and values are stable."""
    assert OrderType.MARKET.name == "MARKET"
    assert OrderType.MARKET.value == "MARKET"
    assert OrderType.LIMIT.name == "LIMIT"
    assert OrderType.LIMIT.value == "LIMIT"
    assert OrderType.STOP_MARKET.name == "STOP_MARKET"
    assert OrderType.STOP_MARKET.value == "STOP_MARKET"
    assert OrderType.STOP_LIMIT.name == "STOP_LIMIT"
    assert OrderType.STOP_LIMIT.value == "STOP_LIMIT"
    assert OrderType.TAKE_PROFIT_MARKET.name == "TAKE_PROFIT_MARKET"
    assert OrderType.TAKE_PROFIT_MARKET.value == "TAKE_PROFIT_MARKET"
    assert OrderType.TAKE_PROFIT_LIMIT.name == "TAKE_PROFIT_LIMIT"
    assert OrderType.TAKE_PROFIT_LIMIT.value == "TAKE_PROFIT_LIMIT"


def test_order_type_enum_names() -> None:
    """OrderType names remain the canonical uppercase identifiers."""
    assert tuple(member.name for member in OrderType) == (
        "MARKET",
        "LIMIT",
        "STOP_MARKET",
        "STOP_LIMIT",
        "TAKE_PROFIT_MARKET",
        "TAKE_PROFIT_LIMIT",
    )


def test_order_type_enum_values() -> None:
    """OrderType values remain the canonical uppercase strings."""
    assert tuple(member.value for member in OrderType) == (
        "MARKET",
        "LIMIT",
        "STOP_MARKET",
        "STOP_LIMIT",
        "TAKE_PROFIT_MARKET",
        "TAKE_PROFIT_LIMIT",
    )


def test_order_status_members() -> None:
    """OrderStatus member names and values are stable."""
    assert OrderStatus.PENDING.name == "PENDING"
    assert OrderStatus.PENDING.value == "PENDING"
    assert OrderStatus.SUBMITTED.name == "SUBMITTED"
    assert OrderStatus.SUBMITTED.value == "SUBMITTED"
    assert OrderStatus.PARTIALLY_FILLED.name == "PARTIALLY_FILLED"
    assert OrderStatus.PARTIALLY_FILLED.value == "PARTIALLY_FILLED"
    assert OrderStatus.FILLED.name == "FILLED"
    assert OrderStatus.FILLED.value == "FILLED"
    assert OrderStatus.CANCELLED.name == "CANCELLED"
    assert OrderStatus.CANCELLED.value == "CANCELLED"
    assert OrderStatus.REJECTED.name == "REJECTED"
    assert OrderStatus.REJECTED.value == "REJECTED"
    assert OrderStatus.EXPIRED.name == "EXPIRED"
    assert OrderStatus.EXPIRED.value == "EXPIRED"


def test_order_status_enum_names() -> None:
    """OrderStatus names remain the canonical uppercase identifiers."""
    assert tuple(member.name for member in OrderStatus) == (
        "PENDING",
        "SUBMITTED",
        "PARTIALLY_FILLED",
        "FILLED",
        "CANCELLED",
        "REJECTED",
        "EXPIRED",
    )


def test_order_status_enum_values() -> None:
    """OrderStatus values remain the canonical uppercase strings."""
    assert tuple(member.value for member in OrderStatus) == (
        "PENDING",
        "SUBMITTED",
        "PARTIALLY_FILLED",
        "FILLED",
        "CANCELLED",
        "REJECTED",
        "EXPIRED",
    )


def test_order_manager_type_members() -> None:
    """OrderManagerType member names and values are stable."""
    assert OrderManagerType.SIMPLE.name == "SIMPLE"
    assert OrderManagerType.SIMPLE.value == "simple"
    assert OrderManagerType.TWAP.name == "TWAP"
    assert OrderManagerType.TWAP.value == "twap"
    assert OrderManagerType.VWAP.name == "VWAP"
    assert OrderManagerType.VWAP.value == "vwap"
    assert OrderManagerType.POV.name == "POV"
    assert OrderManagerType.POV.value == "pov"
    assert OrderManagerType.ICEBERG.name == "ICEBERG"
    assert OrderManagerType.ICEBERG.value == "iceberg"


def test_order_manager_type_enum_names() -> None:
    """OrderManagerType names remain the canonical uppercase identifiers."""
    assert tuple(member.name for member in OrderManagerType) == (
        "SIMPLE",
        "TWAP",
        "VWAP",
        "POV",
        "ICEBERG",
    )


def test_order_manager_type_enum_values() -> None:
    """OrderManagerType values remain the reserved lowercase strings."""
    assert tuple(member.value for member in OrderManagerType) == (
        "simple",
        "twap",
        "vwap",
        "pov",
        "iceberg",
    )


def test_enums_subclass_str_and_enum() -> None:
    """All enumerations subclass str and Enum for natural serialization."""
    for enum_cls in (OrderSide, OrderType, OrderStatus, OrderManagerType):
        assert issubclass(enum_cls, str)
        assert issubclass(enum_cls, Enum)
        for member in enum_cls:
            assert isinstance(member, str)
            assert isinstance(member, enum_cls)
            assert member == member.value


def test_order_sides_helper_output() -> None:
    """order_sides returns every member in declaration order."""
    assert order_sides() == (OrderSide.BUY, OrderSide.SELL)


def test_order_types_helper_output() -> None:
    """order_types returns every member in declaration order."""
    assert order_types() == (
        OrderType.MARKET,
        OrderType.LIMIT,
        OrderType.STOP_MARKET,
        OrderType.STOP_LIMIT,
        OrderType.TAKE_PROFIT_MARKET,
        OrderType.TAKE_PROFIT_LIMIT,
    )


def test_order_statuses_helper_output() -> None:
    """order_statuses returns every member in declaration order."""
    assert order_statuses() == (
        OrderStatus.PENDING,
        OrderStatus.SUBMITTED,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
        OrderStatus.REJECTED,
        OrderStatus.EXPIRED,
    )


def test_order_manager_types_helper_output() -> None:
    """order_manager_types returns every member in declaration order."""
    assert order_manager_types() == (
        OrderManagerType.SIMPLE,
        OrderManagerType.TWAP,
        OrderManagerType.VWAP,
        OrderManagerType.POV,
        OrderManagerType.ICEBERG,
    )


def test_values_helper_output() -> None:
    """values returns every string value for each OMS enumeration."""
    assert values(OrderSide) == ("BUY", "SELL")
    assert values(OrderType) == (
        "MARKET",
        "LIMIT",
        "STOP_MARKET",
        "STOP_LIMIT",
        "TAKE_PROFIT_MARKET",
        "TAKE_PROFIT_LIMIT",
    )
    assert values(OrderStatus) == (
        "PENDING",
        "SUBMITTED",
        "PARTIALLY_FILLED",
        "FILLED",
        "CANCELLED",
        "REJECTED",
        "EXPIRED",
    )
    assert values(OrderManagerType) == (
        "simple",
        "twap",
        "vwap",
        "pov",
        "iceberg",
    )


def test_helper_outputs_are_immutable_tuples() -> None:
    """Helpers return immutable tuples."""
    side_members = order_sides()
    type_members = order_types()
    status_members = order_statuses()
    manager_members = order_manager_types()
    side_values = values(OrderSide)
    type_values = values(OrderType)
    status_values = values(OrderStatus)
    manager_values = values(OrderManagerType)

    assert isinstance(side_members, tuple)
    assert isinstance(type_members, tuple)
    assert isinstance(status_members, tuple)
    assert isinstance(manager_members, tuple)
    assert isinstance(side_values, tuple)
    assert isinstance(type_values, tuple)
    assert isinstance(status_values, tuple)
    assert isinstance(manager_values, tuple)

    with pytest.raises(TypeError):
        side_members[0] = OrderSide.SELL  # type: ignore[index]

    with pytest.raises(TypeError):
        type_members[0] = OrderType.LIMIT  # type: ignore[index]

    with pytest.raises(TypeError):
        status_members[0] = OrderStatus.FILLED  # type: ignore[index]

    with pytest.raises(TypeError):
        manager_members[0] = OrderManagerType.TWAP  # type: ignore[index]

    with pytest.raises(TypeError):
        side_values[0] = "SELL"  # type: ignore[index]

    with pytest.raises(TypeError):
        type_values[0] = "LIMIT"  # type: ignore[index]

    with pytest.raises(TypeError):
        status_values[0] = "FILLED"  # type: ignore[index]

    with pytest.raises(TypeError):
        manager_values[0] = "twap"  # type: ignore[index]


def test_helper_independence() -> None:
    """Helpers return independent copies, not shared mutable state."""
    first_sides = order_sides()
    second_sides = order_sides()
    first_types = order_types()
    second_types = order_types()
    first_statuses = order_statuses()
    second_statuses = order_statuses()
    first_managers = order_manager_types()
    second_managers = order_manager_types()
    first_side_values = values(OrderSide)
    second_side_values = values(OrderSide)
    first_type_values = values(OrderType)
    second_type_values = values(OrderType)
    first_status_values = values(OrderStatus)
    second_status_values = values(OrderStatus)
    first_manager_values = values(OrderManagerType)
    second_manager_values = values(OrderManagerType)

    assert first_sides == second_sides
    assert first_sides is not second_sides
    assert first_types == second_types
    assert first_types is not second_types
    assert first_statuses == second_statuses
    assert first_statuses is not second_statuses
    assert first_managers == second_managers
    assert first_managers is not second_managers
    assert first_side_values == second_side_values
    assert first_side_values is not second_side_values
    assert first_type_values == second_type_values
    assert first_type_values is not second_type_values
    assert first_status_values == second_status_values
    assert first_status_values is not second_status_values
    assert first_manager_values == second_manager_values
    assert first_manager_values is not second_manager_values


def test_enum_members_and_values_are_unique() -> None:
    """Enum names and values contain no duplicates."""
    for enum_cls, members_helper in (
        (OrderSide, order_sides),
        (OrderType, order_types),
        (OrderStatus, order_statuses),
        (OrderManagerType, order_manager_types),
    ):
        names = tuple(member.name for member in enum_cls)
        member_values = tuple(member.value for member in enum_cls)

        assert len(names) == len(set(names))
        assert len(member_values) == len(set(member_values))
        assert len(members_helper()) == len(set(members_helper()))
        assert len(values(enum_cls)) == len(set(values(enum_cls)))


def test_enum_round_trips_from_value() -> None:
    """Enum members can be reconstructed from their string values."""
    for enum_cls in (OrderSide, OrderType, OrderStatus, OrderManagerType):
        for member in enum_cls:
            assert enum_cls(member.value) is member


def test_invalid_value_raises_value_error() -> None:
    """Unknown serialized values raise ValueError."""
    with pytest.raises(ValueError):
        OrderSide("not_a_valid_side")

    with pytest.raises(ValueError):
        OrderType("not_a_valid_type")

    with pytest.raises(ValueError):
        OrderStatus("not_a_valid_status")

    with pytest.raises(ValueError):
        OrderManagerType("not_a_valid_manager")
