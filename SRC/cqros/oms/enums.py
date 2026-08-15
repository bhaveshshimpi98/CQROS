"""CQROS Order Management System enumerations.

Purpose:
    Define the canonical order-side, order-type, order-status, and order-
    manager vocabulary used throughout CQROS OMS.

Responsibilities:
    - Enumerate every supported order side as a string-backed enumeration
    - Enumerate every supported order type as a string-backed enumeration
    - Enumerate every supported order status as a string-backed enumeration
    - Enumerate every reserved order-manager strategy as a string-backed
      enumeration
    - Expose helper accessors that return immutable member and value tuples
    - Remain free of order generation, execution, state-transition, and
      persistence logic

Dependencies:
    Python standard library only (``enum.Enum``).

Public API:
    ``OrderSide``, ``OrderType``, ``OrderStatus``, ``OrderManagerType``,
    ``order_sides``, ``order_types``, ``order_statuses``,
    ``order_manager_types``, ``values``

Notes:
    All enumerations subclass ``str`` and ``Enum`` so members serialize
    naturally into Polars DataFrames without conversion. Enum members
    reserve the public API; no OMS business logic lives here.
"""

from __future__ import annotations

from enum import Enum
from typing import TypeVar

__all__ = [
    "OrderManagerType",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "order_manager_types",
    "order_sides",
    "order_statuses",
    "order_types",
    "values",
]

_EnumT = TypeVar("_EnumT", bound=Enum)


# Prefer str+Enum over StrEnum so enum values embed directly into Polars.
class OrderSide(str, Enum):  # noqa: UP042
    """Canonical order side.

    Attributes:
        BUY: Acquire or increase long exposure.
        SELL: Dispose or increase short exposure.
    """

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):  # noqa: UP042
    """Canonical order type identifiers.

    Attributes:
        MARKET: Immediate execution at the prevailing market price.
        LIMIT: Execution at the specified limit price or better.
        STOP_MARKET: Market order triggered at the stop price.
        STOP_LIMIT: Limit order triggered at the stop price.
        TAKE_PROFIT_MARKET: Market order triggered at the take-profit price.
        TAKE_PROFIT_LIMIT: Limit order triggered at the take-profit price.
    """

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"


class OrderStatus(str, Enum):  # noqa: UP042
    """Canonical order lifecycle status.

    Attributes:
        PENDING: Order created but not yet submitted.
        SUBMITTED: Order submitted to the execution venue.
        PARTIALLY_FILLED: Order filled for a portion of its quantity.
        FILLED: Order completely filled.
        CANCELLED: Order cancelled before completion.
        REJECTED: Order rejected by the venue or risk controls.
        EXPIRED: Order expired without completion.
    """

    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class OrderManagerType(str, Enum):  # noqa: UP042
    """Canonical order-manager strategy identifiers.

    Attributes:
        SIMPLE: Reserved for single-shot / direct order submission.
        TWAP: Reserved for time-weighted average price execution.
        VWAP: Reserved for volume-weighted average price execution.
        POV: Reserved for percentage-of-volume execution.
        ICEBERG: Reserved for iceberg / displayed-quantity execution.
    """

    SIMPLE = "simple"
    TWAP = "twap"
    VWAP = "vwap"
    POV = "pov"
    ICEBERG = "iceberg"


def order_sides() -> tuple[OrderSide, ...]:
    """Return an immutable copy of every ``OrderSide`` member.

    Returns:
        All order-side members in declaration order.
    """
    return (
        OrderSide.BUY,
        OrderSide.SELL,
    )


def order_types() -> tuple[OrderType, ...]:
    """Return an immutable copy of every ``OrderType`` member.

    Returns:
        All order-type members in declaration order.
    """
    return (
        OrderType.MARKET,
        OrderType.LIMIT,
        OrderType.STOP_MARKET,
        OrderType.STOP_LIMIT,
        OrderType.TAKE_PROFIT_MARKET,
        OrderType.TAKE_PROFIT_LIMIT,
    )


def order_statuses() -> tuple[OrderStatus, ...]:
    """Return an immutable copy of every ``OrderStatus`` member.

    Returns:
        All order-status members in declaration order.
    """
    return (
        OrderStatus.PENDING,
        OrderStatus.SUBMITTED,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
        OrderStatus.REJECTED,
        OrderStatus.EXPIRED,
    )


def order_manager_types() -> tuple[OrderManagerType, ...]:
    """Return an immutable copy of every ``OrderManagerType`` member.

    Returns:
        All order-manager-type members in declaration order.
    """
    return (
        OrderManagerType.SIMPLE,
        OrderManagerType.TWAP,
        OrderManagerType.VWAP,
        OrderManagerType.POV,
        OrderManagerType.ICEBERG,
    )


def values(enum_cls: type[_EnumT]) -> tuple[str, ...]:
    """Return an immutable copy of every string value for ``enum_cls``.

    Args:
        enum_cls: Enumeration class whose member values are requested.

    Returns:
        All member string values in declaration order.
    """
    return tuple(member.value for member in enum_cls)
