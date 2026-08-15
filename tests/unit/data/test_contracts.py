"""Unit tests for CQROS trading contract data models."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass
from datetime import UTC, datetime
from enum import StrEnum

import pytest

from cqros.data.contracts import (
    Contract,
    ContractStatus,
    ContractType,
    LeverageFilter,
    NotionalFilter,
    PriceFilter,
    QuantityFilter,
)

_ENUM_TYPES: tuple[type[StrEnum], ...] = (
    ContractType,
    ContractStatus,
)

_EXPECTED_VALUES: dict[type[StrEnum], dict[str, str]] = {
    ContractType: {
        "SPOT": "spot",
        "PERPETUAL": "perpetual",
        "FUTURES": "futures",
        "OPTIONS": "options",
    },
    ContractStatus: {
        "PENDING": "pending",
        "TRADING": "trading",
        "HALTED": "halted",
        "SETTLING": "settling",
        "EXPIRED": "expired",
        "DELISTED": "delisted",
    },
}


def _sample_price_filter() -> PriceFilter:
    return PriceFilter(tick_size=0.01, min_price=0.01, max_price=1_000_000.0)


def _sample_quantity_filter() -> QuantityFilter:
    return QuantityFilter(step_size=0.001, min_quantity=0.001, max_quantity=10_000.0)


def _sample_contract(**overrides: object) -> Contract:
    values: dict[str, object] = {
        "symbol": "BTCUSDT",
        "exchange": "binance",
        "base_asset": "BTC",
        "quote_asset": "USDT",
        "contract_type": ContractType.PERPETUAL,
        "status": ContractStatus.TRADING,
        "price_filter": _sample_price_filter(),
        "quantity_filter": _sample_quantity_filter(),
        "notional_filter": NotionalFilter(min_notional=5.0, max_notional=None),
        "leverage_filter": LeverageFilter(min_leverage=1.0, max_leverage=125.0),
        "margin_asset": "USDT",
        "settlement_asset": "USDT",
        "contract_size": 1.0,
        "expiry": None,
        "listed_at": datetime(2020, 1, 1, tzinfo=UTC),
        "updated_at": datetime(2026, 7, 25, tzinfo=UTC),
    }
    values.update(overrides)
    return Contract(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("enum_type", _ENUM_TYPES)
def test_enums_are_str_enums(enum_type: type[StrEnum]) -> None:
    """Contract enumerations are serializable string enums."""
    assert issubclass(enum_type, StrEnum)
    for member in enum_type:
        assert isinstance(member.value, str)
        assert str(member) == member.value


@pytest.mark.parametrize("enum_type", _ENUM_TYPES)
def test_enum_member_values(enum_type: type[StrEnum]) -> None:
    """Enumeration members expose the documented string values."""
    expected = _EXPECTED_VALUES[enum_type]
    assert {member.name: member.value for member in enum_type} == expected


def test_price_filter_is_frozen_dataclass() -> None:
    """PriceFilter is an immutable slotted dataclass."""
    price_filter = PriceFilter(tick_size=0.1)
    assert is_dataclass(price_filter)
    assert price_filter.tick_size == pytest.approx(0.1)
    assert price_filter.min_price is None
    assert price_filter.max_price is None
    with pytest.raises(FrozenInstanceError):
        price_filter.tick_size = 0.2  # type: ignore[misc]


def test_quantity_filter_is_frozen_dataclass() -> None:
    """QuantityFilter is an immutable slotted dataclass."""
    quantity_filter = QuantityFilter(step_size=0.01, min_quantity=0.01)
    assert is_dataclass(quantity_filter)
    assert quantity_filter.step_size == pytest.approx(0.01)
    assert quantity_filter.min_quantity == pytest.approx(0.01)
    assert quantity_filter.max_quantity is None
    with pytest.raises(FrozenInstanceError):
        quantity_filter.step_size = 0.02  # type: ignore[misc]


def test_notional_filter_is_frozen_dataclass() -> None:
    """NotionalFilter is an immutable slotted dataclass."""
    notional_filter = NotionalFilter(min_notional=10.0, max_notional=1_000_000.0)
    assert is_dataclass(notional_filter)
    assert notional_filter.min_notional == pytest.approx(10.0)
    assert notional_filter.max_notional == pytest.approx(1_000_000.0)
    with pytest.raises(FrozenInstanceError):
        notional_filter.min_notional = 1.0  # type: ignore[misc]


def test_leverage_filter_is_frozen_dataclass() -> None:
    """LeverageFilter is an immutable slotted dataclass."""
    leverage_filter = LeverageFilter(min_leverage=1.0, max_leverage=20.0)
    assert is_dataclass(leverage_filter)
    assert leverage_filter.min_leverage == pytest.approx(1.0)
    assert leverage_filter.max_leverage == pytest.approx(20.0)
    with pytest.raises(FrozenInstanceError):
        leverage_filter.max_leverage = 10.0  # type: ignore[misc]


def test_contract_required_fields_and_defaults() -> None:
    """Contract requires identity and filters; optional fields default to None."""
    contract = Contract(
        symbol="ETHUSDT",
        exchange="binance",
        base_asset="ETH",
        quote_asset="USDT",
        contract_type=ContractType.SPOT,
        status=ContractStatus.TRADING,
        price_filter=PriceFilter(tick_size=0.01),
        quantity_filter=QuantityFilter(step_size=0.0001, min_quantity=0.0001),
    )

    assert contract.symbol == "ETHUSDT"
    assert contract.exchange == "binance"
    assert contract.base_asset == "ETH"
    assert contract.quote_asset == "USDT"
    assert contract.contract_type is ContractType.SPOT
    assert contract.status is ContractStatus.TRADING
    assert contract.notional_filter is None
    assert contract.leverage_filter is None
    assert contract.margin_asset is None
    assert contract.settlement_asset is None
    assert contract.contract_size is None
    assert contract.expiry is None
    assert contract.listed_at is None
    assert contract.updated_at is None


def test_contract_is_frozen_and_slotted() -> None:
    """Contract instances are immutable and use slots."""
    contract = _sample_contract()
    assert is_dataclass(contract)
    assert hasattr(type(contract), "__slots__")
    with pytest.raises(FrozenInstanceError):
        contract.status = ContractStatus.HALTED  # type: ignore[misc]


def test_contract_equality_and_hash() -> None:
    """Equal Contract values compare equal and are hashable."""
    left = _sample_contract()
    right = _sample_contract()
    different = _sample_contract(status=ContractStatus.HALTED)

    assert left == right
    assert hash(left) == hash(right)
    assert left != different
    assert {left, right, different} == {left, different}


def test_contract_field_names_are_stable() -> None:
    """Contract public field names remain stable for serialization consumers."""
    names = tuple(field.name for field in fields(Contract))
    assert names == (
        "symbol",
        "exchange",
        "base_asset",
        "quote_asset",
        "contract_type",
        "status",
        "price_filter",
        "quantity_filter",
        "notional_filter",
        "leverage_filter",
        "margin_asset",
        "settlement_asset",
        "contract_size",
        "expiry",
        "listed_at",
        "updated_at",
    )


def test_package_exports_contract_models() -> None:
    """The data package re-exports the contract public API."""
    import cqros.data as data_package

    for name in data_package.__all__:
        assert hasattr(data_package, name)
    assert data_package.Contract is Contract
    assert data_package.ContractType is ContractType
    assert data_package.ContractStatus is ContractStatus
