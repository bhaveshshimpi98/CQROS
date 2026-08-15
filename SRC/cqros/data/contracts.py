"""CQROS trading contract data models.

Purpose:
    Provide immutable, exchange-agnostic value objects that describe a
    tradeable contract and the price, quantity, notional, and leverage
    constraints associated with it.

Responsibilities:
    - Define contract classification and lifecycle enumerations
    - Define trading filter value objects used by execution validation
    - Define the canonical ``Contract`` model shared across CQROS layers
    - Remain free of business logic, validation, and I/O

Dependencies:
    Python standard library and ``cqros.core.types``.

Public API:
    The enumerations and dataclasses listed in ``__all__``.

Notes:
    Filter bounds that do not apply on a given venue are expressed as
    ``None``. Derivative-only attributes on ``Contract`` are optional so
    spot and future market types share one model without venue-specific
    assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from cqros.core.types import (
    Asset,
    Exchange,
    Leverage,
    Price,
    Quantity,
    Symbol,
    Timestamp,
)

__all__ = [
    "ContractType",
    "ContractStatus",
    "PriceFilter",
    "QuantityFilter",
    "NotionalFilter",
    "LeverageFilter",
    "Contract",
]


class ContractType(StrEnum):
    """Classification of a tradeable contract.

    Attributes:
        SPOT: Immediate settlement spot instrument.
        PERPETUAL: Perpetual swap / perpetual futures instrument.
        FUTURES: Dated futures instrument.
        OPTIONS: Options instrument.
    """

    SPOT = "spot"
    PERPETUAL = "perpetual"
    FUTURES = "futures"
    OPTIONS = "options"


class ContractStatus(StrEnum):
    """Lifecycle status of a trading contract.

    Attributes:
        PENDING: Listed but not yet available for trading.
        TRADING: Actively available for trading.
        HALTED: Temporarily suspended from trading.
        SETTLING: Undergoing settlement or delivery.
        EXPIRED: Reached expiry and is no longer tradeable.
        DELISTED: Permanently removed from the trading venue.
    """

    PENDING = "pending"
    TRADING = "trading"
    HALTED = "halted"
    SETTLING = "settling"
    EXPIRED = "expired"
    DELISTED = "delisted"


@dataclass(frozen=True, slots=True)
class PriceFilter:
    """Price constraint parameters for a contract.

    Attributes:
        tick_size: Minimum allowed price increment in quote-asset units.
        min_price: Lowest permitted order price, if bounded.
        max_price: Highest permitted order price, if bounded.
    """

    tick_size: Price
    min_price: Price | None = None
    max_price: Price | None = None


@dataclass(frozen=True, slots=True)
class QuantityFilter:
    """Quantity constraint parameters for a contract.

    Attributes:
        step_size: Minimum allowed quantity increment in base-asset units.
        min_quantity: Lowest permitted order quantity.
        max_quantity: Highest permitted order quantity, if bounded.
    """

    step_size: Quantity
    min_quantity: Quantity
    max_quantity: Quantity | None = None


@dataclass(frozen=True, slots=True)
class NotionalFilter:
    """Notional (price × quantity) constraint parameters for a contract.

    Notional values are denominated in quote-asset units.

    Attributes:
        min_notional: Lowest permitted order notional value.
        max_notional: Highest permitted order notional value, if bounded.
    """

    min_notional: float
    max_notional: float | None = None


@dataclass(frozen=True, slots=True)
class LeverageFilter:
    """Leverage constraint parameters for a contract.

    Attributes:
        min_leverage: Lowest permitted leverage multiplier.
        max_leverage: Highest permitted leverage multiplier.
    """

    min_leverage: Leverage
    max_leverage: Leverage


@dataclass(frozen=True, slots=True)
class Contract:
    """Canonical trading contract used throughout CQROS.

    A contract identifies a tradeable instrument on an exchange and
    carries the trading filters required by downstream research,
    portfolio, and execution layers. Optional fields support derivative
    instruments without encoding venue-specific semantics.

    Attributes:
        symbol: CQROS instrument symbol (for example ``BTCUSDT``).
        exchange: Exchange identifier (for example ``binance``).
        base_asset: Base asset code (for example ``BTC``).
        quote_asset: Quote asset code (for example ``USDT``).
        contract_type: Instrument classification.
        status: Current trading lifecycle status.
        price_filter: Price tick and bound constraints.
        quantity_filter: Quantity step and bound constraints.
        notional_filter: Notional bound constraints, if applicable.
        leverage_filter: Leverage bound constraints, if applicable.
        margin_asset: Asset used for margin, if applicable.
        settlement_asset: Asset used for settlement, if applicable.
        contract_size: Contract multiplier / size, if applicable.
        expiry: Contract expiry timestamp (UTC), if dated.
        listed_at: Listing timestamp (UTC), if known.
        updated_at: Last metadata update timestamp (UTC), if known.
    """

    symbol: Symbol
    exchange: Exchange
    base_asset: Asset
    quote_asset: Asset
    contract_type: ContractType
    status: ContractStatus
    price_filter: PriceFilter
    quantity_filter: QuantityFilter
    notional_filter: NotionalFilter | None = None
    leverage_filter: LeverageFilter | None = None
    margin_asset: Asset | None = None
    settlement_asset: Asset | None = None
    contract_size: float | None = None
    expiry: Timestamp | None = None
    listed_at: Timestamp | None = None
    updated_at: Timestamp | None = None
