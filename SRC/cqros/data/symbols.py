"""CQROS symbol and research universe data models.

Purpose:
    Provide immutable, exchange-agnostic value objects that describe
    tradeable symbols, research universe membership, selection criteria,
    and point-in-time symbol statistics used throughout CQROS.

Responsibilities:
    - Define symbol lifecycle and category enumerations
    - Define ``SymbolInfo`` for multi-exchange symbol identity and metadata
    - Define ``UniverseSnapshot`` for reproducible historical universes
    - Define ``SymbolFilter`` as a criteria value object (no filtering logic)
    - Define ``SymbolStatistics`` for point-in-time research metrics
    - Remain free of business logic, validation, and I/O

Dependencies:
    Python standard library and ``cqros.core.types``.

Public API:
    The enumerations and dataclasses listed in ``__all__``.

Notes:
    Collections that form part of an immutable value object use ``tuple``
    rather than ``list``. Optional filter fields that are ``None`` express
    unconstrained criteria dimensions. ``native_symbol`` preserves the
    venue-specific symbol string alongside the normalized CQROS ``symbol``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from cqros.core.types import (
    Asset,
    Exchange,
    Id,
    Market,
    Metadata,
    Price,
    Symbol,
    Timeframe,
    Timestamp,
    Volume,
)

__all__ = [
    "SymbolStatus",
    "SymbolCategory",
    "SymbolInfo",
    "UniverseSnapshot",
    "SymbolFilter",
    "SymbolStatistics",
]


class SymbolStatus(StrEnum):
    """Lifecycle status of a tradeable symbol on a venue.

    Attributes:
        PENDING: Listed but not yet available for trading or research use.
        TRADING: Actively available for trading.
        HALTED: Temporarily suspended from trading.
        DELISTED: Permanently removed from the trading venue.
        EXPIRED: Reached expiry and is no longer tradeable.
    """

    PENDING = "pending"
    TRADING = "trading"
    HALTED = "halted"
    DELISTED = "delisted"
    EXPIRED = "expired"


class SymbolCategory(StrEnum):
    """Market-structure category of a tradeable symbol.

    Attributes:
        SPOT: Immediate settlement spot instrument.
        PERPETUAL: Perpetual swap / perpetual futures instrument.
        FUTURES: Dated futures instrument.
        OPTIONS: Options instrument.
        INDEX: Index or synthetic reference instrument.
        OTHER: Category not covered by the explicit members.
    """

    SPOT = "spot"
    PERPETUAL = "perpetual"
    FUTURES = "futures"
    OPTIONS = "options"
    INDEX = "index"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class SymbolInfo:
    """Canonical symbol identity and metadata for research universes.

    Captures the normalized CQROS symbol together with exchange, asset
    pair, lifecycle status, and category so historical universes remain
    reproducible across venues.

    Attributes:
        symbol: CQROS instrument symbol (for example ``BTCUSDT``).
        exchange: Exchange identifier (for example ``binance``).
        base_asset: Base asset code (for example ``BTC``).
        quote_asset: Quote asset code (for example ``USDT``).
        status: Current symbol lifecycle status.
        category: Market-structure category of the symbol.
        market: Market category identifier (for example ``usdt_perpetual``),
            if known.
        native_symbol: Venue-native symbol string, if different from the
            normalized CQROS symbol.
        listed_at: Listing timestamp (UTC), if known.
        delisted_at: Delisting timestamp (UTC), if known.
        updated_at: Last metadata update timestamp (UTC), if known.
        metadata: Additional structured metadata, if recorded.
    """

    symbol: Symbol
    exchange: Exchange
    base_asset: Asset
    quote_asset: Asset
    status: SymbolStatus
    category: SymbolCategory
    market: Market | None = None
    native_symbol: str | None = None
    listed_at: Timestamp | None = None
    delisted_at: Timestamp | None = None
    updated_at: Timestamp | None = None
    metadata: Metadata | None = None


@dataclass(frozen=True, slots=True)
class SymbolFilter:
    """Selection criteria for constructing a research symbol universe.

    This is a criteria value object only. It does not apply filters or
    query repositories. A field set to ``None`` means that dimension is
    unconstrained.

    Attributes:
        exchanges: Allowed exchange identifiers, if constrained.
        symbols: Explicit symbol allowlist, if constrained.
        base_assets: Allowed base asset codes, if constrained.
        quote_assets: Allowed quote asset codes, if constrained.
        statuses: Allowed symbol lifecycle statuses, if constrained.
        categories: Allowed symbol categories, if constrained.
        markets: Allowed market category identifiers, if constrained.
        listed_after: Inclusive lower bound on listing time (UTC), if set.
        listed_before: Exclusive upper bound on listing time (UTC), if set.
        include_delisted: Whether delisted symbols may be retained.
        min_quote_volume: Minimum quote-asset volume threshold, if set.
        max_symbols: Maximum number of symbols to retain, if set.
    """

    exchanges: tuple[Exchange, ...] | None = None
    symbols: tuple[Symbol, ...] | None = None
    base_assets: tuple[Asset, ...] | None = None
    quote_assets: tuple[Asset, ...] | None = None
    statuses: tuple[SymbolStatus, ...] | None = None
    categories: tuple[SymbolCategory, ...] | None = None
    markets: tuple[Market, ...] | None = None
    listed_after: Timestamp | None = None
    listed_before: Timestamp | None = None
    include_delisted: bool = False
    min_quote_volume: Volume | None = None
    max_symbols: int | None = None


@dataclass(frozen=True, slots=True)
class SymbolStatistics:
    """Point-in-time research statistics for a single symbol.

    Attributes:
        symbol: CQROS instrument symbol (for example ``BTCUSDT``).
        exchange: Exchange identifier (for example ``binance``).
        as_of: Statistics observation timestamp (UTC).
        window: Aggregation window identifier (for example ``1d``), if
            applicable.
        quote_volume: Quote-asset traded volume over the window, if known.
        base_volume: Base-asset traded volume over the window, if known.
        trade_count: Number of trades over the window, if known.
        last_price: Last traded price in quote-asset units, if known.
        volatility: Realized volatility over the window, if known.
        open_interest: Open interest in base-asset (or contract) units,
            if known.
    """

    symbol: Symbol
    exchange: Exchange
    as_of: Timestamp
    window: Timeframe | None = None
    quote_volume: Volume | None = None
    base_volume: Volume | None = None
    trade_count: int | None = None
    last_price: Price | None = None
    volatility: float | None = None
    open_interest: float | None = None


@dataclass(frozen=True, slots=True)
class UniverseSnapshot:
    """Point-in-time snapshot of a research symbol universe.

    Snapshots are versioned and checksummed so historical research can
    reproduce the exact symbol membership used at ``as_of``. Multi-exchange
    universes are supported by including ``SymbolInfo`` entries from more
    than one venue.

    Attributes:
        snapshot_id: Stable snapshot identifier.
        version: Snapshot version string.
        created_at: Snapshot creation timestamp (UTC).
        as_of: Historical point-in-time the universe represents (UTC).
        symbols: Immutable sequence of symbol entries in the universe.
        checksum: Content checksum of the snapshot payload.
        exchanges: Immutable sequence of exchange identifiers covered, if
            recorded separately from ``symbols``.
        name: Human-readable snapshot name, if assigned.
        description: Free-text description of the universe, if assigned.
        symbol_filter: Criteria used to construct the snapshot, if recorded.
        statistics: Point-in-time statistics aligned with ``symbols``, if
            recorded.
        metadata: Additional structured metadata, if recorded.
    """

    snapshot_id: Id
    version: str
    created_at: Timestamp
    as_of: Timestamp
    symbols: tuple[SymbolInfo, ...]
    checksum: str
    exchanges: tuple[Exchange, ...] | None = None
    name: str | None = None
    description: str | None = None
    symbol_filter: SymbolFilter | None = None
    statistics: tuple[SymbolStatistics, ...] | None = None
    metadata: Metadata | None = None
