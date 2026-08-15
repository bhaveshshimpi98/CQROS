"""CQROS canonical market data schema models.

Purpose:
    Provide immutable, exchange-agnostic value objects that represent
    market data observations and related metadata used throughout CQROS.

Responsibilities:
    - Define canonical structures for candles, trades, funding, open
      interest, liquidations, order books, and tickers
    - Define kline metadata and market-storage dataset descriptor value
      objects
    - Remain free of business logic, validation, and I/O

Dependencies:
    Python standard library, ``cqros.core.enums``, and ``cqros.core.types``.

Public API:
    The dataclasses listed in ``__all__``.

Notes:
    Collections that form part of an immutable value object use ``tuple``
    rather than ``list``. Optional fields express values that are not
    available on every venue or data source without encoding
    exchange-specific semantics.
    Research reproducibility metadata lives in ``cqros.data.metadata``;
    ``DatasetDescriptor`` describes market-storage dataset artifacts only.
"""

from __future__ import annotations

from dataclasses import dataclass

from cqros.core.enums import DataSource, OrderSide
from cqros.core.types import (
    CompressionCodec,
    Exchange,
    FilePath,
    Id,
    Metadata,
    Price,
    Quantity,
    Symbol,
    Timeframe,
    Timestamp,
    Volume,
)

__all__ = [
    "OHLCV",
    "Trade",
    "FundingRate",
    "OpenInterest",
    "Liquidation",
    "OrderBookLevel",
    "OrderBookSnapshot",
    "Ticker",
    "KlineMetadata",
    "DatasetDescriptor",
    "DatasetMetadata",
]


@dataclass(frozen=True, slots=True)
class OHLCV:
    """Canonical OHLCV (candle / kline) bar.

    Attributes:
        symbol: CQROS instrument symbol (for example ``BTCUSDT``).
        timeframe: Bar interval identifier (for example ``1m``).
        open_time: Bar open timestamp (UTC).
        close_time: Bar close timestamp (UTC).
        open: Opening price in quote-asset units.
        high: Highest price in quote-asset units.
        low: Lowest price in quote-asset units.
        close: Closing price in quote-asset units.
        volume: Base-asset traded volume during the bar.
        quote_volume: Quote-asset traded volume during the bar, if known.
        trade_count: Number of trades aggregated into the bar, if known.
    """

    symbol: Symbol
    timeframe: Timeframe
    open_time: Timestamp
    close_time: Timestamp
    open: Price
    high: Price
    low: Price
    close: Price
    volume: Volume
    quote_volume: Volume | None = None
    trade_count: int | None = None


@dataclass(frozen=True, slots=True)
class Trade:
    """Canonical public market trade.

    Attributes:
        trade_id: Exchange or normalized trade identifier.
        symbol: CQROS instrument symbol (for example ``BTCUSDT``).
        timestamp: Trade execution timestamp (UTC).
        side: Aggressor side relative to the base asset.
        price: Trade price in quote-asset units.
        quantity: Trade quantity in base-asset units.
        buyer_maker: Whether the buyer was the maker, if known.
    """

    trade_id: Id
    symbol: Symbol
    timestamp: Timestamp
    side: OrderSide
    price: Price
    quantity: Quantity
    buyer_maker: bool | None = None


@dataclass(frozen=True, slots=True)
class FundingRate:
    """Canonical perpetual funding rate observation.

    Attributes:
        symbol: CQROS instrument symbol (for example ``BTCUSDT``).
        funding_time: Funding settlement timestamp (UTC).
        funding_rate: Funding rate as a fractional value.
        mark_price: Mark price at funding time, if known.
    """

    symbol: Symbol
    funding_time: Timestamp
    funding_rate: float
    mark_price: Price | None = None


@dataclass(frozen=True, slots=True)
class OpenInterest:
    """Canonical open interest observation.

    Attributes:
        symbol: CQROS instrument symbol (for example ``BTCUSDT``).
        timestamp: Observation timestamp (UTC).
        open_interest: Open interest in base-asset (or contract) units.
        value: Notional open interest value in quote-asset units, if known.
    """

    symbol: Symbol
    timestamp: Timestamp
    open_interest: float
    value: float | None = None


@dataclass(frozen=True, slots=True)
class Liquidation:
    """Canonical forced liquidation observation.

    Attributes:
        symbol: CQROS instrument symbol (for example ``BTCUSDT``).
        timestamp: Liquidation timestamp (UTC).
        side: Side of the liquidating order relative to the base asset.
        price: Liquidation price in quote-asset units.
        quantity: Liquidated quantity in base-asset units.
        average_price: Average fill price, if known.
        order_id: Exchange or normalized liquidation order identifier, if known.
    """

    symbol: Symbol
    timestamp: Timestamp
    side: OrderSide
    price: Price
    quantity: Quantity
    average_price: Price | None = None
    order_id: Id | None = None


@dataclass(frozen=True, slots=True)
class OrderBookLevel:
    """Single price level in an order book.

    Attributes:
        price: Level price in quote-asset units.
        quantity: Resting quantity at the level in base-asset units.
        order_count: Number of orders at the level, if known.
    """

    price: Price
    quantity: Quantity
    order_count: int | None = None


@dataclass(frozen=True, slots=True)
class OrderBookSnapshot:
    """Canonical order book depth snapshot.

    Attributes:
        symbol: CQROS instrument symbol (for example ``BTCUSDT``).
        timestamp: Snapshot timestamp (UTC).
        bids: Bid levels from best to deeper, as an immutable sequence.
        asks: Ask levels from best to deeper, as an immutable sequence.
        checksum: Integrity checksum supplied by the venue, if known.
    """

    symbol: Symbol
    timestamp: Timestamp
    bids: tuple[OrderBookLevel, ...]
    asks: tuple[OrderBookLevel, ...]
    checksum: str | None = None


@dataclass(frozen=True, slots=True)
class Ticker:
    """Canonical top-of-book and last-trade ticker snapshot.

    Attributes:
        symbol: CQROS instrument symbol (for example ``BTCUSDT``).
        timestamp: Ticker timestamp (UTC).
        last_price: Last traded price in quote-asset units.
        bid_price: Best bid price in quote-asset units.
        ask_price: Best ask price in quote-asset units.
        bid_quantity: Best bid quantity in base-asset units, if known.
        ask_quantity: Best ask quantity in base-asset units, if known.
        open_price: Session or rolling-window open price, if known.
        high_price: Session or rolling-window high price, if known.
        low_price: Session or rolling-window low price, if known.
        volume: Base-asset volume over the ticker window, if known.
        quote_volume: Quote-asset volume over the ticker window, if known.
        mark_price: Mark price for derivatives, if known.
        index_price: Index price for derivatives, if known.
    """

    symbol: Symbol
    timestamp: Timestamp
    last_price: Price
    bid_price: Price
    ask_price: Price
    bid_quantity: Quantity | None = None
    ask_quantity: Quantity | None = None
    open_price: Price | None = None
    high_price: Price | None = None
    low_price: Price | None = None
    volume: Volume | None = None
    quote_volume: Volume | None = None
    mark_price: Price | None = None
    index_price: Price | None = None


@dataclass(frozen=True, slots=True)
class KlineMetadata:
    """Provenance and completeness metadata for a kline / OHLCV bar.

    Attributes:
        symbol: CQROS instrument symbol (for example ``BTCUSDT``).
        timeframe: Bar interval identifier (for example ``1m``).
        open_time: Bar open timestamp (UTC).
        close_time: Bar close timestamp (UTC).
        is_closed: Whether the bar is finalized and will not change.
        exchange: Exchange identifier, if known.
        first_trade_id: Identifier of the first trade in the bar, if known.
        last_trade_id: Identifier of the last trade in the bar, if known.
        source: Origin of the kline data, if known.
    """

    symbol: Symbol
    timeframe: Timeframe
    open_time: Timestamp
    close_time: Timestamp
    is_closed: bool
    exchange: Exchange | None = None
    first_trade_id: Id | None = None
    last_trade_id: Id | None = None
    source: DataSource | None = None


@dataclass(frozen=True, slots=True)
class DatasetDescriptor:
    """Canonical descriptor for a market-storage dataset artifact.

    Describes identity, coverage, integrity, and storage attributes of a
    market dataset. Research reproducibility and lineage-oriented metadata
    use ``cqros.data.metadata.DatasetMetadata``.

    Attributes:
        dataset_id: Stable dataset identifier.
        version: Dataset version string.
        created_at: Creation timestamp (UTC).
        exchange: Exchange identifier associated with the dataset.
        symbols: Immutable sequence of instrument symbols covered.
        intervals: Immutable sequence of timeframes / intervals covered.
        rows: Number of rows in the dataset.
        columns: Immutable sequence of column names.
        checksum: Content checksum of the dataset artifact.
        lineage: Structured lineage payload, if recorded.
        storage_location: Storage path or URI, if recorded.
        compression: Compression codec used for the artifact, if recorded.
        quality_score: Dataset quality score, if computed.
        validation_report: Identifier or reference for a validation report,
            if recorded.
    """

    dataset_id: Id
    version: str
    created_at: Timestamp
    exchange: Exchange
    symbols: tuple[Symbol, ...]
    intervals: tuple[Timeframe, ...]
    rows: int
    columns: tuple[str, ...]
    checksum: str
    lineage: Metadata | None = None
    storage_location: FilePath | None = None
    compression: CompressionCodec | None = None
    quality_score: float | None = None
    validation_report: Id | None = None


# Backward-compatible alias. Prefer ``DatasetDescriptor``.
DatasetMetadata = DatasetDescriptor
