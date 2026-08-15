"""Unit tests for CQROS canonical market data schema models."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from cqros.core.enums import DataSource, OrderSide
from cqros.data.schemas import (
    OHLCV,
    DatasetDescriptor,
    DatasetMetadata,
    FundingRate,
    KlineMetadata,
    Liquidation,
    OpenInterest,
    OrderBookLevel,
    OrderBookSnapshot,
    Ticker,
    Trade,
)

_TS = datetime(2026, 7, 25, 12, 0, tzinfo=UTC)
_TS_CLOSE = datetime(2026, 7, 25, 12, 1, tzinfo=UTC)

_SCHEMA_TYPES: tuple[type[object], ...] = (
    OHLCV,
    Trade,
    FundingRate,
    OpenInterest,
    Liquidation,
    OrderBookLevel,
    OrderBookSnapshot,
    Ticker,
    KlineMetadata,
    DatasetDescriptor,
)


def _sample_ohlcv(**overrides: object) -> OHLCV:
    values: dict[str, object] = {
        "symbol": "BTCUSDT",
        "timeframe": "1m",
        "open_time": _TS,
        "close_time": _TS_CLOSE,
        "open": 100.0,
        "high": 110.0,
        "low": 95.0,
        "close": 105.0,
        "volume": 12.5,
        "quote_volume": 1_300.0,
        "trade_count": 42,
    }
    values.update(overrides)
    return OHLCV(**values)  # type: ignore[arg-type]


def _sample_trade(**overrides: object) -> Trade:
    values: dict[str, object] = {
        "trade_id": "123456",
        "symbol": "BTCUSDT",
        "timestamp": _TS,
        "side": OrderSide.BUY,
        "price": 105.0,
        "quantity": 0.01,
        "buyer_maker": False,
    }
    values.update(overrides)
    return Trade(**values)  # type: ignore[arg-type]


def _sample_funding_rate(**overrides: object) -> FundingRate:
    values: dict[str, object] = {
        "symbol": "BTCUSDT",
        "funding_time": _TS,
        "funding_rate": 0.0001,
        "mark_price": 105.0,
    }
    values.update(overrides)
    return FundingRate(**values)  # type: ignore[arg-type]


def _sample_open_interest(**overrides: object) -> OpenInterest:
    values: dict[str, object] = {
        "symbol": "BTCUSDT",
        "timestamp": _TS,
        "open_interest": 10_000.0,
        "value": 1_050_000.0,
    }
    values.update(overrides)
    return OpenInterest(**values)  # type: ignore[arg-type]


def _sample_liquidation(**overrides: object) -> Liquidation:
    values: dict[str, object] = {
        "symbol": "BTCUSDT",
        "timestamp": _TS,
        "side": OrderSide.SELL,
        "price": 95.0,
        "quantity": 1.5,
        "average_price": 94.5,
        "order_id": "liq-1",
    }
    values.update(overrides)
    return Liquidation(**values)  # type: ignore[arg-type]


def _sample_order_book_level(**overrides: object) -> OrderBookLevel:
    values: dict[str, object] = {
        "price": 100.0,
        "quantity": 2.0,
        "order_count": 3,
    }
    values.update(overrides)
    return OrderBookLevel(**values)  # type: ignore[arg-type]


def _sample_order_book_snapshot(**overrides: object) -> OrderBookSnapshot:
    values: dict[str, object] = {
        "symbol": "BTCUSDT",
        "timestamp": _TS,
        "bids": (
            OrderBookLevel(price=104.0, quantity=1.0),
            OrderBookLevel(price=103.0, quantity=2.0),
        ),
        "asks": (
            OrderBookLevel(price=105.0, quantity=1.5),
            OrderBookLevel(price=106.0, quantity=2.5),
        ),
        "checksum": "abc123",
    }
    values.update(overrides)
    return OrderBookSnapshot(**values)  # type: ignore[arg-type]


def _sample_ticker(**overrides: object) -> Ticker:
    values: dict[str, object] = {
        "symbol": "BTCUSDT",
        "timestamp": _TS,
        "last_price": 105.0,
        "bid_price": 104.9,
        "ask_price": 105.1,
        "bid_quantity": 1.0,
        "ask_quantity": 1.2,
        "open_price": 100.0,
        "high_price": 110.0,
        "low_price": 95.0,
        "volume": 50.0,
        "quote_volume": 5_250.0,
        "mark_price": 105.05,
        "index_price": 105.02,
    }
    values.update(overrides)
    return Ticker(**values)  # type: ignore[arg-type]


def _sample_kline_metadata(**overrides: object) -> KlineMetadata:
    values: dict[str, object] = {
        "symbol": "BTCUSDT",
        "timeframe": "1m",
        "open_time": _TS,
        "close_time": _TS_CLOSE,
        "is_closed": True,
        "exchange": "binance",
        "first_trade_id": "100",
        "last_trade_id": "140",
        "source": DataSource.EXCHANGE_REST,
    }
    values.update(overrides)
    return KlineMetadata(**values)  # type: ignore[arg-type]


def _sample_dataset_descriptor(**overrides: object) -> DatasetDescriptor:
    values: dict[str, object] = {
        "dataset_id": "ds-btc-1m",
        "version": "1.0.0",
        "created_at": _TS,
        "exchange": "binance",
        "symbols": ("BTCUSDT", "ETHUSDT"),
        "intervals": ("1m", "5m"),
        "rows": 1_000,
        "columns": ("open_time", "open", "high", "low", "close", "volume"),
        "checksum": "sha256:deadbeef",
        "lineage": {"parents": ["raw-ohlcv-v1"]},
        "storage_location": Path("data/processed/ds-btc-1m"),
        "compression": "zstd",
        "quality_score": 0.99,
        "validation_report": "vr-001",
    }
    values.update(overrides)
    return DatasetDescriptor(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("schema_type", _SCHEMA_TYPES)
def test_schema_types_are_frozen_slotted_dataclasses(schema_type: type[object]) -> None:
    """Schema models are immutable slotted dataclasses."""
    assert is_dataclass(schema_type)
    assert hasattr(schema_type, "__slots__")


def test_ohlcv_required_fields_and_defaults() -> None:
    """OHLCV requires bar identity and prices; optional aggregates default to None."""
    bar = OHLCV(
        symbol="ETHUSDT",
        timeframe="5m",
        open_time=_TS,
        close_time=_TS_CLOSE,
        open=1.0,
        high=2.0,
        low=0.5,
        close=1.5,
        volume=10.0,
    )
    assert bar.quote_volume is None
    assert bar.trade_count is None
    assert bar.symbol == "ETHUSDT"
    assert bar.close == pytest.approx(1.5)


def test_ohlcv_is_frozen() -> None:
    """OHLCV instances reject attribute mutation."""
    bar = _sample_ohlcv()
    with pytest.raises(FrozenInstanceError):
        bar.close = 200.0  # type: ignore[misc]


def test_ohlcv_equality_and_hash() -> None:
    """Equal OHLCV values compare equal and are hashable."""
    left = _sample_ohlcv()
    right = _sample_ohlcv()
    different = _sample_ohlcv(close=200.0)

    assert left == right
    assert hash(left) == hash(right)
    assert left != different
    assert {left, right, different} == {left, different}


def test_ohlcv_field_names_are_stable() -> None:
    """OHLCV public field names remain stable for serialization consumers."""
    names = tuple(field.name for field in fields(OHLCV))
    assert names == (
        "symbol",
        "timeframe",
        "open_time",
        "close_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trade_count",
    )


def test_trade_required_fields_and_defaults() -> None:
    """Trade requires identity and economics; buyer_maker defaults to None."""
    trade = Trade(
        trade_id="1",
        symbol="BTCUSDT",
        timestamp=_TS,
        side=OrderSide.SELL,
        price=100.0,
        quantity=0.5,
    )
    assert trade.buyer_maker is None
    assert trade.side is OrderSide.SELL


def test_trade_is_frozen_and_hashable() -> None:
    """Trade instances are immutable and hashable."""
    trade = _sample_trade()
    with pytest.raises(FrozenInstanceError):
        trade.price = 1.0  # type: ignore[misc]
    assert hash(trade) == hash(_sample_trade())


def test_trade_field_names_are_stable() -> None:
    """Trade public field names remain stable for serialization consumers."""
    names = tuple(field.name for field in fields(Trade))
    assert names == (
        "trade_id",
        "symbol",
        "timestamp",
        "side",
        "price",
        "quantity",
        "buyer_maker",
    )


def test_funding_rate_defaults_and_frozen() -> None:
    """FundingRate mark_price is optional and instances are frozen."""
    funding = FundingRate(
        symbol="BTCUSDT",
        funding_time=_TS,
        funding_rate=0.0002,
    )
    assert funding.mark_price is None
    assert _sample_funding_rate().mark_price == pytest.approx(105.0)
    with pytest.raises(FrozenInstanceError):
        funding.funding_rate = 0.0  # type: ignore[misc]


def test_funding_rate_field_names_are_stable() -> None:
    """FundingRate public field names remain stable for serialization consumers."""
    names = tuple(field.name for field in fields(FundingRate))
    assert names == (
        "symbol",
        "funding_time",
        "funding_rate",
        "mark_price",
    )


def test_open_interest_defaults_and_equality() -> None:
    """OpenInterest value is optional; equal instances compare equal."""
    observation = OpenInterest(
        symbol="BTCUSDT",
        timestamp=_TS,
        open_interest=500.0,
    )
    assert observation.value is None
    assert _sample_open_interest() == _sample_open_interest()


def test_open_interest_field_names_are_stable() -> None:
    """OpenInterest public field names remain stable for serialization consumers."""
    names = tuple(field.name for field in fields(OpenInterest))
    assert names == (
        "symbol",
        "timestamp",
        "open_interest",
        "value",
    )


def test_liquidation_defaults_and_frozen() -> None:
    """Liquidation optional identifiers default to None and instances are frozen."""
    liquidation = Liquidation(
        symbol="BTCUSDT",
        timestamp=_TS,
        side=OrderSide.BUY,
        price=90.0,
        quantity=2.0,
    )
    assert liquidation.average_price is None
    assert liquidation.order_id is None
    assert _sample_liquidation().order_id == "liq-1"
    with pytest.raises(FrozenInstanceError):
        liquidation.quantity = 1.0  # type: ignore[misc]


def test_liquidation_field_names_are_stable() -> None:
    """Liquidation public field names remain stable for serialization consumers."""
    names = tuple(field.name for field in fields(Liquidation))
    assert names == (
        "symbol",
        "timestamp",
        "side",
        "price",
        "quantity",
        "average_price",
        "order_id",
    )


def test_order_book_level_defaults() -> None:
    """OrderBookLevel order_count is optional."""
    level = OrderBookLevel(price=100.0, quantity=5.0)
    assert level.order_count is None
    assert _sample_order_book_level().order_count == 3


def test_order_book_snapshot_uses_tuples() -> None:
    """OrderBookSnapshot stores bids and asks as immutable tuples."""
    snapshot = _sample_order_book_snapshot()
    assert isinstance(snapshot.bids, tuple)
    assert isinstance(snapshot.asks, tuple)
    assert snapshot.bids[0].price == pytest.approx(104.0)
    assert snapshot.asks[0].price == pytest.approx(105.0)


def test_order_book_snapshot_is_frozen_and_hashable() -> None:
    """OrderBookSnapshot instances are immutable and hashable."""
    snapshot = _sample_order_book_snapshot()
    with pytest.raises(FrozenInstanceError):
        snapshot.checksum = "other"  # type: ignore[misc]
    assert hash(snapshot) == hash(_sample_order_book_snapshot())


def test_order_book_snapshot_field_names_are_stable() -> None:
    """OrderBookSnapshot public field names remain stable for serialization consumers."""
    names = tuple(field.name for field in fields(OrderBookSnapshot))
    assert names == (
        "symbol",
        "timestamp",
        "bids",
        "asks",
        "checksum",
    )


def test_ticker_required_fields_and_defaults() -> None:
    """Ticker requires last and top-of-book; optional window fields default to None."""
    ticker = Ticker(
        symbol="BTCUSDT",
        timestamp=_TS,
        last_price=100.0,
        bid_price=99.9,
        ask_price=100.1,
    )
    assert ticker.bid_quantity is None
    assert ticker.ask_quantity is None
    assert ticker.open_price is None
    assert ticker.high_price is None
    assert ticker.low_price is None
    assert ticker.volume is None
    assert ticker.quote_volume is None
    assert ticker.mark_price is None
    assert ticker.index_price is None


def test_ticker_is_frozen() -> None:
    """Ticker instances reject attribute mutation."""
    ticker = _sample_ticker()
    with pytest.raises(FrozenInstanceError):
        ticker.last_price = 1.0  # type: ignore[misc]


def test_ticker_field_names_are_stable() -> None:
    """Ticker public field names remain stable for serialization consumers."""
    names = tuple(field.name for field in fields(Ticker))
    assert names == (
        "symbol",
        "timestamp",
        "last_price",
        "bid_price",
        "ask_price",
        "bid_quantity",
        "ask_quantity",
        "open_price",
        "high_price",
        "low_price",
        "volume",
        "quote_volume",
        "mark_price",
        "index_price",
    )


def test_kline_metadata_defaults_and_frozen() -> None:
    """KlineMetadata optional provenance fields default to None."""
    metadata = KlineMetadata(
        symbol="BTCUSDT",
        timeframe="1h",
        open_time=_TS,
        close_time=_TS_CLOSE,
        is_closed=False,
    )
    assert metadata.exchange is None
    assert metadata.first_trade_id is None
    assert metadata.last_trade_id is None
    assert metadata.source is None
    assert _sample_kline_metadata().source is DataSource.EXCHANGE_REST
    with pytest.raises(FrozenInstanceError):
        metadata.is_closed = True  # type: ignore[misc]


def test_kline_metadata_field_names_are_stable() -> None:
    """KlineMetadata public field names remain stable for serialization consumers."""
    names = tuple(field.name for field in fields(KlineMetadata))
    assert names == (
        "symbol",
        "timeframe",
        "open_time",
        "close_time",
        "is_closed",
        "exchange",
        "first_trade_id",
        "last_trade_id",
        "source",
    )


def test_dataset_descriptor_uses_tuples_for_collections() -> None:
    """DatasetDescriptor stores symbols, intervals, and columns as tuples."""
    descriptor = _sample_dataset_descriptor()
    assert isinstance(descriptor.symbols, tuple)
    assert isinstance(descriptor.intervals, tuple)
    assert isinstance(descriptor.columns, tuple)
    assert descriptor.symbols == ("BTCUSDT", "ETHUSDT")
    assert descriptor.compression == "zstd"


def test_dataset_descriptor_optional_defaults() -> None:
    """DatasetDescriptor optional lineage and quality fields default to None."""
    descriptor = DatasetDescriptor(
        dataset_id="ds-1",
        version="0.1.0",
        created_at=_TS,
        exchange="binance",
        symbols=("BTCUSDT",),
        intervals=("1m",),
        rows=10,
        columns=("open", "close"),
        checksum="sha256:abc",
    )
    assert descriptor.lineage is None
    assert descriptor.storage_location is None
    assert descriptor.compression is None
    assert descriptor.quality_score is None
    assert descriptor.validation_report is None


def test_dataset_descriptor_is_frozen() -> None:
    """DatasetDescriptor instances reject attribute mutation."""
    descriptor = _sample_dataset_descriptor()
    with pytest.raises(FrozenInstanceError):
        descriptor.rows = 0  # type: ignore[misc]


def test_dataset_descriptor_field_names_are_stable() -> None:
    """DatasetDescriptor public field names remain stable for serialization consumers."""
    names = tuple(field.name for field in fields(DatasetDescriptor))
    assert names == (
        "dataset_id",
        "version",
        "created_at",
        "exchange",
        "symbols",
        "intervals",
        "rows",
        "columns",
        "checksum",
        "lineage",
        "storage_location",
        "compression",
        "quality_score",
        "validation_report",
    )


def test_dataset_metadata_alias_points_to_descriptor() -> None:
    """DatasetMetadata remains a backward-compatible alias for DatasetDescriptor."""
    assert DatasetMetadata is DatasetDescriptor
    assert DatasetMetadata.__name__ == "DatasetDescriptor"


def test_package_exports_schema_models() -> None:
    """The data package re-exports the schema public API."""
    import cqros.data as data_package
    from cqros.data.metadata import DatasetMetadata as ResearchDatasetMetadata

    for name in (
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
    ):
        assert name in data_package.__all__
        assert hasattr(data_package, name)

    assert data_package.OHLCV is OHLCV
    assert data_package.Trade is Trade
    assert data_package.OrderBookSnapshot is OrderBookSnapshot
    assert data_package.DatasetDescriptor is DatasetDescriptor
    # Package DatasetMetadata is the research metadata model.
    assert data_package.DatasetMetadata is ResearchDatasetMetadata
    assert data_package.DatasetMetadata is not DatasetDescriptor
