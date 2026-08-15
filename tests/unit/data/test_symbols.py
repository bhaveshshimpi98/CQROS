"""Unit tests for CQROS symbol and research universe data models."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass
from datetime import UTC, datetime
from enum import StrEnum

import pytest

from cqros.data.symbols import (
    SymbolCategory,
    SymbolFilter,
    SymbolInfo,
    SymbolStatistics,
    SymbolStatus,
    UniverseSnapshot,
)

_ENUM_TYPES: tuple[type[StrEnum], ...] = (
    SymbolStatus,
    SymbolCategory,
)

_EXPECTED_VALUES: dict[type[StrEnum], dict[str, str]] = {
    SymbolStatus: {
        "PENDING": "pending",
        "TRADING": "trading",
        "HALTED": "halted",
        "DELISTED": "delisted",
        "EXPIRED": "expired",
    },
    SymbolCategory: {
        "SPOT": "spot",
        "PERPETUAL": "perpetual",
        "FUTURES": "futures",
        "OPTIONS": "options",
        "INDEX": "index",
        "OTHER": "other",
    },
}


def _sample_symbol_info(**overrides: object) -> SymbolInfo:
    values: dict[str, object] = {
        "symbol": "BTCUSDT",
        "exchange": "binance",
        "base_asset": "BTC",
        "quote_asset": "USDT",
        "status": SymbolStatus.TRADING,
        "category": SymbolCategory.PERPETUAL,
        "market": "usdt_perpetual",
        "native_symbol": "BTCUSDT",
        "listed_at": datetime(2020, 1, 1, tzinfo=UTC),
        "delisted_at": None,
        "updated_at": datetime(2026, 7, 25, tzinfo=UTC),
        "metadata": None,
    }
    values.update(overrides)
    return SymbolInfo(**values)  # type: ignore[arg-type]


def _sample_symbol_filter(**overrides: object) -> SymbolFilter:
    values: dict[str, object] = {
        "exchanges": ("binance", "bybit"),
        "symbols": ("BTCUSDT", "ETHUSDT"),
        "base_assets": ("BTC", "ETH"),
        "quote_assets": ("USDT",),
        "statuses": (SymbolStatus.TRADING,),
        "categories": (SymbolCategory.PERPETUAL, SymbolCategory.SPOT),
        "markets": ("usdt_perpetual",),
        "listed_after": datetime(2019, 1, 1, tzinfo=UTC),
        "listed_before": datetime(2026, 1, 1, tzinfo=UTC),
        "include_delisted": False,
        "min_quote_volume": 1_000_000.0,
        "max_symbols": 50,
    }
    values.update(overrides)
    return SymbolFilter(**values)  # type: ignore[arg-type]


def _sample_symbol_statistics(**overrides: object) -> SymbolStatistics:
    values: dict[str, object] = {
        "symbol": "BTCUSDT",
        "exchange": "binance",
        "as_of": datetime(2026, 7, 25, tzinfo=UTC),
        "window": "1d",
        "quote_volume": 2_500_000_000.0,
        "base_volume": 25_000.0,
        "trade_count": 1_000_000,
        "last_price": 100_000.0,
        "volatility": 0.02,
        "open_interest": 50_000.0,
    }
    values.update(overrides)
    return SymbolStatistics(**values)  # type: ignore[arg-type]


def _sample_universe_snapshot(**overrides: object) -> UniverseSnapshot:
    symbol = _sample_symbol_info()
    values: dict[str, object] = {
        "snapshot_id": "universe-2026-07-25",
        "version": "1.0.0",
        "created_at": datetime(2026, 7, 25, 12, 0, tzinfo=UTC),
        "as_of": datetime(2026, 7, 25, 0, 0, tzinfo=UTC),
        "symbols": (symbol,),
        "checksum": "abc123",
        "exchanges": ("binance",),
        "name": "binance_usdt_perp_core",
        "description": "Core USDT-M perpetual universe",
        "symbol_filter": _sample_symbol_filter(),
        "statistics": (_sample_symbol_statistics(),),
        "metadata": None,
    }
    values.update(overrides)
    return UniverseSnapshot(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("enum_type", _ENUM_TYPES)
def test_enums_are_str_enums(enum_type: type[StrEnum]) -> None:
    """Symbol enumerations are serializable string enums."""
    assert issubclass(enum_type, StrEnum)
    for member in enum_type:
        assert isinstance(member.value, str)
        assert str(member) == member.value


@pytest.mark.parametrize("enum_type", _ENUM_TYPES)
def test_enum_member_values(enum_type: type[StrEnum]) -> None:
    """Enumeration members expose the documented string values."""
    expected = _EXPECTED_VALUES[enum_type]
    assert {member.name: member.value for member in enum_type} == expected


def test_symbol_info_required_fields_and_defaults() -> None:
    """SymbolInfo requires identity fields; optional fields default to None."""
    info = SymbolInfo(
        symbol="ETHUSDT",
        exchange="binance",
        base_asset="ETH",
        quote_asset="USDT",
        status=SymbolStatus.TRADING,
        category=SymbolCategory.SPOT,
    )

    assert info.symbol == "ETHUSDT"
    assert info.exchange == "binance"
    assert info.base_asset == "ETH"
    assert info.quote_asset == "USDT"
    assert info.status is SymbolStatus.TRADING
    assert info.category is SymbolCategory.SPOT
    assert info.market is None
    assert info.native_symbol is None
    assert info.listed_at is None
    assert info.delisted_at is None
    assert info.updated_at is None
    assert info.metadata is None


def test_symbol_info_is_frozen_and_slotted() -> None:
    """SymbolInfo instances are immutable and use slots."""
    info = _sample_symbol_info()
    assert is_dataclass(info)
    assert hasattr(type(info), "__slots__")
    with pytest.raises(FrozenInstanceError):
        info.status = SymbolStatus.HALTED  # type: ignore[misc]


def test_symbol_info_equality_and_hash() -> None:
    """Equal SymbolInfo values compare equal and are hashable."""
    left = _sample_symbol_info()
    right = _sample_symbol_info()
    different = _sample_symbol_info(status=SymbolStatus.HALTED)

    assert left == right
    assert hash(left) == hash(right)
    assert left != different
    assert {left, right, different} == {left, different}


def test_symbol_info_accepts_metadata_mapping() -> None:
    """SymbolInfo stores optional metadata without interpreting it."""
    info = _sample_symbol_info(metadata={"venue_id": "BTCUSDT"})
    assert info.metadata == {"venue_id": "BTCUSDT"}


def test_symbol_info_field_names_are_stable() -> None:
    """SymbolInfo public field names remain stable for serialization consumers."""
    names = tuple(field.name for field in fields(SymbolInfo))
    assert names == (
        "symbol",
        "exchange",
        "base_asset",
        "quote_asset",
        "status",
        "category",
        "market",
        "native_symbol",
        "listed_at",
        "delisted_at",
        "updated_at",
        "metadata",
    )


def test_symbol_filter_defaults_are_unconstrained() -> None:
    """SymbolFilter defaults leave all criteria dimensions unconstrained."""
    symbol_filter = SymbolFilter()

    assert symbol_filter.exchanges is None
    assert symbol_filter.symbols is None
    assert symbol_filter.base_assets is None
    assert symbol_filter.quote_assets is None
    assert symbol_filter.statuses is None
    assert symbol_filter.categories is None
    assert symbol_filter.markets is None
    assert symbol_filter.listed_after is None
    assert symbol_filter.listed_before is None
    assert symbol_filter.include_delisted is False
    assert symbol_filter.min_quote_volume is None
    assert symbol_filter.max_symbols is None


def test_symbol_filter_is_frozen_and_slotted() -> None:
    """SymbolFilter instances are immutable and use slots."""
    symbol_filter = _sample_symbol_filter()
    assert is_dataclass(symbol_filter)
    assert hasattr(type(symbol_filter), "__slots__")
    with pytest.raises(FrozenInstanceError):
        symbol_filter.max_symbols = 10  # type: ignore[misc]


def test_symbol_filter_stores_criteria_as_tuples() -> None:
    """SymbolFilter collection criteria are stored as tuples."""
    symbol_filter = _sample_symbol_filter()
    assert isinstance(symbol_filter.exchanges, tuple)
    assert isinstance(symbol_filter.symbols, tuple)
    assert isinstance(symbol_filter.statuses, tuple)
    assert isinstance(symbol_filter.categories, tuple)


def test_symbol_statistics_required_fields_and_defaults() -> None:
    """SymbolStatistics requires identity and as_of; metrics default to None."""
    stats = SymbolStatistics(
        symbol="BTCUSDT",
        exchange="binance",
        as_of=datetime(2026, 7, 25, tzinfo=UTC),
    )

    assert stats.symbol == "BTCUSDT"
    assert stats.exchange == "binance"
    assert stats.as_of == datetime(2026, 7, 25, tzinfo=UTC)
    assert stats.window is None
    assert stats.quote_volume is None
    assert stats.base_volume is None
    assert stats.trade_count is None
    assert stats.last_price is None
    assert stats.volatility is None
    assert stats.open_interest is None


def test_symbol_statistics_is_frozen_and_slotted() -> None:
    """SymbolStatistics instances are immutable and use slots."""
    stats = _sample_symbol_statistics()
    assert is_dataclass(stats)
    assert hasattr(type(stats), "__slots__")
    with pytest.raises(FrozenInstanceError):
        stats.trade_count = 0  # type: ignore[misc]


def test_universe_snapshot_required_fields_and_defaults() -> None:
    """UniverseSnapshot requires identity and membership; optionals default."""
    symbol = _sample_symbol_info()
    snapshot = UniverseSnapshot(
        snapshot_id="snap-1",
        version="1.0.0",
        created_at=datetime(2026, 7, 25, tzinfo=UTC),
        as_of=datetime(2026, 7, 24, tzinfo=UTC),
        symbols=(symbol,),
        checksum="deadbeef",
    )

    assert snapshot.snapshot_id == "snap-1"
    assert snapshot.version == "1.0.0"
    assert snapshot.symbols == (symbol,)
    assert snapshot.checksum == "deadbeef"
    assert snapshot.exchanges is None
    assert snapshot.name is None
    assert snapshot.description is None
    assert snapshot.symbol_filter is None
    assert snapshot.statistics is None
    assert snapshot.metadata is None


def test_universe_snapshot_is_frozen_and_slotted() -> None:
    """UniverseSnapshot instances are immutable and use slots."""
    snapshot = _sample_universe_snapshot()
    assert is_dataclass(snapshot)
    assert hasattr(type(snapshot), "__slots__")
    with pytest.raises(FrozenInstanceError):
        snapshot.version = "2.0.0"  # type: ignore[misc]


def test_universe_snapshot_supports_multi_exchange() -> None:
    """UniverseSnapshot can hold symbols from multiple exchanges."""
    binance = _sample_symbol_info(exchange="binance", symbol="BTCUSDT")
    bybit = _sample_symbol_info(
        exchange="bybit",
        symbol="BTCUSDT",
        native_symbol="BTCUSDT",
        market="linear",
    )
    snapshot = _sample_universe_snapshot(
        symbols=(binance, bybit),
        exchanges=("binance", "bybit"),
    )

    assert len(snapshot.symbols) == 2
    assert snapshot.exchanges == ("binance", "bybit")
    assert {entry.exchange for entry in snapshot.symbols} == {"binance", "bybit"}


def test_universe_snapshot_equality_and_hash() -> None:
    """Equal UniverseSnapshot values compare equal and are hashable."""
    left = _sample_universe_snapshot()
    right = _sample_universe_snapshot()
    different = _sample_universe_snapshot(checksum="different")

    assert left == right
    assert hash(left) == hash(right)
    assert left != different
    assert {left, right, different} == {left, different}


def test_universe_snapshot_field_names_are_stable() -> None:
    """UniverseSnapshot public field names remain stable for consumers."""
    names = tuple(field.name for field in fields(UniverseSnapshot))
    assert names == (
        "snapshot_id",
        "version",
        "created_at",
        "as_of",
        "symbols",
        "checksum",
        "exchanges",
        "name",
        "description",
        "symbol_filter",
        "statistics",
        "metadata",
    )


def test_package_exports_symbol_models() -> None:
    """The data package re-exports the symbol public API."""
    import cqros.data as data_package

    assert data_package.SymbolStatus is SymbolStatus
    assert data_package.SymbolCategory is SymbolCategory
    assert data_package.SymbolInfo is SymbolInfo
    assert data_package.UniverseSnapshot is UniverseSnapshot
    assert data_package.SymbolFilter is SymbolFilter
    assert data_package.SymbolStatistics is SymbolStatistics
    for name in (
        "SymbolStatus",
        "SymbolCategory",
        "SymbolInfo",
        "UniverseSnapshot",
        "SymbolFilter",
        "SymbolStatistics",
    ):
        assert name in data_package.__all__
