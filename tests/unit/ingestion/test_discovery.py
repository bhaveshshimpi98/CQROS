"""Unit tests for Binance USDT-M perpetual symbol discovery."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest

from cqros.core.constants import EXCHANGE_BINANCE
from cqros.core.exceptions import ValidationError
from cqros.data.contracts import ContractStatus, ContractType
from cqros.ingestion.discovery import SymbolDiscovery


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def _price_filter(
    *,
    tick_size: str = "0.10",
    min_price: str = "0.10",
    max_price: str = "1000000",
) -> dict[str, str]:
    return {
        "filterType": "PRICE_FILTER",
        "tickSize": tick_size,
        "minPrice": min_price,
        "maxPrice": max_price,
    }


def _lot_size(
    *,
    step_size: str = "0.001",
    min_qty: str = "0.001",
    max_qty: str = "1000",
) -> dict[str, str]:
    return {
        "filterType": "LOT_SIZE",
        "stepSize": step_size,
        "minQty": min_qty,
        "maxQty": max_qty,
    }


def _min_notional(*, notional: str = "5.0") -> dict[str, str]:
    return {
        "filterType": "MIN_NOTIONAL",
        "notional": notional,
    }


def _symbol_entry(
    *,
    symbol: str = "BTCUSDT",
    contract_type: str = "PERPETUAL",
    quote_asset: str = "USDT",
    status: str = "TRADING",
    base_asset: str = "BTC",
    margin_asset: str = "USDT",
    onboard_date: int | None = 1_598_252_400_000,
    filters: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "symbol": symbol,
        "contractType": contract_type,
        "quoteAsset": quote_asset,
        "status": status,
        "baseAsset": base_asset,
        "marginAsset": margin_asset,
        "filters": (
            filters if filters is not None else [_price_filter(), _lot_size(), _min_notional()]
        ),
    }
    if onboard_date is not None:
        entry["onboardDate"] = onboard_date
    return entry


def _exchange_info(*symbols: dict[str, Any]) -> dict[str, Any]:
    return {"timezone": "UTC", "symbols": list(symbols)}


def _client_with_payload(payload: object) -> AsyncMock:
    client = AsyncMock()
    client.get_exchange_info = AsyncMock(return_value=payload)
    return client


def test_discover_returns_only_usdt_perpetual_trading_contracts() -> None:
    """Unsupported and non-trading symbols are excluded from discovery."""
    payload = _exchange_info(
        _symbol_entry(symbol="BTCUSDT"),
        _symbol_entry(symbol="ETHUSD_PERP", quote_asset="USD"),
        _symbol_entry(
            symbol="BTCUSDT_250926",
            contract_type="CURRENT_QUARTER",
        ),
        _symbol_entry(symbol="SOLUSDT", status="PENDING"),
        _symbol_entry(
            symbol="ETHUSDT",
            base_asset="ETH",
            onboard_date=1_600_000_000_000,
        ),
    )
    discovery = SymbolDiscovery(_client_with_payload(payload))

    contracts = _run(discovery.discover())

    assert len(contracts) == 2
    assert tuple(contract.symbol for contract in contracts) == ("BTCUSDT", "ETHUSDT")
    assert all(contract.exchange == EXCHANGE_BINANCE for contract in contracts)
    assert all(contract.contract_type is ContractType.PERPETUAL for contract in contracts)
    assert all(contract.status is ContractStatus.TRADING for contract in contracts)
    assert all(contract.quote_asset == "USDT" for contract in contracts)


def test_discover_maps_filters_and_listing_timestamp() -> None:
    """Price, quantity, notional filters and onboardDate are mapped correctly."""
    payload = _exchange_info(
        _symbol_entry(
            symbol="BTCUSDT",
            filters=[
                _price_filter(tick_size="0.10", min_price="0.10", max_price="100000"),
                _lot_size(step_size="0.001", min_qty="0.001", max_qty="1000"),
                _min_notional(notional="5.0"),
            ],
            onboard_date=1_598_252_400_000,
        )
    )
    discovery = SymbolDiscovery(_client_with_payload(payload))

    (contract,) = _run(discovery.discover())

    assert contract.base_asset == "BTC"
    assert contract.margin_asset == "USDT"
    assert contract.settlement_asset == "USDT"
    assert contract.leverage_filter is None
    assert contract.contract_size is None
    assert contract.expiry is None
    assert contract.updated_at is None
    assert contract.price_filter.tick_size == 0.10
    assert contract.price_filter.min_price == 0.10
    assert contract.price_filter.max_price == 100000.0
    assert contract.quantity_filter.step_size == 0.001
    assert contract.quantity_filter.min_quantity == 0.001
    assert contract.quantity_filter.max_quantity == 1000.0
    assert contract.notional_filter is not None
    assert contract.notional_filter.min_notional == 5.0
    assert contract.notional_filter.max_notional is None
    assert contract.listed_at == datetime(2020, 8, 24, 7, 0, tzinfo=UTC)


def test_discover_treats_zero_bounds_as_unbounded() -> None:
    """Binance zero price/quantity bounds map to ``None`` in CQROS filters."""
    payload = _exchange_info(
        _symbol_entry(
            filters=[
                _price_filter(min_price="0", max_price="0"),
                _lot_size(max_qty="0"),
                _min_notional(),
            ]
        )
    )
    discovery = SymbolDiscovery(_client_with_payload(payload))

    (contract,) = _run(discovery.discover())

    assert contract.price_filter.min_price is None
    assert contract.price_filter.max_price is None
    assert contract.quantity_filter.max_quantity is None


def test_discover_allows_missing_min_notional() -> None:
    """Symbols without MIN_NOTIONAL still discover with notional_filter=None."""
    payload = _exchange_info(_symbol_entry(filters=[_price_filter(), _lot_size()]))
    discovery = SymbolDiscovery(_client_with_payload(payload))

    (contract,) = _run(discovery.discover())

    assert contract.notional_filter is None


def test_discover_rejects_invalid_payload_structure() -> None:
    """Malformed exchangeInfo payloads raise ValidationError."""
    discovery = SymbolDiscovery(_client_with_payload(["not", "an", "object"]))

    with pytest.raises(ValidationError, match="JSON object"):
        _run(discovery.discover())


def test_discover_rejects_missing_required_filters() -> None:
    """Eligible symbols missing PRICE_FILTER or LOT_SIZE raise ValidationError."""
    payload = _exchange_info(_symbol_entry(filters=[_min_notional()]))
    discovery = SymbolDiscovery(_client_with_payload(payload))

    with pytest.raises(ValidationError, match="PRICE_FILTER"):
        _run(discovery.discover())


def test_discover_rejects_invalid_onboard_date() -> None:
    """Non-integer onboardDate values raise ValidationError."""
    payload = _exchange_info(_symbol_entry(onboard_date=None))
    payload["symbols"][0]["onboardDate"] = "not-a-timestamp"
    discovery = SymbolDiscovery(_client_with_payload(payload))

    with pytest.raises(ValidationError, match="onboardDate"):
        _run(discovery.discover())


def test_package_exports_symbol_discovery() -> None:
    """SymbolDiscovery is part of the ingestion package public API."""
    import cqros.ingestion as ingestion

    assert ingestion.SymbolDiscovery is SymbolDiscovery
    assert "SymbolDiscovery" in ingestion.__all__
