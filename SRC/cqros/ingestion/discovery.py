"""Binance USDⓈ-M Futures symbol discovery.

Purpose:
    Discover tradeable USDT-margined perpetual futures contracts from
    Binance exchange information and map them into immutable CQROS
    ``Contract`` value objects.

Responsibilities:
    - Fetch exchange information through ``BinanceClient``
    - Select USDT perpetual futures with trading status ``TRADING``
    - Parse venue filters into CQROS filter value objects
    - Return an immutable tuple of ``Contract`` instances

Dependencies:
    ``cqros.core.constants``, ``cqros.core.exceptions``,
    ``cqros.data.contracts``, and ``cqros.ingestion.client``.

Public API:
    ``SymbolDiscovery``.

Notes:
    Parsing helpers are intentionally private so additional exchanges or
    fields can be introduced without changing the public discovery API.
    This module performs no file I/O and does not apply research or
    trading business logic.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Final, cast

from cqros.core.constants import (
    DEFAULT_QUOTE_ASSET,
    EXCHANGE_BINANCE,
    MILLISECONDS_PER_SECOND,
)
from cqros.core.exceptions import ValidationError
from cqros.data.contracts import (
    Contract,
    ContractStatus,
    ContractType,
    NotionalFilter,
    PriceFilter,
    QuantityFilter,
)
from cqros.ingestion.client import BinanceClient

__all__ = [
    "SymbolDiscovery",
]

_BINANCE_CONTRACT_TYPE_PERPETUAL: Final[str] = "PERPETUAL"
_BINANCE_STATUS_TRADING: Final[str] = "TRADING"
_FILTER_PRICE: Final[str] = "PRICE_FILTER"
_FILTER_LOT_SIZE: Final[str] = "LOT_SIZE"
_FILTER_MIN_NOTIONAL: Final[str] = "MIN_NOTIONAL"

_logger = logging.getLogger(__name__)


class SymbolDiscovery:
    """Discover Binance USDT-M perpetual futures contracts.

    Args:
        client: Open ``BinanceClient`` used to fetch exchange information.
        logger: Optional logger instance. Defaults to the module logger.
    """

    __slots__ = ("_client", "_logger")

    def __init__(
        self,
        client: BinanceClient,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize discovery with an injected Binance client.

        Args:
            client: Binance USDⓈ-M Futures REST client.
            logger: Optional logger instance.
        """
        self._client = client
        self._logger = logger if logger is not None else _logger

    async def discover(self) -> tuple[Contract, ...]:
        """Fetch and parse tradeable USDT perpetual futures contracts.

        Returns:
            Immutable tuple of ``Contract`` objects for symbols that are
            USDT-quoted perpetual futures currently in ``TRADING`` status.

        Raises:
            ExchangeError: Propagated from ``BinanceClient`` transport failures.
            ValidationError: If the exchange payload cannot be parsed into
                contracts.
        """
        self._logger.debug(
            "Discovering Binance USDT perpetual contracts",
            extra={"exchange": EXCHANGE_BINANCE},
        )
        payload = await self._client.get_exchange_info()
        contracts = self._parse_exchange_info(payload)
        self._logger.info(
            "Discovered Binance USDT perpetual contracts",
            extra={
                "exchange": EXCHANGE_BINANCE,
                "contract_count": len(contracts),
            },
        )
        return contracts

    def _parse_exchange_info(self, payload: object) -> tuple[Contract, ...]:
        """Parse a Binance ``exchangeInfo`` payload into contracts.

        Args:
            payload: Decoded JSON body returned by ``get_exchange_info``.

        Returns:
            Immutable tuple of supported trading contracts.

        Raises:
            ValidationError: If the payload structure is invalid.
        """
        root = _require_mapping(payload, context="exchangeInfo")
        symbols = root.get("symbols")
        if not isinstance(symbols, Sequence) or isinstance(symbols, (str, bytes)):
            raise ValidationError(
                "exchangeInfo.symbols must be a sequence",
                error_code="INGESTION-DISCOVERY-001",
                details={"field": "symbols", "type": type(symbols).__name__},
            )

        symbol_entries = cast(Sequence[object], symbols)
        contracts: list[Contract] = []
        for index, entry in enumerate(symbol_entries):
            symbol_entry = _require_mapping(
                entry,
                context=f"exchangeInfo.symbols[{index}]",
            )
            if not self._is_supported_symbol(symbol_entry):
                continue
            contracts.append(self._parse_contract(symbol_entry))
        return tuple(contracts)

    def _is_supported_symbol(self, entry: Mapping[str, object]) -> bool:
        """Return whether a symbol entry is a tradeable USDT perpetual.

        Args:
            entry: Single symbol object from ``exchangeInfo.symbols``.

        Returns:
            ``True`` when the symbol is a USDT-quoted perpetual futures
            contract with Binance status ``TRADING``.
        """
        return (
            entry.get("contractType") == _BINANCE_CONTRACT_TYPE_PERPETUAL
            and entry.get("quoteAsset") == DEFAULT_QUOTE_ASSET
            and entry.get("status") == _BINANCE_STATUS_TRADING
        )

    def _parse_contract(self, entry: Mapping[str, object]) -> Contract:
        """Build an immutable ``Contract`` from a supported symbol entry.

        Args:
            entry: Supported Binance symbol mapping.

        Returns:
            Parsed ``Contract`` value object.

        Raises:
            ValidationError: If required fields or filters are missing or
                invalid.
        """
        symbol = _require_str(entry, "symbol")
        base_asset = _require_str(entry, "baseAsset")
        quote_asset = _require_str(entry, "quoteAsset")
        margin_asset = _optional_str(entry, "marginAsset")
        filters = self._index_filters(entry.get("filters"), symbol=symbol)

        return Contract(
            symbol=symbol,
            exchange=EXCHANGE_BINANCE,
            base_asset=base_asset,
            quote_asset=quote_asset,
            contract_type=ContractType.PERPETUAL,
            status=ContractStatus.TRADING,
            price_filter=self._parse_price_filter(filters, symbol=symbol),
            quantity_filter=self._parse_quantity_filter(filters, symbol=symbol),
            notional_filter=self._parse_notional_filter(filters, symbol=symbol),
            leverage_filter=None,
            margin_asset=margin_asset,
            settlement_asset=margin_asset,
            contract_size=None,
            expiry=None,
            listed_at=self._parse_listed_at(entry.get("onboardDate"), symbol=symbol),
            updated_at=None,
        )

    def _index_filters(
        self,
        filters: object,
        *,
        symbol: str,
    ) -> dict[str, Mapping[str, object]]:
        """Index symbol filters by Binance ``filterType``.

        Args:
            filters: Raw ``filters`` array from a symbol entry.
            symbol: Symbol being parsed, for error context.

        Returns:
            Mapping of filter type name to filter payload.

        Raises:
            ValidationError: If ``filters`` is not a sequence of mappings.
        """
        if not isinstance(filters, Sequence) or isinstance(filters, (str, bytes)):
            raise ValidationError(
                "symbol filters must be a sequence",
                error_code="INGESTION-DISCOVERY-002",
                details={"symbol": symbol, "type": type(filters).__name__},
            )

        filter_entries = cast(Sequence[object], filters)
        indexed: dict[str, Mapping[str, object]] = {}
        for index, item in enumerate(filter_entries):
            filter_entry = _require_mapping(
                item,
                context=f"{symbol}.filters[{index}]",
            )
            filter_type = filter_entry.get("filterType")
            if not isinstance(filter_type, str) or not filter_type:
                raise ValidationError(
                    "symbol filter is missing filterType",
                    error_code="INGESTION-DISCOVERY-003",
                    details={"symbol": symbol, "index": index},
                )
            indexed[filter_type] = filter_entry
        return indexed

    def _parse_price_filter(
        self,
        filters: Mapping[str, Mapping[str, object]],
        *,
        symbol: str,
    ) -> PriceFilter:
        """Parse the Binance ``PRICE_FILTER`` into a ``PriceFilter``.

        Args:
            filters: Indexed symbol filters.
            symbol: Symbol being parsed, for error context.

        Returns:
            Parsed price constraints.

        Raises:
            ValidationError: If the price filter is missing or invalid.
        """
        entry = _require_filter(filters, _FILTER_PRICE, symbol=symbol)
        tick_size = _require_positive_float(entry, "tickSize", symbol=symbol)
        return PriceFilter(
            tick_size=tick_size,
            min_price=_optional_bounded_float(entry, "minPrice", symbol=symbol),
            max_price=_optional_bounded_float(entry, "maxPrice", symbol=symbol),
        )

    def _parse_quantity_filter(
        self,
        filters: Mapping[str, Mapping[str, object]],
        *,
        symbol: str,
    ) -> QuantityFilter:
        """Parse the Binance ``LOT_SIZE`` filter into a ``QuantityFilter``.

        Args:
            filters: Indexed symbol filters.
            symbol: Symbol being parsed, for error context.

        Returns:
            Parsed quantity constraints.

        Raises:
            ValidationError: If the lot-size filter is missing or invalid.
        """
        entry = _require_filter(filters, _FILTER_LOT_SIZE, symbol=symbol)
        step_size = _require_positive_float(entry, "stepSize", symbol=symbol)
        min_quantity = _require_float(entry, "minQty", symbol=symbol)
        return QuantityFilter(
            step_size=step_size,
            min_quantity=min_quantity,
            max_quantity=_optional_bounded_float(entry, "maxQty", symbol=symbol),
        )

    def _parse_notional_filter(
        self,
        filters: Mapping[str, Mapping[str, object]],
        *,
        symbol: str,
    ) -> NotionalFilter | None:
        """Parse the Binance ``MIN_NOTIONAL`` filter when present.

        Args:
            filters: Indexed symbol filters.
            symbol: Symbol being parsed, for error context.

        Returns:
            Parsed notional constraints, or ``None`` when absent.

        Raises:
            ValidationError: If the notional filter is present but invalid.
        """
        entry = filters.get(_FILTER_MIN_NOTIONAL)
        if entry is None:
            return None
        min_notional = _require_float(entry, "notional", symbol=symbol)
        return NotionalFilter(min_notional=min_notional, max_notional=None)

    def _parse_listed_at(self, value: object, *, symbol: str) -> datetime | None:
        """Parse Binance ``onboardDate`` milliseconds into a UTC timestamp.

        Args:
            value: Raw onboard date value from exchange information.
            symbol: Symbol being parsed, for error context.

        Returns:
            Timezone-aware UTC listing timestamp, or ``None`` when absent.

        Raises:
            ValidationError: If the value is present but not a valid epoch
                millisecond timestamp.
        """
        if value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValidationError(
                "onboardDate must be an integer unix timestamp in milliseconds",
                error_code="INGESTION-DISCOVERY-004",
                details={"symbol": symbol, "value": value},
            )
        if value < 0:
            raise ValidationError(
                "onboardDate must be non-negative",
                error_code="INGESTION-DISCOVERY-005",
                details={"symbol": symbol, "value": value},
            )
        return datetime.fromtimestamp(
            value / MILLISECONDS_PER_SECOND,
            tz=UTC,
        )


def _require_mapping(value: object, *, context: str) -> Mapping[str, object]:
    """Validate that a JSON value is an object mapping.

    Args:
        value: Candidate JSON value.
        context: Human-readable location used in error details.

    Returns:
        The value cast as a string-keyed mapping.

    Raises:
        ValidationError: If ``value`` is not a mapping.
    """
    if not isinstance(value, Mapping):
        raise ValidationError(
            f"{context} must be a JSON object",
            error_code="INGESTION-DISCOVERY-006",
            details={"context": context, "type": type(value).__name__},
        )
    return cast(Mapping[str, object], value)


def _require_filter(
    filters: Mapping[str, Mapping[str, object]],
    filter_type: str,
    *,
    symbol: str,
) -> Mapping[str, object]:
    """Return a required filter by type.

    Args:
        filters: Indexed symbol filters.
        filter_type: Expected Binance filter type name.
        symbol: Symbol being parsed, for error context.

    Returns:
        Filter payload for ``filter_type``.

    Raises:
        ValidationError: If the filter is missing.
    """
    entry = filters.get(filter_type)
    if entry is None:
        raise ValidationError(
            f"symbol is missing required filter {filter_type}",
            error_code="INGESTION-DISCOVERY-007",
            details={"symbol": symbol, "filter_type": filter_type},
        )
    return entry


def _require_str(entry: Mapping[str, object], field: str) -> str:
    """Return a required non-empty string field.

    Args:
        entry: Symbol or filter mapping.
        field: Field name to read.

    Returns:
        Non-empty string field value.

    Raises:
        ValidationError: If the field is missing or not a non-empty string.
    """
    value = entry.get(field)
    if not isinstance(value, str) or not value:
        raise ValidationError(
            f"{field} must be a non-empty string",
            error_code="INGESTION-DISCOVERY-008",
            details={"field": field, "value": value},
        )
    return value


def _optional_str(entry: Mapping[str, object], field: str) -> str | None:
    """Return an optional string field.

    Args:
        entry: Symbol mapping.
        field: Field name to read.

    Returns:
        String value, or ``None`` when the field is absent.

    Raises:
        ValidationError: If the field is present but not a non-empty string.
    """
    if field not in entry or entry[field] is None:
        return None
    return _require_str(entry, field)


def _require_float(entry: Mapping[str, object], field: str, *, symbol: str) -> float:
    """Parse a required numeric field as ``float``.

    Args:
        entry: Filter mapping.
        field: Field name to read.
        symbol: Symbol being parsed, for error context.

    Returns:
        Parsed floating-point value.

    Raises:
        ValidationError: If the field is missing or not numeric.
    """
    value = entry.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValidationError(
            f"{field} must be numeric",
            error_code="INGESTION-DISCOVERY-009",
            details={"symbol": symbol, "field": field, "value": value},
        )
    try:
        return float(value)
    except ValueError as exc:
        raise ValidationError(
            f"{field} must be numeric",
            error_code="INGESTION-DISCOVERY-009",
            details={"symbol": symbol, "field": field, "value": value},
        ) from exc


def _require_positive_float(
    entry: Mapping[str, object],
    field: str,
    *,
    symbol: str,
) -> float:
    """Parse a required numeric field that must be strictly positive.

    Args:
        entry: Filter mapping.
        field: Field name to read.
        symbol: Symbol being parsed, for error context.

    Returns:
        Parsed positive floating-point value.

    Raises:
        ValidationError: If the field is missing, non-numeric, or not positive.
    """
    value = _require_float(entry, field, symbol=symbol)
    if value <= 0:
        raise ValidationError(
            f"{field} must be greater than 0",
            error_code="INGESTION-DISCOVERY-010",
            details={"symbol": symbol, "field": field, "value": value},
        )
    return value


def _optional_bounded_float(
    entry: Mapping[str, object],
    field: str,
    *,
    symbol: str,
) -> float | None:
    """Parse an optional bound field, treating ``0`` as unbounded.

    Binance documents ``0`` as a disabled bound for price and quantity
    filters. CQROS represents disabled bounds as ``None``.

    Args:
        entry: Filter mapping.
        field: Field name to read.
        symbol: Symbol being parsed, for error context.

    Returns:
        Parsed bound, or ``None`` when absent or disabled.

    Raises:
        ValidationError: If the field is present but not numeric or negative.
    """
    if field not in entry or entry[field] is None:
        return None
    value = _require_float(entry, field, symbol=symbol)
    if value == 0:
        return None
    if value < 0:
        raise ValidationError(
            f"{field} must be greater than or equal to 0",
            error_code="INGESTION-DISCOVERY-011",
            details={"symbol": symbol, "field": field, "value": value},
        )
    return value
