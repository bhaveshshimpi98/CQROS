"""Binance USDⓈ-M Futures REST HTTP client.

Purpose:
    Provide a production-ready, async-only HTTP transport for Binance
    USDⓈ-M Futures public REST endpoints used by CQROS ingestion.

Responsibilities:
    - Own or share an ``httpx.AsyncClient`` session
    - Execute GET requests with configurable timeouts and retries
    - Enforce local async token-bucket pacing and adaptive weight throttling
    - Parse Binance rate-limit response headers and IP-ban bodies
    - Translate transport and HTTP failures into CQROS exchange exceptions
    - Decode JSON response bodies without applying domain transformations

Dependencies:
    ``httpx`` and ``cqros.core.exceptions``.

Public API:
    ``BinanceClient``, ``AsyncTokenBucket``, and the default base-URL / retry
    / rate-limit constants listed in ``__all__``.

Notes:
    Endpoint helpers are thin wrappers around ``_request``. Additional
    Binance REST paths can be exposed by adding helpers that call
    ``_request`` without changing existing method signatures.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Final, Self, cast

import httpx

from cqros.core.exceptions import (
    ExchangeAuthenticationError,
    ExchangeError,
    ExchangePermissionError,
    ExchangeRateLimitError,
    ExchangeSymbolNotFoundError,
    ExchangeTimeoutError,
    ExchangeUnavailableError,
    ExchangeValidationError,
    ValidationError,
)

__all__ = [
    "DEFAULT_BINANCE_FUTURES_REST_BASE_URL",
    "DEFAULT_BINANCE_FUTURES_TESTNET_REST_BASE_URL",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_BACKOFF_FACTOR_SECONDS",
    "DEFAULT_JITTER_RATIO",
    "DEFAULT_WEIGHT_LIMIT_1M",
    "DEFAULT_WEIGHT_THROTTLE_RATIO",
    "DEFAULT_WEIGHT_WINDOW_SECONDS",
    "DEFAULT_REQUEST_WEIGHT",
    "DEFAULT_BAN_SAFETY_MARGIN_SECONDS",
    "DEFAULT_TOKEN_BUCKET_CAPACITY",
    "DEFAULT_TOKEN_BUCKET_REFILL_PER_SECOND",
    "AsyncTokenBucket",
    "BinanceClient",
]

DEFAULT_BINANCE_FUTURES_REST_BASE_URL: Final[str] = "https://fapi.binance.com"
DEFAULT_BINANCE_FUTURES_TESTNET_REST_BASE_URL: Final[str] = "https://testnet.binancefuture.com"
DEFAULT_TIMEOUT_SECONDS: Final[float] = 30.0
DEFAULT_MAX_RETRIES: Final[int] = 3
DEFAULT_BACKOFF_FACTOR_SECONDS: Final[float] = 0.5
DEFAULT_JITTER_RATIO: Final[float] = 0.25
DEFAULT_WEIGHT_LIMIT_1M: Final[int] = 2400
DEFAULT_WEIGHT_THROTTLE_RATIO: Final[float] = 0.8
DEFAULT_WEIGHT_WINDOW_SECONDS: Final[float] = 60.0
DEFAULT_REQUEST_WEIGHT: Final[int] = 1
DEFAULT_BAN_SAFETY_MARGIN_SECONDS: Final[float] = 1.0
DEFAULT_TOKEN_BUCKET_CAPACITY: Final[float] = 40.0
DEFAULT_TOKEN_BUCKET_REFILL_PER_SECOND: Final[float] = 40.0

_PATH_EXCHANGE_INFO: Final[str] = "/fapi/v1/exchangeInfo"
_PATH_KLINES: Final[str] = "/fapi/v1/klines"
_PATH_FUNDING_RATE: Final[str] = "/fapi/v1/fundingRate"
_PATH_OPEN_INTEREST: Final[str] = "/fapi/v1/openInterest"
_PATH_OPEN_INTEREST_HIST: Final[str] = "/futures/data/openInterestHist"
_PATH_TAKER_LONG_SHORT_RATIO: Final[str] = "/futures/data/takerlongshortRatio"
_PATH_GLOBAL_LONG_SHORT_ACCOUNT_RATIO: Final[str] = "/futures/data/globalLongShortAccountRatio"
_PATH_TOP_LONG_SHORT_ACCOUNT_RATIO: Final[str] = "/futures/data/topLongShortAccountRatio"
_PATH_TOP_LONG_SHORT_POSITION_RATIO: Final[str] = "/futures/data/topLongShortPositionRatio"

_RETRYABLE_STATUS_CODES: Final[frozenset[int]] = frozenset({429, 500, 502, 503, 504})
_EXCHANGE_NAME: Final[str] = "binance"
_HEADER_USED_WEIGHT_1M: Final[str] = "X-MBX-USED-WEIGHT-1M"
_HEADER_ORDER_COUNT_1M: Final[str] = "X-MBX-ORDER-COUNT-1M"
_HEADER_RETRY_AFTER: Final[str] = "Retry-After"
_BAN_UNTIL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"banned until\s+(\d+)",
    re.IGNORECASE,
)
_MAX_BACKOFF_SECONDS: Final[float] = 60.0

_logger = logging.getLogger(__name__)


class AsyncTokenBucket:
    """Async token-bucket rate limiter for pacing outbound requests.

    Tokens refill continuously at a fixed rate up to ``capacity``. Callers
    block in ``acquire`` until enough tokens are available.

    Args:
        capacity: Maximum number of tokens the bucket may hold.
        refill_rate: Tokens added per second.

    Raises:
        ValidationError: If ``capacity`` or ``refill_rate`` is not positive.
    """

    __slots__ = ("_capacity", "_refill_rate", "_tokens", "_updated_at", "_lock")

    def __init__(self, *, capacity: float, refill_rate: float) -> None:
        """Initialize bucket capacity and refill rate.

        Args:
            capacity: Maximum number of tokens the bucket may hold.
            refill_rate: Tokens added per second.

        Raises:
            ValidationError: If ``capacity`` or ``refill_rate`` is not positive.
        """
        if capacity <= 0:
            raise ValidationError(
                "capacity must be greater than 0",
                error_code="INGESTION-CLIENT-005",
                details={"parameter": "capacity", "value": capacity},
            )
        if refill_rate <= 0:
            raise ValidationError(
                "refill_rate must be greater than 0",
                error_code="INGESTION-CLIENT-006",
                details={"parameter": "refill_rate", "value": refill_rate},
            )
        self._capacity = float(capacity)
        self._refill_rate = float(refill_rate)
        self._tokens = float(capacity)
        self._updated_at = time.monotonic()
        self._lock = asyncio.Lock()

    @property
    def capacity(self) -> float:
        """Return the configured maximum token capacity."""
        return self._capacity

    @property
    def refill_rate(self) -> float:
        """Return the configured token refill rate per second."""
        return self._refill_rate

    @property
    def tokens(self) -> float:
        """Return the current token balance after applying pending refill."""
        self._refill()
        return self._tokens

    def _refill(self) -> None:
        """Add tokens accrued since the last refill, capped at capacity."""
        now = time.monotonic()
        elapsed = now - self._updated_at
        if elapsed <= 0:
            return
        self._updated_at = now
        self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_rate)

    async def acquire(self, tokens: float = 1.0) -> None:
        """Block until ``tokens`` are available, then consume them.

        Args:
            tokens: Number of tokens to consume. Must be positive.

        Raises:
            ValidationError: If ``tokens`` is not positive.
        """
        if tokens <= 0:
            raise ValidationError(
                "tokens must be greater than 0",
                error_code="INGESTION-CLIENT-007",
                details={"parameter": "tokens", "value": tokens},
            )
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                deficit = tokens - self._tokens
                wait_seconds = deficit / self._refill_rate
            await asyncio.sleep(wait_seconds)


class BinanceClient:
    """Async HTTP client for Binance USDⓈ-M Futures public REST APIs.

    The client manages a shared ``httpx.AsyncClient`` either by creating one
    internally or by accepting an externally owned instance. Retryable
    transport failures and selected HTTP status codes are retried with
    exponential backoff and jitter. An owned ``AsyncTokenBucket`` paces
    requests, and Binance weight headers drive adaptive pre-request
    throttling. Response bodies are returned as decoded JSON with no
    business-level interpretation.

    Args:
        base_url: Binance Futures REST base URL (no trailing slash required).
        timeout: Per-request timeout in seconds applied to the HTTP client.
        max_retries: Maximum retry attempts after the initial request.
        backoff_factor: Base delay in seconds for exponential backoff.
        client: Optional shared ``httpx.AsyncClient``. When provided, the
            caller retains ownership and this client will not close it.
        logger: Optional logger instance. Defaults to the module logger.
        jitter_ratio: Fractional jitter applied to exponential backoff.
        weight_limit_1m: Configured Binance ``REQUEST_WEIGHT`` limit per minute.
        weight_throttle_ratio: Fraction of ``weight_limit_1m`` at which
            adaptive throttling begins.
        weight_window_seconds: Assumed weight window length in seconds.
        request_weight: Default request weight consumed per call.
        ban_safety_margin_seconds: Extra sleep after an IP ban expires.
        token_bucket_capacity: Capacity for the owned token bucket.
        token_bucket_refill_per_second: Refill rate for the owned token bucket.
        rate_limiter: Optional externally owned ``AsyncTokenBucket``. When
            omitted, the client creates and owns one.

    Raises:
        ValidationError: If constructor parameters are invalid.
    """

    __slots__ = (
        "_base_url",
        "_timeout",
        "_max_retries",
        "_backoff_factor",
        "_jitter_ratio",
        "_weight_limit_1m",
        "_weight_throttle_ratio",
        "_weight_window_seconds",
        "_request_weight",
        "_ban_safety_margin_seconds",
        "_client",
        "_owns_client",
        "_logger",
        "_rate_limiter",
        "_used_weight_1m",
        "_order_count_1m",
        "_weight_updated_at",
        "_rng",
    )

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BINANCE_FUTURES_REST_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR_SECONDS,
        client: httpx.AsyncClient | None = None,
        logger: logging.Logger | None = None,
        jitter_ratio: float = DEFAULT_JITTER_RATIO,
        weight_limit_1m: int = DEFAULT_WEIGHT_LIMIT_1M,
        weight_throttle_ratio: float = DEFAULT_WEIGHT_THROTTLE_RATIO,
        weight_window_seconds: float = DEFAULT_WEIGHT_WINDOW_SECONDS,
        request_weight: int = DEFAULT_REQUEST_WEIGHT,
        ban_safety_margin_seconds: float = DEFAULT_BAN_SAFETY_MARGIN_SECONDS,
        token_bucket_capacity: float = DEFAULT_TOKEN_BUCKET_CAPACITY,
        token_bucket_refill_per_second: float = DEFAULT_TOKEN_BUCKET_REFILL_PER_SECOND,
        rate_limiter: AsyncTokenBucket | None = None,
    ) -> None:
        """Initialize client configuration and optional shared session.

        Args:
            base_url: Binance Futures REST base URL.
            timeout: Per-request timeout in seconds.
            max_retries: Maximum retry attempts after the initial request.
            backoff_factor: Base delay in seconds for exponential backoff.
            client: Optional shared ``httpx.AsyncClient``.
            logger: Optional logger instance.
            jitter_ratio: Fractional jitter applied to exponential backoff.
            weight_limit_1m: Configured Binance weight limit per minute.
            weight_throttle_ratio: Fraction of the weight limit that triggers
                adaptive throttling.
            weight_window_seconds: Assumed weight window length in seconds.
            request_weight: Default request weight consumed per call.
            ban_safety_margin_seconds: Extra sleep after an IP ban expires.
            token_bucket_capacity: Capacity for the owned token bucket.
            token_bucket_refill_per_second: Refill rate for the owned token
                bucket.
            rate_limiter: Optional externally owned ``AsyncTokenBucket``.

        Raises:
            ValidationError: If constructor parameters are invalid.
        """
        normalized_base_url = base_url.strip().rstrip("/")
        if not normalized_base_url:
            raise ValidationError(
                "base_url must be a non-empty string",
                error_code="INGESTION-CLIENT-001",
                details={"parameter": "base_url"},
            )
        if timeout <= 0:
            raise ValidationError(
                "timeout must be greater than 0",
                error_code="INGESTION-CLIENT-002",
                details={"parameter": "timeout", "value": timeout},
            )
        if max_retries < 0:
            raise ValidationError(
                "max_retries must be greater than or equal to 0",
                error_code="INGESTION-CLIENT-003",
                details={"parameter": "max_retries", "value": max_retries},
            )
        if backoff_factor < 0:
            raise ValidationError(
                "backoff_factor must be greater than or equal to 0",
                error_code="INGESTION-CLIENT-004",
                details={"parameter": "backoff_factor", "value": backoff_factor},
            )
        if jitter_ratio < 0:
            raise ValidationError(
                "jitter_ratio must be greater than or equal to 0",
                error_code="INGESTION-CLIENT-008",
                details={"parameter": "jitter_ratio", "value": jitter_ratio},
            )
        if weight_limit_1m <= 0:
            raise ValidationError(
                "weight_limit_1m must be greater than 0",
                error_code="INGESTION-CLIENT-009",
                details={"parameter": "weight_limit_1m", "value": weight_limit_1m},
            )
        if not 0.0 < weight_throttle_ratio <= 1.0:
            raise ValidationError(
                "weight_throttle_ratio must be in the interval (0, 1]",
                error_code="INGESTION-CLIENT-010",
                details={
                    "parameter": "weight_throttle_ratio",
                    "value": weight_throttle_ratio,
                },
            )
        if weight_window_seconds <= 0:
            raise ValidationError(
                "weight_window_seconds must be greater than 0",
                error_code="INGESTION-CLIENT-011",
                details={
                    "parameter": "weight_window_seconds",
                    "value": weight_window_seconds,
                },
            )
        if request_weight <= 0:
            raise ValidationError(
                "request_weight must be greater than 0",
                error_code="INGESTION-CLIENT-012",
                details={"parameter": "request_weight", "value": request_weight},
            )
        if ban_safety_margin_seconds < 0:
            raise ValidationError(
                "ban_safety_margin_seconds must be greater than or equal to 0",
                error_code="INGESTION-CLIENT-013",
                details={
                    "parameter": "ban_safety_margin_seconds",
                    "value": ban_safety_margin_seconds,
                },
            )

        self._base_url = normalized_base_url
        self._timeout = timeout
        self._max_retries = max_retries
        self._backoff_factor = backoff_factor
        self._jitter_ratio = jitter_ratio
        self._weight_limit_1m = weight_limit_1m
        self._weight_throttle_ratio = weight_throttle_ratio
        self._weight_window_seconds = weight_window_seconds
        self._request_weight = request_weight
        self._ban_safety_margin_seconds = ban_safety_margin_seconds
        self._client = client
        self._owns_client = client is None
        self._logger = logger if logger is not None else _logger
        if rate_limiter is None:
            self._rate_limiter = AsyncTokenBucket(
                capacity=token_bucket_capacity,
                refill_rate=token_bucket_refill_per_second,
            )
        else:
            self._rate_limiter = rate_limiter
        self._used_weight_1m = 0
        self._order_count_1m = 0
        self._weight_updated_at = time.monotonic()
        self._rng = random.Random()

    @property
    def base_url(self) -> str:
        """Return the configured Binance REST base URL."""
        return self._base_url

    @property
    def timeout(self) -> float:
        """Return the configured request timeout in seconds."""
        return self._timeout

    @property
    def max_retries(self) -> int:
        """Return the configured maximum retry count."""
        return self._max_retries

    @property
    def backoff_factor(self) -> float:
        """Return the configured exponential backoff factor in seconds."""
        return self._backoff_factor

    @property
    def jitter_ratio(self) -> float:
        """Return the configured backoff jitter ratio."""
        return self._jitter_ratio

    @property
    def weight_limit_1m(self) -> int:
        """Return the configured per-minute request weight limit."""
        return self._weight_limit_1m

    @property
    def used_weight_1m(self) -> int:
        """Return the latest observed ``X-MBX-USED-WEIGHT-1M`` value."""
        return self._used_weight_1m

    @property
    def order_count_1m(self) -> int:
        """Return the latest observed ``X-MBX-ORDER-COUNT-1M`` value."""
        return self._order_count_1m

    @property
    def rate_limiter(self) -> AsyncTokenBucket:
        """Return the token-bucket rate limiter used by this client."""
        return self._rate_limiter

    @property
    def is_open(self) -> bool:
        """Return whether an HTTP session is available for requests."""
        return self._client is not None

    async def __aenter__(self) -> Self:
        """Open the client session for use as an async context manager.

        Returns:
            This client instance with an active HTTP session.
        """
        await self.open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        """Close the client session when leaving an async context.

        Args:
            exc_type: Exception type raised in the context, if any.
            exc: Exception instance raised in the context, if any.
            tb: Traceback associated with the exception, if any.
        """
        del exc_type, exc, tb
        await self.close()

    async def open(self) -> None:
        """Create an owned ``httpx.AsyncClient`` when none is configured.

        Shared clients provided at construction are left unchanged.
        """
        if self._client is not None:
            return
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(self._timeout),
        )
        self._owns_client = True
        self._logger.debug(
            "Opened BinanceClient session",
            extra={"exchange": _EXCHANGE_NAME, "base_url": self._base_url},
        )

    async def close(self) -> None:
        """Close the owned HTTP session.

        Externally supplied shared clients are never closed.
        """
        if self._client is None:
            return
        if self._owns_client:
            await self._client.aclose()
            self._logger.debug(
                "Closed BinanceClient session",
                extra={"exchange": _EXCHANGE_NAME, "base_url": self._base_url},
            )
        self._client = None

    async def get_exchange_info(self) -> object:
        """Fetch Binance Futures exchange information.

        Returns:
            Decoded JSON payload from ``GET /fapi/v1/exchangeInfo``.

        Raises:
            ExchangeError: On transport, HTTP, or JSON failures.
        """
        return await self._request("GET", _PATH_EXCHANGE_INFO)

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        *,
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int | None = None,
    ) -> object:
        """Fetch kline / candlestick data for a symbol.

        Args:
            symbol: Binance futures symbol (for example, ``BTCUSDT``).
            interval: Binance kline interval (for example, ``1m``).
            start_time: Optional inclusive start time in milliseconds.
            end_time: Optional inclusive end time in milliseconds.
            limit: Optional maximum number of klines to return.

        Returns:
            Decoded JSON payload from ``GET /fapi/v1/klines``.

        Raises:
            ExchangeError: On transport, HTTP, or JSON failures.
        """
        params: dict[str, str | int] = {
            "symbol": symbol,
            "interval": interval,
        }
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time
        if limit is not None:
            params["limit"] = limit
        return await self._request("GET", _PATH_KLINES, params=params)

    async def get_funding_rates(
        self,
        symbol: str,
        *,
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int | None = None,
    ) -> object:
        """Fetch funding rate history for a symbol.

        Args:
            symbol: Binance futures symbol (for example, ``BTCUSDT``).
            start_time: Optional inclusive start time in milliseconds.
            end_time: Optional inclusive end time in milliseconds.
            limit: Optional maximum number of records to return.

        Returns:
            Decoded JSON payload from ``GET /fapi/v1/fundingRate``.

        Raises:
            ExchangeError: On transport, HTTP, or JSON failures.
        """
        params: dict[str, str | int] = {"symbol": symbol}
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time
        if limit is not None:
            params["limit"] = limit
        return await self._request("GET", _PATH_FUNDING_RATE, params=params)

    async def get_open_interest(self, symbol: str) -> object:
        """Fetch the current open interest for a symbol.

        Args:
            symbol: Binance futures symbol (for example, ``BTCUSDT``).

        Returns:
            Decoded JSON payload from ``GET /fapi/v1/openInterest``.

        Raises:
            ExchangeError: On transport, HTTP, or JSON failures.
        """
        return await self._request(
            "GET",
            _PATH_OPEN_INTEREST,
            params={"symbol": symbol},
        )

    async def get_open_interest_history(
        self,
        symbol: str,
        period: str,
        *,
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int | None = None,
    ) -> object:
        """Fetch historical open-interest statistics for a symbol.

        Args:
            symbol: Binance futures symbol (for example, ``BTCUSDT``).
            period: Aggregation period (for example ``5m`` or ``1h``).
            start_time: Optional inclusive start time in milliseconds.
            end_time: Optional inclusive end time in milliseconds.
            limit: Optional maximum number of records to return.

        Returns:
            Decoded JSON payload from ``GET /futures/data/openInterestHist``.

        Raises:
            ExchangeError: On transport, HTTP, or JSON failures.
        """
        params: dict[str, str | int] = {
            "symbol": symbol,
            "period": period,
        }
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time
        if limit is not None:
            params["limit"] = limit
        return await self._request("GET", _PATH_OPEN_INTEREST_HIST, params=params)

    async def get_taker_buy_sell_volume(
        self,
        symbol: str,
        period: str,
        *,
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int | None = None,
    ) -> object:
        """Fetch historical taker buy/sell volume for a symbol.

        Args:
            symbol: Binance futures symbol (for example, ``BTCUSDT``).
            period: Aggregation period (for example ``5m`` or ``1h``).
            start_time: Optional inclusive start time in milliseconds.
            end_time: Optional inclusive end time in milliseconds.
            limit: Optional maximum number of records to return.

        Returns:
            Decoded JSON payload from ``GET /futures/data/takerlongshortRatio``.

        Raises:
            ExchangeError: On transport, HTTP, or JSON failures.
        """
        params: dict[str, str | int] = {
            "symbol": symbol,
            "period": period,
        }
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time
        if limit is not None:
            params["limit"] = limit
        return await self._request("GET", _PATH_TAKER_LONG_SHORT_RATIO, params=params)

    async def get_global_long_short_account_ratio(
        self,
        symbol: str,
        period: str,
        *,
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int | None = None,
    ) -> object:
        """Fetch global long/short account ratio history for a symbol.

        Args:
            symbol: Binance futures symbol (for example, ``BTCUSDT``).
            period: Aggregation period (for example ``5m`` or ``1h``).
            start_time: Optional inclusive start time in milliseconds.
            end_time: Optional inclusive end time in milliseconds.
            limit: Optional maximum number of records to return.

        Returns:
            Decoded JSON payload from
            ``GET /futures/data/globalLongShortAccountRatio``.

        Raises:
            ExchangeError: On transport, HTTP, or JSON failures.
        """
        return await self._get_symbol_period_history(
            _PATH_GLOBAL_LONG_SHORT_ACCOUNT_RATIO,
            symbol,
            period,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )

    async def get_top_long_short_account_ratio(
        self,
        symbol: str,
        period: str,
        *,
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int | None = None,
    ) -> object:
        """Fetch top-trader long/short account ratio history for a symbol.

        Args:
            symbol: Binance futures symbol (for example, ``BTCUSDT``).
            period: Aggregation period (for example ``5m`` or ``1h``).
            start_time: Optional inclusive start time in milliseconds.
            end_time: Optional inclusive end time in milliseconds.
            limit: Optional maximum number of records to return.

        Returns:
            Decoded JSON payload from
            ``GET /futures/data/topLongShortAccountRatio``.

        Raises:
            ExchangeError: On transport, HTTP, or JSON failures.
        """
        return await self._get_symbol_period_history(
            _PATH_TOP_LONG_SHORT_ACCOUNT_RATIO,
            symbol,
            period,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )

    async def get_top_long_short_position_ratio(
        self,
        symbol: str,
        period: str,
        *,
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int | None = None,
    ) -> object:
        """Fetch top-trader long/short position ratio history for a symbol.

        Args:
            symbol: Binance futures symbol (for example, ``BTCUSDT``).
            period: Aggregation period (for example ``5m`` or ``1h``).
            start_time: Optional inclusive start time in milliseconds.
            end_time: Optional inclusive end time in milliseconds.
            limit: Optional maximum number of records to return.

        Returns:
            Decoded JSON payload from
            ``GET /futures/data/topLongShortPositionRatio``.

        Raises:
            ExchangeError: On transport, HTTP, or JSON failures.
        """
        return await self._get_symbol_period_history(
            _PATH_TOP_LONG_SHORT_POSITION_RATIO,
            symbol,
            period,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )

    async def _get_symbol_period_history(
        self,
        path: str,
        symbol: str,
        period: str,
        *,
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int | None = None,
    ) -> object:
        """Fetch a symbol/period futures-data history endpoint.

        Args:
            path: Absolute Binance futures-data path.
            symbol: Binance futures symbol (for example, ``BTCUSDT``).
            period: Aggregation period (for example ``5m`` or ``1h``).
            start_time: Optional inclusive start time in milliseconds.
            end_time: Optional inclusive end time in milliseconds.
            limit: Optional maximum number of records to return.

        Returns:
            Decoded JSON payload from the requested endpoint.

        Raises:
            ExchangeError: On transport, HTTP, or JSON failures.
        """
        params: dict[str, str | int] = {
            "symbol": symbol,
            "period": period,
        }
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time
        if limit is not None:
            params["limit"] = limit
        return await self._request("GET", path, params=params)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str | int | float | bool | None] | None = None,
        weight: int | None = None,
    ) -> object:
        """Execute an HTTP request with pacing, retry, and JSON decoding.

        This is the extension point for additional Binance endpoints.
        New public helpers should call this method with the appropriate
        HTTP method, path, and query parameters.

        Args:
            method: HTTP method name (for example, ``GET``).
            path: Absolute path under the configured base URL.
            params: Optional query-string parameters.
            weight: Optional request weight override. Defaults to the
                client ``request_weight``.

        Returns:
            Decoded JSON response body.

        Raises:
            ExchangeError: When the client is closed or the request fails.
            ExchangeTimeoutError: When the request times out.
            ExchangeUnavailableError: When the endpoint is unavailable.
            ExchangeRateLimitError: When rate limits are exceeded.
            ExchangeAuthenticationError: When authentication fails.
            ExchangePermissionError: When permissions are insufficient.
            ExchangeSymbolNotFoundError: When a symbol cannot be resolved.
            ExchangeValidationError: When the request or response is invalid.
        """
        client = self._require_client()
        query = _sanitize_params(params)
        url = f"{self._base_url}{path}"
        request_weight = self._request_weight if weight is None else weight
        if request_weight <= 0:
            raise ValidationError(
                "weight must be greater than 0",
                error_code="INGESTION-CLIENT-012",
                details={"parameter": "weight", "value": request_weight},
            )
        attempt = 0

        while True:
            await self._prepare_request(request_weight)

            try:
                response = await client.request(method, url, params=query)
            except httpx.TimeoutException as exc:
                if attempt >= self._max_retries:
                    raise ExchangeTimeoutError(
                        "Binance request timed out",
                        error_code="INGESTION-CLIENT-TIMEOUT",
                        details=self._error_details(method, path, attempt=attempt),
                        recovery_suggestion="Increase timeout or retry later.",
                    ) from exc
                await self._backoff(attempt, method, path, reason="timeout")
                attempt += 1
                continue
            except httpx.ConnectError as exc:
                if attempt >= self._max_retries:
                    raise ExchangeUnavailableError(
                        "Binance transport failure",
                        error_code="INGESTION-CLIENT-TRANSPORT",
                        details=self._error_details(
                            method,
                            path,
                            attempt=attempt,
                            cause=type(exc).__name__,
                        ),
                        recovery_suggestion="Check network connectivity and retry.",
                    ) from exc
                await self._backoff(attempt, method, path, reason="connect")
                attempt += 1
                continue
            except httpx.TransportError as exc:
                if attempt >= self._max_retries:
                    raise ExchangeUnavailableError(
                        "Binance transport failure",
                        error_code="INGESTION-CLIENT-TRANSPORT",
                        details=self._error_details(
                            method,
                            path,
                            attempt=attempt,
                            cause=type(exc).__name__,
                        ),
                        recovery_suggestion="Check network connectivity and retry.",
                    ) from exc
                await self._backoff(attempt, method, path, reason="transport")
                attempt += 1
                continue

            self._update_rate_limit_state(response)

            if response.status_code == 418:
                await self._recover_from_ip_ban(
                    response,
                    method=method,
                    path=path,
                    attempt=attempt,
                )
                continue

            if response.status_code in _RETRYABLE_STATUS_CODES and attempt < self._max_retries:
                retry_after = _parse_retry_after_seconds(response)
                await self._backoff(
                    attempt,
                    method,
                    path,
                    reason=f"http_{response.status_code}",
                    minimum_delay=retry_after,
                )
                attempt += 1
                continue

            return self._handle_response(response, method=method, path=path)

    async def _prepare_request(self, request_weight: int) -> None:
        """Acquire token-bucket capacity and apply adaptive weight throttling.

        Args:
            request_weight: Weight that the upcoming request is expected to
                consume.
        """
        await self._rate_limiter.acquire(float(request_weight))
        await self._throttle_for_weight(request_weight)

    async def _throttle_for_weight(self, request_weight: int) -> None:
        """Sleep when projected used weight approaches the configured limit.

        Args:
            request_weight: Weight that the upcoming request is expected to
                consume.
        """
        threshold = self._weight_limit_1m * self._weight_throttle_ratio
        projected = self._used_weight_1m + request_weight
        if projected < threshold:
            return

        if self._used_weight_1m >= self._weight_limit_1m:
            delay = self._weight_window_seconds
        else:
            span = max(1.0, float(self._weight_limit_1m) - threshold)
            delay = self._weight_window_seconds * ((projected - threshold) / span)
            delay = min(delay, self._weight_window_seconds)

        self._logger.warning(
            "Throttling Binance request due to used weight",
            extra={
                "exchange": _EXCHANGE_NAME,
                "used_weight_1m": self._used_weight_1m,
                "weight_limit_1m": self._weight_limit_1m,
                "request_weight": request_weight,
                "delay_seconds": delay,
            },
        )
        await asyncio.sleep(delay)
        # Assume the rolling window has recovered after an intentional pause.
        self._used_weight_1m = 0
        self._weight_updated_at = time.monotonic()

    def _update_rate_limit_state(self, response: httpx.Response) -> None:
        """Update local weight/order counters from Binance response headers.

        Args:
            response: HTTP response that may include rate-limit headers.
        """
        used_weight = _parse_optional_int_header(response, _HEADER_USED_WEIGHT_1M)
        if used_weight is not None:
            self._used_weight_1m = used_weight
            self._weight_updated_at = time.monotonic()

        order_count = _parse_optional_int_header(response, _HEADER_ORDER_COUNT_1M)
        if order_count is not None:
            self._order_count_1m = order_count

    async def _recover_from_ip_ban(
        self,
        response: httpx.Response,
        *,
        method: str,
        path: str,
        attempt: int,
    ) -> None:
        """Sleep until an HTTP 418 IP ban expires, then resume.

        HTTP 418 never fails the caller. When Binance includes a ban-until
        timestamp the client waits until that instant plus the configured
        safety margin. Otherwise ``Retry-After`` or exponential backoff is
        used.

        Args:
            response: HTTP 418 response from Binance.
            method: HTTP method being retried.
            path: Request path being retried.
            attempt: Zero-based attempt index used for backoff fallback.
        """
        body_preview = _safe_body_preview(response)
        ban_until_ms = _parse_ban_until_ms(body_preview)
        retry_after = _parse_retry_after_seconds(response)

        if ban_until_ms is not None:
            now_ms = int(datetime.now(UTC).timestamp() * 1000)
            delay = max(0.0, (ban_until_ms - now_ms) / 1000.0)
            delay += self._ban_safety_margin_seconds
            reason = "ip_ban"
        elif retry_after is not None:
            delay = retry_after + self._ban_safety_margin_seconds
            reason = "ip_ban_retry_after"
        else:
            delay = self._compute_backoff_delay(attempt)
            reason = "ip_ban_backoff"

        self._logger.warning(
            "Binance IP ban received; waiting before resume",
            extra={
                "exchange": _EXCHANGE_NAME,
                "method": method,
                "path": path,
                "attempt": attempt,
                "delay_seconds": delay,
                "ban_until_ms": ban_until_ms,
                "reason": reason,
                "body": body_preview,
            },
        )
        await asyncio.sleep(delay)
        self._used_weight_1m = 0
        self._weight_updated_at = time.monotonic()

    def _require_client(self) -> httpx.AsyncClient:
        """Return the active HTTP session or raise if closed.

        Returns:
            Active ``httpx.AsyncClient`` instance.

        Raises:
            ExchangeError: If no session is available.
        """
        if self._client is None:
            raise ExchangeError(
                "BinanceClient is not open; call open() or use async with",
                error_code="INGESTION-CLIENT-CLOSED",
                details={"exchange": _EXCHANGE_NAME, "base_url": self._base_url},
                recovery_suggestion="Enter the async context manager before requesting.",
            )
        return self._client

    def _compute_backoff_delay(self, attempt: int) -> float:
        """Compute an exponential backoff delay with optional jitter.

        Args:
            attempt: Zero-based attempt index that just failed.

        Returns:
            Delay in seconds, capped at ``_MAX_BACKOFF_SECONDS``.
        """
        delay = self._backoff_factor * (2**attempt)
        if self._jitter_ratio > 0 and delay > 0:
            spread = delay * self._jitter_ratio
            delay = max(0.0, delay + self._rng.uniform(-spread, spread))
        return min(delay, _MAX_BACKOFF_SECONDS)

    async def _backoff(
        self,
        attempt: int,
        method: str,
        path: str,
        *,
        reason: str,
        minimum_delay: float | None = None,
    ) -> None:
        """Sleep for the exponential backoff delay for the given attempt.

        Args:
            attempt: Zero-based attempt index that just failed.
            method: HTTP method being retried.
            path: Request path being retried.
            reason: Short reason code for logging.
            minimum_delay: Optional lower bound from ``Retry-After``.
        """
        delay = self._compute_backoff_delay(attempt)
        if minimum_delay is not None:
            delay = max(delay, minimum_delay)
        self._logger.warning(
            "Retrying Binance request",
            extra={
                "exchange": _EXCHANGE_NAME,
                "method": method,
                "path": path,
                "attempt": attempt + 1,
                "max_retries": self._max_retries,
                "delay_seconds": delay,
                "reason": reason,
            },
        )
        await asyncio.sleep(delay)

    def _handle_response(
        self,
        response: httpx.Response,
        *,
        method: str,
        path: str,
    ) -> object:
        """Translate an HTTP response into JSON or a CQROS exception.

        Args:
            response: HTTP response returned by httpx.
            method: HTTP method that produced the response.
            path: Request path that produced the response.

        Returns:
            Decoded JSON body for successful responses.

        Raises:
            ExchangeError: For non-success HTTP responses or invalid JSON.
        """
        if response.status_code == 429 or response.status_code == 418:
            raise ExchangeRateLimitError(
                "Binance rate limit exceeded",
                error_code="INGESTION-CLIENT-RATE-LIMIT",
                details=self._error_details(
                    method,
                    path,
                    status_code=response.status_code,
                    body=_safe_body_preview(response),
                ),
                recovery_suggestion="Back off and retry after the rate-limit window.",
            )

        if response.status_code in {401, 403}:
            exception_type: type[ExchangeError] = (
                ExchangeAuthenticationError
                if response.status_code == 401
                else ExchangePermissionError
            )
            raise exception_type(
                "Binance rejected the request due to authentication or permissions",
                error_code="INGESTION-CLIENT-AUTH",
                details=self._error_details(
                    method,
                    path,
                    status_code=response.status_code,
                    body=_safe_body_preview(response),
                ),
            )

        if response.status_code >= 500:
            raise ExchangeUnavailableError(
                "Binance endpoint unavailable",
                error_code="INGESTION-CLIENT-UNAVAILABLE",
                details=self._error_details(
                    method,
                    path,
                    status_code=response.status_code,
                    body=_safe_body_preview(response),
                ),
                recovery_suggestion="Retry after a short delay.",
            )

        payload = _decode_json(response, method=method, path=path, base_url=self._base_url)

        if response.status_code >= 400:
            raise _map_binance_client_error(
                payload,
                method=method,
                path=path,
                base_url=self._base_url,
                status_code=response.status_code,
            )

        error_payload = _as_string_object_mapping(payload)
        if error_payload is not None and "code" in error_payload and "msg" in error_payload:
            code = error_payload.get("code")
            if isinstance(code, int) and code < 0:
                raise _map_binance_client_error(
                    error_payload,
                    method=method,
                    path=path,
                    base_url=self._base_url,
                    status_code=response.status_code,
                )

        return payload

    def _error_details(
        self,
        method: str,
        path: str,
        *,
        attempt: int | None = None,
        status_code: int | None = None,
        cause: str | None = None,
        body: str | None = None,
    ) -> dict[str, object]:
        """Build structured diagnostic details for raised exceptions.

        Args:
            method: HTTP method.
            path: Request path.
            attempt: Optional zero-based attempt index.
            status_code: Optional HTTP status code.
            cause: Optional transport failure type name.
            body: Optional response body preview.

        Returns:
            Mapping suitable for ``CQROSError.details``.
        """
        details: dict[str, object] = {
            "exchange": _EXCHANGE_NAME,
            "base_url": self._base_url,
            "method": method,
            "path": path,
        }
        if attempt is not None:
            details["attempt"] = attempt
        if status_code is not None:
            details["status_code"] = status_code
        if cause is not None:
            details["cause"] = cause
        if body is not None:
            details["body"] = body
        return details


def _as_string_object_mapping(payload: object) -> Mapping[str, object] | None:
    """Return ``payload`` as a string-keyed mapping when possible.

    Args:
        payload: Decoded JSON value.

    Returns:
        A ``Mapping[str, object]`` view of ``payload``, or ``None`` when the
        value is not a mapping.
    """
    if isinstance(payload, dict):
        return cast(Mapping[str, object], payload)
    return None


def _sanitize_params(
    params: Mapping[str, str | int | float | bool | None] | None,
) -> dict[str, str | int | float | bool] | None:
    """Drop ``None`` query values so httpx omits unbound parameters.

    Args:
        params: Optional raw query parameter mapping.

    Returns:
        Sanitized mapping, or ``None`` when no parameters remain.
    """
    if params is None:
        return None
    cleaned = {key: value for key, value in params.items() if value is not None}
    return cleaned or None


def _decode_json(
    response: httpx.Response,
    *,
    method: str,
    path: str,
    base_url: str,
) -> object:
    """Decode a response body as JSON.

    Args:
        response: HTTP response to decode.
        method: HTTP method for error context.
        path: Request path for error context.
        base_url: Configured base URL for error context.

    Returns:
        Decoded JSON value.

    Raises:
        ExchangeValidationError: If the body is not valid JSON.
    """
    try:
        return response.json()
    except ValueError as exc:
        raise ExchangeValidationError(
            "Binance response was not valid JSON",
            error_code="INGESTION-CLIENT-JSON",
            details={
                "exchange": _EXCHANGE_NAME,
                "base_url": base_url,
                "method": method,
                "path": path,
                "status_code": response.status_code,
                "body": _safe_body_preview(response),
            },
            recovery_suggestion="Inspect the upstream response and retry.",
        ) from exc


def _map_binance_client_error(
    payload: object,
    *,
    method: str,
    path: str,
    base_url: str,
    status_code: int,
) -> ExchangeError:
    """Map a Binance error payload to a CQROS exchange exception.

    Args:
        payload: Decoded JSON error body.
        method: HTTP method for error context.
        path: Request path for error context.
        base_url: Configured base URL for error context.
        status_code: HTTP status code.

    Returns:
        An appropriate ``ExchangeError`` subclass instance.
    """
    code: int | None = None
    message = "Binance rejected the request"
    mapping = _as_string_object_mapping(payload)
    if mapping is not None:
        raw_code = mapping.get("code")
        raw_msg = mapping.get("msg")
        if isinstance(raw_code, int):
            code = raw_code
        if isinstance(raw_msg, str) and raw_msg:
            message = raw_msg

    details: dict[str, object] = {
        "exchange": _EXCHANGE_NAME,
        "base_url": base_url,
        "method": method,
        "path": path,
        "status_code": status_code,
        "binance_code": code,
        "body": payload,
    }

    if code == -1121:
        return ExchangeSymbolNotFoundError(
            message,
            error_code="INGESTION-CLIENT-SYMBOL",
            details=details,
        )
    if code in {-2014, -2015, -1022}:
        return ExchangeAuthenticationError(
            message,
            error_code="INGESTION-CLIENT-AUTH",
            details=details,
        )
    if code in {-2010, -2011}:
        return ExchangePermissionError(
            message,
            error_code="INGESTION-CLIENT-PERMISSION",
            details=details,
        )
    return ExchangeValidationError(
        message,
        error_code="INGESTION-CLIENT-VALIDATION",
        details=details,
    )


def _safe_body_preview(response: httpx.Response, *, limit: int = 512) -> str:
    """Return a truncated text preview of a response body.

    Args:
        response: HTTP response whose body should be previewed.
        limit: Maximum number of characters to include.

    Returns:
        Truncated UTF-8 text preview of the response body.
    """
    return response.content[:limit].decode("utf-8", errors="replace")


def _parse_optional_int_header(response: httpx.Response, header_name: str) -> int | None:
    """Parse an optional integer HTTP header.

    Args:
        response: HTTP response containing headers.
        header_name: Header name to read.

    Returns:
        Parsed integer value, or ``None`` when missing or invalid.
    """
    raw_value = response.headers.get(header_name)
    if raw_value is None:
        return None
    try:
        return int(raw_value)
    except ValueError:
        return None


def _parse_retry_after_seconds(response: httpx.Response) -> float | None:
    """Parse a ``Retry-After`` header expressed in seconds.

    Args:
        response: HTTP response that may include ``Retry-After``.

    Returns:
        Delay in seconds, or ``None`` when the header is absent or invalid.
    """
    raw_value = response.headers.get(_HEADER_RETRY_AFTER)
    if raw_value is None:
        return None
    try:
        seconds = float(raw_value)
    except ValueError:
        return None
    if seconds < 0:
        return None
    return seconds


def _parse_ban_until_ms(body: str) -> int | None:
    """Extract a Binance IP ban-until timestamp in milliseconds.

    Args:
        body: Response body text that may contain a ban-until message.

    Returns:
        Ban-until epoch milliseconds, or ``None`` when not present.
    """
    match = _BAN_UNTIL_PATTERN.search(body)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None
