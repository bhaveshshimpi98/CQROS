"""Unit tests for the Binance USDⓈ-M Futures REST client."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from cqros.core.exceptions import (
    ExchangeAuthenticationError,
    ExchangeError,
    ExchangeRateLimitError,
    ExchangeSymbolNotFoundError,
    ExchangeTimeoutError,
    ExchangeUnavailableError,
    ExchangeValidationError,
    ValidationError,
)
from cqros.ingestion.client import (
    DEFAULT_BACKOFF_FACTOR_SECONDS,
    DEFAULT_BINANCE_FUTURES_REST_BASE_URL,
    DEFAULT_JITTER_RATIO,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_WEIGHT_LIMIT_1M,
    AsyncTokenBucket,
    BinanceClient,
)

_Handler = Callable[[httpx.Request], httpx.Response]


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def _mock_client(handler: _Handler) -> httpx.AsyncClient:
    """Create an AsyncClient backed by MockTransport."""
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url=DEFAULT_BINANCE_FUTURES_REST_BASE_URL,
    )


def _fast_limiter() -> AsyncTokenBucket:
    """Return a token bucket that never blocks unit tests."""
    return AsyncTokenBucket(capacity=10_000.0, refill_rate=10_000.0)


def test_constructor_rejects_invalid_parameters() -> None:
    """Invalid timeout, retry, backoff, and base_url values raise ValidationError."""
    with pytest.raises(ValidationError, match="base_url"):
        BinanceClient(base_url="   ")
    with pytest.raises(ValidationError, match="timeout"):
        BinanceClient(timeout=0)
    with pytest.raises(ValidationError, match="max_retries"):
        BinanceClient(max_retries=-1)
    with pytest.raises(ValidationError, match="backoff_factor"):
        BinanceClient(backoff_factor=-0.1)
    with pytest.raises(ValidationError, match="jitter_ratio"):
        BinanceClient(jitter_ratio=-0.1)
    with pytest.raises(ValidationError, match="weight_limit_1m"):
        BinanceClient(weight_limit_1m=0)
    with pytest.raises(ValidationError, match="weight_throttle_ratio"):
        BinanceClient(weight_throttle_ratio=0)
    with pytest.raises(ValidationError, match="request_weight"):
        BinanceClient(request_weight=0)


def test_request_requires_open_session() -> None:
    """Requests fail fast when the owned session has not been opened."""
    client = BinanceClient(rate_limiter=_fast_limiter())

    with pytest.raises(ExchangeError, match="not open"):
        _run(client.get_exchange_info())


def test_async_context_manager_opens_and_closes_owned_client() -> None:
    """Context manager creates and closes an owned httpx session."""
    mock_session = _mock_client(lambda request: httpx.Response(200, json={"timezone": "UTC"}))

    async def _exercise() -> None:
        with patch(
            "cqros.ingestion.client.httpx.AsyncClient",
            return_value=mock_session,
        ) as ctor:
            async with BinanceClient(rate_limiter=_fast_limiter()) as client:
                assert client.is_open
                payload = await client.get_exchange_info()
                assert payload == {"timezone": "UTC"}
            ctor.assert_called_once()
        assert not client.is_open
        assert mock_session.is_closed

    _run(_exercise())


def test_shared_client_is_not_closed() -> None:
    """Externally supplied AsyncClient remains open after BinanceClient.close()."""

    async def _exercise() -> None:
        shared = _mock_client(lambda request: httpx.Response(200, json={"ok": True}))
        client = BinanceClient(client=shared, rate_limiter=_fast_limiter())
        assert client.is_open
        assert await client.get_exchange_info() == {"ok": True}
        await client.close()
        assert not client.is_open
        assert not shared.is_closed
        await shared.aclose()

    _run(_exercise())


def test_get_klines_funding_and_open_interest_pass_query_params() -> None:
    """Endpoint helpers forward the expected query parameters."""
    seen: dict[str, httpx.URL] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen[request.url.path] = request.url
        if request.url.path.endswith("/klines"):
            return httpx.Response(200, json=[[1, "2"]])
        if request.url.path.endswith("/fundingRate"):
            return httpx.Response(200, json=[{"symbol": "BTCUSDT"}])
        if request.url.path.endswith("/openInterestHist"):
            return httpx.Response(
                200,
                json=[{"symbol": "BTCUSDT", "sumOpenInterest": "1", "timestamp": "2"}],
            )
        if request.url.path.endswith("/takerlongshortRatio"):
            return httpx.Response(
                200,
                json=[
                    {
                        "buyVol": "1",
                        "sellVol": "2",
                        "buySellRatio": "0.5",
                        "timestamp": "3",
                    }
                ],
            )
        if request.url.path.endswith("/globalLongShortAccountRatio"):
            return httpx.Response(200, json=[{"symbol": "BTCUSDT", "kind": "global"}])
        if request.url.path.endswith("/topLongShortAccountRatio"):
            return httpx.Response(200, json=[{"symbol": "BTCUSDT", "kind": "top_acct"}])
        if request.url.path.endswith("/topLongShortPositionRatio"):
            return httpx.Response(200, json=[{"symbol": "BTCUSDT", "kind": "top_pos"}])
        return httpx.Response(200, json={"openInterest": "100"})

    async def _exercise() -> None:
        async with BinanceClient(
            client=_mock_client(handler),
            rate_limiter=_fast_limiter(),
        ) as client:
            klines = await client.get_klines(
                "BTCUSDT",
                "1m",
                start_time=10,
                end_time=20,
                limit=5,
            )
            funding = await client.get_funding_rates(
                "BTCUSDT",
                start_time=11,
                end_time=21,
                limit=6,
            )
            open_interest = await client.get_open_interest("BTCUSDT")
            open_interest_hist = await client.get_open_interest_history(
                "BTCUSDT",
                "5m",
                start_time=12,
                end_time=22,
                limit=7,
            )
            taker_volume = await client.get_taker_buy_sell_volume(
                "BTCUSDT",
                "5m",
                start_time=13,
                end_time=23,
                limit=8,
            )
            global_ratio = await client.get_global_long_short_account_ratio(
                "BTCUSDT",
                "5m",
                start_time=14,
                end_time=24,
                limit=9,
            )
            top_account = await client.get_top_long_short_account_ratio(
                "BTCUSDT",
                "15m",
                start_time=15,
                end_time=25,
                limit=10,
            )
            top_position = await client.get_top_long_short_position_ratio(
                "BTCUSDT",
                "1h",
                start_time=16,
                end_time=26,
                limit=11,
            )

        assert klines == [[1, "2"]]
        assert funding == [{"symbol": "BTCUSDT"}]
        assert open_interest == {"openInterest": "100"}
        assert open_interest_hist == [
            {"symbol": "BTCUSDT", "sumOpenInterest": "1", "timestamp": "2"}
        ]
        assert taker_volume == [
            {
                "buyVol": "1",
                "sellVol": "2",
                "buySellRatio": "0.5",
                "timestamp": "3",
            }
        ]
        assert global_ratio == [{"symbol": "BTCUSDT", "kind": "global"}]
        assert top_account == [{"symbol": "BTCUSDT", "kind": "top_acct"}]
        assert top_position == [{"symbol": "BTCUSDT", "kind": "top_pos"}]
        assert seen["/fapi/v1/klines"].params["symbol"] == "BTCUSDT"
        assert seen["/fapi/v1/klines"].params["interval"] == "1m"
        assert seen["/fapi/v1/klines"].params["startTime"] == "10"
        assert seen["/fapi/v1/klines"].params["endTime"] == "20"
        assert seen["/fapi/v1/klines"].params["limit"] == "5"
        assert seen["/fapi/v1/fundingRate"].params["startTime"] == "11"
        assert seen["/fapi/v1/openInterest"].params["symbol"] == "BTCUSDT"
        assert seen["/futures/data/openInterestHist"].params["symbol"] == "BTCUSDT"
        assert seen["/futures/data/openInterestHist"].params["period"] == "5m"
        assert seen["/futures/data/openInterestHist"].params["startTime"] == "12"
        assert seen["/futures/data/openInterestHist"].params["endTime"] == "22"
        assert seen["/futures/data/openInterestHist"].params["limit"] == "7"
        assert seen["/futures/data/takerlongshortRatio"].params["symbol"] == "BTCUSDT"
        assert seen["/futures/data/takerlongshortRatio"].params["period"] == "5m"
        assert seen["/futures/data/takerlongshortRatio"].params["startTime"] == "13"
        assert seen["/futures/data/takerlongshortRatio"].params["endTime"] == "23"
        assert seen["/futures/data/takerlongshortRatio"].params["limit"] == "8"
        assert seen["/futures/data/globalLongShortAccountRatio"].params["startTime"] == "14"
        assert seen["/futures/data/topLongShortAccountRatio"].params["period"] == "15m"
        assert seen["/futures/data/topLongShortPositionRatio"].params["limit"] == "11"

    _run(_exercise())


def test_retries_on_server_error_with_exponential_backoff() -> None:
    """HTTP 503 responses are retried with exponential backoff delays."""
    calls = {"count": 0}
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        calls["count"] += 1
        if calls["count"] < 3:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, json={"recovered": True})

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)

    async def _exercise() -> None:
        with patch("cqros.ingestion.client.asyncio.sleep", side_effect=_sleep):
            async with BinanceClient(
                client=_mock_client(handler),
                max_retries=3,
                backoff_factor=0.5,
                jitter_ratio=0.0,
                rate_limiter=_fast_limiter(),
            ) as client:
                payload = await client.get_exchange_info()

        assert payload == {"recovered": True}
        assert calls["count"] == 3
        assert sleeps == [0.5, 1.0]

    _run(_exercise())


def test_exponential_backoff_applies_jitter() -> None:
    """Backoff delays include configured jitter around the exponential base."""
    calls = {"count": 0}
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        calls["count"] += 1
        if calls["count"] < 2:
            return httpx.Response(500, text="error")
        return httpx.Response(200, json={"ok": True})

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)

    async def _exercise() -> None:
        with patch("cqros.ingestion.client.asyncio.sleep", side_effect=_sleep):
            with patch("random.Random.uniform", return_value=0.1):
                async with BinanceClient(
                    client=_mock_client(handler),
                    max_retries=2,
                    backoff_factor=1.0,
                    jitter_ratio=0.25,
                    rate_limiter=_fast_limiter(),
                ) as client:
                    payload = await client.get_exchange_info()

        assert payload == {"ok": True}
        assert len(sleeps) == 1
        # base=1.0, jitter_ratio=0.25, uniform returns 0.1 -> delay = 1.0 + 0.1 = 1.1
        assert sleeps[0] == pytest.approx(1.1)

    _run(_exercise())


def test_does_not_retry_client_validation_errors() -> None:
    """Non-retryable 400 responses are raised immediately."""
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        calls["count"] += 1
        return httpx.Response(400, json={"code": -1102, "msg": "Mandatory parameter"})

    async def _exercise() -> None:
        async with BinanceClient(
            client=_mock_client(handler),
            max_retries=3,
            rate_limiter=_fast_limiter(),
        ) as client:
            with pytest.raises(ExchangeValidationError, match="Mandatory parameter"):
                await client.get_exchange_info()

    _run(_exercise())
    assert calls["count"] == 1


def test_maps_rate_limit_timeout_and_symbol_errors() -> None:
    """Status and Binance error codes map to specialized CQROS exceptions."""

    async def _rate_limit() -> None:
        with patch("cqros.ingestion.client.asyncio.sleep", new_callable=AsyncMock):
            async with BinanceClient(
                client=_mock_client(lambda request: httpx.Response(429, text="slow down")),
                max_retries=0,
                rate_limiter=_fast_limiter(),
            ) as client:
                with pytest.raises(ExchangeRateLimitError):
                    await client.get_exchange_info()

    async def _symbol() -> None:
        async with BinanceClient(
            client=_mock_client(
                lambda request: httpx.Response(
                    400,
                    json={"code": -1121, "msg": "Invalid symbol."},
                )
            ),
            rate_limiter=_fast_limiter(),
        ) as client:
            with pytest.raises(ExchangeSymbolNotFoundError, match="Invalid symbol"):
                await client.get_klines("BAD", "1m")

    async def _auth() -> None:
        async with BinanceClient(
            client=_mock_client(lambda request: httpx.Response(401, text="unauthorized")),
            rate_limiter=_fast_limiter(),
        ) as client:
            with pytest.raises(ExchangeAuthenticationError):
                await client.get_exchange_info()

    _run(_rate_limit())
    _run(_symbol())
    _run(_auth())


def test_timeout_and_transport_failures_are_translated() -> None:
    """Timeout and transport errors become CQROS exchange exceptions."""

    async def _timeout() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            del request
            raise httpx.ReadTimeout("read timed out")

        with patch("cqros.ingestion.client.asyncio.sleep", new_callable=AsyncMock):
            async with BinanceClient(
                client=_mock_client(handler),
                max_retries=1,
                backoff_factor=0,
                jitter_ratio=0.0,
                rate_limiter=_fast_limiter(),
            ) as client:
                with pytest.raises(ExchangeTimeoutError):
                    await client.get_exchange_info()

    async def _transport() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            del request
            raise httpx.ConnectError("connection refused")

        with patch("cqros.ingestion.client.asyncio.sleep", new_callable=AsyncMock):
            async with BinanceClient(
                client=_mock_client(handler),
                max_retries=0,
                rate_limiter=_fast_limiter(),
            ) as client:
                with pytest.raises(ExchangeUnavailableError):
                    await client.get_exchange_info()

    _run(_timeout())
    _run(_transport())


def test_invalid_json_raises_validation_error() -> None:
    """Non-JSON success bodies raise ExchangeValidationError."""

    async def _exercise() -> None:
        async with BinanceClient(
            client=_mock_client(lambda request: httpx.Response(200, text="not-json")),
            rate_limiter=_fast_limiter(),
        ) as client:
            with pytest.raises(ExchangeValidationError, match="not valid JSON"):
                await client.get_exchange_info()

    _run(_exercise())


def test_default_configuration_constants() -> None:
    """Constructor defaults match the published module constants."""
    client = BinanceClient()
    assert client.base_url == DEFAULT_BINANCE_FUTURES_REST_BASE_URL
    assert client.timeout == DEFAULT_TIMEOUT_SECONDS
    assert client.max_retries == DEFAULT_MAX_RETRIES
    assert client.backoff_factor == DEFAULT_BACKOFF_FACTOR_SECONDS
    assert client.jitter_ratio == DEFAULT_JITTER_RATIO
    assert client.weight_limit_1m == DEFAULT_WEIGHT_LIMIT_1M
    assert isinstance(client.rate_limiter, AsyncTokenBucket)


def test_none_query_params_are_omitted() -> None:
    """Optional query parameters set to None are not sent."""
    seen: dict[str, httpx.URL] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen[request.url.path] = request.url
        return httpx.Response(200, json=[])

    async def _exercise() -> None:
        async with BinanceClient(
            client=_mock_client(handler),
            rate_limiter=_fast_limiter(),
        ) as client:
            await client.get_klines("BTCUSDT", "1m")

        params = seen["/fapi/v1/klines"].params
        assert "startTime" not in params
        assert "endTime" not in params
        assert "limit" not in params

    _run(_exercise())


def test_token_bucket_rejects_invalid_configuration() -> None:
    """Token bucket construction validates capacity and refill rate."""
    with pytest.raises(ValidationError, match="capacity"):
        AsyncTokenBucket(capacity=0, refill_rate=1.0)
    with pytest.raises(ValidationError, match="refill_rate"):
        AsyncTokenBucket(capacity=1.0, refill_rate=0)
    with pytest.raises(ValidationError, match="tokens"):
        _run(AsyncTokenBucket(capacity=1.0, refill_rate=1.0).acquire(0))


def test_token_bucket_acquire_consumes_and_refills() -> None:
    """Acquire consumes tokens and waits when the bucket is empty."""
    sleeps: list[float] = []
    clock = {"now": 100.0}

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)
        clock["now"] += delay

    async def _exercise() -> None:
        with patch("cqros.ingestion.client.asyncio.sleep", side_effect=_sleep):
            with patch(
                "cqros.ingestion.client.time.monotonic",
                side_effect=lambda: clock["now"],
            ):
                bucket = AsyncTokenBucket(capacity=1.0, refill_rate=2.0)
                await bucket.acquire(1.0)
                assert bucket.tokens == pytest.approx(0.0)
                await bucket.acquire(1.0)
                assert bucket.tokens == pytest.approx(0.0)

        assert sleeps == [0.5]

    _run(_exercise())


def test_binance_client_reuses_owned_token_bucket_across_requests() -> None:
    """Every request acquires from the same client-owned rate limiter."""
    acquires: list[float] = []

    class _TrackingBucket(AsyncTokenBucket):
        async def acquire(self, tokens: float = 1.0) -> None:
            acquires.append(tokens)
            await super().acquire(tokens)

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json={"ok": True})

    async def _exercise() -> None:
        limiter = _TrackingBucket(capacity=100.0, refill_rate=100.0)
        async with BinanceClient(
            client=_mock_client(handler),
            rate_limiter=limiter,
            request_weight=2,
        ) as client:
            assert client.rate_limiter is limiter
            await client.get_exchange_info()
            await client.get_exchange_info()

        assert acquires == [2.0, 2.0]

    _run(_exercise())


def test_parses_binance_rate_limit_headers() -> None:
    """Response weight and order-count headers update client state."""

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={"ok": True},
            headers={
                "X-MBX-USED-WEIGHT-1M": "1234",
                "X-MBX-ORDER-COUNT-1M": "7",
            },
        )

    async def _exercise() -> None:
        async with BinanceClient(
            client=_mock_client(handler),
            rate_limiter=_fast_limiter(),
        ) as client:
            await client.get_exchange_info()
            assert client.used_weight_1m == 1234
            assert client.order_count_1m == 7

    _run(_exercise())


def test_adaptive_throttling_before_request_when_weight_high() -> None:
    """Client sleeps before issuing a request when used weight nears the limit."""
    sleeps: list[float] = []
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(
                200,
                json={"ok": True},
                headers={"X-MBX-USED-WEIGHT-1M": "900"},
            )
        return httpx.Response(
            200,
            json={"ok": True},
            headers={"X-MBX-USED-WEIGHT-1M": "100"},
        )

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)

    async def _exercise() -> None:
        with patch("cqros.ingestion.client.asyncio.sleep", side_effect=_sleep):
            async with BinanceClient(
                client=_mock_client(handler),
                rate_limiter=_fast_limiter(),
                weight_limit_1m=1000,
                weight_throttle_ratio=0.8,
                weight_window_seconds=60.0,
                request_weight=1,
            ) as client:
                await client.get_exchange_info()
                assert client.used_weight_1m == 900
                payload = await client.get_exchange_info()

        assert payload == {"ok": True}
        assert calls["count"] == 2
        assert len(sleeps) == 1
        assert sleeps[0] == pytest.approx(30.3, rel=1e-3)
        assert client.used_weight_1m == 100

    _run(_exercise())


def test_http_429_retries_and_honors_retry_after() -> None:
    """HTTP 429 is retried using Retry-After when larger than backoff."""
    calls = {"count": 0}
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(
                429,
                text="too many requests",
                headers={"Retry-After": "3"},
            )
        return httpx.Response(200, json={"recovered": True})

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)

    async def _exercise() -> None:
        with patch("cqros.ingestion.client.asyncio.sleep", side_effect=_sleep):
            async with BinanceClient(
                client=_mock_client(handler),
                max_retries=2,
                backoff_factor=0.5,
                jitter_ratio=0.0,
                rate_limiter=_fast_limiter(),
            ) as client:
                payload = await client.get_exchange_info()

        assert payload == {"recovered": True}
        assert calls["count"] == 2
        assert sleeps == [3.0]

    _run(_exercise())


def test_http_429_raises_after_retries_exhausted() -> None:
    """Persistent HTTP 429 raises ExchangeRateLimitError after max retries."""
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        calls["count"] += 1
        return httpx.Response(429, text="slow down")

    async def _exercise() -> None:
        with patch("cqros.ingestion.client.asyncio.sleep", new_callable=AsyncMock):
            async with BinanceClient(
                client=_mock_client(handler),
                max_retries=2,
                backoff_factor=0.1,
                jitter_ratio=0.0,
                rate_limiter=_fast_limiter(),
            ) as client:
                with pytest.raises(ExchangeRateLimitError):
                    await client.get_exchange_info()

        assert calls["count"] == 3

    _run(_exercise())


def test_http_418_waits_for_ban_until_and_resumes() -> None:
    """HTTP 418 parses ban-until, waits with safety margin, then resumes."""
    calls = {"count": 0}
    sleeps: list[float] = []
    ban_until_ms = int(datetime.now(UTC).timestamp() * 1000) + 5_000

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(
                418,
                json={
                    "code": -1003,
                    "msg": (
                        "Way too many requests; IP banned until "
                        f"{ban_until_ms}. Please use websocket."
                    ),
                },
            )
        return httpx.Response(200, json={"resumed": True})

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)

    async def _exercise() -> None:
        with patch("cqros.ingestion.client.asyncio.sleep", side_effect=_sleep):
            async with BinanceClient(
                client=_mock_client(handler),
                max_retries=0,
                ban_safety_margin_seconds=1.0,
                rate_limiter=_fast_limiter(),
            ) as client:
                payload = await client.get_exchange_info()

        assert payload == {"resumed": True}
        assert calls["count"] == 2
        assert len(sleeps) == 1
        assert 4.0 <= sleeps[0] <= 7.0

    _run(_exercise())


def test_timeout_is_retried_then_succeeds() -> None:
    """TimeoutException is retried with backoff before a successful response."""
    calls = {"count": 0}
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.ReadTimeout("read timed out")
        return httpx.Response(200, json={"ok": True})

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)

    async def _exercise() -> None:
        with patch("cqros.ingestion.client.asyncio.sleep", side_effect=_sleep):
            async with BinanceClient(
                client=_mock_client(handler),
                max_retries=2,
                backoff_factor=0.25,
                jitter_ratio=0.0,
                rate_limiter=_fast_limiter(),
            ) as client:
                payload = await client.get_exchange_info()

        assert payload == {"ok": True}
        assert calls["count"] == 2
        assert sleeps == [0.25]

    _run(_exercise())


def test_connect_error_is_retried_then_succeeds() -> None:
    """ConnectError is retried as a transient network failure."""
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.ConnectError("connection refused")
        return httpx.Response(200, json={"ok": True})

    async def _exercise() -> None:
        with patch("cqros.ingestion.client.asyncio.sleep", new_callable=AsyncMock):
            async with BinanceClient(
                client=_mock_client(handler),
                max_retries=2,
                backoff_factor=0.1,
                jitter_ratio=0.0,
                rate_limiter=_fast_limiter(),
            ) as client:
                payload = await client.get_exchange_info()

        assert payload == {"ok": True}
        assert calls["count"] == 2

    _run(_exercise())


def test_retries_http_502_and_504() -> None:
    """HTTP 502 and 504 are included in the automatic retry set."""

    async def _recover(status: int) -> object:
        state = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            del request
            state["count"] += 1
            if state["count"] == 1:
                return httpx.Response(status, text="gateway")
            return httpx.Response(200, json={"status": status})

        with patch("cqros.ingestion.client.asyncio.sleep", new_callable=AsyncMock):
            async with BinanceClient(
                client=_mock_client(handler),
                max_retries=1,
                backoff_factor=0.0,
                jitter_ratio=0.0,
                rate_limiter=_fast_limiter(),
            ) as client:
                return await client.get_exchange_info()

    assert _run(_recover(502)) == {"status": 502}
    assert _run(_recover(504)) == {"status": 504}
