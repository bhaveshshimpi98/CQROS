"""CQROS Binance historical long/short ratio downloader.

Purpose:
    Plan and execute historical long/short ratio downloads from Binance
    USDⓈ-M Futures across the three supported positioning endpoints,
    persisting each dataset into an independent repository namespace.

Responsibilities:
    - Represent immutable ``LongShortDownloadTask`` time-range units
    - Split long ranges into sequential tasks via ``LongShortDownloadPlanner``
    - Fetch global account, top-trader account, and top-trader position ratios
      through ``BinanceClient``
    - Persist each dataset independently through ``MarketDataRepository``
    - Paginate until each requested window is complete
    - Validate every exchange response before conversion
    - Keep planning and execution separated
    - Remain free of filesystem path construction, feature engineering, and
      research logic

Dependencies:
    ``polars``, ``cqros.core``, ``cqros.ingestion.chunk_sizing``,
    ``cqros.ingestion.client``, and ``cqros.storage.repository``.

Public API:
    ``LongShortRatioKind``, ``LongShortDownloadTask``,
    ``LongShortDownloadPlanner``, ``LongShortDownloader``, and the default
    limit / period / chunk constants listed in ``__all__``.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, cast

import polars as pl

from cqros.core.constants import (
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
)
from cqros.core.exceptions import ValidationError
from cqros.core.types import (
    Exchange,
    Market,
    Symbol,
    Timeframe,
    UnixTimestampMs,
)
from cqros.ingestion.chunk_sizing import (
    DEFAULT_CHUNK_SAFETY_FACTOR,
    timeframe_duration_ms,
)
from cqros.ingestion.client import BinanceClient
from cqros.ingestion.resume import (
    DownloadResult,
    DownloadStatus,
    coerce_latest_timestamp,
    resolve_resume_window,
)
from cqros.storage.repository import MarketDataRepository

__all__ = [
    "DEFAULT_LONG_SHORT_CHUNK_SAFETY_FACTOR",
    "DEFAULT_LONG_SHORT_PERIOD",
    "DEFAULT_LONG_SHORT_REQUEST_LIMIT",
    "LONG_SHORT_PERIODS",
    "LongShortRatioKind",
    "LongShortDownloadTask",
    "LongShortDownloadPlanner",
    "LongShortDownloader",
]

LONG_SHORT_PERIODS: Final[frozenset[str]] = frozenset(
    {
        "5m",
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "6h",
        "12h",
        "1d",
    }
)
DEFAULT_LONG_SHORT_PERIOD: Final[Timeframe] = "5m"
DEFAULT_LONG_SHORT_REQUEST_LIMIT: Final[int] = 500
DEFAULT_LONG_SHORT_CHUNK_SAFETY_FACTOR: Final[float] = DEFAULT_CHUNK_SAFETY_FACTOR

_EXCHANGE: Final[Exchange] = EXCHANGE_BINANCE
_MARKET: Final[Market] = MARKET_USDT_PERPETUAL

_LONG_SHORT_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "symbol": pl.String,
        "timestamp": pl.Int64,
        "long_account": pl.Float64,
        "short_account": pl.Float64,
        "long_short_ratio": pl.Float64,
    }
)

_logger = logging.getLogger(__name__)

type _HistoryFetcher = Callable[
    ...,
    Awaitable[object],
]
type _PartitionSaver = Callable[..., None]


class LongShortRatioKind(StrEnum):
    """Supported Binance long/short ratio dataset kinds.

    Values are also used as repository dataset identifiers / storage
    namespaces.
    """

    GLOBAL_ACCOUNT = "global_long_short_account_ratio"
    TOP_TRADER_ACCOUNT = "top_long_short_account_ratio"
    TOP_TRADER_POSITION = "top_long_short_position_ratio"


@dataclass(frozen=True, slots=True)
class LongShortDownloadTask:
    """Immutable unit of work for a single long/short ratio download window.

    Attributes:
        symbol: Tradeable symbol (for example ``BTCUSDT``).
        period: Binance aggregation period (for example ``5m``).
        kind: Long/short dataset kind selecting the Binance endpoint and
            storage namespace.
        start_time: Inclusive window start as UTC Unix milliseconds.
        end_time: Inclusive window end as UTC Unix milliseconds.
    """

    symbol: Symbol
    period: Timeframe
    kind: LongShortRatioKind
    start_time: UnixTimestampMs
    end_time: UnixTimestampMs


class LongShortDownloadPlanner:
    """Split a historical long/short range into sequential download tasks.

    By default, chunk duration is derived from the requested period duration,
    request limit, and safety factor so each task stays near exchange page
    capacity. Callers may supply a fixed ``chunk_size_ms`` override. Produced
    tasks are contiguous, covering ``[start_time, end_time]`` without gaps or
    overlaps.

    Args:
        chunk_size_ms: Optional fixed inclusive window length per task in
            milliseconds. When set, period-aware sizing is disabled.
        request_limit: Maximum rows per request used by the default adaptive
            chunk sizing.
        safety_factor: Fraction of ``request_limit`` retained as headroom by
            the default adaptive sizing.

    Raises:
        ValidationError: If configuration is invalid.
    """

    __slots__ = ("_chunk_size_ms", "_request_limit", "_safety_factor")

    _chunk_size_ms: int | None
    _request_limit: int
    _safety_factor: float

    def __init__(
        self,
        *,
        chunk_size_ms: int | None = None,
        request_limit: int = DEFAULT_LONG_SHORT_REQUEST_LIMIT,
        safety_factor: float = DEFAULT_LONG_SHORT_CHUNK_SAFETY_FACTOR,
    ) -> None:
        """Initialize planner configuration.

        Args:
            chunk_size_ms: Optional fixed inclusive window length per task.
            request_limit: Maximum rows per request for adaptive sizing.
            safety_factor: Adaptive safety factor in ``(0, 1]``.

        Raises:
            ValidationError: If configuration is invalid.
        """
        if chunk_size_ms is not None and chunk_size_ms <= 0:
            raise ValidationError(
                "chunk_size_ms must be greater than 0",
                error_code="INGESTION-LONG-SHORT-001",
                details={"parameter": "chunk_size_ms", "value": chunk_size_ms},
            )
        if request_limit <= 0:
            raise ValidationError(
                "request_limit must be greater than 0",
                error_code="INGESTION-LONG-SHORT-002",
                details={"parameter": "request_limit", "value": request_limit},
            )
        if not (0.0 < safety_factor <= 1.0):
            raise ValidationError(
                "safety_factor must be in (0, 1]",
                error_code="INGESTION-LONG-SHORT-003",
                details={"parameter": "safety_factor", "value": safety_factor},
            )
        self._chunk_size_ms = chunk_size_ms
        self._request_limit = request_limit
        self._safety_factor = safety_factor

    @property
    def chunk_size_ms(self) -> int | None:
        """Return the fixed chunk size when configured; otherwise ``None``."""
        return self._chunk_size_ms

    def plan(
        self,
        *,
        symbol: Symbol,
        period: Timeframe,
        kind: LongShortRatioKind,
        start_time: UnixTimestampMs,
        end_time: UnixTimestampMs,
    ) -> tuple[LongShortDownloadTask, ...]:
        """Produce sequential long/short download tasks covering the range.

        Args:
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            period: Binance aggregation period.
            kind: Long/short dataset kind.
            start_time: Inclusive range start as UTC Unix milliseconds.
            end_time: Inclusive range end as UTC Unix milliseconds.

        Returns:
            Immutable tuple of ``LongShortDownloadTask`` instances in
            chronological order. Returns an empty tuple when ``start_time`` is
            greater than ``end_time``.

        Raises:
            ValidationError: If timestamps, ``period``, or ``kind`` are invalid.
        """
        _require_unix_ms(start_time, parameter="start_time")
        _require_unix_ms(end_time, parameter="end_time")
        _require_period(period)
        kind = _require_kind(kind)

        if start_time > end_time:
            return ()

        chunk_size = self._resolve_chunk_size_ms(period)
        tasks: list[LongShortDownloadTask] = []
        cursor = start_time
        while cursor <= end_time:
            chunk_end = min(cursor + chunk_size - 1, end_time)
            tasks.append(
                LongShortDownloadTask(
                    symbol=symbol,
                    period=period,
                    kind=kind,
                    start_time=cursor,
                    end_time=chunk_end,
                )
            )
            cursor = chunk_end + 1

        return tuple(tasks)

    def _resolve_chunk_size_ms(self, period: Timeframe) -> int:
        """Resolve the inclusive chunk duration for ``period``.

        Args:
            period: Validated aggregation period.

        Returns:
            Inclusive chunk duration in milliseconds.

        Raises:
            ValidationError: If adaptive sizing cannot derive a positive chunk.
        """
        if self._chunk_size_ms is not None:
            return self._chunk_size_ms

        period_ms = timeframe_duration_ms(period)
        effective_count = max(1, int(self._request_limit * self._safety_factor))
        chunk_size = effective_count * period_ms
        if chunk_size <= 0:
            raise ValidationError(
                "derived long/short chunk size must be greater than 0",
                error_code="INGESTION-LONG-SHORT-004",
                details={
                    "period": period,
                    "period_ms": period_ms,
                    "request_limit": self._request_limit,
                    "safety_factor": self._safety_factor,
                    "chunk_size_ms": chunk_size,
                },
            )
        return chunk_size


class LongShortDownloader:
    """Execute planned historical long/short ratio downloads into storage.

    Planning remains the responsibility of ``LongShortDownloadPlanner``. This
    class fetches one of the three supported long/short endpoints for each
    task, converts vendor rows into tabular frames, and persists year
    partitions through the matching ``MarketDataRepository`` namespace.
    Filesystem paths are never constructed here.

    Args:
        client: Open ``BinanceClient`` used for long/short requests.
        repository: Market-data repository used for persistence.
        planner: Planner that splits long ranges into sequential tasks.
        request_limit: Maximum records requested per Binance API call.
        logger: Optional logger instance. Defaults to the module logger.

    Raises:
        ValidationError: If ``request_limit`` is not positive.
    """

    __slots__ = (
        "_client",
        "_repository",
        "_planner",
        "_request_limit",
        "_logger",
        "_fetchers",
        "_savers",
    )

    _client: BinanceClient
    _repository: MarketDataRepository
    _planner: LongShortDownloadPlanner
    _request_limit: int
    _logger: logging.Logger
    _fetchers: Mapping[LongShortRatioKind, _HistoryFetcher]
    _savers: Mapping[LongShortRatioKind, _PartitionSaver]

    def __init__(
        self,
        client: BinanceClient,
        repository: MarketDataRepository,
        planner: LongShortDownloadPlanner,
        *,
        request_limit: int = DEFAULT_LONG_SHORT_REQUEST_LIMIT,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the downloader with injected collaborators.

        Args:
            client: Binance USDⓈ-M Futures REST client.
            repository: Repository used to persist long/short partitions.
            planner: Long/short download range planner.
            request_limit: Maximum records per API call.
            logger: Optional logger instance.

        Raises:
            ValidationError: If ``request_limit`` is not positive.
        """
        if request_limit <= 0:
            raise ValidationError(
                "request_limit must be greater than 0",
                error_code="INGESTION-LONG-SHORT-005",
                details={"parameter": "request_limit", "value": request_limit},
            )
        self._client = client
        self._repository = repository
        self._planner = planner
        self._request_limit = request_limit
        self._logger = logger if logger is not None else _logger
        self._fetchers = {
            LongShortRatioKind.GLOBAL_ACCOUNT: client.get_global_long_short_account_ratio,
            LongShortRatioKind.TOP_TRADER_ACCOUNT: (client.get_top_long_short_account_ratio),
            LongShortRatioKind.TOP_TRADER_POSITION: (client.get_top_long_short_position_ratio),
        }
        self._savers = {
            LongShortRatioKind.GLOBAL_ACCOUNT: (repository.save_global_long_short_account_ratio),
            LongShortRatioKind.TOP_TRADER_ACCOUNT: (repository.save_top_long_short_account_ratio),
            LongShortRatioKind.TOP_TRADER_POSITION: (repository.save_top_long_short_position_ratio),
        }

    @property
    def request_limit(self) -> int:
        """Return the configured per-request long/short limit."""
        return self._request_limit

    async def fetch_symbol(
        self,
        *,
        symbol: Symbol,
        kind: LongShortRatioKind,
        start_time: UnixTimestampMs,
        end_time: UnixTimestampMs,
        period: Timeframe = DEFAULT_LONG_SHORT_PERIOD,
    ) -> pl.DataFrame:
        """Download historical long/short ratios for a symbol without persisting.

        Args:
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            kind: Long/short dataset kind selecting the Binance endpoint.
            start_time: Inclusive range start as UTC Unix milliseconds.
            end_time: Inclusive range end as UTC Unix milliseconds.
            period: Binance aggregation period.

        Returns:
            Combined long/short DataFrame for the requested range. Empty when
            the exchange returns no rows or ``start_time`` is after
            ``end_time``.

        Raises:
            ValidationError: If planner inputs or payloads are invalid.
            ExchangeError: Propagated from ``BinanceClient`` transport failures.
        """
        kind = _require_kind(kind)
        tasks = self._planner.plan(
            symbol=symbol,
            period=period,
            kind=kind,
            start_time=start_time,
            end_time=end_time,
        )
        self._logger.info(
            "Starting symbol long/short fetch",
            extra={
                "exchange": _EXCHANGE,
                "market": _MARKET,
                "symbol": symbol,
                "kind": kind.value,
                "period": period,
                "start_time": start_time,
                "end_time": end_time,
                "task_count": len(tasks),
            },
        )

        frames: list[pl.DataFrame] = []
        for task in tasks:
            frame = await self._execute_task(task)
            if frame.height > 0:
                frames.append(frame)

        if not frames:
            self._logger.info(
                "Symbol long/short fetch produced no rows",
                extra={
                    "symbol": symbol,
                    "kind": kind.value,
                    "period": period,
                    "start_time": start_time,
                    "end_time": end_time,
                },
            )
            return pl.DataFrame(schema=_LONG_SHORT_SCHEMA)

        combined = pl.concat(frames, how="vertical")
        self._logger.info(
            "Completed symbol long/short fetch",
            extra={
                "symbol": symbol,
                "kind": kind.value,
                "period": period,
                "start_time": start_time,
                "end_time": end_time,
                "rows": combined.height,
            },
        )
        return combined

    async def download_symbol(
        self,
        *,
        symbol: Symbol,
        kind: LongShortRatioKind,
        start_time: UnixTimestampMs,
        end_time: UnixTimestampMs,
        period: Timeframe = DEFAULT_LONG_SHORT_PERIOD,
    ) -> DownloadResult:
        """Download and persist historical long/short ratios for one symbol.

        Resumes from the latest persisted ``timestamp`` for ``symbol``/``period``
        and ``kind``.

        Args:
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            kind: Long/short dataset kind selecting the endpoint and storage
                namespace.
            start_time: Inclusive range start as UTC Unix milliseconds.
            end_time: Inclusive range end as UTC Unix milliseconds.
            period: Binance aggregation period.

        Returns:
            Immutable ``DownloadResult`` describing full, updated, or skipped.

        Raises:
            ValidationError: If planner inputs or payloads are invalid.
            ExchangeError: Propagated from ``BinanceClient`` transport failures.
        """
        kind = _require_kind(kind)
        latest = coerce_latest_timestamp(
            self._repository.get_latest_long_short_timestamp(
                exchange=_EXCHANGE,
                market=_MARKET,
                symbol=symbol,
                timeframe=period,
                dataset=kind.value,
            )
        )
        window = resolve_resume_window(
            latest_timestamp=latest,
            requested_start=start_time,
            requested_end=end_time,
            interval_ms=timeframe_duration_ms(period),
        )
        if window.status is DownloadStatus.SKIPPED:
            self._logger.info(
                "Skipping long/short download; storage already current",
                extra={
                    "symbol": symbol,
                    "kind": kind.value,
                    "period": period,
                    "start_time": start_time,
                    "end_time": end_time,
                    "adjusted_start": window.start_time,
                },
            )
            return DownloadResult(status=DownloadStatus.SKIPPED, rows_downloaded=0)

        combined = await self.fetch_symbol(
            symbol=symbol,
            kind=kind,
            period=period,
            start_time=window.start_time,
            end_time=window.end_time,
        )
        if combined.height > 0:
            self._persist_long_short(
                combined,
                symbol=symbol,
                period=period,
                kind=kind,
            )
            self._logger.info(
                "Completed symbol long/short download",
                extra={
                    "symbol": symbol,
                    "kind": kind.value,
                    "period": period,
                    "start_time": window.start_time,
                    "end_time": window.end_time,
                    "rows": combined.height,
                    "status": window.status.value,
                },
            )
        return DownloadResult(
            status=window.status,
            rows_downloaded=combined.height,
        )

    async def download_universe(
        self,
        symbols: Sequence[Symbol],
        *,
        kind: LongShortRatioKind,
        start_time: UnixTimestampMs,
        end_time: UnixTimestampMs,
        period: Timeframe = DEFAULT_LONG_SHORT_PERIOD,
    ) -> None:
        """Download and persist historical long/short ratios for many symbols.

        Symbols are processed sequentially so rate-limit pressure remains
        predictable. Each symbol is planned and executed independently.

        Args:
            symbols: Ordered symbols to download.
            kind: Long/short dataset kind selecting the endpoint and storage
                namespace.
            start_time: Inclusive range start as UTC Unix milliseconds.
            end_time: Inclusive range end as UTC Unix milliseconds.
            period: Binance aggregation period.

        Raises:
            ValidationError: If planner inputs or payloads are invalid.
            ExchangeError: Propagated from ``BinanceClient`` transport failures.
        """
        kind = _require_kind(kind)
        self._logger.info(
            "Starting universe long/short download",
            extra={
                "exchange": _EXCHANGE,
                "market": _MARKET,
                "symbol_count": len(symbols),
                "kind": kind.value,
                "period": period,
                "start_time": start_time,
                "end_time": end_time,
            },
        )
        for symbol in symbols:
            await self.download_symbol(
                symbol=symbol,
                kind=kind,
                period=period,
                start_time=start_time,
                end_time=end_time,
            )
        self._logger.info(
            "Completed universe long/short download",
            extra={
                "symbol_count": len(symbols),
                "kind": kind.value,
                "period": period,
                "start_time": start_time,
                "end_time": end_time,
            },
        )

    async def _execute_task(self, task: LongShortDownloadTask) -> pl.DataFrame:
        """Fetch all long/short rows for a single task window.

        Args:
            task: Immutable download window to execute.

        Returns:
            Long/short DataFrame for the task range. May be empty.

        Raises:
            ValidationError: If a payload cannot be parsed.
            ExchangeError: Propagated from ``BinanceClient`` transport failures.
        """
        self._logger.debug(
            "Executing long/short download task",
            extra={
                "symbol": task.symbol,
                "kind": task.kind.value,
                "period": task.period,
                "start_time": task.start_time,
                "end_time": task.end_time,
            },
        )

        fetch = self._fetchers[task.kind]
        frames: list[pl.DataFrame] = []
        cursor = task.start_time
        while cursor <= task.end_time:
            payload = await fetch(
                task.symbol,
                task.period,
                start_time=cursor,
                end_time=task.end_time,
                limit=self._request_limit,
            )
            rows = _parse_long_short_rows(
                payload,
                expected_symbol=task.symbol,
            )
            if not rows:
                break

            frames.append(_long_short_rows_to_dataframe(rows))
            last_timestamp = rows[-1][1]
            next_cursor = last_timestamp + 1
            if next_cursor <= cursor:
                break
            if len(rows) < self._request_limit:
                break
            cursor = next_cursor

        if not frames:
            return pl.DataFrame(schema=_LONG_SHORT_SCHEMA)
        return pl.concat(frames, how="vertical")

    def _persist_long_short(
        self,
        frame: pl.DataFrame,
        *,
        symbol: Symbol,
        period: Timeframe,
        kind: LongShortRatioKind,
    ) -> None:
        """Persist a long/short frame as year partitions for ``kind``.

        Args:
            frame: Combined long/short rows for one symbol/period/kind download.
            symbol: Tradeable symbol.
            period: Aggregation period used as the repository timeframe key.
            kind: Dataset kind selecting the repository save method.
        """
        save = self._savers[kind]
        year_expr = pl.from_epoch(pl.col("timestamp"), time_unit="ms").dt.year()
        # Polars DataFrame stubs expose Unknown aliases under pyright strict.
        year_frame = frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            year_expr.alias("_year")
        )
        years = year_frame.get_column("_year").unique().sort().to_list()
        for year in years:
            partition = year_frame.filter(  # pyright: ignore[reportUnknownMemberType]
                pl.col("_year") == year
            ).drop("_year")
            save(
                partition,
                exchange=_EXCHANGE,
                market=_MARKET,
                symbol=symbol,
                timeframe=period,
                year=int(year),
            )
            self._logger.debug(
                "Persisted long/short year partition",
                extra={
                    "symbol": symbol,
                    "kind": kind.value,
                    "period": period,
                    "year": int(year),
                    "rows": partition.height,
                },
            )


def _require_unix_ms(value: object, *, parameter: str) -> UnixTimestampMs:
    """Validate a Unix-millisecond timestamp parameter.

    Args:
        value: Candidate timestamp value.
        parameter: Parameter name for error context.

    Returns:
        The validated integer timestamp.

    Raises:
        ValidationError: If ``value`` is not an ``int``.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(
            f"{parameter} must be an int Unix timestamp in milliseconds",
            error_code="INGESTION-LONG-SHORT-006",
            details={"parameter": parameter, "type": type(value).__name__},
        )
    return value


def _require_period(period: object) -> Timeframe:
    """Validate a Binance long/short aggregation period.

    Args:
        period: Candidate period identifier.

    Returns:
        The validated period string.

    Raises:
        ValidationError: If ``period`` is unsupported.
    """
    if not isinstance(period, str):
        raise ValidationError(
            "period must be a string long/short interval",
            error_code="INGESTION-LONG-SHORT-007",
            details={"parameter": "period", "type": type(period).__name__},
        )
    if period not in LONG_SHORT_PERIODS:
        raise ValidationError(
            f"unsupported long/short period: {period!r}",
            error_code="INGESTION-LONG-SHORT-008",
            details={
                "parameter": "period",
                "value": period,
                "supported": sorted(LONG_SHORT_PERIODS),
            },
        )
    return period


def _require_kind(kind: object) -> LongShortRatioKind:
    """Validate a long/short dataset kind.

    Args:
        kind: Candidate dataset kind.

    Returns:
        The validated ``LongShortRatioKind``.

    Raises:
        ValidationError: If ``kind`` is unsupported.
    """
    if isinstance(kind, LongShortRatioKind):
        return kind
    if isinstance(kind, str):
        try:
            return LongShortRatioKind(kind)
        except ValueError:
            pass
    raise ValidationError(
        "kind must be a supported LongShortRatioKind",
        error_code="INGESTION-LONG-SHORT-009",
        details={
            "parameter": "kind",
            "value": kind if isinstance(kind, str) else type(kind).__name__,
            "supported": [member.value for member in LongShortRatioKind],
        },
    )


def _parse_long_short_rows(
    payload: object,
    *,
    expected_symbol: Symbol,
) -> list[tuple[str, int, float, float, float]]:
    """Parse a Binance long/short payload into typed row tuples.

    Args:
        payload: Decoded JSON body from a long/short history endpoint.
        expected_symbol: Symbol requested from the exchange.

    Returns:
        List of
        ``(symbol, timestamp, long_account, short_account, long_short_ratio)``
        tuples.

    Raises:
        ValidationError: If the payload structure is invalid.
    """
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise ValidationError(
            "long/short payload must be a sequence",
            error_code="INGESTION-LONG-SHORT-010",
            details={"type": type(payload).__name__},
        )

    rows: list[tuple[str, int, float, float, float]] = []
    for index, entry in enumerate(cast(Sequence[object], payload)):
        rows.append(
            _parse_long_short_row(
                entry,
                index=index,
                expected_symbol=expected_symbol,
            )
        )
    return rows


def _parse_long_short_row(
    entry: object,
    *,
    index: int,
    expected_symbol: Symbol,
) -> tuple[str, int, float, float, float]:
    """Parse a single Binance long/short object into typed fields.

    Args:
        entry: Raw long/short object from the exchange payload.
        index: Row index for error context.
        expected_symbol: Symbol requested from the exchange.

    Returns:
        Tuple of
        ``(symbol, timestamp, long_account, short_account, long_short_ratio)``.

    Raises:
        ValidationError: If the row cannot be parsed.
    """
    if not isinstance(entry, Mapping):
        raise ValidationError(
            "long/short row must be a mapping",
            error_code="INGESTION-LONG-SHORT-011",
            details={"index": index, "type": type(entry).__name__},
        )

    values = cast(Mapping[str, object], entry)
    required = (
        "symbol",
        "timestamp",
        "longAccount",
        "shortAccount",
        "longShortRatio",
    )
    missing = [field for field in required if field not in values]
    if missing:
        raise ValidationError(
            "long/short row is missing required fields",
            error_code="INGESTION-LONG-SHORT-012",
            details={"index": index, "missing": missing},
        )

    symbol = _as_symbol(values["symbol"], index=index)
    if symbol != expected_symbol:
        raise ValidationError(
            "long/short row symbol does not match requested symbol",
            error_code="INGESTION-LONG-SHORT-013",
            details={
                "index": index,
                "expected": expected_symbol,
                "actual": symbol,
            },
        )

    return (
        symbol,
        _as_int(values["timestamp"], field="timestamp", index=index),
        _as_float(values["longAccount"], field="long_account", index=index),
        _as_float(values["shortAccount"], field="short_account", index=index),
        _as_float(values["longShortRatio"], field="long_short_ratio", index=index),
    )


def _long_short_rows_to_dataframe(
    rows: Sequence[tuple[str, int, float, float, float]],
) -> pl.DataFrame:
    """Convert parsed long/short rows into a canonical DataFrame.

    Args:
        rows: Parsed long/short tuples.

    Returns:
        Eager Polars DataFrame with the canonical raw long/short columns.
    """
    return pl.DataFrame(
        {
            "symbol": [row[0] for row in rows],
            "timestamp": [row[1] for row in rows],
            "long_account": [row[2] for row in rows],
            "short_account": [row[3] for row in rows],
            "long_short_ratio": [row[4] for row in rows],
        },
        schema=_LONG_SHORT_SCHEMA,
    )


def _as_symbol(value: object, *, index: int) -> str:
    """Coerce a long/short symbol field to ``str``.

    Args:
        value: Raw field value.
        index: Row index for error context.

    Returns:
        Symbol string.

    Raises:
        ValidationError: If coercion fails.
    """
    if not isinstance(value, str) or not value:
        raise ValidationError(
            "long_short.symbol must be a non-empty string",
            error_code="INGESTION-LONG-SHORT-014",
            details={"index": index, "type": type(value).__name__},
        )
    return value


def _as_int(value: object, *, field: str, index: int) -> int:
    """Coerce a long/short field to ``int``.

    Args:
        value: Raw field value.
        field: Field name for error context.
        index: Row index for error context.

    Returns:
        Integer value.

    Raises:
        ValidationError: If coercion fails.
    """
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValidationError(
            f"long_short.{field} must be an int-compatible value",
            error_code="INGESTION-LONG-SHORT-015",
            details={"index": index, "field": field, "type": type(value).__name__},
        )
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"long_short.{field} must be an int-compatible value",
            error_code="INGESTION-LONG-SHORT-015",
            details={"index": index, "field": field, "value": value},
        ) from exc


def _as_float(value: object, *, field: str, index: int) -> float:
    """Coerce a long/short field to ``float``.

    Args:
        value: Raw field value.
        field: Field name for error context.
        index: Row index for error context.

    Returns:
        Floating-point value.

    Raises:
        ValidationError: If coercion fails.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValidationError(
            f"long_short.{field} must be a float-compatible value",
            error_code="INGESTION-LONG-SHORT-016",
            details={"index": index, "field": field, "type": type(value).__name__},
        )
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"long_short.{field} must be a float-compatible value",
            error_code="INGESTION-LONG-SHORT-016",
            details={"index": index, "field": field, "value": value},
        ) from exc
