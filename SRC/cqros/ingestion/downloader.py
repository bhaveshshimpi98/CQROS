"""CQROS Binance historical market-data downloader.

Purpose:
    Plan and execute historical kline downloads from Binance USDⓈ-M Futures,
    persisting raw OHLCV partitions through ``MarketDataRepository``.

Responsibilities:
    - Represent immutable ``DownloadTask`` time-range units
    - Split long ranges into sequential tasks via ``DownloadPlanner``
    - Derive timeframe-aware chunk sizes through injectable strategies
    - Fetch klines through ``BinanceClient`` and persist via repository
    - Keep planning and execution separated
    - Remain free of filesystem path construction and feature engineering

Dependencies:
    ``polars``, ``cqros.core``, ``cqros.ingestion.chunk_sizing``,
    ``cqros.ingestion.client``, and ``cqros.storage.repository``.

Public API:
    ``DownloadTask``, ``DownloadPlanner``, ``HistoricalDownloader``
    (including ``fetch_symbol`` / ``download_symbol``), and the default chunk /
    request-limit constants listed in ``__all__``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, cast

import polars as pl

from cqros.core.constants import (
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
    MILLISECONDS_PER_DAY,
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
    AdaptiveChunkSizingStrategy,
    ChunkSizingStrategy,
    FixedChunkSizingStrategy,
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
    "DEFAULT_DOWNLOAD_BATCH_SIZE",
    "DEFAULT_DOWNLOAD_CHUNK_SIZE_MS",
    "DEFAULT_DOWNLOAD_WORKERS",
    "DEFAULT_KLINE_REQUEST_LIMIT",
    "DownloadTask",
    "DownloadPlanner",
    "HistoricalDownloader",
]

DEFAULT_DOWNLOAD_CHUNK_SIZE_MS: Final[int] = MILLISECONDS_PER_DAY
DEFAULT_KLINE_REQUEST_LIMIT: Final[int] = 1_500
DEFAULT_DOWNLOAD_WORKERS: Final[int] = 8
DEFAULT_DOWNLOAD_BATCH_SIZE: Final[int] = 50

_EXCHANGE: Final[Exchange] = EXCHANGE_BINANCE
_MARKET: Final[Market] = MARKET_USDT_PERPETUAL

_OHLCV_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "symbol": pl.String,
        "timeframe": pl.String,
        "open_time": pl.Int64,
        "close_time": pl.Int64,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
        "quote_volume": pl.Float64,
        "trade_count": pl.Int64,
    }
)

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DownloadTask:
    """Immutable unit of work for a single historical download window.

    Attributes:
        symbol: Tradeable symbol (for example ``BTCUSDT``).
        timeframe: Bar interval identifier (for example ``1m``).
        start_time: Inclusive window start as UTC Unix milliseconds.
        end_time: Inclusive window end as UTC Unix milliseconds.
    """

    symbol: Symbol
    timeframe: Timeframe
    start_time: UnixTimestampMs
    end_time: UnixTimestampMs


class DownloadPlanner:
    """Split a historical download range into sequential ``DownloadTask`` values.

    By default, chunk duration is derived adaptively from timeframe duration,
    ``kline_limit``, and safety factor so each task stays near exchange request
    capacity. Callers may inject a ``ChunkSizingStrategy`` or supply a fixed
    ``chunk_size_ms`` override. Produced tasks are contiguous, covering
    ``[start_time, end_time]`` without gaps or overlaps.

    Args:
        chunk_size_ms: Optional fixed inclusive window length per task in
            milliseconds. When set, timeframe-aware sizing is disabled.
        chunk_strategy: Optional injected sizing strategy. Mutually exclusive
            with ``chunk_size_ms``.
        kline_limit: Maximum klines per request used by the default adaptive
            strategy.
        safety_factor: Fraction of ``kline_limit`` retained as headroom by the
            default adaptive strategy.

    Raises:
        ValidationError: If configuration is invalid.
    """

    __slots__ = ("_chunk_strategy", "_fixed_chunk_size_ms")

    _chunk_strategy: ChunkSizingStrategy
    _fixed_chunk_size_ms: int | None

    def __init__(
        self,
        *,
        chunk_size_ms: int | None = None,
        chunk_strategy: ChunkSizingStrategy | None = None,
        kline_limit: int = DEFAULT_KLINE_REQUEST_LIMIT,
        safety_factor: float = DEFAULT_CHUNK_SAFETY_FACTOR,
    ) -> None:
        """Initialize planner configuration.

        Args:
            chunk_size_ms: Optional fixed inclusive window length per task in
                milliseconds.
            chunk_strategy: Optional injected sizing strategy.
            kline_limit: Maximum klines per request for default adaptive sizing.
            safety_factor: Adaptive safety factor in ``(0, 1]``.

        Raises:
            ValidationError: If configuration is invalid.
        """
        if chunk_size_ms is not None and chunk_strategy is not None:
            raise ValidationError(
                "chunk_size_ms and chunk_strategy are mutually exclusive",
                error_code="INGESTION-DOWNLOADER-009",
                details={
                    "chunk_size_ms": chunk_size_ms,
                    "chunk_strategy": type(chunk_strategy).__name__,
                },
            )

        if chunk_size_ms is not None:
            if chunk_size_ms <= 0:
                raise ValidationError(
                    "chunk_size_ms must be greater than 0",
                    error_code="INGESTION-DOWNLOADER-001",
                    details={"parameter": "chunk_size_ms", "value": chunk_size_ms},
                )
            self._fixed_chunk_size_ms = chunk_size_ms
            self._chunk_strategy = FixedChunkSizingStrategy(size_ms=chunk_size_ms)
            return

        if chunk_strategy is not None:
            self._fixed_chunk_size_ms = None
            self._chunk_strategy = chunk_strategy
            return

        self._fixed_chunk_size_ms = None
        self._chunk_strategy = AdaptiveChunkSizingStrategy(
            kline_limit=kline_limit,
            safety_factor=safety_factor,
        )

    @property
    def chunk_size_ms(self) -> int | None:
        """Return the fixed chunk size when configured; otherwise ``None``."""
        return self._fixed_chunk_size_ms

    @property
    def chunk_strategy(self) -> ChunkSizingStrategy:
        """Return the injected or default chunk sizing strategy."""
        return self._chunk_strategy

    def plan(
        self,
        *,
        symbol: Symbol,
        timeframe: Timeframe,
        start_time: UnixTimestampMs,
        end_time: UnixTimestampMs,
    ) -> tuple[DownloadTask, ...]:
        """Produce sequential download tasks covering the requested range.

        Args:
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Bar interval identifier (for example ``1m``).
            start_time: Inclusive range start as UTC Unix milliseconds.
            end_time: Inclusive range end as UTC Unix milliseconds.

        Returns:
            Immutable tuple of ``DownloadTask`` instances in chronological
            order. Returns an empty tuple when ``start_time`` is greater than
            ``end_time``.

        Raises:
            ValidationError: If timestamps are invalid integers, or if the
                active strategy rejects ``timeframe``.
        """
        _require_unix_ms(start_time, parameter="start_time")
        _require_unix_ms(end_time, parameter="end_time")

        if start_time > end_time:
            return ()

        chunk_size = self._chunk_strategy.chunk_size_ms(timeframe)
        if chunk_size <= 0:
            raise ValidationError(
                "chunk strategy returned a non-positive chunk size",
                error_code="INGESTION-DOWNLOADER-010",
                details={
                    "timeframe": timeframe,
                    "chunk_size_ms": chunk_size,
                    "strategy": type(self._chunk_strategy).__name__,
                },
            )

        tasks: list[DownloadTask] = []
        cursor = start_time
        while cursor <= end_time:
            chunk_end = min(cursor + chunk_size - 1, end_time)
            tasks.append(
                DownloadTask(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=cursor,
                    end_time=chunk_end,
                )
            )
            cursor = chunk_end + 1

        return tuple(tasks)


class HistoricalDownloader:
    """Execute planned historical kline downloads into market-data storage.

    Planning remains the responsibility of ``DownloadPlanner``. This class
    fetches klines for each task, converts vendor rows into tabular OHLCV
    frames, and persists year partitions through ``MarketDataRepository``.
    Filesystem paths are never constructed here.

    Args:
        client: Open ``BinanceClient`` used for kline requests.
        repository: Market-data repository used for OHLCV persistence.
        planner: Planner that splits long ranges into sequential tasks.
        kline_limit: Maximum klines requested per Binance API call.
        workers: Maximum concurrent workers retained for pipeline configuration.
        batch_size: Maximum batch size retained for pipeline configuration.
        logger: Optional logger instance. Defaults to the module logger.

    Raises:
        ValidationError: If ``kline_limit``, ``workers``, or ``batch_size`` is
            not positive.
    """

    __slots__ = (
        "_client",
        "_repository",
        "_planner",
        "_kline_limit",
        "_workers",
        "_batch_size",
        "_logger",
    )

    _client: BinanceClient
    _repository: MarketDataRepository
    _planner: DownloadPlanner
    _kline_limit: int
    _workers: int
    _batch_size: int
    _logger: logging.Logger

    def __init__(
        self,
        client: BinanceClient,
        repository: MarketDataRepository,
        planner: DownloadPlanner,
        *,
        kline_limit: int = DEFAULT_KLINE_REQUEST_LIMIT,
        workers: int = DEFAULT_DOWNLOAD_WORKERS,
        batch_size: int = DEFAULT_DOWNLOAD_BATCH_SIZE,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the downloader with injected collaborators.

        Args:
            client: Binance USDⓈ-M Futures REST client.
            repository: Repository used to persist OHLCV partitions.
            planner: Download range planner.
            kline_limit: Maximum klines requested per API call.
            workers: Maximum concurrent workers for pipeline configuration.
            batch_size: Maximum batch size for pipeline configuration.
            logger: Optional logger instance.

        Raises:
            ValidationError: If ``kline_limit``, ``workers``, or ``batch_size``
                is not positive.
        """
        if kline_limit <= 0:
            raise ValidationError(
                "kline_limit must be greater than 0",
                error_code="INGESTION-DOWNLOADER-002",
                details={"parameter": "kline_limit", "value": kline_limit},
            )
        if workers < 1:
            raise ValidationError(
                "workers must be greater than or equal to 1",
                error_code="INGESTION-DOWNLOADER-011",
                details={"parameter": "workers", "value": workers},
            )
        if batch_size < 1:
            raise ValidationError(
                "batch_size must be greater than or equal to 1",
                error_code="INGESTION-DOWNLOADER-012",
                details={"parameter": "batch_size", "value": batch_size},
            )
        self._client = client
        self._repository = repository
        self._planner = planner
        self._kline_limit = kline_limit
        self._workers = workers
        self._batch_size = batch_size
        self._logger = logger if logger is not None else _logger

    @property
    def kline_limit(self) -> int:
        """Return the configured per-request kline limit."""
        return self._kline_limit

    @property
    def workers(self) -> int:
        """Return the configured maximum concurrent worker count."""
        return self._workers

    @property
    def batch_size(self) -> int:
        """Return the configured download batch size."""
        return self._batch_size

    async def fetch_symbol(
        self,
        *,
        symbol: Symbol,
        timeframe: Timeframe,
        start_time: UnixTimestampMs,
        end_time: UnixTimestampMs,
    ) -> pl.DataFrame:
        """Download historical OHLCV for a single symbol without persisting.

        Args:
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Bar interval identifier (for example ``1m``).
            start_time: Inclusive range start as UTC Unix milliseconds.
            end_time: Inclusive range end as UTC Unix milliseconds.

        Returns:
            Combined OHLCV DataFrame for the requested range. Empty when the
            exchange returns no rows or ``start_time`` is after ``end_time``.

        Raises:
            ValidationError: If planner inputs or kline payloads are invalid.
            ExchangeError: Propagated from ``BinanceClient`` transport failures.
        """
        tasks = self._planner.plan(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
        )
        self._logger.info(
            "Starting symbol historical fetch",
            extra={
                "exchange": _EXCHANGE,
                "market": _MARKET,
                "symbol": symbol,
                "timeframe": timeframe,
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
                "Symbol historical fetch produced no rows",
                extra={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "start_time": start_time,
                    "end_time": end_time,
                },
            )
            return pl.DataFrame(schema=_OHLCV_SCHEMA)

        combined = pl.concat(frames, how="vertical")
        self._logger.info(
            "Completed symbol historical fetch",
            extra={
                "symbol": symbol,
                "timeframe": timeframe,
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
        timeframe: Timeframe,
        start_time: UnixTimestampMs,
        end_time: UnixTimestampMs,
    ) -> DownloadResult:
        """Download and persist historical OHLCV for a single symbol.

        Resumes from the latest persisted ``open_time`` when storage already
        contains data for ``symbol``/``timeframe``. Overlapping rows are
        deduplicated by the repository on save.

        Args:
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Bar interval identifier (for example ``1m``).
            start_time: Inclusive range start as UTC Unix milliseconds.
            end_time: Inclusive range end as UTC Unix milliseconds.

        Returns:
            Immutable ``DownloadResult`` describing full, updated, or skipped.

        Raises:
            ValidationError: If planner inputs or kline payloads are invalid.
            ExchangeError: Propagated from ``BinanceClient`` transport failures.
        """
        latest = coerce_latest_timestamp(
            self._repository.get_latest_ohlcv_timestamp(
                exchange=_EXCHANGE,
                market=_MARKET,
                symbol=symbol,
                timeframe=timeframe,
            )
        )
        window = resolve_resume_window(
            latest_timestamp=latest,
            requested_start=start_time,
            requested_end=end_time,
            interval_ms=timeframe_duration_ms(timeframe),
        )
        if window.status is DownloadStatus.SKIPPED:
            self._logger.info(
                "Skipping historical download; storage already current",
                extra={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "start_time": start_time,
                    "end_time": end_time,
                    "adjusted_start": window.start_time,
                },
            )
            return DownloadResult(status=DownloadStatus.SKIPPED, rows_downloaded=0)

        combined = await self.fetch_symbol(
            symbol=symbol,
            timeframe=timeframe,
            start_time=window.start_time,
            end_time=window.end_time,
        )
        if combined.height > 0:
            self._persist_ohlcv(combined, symbol=symbol, timeframe=timeframe)
            self._logger.info(
                "Completed symbol historical download",
                extra={
                    "symbol": symbol,
                    "timeframe": timeframe,
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
        timeframe: Timeframe,
        start_time: UnixTimestampMs,
        end_time: UnixTimestampMs,
    ) -> None:
        """Download and persist historical OHLCV for multiple symbols.

        Symbols are processed sequentially so rate-limit pressure remains
        predictable. Each symbol is planned and executed independently.

        Args:
            symbols: Ordered symbols to download.
            timeframe: Bar interval identifier (for example ``1m``).
            start_time: Inclusive range start as UTC Unix milliseconds.
            end_time: Inclusive range end as UTC Unix milliseconds.

        Raises:
            ValidationError: If planner inputs or kline payloads are invalid.
            ExchangeError: Propagated from ``BinanceClient`` transport failures.
        """
        self._logger.info(
            "Starting universe historical download",
            extra={
                "exchange": _EXCHANGE,
                "market": _MARKET,
                "symbol_count": len(symbols),
                "timeframe": timeframe,
                "start_time": start_time,
                "end_time": end_time,
            },
        )
        for symbol in symbols:
            await self.download_symbol(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
            )
        self._logger.info(
            "Completed universe historical download",
            extra={
                "symbol_count": len(symbols),
                "timeframe": timeframe,
                "start_time": start_time,
                "end_time": end_time,
            },
        )

    async def _execute_task(self, task: DownloadTask) -> pl.DataFrame:
        """Fetch all klines for a single task window.

        Args:
            task: Immutable download window to execute.

        Returns:
            OHLCV DataFrame for the task range. May be empty.

        Raises:
            ValidationError: If a kline payload cannot be parsed.
            ExchangeError: Propagated from ``BinanceClient`` transport failures.
        """
        self._logger.debug(
            "Executing download task",
            extra={
                "symbol": task.symbol,
                "timeframe": task.timeframe,
                "start_time": task.start_time,
                "end_time": task.end_time,
            },
        )

        frames: list[pl.DataFrame] = []
        cursor = task.start_time
        while cursor <= task.end_time:
            payload = await self._client.get_klines(
                task.symbol,
                task.timeframe,
                start_time=cursor,
                end_time=task.end_time,
                limit=self._kline_limit,
            )
            rows = _parse_kline_rows(payload)
            if not rows:
                break

            frames.append(
                _klines_to_dataframe(
                    rows,
                    symbol=task.symbol,
                    timeframe=task.timeframe,
                )
            )
            last_open_time = rows[-1][0]
            next_cursor = last_open_time + 1
            if next_cursor <= cursor:
                break
            if len(rows) < self._kline_limit:
                break
            cursor = next_cursor

        if not frames:
            return pl.DataFrame(schema=_OHLCV_SCHEMA)
        return pl.concat(frames, how="vertical")

    def _persist_ohlcv(
        self,
        frame: pl.DataFrame,
        *,
        symbol: Symbol,
        timeframe: Timeframe,
    ) -> None:
        """Persist an OHLCV frame as year partitions through the repository.

        Args:
            frame: Combined OHLCV rows for one symbol/timeframe download.
            symbol: Tradeable symbol.
            timeframe: Bar interval identifier.
        """
        year_frame = frame.with_columns(
            pl.from_epoch(pl.col("open_time"), time_unit="ms")
            .dt.replace_time_zone("UTC")
            .dt.year()
            .alias("_year")
        )
        years = year_frame.get_column("_year").unique().sort().to_list()
        for year in years:
            partition = year_frame.filter(pl.col("_year") == year).drop("_year")
            self._repository.save_ohlcv(
                partition,
                exchange=_EXCHANGE,
                market=_MARKET,
                symbol=symbol,
                timeframe=timeframe,
                year=int(year),
            )
            self._logger.debug(
                "Persisted OHLCV year partition",
                extra={
                    "symbol": symbol,
                    "timeframe": timeframe,
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
            error_code="INGESTION-DOWNLOADER-003",
            details={"parameter": parameter, "type": type(value).__name__},
        )
    return value


def _parse_kline_rows(
    payload: object,
) -> list[tuple[int, float, float, float, float, float, int, float, int]]:
    """Parse a Binance klines payload into typed row tuples.

    Args:
        payload: Decoded JSON body from ``get_klines``.

    Returns:
        List of parsed kline row tuples. Only the fields required for OHLCV
        persistence are retained.

    Raises:
        ValidationError: If the payload structure is invalid.
    """
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise ValidationError(
            "klines payload must be a sequence",
            error_code="INGESTION-DOWNLOADER-004",
            details={"type": type(payload).__name__},
        )

    rows: list[tuple[int, float, float, float, float, float, int, float, int]] = []
    for index, entry in enumerate(cast(Sequence[object], payload)):
        rows.append(_parse_kline_row(entry, index=index))
    return rows


def _parse_kline_row(
    entry: object, *, index: int
) -> tuple[int, float, float, float, float, float, int, float, int]:
    """Parse a single Binance kline array into typed fields.

    Args:
        entry: Raw kline array from the exchange payload.
        index: Row index for error context.

    Returns:
        Tuple of
        ``(open_time, open, high, low, close, volume, close_time,
        quote_volume, trade_count)``.

    Raises:
        ValidationError: If the row cannot be parsed.
    """
    if not isinstance(entry, Sequence) or isinstance(entry, (str, bytes)):
        raise ValidationError(
            "klines row must be a sequence",
            error_code="INGESTION-DOWNLOADER-005",
            details={"index": index, "type": type(entry).__name__},
        )

    values = cast(Sequence[object], entry)
    if len(values) < 9:
        raise ValidationError(
            "klines row is missing required fields",
            error_code="INGESTION-DOWNLOADER-006",
            details={"index": index, "length": len(values)},
        )

    return (
        _as_int(values[0], field="open_time", index=index),
        _as_float(values[1], field="open", index=index),
        _as_float(values[2], field="high", index=index),
        _as_float(values[3], field="low", index=index),
        _as_float(values[4], field="close", index=index),
        _as_float(values[5], field="volume", index=index),
        _as_int(values[6], field="close_time", index=index),
        _as_float(values[7], field="quote_volume", index=index),
        _as_int(values[8], field="trade_count", index=index),
    )


def _klines_to_dataframe(
    rows: Sequence[tuple[int, float, float, float, float, float, int, float, int]],
    *,
    symbol: Symbol,
    timeframe: Timeframe,
) -> pl.DataFrame:
    """Convert parsed kline rows into an OHLCV DataFrame.

    Args:
        rows: Parsed kline tuples.
        symbol: Tradeable symbol stamped onto each row.
        timeframe: Bar interval stamped onto each row.

    Returns:
        Eager Polars DataFrame with the canonical raw OHLCV columns.
    """
    return pl.DataFrame(
        {
            "symbol": [symbol] * len(rows),
            "timeframe": [timeframe] * len(rows),
            "open_time": [row[0] for row in rows],
            "close_time": [row[6] for row in rows],
            "open": [row[1] for row in rows],
            "high": [row[2] for row in rows],
            "low": [row[3] for row in rows],
            "close": [row[4] for row in rows],
            "volume": [row[5] for row in rows],
            "quote_volume": [row[7] for row in rows],
            "trade_count": [row[8] for row in rows],
        },
        schema=_OHLCV_SCHEMA,
    )


def _as_int(value: object, *, field: str, index: int) -> int:
    """Coerce a kline field to ``int``.

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
            f"klines.{field} must be an int-compatible value",
            error_code="INGESTION-DOWNLOADER-007",
            details={"index": index, "field": field, "type": type(value).__name__},
        )
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"klines.{field} must be an int-compatible value",
            error_code="INGESTION-DOWNLOADER-007",
            details={"index": index, "field": field, "value": value},
        ) from exc


def _as_float(value: object, *, field: str, index: int) -> float:
    """Coerce a kline field to ``float``.

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
            f"klines.{field} must be a float-compatible value",
            error_code="INGESTION-DOWNLOADER-008",
            details={"index": index, "field": field, "type": type(value).__name__},
        )
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"klines.{field} must be a float-compatible value",
            error_code="INGESTION-DOWNLOADER-008",
            details={"index": index, "field": field, "value": value},
        ) from exc
