"""CQROS Binance historical funding-rate downloader.

Purpose:
    Plan and execute historical funding-rate downloads from Binance USDⓈ-M
    Futures, persisting raw funding partitions through
    ``MarketDataRepository``.

Responsibilities:
    - Represent immutable ``FundingDownloadTask`` time-range units
    - Split long ranges into sequential tasks via ``FundingDownloadPlanner``
    - Fetch funding history through ``BinanceClient`` and persist via repository
    - Paginate until each requested window is complete
    - Validate every exchange response before conversion
    - Keep planning and execution separated
    - Remain free of filesystem path construction, feature engineering, and
      research logic

Dependencies:
    ``polars``, ``cqros.core``, ``cqros.ingestion.chunk_sizing``,
    ``cqros.ingestion.client``, and ``cqros.storage.repository``.

Public API:
    ``FundingDownloadTask``, ``FundingDownloadPlanner``, ``FundingDownloader``,
    and the default limit / chunk / timeframe constants listed in ``__all__``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, cast

import polars as pl

from cqros.core.constants import (
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
    MILLISECONDS_PER_HOUR,
)
from cqros.core.exceptions import ValidationError
from cqros.core.types import (
    Exchange,
    Market,
    Symbol,
    Timeframe,
    UnixTimestampMs,
)
from cqros.ingestion.chunk_sizing import DEFAULT_CHUNK_SAFETY_FACTOR
from cqros.ingestion.client import BinanceClient
from cqros.ingestion.resume import (
    DownloadResult,
    DownloadStatus,
    coerce_latest_timestamp,
    resolve_resume_window,
)
from cqros.storage.repository import MarketDataRepository

__all__ = [
    "DEFAULT_FUNDING_CHUNK_SIZE_MS",
    "DEFAULT_FUNDING_INTERVAL_MS",
    "DEFAULT_FUNDING_REQUEST_LIMIT",
    "DEFAULT_FUNDING_TIMEFRAME",
    "FundingDownloadTask",
    "FundingDownloadPlanner",
    "FundingDownloader",
]

# Binance USDⓈ-M funding settles on an 8-hour cadence for most contracts.
DEFAULT_FUNDING_INTERVAL_MS: Final[int] = 8 * MILLISECONDS_PER_HOUR
DEFAULT_FUNDING_REQUEST_LIMIT: Final[int] = 1_000
DEFAULT_FUNDING_TIMEFRAME: Final[Timeframe] = "8h"
DEFAULT_FUNDING_CHUNK_SIZE_MS: Final[int] = int(
    DEFAULT_FUNDING_REQUEST_LIMIT * DEFAULT_CHUNK_SAFETY_FACTOR * DEFAULT_FUNDING_INTERVAL_MS
)

_EXCHANGE: Final[Exchange] = EXCHANGE_BINANCE
_MARKET: Final[Market] = MARKET_USDT_PERPETUAL

_FUNDING_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "symbol": pl.String,
        "funding_time": pl.Int64,
        "funding_rate": pl.Float64,
        "mark_price": pl.Float64,
    }
)

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FundingDownloadTask:
    """Immutable unit of work for a single historical funding download window.

    Attributes:
        symbol: Tradeable symbol (for example ``BTCUSDT``).
        start_time: Inclusive window start as UTC Unix milliseconds.
        end_time: Inclusive window end as UTC Unix milliseconds.
    """

    symbol: Symbol
    start_time: UnixTimestampMs
    end_time: UnixTimestampMs


class FundingDownloadPlanner:
    """Split a historical funding range into sequential download tasks.

    Produced tasks are contiguous, covering ``[start_time, end_time]`` without
    gaps or overlaps. Chunk duration is fixed because funding history is
    event-based rather than bar-interval based.

    Args:
        chunk_size_ms: Inclusive window length per task in milliseconds.

    Raises:
        ValidationError: If ``chunk_size_ms`` is not positive.
    """

    __slots__ = ("_chunk_size_ms",)

    _chunk_size_ms: int

    def __init__(
        self,
        *,
        chunk_size_ms: int = DEFAULT_FUNDING_CHUNK_SIZE_MS,
    ) -> None:
        """Initialize planner configuration.

        Args:
            chunk_size_ms: Inclusive window length per task in milliseconds.

        Raises:
            ValidationError: If ``chunk_size_ms`` is not positive.
        """
        if chunk_size_ms <= 0:
            raise ValidationError(
                "chunk_size_ms must be greater than 0",
                error_code="INGESTION-FUNDING-001",
                details={"parameter": "chunk_size_ms", "value": chunk_size_ms},
            )
        self._chunk_size_ms = chunk_size_ms

    @property
    def chunk_size_ms(self) -> int:
        """Return the configured inclusive chunk size in milliseconds."""
        return self._chunk_size_ms

    def plan(
        self,
        *,
        symbol: Symbol,
        start_time: UnixTimestampMs,
        end_time: UnixTimestampMs,
    ) -> tuple[FundingDownloadTask, ...]:
        """Produce sequential funding download tasks covering the range.

        Args:
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            start_time: Inclusive range start as UTC Unix milliseconds.
            end_time: Inclusive range end as UTC Unix milliseconds.

        Returns:
            Immutable tuple of ``FundingDownloadTask`` instances in
            chronological order. Returns an empty tuple when ``start_time`` is
            greater than ``end_time``.

        Raises:
            ValidationError: If timestamps are invalid integers.
        """
        _require_unix_ms(start_time, parameter="start_time")
        _require_unix_ms(end_time, parameter="end_time")

        if start_time > end_time:
            return ()

        tasks: list[FundingDownloadTask] = []
        cursor = start_time
        while cursor <= end_time:
            chunk_end = min(cursor + self._chunk_size_ms - 1, end_time)
            tasks.append(
                FundingDownloadTask(
                    symbol=symbol,
                    start_time=cursor,
                    end_time=chunk_end,
                )
            )
            cursor = chunk_end + 1

        return tuple(tasks)


class FundingDownloader:
    """Execute planned historical funding downloads into market-data storage.

    Planning remains the responsibility of ``FundingDownloadPlanner``. This
    class fetches funding history for each task, converts vendor rows into
    tabular frames, and persists year partitions through
    ``MarketDataRepository``. Filesystem paths are never constructed here.

    Args:
        client: Open ``BinanceClient`` used for funding requests.
        repository: Market-data repository used for funding persistence.
        planner: Planner that splits long ranges into sequential tasks.
        funding_limit: Maximum funding records requested per Binance API call.
        funding_timeframe: Sampling interval used only for repository
            partition keys. Does not appear in persisted DataFrame columns.
        logger: Optional logger instance. Defaults to the module logger.

    Raises:
        ValidationError: If ``funding_limit`` is not positive.
    """

    __slots__ = (
        "_client",
        "_repository",
        "_planner",
        "_funding_limit",
        "_funding_timeframe",
        "_logger",
    )

    _client: BinanceClient
    _repository: MarketDataRepository
    _planner: FundingDownloadPlanner
    _funding_limit: int
    _funding_timeframe: Timeframe
    _logger: logging.Logger

    def __init__(
        self,
        client: BinanceClient,
        repository: MarketDataRepository,
        planner: FundingDownloadPlanner,
        *,
        funding_limit: int = DEFAULT_FUNDING_REQUEST_LIMIT,
        funding_timeframe: Timeframe = DEFAULT_FUNDING_TIMEFRAME,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the downloader with injected collaborators.

        Args:
            client: Binance USDⓈ-M Futures REST client.
            repository: Repository used to persist funding partitions.
            planner: Funding download range planner.
            funding_limit: Maximum funding records requested per API call.
            funding_timeframe: Sampling interval used for repository partition
                keys only.
            logger: Optional logger instance.

        Raises:
            ValidationError: If ``funding_limit`` is not positive.
        """
        if funding_limit <= 0:
            raise ValidationError(
                "funding_limit must be greater than 0",
                error_code="INGESTION-FUNDING-002",
                details={"parameter": "funding_limit", "value": funding_limit},
            )
        self._client = client
        self._repository = repository
        self._planner = planner
        self._funding_limit = funding_limit
        self._funding_timeframe = funding_timeframe
        self._logger = logger if logger is not None else _logger

    @property
    def funding_limit(self) -> int:
        """Return the configured per-request funding limit."""
        return self._funding_limit

    @property
    def funding_timeframe(self) -> Timeframe:
        """Return the sampling interval used for repository partition keys."""
        return self._funding_timeframe

    async def fetch_symbol(
        self,
        *,
        symbol: Symbol,
        start_time: UnixTimestampMs,
        end_time: UnixTimestampMs,
    ) -> pl.DataFrame:
        """Download historical funding rates for a symbol without persisting.

        Args:
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            start_time: Inclusive range start as UTC Unix milliseconds.
            end_time: Inclusive range end as UTC Unix milliseconds.

        Returns:
            Combined funding DataFrame for the requested range. Empty when the
            exchange returns no rows or ``start_time`` is after ``end_time``.

        Raises:
            ValidationError: If planner inputs or funding payloads are invalid.
            ExchangeError: Propagated from ``BinanceClient`` transport failures.
        """
        tasks = self._planner.plan(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
        )
        self._logger.info(
            "Starting symbol funding fetch",
            extra={
                "exchange": _EXCHANGE,
                "market": _MARKET,
                "symbol": symbol,
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
                "Symbol funding fetch produced no rows",
                extra={
                    "symbol": symbol,
                    "start_time": start_time,
                    "end_time": end_time,
                },
            )
            return pl.DataFrame(schema=_FUNDING_SCHEMA)

        combined = pl.concat(frames, how="vertical")
        self._logger.info(
            "Completed symbol funding fetch",
            extra={
                "symbol": symbol,
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
        start_time: UnixTimestampMs,
        end_time: UnixTimestampMs,
    ) -> DownloadResult:
        """Download and persist historical funding rates for a single symbol.

        Resumes from the latest persisted ``funding_time`` when storage already
        contains funding for ``symbol``. Overlapping rows are deduplicated by
        the repository on save.

        Args:
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            start_time: Inclusive range start as UTC Unix milliseconds.
            end_time: Inclusive range end as UTC Unix milliseconds.

        Returns:
            Immutable ``DownloadResult`` describing full, updated, or skipped.

        Raises:
            ValidationError: If planner inputs or funding payloads are invalid.
            ExchangeError: Propagated from ``BinanceClient`` transport failures.
        """
        latest = coerce_latest_timestamp(
            self._repository.get_latest_funding_timestamp(
                exchange=_EXCHANGE,
                market=_MARKET,
                symbol=symbol,
                timeframe=self._funding_timeframe,
            )
        )
        window = resolve_resume_window(
            latest_timestamp=latest,
            requested_start=start_time,
            requested_end=end_time,
            interval_ms=DEFAULT_FUNDING_INTERVAL_MS,
        )
        if window.status is DownloadStatus.SKIPPED:
            self._logger.info(
                "Skipping funding download; storage already current",
                extra={
                    "symbol": symbol,
                    "start_time": start_time,
                    "end_time": end_time,
                    "adjusted_start": window.start_time,
                },
            )
            return DownloadResult(status=DownloadStatus.SKIPPED, rows_downloaded=0)

        combined = await self.fetch_symbol(
            symbol=symbol,
            start_time=window.start_time,
            end_time=window.end_time,
        )
        if combined.height > 0:
            self._persist_funding(combined, symbol=symbol)
            self._logger.info(
                "Completed symbol funding download",
                extra={
                    "symbol": symbol,
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
        start_time: UnixTimestampMs,
        end_time: UnixTimestampMs,
    ) -> None:
        """Download and persist historical funding rates for multiple symbols.

        Symbols are processed sequentially so rate-limit pressure remains
        predictable. Each symbol is planned and executed independently.

        Args:
            symbols: Ordered symbols to download.
            start_time: Inclusive range start as UTC Unix milliseconds.
            end_time: Inclusive range end as UTC Unix milliseconds.

        Raises:
            ValidationError: If planner inputs or funding payloads are invalid.
            ExchangeError: Propagated from ``BinanceClient`` transport failures.
        """
        self._logger.info(
            "Starting universe funding download",
            extra={
                "exchange": _EXCHANGE,
                "market": _MARKET,
                "symbol_count": len(symbols),
                "start_time": start_time,
                "end_time": end_time,
            },
        )
        for symbol in symbols:
            await self.download_symbol(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
            )
        self._logger.info(
            "Completed universe funding download",
            extra={
                "symbol_count": len(symbols),
                "start_time": start_time,
                "end_time": end_time,
            },
        )

    async def _execute_task(self, task: FundingDownloadTask) -> pl.DataFrame:
        """Fetch all funding rates for a single task window.

        Args:
            task: Immutable download window to execute.

        Returns:
            Funding DataFrame for the task range. May be empty.

        Raises:
            ValidationError: If a funding payload cannot be parsed.
            ExchangeError: Propagated from ``BinanceClient`` transport failures.
        """
        self._logger.debug(
            "Executing funding download task",
            extra={
                "symbol": task.symbol,
                "start_time": task.start_time,
                "end_time": task.end_time,
            },
        )

        frames: list[pl.DataFrame] = []
        cursor = task.start_time
        while cursor <= task.end_time:
            payload = await self._client.get_funding_rates(
                task.symbol,
                start_time=cursor,
                end_time=task.end_time,
                limit=self._funding_limit,
            )
            rows = _parse_funding_rows(payload, expected_symbol=task.symbol)
            if not rows:
                break

            frames.append(_funding_rows_to_dataframe(rows))
            last_funding_time = rows[-1][1]
            next_cursor = last_funding_time + 1
            if next_cursor <= cursor:
                break
            if len(rows) < self._funding_limit:
                break
            cursor = next_cursor

        if not frames:
            return pl.DataFrame(schema=_FUNDING_SCHEMA)
        return pl.concat(frames, how="vertical")

    def _persist_funding(
        self,
        frame: pl.DataFrame,
        *,
        symbol: Symbol,
    ) -> None:
        """Persist a funding frame as year partitions through the repository.

        Args:
            frame: Combined funding rows for one symbol download.
            symbol: Tradeable symbol.
        """
        year_expr = pl.from_epoch(pl.col("funding_time"), time_unit="ms").dt.year()
        # Polars DataFrame stubs expose Unknown aliases under pyright strict.
        year_frame = frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            year_expr.alias("_year")
        )
        years = year_frame.get_column("_year").unique().sort().to_list()
        for year in years:
            partition = year_frame.filter(  # pyright: ignore[reportUnknownMemberType]
                pl.col("_year") == year
            ).drop("_year")
            self._repository.save_funding(
                partition,
                exchange=_EXCHANGE,
                market=_MARKET,
                symbol=symbol,
                timeframe=self._funding_timeframe,
                year=int(year),
            )
            self._logger.debug(
                "Persisted funding year partition",
                extra={
                    "symbol": symbol,
                    "timeframe": self._funding_timeframe,
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
            error_code="INGESTION-FUNDING-003",
            details={"parameter": parameter, "type": type(value).__name__},
        )
    return value


def _parse_funding_rows(
    payload: object,
    *,
    expected_symbol: Symbol,
) -> list[tuple[str, int, float, float | None]]:
    """Parse a Binance funding-rate payload into typed row tuples.

    Args:
        payload: Decoded JSON body from ``get_funding_rates``.
        expected_symbol: Symbol requested from the exchange.

    Returns:
        List of ``(symbol, funding_time, funding_rate, mark_price)`` tuples.
        ``mark_price`` may be ``None`` when absent from the vendor payload.

    Raises:
        ValidationError: If the payload structure is invalid.
    """
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise ValidationError(
            "funding payload must be a sequence",
            error_code="INGESTION-FUNDING-004",
            details={"type": type(payload).__name__},
        )

    rows: list[tuple[str, int, float, float | None]] = []
    for index, entry in enumerate(cast(Sequence[object], payload)):
        rows.append(
            _parse_funding_row(
                entry,
                index=index,
                expected_symbol=expected_symbol,
            )
        )
    return rows


def _parse_funding_row(
    entry: object,
    *,
    index: int,
    expected_symbol: Symbol,
) -> tuple[str, int, float, float | None]:
    """Parse a single Binance funding object into typed fields.

    Args:
        entry: Raw funding object from the exchange payload.
        index: Row index for error context.
        expected_symbol: Symbol requested from the exchange.

    Returns:
        Tuple of ``(symbol, funding_time, funding_rate, mark_price)``.

    Raises:
        ValidationError: If the row cannot be parsed.
    """
    if not isinstance(entry, Mapping):
        raise ValidationError(
            "funding row must be a mapping",
            error_code="INGESTION-FUNDING-005",
            details={"index": index, "type": type(entry).__name__},
        )

    values = cast(Mapping[str, object], entry)
    required = ("symbol", "fundingTime", "fundingRate")
    missing = [field for field in required if field not in values]
    if missing:
        raise ValidationError(
            "funding row is missing required fields",
            error_code="INGESTION-FUNDING-006",
            details={"index": index, "missing": missing},
        )

    symbol = _as_symbol(values["symbol"], index=index)
    if symbol != expected_symbol:
        raise ValidationError(
            "funding row symbol does not match requested symbol",
            error_code="INGESTION-FUNDING-007",
            details={
                "index": index,
                "expected": expected_symbol,
                "actual": symbol,
            },
        )

    return (
        symbol,
        _as_int(values["fundingTime"], field="funding_time", index=index),
        _as_float(values["fundingRate"], field="funding_rate", index=index),
        _as_optional_float(values.get("markPrice"), field="mark_price", index=index),
    )


def _funding_rows_to_dataframe(
    rows: Sequence[tuple[str, int, float, float | None]],
) -> pl.DataFrame:
    """Convert parsed funding rows into a canonical funding DataFrame.

    Args:
        rows: Parsed funding tuples.

    Returns:
        Eager Polars DataFrame with the canonical raw funding columns.
    """
    return pl.DataFrame(
        {
            "symbol": [row[0] for row in rows],
            "funding_time": [row[1] for row in rows],
            "funding_rate": [row[2] for row in rows],
            "mark_price": [row[3] for row in rows],
        },
        schema=_FUNDING_SCHEMA,
    )


def _as_symbol(value: object, *, index: int) -> str:
    """Coerce a funding symbol field to ``str``.

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
            "funding.symbol must be a non-empty string",
            error_code="INGESTION-FUNDING-008",
            details={"index": index, "type": type(value).__name__},
        )
    return value


def _as_int(value: object, *, field: str, index: int) -> int:
    """Coerce a funding field to ``int``.

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
            f"funding.{field} must be an int-compatible value",
            error_code="INGESTION-FUNDING-009",
            details={"index": index, "field": field, "type": type(value).__name__},
        )
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"funding.{field} must be an int-compatible value",
            error_code="INGESTION-FUNDING-009",
            details={"index": index, "field": field, "value": value},
        ) from exc


def _as_float(value: object, *, field: str, index: int) -> float:
    """Coerce a funding field to ``float``.

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
            f"funding.{field} must be a float-compatible value",
            error_code="INGESTION-FUNDING-010",
            details={"index": index, "field": field, "type": type(value).__name__},
        )
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"funding.{field} must be a float-compatible value",
            error_code="INGESTION-FUNDING-010",
            details={"index": index, "field": field, "value": value},
        ) from exc


def _as_optional_float(
    value: object,
    *,
    field: str,
    index: int,
) -> float | None:
    """Coerce an optional funding field to ``float`` or ``None``.

    Empty strings and missing values become ``None``. Present values must be
    float-compatible.

    Args:
        value: Raw field value, possibly absent.
        field: Field name for error context.
        index: Row index for error context.

    Returns:
        Floating-point value, or ``None`` when absent.

    Raises:
        ValidationError: If a present value cannot be coerced.
    """
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return _as_float(value, field=field, index=index)
