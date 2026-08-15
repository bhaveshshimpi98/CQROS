"""CQROS Processing Framework orchestration runner.

Purpose:
    Load raw market partitions, execute an injected processing pipeline, apply
    the appropriate cleaner, and persist research-ready frames through the
    processed repository.

Responsibilities:
    - Load raw datasets from ``MarketDataRepository``
    - Execute ``ProcessingPipeline.run`` (or compatible ``run`` executors)
    - Execute the injected dataset cleaner
    - Save cleaned datasets through ``ProcessedMarketDataRepository``
    - Continue remaining assets after individual partition failures
    - Return an immutable ``ProcessingSummary``
    - Remain free of CLI, multiprocessing, feature, and factor logic

Dependencies:
    ``polars``, the Python standard library, ``cqros.core``,
    ``cqros.processing.cleaning``, ``cqros.processing.exceptions``, and
    ``cqros.storage`` repositories.

Public API:
    ``ProcessingTaskResult``, ``ProcessingSummary``, ``ProcessingRunner``
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Literal, Protocol, cast

import polars as pl

from cqros.core.constants import EXCHANGE_BINANCE, MARKET_USDT_PERPETUAL
from cqros.core.exceptions import CQROSError
from cqros.core.types import Exchange, Market, Symbol, Timeframe
from cqros.processing.cleaning import (
    CleaningReport,
    FundingCleaner,
    LongShortCleaner,
    OHLCVCleaner,
    OpenInterestCleaner,
    TakerVolumeCleaner,
)
from cqros.processing.exceptions import ProcessingError
from cqros.storage.processed_repository import ProcessedMarketDataRepository
from cqros.storage.repository import MarketDataRepository

__all__ = [
    "ProcessingRunner",
    "ProcessingSummary",
    "ProcessingTaskResult",
]

_logger = logging.getLogger(__name__)

_STATUS_SUCCEEDED: Final[Literal["succeeded"]] = "succeeded"
_STATUS_FAILED: Final[Literal["failed"]] = "failed"

_ERROR_SYMBOLS_EMPTY: Final[str] = "PROCESSING-RUNNER-001"
_ERROR_TIMEFRAMES_EMPTY: Final[str] = "PROCESSING-RUNNER-002"
_ERROR_YEARS_EMPTY: Final[str] = "PROCESSING-RUNNER-003"
_ERROR_YEAR_INVALID: Final[str] = "PROCESSING-RUNNER-004"

type _LoadPartition = Callable[..., pl.DataFrame]
type _SavePartition = Callable[..., None]


class SupportsProcessingRun(Protocol):
    """Structural contract for pipeline executors accepted by the runner."""

    def run(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Transform ``frame`` and return a new DataFrame."""
        ...


class SupportsCleaning(Protocol):
    """Structural contract for cleaners accepted by the runner."""

    def clean(self, frame: pl.DataFrame) -> tuple[pl.DataFrame, CleaningReport]:
        """Clean ``frame`` and return ``(cleaned_frame, CleaningReport)``."""
        ...


@dataclass(frozen=True, slots=True)
class ProcessingTaskResult:
    """Immutable outcome for one symbol/timeframe/year processing task.

    Attributes:
        symbol: Symbol processed by the task.
        timeframe: Timeframe processed by the task.
        year: Calendar year partition processed by the task.
        status: ``succeeded`` or ``failed``.
        rows_loaded: Rows loaded from raw storage when available.
        rows_saved: Rows written to processed storage when available.
        cleaning_report: Cleaner report when the task succeeded.
        error_type: Exception type name when the task failed.
        error_message: Human-readable failure summary when the task failed.
        error_code: Optional CQROS error code when the task failed.
    """

    symbol: Symbol
    timeframe: Timeframe
    year: int
    status: Literal["succeeded", "failed"]
    rows_loaded: int | None = None
    rows_saved: int | None = None
    cleaning_report: CleaningReport | None = None
    error_type: str | None = None
    error_message: str | None = None
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class ProcessingSummary:
    """Immutable summary of a multi-asset processing run.

    Attributes:
        dataset: Logical dataset name processed by the run.
        exchange: Exchange identifier used for load and save.
        market: Market segment used for load and save.
        results: Ordered per-partition task outcomes.
    """

    dataset: str
    exchange: Exchange
    market: Market
    results: tuple[ProcessingTaskResult, ...]

    @property
    def total_count(self) -> int:
        """Return the number of partition tasks attempted."""
        return len(self.results)

    @property
    def succeeded_count(self) -> int:
        """Return the number of succeeded partition tasks."""
        return sum(1 for result in self.results if result.status == _STATUS_SUCCEEDED)

    @property
    def failed_count(self) -> int:
        """Return the number of failed partition tasks."""
        return sum(1 for result in self.results if result.status == _STATUS_FAILED)

    @property
    def succeeded(self) -> tuple[ProcessingTaskResult, ...]:
        """Return succeeded task results in discovery order."""
        return tuple(result for result in self.results if result.status == _STATUS_SUCCEEDED)

    @property
    def failed(self) -> tuple[ProcessingTaskResult, ...]:
        """Return failed task results in discovery order."""
        return tuple(result for result in self.results if result.status == _STATUS_FAILED)


class ProcessingRunner:
    """Orchestrate raw → pipeline → cleaner → processed persistence.

    The runner never mutates caller-supplied frames, never invents processing
    or cleaning rules, and never stops the overall run when a single partition
    fails.

    Args:
        raw_repository: Repository used to load raw partitions.
        processed_repository: Repository used to save processed partitions.
        pipeline: Processing executor exposing ``run(frame) -> DataFrame``.
        ohlcv_cleaner: Cleaner for OHLCV frames. Defaults to ``OHLCVCleaner``.
        funding_cleaner: Cleaner for funding frames. Defaults to
            ``FundingCleaner``.
        open_interest_cleaner: Cleaner for open-interest frames. Defaults to
            ``OpenInterestCleaner``.
        taker_volume_cleaner: Cleaner for taker-volume frames. Defaults to
            ``TakerVolumeCleaner``.
        long_short_cleaner: Cleaner for long/short ratio frames. Defaults to
            ``LongShortCleaner``.
        logger: Optional logger instance. Defaults to the module logger.
    """

    __slots__ = (
        "_funding_cleaner",
        "_logger",
        "_long_short_cleaner",
        "_ohlcv_cleaner",
        "_open_interest_cleaner",
        "_pipeline",
        "_processed",
        "_raw",
        "_taker_volume_cleaner",
    )

    def __init__(
        self,
        raw_repository: MarketDataRepository,
        processed_repository: ProcessedMarketDataRepository,
        pipeline: SupportsProcessingRun,
        *,
        ohlcv_cleaner: SupportsCleaning | None = None,
        funding_cleaner: SupportsCleaning | None = None,
        open_interest_cleaner: SupportsCleaning | None = None,
        taker_volume_cleaner: SupportsCleaning | None = None,
        long_short_cleaner: SupportsCleaning | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the runner with injected repositories, pipeline, and cleaners.

        Args:
            raw_repository: Raw market-data repository.
            processed_repository: Processed market-data repository.
            pipeline: Processing pipeline or compatible executor.
            ohlcv_cleaner: Optional OHLCV cleaner override.
            funding_cleaner: Optional funding cleaner override.
            open_interest_cleaner: Optional open-interest cleaner override.
            taker_volume_cleaner: Optional taker-volume cleaner override.
            long_short_cleaner: Optional long/short cleaner override.
            logger: Optional logger instance.
        """
        self._raw = raw_repository
        self._processed = processed_repository
        self._pipeline = pipeline
        self._ohlcv_cleaner = ohlcv_cleaner if ohlcv_cleaner is not None else OHLCVCleaner()
        self._funding_cleaner = funding_cleaner if funding_cleaner is not None else FundingCleaner()
        self._open_interest_cleaner = (
            open_interest_cleaner if open_interest_cleaner is not None else OpenInterestCleaner()
        )
        self._taker_volume_cleaner = (
            taker_volume_cleaner if taker_volume_cleaner is not None else TakerVolumeCleaner()
        )
        self._long_short_cleaner = (
            long_short_cleaner if long_short_cleaner is not None else LongShortCleaner()
        )
        self._logger = logger if logger is not None else _logger

    def process_ohlcv(
        self,
        *,
        symbols: Sequence[Symbol],
        timeframes: Sequence[Timeframe],
        years: Sequence[int],
        exchange: Exchange = EXCHANGE_BINANCE,
        market: Market = MARKET_USDT_PERPETUAL,
    ) -> ProcessingSummary:
        """Process raw OHLCV partitions into processed storage.

        Args:
            symbols: Symbols to process.
            timeframes: Timeframes to process.
            years: Calendar years to process.
            exchange: Exchange identifier.
            market: Market segment.

        Returns:
            Immutable processing summary covering every requested partition.
        """
        return self._process_dataset(
            dataset="ohlcv",
            load=self._raw.load_ohlcv,
            save=self._processed.save_ohlcv,
            cleaner=self._ohlcv_cleaner,
            symbols=symbols,
            timeframes=timeframes,
            years=years,
            exchange=exchange,
            market=market,
        )

    def process_funding(
        self,
        *,
        symbols: Sequence[Symbol],
        timeframes: Sequence[Timeframe],
        years: Sequence[int],
        exchange: Exchange = EXCHANGE_BINANCE,
        market: Market = MARKET_USDT_PERPETUAL,
    ) -> ProcessingSummary:
        """Process raw funding partitions into processed storage."""
        return self._process_dataset(
            dataset="funding",
            load=self._raw.load_funding,
            save=self._processed.save_funding,
            cleaner=self._funding_cleaner,
            symbols=symbols,
            timeframes=timeframes,
            years=years,
            exchange=exchange,
            market=market,
        )

    def process_open_interest(
        self,
        *,
        symbols: Sequence[Symbol],
        timeframes: Sequence[Timeframe],
        years: Sequence[int],
        exchange: Exchange = EXCHANGE_BINANCE,
        market: Market = MARKET_USDT_PERPETUAL,
    ) -> ProcessingSummary:
        """Process raw open-interest partitions into processed storage."""
        return self._process_dataset(
            dataset="open_interest",
            load=self._raw.load_open_interest,
            save=self._processed.save_open_interest,
            cleaner=self._open_interest_cleaner,
            symbols=symbols,
            timeframes=timeframes,
            years=years,
            exchange=exchange,
            market=market,
        )

    def process_taker_volume(
        self,
        *,
        symbols: Sequence[Symbol],
        timeframes: Sequence[Timeframe],
        years: Sequence[int],
        exchange: Exchange = EXCHANGE_BINANCE,
        market: Market = MARKET_USDT_PERPETUAL,
    ) -> ProcessingSummary:
        """Process raw taker-volume partitions into processed storage."""
        return self._process_dataset(
            dataset="taker_volume",
            load=self._raw.load_taker_volume,
            save=self._processed.save_taker_volume,
            cleaner=self._taker_volume_cleaner,
            symbols=symbols,
            timeframes=timeframes,
            years=years,
            exchange=exchange,
            market=market,
        )

    def process_global_long_short_account_ratio(
        self,
        *,
        symbols: Sequence[Symbol],
        timeframes: Sequence[Timeframe],
        years: Sequence[int],
        exchange: Exchange = EXCHANGE_BINANCE,
        market: Market = MARKET_USDT_PERPETUAL,
    ) -> ProcessingSummary:
        """Process raw global long/short account-ratio partitions."""
        return self._process_dataset(
            dataset="global_long_short_account_ratio",
            load=self._raw.load_global_long_short_account_ratio,
            save=self._processed.save_global_long_short_account_ratio,
            cleaner=self._long_short_cleaner,
            symbols=symbols,
            timeframes=timeframes,
            years=years,
            exchange=exchange,
            market=market,
        )

    def process_top_long_short_account_ratio(
        self,
        *,
        symbols: Sequence[Symbol],
        timeframes: Sequence[Timeframe],
        years: Sequence[int],
        exchange: Exchange = EXCHANGE_BINANCE,
        market: Market = MARKET_USDT_PERPETUAL,
    ) -> ProcessingSummary:
        """Process raw top-trader long/short account-ratio partitions."""
        return self._process_dataset(
            dataset="top_long_short_account_ratio",
            load=self._raw.load_top_long_short_account_ratio,
            save=self._processed.save_top_long_short_account_ratio,
            cleaner=self._long_short_cleaner,
            symbols=symbols,
            timeframes=timeframes,
            years=years,
            exchange=exchange,
            market=market,
        )

    def process_top_long_short_position_ratio(
        self,
        *,
        symbols: Sequence[Symbol],
        timeframes: Sequence[Timeframe],
        years: Sequence[int],
        exchange: Exchange = EXCHANGE_BINANCE,
        market: Market = MARKET_USDT_PERPETUAL,
    ) -> ProcessingSummary:
        """Process raw top-trader long/short position-ratio partitions."""
        return self._process_dataset(
            dataset="top_long_short_position_ratio",
            load=self._raw.load_top_long_short_position_ratio,
            save=self._processed.save_top_long_short_position_ratio,
            cleaner=self._long_short_cleaner,
            symbols=symbols,
            timeframes=timeframes,
            years=years,
            exchange=exchange,
            market=market,
        )

    def _process_dataset(
        self,
        *,
        dataset: str,
        load: _LoadPartition,
        save: _SavePartition,
        cleaner: SupportsCleaning,
        symbols: Sequence[Symbol],
        timeframes: Sequence[Timeframe],
        years: Sequence[int],
        exchange: Exchange,
        market: Market,
    ) -> ProcessingSummary:
        """Process every requested partition for one dataset type."""
        symbol_list = _require_non_empty_symbols(symbols)
        timeframe_list = _require_non_empty_timeframes(timeframes)
        year_list = _require_years(years)

        results: list[ProcessingTaskResult] = []
        for symbol in symbol_list:
            for timeframe in timeframe_list:
                for year in year_list:
                    results.append(
                        self._process_partition(
                            dataset=dataset,
                            load=load,
                            save=save,
                            cleaner=cleaner,
                            exchange=exchange,
                            market=market,
                            symbol=symbol,
                            timeframe=timeframe,
                            year=year,
                        )
                    )

        summary = ProcessingSummary(
            dataset=dataset,
            exchange=exchange,
            market=market,
            results=tuple(results),
        )
        self._logger.info(
            "Completed processing run",
            extra={
                "dataset": dataset,
                "exchange": exchange,
                "market": market,
                "total": summary.total_count,
                "succeeded": summary.succeeded_count,
                "failed": summary.failed_count,
            },
        )
        return summary

    def _process_partition(
        self,
        *,
        dataset: str,
        load: _LoadPartition,
        save: _SavePartition,
        cleaner: SupportsCleaning,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> ProcessingTaskResult:
        """Load, process, clean, and save one partition, capturing failures."""
        rows_loaded: int | None = None
        try:
            raw_frame = load(
                exchange=exchange,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
            )
            rows_loaded = raw_frame.height
            processed_frame = self._pipeline.run(raw_frame)
            cleaned_frame, cleaning_report = cleaner.clean(processed_frame)
            save(
                cleaned_frame,
                exchange=exchange,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
            )
            self._logger.info(
                "Processed market partition",
                extra={
                    "dataset": dataset,
                    "exchange": exchange,
                    "market": market,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "year": year,
                    "rows_loaded": rows_loaded,
                    "rows_saved": cleaned_frame.height,
                    "rows_before_cleaning": cleaning_report.rows_before,
                    "rows_after_cleaning": cleaning_report.rows_after,
                },
            )
            return ProcessingTaskResult(
                symbol=symbol,
                timeframe=timeframe,
                year=year,
                status=_STATUS_SUCCEEDED,
                rows_loaded=rows_loaded,
                rows_saved=cleaned_frame.height,
                cleaning_report=cleaning_report,
            )
        except Exception as exc:
            error_code = exc.error_code if isinstance(exc, CQROSError) else None
            self._logger.warning(
                "Failed to process market partition; continuing",
                extra={
                    "dataset": dataset,
                    "exchange": exchange,
                    "market": market,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "year": year,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "error_code": error_code,
                },
            )
            return ProcessingTaskResult(
                symbol=symbol,
                timeframe=timeframe,
                year=year,
                status=_STATUS_FAILED,
                rows_loaded=rows_loaded,
                error_type=type(exc).__name__,
                error_message=str(exc),
                error_code=error_code,
            )


def _require_non_empty_symbols(symbols: Sequence[Symbol]) -> tuple[Symbol, ...]:
    """Validate and return a non-empty symbol sequence."""
    frozen = tuple(symbols)
    if len(frozen) == 0:
        raise ProcessingError(
            "symbols must contain at least one entry",
            error_code=_ERROR_SYMBOLS_EMPTY,
            details={"parameter": "symbols"},
        )
    return frozen


def _require_non_empty_timeframes(
    timeframes: Sequence[Timeframe],
) -> tuple[Timeframe, ...]:
    """Validate and return a non-empty timeframe sequence."""
    frozen = tuple(timeframes)
    if len(frozen) == 0:
        raise ProcessingError(
            "timeframes must contain at least one entry",
            error_code=_ERROR_TIMEFRAMES_EMPTY,
            details={"parameter": "timeframes"},
        )
    return frozen


def _require_years(years: Sequence[int]) -> tuple[int, ...]:
    """Validate and return a non-empty tuple of calendar years."""
    frozen = tuple(years)
    if len(frozen) == 0:
        raise ProcessingError(
            "years must contain at least one entry",
            error_code=_ERROR_YEARS_EMPTY,
            details={"parameter": "years"},
        )
    for index, year in enumerate(frozen):
        year_value = cast(object, year)
        if not isinstance(year_value, int) or isinstance(year_value, bool) or year_value < 1:
            raise ProcessingError(
                "years entries must be positive integers",
                error_code=_ERROR_YEAR_INVALID,
                details={"parameter": "years", "index": index, "value": year},
            )
    return frozen
