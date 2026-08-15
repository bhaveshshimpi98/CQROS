"""CQROS processed-market verification orchestration runner.

Purpose:
    Load processed market partitions and execute the injected dataset
    verifier for each requested symbol/timeframe/year partition.

Responsibilities:
    - Load processed datasets from ``ProcessedMarketDataRepository``
    - Execute the injected ``DataVerifier.verify`` pass
    - Continue remaining partitions after individual partition failures
    - Return an immutable ``VerificationSummary``
    - Remain free of CLI, concurrency, feature, and factor logic

Dependencies:
    ``polars``, the Python standard library, ``cqros.core``,
    ``cqros.processing.verification``, and ``cqros.storage``.

Public API:
    ``VerificationTaskResult``, ``VerificationSummary``, ``VerificationRunner``
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Literal, cast

import polars as pl

from cqros.core.constants import EXCHANGE_BINANCE, MARKET_USDT_PERPETUAL
from cqros.core.exceptions import CQROSError
from cqros.core.types import Exchange, Market, Symbol, Timeframe
from cqros.processing.exceptions import ProcessingError
from cqros.processing.verification.funding import FundingVerifier
from cqros.processing.verification.interfaces import DataVerifier
from cqros.processing.verification.long_short import LongShortVerifier
from cqros.processing.verification.ohlcv import OHLCVVerifier
from cqros.processing.verification.open_interest import OpenInterestVerifier
from cqros.processing.verification.report import VerificationReport
from cqros.processing.verification.taker_volume import TakerVolumeVerifier
from cqros.storage.processed_repository import ProcessedMarketDataRepository

__all__ = [
    "VerificationRunner",
    "VerificationSummary",
    "VerificationTaskResult",
]

_logger = logging.getLogger(__name__)

_STATUS_SUCCEEDED: Final[Literal["succeeded"]] = "succeeded"
_STATUS_FAILED: Final[Literal["failed"]] = "failed"

_ERROR_SYMBOLS_EMPTY: Final[str] = "PROCESSING-VERIFICATION-005"
_ERROR_TIMEFRAMES_EMPTY: Final[str] = "PROCESSING-VERIFICATION-006"
_ERROR_YEARS_EMPTY: Final[str] = "PROCESSING-VERIFICATION-007"
_ERROR_YEAR_INVALID: Final[str] = "PROCESSING-VERIFICATION-008"

type _LoadPartition = Callable[..., pl.DataFrame]


@dataclass(frozen=True, slots=True)
class VerificationTaskResult:
    """Immutable outcome for one symbol/timeframe/year verification task.

    Attributes:
        symbol: Symbol verified by the task.
        timeframe: Timeframe verified by the task.
        year: Calendar year partition verified by the task.
        status: ``succeeded`` when load and verify completed without raising;
            ``failed`` when an exception was captured.
        report: Verifier report when the task succeeded.
        error_type: Exception type name when the task failed.
        error_message: Human-readable failure summary when the task failed.
        error_code: Optional CQROS error code when the task failed.
    """

    symbol: Symbol
    timeframe: Timeframe
    year: int
    status: Literal["succeeded", "failed"]
    report: VerificationReport | None = None
    error_type: str | None = None
    error_message: str | None = None
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class VerificationSummary:
    """Immutable summary of a multi-partition verification run.

    Attributes:
        dataset: Logical dataset name verified by the run.
        exchange: Exchange identifier used for load.
        market: Market segment used for load.
        results: Ordered per-partition task outcomes.
    """

    dataset: str
    exchange: Exchange
    market: Market
    results: tuple[VerificationTaskResult, ...]

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
    def succeeded(self) -> tuple[VerificationTaskResult, ...]:
        """Return succeeded task results in discovery order."""
        return tuple(result for result in self.results if result.status == _STATUS_SUCCEEDED)

    @property
    def failed(self) -> tuple[VerificationTaskResult, ...]:
        """Return failed task results in discovery order."""
        return tuple(result for result in self.results if result.status == _STATUS_FAILED)


class VerificationRunner:
    """Orchestrate processed-partition load → verifier inspection.

    The runner never mutates caller-supplied frames, never invents
    verification rules, and never stops the overall run when a single
    partition fails.

    Args:
        processed_repository: Repository used to load processed partitions.
        ohlcv_verifier: Verifier for OHLCV frames. Defaults to
            ``OHLCVVerifier``.
        funding_verifier: Verifier for funding frames. Defaults to
            ``FundingVerifier``.
        open_interest_verifier: Verifier for open-interest frames. Defaults
            to ``OpenInterestVerifier``.
        taker_volume_verifier: Verifier for taker-volume frames. Defaults to
            ``TakerVolumeVerifier``.
        long_short_verifier: Verifier for long/short ratio frames. Defaults
            to ``LongShortVerifier``.
        logger: Optional logger instance. Defaults to the module logger.
    """

    __slots__ = (
        "_funding_verifier",
        "_logger",
        "_long_short_verifier",
        "_ohlcv_verifier",
        "_open_interest_verifier",
        "_processed",
        "_taker_volume_verifier",
    )

    def __init__(
        self,
        processed_repository: ProcessedMarketDataRepository,
        *,
        ohlcv_verifier: DataVerifier | None = None,
        funding_verifier: DataVerifier | None = None,
        open_interest_verifier: DataVerifier | None = None,
        taker_volume_verifier: DataVerifier | None = None,
        long_short_verifier: DataVerifier | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the runner with an injected repository and verifiers.

        Args:
            processed_repository: Processed market-data repository.
            ohlcv_verifier: Optional OHLCV verifier override.
            funding_verifier: Optional funding verifier override.
            open_interest_verifier: Optional open-interest verifier override.
            taker_volume_verifier: Optional taker-volume verifier override.
            long_short_verifier: Optional long/short verifier override.
            logger: Optional logger instance.
        """
        self._processed = processed_repository
        self._ohlcv_verifier = ohlcv_verifier if ohlcv_verifier is not None else OHLCVVerifier()
        self._funding_verifier = (
            funding_verifier if funding_verifier is not None else FundingVerifier()
        )
        self._open_interest_verifier = (
            open_interest_verifier if open_interest_verifier is not None else OpenInterestVerifier()
        )
        self._taker_volume_verifier = (
            taker_volume_verifier if taker_volume_verifier is not None else TakerVolumeVerifier()
        )
        self._long_short_verifier = (
            long_short_verifier if long_short_verifier is not None else LongShortVerifier()
        )
        self._logger = logger if logger is not None else _logger

    def verify_ohlcv(
        self,
        *,
        symbols: Sequence[Symbol],
        timeframes: Sequence[Timeframe],
        years: Sequence[int],
        exchange: Exchange = EXCHANGE_BINANCE,
        market: Market = MARKET_USDT_PERPETUAL,
    ) -> VerificationSummary:
        """Verify processed OHLCV partitions.

        Args:
            symbols: Symbols to verify.
            timeframes: Timeframes to verify.
            years: Calendar years to verify.
            exchange: Exchange identifier.
            market: Market segment.

        Returns:
            Immutable verification summary covering every requested partition.
        """
        return self._verify_dataset(
            dataset="ohlcv",
            load=self._processed.load_ohlcv,
            verifier=self._ohlcv_verifier,
            symbols=symbols,
            timeframes=timeframes,
            years=years,
            exchange=exchange,
            market=market,
        )

    def verify_funding(
        self,
        *,
        symbols: Sequence[Symbol],
        timeframes: Sequence[Timeframe],
        years: Sequence[int],
        exchange: Exchange = EXCHANGE_BINANCE,
        market: Market = MARKET_USDT_PERPETUAL,
    ) -> VerificationSummary:
        """Verify processed funding partitions."""
        return self._verify_dataset(
            dataset="funding",
            load=self._processed.load_funding,
            verifier=self._funding_verifier,
            symbols=symbols,
            timeframes=timeframes,
            years=years,
            exchange=exchange,
            market=market,
        )

    def verify_open_interest(
        self,
        *,
        symbols: Sequence[Symbol],
        timeframes: Sequence[Timeframe],
        years: Sequence[int],
        exchange: Exchange = EXCHANGE_BINANCE,
        market: Market = MARKET_USDT_PERPETUAL,
    ) -> VerificationSummary:
        """Verify processed open-interest partitions."""
        return self._verify_dataset(
            dataset="open_interest",
            load=self._processed.load_open_interest,
            verifier=self._open_interest_verifier,
            symbols=symbols,
            timeframes=timeframes,
            years=years,
            exchange=exchange,
            market=market,
        )

    def verify_taker_volume(
        self,
        *,
        symbols: Sequence[Symbol],
        timeframes: Sequence[Timeframe],
        years: Sequence[int],
        exchange: Exchange = EXCHANGE_BINANCE,
        market: Market = MARKET_USDT_PERPETUAL,
    ) -> VerificationSummary:
        """Verify processed taker-volume partitions."""
        return self._verify_dataset(
            dataset="taker_volume",
            load=self._processed.load_taker_volume,
            verifier=self._taker_volume_verifier,
            symbols=symbols,
            timeframes=timeframes,
            years=years,
            exchange=exchange,
            market=market,
        )

    def verify_global_long_short_account_ratio(
        self,
        *,
        symbols: Sequence[Symbol],
        timeframes: Sequence[Timeframe],
        years: Sequence[int],
        exchange: Exchange = EXCHANGE_BINANCE,
        market: Market = MARKET_USDT_PERPETUAL,
    ) -> VerificationSummary:
        """Verify processed global long/short account-ratio partitions."""
        return self._verify_dataset(
            dataset="global_long_short_account_ratio",
            load=self._processed.load_global_long_short_account_ratio,
            verifier=self._long_short_verifier,
            symbols=symbols,
            timeframes=timeframes,
            years=years,
            exchange=exchange,
            market=market,
        )

    def verify_top_long_short_account_ratio(
        self,
        *,
        symbols: Sequence[Symbol],
        timeframes: Sequence[Timeframe],
        years: Sequence[int],
        exchange: Exchange = EXCHANGE_BINANCE,
        market: Market = MARKET_USDT_PERPETUAL,
    ) -> VerificationSummary:
        """Verify processed top-trader long/short account-ratio partitions."""
        return self._verify_dataset(
            dataset="top_long_short_account_ratio",
            load=self._processed.load_top_long_short_account_ratio,
            verifier=self._long_short_verifier,
            symbols=symbols,
            timeframes=timeframes,
            years=years,
            exchange=exchange,
            market=market,
        )

    def verify_top_long_short_position_ratio(
        self,
        *,
        symbols: Sequence[Symbol],
        timeframes: Sequence[Timeframe],
        years: Sequence[int],
        exchange: Exchange = EXCHANGE_BINANCE,
        market: Market = MARKET_USDT_PERPETUAL,
    ) -> VerificationSummary:
        """Verify processed top-trader long/short position-ratio partitions."""
        return self._verify_dataset(
            dataset="top_long_short_position_ratio",
            load=self._processed.load_top_long_short_position_ratio,
            verifier=self._long_short_verifier,
            symbols=symbols,
            timeframes=timeframes,
            years=years,
            exchange=exchange,
            market=market,
        )

    def _verify_dataset(
        self,
        *,
        dataset: str,
        load: _LoadPartition,
        verifier: DataVerifier,
        symbols: Sequence[Symbol],
        timeframes: Sequence[Timeframe],
        years: Sequence[int],
        exchange: Exchange,
        market: Market,
    ) -> VerificationSummary:
        """Verify every requested partition for one dataset type."""
        symbol_list = _require_non_empty_symbols(symbols)
        timeframe_list = _require_non_empty_timeframes(timeframes)
        year_list = _require_years(years)

        results: list[VerificationTaskResult] = []
        for symbol in symbol_list:
            for timeframe in timeframe_list:
                for year in year_list:
                    results.append(
                        self._verify_partition(
                            dataset=dataset,
                            load=load,
                            verifier=verifier,
                            exchange=exchange,
                            market=market,
                            symbol=symbol,
                            timeframe=timeframe,
                            year=year,
                        )
                    )

        summary = VerificationSummary(
            dataset=dataset,
            exchange=exchange,
            market=market,
            results=tuple(results),
        )
        self._logger.info(
            "Completed verification run",
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

    def _verify_partition(
        self,
        *,
        dataset: str,
        load: _LoadPartition,
        verifier: DataVerifier,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> VerificationTaskResult:
        """Load and verify one partition, capturing failures."""
        try:
            frame = load(
                exchange=exchange,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
            )
            report = verifier.verify(frame)
            self._logger.info(
                "Verified market partition",
                extra={
                    "dataset": dataset,
                    "exchange": exchange,
                    "market": market,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "year": year,
                    "rows_checked": report.rows_checked,
                    "passed": report.passed,
                },
            )
            return VerificationTaskResult(
                symbol=symbol,
                timeframe=timeframe,
                year=year,
                status=_STATUS_SUCCEEDED,
                report=report,
            )
        except Exception as exc:
            error_code = exc.error_code if isinstance(exc, CQROSError) else None
            self._logger.warning(
                "Failed to verify market partition; continuing",
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
            return VerificationTaskResult(
                symbol=symbol,
                timeframe=timeframe,
                year=year,
                status=_STATUS_FAILED,
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
