"""CQROS resumable historical universe bootstrap orchestration.

Purpose:
    Orchestrate restart-safe historical market-data bootstrapping across an
    exchange universe by composing existing CQROS bootstrap and ingestion
    services.

Responsibilities:
    - Discover symbols through ``HistoricalBootstrap.discover_symbols``
    - Process symbols with a bounded asyncio worker pool
    - For every ``(symbol, timeframe)``, load existing storage via
      ``MarketDataRepository`` and choose download, repair, update, or skip
    - Never redownload a completed valid dataset
    - Continue remaining symbols and timeframes when an individual pair fails
    - Return an immutable ``UniverseBootstrapResult`` summarizing outcomes

Dependencies:
    ``polars``, ``cqros.bootstrap.historical``, ``cqros.core``,
    ``cqros.data.contracts``, ``cqros.ingestion``, and ``cqros.storage``.

Public API:
    ``DEFAULT_UNIVERSE_WORKER_COUNT``, ``UniverseBootstrap``, and
    ``UniverseBootstrapResult``.

Notes:
    This module is orchestration only. It does not construct storage paths,
    download from the exchange directly, merge partitions, repair datasets,
    or duplicate validator logic. Those responsibilities remain owned by the
    injected collaborators. Concurrency is bounded to ``worker_count`` symbols
    and never materializes one task per symbol.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

import polars as pl

from cqros.bootstrap.historical import HistoricalBootstrap
from cqros.core.constants import (
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
    MILLISECONDS_PER_DAY,
)
from cqros.core.exceptions import ValidationError
from cqros.core.types import Symbol, Timeframe, UnixTimestampMs
from cqros.data.contracts import Contract
from cqros.ingestion.repair import DatasetRepairEngine
from cqros.ingestion.updater import IncrementalUpdater
from cqros.ingestion.validator import MarketDataValidator
from cqros.storage.exceptions import CorruptedDatasetError, DatasetNotFoundError
from cqros.storage.repository import MarketDataRepository

__all__ = [
    "DEFAULT_UNIVERSE_WORKER_COUNT",
    "UniverseBootstrap",
    "UniverseBootstrapResult",
]

_EXCHANGE: Final[str] = EXCHANGE_BINANCE
_MARKET: Final[str] = MARKET_USDT_PERPETUAL

# Binance USDⓈ-M Futures launched in 2019; partitions cannot precede this year.
_EARLIEST_PARTITION_YEAR: Final[int] = 2019

_ERROR_DOWNLOAD_RANGE: Final[str] = "BOOTSTRAP-UNIVERSE-001"
_ERROR_WORKER_COUNT: Final[str] = "BOOTSTRAP-UNIVERSE-002"

DEFAULT_UNIVERSE_WORKER_COUNT: Final[int] = 4


@dataclass(frozen=True, slots=True)
class UniverseBootstrapResult:
    """Immutable summary of a resumable universe historical bootstrap run.

    Attributes:
        successful_symbols: Symbols that completed every timeframe without
            error.
        failed_symbols: Symbols for which at least one timeframe failed.
        total_symbols: Count of discovered contracts attempted.
        successful_downloads: Count of symbols in ``successful_symbols``.
        failed_downloads: Count of symbols in ``failed_symbols``.
        downloaded_symbols: Symbols that required a fresh historical download
            for at least one timeframe.
        updated_symbols: Symbols that received an incremental update for at
            least one timeframe.
        repaired_symbols: Symbols that required dataset repair for at least
            one timeframe.
        skipped_symbols: Symbols whose existing valid datasets were already
            current for at least one timeframe (download skipped).
    """

    successful_symbols: tuple[Symbol, ...]
    failed_symbols: tuple[Symbol, ...]
    total_symbols: int
    successful_downloads: int
    failed_downloads: int
    downloaded_symbols: tuple[Symbol, ...]
    updated_symbols: tuple[Symbol, ...]
    repaired_symbols: tuple[Symbol, ...]
    skipped_symbols: tuple[Symbol, ...]


@dataclass(slots=True)
class _SymbolOutcome:
    """Mutable per-symbol outcome flags collected by bootstrap workers."""

    failed: bool = False
    downloaded: bool = False
    updated: bool = False
    repaired: bool = False
    skipped: bool = False


class UniverseBootstrap:
    """Orchestrate resumable historical bootstrap for a discovered universe.

    For every discovered symbol and configured timeframe, loads any existing
    dataset through ``MarketDataRepository``. Missing datasets are seeded via
    ``HistoricalBootstrap.download_symbol``. Existing datasets are validated;
    invalid data is repaired and valid data is incrementally updated (already
    current pairs are recorded as skipped). Per-pair failures are recorded and
    remaining work continues.

    Symbols are processed by a bounded asyncio worker pool so at most
    ``worker_count`` symbols run concurrently. Timeframes within a symbol remain
    sequential.

    Args:
        historical_bootstrap: Injected historical bootstrap for discovery and
            fresh downloads.
        repository: Injected market-data repository for dataset loads.
        validator: Injected market-data validator for existing frames.
        updater: Injected incremental updater for valid datasets.
        repair_engine: Injected repair engine for invalid datasets.
        worker_count: Maximum number of symbols processed concurrently.
            When omitted, uses ``historical_bootstrap.options.workers``.

    Raises:
        ValidationError: If ``worker_count`` is not positive.
    """

    __slots__ = (
        "_historical_bootstrap",
        "_repository",
        "_validator",
        "_updater",
        "_repair_engine",
        "_worker_count",
    )

    _historical_bootstrap: HistoricalBootstrap
    _repository: MarketDataRepository
    _validator: MarketDataValidator
    _updater: IncrementalUpdater
    _repair_engine: DatasetRepairEngine
    _worker_count: int

    def __init__(
        self,
        historical_bootstrap: HistoricalBootstrap,
        repository: MarketDataRepository,
        validator: MarketDataValidator,
        updater: IncrementalUpdater,
        repair_engine: DatasetRepairEngine,
        *,
        worker_count: int | None = None,
    ) -> None:
        """Initialize universe orchestration with injected collaborators.

        Args:
            historical_bootstrap: Historical bootstrap used for discovery and
                per-symbol download.
            repository: Repository used to load existing OHLCV partitions.
            validator: Validator applied to existing OHLCV frames.
            updater: Incremental updater for valid existing datasets.
            repair_engine: Repair engine for invalid existing datasets.
            worker_count: Maximum concurrent symbol workers. When ``None``,
                uses ``historical_bootstrap.options.workers``.

        Raises:
            ValidationError: If ``worker_count`` is not positive.
        """
        resolved_workers = (
            worker_count if worker_count is not None else historical_bootstrap.options.workers
        )
        if resolved_workers <= 0:
            raise ValidationError(
                "worker_count must be greater than 0",
                error_code=_ERROR_WORKER_COUNT,
                details={"parameter": "worker_count", "value": resolved_workers},
            )
        self._historical_bootstrap = historical_bootstrap
        self._repository = repository
        self._validator = validator
        self._updater = updater
        self._repair_engine = repair_engine
        self._worker_count = resolved_workers

    @property
    def historical_bootstrap(self) -> HistoricalBootstrap:
        """Return the injected ``HistoricalBootstrap`` instance."""
        return self._historical_bootstrap

    @property
    def repository(self) -> MarketDataRepository:
        """Return the injected ``MarketDataRepository`` instance."""
        return self._repository

    @property
    def validator(self) -> MarketDataValidator:
        """Return the injected ``MarketDataValidator`` instance."""
        return self._validator

    @property
    def updater(self) -> IncrementalUpdater:
        """Return the injected ``IncrementalUpdater`` instance."""
        return self._updater

    @property
    def repair_engine(self) -> DatasetRepairEngine:
        """Return the injected ``DatasetRepairEngine`` instance."""
        return self._repair_engine

    @property
    def worker_count(self) -> int:
        """Return the configured maximum concurrent symbol worker count."""
        return self._worker_count

    async def run(self) -> UniverseBootstrapResult:
        """Discover the universe and resumably bootstrap each symbol/timeframe.

        Decision flow per ``(symbol, timeframe)``:

        1. Load existing dataset. If missing, download via
           ``HistoricalBootstrap.download_symbol``.
        2. If present, validate. If invalid, repair via
           ``DatasetRepairEngine.repair_symbol``.
        3. If valid, update via ``IncrementalUpdater.update_symbol``. Already
           current datasets are recorded as skipped after the updater no-op.
        4. Continue to the next timeframe.
        5. Continue to the next symbol even when one pair fails.

        Symbols are drained from a shared queue by ``worker_count`` async
        workers. Each worker processes one symbol at a time and runs that
        symbol's timeframes sequentially.

        Returns:
            Immutable ``UniverseBootstrapResult`` summarizing outcomes.
            Category tuples follow discovery order for deterministic reporting.

        Raises:
            Exception: Propagated from discovery or coverage-window resolution
                failures. Per-pair exceptions are collected and do not abort
                the run.
        """
        contracts = await self._historical_bootstrap.discover_symbols()
        timeframes = self._historical_bootstrap.options.timeframes
        start_time, end_time = _resolve_coverage_window(self._historical_bootstrap)

        if not contracts:
            return _empty_result()

        outcomes: dict[Symbol, _SymbolOutcome] = {
            contract.symbol: _SymbolOutcome() for contract in contracts
        }
        await self._run_worker_pool(
            contracts=contracts,
            timeframes=timeframes,
            start_time=start_time,
            end_time=end_time,
            outcomes=outcomes,
        )
        return _build_result(contracts=contracts, outcomes=outcomes)

    async def _run_worker_pool(
        self,
        *,
        contracts: Sequence[Contract],
        timeframes: Sequence[Timeframe],
        start_time: UnixTimestampMs,
        end_time: UnixTimestampMs,
        outcomes: dict[Symbol, _SymbolOutcome],
    ) -> None:
        """Drain discovered contracts through a bounded asyncio worker pool.

        Args:
            contracts: Discovered contracts in discovery order.
            timeframes: Configured bar intervals.
            start_time: Inclusive coverage window start (Unix ms, UTC).
            end_time: Inclusive coverage window end (Unix ms, UTC).
            outcomes: Mutable per-symbol outcome map shared by workers.
        """
        queue: asyncio.Queue[Contract | None] = asyncio.Queue()
        for contract in contracts:
            queue.put_nowait(contract)
        for _ in range(self._worker_count):
            queue.put_nowait(None)

        async def worker() -> None:
            while True:
                item = await queue.get()
                try:
                    if item is None:
                        return
                    await self._process_symbol(
                        symbol=item.symbol,
                        timeframes=timeframes,
                        start_time=start_time,
                        end_time=end_time,
                        outcome=outcomes[item.symbol],
                    )
                finally:
                    queue.task_done()

        worker_tasks = [
            asyncio.create_task(
                worker(),
                name=f"universe-bootstrap-worker-{index}",
            )
            for index in range(self._worker_count)
        ]
        try:
            await asyncio.gather(*worker_tasks)
        finally:
            for task in worker_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*worker_tasks, return_exceptions=True)

    async def _process_symbol(
        self,
        *,
        symbol: Symbol,
        timeframes: Sequence[Timeframe],
        start_time: UnixTimestampMs,
        end_time: UnixTimestampMs,
        outcome: _SymbolOutcome,
    ) -> None:
        """Process every configured timeframe for one symbol sequentially.

        Args:
            symbol: Tradeable symbol.
            timeframes: Configured bar intervals for this run.
            start_time: Inclusive coverage window start (Unix ms, UTC).
            end_time: Inclusive coverage window end (Unix ms, UTC).
            outcome: Mutable outcome flags for ``symbol``.
        """
        for timeframe in timeframes:
            try:
                label = await self._process_symbol_timeframe(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_time,
                    end_time=end_time,
                )
            except Exception:
                outcome.failed = True
                continue

            if label == "downloaded":
                outcome.downloaded = True
            elif label == "repaired":
                outcome.repaired = True
            elif label == "updated":
                outcome.updated = True
            elif label == "skipped":
                outcome.skipped = True

    async def _process_symbol_timeframe(
        self,
        *,
        symbol: Symbol,
        timeframe: Timeframe,
        start_time: UnixTimestampMs,
        end_time: UnixTimestampMs,
    ) -> str:
        """Apply the resumable bootstrap decision flow for one pair.

        Args:
            symbol: Tradeable symbol.
            timeframe: Bar interval identifier.
            start_time: Inclusive coverage window start (Unix ms, UTC).
            end_time: Inclusive coverage window end (Unix ms, UTC).

        Returns:
            Outcome label: ``downloaded``, ``repaired``, ``updated``, or
            ``skipped``.

        Raises:
            Exception: Propagated from injected collaborator failures.
        """
        try:
            existing = self._load_existing_dataset(symbol=symbol, timeframe=timeframe)
        except CorruptedDatasetError:
            await self._repair_engine.repair_symbol(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
            )
            return "repaired"

        if existing is None:
            await self._historical_bootstrap.download_symbol(
                symbol=symbol,
                timeframe=timeframe,
            )
            return "downloaded"

        report = self._validator.validate(existing, timeframe)
        if not report.is_valid:
            await self._repair_engine.repair_symbol(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
            )
            return "repaired"

        already_current = _is_already_current(existing, end_time=end_time)
        await self._updater.update_symbol(
            symbol=symbol,
            timeframe=timeframe,
            end_time=end_time,
        )
        return "skipped" if already_current else "updated"

    def _load_existing_dataset(
        self,
        *,
        symbol: Symbol,
        timeframe: Timeframe,
    ) -> pl.DataFrame | None:
        """Load and concatenate existing OHLCV year partitions, if any.

        Args:
            symbol: Tradeable symbol.
            timeframe: Bar interval identifier.

        Returns:
            Combined OHLCV frame when at least one partition exists; otherwise
            ``None``.

        Raises:
            CorruptedDatasetError: Propagated when a partition exists but cannot
                be read. Callers treat corruption as a repair path.
        """
        frames: list[pl.DataFrame] = []
        current_year = datetime.now(UTC).year
        for year in range(_EARLIEST_PARTITION_YEAR, current_year + 1):
            try:
                frame = self._repository.load_ohlcv(
                    exchange=_EXCHANGE,
                    market=_MARKET,
                    symbol=symbol,
                    timeframe=timeframe,
                    year=year,
                )
            except DatasetNotFoundError:
                continue
            if frame.height == 0:
                continue
            frames.append(frame)

        if not frames:
            return None
        if len(frames) == 1:
            return frames[0]
        return pl.concat(frames, how="vertical")


def _empty_result() -> UniverseBootstrapResult:
    """Return an empty bootstrap result for an empty discovered universe."""
    return UniverseBootstrapResult(
        successful_symbols=(),
        failed_symbols=(),
        total_symbols=0,
        successful_downloads=0,
        failed_downloads=0,
        downloaded_symbols=(),
        updated_symbols=(),
        repaired_symbols=(),
        skipped_symbols=(),
    )


def _build_result(
    *,
    contracts: Sequence[Contract],
    outcomes: Mapping[Symbol, _SymbolOutcome],
) -> UniverseBootstrapResult:
    """Assemble a deterministic result from discovery-ordered contracts.

    Args:
        contracts: Discovered contracts in discovery order.
        outcomes: Per-symbol outcome flags collected by workers.

    Returns:
        Immutable bootstrap result with category tuples in discovery order.
    """
    successful = tuple(
        contract.symbol for contract in contracts if not outcomes[contract.symbol].failed
    )
    failed = tuple(contract.symbol for contract in contracts if outcomes[contract.symbol].failed)
    downloaded = tuple(
        contract.symbol for contract in contracts if outcomes[contract.symbol].downloaded
    )
    updated = tuple(contract.symbol for contract in contracts if outcomes[contract.symbol].updated)
    repaired = tuple(
        contract.symbol for contract in contracts if outcomes[contract.symbol].repaired
    )
    skipped = tuple(contract.symbol for contract in contracts if outcomes[contract.symbol].skipped)
    return UniverseBootstrapResult(
        successful_symbols=successful,
        failed_symbols=failed,
        total_symbols=len(contracts),
        successful_downloads=len(successful),
        failed_downloads=len(failed),
        downloaded_symbols=downloaded,
        updated_symbols=updated,
        repaired_symbols=repaired,
        skipped_symbols=skipped,
    )


def _resolve_coverage_window(
    historical_bootstrap: HistoricalBootstrap,
) -> tuple[UnixTimestampMs, UnixTimestampMs]:
    """Resolve inclusive coverage start/end from bootstrap options.

    Args:
        historical_bootstrap: Bootstrap providing validated options.

    Returns:
        ``(start_time, end_time)`` as UTC Unix milliseconds.

    Raises:
        ValidationError: If ``start_time`` cannot be derived because both
            ``start_time`` and ``history_days`` are omitted.
    """
    options = historical_bootstrap.options
    end_time = (
        options.end_time
        if options.end_time is not None
        else int(datetime.now(UTC).timestamp() * 1000)
    )

    if options.start_time is not None:
        start_time = options.start_time
    elif options.history_days is not None:
        start_time = max(
            0,
            end_time - (options.history_days * MILLISECONDS_PER_DAY),
        )
    else:
        raise ValidationError(
            "coverage window requires start_time or history_days",
            error_code=_ERROR_DOWNLOAD_RANGE,
            details={
                "start_time": options.start_time,
                "end_time": options.end_time,
                "history_days": options.history_days,
            },
            recovery_suggestion=(
                "Set BootstrapOptions.start_time or BootstrapOptions.history_days "
                "before running UniverseBootstrap."
            ),
        )

    return start_time, end_time


def _is_already_current(
    frame: pl.DataFrame,
    *,
    end_time: UnixTimestampMs,
) -> bool:
    """Return whether stored data already reaches the coverage end.

    Args:
        frame: Existing OHLCV dataset.
        end_time: Inclusive coverage window end (Unix ms, UTC).

    Returns:
        ``True`` when the next missing open time would start after ``end_time``.
    """
    latest = frame.get_column(
        "open_time"
    ).max()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    if latest is None:
        return False
    return int(latest) + 1 > end_time  # pyright: ignore[reportArgumentType]
