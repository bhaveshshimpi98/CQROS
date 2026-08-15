"""Unit tests for CQROS resumable historical universe bootstrap orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from dataclasses import FrozenInstanceError, is_dataclass
from unittest.mock import AsyncMock, MagicMock

import polars as pl
import pytest

from cqros.bootstrap.historical import BootstrapOptions, HistoricalBootstrap
from cqros.bootstrap.universe import (
    UniverseBootstrap,
    UniverseBootstrapResult,
)
from cqros.core.exceptions import ValidationError
from cqros.data.contracts import (
    Contract,
    ContractStatus,
    ContractType,
    PriceFilter,
    QuantityFilter,
)
from cqros.ingestion.downloader import DEFAULT_DOWNLOAD_WORKERS
from cqros.ingestion.repair import DatasetRepairEngine
from cqros.ingestion.updater import IncrementalUpdater
from cqros.ingestion.validator import (
    MarketDataValidator,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)
from cqros.storage.exceptions import DatasetNotFoundError
from cqros.storage.repository import MarketDataRepository

_END_TIME = 1_700_000_120_000
_START_TIME = 1_700_000_000_000
_TIMEFRAME = "1h"


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def _contract(symbol: str) -> Contract:
    """Build a minimal immutable contract for universe tests."""
    return Contract(
        symbol=symbol,
        exchange="binance",
        base_asset=symbol.removesuffix("USDT"),
        quote_asset="USDT",
        contract_type=ContractType.PERPETUAL,
        status=ContractStatus.TRADING,
        price_filter=PriceFilter(tick_size=0.1),
        quantity_filter=QuantityFilter(step_size=0.001, min_quantity=0.001),
    )


def _ohlcv_frame(symbol: str, *open_times: int) -> pl.DataFrame:
    """Build a minimal OHLCV frame for repository/validator stubs."""
    return pl.DataFrame(
        [
            {
                "symbol": symbol,
                "timeframe": _TIMEFRAME,
                "open_time": open_time,
                "close_time": open_time + 3_599_999,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10.0,
                "quote_volume": 1000.0,
                "trade_count": 42,
            }
            for open_time in open_times
        ]
    )


def _empty_result(**overrides: object) -> UniverseBootstrapResult:
    """Build a zeroed result with optional field overrides."""
    payload: dict[str, object] = {
        "successful_symbols": (),
        "failed_symbols": (),
        "total_symbols": 0,
        "successful_downloads": 0,
        "failed_downloads": 0,
        "downloaded_symbols": (),
        "updated_symbols": (),
        "repaired_symbols": (),
        "skipped_symbols": (),
    }
    payload.update(overrides)
    return UniverseBootstrapResult(**payload)  # type: ignore[arg-type]


def _build_universe(
    *,
    contracts: tuple[Contract, ...] = (),
    timeframes: tuple[str, ...] = (_TIMEFRAME,),
    existing: dict[tuple[str, str], pl.DataFrame | None] | None = None,
    valid: bool = True,
    download_error: Exception | None = None,
    update_error: Exception | None = None,
    repair_error: Exception | None = None,
    worker_count: int = 1,
) -> tuple[UniverseBootstrap, MagicMock, MagicMock, MagicMock, MagicMock, MagicMock]:
    """Compose a UniverseBootstrap with mocked collaborators."""
    historical = MagicMock(spec=HistoricalBootstrap)
    historical.discover_symbols = AsyncMock(return_value=contracts)
    historical.options = BootstrapOptions(
        timeframes=timeframes,
        start_time=_START_TIME,
        end_time=_END_TIME,
        history_days=None,
    )
    if download_error is None:
        historical.download_symbol = AsyncMock(return_value=None)
    else:
        historical.download_symbol = AsyncMock(side_effect=download_error)

    frames = existing if existing is not None else {}

    def _load_ohlcv(
        *,
        exchange: str,
        market: str,
        symbol: str,
        timeframe: str,
        year: int,
    ) -> pl.DataFrame:
        del exchange, market
        key = (symbol, timeframe)
        if key not in frames or frames[key] is None or year != 2023:
            raise DatasetNotFoundError(
                "missing",
                error_code="STORAGE-TEST-001",
                details={"symbol": symbol, "timeframe": timeframe, "year": year},
            )
        frame = frames[key]
        assert frame is not None
        return frame

    repository = MagicMock(spec=MarketDataRepository)
    repository.load_ohlcv = MagicMock(side_effect=_load_ohlcv)

    validator = MagicMock(spec=MarketDataValidator)
    if valid:
        validator.validate = MagicMock(
            return_value=ValidationReport(timeframe=_TIMEFRAME, row_count=1, issues=())
        )
    else:
        validator.validate = MagicMock(
            return_value=ValidationReport(
                timeframe=_TIMEFRAME,
                row_count=1,
                issues=(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        check="schema",
                        message="invalid",
                    ),
                ),
            )
        )

    updater = MagicMock(spec=IncrementalUpdater)
    if update_error is None:
        updater.update_symbol = AsyncMock(return_value=None)
    else:
        updater.update_symbol = AsyncMock(side_effect=update_error)

    repair_engine = MagicMock(spec=DatasetRepairEngine)
    if repair_error is None:
        repair_engine.repair_symbol = AsyncMock(return_value=None)
    else:
        repair_engine.repair_symbol = AsyncMock(side_effect=repair_error)

    universe = UniverseBootstrap(
        historical,
        repository,
        validator,
        updater,
        repair_engine,
        worker_count=worker_count,
    )
    return universe, historical, repository, validator, updater, repair_engine


def test_universe_bootstrap_result_is_immutable() -> None:
    """UniverseBootstrapResult rejects attribute assignment."""
    result = _empty_result(
        successful_symbols=("BTCUSDT",),
        total_symbols=1,
        successful_downloads=1,
        downloaded_symbols=("BTCUSDT",),
    )
    assert is_dataclass(result)
    with pytest.raises(FrozenInstanceError):
        result.total_symbols = 2  # type: ignore[misc]


def test_universe_bootstrap_dependency_injection() -> None:
    """UniverseBootstrap stores all injected collaborators."""
    historical = HistoricalBootstrap(BootstrapOptions(history_days=1, timeframes=(_TIMEFRAME,)))
    repository = MagicMock(spec=MarketDataRepository)
    validator = MagicMock(spec=MarketDataValidator)
    updater = MagicMock(spec=IncrementalUpdater)
    repair_engine = MagicMock(spec=DatasetRepairEngine)

    universe = UniverseBootstrap(
        historical,
        repository,
        validator,
        updater,
        repair_engine,
    )

    assert universe.historical_bootstrap is historical
    assert universe.repository is repository
    assert universe.validator is validator
    assert universe.updater is updater
    assert universe.repair_engine is repair_engine
    assert universe.worker_count == historical.options.workers
    assert universe.worker_count == DEFAULT_DOWNLOAD_WORKERS


def test_universe_bootstrap_uses_options_workers_when_omitted() -> None:
    """Omitted worker_count resolves from HistoricalBootstrap options."""
    historical = HistoricalBootstrap(
        BootstrapOptions(history_days=1, timeframes=(_TIMEFRAME,), workers=6)
    )
    universe = UniverseBootstrap(
        historical,
        MagicMock(spec=MarketDataRepository),
        MagicMock(spec=MarketDataValidator),
        MagicMock(spec=IncrementalUpdater),
        MagicMock(spec=DatasetRepairEngine),
    )
    assert universe.worker_count == 6


def test_universe_bootstrap_rejects_non_positive_worker_count() -> None:
    """Construction fails fast when worker_count is not positive."""
    with pytest.raises(ValidationError) as exc_info:
        UniverseBootstrap(
            MagicMock(spec=HistoricalBootstrap),
            MagicMock(spec=MarketDataRepository),
            MagicMock(spec=MarketDataValidator),
            MagicMock(spec=IncrementalUpdater),
            MagicMock(spec=DatasetRepairEngine),
            worker_count=0,
        )
    assert exc_info.value.error_code == "BOOTSTRAP-UNIVERSE-002"


def test_run_fresh_bootstrap_downloads_missing_datasets() -> None:
    """Fresh bootstrap downloads every missing symbol/timeframe pair."""
    contracts = (_contract("BTCUSDT"), _contract("ETHUSDT"))
    universe, historical, _, validator, updater, repair_engine = _build_universe(
        contracts=contracts,
        existing={},
    )

    result = _run(universe.run())

    historical.discover_symbols.assert_awaited_once_with()
    assert historical.download_symbol.await_count == 2
    historical.download_symbol.assert_any_await(symbol="BTCUSDT", timeframe=_TIMEFRAME)
    historical.download_symbol.assert_any_await(symbol="ETHUSDT", timeframe=_TIMEFRAME)
    validator.validate.assert_not_called()
    updater.update_symbol.assert_not_awaited()
    repair_engine.repair_symbol.assert_not_awaited()
    assert result == _empty_result(
        successful_symbols=("BTCUSDT", "ETHUSDT"),
        total_symbols=2,
        successful_downloads=2,
        downloaded_symbols=("BTCUSDT", "ETHUSDT"),
    )


def test_run_existing_valid_dataset_updates_when_stale() -> None:
    """Existing valid but stale datasets are incrementally updated."""
    contracts = (_contract("BTCUSDT"),)
    frame = _ohlcv_frame("BTCUSDT", _START_TIME)
    universe, historical, _, validator, updater, repair_engine = _build_universe(
        contracts=contracts,
        existing={("BTCUSDT", _TIMEFRAME): frame},
        valid=True,
    )

    result = _run(universe.run())

    historical.download_symbol.assert_not_awaited()
    validator.validate.assert_called_once()
    updater.update_symbol.assert_awaited_once_with(
        symbol="BTCUSDT",
        timeframe=_TIMEFRAME,
        end_time=_END_TIME,
    )
    repair_engine.repair_symbol.assert_not_awaited()
    assert result == _empty_result(
        successful_symbols=("BTCUSDT",),
        total_symbols=1,
        successful_downloads=1,
        updated_symbols=("BTCUSDT",),
    )


def test_run_existing_valid_current_dataset_is_skipped() -> None:
    """Existing valid datasets already at end_time skip redownload."""
    contracts = (_contract("BTCUSDT"),)
    frame = _ohlcv_frame("BTCUSDT", _END_TIME)
    universe, historical, _, _, updater, repair_engine = _build_universe(
        contracts=contracts,
        existing={("BTCUSDT", _TIMEFRAME): frame},
        valid=True,
    )

    result = _run(universe.run())

    historical.download_symbol.assert_not_awaited()
    updater.update_symbol.assert_awaited_once_with(
        symbol="BTCUSDT",
        timeframe=_TIMEFRAME,
        end_time=_END_TIME,
    )
    repair_engine.repair_symbol.assert_not_awaited()
    assert result == _empty_result(
        successful_symbols=("BTCUSDT",),
        total_symbols=1,
        successful_downloads=1,
        skipped_symbols=("BTCUSDT",),
    )


def test_run_existing_invalid_dataset_is_repaired() -> None:
    """Existing invalid datasets are repaired instead of redownloaded."""
    contracts = (_contract("BTCUSDT"),)
    frame = _ohlcv_frame("BTCUSDT", _START_TIME)
    universe, historical, _, validator, updater, repair_engine = _build_universe(
        contracts=contracts,
        existing={("BTCUSDT", _TIMEFRAME): frame},
        valid=False,
    )

    result = _run(universe.run())

    historical.download_symbol.assert_not_awaited()
    validator.validate.assert_called_once()
    repair_engine.repair_symbol.assert_awaited_once_with(
        symbol="BTCUSDT",
        timeframe=_TIMEFRAME,
        start_time=_START_TIME,
        end_time=_END_TIME,
    )
    updater.update_symbol.assert_not_awaited()
    assert result == _empty_result(
        successful_symbols=("BTCUSDT",),
        total_symbols=1,
        successful_downloads=1,
        repaired_symbols=("BTCUSDT",),
    )


def test_run_missing_dataset_downloads_symbol() -> None:
    """Missing datasets call HistoricalBootstrap.download_symbol."""
    contracts = (_contract("SOLUSDT"),)
    universe, historical, repository, _, updater, repair_engine = _build_universe(
        contracts=contracts,
        existing={("SOLUSDT", _TIMEFRAME): None},
    )

    result = _run(universe.run())

    assert repository.load_ohlcv.called
    historical.download_symbol.assert_awaited_once_with(
        symbol="SOLUSDT",
        timeframe=_TIMEFRAME,
    )
    updater.update_symbol.assert_not_awaited()
    repair_engine.repair_symbol.assert_not_awaited()
    assert result.downloaded_symbols == ("SOLUSDT",)
    assert result.failed_symbols == ()


def test_run_partial_failures_continues_remaining_symbols() -> None:
    """Per-symbol failures are recorded while remaining symbols continue."""
    contracts = (
        _contract("BTCUSDT"),
        _contract("ETHUSDT"),
        _contract("SOLUSDT"),
    )

    historical = MagicMock(spec=HistoricalBootstrap)
    historical.discover_symbols = AsyncMock(return_value=contracts)
    historical.options = BootstrapOptions(
        timeframes=(_TIMEFRAME,),
        start_time=_START_TIME,
        end_time=_END_TIME,
    )

    async def _download(*, symbol: str, timeframe: str) -> None:
        del timeframe
        if symbol == "ETHUSDT":
            raise ValidationError(
                "simulated download failure",
                error_code="TEST-UNIVERSE-001",
                details={"symbol": symbol},
            )

    historical.download_symbol = AsyncMock(side_effect=_download)

    repository = MagicMock(spec=MarketDataRepository)
    repository.load_ohlcv = MagicMock(
        side_effect=DatasetNotFoundError(
            "missing",
            error_code="STORAGE-TEST-001",
        )
    )
    validator = MagicMock(spec=MarketDataValidator)
    updater = MagicMock(spec=IncrementalUpdater)
    updater.update_symbol = AsyncMock(return_value=None)
    repair_engine = MagicMock(spec=DatasetRepairEngine)
    repair_engine.repair_symbol = AsyncMock(return_value=None)

    result = _run(
        UniverseBootstrap(
            historical,
            repository,
            validator,
            updater,
            repair_engine,
            worker_count=1,
        ).run()
    )

    assert historical.download_symbol.await_count == 3
    assert result == _empty_result(
        successful_symbols=("BTCUSDT", "SOLUSDT"),
        failed_symbols=("ETHUSDT",),
        total_symbols=3,
        successful_downloads=2,
        failed_downloads=1,
        downloaded_symbols=("BTCUSDT", "SOLUSDT"),
    )


def test_run_restart_after_interruption_skips_completed_datasets() -> None:
    """Restart does not redownload completed valid datasets."""
    contracts = (_contract("BTCUSDT"), _contract("ETHUSDT"))
    existing = {
        ("BTCUSDT", _TIMEFRAME): _ohlcv_frame("BTCUSDT", _END_TIME),
        ("ETHUSDT", _TIMEFRAME): None,
    }
    universe, historical, _, _, updater, repair_engine = _build_universe(
        contracts=contracts,
        existing=existing,
        valid=True,
    )

    result = _run(universe.run())

    historical.download_symbol.assert_awaited_once_with(
        symbol="ETHUSDT",
        timeframe=_TIMEFRAME,
    )
    updater.update_symbol.assert_awaited_once_with(
        symbol="BTCUSDT",
        timeframe=_TIMEFRAME,
        end_time=_END_TIME,
    )
    repair_engine.repair_symbol.assert_not_awaited()
    assert result == _empty_result(
        successful_symbols=("BTCUSDT", "ETHUSDT"),
        total_symbols=2,
        successful_downloads=2,
        downloaded_symbols=("ETHUSDT",),
        skipped_symbols=("BTCUSDT",),
    )


def test_run_empty_universe() -> None:
    """run() returns an empty result when discovery finds no contracts."""
    universe, historical, _, _, updater, _ = _build_universe(contracts=())

    result = _run(universe.run())

    historical.discover_symbols.assert_awaited_once_with()
    historical.download_symbol.assert_not_awaited()
    updater.update_symbol.assert_not_awaited()
    assert result == _empty_result()


def test_run_processes_multiple_timeframes_independently() -> None:
    """Each timeframe for a symbol follows the decision flow independently."""
    contracts = (_contract("BTCUSDT"),)
    frame_4h = _ohlcv_frame("BTCUSDT", _START_TIME).with_columns(pl.lit("4h").alias("timeframe"))
    existing = {
        ("BTCUSDT", "1h"): None,
        ("BTCUSDT", "4h"): frame_4h,
    }

    universe, historical, _, validator, updater, repair_engine = _build_universe(
        contracts=contracts,
        timeframes=("1h", "4h"),
        existing=existing,
        valid=True,
    )

    result = _run(universe.run())

    historical.download_symbol.assert_awaited_once_with(symbol="BTCUSDT", timeframe="1h")
    updater.update_symbol.assert_awaited_once_with(
        symbol="BTCUSDT",
        timeframe="4h",
        end_time=_END_TIME,
    )
    repair_engine.repair_symbol.assert_not_awaited()
    assert validator.validate.call_count == 1
    assert result.downloaded_symbols == ("BTCUSDT",)
    assert result.updated_symbols == ("BTCUSDT",)
    assert result.failed_symbols == ()


def test_run_worker_count_one_matches_sequential_accounting() -> None:
    """worker_count=1 preserves sequential-style discovery-ordered results."""
    contracts = (_contract("BTCUSDT"), _contract("ETHUSDT"), _contract("SOLUSDT"))
    universe, historical, _, _, _, _ = _build_universe(
        contracts=contracts,
        existing={},
        worker_count=1,
    )

    result = _run(universe.run())

    assert historical.download_symbol.await_count == 3
    assert result == _empty_result(
        successful_symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
        total_symbols=3,
        successful_downloads=3,
        downloaded_symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
    )


def test_run_worker_count_greater_than_one_processes_all_symbols() -> None:
    """Multiple workers drain the full universe without dropping symbols."""
    contracts = tuple(_contract(symbol) for symbol in ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT"))
    universe, historical, _, _, _, _ = _build_universe(
        contracts=contracts,
        existing={},
        worker_count=3,
    )

    result = _run(universe.run())

    assert historical.download_symbol.await_count == 4
    assert set(result.successful_symbols) == {contract.symbol for contract in contracts}
    assert result.successful_symbols == ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT")
    assert result.downloaded_symbols == ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT")
    assert result.failed_symbols == ()
    assert result.total_symbols == 4
    assert result.successful_downloads == 4


def test_run_bounds_maximum_concurrent_symbols_to_worker_count() -> None:
    """At most worker_count symbols are in-flight concurrently."""
    contracts = tuple(
        _contract(symbol)
        for symbol in (
            "S1USDT",
            "S2USDT",
            "S3USDT",
            "S4USDT",
            "S5USDT",
            "S6USDT",
        )
    )
    worker_count = 2
    current = 0
    max_seen = 0
    lock = asyncio.Lock()

    historical = MagicMock(spec=HistoricalBootstrap)
    historical.discover_symbols = AsyncMock(return_value=contracts)
    historical.options = BootstrapOptions(
        timeframes=(_TIMEFRAME,),
        start_time=_START_TIME,
        end_time=_END_TIME,
    )

    async def _download(*, symbol: str, timeframe: str) -> None:
        nonlocal current, max_seen
        del symbol, timeframe
        async with lock:
            current += 1
            max_seen = max(max_seen, current)
        await asyncio.sleep(0.02)
        async with lock:
            current -= 1

    historical.download_symbol = AsyncMock(side_effect=_download)
    repository = MagicMock(spec=MarketDataRepository)
    repository.load_ohlcv = MagicMock(
        side_effect=DatasetNotFoundError(
            "missing",
            error_code="STORAGE-TEST-001",
        )
    )

    result = _run(
        UniverseBootstrap(
            historical,
            repository,
            MagicMock(spec=MarketDataValidator),
            MagicMock(spec=IncrementalUpdater),
            MagicMock(spec=DatasetRepairEngine),
            worker_count=worker_count,
        ).run()
    )

    assert max_seen <= worker_count
    assert max_seen >= 1
    assert result.successful_downloads == 6
    assert result.failed_downloads == 0


def test_run_partial_failures_with_multiple_workers() -> None:
    """Worker-pool failures are isolated and discovery-ordered in the result."""
    contracts = (
        _contract("BTCUSDT"),
        _contract("ETHUSDT"),
        _contract("SOLUSDT"),
        _contract("XRPUSDT"),
    )

    historical = MagicMock(spec=HistoricalBootstrap)
    historical.discover_symbols = AsyncMock(return_value=contracts)
    historical.options = BootstrapOptions(
        timeframes=(_TIMEFRAME,),
        start_time=_START_TIME,
        end_time=_END_TIME,
    )

    async def _download(*, symbol: str, timeframe: str) -> None:
        del timeframe
        if symbol in {"ETHUSDT", "XRPUSDT"}:
            raise ValidationError(
                "simulated download failure",
                error_code="TEST-UNIVERSE-002",
                details={"symbol": symbol},
            )
        await asyncio.sleep(0)

    historical.download_symbol = AsyncMock(side_effect=_download)
    repository = MagicMock(spec=MarketDataRepository)
    repository.load_ohlcv = MagicMock(
        side_effect=DatasetNotFoundError(
            "missing",
            error_code="STORAGE-TEST-001",
        )
    )

    result = _run(
        UniverseBootstrap(
            historical,
            repository,
            MagicMock(spec=MarketDataValidator),
            MagicMock(spec=IncrementalUpdater),
            MagicMock(spec=DatasetRepairEngine),
            worker_count=3,
        ).run()
    )

    assert historical.download_symbol.await_count == 4
    assert result == _empty_result(
        successful_symbols=("BTCUSDT", "SOLUSDT"),
        failed_symbols=("ETHUSDT", "XRPUSDT"),
        total_symbols=4,
        successful_downloads=2,
        failed_downloads=2,
        downloaded_symbols=("BTCUSDT", "SOLUSDT"),
    )


def test_run_empty_universe_does_not_start_downloads() -> None:
    """Empty discovery returns an empty result without download attempts."""
    universe, historical, _, _, updater, _ = _build_universe(
        contracts=(),
        worker_count=4,
    )

    result = _run(universe.run())

    historical.discover_symbols.assert_awaited_once_with()
    historical.download_symbol.assert_not_awaited()
    updater.update_symbol.assert_not_awaited()
    assert result == _empty_result()


def test_run_all_workers_exit_after_queue_drains() -> None:
    """Worker tasks complete cleanly after the contract queue is exhausted."""
    contracts = (_contract("BTCUSDT"), _contract("ETHUSDT"))
    universe, _, _, _, _, _ = _build_universe(
        contracts=contracts,
        existing={},
        worker_count=4,
    )

    created: list[asyncio.Task[None]] = []
    real_create_task = asyncio.create_task

    def _track_create_task(
        coro: Coroutine[object, object, None],
        **kwargs: object,
    ) -> asyncio.Task[None]:
        task = real_create_task(coro, **kwargs)  # type: ignore[arg-type]
        created.append(task)
        return task

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "cqros.bootstrap.universe.asyncio.create_task",
            _track_create_task,
        )
        result = _run(universe.run())

    assert len(created) == 4
    assert all(task.done() for task in created)
    assert all(not task.cancelled() for task in created)
    assert result.total_symbols == 2
    assert result.successful_downloads == 2


def test_run_result_accounting_is_discovery_ordered() -> None:
    """Category tuples remain discovery-ordered under concurrent workers."""
    contracts = (
        _contract("AAAUSDT"),
        _contract("BBBUSDT"),
        _contract("CCCUSDT"),
    )
    existing = {
        ("AAAUSDT", _TIMEFRAME): _ohlcv_frame("AAAUSDT", _END_TIME),
        ("BBBUSDT", _TIMEFRAME): None,
        ("CCCUSDT", _TIMEFRAME): _ohlcv_frame("CCCUSDT", _START_TIME),
    }
    universe, _, _, _, _, _ = _build_universe(
        contracts=contracts,
        existing=existing,
        valid=True,
        worker_count=3,
    )

    result = _run(universe.run())

    assert result.successful_symbols == ("AAAUSDT", "BBBUSDT", "CCCUSDT")
    assert result.downloaded_symbols == ("BBBUSDT",)
    assert result.updated_symbols == ("CCCUSDT",)
    assert result.skipped_symbols == ("AAAUSDT",)
    assert result.failed_symbols == ()
    assert result.successful_downloads == 3
    assert result.failed_downloads == 0
