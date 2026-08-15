"""Unit tests for CQROS trade-management verification CLI."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

import cqros.cli.verify_trade_management as verify_trade_management_module
from cqros.cli.verify_trade_management import (
    DiscoveredWorkItem,
    VerifyTradeManagementOptions,
    VerifyTradeManagementSummary,
    build_options,
    build_parser,
    discover_work,
    format_partition_failure,
    format_summary,
    run_verification,
)
from cqros.config.models import ResearchConfig
from cqros.core.constants import (
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
)
from cqros.core.exceptions import ValidationError
from cqros.storage import ParquetStore, StorageLayout
from cqros.trade_management import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    ManagementAction,
    ShutdownReason,
    TradeManagementRepository,
    TradeManagementVerifier,
)

_MANAGER = "simple"
_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1h"
_YEAR = 2026


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def _options(
    *,
    storage_root: Path,
    manager: str | None = _MANAGER,
    model: str | None = None,
    version: str | None = None,
    symbols: tuple[str, ...] | None = None,
    timeframes: tuple[str, ...] | None = None,
    years: tuple[int, ...] | None = None,
    workers: int = ResearchConfig().worker_count,
    verbose: bool = False,
    debug: bool = False,
) -> VerifyTradeManagementOptions:
    """Build verification options for tests."""
    return VerifyTradeManagementOptions(
        storage_root=storage_root,
        manager=manager,
        model=model,
        version=version,
        symbols=symbols,
        timeframes=timeframes,
        years=years,
        workers=workers,
        verbose=verbose,
        debug=debug,
    )


def _trade_management_frame(*, symbol: str = _SYMBOL) -> pl.DataFrame:
    """Return a canonical passing trade-management frame."""
    open_time = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
    return pl.DataFrame(
        {
            "symbol": [symbol],
            "timeframe": [_TIMEFRAME],
            "open_time": [open_time],
            "manager": [_MANAGER],
            "position_id": ["pos-00000001"],
            "position_status": ["OPEN"],
            "quantity": [1.0],
            "entry_price": [100.0],
            "current_price": [104.0],
            "highest_price": [104.0],
            "lowest_price": [104.0],
            "unrealized_pnl": [0.0],
            "risk_state": ["NORMAL"],
            "management_action": [ManagementAction.NONE.value],
            "action_reason": [ShutdownReason.NONE.value],
            "stop_price": [None],
            "take_profit_price": [None],
            "trail_price": [98.8],
            "breakeven_price": [None],
            "allow_pyramid": [False],
            "exit_quantity": [0.0],
            "model_name": ["alpha-lgbm"],
            "model_version": ["1.0.0"],
            "optimizer": ["equal_weight"],
            "policy": ["fixed_risk"],
        },
        schema=dict(COLUMN_DTYPES),
    ).select(list(CANONICAL_COLUMN_ORDER))


def test_build_parser_and_options(tmp_path: Path) -> None:
    """Parser accepts filters; options reject invalid workers."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "simple", "--workers", "2"])
    with patch.object(verify_trade_management_module, "DEFAULT_STORAGE_ROOT", tmp_path):
        options = build_options(args)
    assert options.manager == "simple"
    assert options.workers == 2

    args = parser.parse_args(["--workers", "0"])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-VERIFY-TRADE-MANAGEMENT-001"


def test_discover_work_and_verify_pass(tmp_path: Path) -> None:
    """Discovery finds partitions and verification passes for clean frames."""
    repository = TradeManagementRepository(StorageLayout(tmp_path), ParquetStore())
    repository.save(
        _trade_management_frame(),
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    options = _options(storage_root=tmp_path)
    work = discover_work(repository, options)
    assert work == (
        DiscoveredWorkItem(
            manager=_MANAGER,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            years=(_YEAR,),
        ),
    )
    summary = _run(
        run_verification(
            repository=repository,
            verifier=TradeManagementVerifier(),
            options=options,
            work=work,
        )
    )
    assert summary.repository_passed is True
    assert summary.successful_tasks == 1
    assert summary.rows_checked == 1


def test_run_verification_empty_work(tmp_path: Path) -> None:
    """Empty work produces a passing zeroed verification summary."""
    summary = _run(
        run_verification(
            repository=TradeManagementRepository(StorageLayout(tmp_path), ParquetStore()),
            verifier=TradeManagementVerifier(),
            options=_options(storage_root=tmp_path),
            work=(),
        )
    )
    assert summary.repository_passed is True
    assert summary.successful_tasks == 0
    assert summary.failed_tasks == 0
    assert summary.rows_checked == 0


def test_format_summary_and_partition_failure() -> None:
    """Summary and failure formatters render expected labels."""
    summary = VerifyTradeManagementSummary(
        symbols_verified=1,
        datasets_verified=1,
        timeframes_verified=1,
        successful_tasks=1,
        failed_tasks=0,
        rows_checked=1,
        duplicate_timestamps=0,
        null_rows=0,
        nan_rows=0,
        invalid_timestamps=0,
        invalid_status_rows=0,
        warnings=0,
        duration_seconds=0.5,
        repository_passed=True,
    )
    text = format_summary(summary)
    assert "CQROS Verification Summary" in text
    assert "PASS" in text
    failure = format_partition_failure(
        dataset="Trade Management",
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        partition="2026.parquet",
        verifier="TradeManagementVerifier",
        exception_type="TradeManagementValidationError",
        message="missing required columns",
        code="TME-VERIFICATION-001",
    )
    assert "FAILED" in failure
    assert _SYMBOL in failure
    assert "TradeManagementVerifier" in failure
    assert "TME-VERIFICATION-001" in failure


def test_run_verification_fail_on_invalid_frame(tmp_path: Path) -> None:
    """Verification fails when a partition raises a schema validation error."""
    repository = TradeManagementRepository(StorageLayout(tmp_path), ParquetStore())
    invalid = _trade_management_frame().drop("entry_price")
    repository.save(
        invalid,
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    options = _options(storage_root=tmp_path)
    work = discover_work(repository, options)
    summary = _run(
        run_verification(
            repository=repository,
            verifier=TradeManagementVerifier(),
            options=options,
            work=work,
        )
    )
    assert summary.repository_passed is False
    assert summary.failed_tasks == 1
    assert summary.successful_tasks == 0


def test_run_verification_multiple_partitions(tmp_path: Path) -> None:
    """Verification aggregates rows across multiple symbols and years."""
    repository = TradeManagementRepository(StorageLayout(tmp_path), ParquetStore())
    for symbol, year in (("BTCUSDT", 2025), ("ETHUSDT", 2026)):
        repository.save(
            _trade_management_frame(symbol=symbol),
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            symbol=symbol,
            timeframe=_TIMEFRAME,
            year=year,
        )
    options = _options(storage_root=tmp_path)
    work = discover_work(repository, options)
    summary = _run(
        run_verification(
            repository=repository,
            verifier=TradeManagementVerifier(),
            options=options,
            work=work,
        )
    )
    assert summary.repository_passed is True
    assert summary.successful_tasks == 2
    assert summary.symbols_verified == 2
    assert summary.rows_checked == 2
