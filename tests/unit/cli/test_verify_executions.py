"""Unit tests for CQROS executed-trade verification CLI."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

import cqros.cli.verify_executions as verify_executions_module
from cqros.cli.verify_executions import (
    DiscoveredWorkItem,
    VerifyExecutionOptions,
    VerifyExecutionSummary,
    build_options,
    build_parser,
    discover_work,
    format_partition_failure,
    format_summary,
    main,
    run_verification,
)
from cqros.config.models import ResearchConfig
from cqros.core.constants import (
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
    STORAGE_DIR_EXECUTIONS,
)
from cqros.core.exceptions import ValidationError
from cqros.execution import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    ExecutionStatus,
    ExecutionVerifier,
    TradeRepository,
)
from cqros.storage import ParquetStore, StorageLayout

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
) -> VerifyExecutionOptions:
    """Build verification options for tests."""
    return VerifyExecutionOptions(
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


def _trade_frame() -> pl.DataFrame:
    """Return a canonical passing trade frame."""
    open_time = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
    return pl.DataFrame(
        {
            "symbol": [_SYMBOL],
            "timeframe": [_TIMEFRAME],
            "open_time": [open_time],
            "model_name": ["alpha-lgbm"],
            "model_version": ["1.0.0"],
            "optimizer": ["equal_weight"],
            "policy": ["fixed_risk"],
            "manager": [_MANAGER],
            "signal": ["BUY"],
            "side": ["BUY"],
            "order_type": ["MARKET"],
            "requested_quantity": [1.0],
            "executed_quantity": [1.0],
            "requested_price": [100.0],
            "executed_price": [100.0],
            "fees": [0.0],
            "slippage": [0.0],
            "status": [ExecutionStatus.FILLED.value],
            "execution_time": [open_time],
        },
        schema=dict(COLUMN_DTYPES),
    ).select(list(CANONICAL_COLUMN_ORDER))


def test_build_parser_and_options(tmp_path: Path) -> None:
    """Parser accepts filters; options reject invalid workers."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "simple", "--workers", "2"])
    with patch.object(verify_executions_module, "DEFAULT_STORAGE_ROOT", tmp_path):
        options = build_options(args)
    assert options.manager == "simple"
    assert options.workers == 2

    args = parser.parse_args(["--workers", "0"])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-VERIFY-EXECUTIONS-001"


def test_discover_work_and_verify_pass(tmp_path: Path) -> None:
    """Discovery finds partitions and verification passes for clean frames."""
    repository = TradeRepository(StorageLayout(tmp_path), ParquetStore())
    repository.save(
        _trade_frame(),
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
            verifier=ExecutionVerifier(),
            options=options,
            work=work,
        )
    )
    assert summary.repository_passed is True
    assert summary.successful_tasks == 1
    assert summary.rows_checked == 1


def test_format_summary_and_partition_failure() -> None:
    """Summary and failure formatters render expected labels."""
    summary = VerifyExecutionSummary(
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
        dataset="Executions",
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        partition="2026.parquet",
        verifier="ExecutionVerifier",
        exception_type="ExecutionValidationError",
        message="missing required columns",
        code="EXEC-VERIFICATION-001",
    )
    assert "FAILED" in failure
    assert "ExecutionVerifier" in failure
    assert "EXEC-VERIFICATION-001" in failure


def test_main_returns_success_for_empty_universe(tmp_path: Path) -> None:
    """main returns 0 when no execution partitions are discovered."""
    with patch.object(verify_executions_module, "DEFAULT_STORAGE_ROOT", tmp_path):
        code = _run(main([]))
    assert code == 0
    assert (tmp_path / STORAGE_DIR_EXECUTIONS).exists() is False
