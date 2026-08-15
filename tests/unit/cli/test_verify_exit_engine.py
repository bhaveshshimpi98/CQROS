"""Unit tests for CQROS exit-engine verification CLI."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

import cqros.cli.verify_exit_engine as verify_exit_engine_module
from cqros.cli.verify_exit_engine import (
    DiscoveredWorkItem,
    VerifyExitEngineOptions,
    VerifyExitEngineSummary,
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
from cqros.exit_engine import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    ExitAction,
    ExitEngineVerifier,
    ExitReason,
    ExitRepository,
)
from cqros.storage import ParquetStore, StorageLayout

_MANAGER = "simple"
_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1h"
_YEAR = 2026
_OPEN_TIME = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
_POSITION_ID = "pos-00000001"


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
) -> VerifyExitEngineOptions:
    """Build VerifyExitEngineOptions against a temporary storage root."""
    return VerifyExitEngineOptions(
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


def _canonical_frame(
    *,
    symbol: str = _SYMBOL,
    position_id: str = _POSITION_ID,
    exit_action: str = ExitAction.HOLD.value,
    exit_reason: str = ExitReason.NONE.value,
    recommended_percent: float = 0.0,
    recommended_quantity: float = 0.0,
    priority: int = 0,
) -> pl.DataFrame:
    """Return a canonical passing exit-engine frame for CLI verification tests."""
    return pl.DataFrame(
        {
            "symbol": [symbol],
            "timeframe": [_TIMEFRAME],
            "open_time": [_OPEN_TIME],
            "position_id": [position_id],
            "manager": [_MANAGER],
            "entry_price": [100.0],
            "current_price": [102.0],
            "quantity": [1.0],
            "risk_reward_ratio": [0.4],
            "risk_state": ["NORMAL"],
            "trade_state": ["NONE"],
            "pyramid_state": ["READY_TO_ADD"],
            "exit_action": [exit_action],
            "exit_reason": [exit_reason],
            "recommended_quantity": [recommended_quantity],
            "recommended_percent": [recommended_percent],
            "priority": [priority],
            "created_at": [_OPEN_TIME],
        },
        schema=dict(COLUMN_DTYPES),
    ).select(list(CANONICAL_COLUMN_ORDER))


# ---------------------------------------------------------------------------
# Parser and options
# ---------------------------------------------------------------------------


def test_build_parser_and_options(tmp_path: Path) -> None:
    """Parser accepts filter flags; options reject invalid workers."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "simple", "--workers", "2"])
    with patch.object(verify_exit_engine_module, "DEFAULT_STORAGE_ROOT", str(tmp_path)):
        options = build_options(args)
    assert options.manager == "simple"
    assert options.workers == 2

    args_bad = parser.parse_args(["--workers", "0"])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args_bad)
    assert exc_info.value.error_code == "CLI-VERIFY-EXIT-ENGINE-001"


def test_build_parser_manager_defaults_to_none() -> None:
    """Parser defaults manager to None (discovers all managers)."""
    parser = build_parser()
    args = parser.parse_args([])
    assert args.manager is None


def test_build_options_rejects_blank_manager() -> None:
    """build_options rejects an explicitly blank manager string."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "   "])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-VERIFY-EXIT-ENGINE-006"


# ---------------------------------------------------------------------------
# discover_work and basic verification pass
# ---------------------------------------------------------------------------


def test_discover_work_and_verify_pass(tmp_path: Path) -> None:
    """Discovery finds partitions and verification passes for clean frames."""
    repository = ExitRepository(StorageLayout(tmp_path), ParquetStore())
    repository.save(
        _canonical_frame(),
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
            verifier=ExitEngineVerifier(),
            options=options,
            work=work,
        )
    )
    assert summary.repository_passed is True
    assert summary.successful_tasks == 1
    assert summary.rows_checked == 1


# ---------------------------------------------------------------------------
# Empty work
# ---------------------------------------------------------------------------


def test_run_verification_empty_work(tmp_path: Path) -> None:
    """Empty work produces a passing zeroed verification summary."""
    summary = _run(
        run_verification(
            repository=ExitRepository(StorageLayout(tmp_path), ParquetStore()),
            verifier=ExitEngineVerifier(),
            options=_options(storage_root=tmp_path),
            work=(),
        )
    )
    assert summary.repository_passed is True
    assert summary.successful_tasks == 0
    assert summary.failed_tasks == 0
    assert summary.rows_checked == 0


# ---------------------------------------------------------------------------
# format_summary and format_partition_failure
# ---------------------------------------------------------------------------


def test_format_summary_shows_pass_status() -> None:
    """Summary renders PASS status when repository_passed is True."""
    summary = VerifyExitEngineSummary(
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


def test_format_summary_shows_fail_status() -> None:
    """Summary renders FAIL status when repository_passed is False."""
    summary = VerifyExitEngineSummary(
        symbols_verified=1,
        datasets_verified=1,
        timeframes_verified=1,
        successful_tasks=0,
        failed_tasks=1,
        rows_checked=0,
        duplicate_timestamps=0,
        null_rows=0,
        nan_rows=0,
        invalid_timestamps=0,
        invalid_status_rows=0,
        warnings=0,
        duration_seconds=0.1,
        repository_passed=False,
    )
    text = format_summary(summary)
    assert "FAIL" in text


def test_format_partition_failure_renders_expected_fields() -> None:
    """format_partition_failure renders all diagnostic fields."""
    failure = format_partition_failure(
        dataset="Exit Engine",
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        partition="2026.parquet",
        verifier="ExitEngineVerifier",
        exception_type="ExitEngineValidationError",
        message="missing required columns",
        code="EXIT-VERIFICATION-001",
    )
    assert "FAILED" in failure
    assert _SYMBOL in failure
    assert "ExitEngineVerifier" in failure
    assert "EXIT-VERIFICATION-001" in failure
    assert "Exit Engine" in failure
    assert "2026.parquet" in failure
    assert "missing required columns" in failure


def test_format_partition_failure_without_code() -> None:
    """format_partition_failure omits Code line when code is None."""
    failure = format_partition_failure(
        dataset="Exit Engine",
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        partition="2026.parquet",
        verifier="ExitEngineVerifier",
        exception_type="RuntimeError",
        message="unexpected error",
        code=None,
    )
    assert "Code:" not in failure
    assert "FAILED" in failure


# ---------------------------------------------------------------------------
# Verification failure on invalid partition
# ---------------------------------------------------------------------------


def test_run_verification_fail_on_invalid_frame(tmp_path: Path) -> None:
    """Verification fails when a partition raises a schema validation error."""
    repository = ExitRepository(StorageLayout(tmp_path), ParquetStore())
    invalid = _canonical_frame().drop("exit_action")
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
            verifier=ExitEngineVerifier(),
            options=options,
            work=work,
        )
    )
    assert summary.repository_passed is False
    assert summary.failed_tasks == 1
    assert summary.successful_tasks == 0


# ---------------------------------------------------------------------------
# Multiple partitions aggregation
# ---------------------------------------------------------------------------


def test_run_verification_multiple_partitions(tmp_path: Path) -> None:
    """Verification aggregates rows across multiple symbols and years."""
    repository = ExitRepository(StorageLayout(tmp_path), ParquetStore())
    for symbol, year in (("BTCUSDT", 2025), ("ETHUSDT", 2026)):
        repository.save(
            _canonical_frame(symbol=symbol),
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
            verifier=ExitEngineVerifier(),
            options=options,
            work=work,
        )
    )
    assert summary.repository_passed is True
    assert summary.successful_tasks == 2
    assert summary.symbols_verified == 2
    assert summary.rows_checked == 2


# ---------------------------------------------------------------------------
# Year filter via options
# ---------------------------------------------------------------------------


def test_discover_work_filters_by_year(tmp_path: Path) -> None:
    """discover_work respects year allowlist from options."""
    repository = ExitRepository(StorageLayout(tmp_path), ParquetStore())
    for year in (2024, 2025, 2026):
        repository.save(
            _canonical_frame(),
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            year=year,
        )
    options = _options(storage_root=tmp_path, years=(2025, 2026))
    work = discover_work(repository, options)
    assert len(work) == 1
    assert work[0].years == (2025, 2026)


# ---------------------------------------------------------------------------
# Verify all valid enum combinations pass
# ---------------------------------------------------------------------------


def test_run_verification_passes_for_all_exit_actions(tmp_path: Path) -> None:
    """All valid exit_action values pass verification."""
    repository = ExitRepository(StorageLayout(tmp_path), ParquetStore())
    verifier = ExitEngineVerifier()

    test_cases = [
        (ExitAction.HOLD.value, ExitReason.NONE.value, 0.0, 0.0, 0),
        (ExitAction.PARTIAL_EXIT.value, ExitReason.TAKE_PROFIT.value, 0.5, 0.5, 5),
        (ExitAction.FULL_EXIT.value, ExitReason.PORTFOLIO_SHUTDOWN.value, 1.0, 1.0, 1),
    ]

    for index, (action, reason, pct, qty, priority) in enumerate(test_cases):
        position_id = f"pos-{index:08d}"
        frame = _canonical_frame(
            position_id=position_id,
            exit_action=action,
            exit_reason=reason,
            recommended_percent=pct,
            recommended_quantity=qty,
            priority=priority,
        )
        year = 2020 + index
        repository.save(
            frame,
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            year=year,
        )

    options = _options(storage_root=tmp_path)
    work = discover_work(repository, options)
    summary = _run(
        run_verification(
            repository=repository,
            verifier=verifier,
            options=options,
            work=work,
        )
    )
    assert summary.repository_passed is True
    assert summary.successful_tasks == len(test_cases)
