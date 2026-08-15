"""Unit tests for CQROS pyramiding-verification CLI."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

import cqros.cli.verify_pyramiding as verify_pyramiding_module
from cqros.cli.verify_pyramiding import (
    DiscoveredWorkItem,
    VerifyPyramidingOptions,
    VerifyPyramidingSummary,
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
from cqros.pyramiding import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    PyramidingReason,
    PyramidingRepository,
    PyramidingVerifier,
)
from cqros.storage import ParquetStore, StorageLayout

_MANAGER = "simple"
_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1h"
_YEAR = 2026
_OPEN_TIME = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)


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
) -> VerifyPyramidingOptions:
    """Build VerifyPyramidingOptions against a temporary storage root."""
    return VerifyPyramidingOptions(
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


def _pyramiding_frame(*, symbol: str = _SYMBOL) -> pl.DataFrame:
    """Return a canonical passing pyramiding frame for CLI verification tests."""
    return pl.DataFrame(
        {
            "manager": [_MANAGER],
            "symbol": [symbol],
            "timeframe": [_TIMEFRAME],
            "open_time": [_OPEN_TIME],
            "position_id": ["pos-00000001"],
            "trade_id": ["pos-00000001"],
            "entry_price": [100.0],
            "current_price": [102.0],
            "highest_price": [102.0],
            "position_size": [1.0],
            "add_number": [0],
            "max_adds": [3],
            "additional_size": [0.0],
            "recommended_size": [1.0],
            "profit_pct": [0.02],
            "allow_pyramid": [False],
            "reason": [PyramidingReason.INSUFFICIENT_PROFIT.value],
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
    with patch.object(verify_pyramiding_module, "DEFAULT_STORAGE_ROOT", tmp_path):
        options = build_options(args)
    assert options.manager == "simple"
    assert options.workers == 2

    args = parser.parse_args(["--workers", "0"])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-VERIFY-PYRAMIDING-001"


def test_build_parser_manager_defaults_to_none() -> None:
    """Parser defaults manager to None (discovers all managers)."""
    parser = build_parser()
    args = parser.parse_args([])
    assert args.manager is None


def test_build_options_rejects_blank_manager(tmp_path: Path) -> None:
    """build_options rejects an explicitly blank manager string."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "   "])
    with patch.object(verify_pyramiding_module, "DEFAULT_STORAGE_ROOT", tmp_path):
        with pytest.raises(ValidationError) as exc_info:
            build_options(args)
    assert exc_info.value.error_code == "CLI-VERIFY-PYRAMIDING-006"


# ---------------------------------------------------------------------------
# discover_work and basic verification pass
# ---------------------------------------------------------------------------


def test_discover_work_and_verify_pass(tmp_path: Path) -> None:
    """Discovery finds partitions and verification passes for clean frames."""
    repository = PyramidingRepository(StorageLayout(tmp_path), ParquetStore())
    repository.save(
        _pyramiding_frame(),
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
            verifier=PyramidingVerifier(),
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
            repository=PyramidingRepository(StorageLayout(tmp_path), ParquetStore()),
            verifier=PyramidingVerifier(),
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


def test_format_summary_and_partition_failure() -> None:
    """Summary and failure formatters render expected labels."""
    summary = VerifyPyramidingSummary(
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
        dataset="Pyramiding",
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        partition="2026.parquet",
        verifier="PyramidingVerifier",
        exception_type="PyramidingValidationError",
        message="missing required columns",
        code="PYR-VERIFICATION-001",
    )
    assert "FAILED" in failure
    assert _SYMBOL in failure
    assert "PyramidingVerifier" in failure
    assert "PYR-VERIFICATION-001" in failure
    assert "Pyramiding" in failure


def test_format_summary_shows_fail_status() -> None:
    """Summary renders FAIL status when repository_passed is False."""
    summary = VerifyPyramidingSummary(
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


# ---------------------------------------------------------------------------
# Verification failure on invalid partition
# ---------------------------------------------------------------------------


def test_run_verification_fail_on_invalid_frame(tmp_path: Path) -> None:
    """Verification fails when a partition raises a schema validation error."""
    repository = PyramidingRepository(StorageLayout(tmp_path), ParquetStore())
    invalid = _pyramiding_frame().drop("allow_pyramid")
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
            verifier=PyramidingVerifier(),
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
    repository = PyramidingRepository(StorageLayout(tmp_path), ParquetStore())
    for symbol, year in (("BTCUSDT", 2025), ("ETHUSDT", 2026)):
        repository.save(
            _pyramiding_frame(symbol=symbol),
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
            verifier=PyramidingVerifier(),
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
    repository = PyramidingRepository(StorageLayout(tmp_path), ParquetStore())
    for year in (2024, 2025, 2026):
        repository.save(
            _pyramiding_frame(),
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
