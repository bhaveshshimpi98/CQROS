"""Unit tests for CQROS analytics verification CLI."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

import cqros.cli.verify_analytics as verify_analytics_module
from cqros.analytics import AnalyticsRepository, AnalyticsStatus, AnalyticsVerifier
from cqros.analytics.schema import CANONICAL_COLUMN_ORDER, COLUMN_DTYPES
from cqros.cli.verify_analytics import (
    DiscoveredWorkItem,
    VerifyAnalyticsOptions,
    VerifyAnalyticsSummary,
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
)
from cqros.core.exceptions import ValidationError
from cqros.storage import ParquetStore, StorageLayout

_MANAGER = "simple"
_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1h"
_YEAR = 2026
_OPEN_TIME_MS = int(datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC).timestamp() * 1000.0)


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def _options(
    *,
    storage_root: Path,
    manager: str | None = _MANAGER,
    symbols: tuple[str, ...] | None = None,
    timeframes: tuple[str, ...] | None = None,
    years: tuple[int, ...] | None = None,
    workers: int = ResearchConfig().worker_count,
    verbose: bool = False,
    debug: bool = False,
) -> VerifyAnalyticsOptions:
    """Build VerifyAnalyticsOptions against a temporary storage root."""
    return VerifyAnalyticsOptions(
        storage_root=storage_root,
        manager=manager,
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
    status: str = AnalyticsStatus.FINISHED.value,
    rolling_return: float = 0.15,
    rolling_max_drawdown: float = 0.02,
) -> pl.DataFrame:
    """Return a canonical passing analytics frame for CLI verification tests."""
    return pl.DataFrame(
        {
            "symbol": [symbol],
            "timeframe": [_TIMEFRAME],
            "open_time": [_OPEN_TIME_MS],
            "manager": [_MANAGER],
            "rolling_return": [rolling_return],
            "rolling_volatility": [0.0],
            "rolling_sharpe": [None],
            "rolling_sortino": [None],
            "rolling_max_drawdown": [rolling_max_drawdown],
            "rolling_win_rate": [0.0],
            "rolling_profit_factor": [None],
            "rolling_expectancy": [0.0],
            "rolling_cagr": [0.0],
            "rolling_calmar": [None],
            "rolling_recovery_factor": [None],
            "benchmark_return": [0.0],
            "benchmark_alpha": [0.0],
            "benchmark_beta": [0.0],
            "benchmark_correlation": [0.0],
            "benchmark_tracking_error": [0.0],
            "benchmark_information_ratio": [0.0],
            "status": [status],
        },
        schema=dict(COLUMN_DTYPES),
    ).select(list(CANONICAL_COLUMN_ORDER))


# ---------------------------------------------------------------------------
# Parser and options
# ---------------------------------------------------------------------------


def test_build_parser_and_options(tmp_path: Path) -> None:
    """Parser accepts filter flags; options reject invalid workers."""
    parser = build_parser()
    args = parser.parse_args(
        ["--manager", "simple", "--workers", "2", "--storage-root", str(tmp_path)]
    )
    options = build_options(args)
    assert options.manager == "simple"
    assert options.workers == 2
    assert options.storage_root == tmp_path

    args_bad = parser.parse_args(["--workers", "0"])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args_bad)
    assert exc_info.value.error_code == "CLI-VERIFY-ANALYTICS-001"


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
    assert exc_info.value.error_code == "CLI-VERIFY-ANALYTICS-004"


# ---------------------------------------------------------------------------
# discover_work and basic verification pass
# ---------------------------------------------------------------------------


def test_discover_work_and_verify_pass(tmp_path: Path) -> None:
    """Discovery finds partitions and verification passes for clean frames."""
    repository = AnalyticsRepository(StorageLayout(tmp_path), ParquetStore())
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
            verifier=AnalyticsVerifier(),
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
            repository=AnalyticsRepository(StorageLayout(tmp_path), ParquetStore()),
            verifier=AnalyticsVerifier(),
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
    summary = VerifyAnalyticsSummary(
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
    summary = VerifyAnalyticsSummary(
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
        dataset="Analytics",
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        partition="2026.parquet",
        verifier="AnalyticsVerifier",
        exception_type="AnalyticsValidationError",
        message="missing required columns",
        code="ANA-VERIFICATION-001",
    )
    assert "FAILED" in failure
    assert _SYMBOL in failure
    assert "AnalyticsVerifier" in failure
    assert "ANA-VERIFICATION-001" in failure
    assert "Analytics" in failure
    assert "2026.parquet" in failure
    assert "missing required columns" in failure


def test_format_partition_failure_without_code() -> None:
    """format_partition_failure omits Code line when code is None."""
    failure = format_partition_failure(
        dataset="Analytics",
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        partition="2026.parquet",
        verifier="AnalyticsVerifier",
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
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    # Bypass repository schema validation so load/verify can surface the failure.
    path = layout.analytics_path(
        _MANAGER,
        EXCHANGE_BINANCE,
        MARKET_USDT_PERPETUAL,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    datastore.write(path, pl.DataFrame({"symbol": [_SYMBOL]}))

    repository = AnalyticsRepository(layout, datastore)
    options = _options(storage_root=tmp_path)
    work = discover_work(repository, options)
    summary = _run(
        run_verification(
            repository=repository,
            verifier=AnalyticsVerifier(),
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
    repository = AnalyticsRepository(StorageLayout(tmp_path), ParquetStore())
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
            verifier=AnalyticsVerifier(),
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
    repository = AnalyticsRepository(StorageLayout(tmp_path), ParquetStore())
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
# main
# ---------------------------------------------------------------------------


def test_main_returns_success_for_empty_universe(tmp_path: Path) -> None:
    """main returns 0 when no analytics partitions are discovered."""
    with patch.object(verify_analytics_module, "DEFAULT_STORAGE_ROOT", tmp_path):
        code = _run(main(["--storage-root", str(tmp_path)]))
    assert code == 0


def test_main_exit_code_pass(tmp_path: Path) -> None:
    """main returns 0 when verification passes."""
    repository = AnalyticsRepository(StorageLayout(tmp_path), ParquetStore())
    repository.save(
        _canonical_frame(),
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    with patch.object(verify_analytics_module, "DEFAULT_STORAGE_ROOT", tmp_path):
        code = _run(main(["--storage-root", str(tmp_path)]))
    assert code == 0


def test_main_exit_code_fail(tmp_path: Path) -> None:
    """main returns 1 when verification fails."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    path = layout.analytics_path(
        _MANAGER,
        EXCHANGE_BINANCE,
        MARKET_USDT_PERPETUAL,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    datastore.write(path, pl.DataFrame({"symbol": [_SYMBOL]}))
    with patch.object(verify_analytics_module, "DEFAULT_STORAGE_ROOT", tmp_path):
        code = _run(main(["--storage-root", str(tmp_path)]))
    assert code == 1


def test_main_validation_error_exit_code() -> None:
    """main returns 1 when build_options raises ValidationError."""
    code = _run(main(["--workers", "0"]))
    assert code == 1
