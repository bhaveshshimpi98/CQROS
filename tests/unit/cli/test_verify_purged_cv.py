"""Unit tests for CQROS purged-CV verification CLI."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

import cqros.cli.verify_purged_cv as verify_purged_cv_module
from cqros.cli.verify_purged_cv import (
    DiscoveredWorkItem,
    VerifyPurgedCVOptions,
    VerifyPurgedCVSummary,
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
from cqros.purged_cv import PurgedCVRepository, PurgedCVStatus, PurgedCVVerifier
from cqros.purged_cv.engine import SimplePurgedCVEngine
from cqros.purged_cv.schema import CANONICAL_COLUMN_ORDER, COLUMN_DTYPES
from cqros.storage import ParquetStore, StorageLayout

_MANAGER = "simple"
_TIMEFRAME = "1h"
_YEAR = 2026
_TRAIN_START = 1_704_067_200_000
_TRAIN_END = 1_704_070_800_000
_TEST_START = 1_704_074_400_000
_TEST_END = 1_704_078_000_000


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def _options(
    *,
    storage_root: Path,
    manager: str | None = _MANAGER,
    timeframes: tuple[str, ...] | None = None,
    years: tuple[int, ...] | None = None,
    workers: int = ResearchConfig().worker_count,
    verbose: bool = False,
    debug: bool = False,
) -> VerifyPurgedCVOptions:
    """Build VerifyPurgedCVOptions against a temporary storage root."""
    return VerifyPurgedCVOptions(
        storage_root=storage_root,
        manager=manager,
        timeframes=timeframes,
        years=years,
        workers=workers,
        verbose=verbose,
        debug=debug,
    )


def _canonical_frame(
    *,
    timeframe: str = _TIMEFRAME,
    fold_id: int = 1,
    status: str = PurgedCVStatus.PASS.value,
    train_score: float = 0.10,
) -> pl.DataFrame:
    """Return a canonical passing purged-CV frame for CLI verification tests."""
    return pl.DataFrame(
        {
            "strategy_name": ["default_strategy"],
            "strategy_version": ["v1"],
            "timeframe": [timeframe],
            "fold_id": [fold_id],
            "train_start_time": [_TRAIN_START],
            "train_end_time": [_TRAIN_END],
            "test_start_time": [_TEST_START],
            "test_end_time": [_TEST_END],
            "purge_size": [5],
            "embargo_size": [5],
            "train_rows": [2],
            "test_rows": [2],
            "train_score": [train_score],
            "test_score": [0.05],
            "overfit_gap": [0.05],
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
    assert exc_info.value.error_code == "CLI-VERIFY-PURGED-CV-001"


def test_build_parser_manager_defaults_to_none() -> None:
    """Parser defaults manager to None (discovers all managers)."""
    parser = build_parser()
    args = parser.parse_args([])
    assert args.manager is None


def test_build_parser_has_no_symbols_flag() -> None:
    """Panel-based purged-CV verification does not expose --symbols."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--symbols", "BTCUSDT"])


def test_build_options_rejects_blank_manager() -> None:
    """build_options rejects an explicitly blank manager string."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "   "])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-VERIFY-PURGED-CV-004"


# ---------------------------------------------------------------------------
# discover_work and basic verification pass
# ---------------------------------------------------------------------------


def test_discover_work_and_verify_pass(tmp_path: Path) -> None:
    """Discovery finds panels and verification passes for clean frames."""
    repository = PurgedCVRepository(StorageLayout(tmp_path), ParquetStore())
    repository.save(
        _canonical_frame(),
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    options = _options(storage_root=tmp_path)
    work = discover_work(repository, options)
    assert work == (
        DiscoveredWorkItem(
            manager=_MANAGER,
            timeframe=_TIMEFRAME,
            years=(_YEAR,),
        ),
    )
    summary = _run(
        run_verification(
            repository=repository,
            verifier=PurgedCVVerifier(),
            options=options,
            work=work,
        )
    )
    assert summary.repository_passed is True
    assert summary.successful_tasks == 1
    assert summary.panels_verified == 1
    assert summary.rows_checked == 1


# ---------------------------------------------------------------------------
# Empty work
# ---------------------------------------------------------------------------


def test_run_verification_empty_work(tmp_path: Path) -> None:
    """Empty work produces a passing zeroed verification summary."""
    summary = _run(
        run_verification(
            repository=PurgedCVRepository(StorageLayout(tmp_path), ParquetStore()),
            verifier=PurgedCVVerifier(),
            options=_options(storage_root=tmp_path),
            work=(),
        )
    )
    assert summary.repository_passed is True
    assert summary.successful_tasks == 0
    assert summary.failed_tasks == 0
    assert summary.panels_verified == 0
    assert summary.rows_checked == 0


# ---------------------------------------------------------------------------
# format_summary and format_partition_failure
# ---------------------------------------------------------------------------


def test_format_summary_shows_pass_status() -> None:
    """Summary renders PASS status when repository_passed is True."""
    summary = VerifyPurgedCVSummary(
        panels_verified=1,
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
    assert "Panels verified: 1" in text
    assert "Successful tasks: 1" in text


def test_format_summary_shows_fail_status() -> None:
    """Summary renders FAIL status when repository_passed is False."""
    summary = VerifyPurgedCVSummary(
        panels_verified=1,
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
        dataset="Purged-CV",
        timeframe=_TIMEFRAME,
        year=_YEAR,
        partition="2026.parquet",
        verifier="PurgedCVVerifier",
        exception_type="PurgedCVError",
        message="missing required columns",
        code="PCV-VERIFICATION-001",
    )
    assert "FAILED" in failure
    assert f"Timeframe: {_TIMEFRAME}" in failure
    assert f"Year: {_YEAR}" in failure
    assert "Symbol:" not in failure
    assert "PurgedCVVerifier" in failure
    assert "PCV-VERIFICATION-001" in failure
    assert "Purged-CV" in failure
    assert "2026.parquet" in failure
    assert "missing required columns" in failure


def test_format_partition_failure_without_code() -> None:
    """format_partition_failure omits Code line when code is None."""
    failure = format_partition_failure(
        dataset="Purged-CV",
        timeframe=_TIMEFRAME,
        year=_YEAR,
        partition="2026.parquet",
        verifier="PurgedCVVerifier",
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
    path = layout.purged_cv_path(
        _MANAGER,
        EXCHANGE_BINANCE,
        MARKET_USDT_PERPETUAL,
        _TIMEFRAME,
        _YEAR,
    )
    datastore.write(path, pl.DataFrame({"strategy_name": ["default_strategy"]}))

    repository = PurgedCVRepository(layout, datastore)
    options = _options(storage_root=tmp_path)
    work = discover_work(repository, options)
    summary = _run(
        run_verification(
            repository=repository,
            verifier=PurgedCVVerifier(),
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
    """Verification aggregates rows across multiple timeframes and years."""
    repository = PurgedCVRepository(StorageLayout(tmp_path), ParquetStore())
    for timeframe, year in (("1h", 2025), ("4h", 2026)):
        repository.save(
            _canonical_frame(timeframe=timeframe),
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            timeframe=timeframe,
            year=year,
        )
    options = _options(storage_root=tmp_path)
    work = discover_work(repository, options)
    summary = _run(
        run_verification(
            repository=repository,
            verifier=PurgedCVVerifier(),
            options=options,
            work=work,
        )
    )
    assert summary.repository_passed is True
    assert summary.successful_tasks == 2
    assert summary.panels_verified == 2
    assert summary.datasets_verified == 2
    assert summary.timeframes_verified == 2
    assert summary.rows_checked == 2
    assert summary.warnings == 0


def test_run_verification_five_timeframe_engine_panels_pass(tmp_path: Path) -> None:
    """Five engine-generated timeframe panels verify with zero warnings.

    Reproduces the production failure mode where purged-CV training extents
    wrap the test block and ``train_start_time`` is non-monotonic across
    ``fold_id`` order — previously emitting two false-positive warnings per
    panel (10 total) and FAIL repository status.
    """
    repository = PurgedCVRepository(StorageLayout(tmp_path), ParquetStore())
    timeframes = ("5m", "15m", "1h", "4h", "1d")
    engine = SimplePurgedCVEngine(n_folds=5, purge_size=2, embargo_size=1)
    for timeframe in timeframes:
        times = [_TRAIN_START + index * 3_600_000 for index in range(20)]
        walk_forward = pl.DataFrame(
            {
                "strategy_name": ["default_strategy"] * 20,
                "strategy_version": ["v1"] * 20,
                "timeframe": [timeframe] * 20,
                "test_start": times,
                "train_score": [float(index) for index in range(20)],
                "test_score": [float(index) for index in range(20)],
            }
        )
        frame = engine.build(walk_forward)
        repository.save(
            frame,
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            timeframe=timeframe,
            year=_YEAR,
        )

    options = _options(storage_root=tmp_path, workers=1)
    work = discover_work(repository, options)
    assert len(work) == 5
    assert {item.timeframe for item in work} == set(timeframes)

    # Filesystem discovery order must not affect verification outcome.
    reversed_work = tuple(reversed(work))
    summary_a = _run(
        run_verification(
            repository=repository,
            verifier=PurgedCVVerifier(),
            options=options,
            work=work,
        )
    )
    summary_b = _run(
        run_verification(
            repository=repository,
            verifier=PurgedCVVerifier(),
            options=options,
            work=reversed_work,
        )
    )

    for summary in (summary_a, summary_b):
        assert summary.panels_verified == 5
        assert summary.datasets_verified == 5
        assert summary.timeframes_verified == 5
        assert summary.successful_tasks == 5
        assert summary.failed_tasks == 0
        assert summary.rows_checked == 25
        assert summary.duplicate_timestamps == 0
        assert summary.null_rows == 0
        assert summary.nan_rows == 0
        assert summary.invalid_timestamps == 0
        assert summary.invalid_status_rows == 0
        assert summary.warnings == 0
        assert summary.repository_passed is True

    assert summary_a.warnings == summary_b.warnings
    assert summary_a.repository_passed == summary_b.repository_passed
    assert summary_a.panels_verified == summary_b.panels_verified
    assert summary_a.datasets_verified == summary_b.datasets_verified
    assert summary_a.rows_checked == summary_b.rows_checked
    assert summary_a.successful_tasks == summary_b.successful_tasks
    assert summary_a.failed_tasks == summary_b.failed_tasks


def test_run_verification_does_not_modify_parquet_artifacts(tmp_path: Path) -> None:
    """Verification is read-only with respect to on-disk parquet bytes."""
    import hashlib

    repository = PurgedCVRepository(StorageLayout(tmp_path), ParquetStore())
    repository.save(
        _canonical_frame(),
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    path = StorageLayout(tmp_path).purged_cv_path(
        _MANAGER,
        EXCHANGE_BINANCE,
        MARKET_USDT_PERPETUAL,
        _TIMEFRAME,
        _YEAR,
    )
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    options = _options(storage_root=tmp_path)
    work = discover_work(repository, options)
    summary = _run(
        run_verification(
            repository=repository,
            verifier=PurgedCVVerifier(),
            options=options,
            work=work,
        )
    )
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    assert summary.repository_passed is True
    assert before == after


def test_run_verification_still_fails_on_structural_corruption(tmp_path: Path) -> None:
    """Duplicate keys and invalid status values still fail repository status."""
    repository = PurgedCVRepository(StorageLayout(tmp_path), ParquetStore())
    good = _canonical_frame(fold_id=1)
    duplicate = pl.concat([good, good])
    repository.save(
        duplicate,
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    options = _options(storage_root=tmp_path)
    work = discover_work(repository, options)
    summary = _run(
        run_verification(
            repository=repository,
            verifier=PurgedCVVerifier(),
            options=options,
            work=work,
        )
    )
    assert summary.repository_passed is False
    assert summary.warnings > 0
    assert summary.duplicate_timestamps > 0


# ---------------------------------------------------------------------------
# Year / timeframe filters via options
# ---------------------------------------------------------------------------


def test_discover_work_filters_by_year(tmp_path: Path) -> None:
    """discover_work respects year allowlist from options."""
    repository = PurgedCVRepository(StorageLayout(tmp_path), ParquetStore())
    for year in (2024, 2025, 2026):
        repository.save(
            _canonical_frame(),
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            timeframe=_TIMEFRAME,
            year=year,
        )
    options = _options(storage_root=tmp_path, years=(2025, 2026))
    work = discover_work(repository, options)
    assert len(work) == 1
    assert work[0].years == (2025, 2026)


def test_discover_work_filters_by_timeframe(tmp_path: Path) -> None:
    """discover_work respects timeframe allowlist from options."""
    repository = PurgedCVRepository(StorageLayout(tmp_path), ParquetStore())
    for timeframe in ("1h", "4h"):
        repository.save(
            _canonical_frame(timeframe=timeframe),
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            timeframe=timeframe,
            year=_YEAR,
        )
    options = _options(storage_root=tmp_path, timeframes=("1h",))
    work = discover_work(repository, options)
    assert len(work) == 1
    assert work[0].timeframe == "1h"


# ---------------------------------------------------------------------------
# Deterministic summary formatting
# ---------------------------------------------------------------------------


def test_format_summary_is_deterministic() -> None:
    """Identical summaries render identical text."""
    summary = VerifyPurgedCVSummary(
        panels_verified=2,
        datasets_verified=2,
        timeframes_verified=1,
        successful_tasks=2,
        failed_tasks=0,
        rows_checked=4,
        duplicate_timestamps=0,
        null_rows=0,
        nan_rows=0,
        invalid_timestamps=0,
        invalid_status_rows=0,
        warnings=0,
        duration_seconds=1.25,
        repository_passed=True,
    )
    assert format_summary(summary) == format_summary(summary)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_returns_success_for_empty_universe(tmp_path: Path) -> None:
    """main returns 0 when no purged-CV partitions are discovered."""
    with patch.object(verify_purged_cv_module, "DEFAULT_STORAGE_ROOT", tmp_path):
        code = _run(main(["--storage-root", str(tmp_path)]))
    assert code == 0


def test_main_exit_code_pass(tmp_path: Path) -> None:
    """main returns 0 when verification passes."""
    repository = PurgedCVRepository(StorageLayout(tmp_path), ParquetStore())
    repository.save(
        _canonical_frame(),
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    with patch.object(verify_purged_cv_module, "DEFAULT_STORAGE_ROOT", tmp_path):
        code = _run(main(["--storage-root", str(tmp_path)]))
    assert code == 0


def test_main_exit_code_fail(tmp_path: Path) -> None:
    """main returns 1 when verification fails."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    path = layout.purged_cv_path(
        _MANAGER,
        EXCHANGE_BINANCE,
        MARKET_USDT_PERPETUAL,
        _TIMEFRAME,
        _YEAR,
    )
    datastore.write(path, pl.DataFrame({"strategy_name": ["default_strategy"]}))
    with patch.object(verify_purged_cv_module, "DEFAULT_STORAGE_ROOT", tmp_path):
        code = _run(main(["--storage-root", str(tmp_path)]))
    assert code == 1


def test_main_validation_error_exit_code() -> None:
    """main returns 1 when build_options raises ValidationError."""
    code = _run(main(["--workers", "0"]))
    assert code == 1
