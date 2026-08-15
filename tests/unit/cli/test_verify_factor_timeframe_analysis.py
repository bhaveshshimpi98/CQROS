"""Unit tests for CQROS factor timeframe analysis verification CLI."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from pathlib import Path

import polars as pl
import pytest

from cqros.cli.verify_factor_timeframe_analysis import (
    DiscoveredWorkItem,
    VerifyFactorTimeframeAnalysisOptions,
    VerifyFactorTimeframeAnalysisSummary,
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
    DEFAULT_STORAGE_ROOT,
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
)
from cqros.core.exceptions import ValidationError
from cqros.factor_selection import FactorSelectionRepository
from cqros.factor_selection.schema import FACTOR_SELECTION_SCHEMA, FactorSelectionStatus
from cqros.factor_timeframe_analysis import (
    FactorTimeframeAnalysisRepository,
    FactorTimeframeAnalysisVerifier,
    SimpleFactorTimeframeAnalysisEngine,
)
from cqros.storage import ParquetStore, StorageLayout

_MANAGER = "simple"
_YEAR = 2026
_SELECTION_TIME = 1_700_000_000_000


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def _options(
    *,
    storage_root: Path,
    manager: str | None = _MANAGER,
    years: tuple[int, ...] | None = None,
    workers: int = ResearchConfig().worker_count,
    verbose: bool = False,
    debug: bool = False,
) -> VerifyFactorTimeframeAnalysisOptions:
    """Build VerifyFactorTimeframeAnalysisOptions against a temporary storage root."""
    return VerifyFactorTimeframeAnalysisOptions(
        storage_root=storage_root,
        manager=manager,
        years=years,
        workers=workers,
        verbose=verbose,
        debug=debug,
    )


def _factor_selection_frame(
    *,
    factor_name: str = "momentum",
    timeframe: str = "1h",
    selected: bool = True,
) -> pl.DataFrame:
    """Return a minimal canonical Factor Selection frame."""
    return pl.DataFrame(
        {
            "factor_name": [factor_name],
            "factor_version": ["1.0.0"],
            "timeframe": [timeframe],
            "selection_time": [_SELECTION_TIME],
            "factor_category": ["price"],
            "selected": [selected],
            "selection_score": [0.5],
            "selection_rank": [1],
            "selection_reason": ["top_n"],
            "selection_ic": [0.08],
            "selected_direction": [1],
            "orientation_policy": ["signed_ic_v1"],
            "status": [FactorSelectionStatus.SELECTED.value],
        },
        schema=FACTOR_SELECTION_SCHEMA,
    )


def _seed_fs_and_fta(
    *,
    layout: StorageLayout,
    datastore: ParquetStore,
    factor_name: str = "momentum",
    timeframe: str = "1h",
    year: int = _YEAR,
) -> None:
    """Seed one Factor Selection partition and derive FTA from it."""
    fs_frame = _factor_selection_frame(factor_name=factor_name, timeframe=timeframe)
    FactorSelectionRepository(layout, datastore).save(
        fs_frame,
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe=timeframe,
        year=year,
    )
    fta_engine = SimpleFactorTimeframeAnalysisEngine(source_selection_version=str(year))
    fta_frame = fta_engine.build(fs_frame)
    FactorTimeframeAnalysisRepository(layout, datastore).save(
        fta_frame,
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        year=year,
    )


# ---------------------------------------------------------------------------
# build_parser
# ---------------------------------------------------------------------------


class TestBuildParser:
    def test_no_required_arguments(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])
        assert args.manager is None

    def test_manager_accepted(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--manager", "simple"])
        assert args.manager == "simple"

    def test_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])
        assert args.years is None
        assert args.verbose is False
        assert args.debug is False
        assert args.storage_root is None

    def test_years_multiple(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--years", "2024", "2025"])
        assert args.years == ["2024", "2025"]


# ---------------------------------------------------------------------------
# build_options
# ---------------------------------------------------------------------------


class TestBuildOptions:
    def _parse(self, *argv: str) -> VerifyFactorTimeframeAnalysisOptions:
        parser = build_parser()
        args = parser.parse_args(list(argv))
        return build_options(args)

    def test_valid_minimal(self) -> None:
        opts = self._parse()
        assert opts.manager is None
        assert opts.workers > 0

    def test_invalid_workers(self) -> None:
        with pytest.raises(ValidationError):
            self._parse("--workers", "-1")

    def test_empty_manager_rejected(self) -> None:
        with pytest.raises(ValidationError):
            parser = build_parser()
            args = parser.parse_args(["--manager", "  "])
            build_options(args)

    def test_invalid_year_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._parse("--years", "bad")

    def test_storage_root_default(self) -> None:
        opts = self._parse()
        assert opts.storage_root == Path(DEFAULT_STORAGE_ROOT)

    def test_years_sorted(self) -> None:
        opts = self._parse("--years", "2026", "2024")
        assert opts.years == (2024, 2026)


# ---------------------------------------------------------------------------
# discover_work
# ---------------------------------------------------------------------------


class TestDiscoverWork:
    def test_empty_when_no_partitions(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        fta_repo = FactorTimeframeAnalysisRepository(layout, ParquetStore())
        opts = _options(storage_root=tmp_path)
        work = discover_work(fta_repo, opts)
        assert work == ()

    def test_discovers_seeded_partition(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        _seed_fs_and_fta(layout=layout, datastore=datastore)
        fta_repo = FactorTimeframeAnalysisRepository(layout, datastore)
        opts = _options(storage_root=tmp_path)
        work = discover_work(fta_repo, opts)
        assert len(work) == 1
        assert work[0].manager == _MANAGER
        assert work[0].year == _YEAR

    def test_year_filter_excludes(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        _seed_fs_and_fta(layout=layout, datastore=datastore, year=2024)
        _seed_fs_and_fta(layout=layout, datastore=datastore, year=2025)
        fta_repo = FactorTimeframeAnalysisRepository(layout, datastore)
        opts = _options(storage_root=tmp_path, years=(2025,))
        work = discover_work(fta_repo, opts)
        assert len(work) == 1
        assert work[0].year == 2025

    def test_none_manager_discovers_all(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        _seed_fs_and_fta(layout=layout, datastore=datastore)
        fta_repo = FactorTimeframeAnalysisRepository(layout, datastore)
        opts = _options(storage_root=tmp_path, manager=None)
        work = discover_work(fta_repo, opts)
        assert len(work) >= 1


# ---------------------------------------------------------------------------
# run_verification
# ---------------------------------------------------------------------------


class TestRunVerification:
    def test_empty_work_returns_pass(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        opts = _options(storage_root=tmp_path)
        summary = _run(
            run_verification(
                fta_repository=FactorTimeframeAnalysisRepository(layout, datastore),
                fs_repository=FactorSelectionRepository(layout, datastore),
                verifier=FactorTimeframeAnalysisVerifier(),
                options=opts,
                work=(),
            )
        )
        assert summary.repository_passed is True
        assert summary.successful_tasks == 0

    def test_succeeds_with_valid_fta(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        _seed_fs_and_fta(layout=layout, datastore=datastore)
        fta_repo = FactorTimeframeAnalysisRepository(layout, datastore)
        fs_repo = FactorSelectionRepository(layout, datastore)
        opts = _options(storage_root=tmp_path)
        work = discover_work(fta_repo, opts)
        summary = _run(
            run_verification(
                fta_repository=fta_repo,
                fs_repository=fs_repo,
                verifier=FactorTimeframeAnalysisVerifier(),
                options=opts,
                work=work,
            )
        )
        assert summary.repository_passed is True
        assert summary.successful_tasks == 1
        assert summary.failed_tasks == 0
        assert summary.rows_checked > 0

    def test_fails_when_fta_partition_missing(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        opts = _options(storage_root=tmp_path)
        work = (DiscoveredWorkItem(manager=_MANAGER, year=_YEAR),)
        summary = _run(
            run_verification(
                fta_repository=FactorTimeframeAnalysisRepository(layout, datastore),
                fs_repository=FactorSelectionRepository(layout, datastore),
                verifier=FactorTimeframeAnalysisVerifier(),
                options=opts,
                work=work,
            )
        )
        assert summary.failed_tasks == 1
        assert summary.repository_passed is False


# ---------------------------------------------------------------------------
# format_partition_failure / format_summary
# ---------------------------------------------------------------------------


class TestFormatters:
    def test_format_partition_failure_contains_fields(self) -> None:
        text = format_partition_failure(
            dataset="Factor Timeframe Analysis",
            manager="mgr",
            year=2026,
            partition="2026.parquet",
            verifier="FactorTimeframeAnalysisVerifier",
            exception_type="ValueError",
            message="something went wrong",
            code="FTA-001",
        )
        assert "mgr" in text
        assert "2026" in text
        assert "ValueError" in text
        assert "FTA-001" in text

    def test_format_summary_pass(self, tmp_path: Path) -> None:
        summary = VerifyFactorTimeframeAnalysisSummary(
            panels_verified=1,
            datasets_verified=1,
            managers_verified=1,
            successful_tasks=1,
            failed_tasks=0,
            rows_checked=5,
            duplicate_timestamps=0,
            null_rows=0,
            nan_rows=0,
            invalid_timestamps=0,
            invalid_numeric_rows=0,
            warnings=0,
            duration_seconds=0.1,
            repository_passed=True,
        )
        text = format_summary(summary)
        assert "PASS" in text

    def test_format_summary_fail(self, tmp_path: Path) -> None:
        summary = VerifyFactorTimeframeAnalysisSummary(
            panels_verified=1,
            datasets_verified=1,
            managers_verified=1,
            successful_tasks=0,
            failed_tasks=1,
            rows_checked=0,
            duplicate_timestamps=0,
            null_rows=0,
            nan_rows=0,
            invalid_timestamps=0,
            invalid_numeric_rows=0,
            warnings=0,
            duration_seconds=0.2,
            repository_passed=False,
        )
        text = format_summary(summary)
        assert "FAIL" in text


# ---------------------------------------------------------------------------
# main (smoke test)
# ---------------------------------------------------------------------------


class TestMain:
    def test_exits_0_when_no_partitions(self, tmp_path: Path) -> None:
        exit_code = _run(main(["--storage-root", str(tmp_path)]))
        assert exit_code == 0

    def test_exits_0_with_valid_fta(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        _seed_fs_and_fta(layout=layout, datastore=datastore)
        exit_code = _run(main(["--manager", _MANAGER, "--storage-root", str(tmp_path)]))
        assert exit_code == 0

    def test_exits_1_on_bad_options(self) -> None:
        exit_code = _run(main(["--workers", "0"]))
        assert exit_code == 1
