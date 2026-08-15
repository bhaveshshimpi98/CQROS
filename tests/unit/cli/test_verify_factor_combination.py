"""Unit tests for CQROS factor combination verification CLI."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from pathlib import Path

import polars as pl
import pytest

from cqros.cli.verify_factor_combination import (
    DiscoveredWorkItem,
    VerifyFactorCombinationOptions,
    VerifyFactorCombinationSummary,
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
from cqros.factor_combination import (
    FactorCombinationRepository,
    FactorCombinationVerifier,
    SimpleFactorCombinationEngine,
)
from cqros.factor_selection.schema import FACTOR_SELECTION_SCHEMA, FactorSelectionStatus
from cqros.factor_timeframe_analysis import (
    FactorTimeframeAnalysisRepository,
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
) -> VerifyFactorCombinationOptions:
    """Build VerifyFactorCombinationOptions against a temporary storage root."""
    return VerifyFactorCombinationOptions(
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
    selection_score: float = 0.5,
) -> pl.DataFrame:
    """Return a minimal canonical Factor Selection frame."""
    return pl.DataFrame(
        {
            "factor_name": [factor_name],
            "factor_version": ["1.0.0"],
            "timeframe": [timeframe],
            "selection_time": [_SELECTION_TIME],
            "factor_category": ["price"],
            "selected": [True],
            "selection_score": [selection_score],
            "selection_rank": [1],
            "selection_reason": ["top_n"],
            "selection_ic": [0.08],
            "selected_direction": [1],
            "orientation_policy": ["signed_ic_v1"],
            "status": [FactorSelectionStatus.SELECTED.value],
        },
        schema=FACTOR_SELECTION_SCHEMA,
    )


def _seed_fta_and_combination(
    *,
    layout: StorageLayout,
    datastore: ParquetStore,
    year: int = _YEAR,
) -> None:
    """Seed FTA and combination partitions for two factors."""
    fs_frame = pl.concat(
        [
            _factor_selection_frame(factor_name="momentum", selection_score=0.7),
            _factor_selection_frame(factor_name="rsi", selection_score=0.5),
        ],
        how="vertical",
    )
    fta_engine = SimpleFactorTimeframeAnalysisEngine(source_selection_version=str(year))
    fta_frame = fta_engine.build(fs_frame)
    fta_repo = FactorTimeframeAnalysisRepository(layout, datastore)
    fta_repo.save(
        fta_frame,
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        year=year,
    )

    comb_engine = SimpleFactorCombinationEngine()
    combination_output = comb_engine.build(fta_frame)
    comb_repo = FactorCombinationRepository(layout, datastore)
    timeframes = combination_output.select("timeframe").unique().to_series().to_list()
    for timeframe in timeframes:
        partition = combination_output.filter(pl.col("timeframe") == timeframe)
        comb_repo.save(
            partition,
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            timeframe=timeframe,
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

    def test_years_list(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--years", "2024", "2026"])
        assert args.years == ["2024", "2026"]


# ---------------------------------------------------------------------------
# build_options
# ---------------------------------------------------------------------------


class TestBuildOptions:
    def _parse(self, *argv: str) -> VerifyFactorCombinationOptions:
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
            self._parse("--years", "notayear")

    def test_years_sorted(self) -> None:
        opts = self._parse("--years", "2026", "2024")
        assert opts.years == (2024, 2026)

    def test_storage_root_default(self) -> None:
        opts = self._parse()
        assert opts.storage_root == Path(DEFAULT_STORAGE_ROOT)


# ---------------------------------------------------------------------------
# discover_work
# ---------------------------------------------------------------------------


class TestDiscoverWork:
    def test_empty_when_no_partitions(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        comb_repo = FactorCombinationRepository(layout, ParquetStore())
        opts = _options(storage_root=tmp_path)
        work = discover_work(comb_repo, opts)
        assert work == ()

    def test_discovers_seeded_combination_partitions(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        _seed_fta_and_combination(layout=layout, datastore=datastore)
        comb_repo = FactorCombinationRepository(layout, datastore)
        opts = _options(storage_root=tmp_path)
        work = discover_work(comb_repo, opts)
        assert len(work) >= 1
        assert all(item.manager == _MANAGER for item in work)
        assert all(item.year == _YEAR for item in work)

    def test_year_filter_excludes(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        _seed_fta_and_combination(layout=layout, datastore=datastore, year=2024)
        _seed_fta_and_combination(layout=layout, datastore=datastore, year=2025)
        comb_repo = FactorCombinationRepository(layout, datastore)
        opts = _options(storage_root=tmp_path, years=(2025,))
        work = discover_work(comb_repo, opts)
        assert all(item.year == 2025 for item in work)


# ---------------------------------------------------------------------------
# run_verification (end-to-end)
# ---------------------------------------------------------------------------


class TestRunVerification:
    def test_empty_work_returns_pass(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        opts = _options(storage_root=tmp_path)
        summary = _run(
            run_verification(
                combination_repository=FactorCombinationRepository(layout, datastore),
                fta_repository=FactorTimeframeAnalysisRepository(layout, datastore),
                verifier=FactorCombinationVerifier(),
                options=opts,
                work=(),
            )
        )
        assert summary.repository_passed is True
        assert summary.successful_tasks == 0

    def test_succeeds_with_valid_combination(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        _seed_fta_and_combination(layout=layout, datastore=datastore)
        comb_repo = FactorCombinationRepository(layout, datastore)
        fta_repo = FactorTimeframeAnalysisRepository(layout, datastore)
        opts = _options(storage_root=tmp_path)
        work = discover_work(comb_repo, opts)
        summary = _run(
            run_verification(
                combination_repository=comb_repo,
                fta_repository=fta_repo,
                verifier=FactorCombinationVerifier(),
                options=opts,
                work=work,
            )
        )
        assert summary.repository_passed is True
        assert summary.successful_tasks >= 1
        assert summary.failed_tasks == 0
        assert summary.rows_checked > 0

    def test_fails_when_combination_partition_missing(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        opts = _options(storage_root=tmp_path)
        work = (DiscoveredWorkItem(manager=_MANAGER, timeframe="1h", year=_YEAR),)
        summary = _run(
            run_verification(
                combination_repository=FactorCombinationRepository(layout, datastore),
                fta_repository=FactorTimeframeAnalysisRepository(layout, datastore),
                verifier=FactorCombinationVerifier(),
                options=opts,
                work=work,
            )
        )
        assert summary.failed_tasks == 1
        assert summary.repository_passed is False

    def test_verifies_against_fta_lineage(self, tmp_path: Path) -> None:
        """Confirm cross-frame lineage check runs without errors for valid data."""
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        _seed_fta_and_combination(layout=layout, datastore=datastore)
        comb_repo = FactorCombinationRepository(layout, datastore)
        fta_repo = FactorTimeframeAnalysisRepository(layout, datastore)
        opts = _options(storage_root=tmp_path)
        work = discover_work(comb_repo, opts)
        summary = _run(
            run_verification(
                combination_repository=comb_repo,
                fta_repository=fta_repo,
                verifier=FactorCombinationVerifier(),
                options=opts,
                work=work,
            )
        )
        assert summary.successful_tasks >= 1
        assert summary.repository_passed is True


# ---------------------------------------------------------------------------
# format_partition_failure / format_summary
# ---------------------------------------------------------------------------


class TestFormatters:
    def test_format_partition_failure_contains_fields(self) -> None:
        text = format_partition_failure(
            dataset="Factor Combination",
            manager="mgr",
            timeframe="1h",
            year=2026,
            partition="2026.parquet",
            verifier="FactorCombinationVerifier",
            exception_type="FactorCombinationError",
            message="lineage fail",
            code="FCOMB-001",
        )
        assert "mgr" in text
        assert "1h" in text
        assert "2026" in text
        assert "FCOMB-001" in text

    def test_format_summary_pass(self) -> None:
        summary = VerifyFactorCombinationSummary(
            panels_verified=2,
            datasets_verified=1,
            timeframes_verified=1,
            successful_tasks=2,
            failed_tasks=0,
            rows_checked=10,
            duration_seconds=0.1,
            repository_passed=True,
        )
        text = format_summary(summary)
        assert "PASS" in text

    def test_format_summary_fail(self) -> None:
        summary = VerifyFactorCombinationSummary(
            panels_verified=1,
            datasets_verified=1,
            timeframes_verified=1,
            successful_tasks=0,
            failed_tasks=1,
            rows_checked=0,
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

    def test_exits_0_with_valid_combination(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        _seed_fta_and_combination(layout=layout, datastore=datastore)
        exit_code = _run(main(["--manager", _MANAGER, "--storage-root", str(tmp_path)]))
        assert exit_code == 0

    def test_exits_1_on_bad_options(self) -> None:
        exit_code = _run(main(["--workers", "0"]))
        assert exit_code == 1
