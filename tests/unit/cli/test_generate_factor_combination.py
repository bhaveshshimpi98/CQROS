"""Unit tests for CQROS factor combination generation CLI."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from pathlib import Path

import polars as pl
import pytest

from cqros.cli.generate_factor_combination import (
    FactorCombinationGenerationOptions,
    FactorCombinationGenerationSummary,
    build_options,
    build_parser,
    discover_work,
    format_summary,
    main,
    run_generation,
)
from cqros.config.models import ResearchConfig
from cqros.core.constants import (
    DEFAULT_STORAGE_ROOT,
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
)
from cqros.core.exceptions import ValidationError
from cqros.factor_combination import FactorCombinationRepository
from cqros.factor_selection import FactorSelectionRepository
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
    manager: str = _MANAGER,
    years: tuple[int, ...] | None = None,
    overwrite: bool = False,
    export_detailed_csv: bool = False,
    workers: int = ResearchConfig().worker_count,
    verbose: bool = False,
    debug: bool = False,
) -> FactorCombinationGenerationOptions:
    """Build FactorCombinationGenerationOptions against a temporary storage root."""
    return FactorCombinationGenerationOptions(
        storage_root=storage_root,
        manager=manager,
        years=years,
        overwrite=overwrite,
        export_detailed_csv=export_detailed_csv,
        workers=workers,
        verbose=verbose,
        debug=debug,
    )


def _factor_selection_frame(
    *,
    factor_name: str = "momentum",
    factor_version: str = "1.0.0",
    timeframe: str = "1h",
    selection_score: float = 0.5,
) -> pl.DataFrame:
    """Return a minimal canonical Factor Selection frame."""
    return pl.DataFrame(
        {
            "factor_name": [factor_name],
            "factor_version": [factor_version],
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


def _seed_fta_with_two_factors(
    *,
    layout: StorageLayout,
    datastore: ParquetStore,
    year: int = _YEAR,
) -> None:
    """Seed FTA partition with two selected factors for combination generation."""
    fs_frame = pl.concat(
        [
            _factor_selection_frame(factor_name="momentum", selection_score=0.7),
            _factor_selection_frame(factor_name="rsi", selection_score=0.5),
        ],
        how="vertical",
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
    def test_manager_required(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_manager_accepted(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--manager", "simple"])
        assert args.manager == "simple"

    def test_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--manager", "simple"])
        assert args.overwrite is False
        assert args.export_detailed_csv is False
        assert args.years is None
        assert args.storage_root is None
        assert args.verbose is False
        assert args.debug is False

    def test_overwrite_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--manager", "simple", "--overwrite"])
        assert args.overwrite is True

    def test_years_multiple(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--manager", "simple", "--years", "2025", "2026"])
        assert args.years == ["2025", "2026"]


# ---------------------------------------------------------------------------
# build_options
# ---------------------------------------------------------------------------


class TestBuildOptions:
    def _parse(self, *argv: str) -> FactorCombinationGenerationOptions:
        parser = build_parser()
        args = parser.parse_args(list(argv))
        return build_options(args)

    def test_valid_minimal(self) -> None:
        opts = self._parse("--manager", "simple")
        assert opts.manager == "simple"
        assert opts.overwrite is False

    def test_invalid_workers(self) -> None:
        with pytest.raises(ValidationError):
            self._parse("--manager", "simple", "--workers", "0")

    def test_empty_manager_rejected(self) -> None:
        with pytest.raises(ValidationError):
            parser = build_parser()
            args = parser.parse_args(["--manager", ""])
            build_options(args)

    def test_invalid_year_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._parse("--manager", "simple", "--years", "bad")

    def test_years_sorted(self) -> None:
        opts = self._parse("--manager", "simple", "--years", "2026", "2024")
        assert opts.years == (2024, 2026)

    def test_storage_root_default(self) -> None:
        opts = self._parse("--manager", "simple")
        assert opts.storage_root == Path(DEFAULT_STORAGE_ROOT)


# ---------------------------------------------------------------------------
# discover_work
# ---------------------------------------------------------------------------


class TestDiscoverWork:
    def test_empty_when_no_fta_partitions(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        fta_repo = FactorTimeframeAnalysisRepository(layout, ParquetStore())
        opts = _options(storage_root=tmp_path)
        work = discover_work(fta_repo, opts)
        assert work == ()

    def test_discovers_seeded_fta_partition(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        _seed_fta_with_two_factors(layout=layout, datastore=datastore)
        fta_repo = FactorTimeframeAnalysisRepository(layout, datastore)
        opts = _options(storage_root=tmp_path)
        work = discover_work(fta_repo, opts)
        assert len(work) == 1
        assert work[0].manager == _MANAGER
        assert work[0].year == _YEAR

    def test_year_filter_excludes(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        _seed_fta_with_two_factors(layout=layout, datastore=datastore, year=2024)
        _seed_fta_with_two_factors(layout=layout, datastore=datastore, year=2025)
        fta_repo = FactorTimeframeAnalysisRepository(layout, datastore)
        opts = _options(storage_root=tmp_path, years=(2025,))
        work = discover_work(fta_repo, opts)
        assert len(work) == 1
        assert work[0].year == 2025


# ---------------------------------------------------------------------------
# run_generation (end-to-end)
# ---------------------------------------------------------------------------


class TestRunGeneration:
    def test_empty_work_returns_zero_summary(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        opts = _options(storage_root=tmp_path)
        summary = _run(
            run_generation(
                fta_repository=FactorTimeframeAnalysisRepository(layout, datastore),
                combination_repository=FactorCombinationRepository(layout, datastore),
                options=opts,
                work=(),
            )
        )
        assert summary.panels == 0
        assert summary.successful_tasks == 0
        assert summary.failed_tasks == 0

    def test_succeeds_with_two_selected_factors(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        _seed_fta_with_two_factors(layout=layout, datastore=datastore)
        fta_repo = FactorTimeframeAnalysisRepository(layout, datastore)
        comb_repo = FactorCombinationRepository(layout, datastore)
        opts = _options(storage_root=tmp_path)
        work = discover_work(fta_repo, opts)
        summary = _run(
            run_generation(
                fta_repository=fta_repo,
                combination_repository=comb_repo,
                options=opts,
                work=work,
            )
        )
        assert summary.successful_tasks >= 1
        assert summary.failed_tasks == 0
        assert summary.rows > 0

    def test_skip_when_partition_exists(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        _seed_fta_with_two_factors(layout=layout, datastore=datastore)
        fta_repo = FactorTimeframeAnalysisRepository(layout, datastore)
        comb_repo = FactorCombinationRepository(layout, datastore)
        opts = _options(storage_root=tmp_path)
        work = discover_work(fta_repo, opts)
        _run(
            run_generation(
                fta_repository=fta_repo,
                combination_repository=comb_repo,
                options=opts,
                work=work,
            )
        )
        summary2 = _run(
            run_generation(
                fta_repository=fta_repo,
                combination_repository=comb_repo,
                options=opts,
                work=work,
            )
        )
        assert summary2.skipped_tasks >= 1
        assert summary2.successful_tasks == 0

    def test_overwrite_regenerates(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        _seed_fta_with_two_factors(layout=layout, datastore=datastore)
        fta_repo = FactorTimeframeAnalysisRepository(layout, datastore)
        comb_repo = FactorCombinationRepository(layout, datastore)
        opts = _options(storage_root=tmp_path)
        work = discover_work(fta_repo, opts)
        _run(
            run_generation(
                fta_repository=fta_repo,
                combination_repository=comb_repo,
                options=opts,
                work=work,
            )
        )
        opts_overwrite = _options(storage_root=tmp_path, overwrite=True)
        summary2 = _run(
            run_generation(
                fta_repository=fta_repo,
                combination_repository=comb_repo,
                options=opts_overwrite,
                work=work,
            )
        )
        assert summary2.successful_tasks >= 1
        assert summary2.skipped_tasks == 0

    def test_combination_partitioned_by_timeframe(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        _seed_fta_with_two_factors(layout=layout, datastore=datastore)
        fta_repo = FactorTimeframeAnalysisRepository(layout, datastore)
        comb_repo = FactorCombinationRepository(layout, datastore)
        opts = _options(storage_root=tmp_path)
        work = discover_work(fta_repo, opts)
        _run(
            run_generation(
                fta_repository=fta_repo,
                combination_repository=comb_repo,
                options=opts,
                work=work,
            )
        )
        timeframes = comb_repo.discover_timeframes(
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
        )
        assert len(timeframes) >= 1

    def test_combination_does_not_load_factor_selection(self, tmp_path: Path) -> None:
        """Verify generate_combination does not depend on Factor Selection repo."""
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        _seed_fta_with_two_factors(layout=layout, datastore=datastore)
        fta_repo = FactorTimeframeAnalysisRepository(layout, datastore)
        comb_repo = FactorCombinationRepository(layout, datastore)
        fs_repo = FactorSelectionRepository(layout, datastore)
        assert fs_repo.discover_managers() == ()
        opts = _options(storage_root=tmp_path)
        work = discover_work(fta_repo, opts)
        summary = _run(
            run_generation(
                fta_repository=fta_repo,
                combination_repository=comb_repo,
                options=opts,
                work=work,
            )
        )
        assert summary.successful_tasks >= 1

    def test_export_detailed_csv(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        _seed_fta_with_two_factors(layout=layout, datastore=datastore)
        fta_repo = FactorTimeframeAnalysisRepository(layout, datastore)
        comb_repo = FactorCombinationRepository(layout, datastore)
        opts = _options(storage_root=tmp_path, export_detailed_csv=True)
        work = discover_work(fta_repo, opts)
        _run(
            run_generation(
                fta_repository=fta_repo,
                combination_repository=comb_repo,
                options=opts,
                work=work,
            )
        )
        from cqros.factor_combination import combined_detailed_csv_path

        combined = combined_detailed_csv_path(
            tmp_path,
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
        )
        assert combined.exists()


# ---------------------------------------------------------------------------
# format_summary
# ---------------------------------------------------------------------------


class TestFormatSummary:
    def test_includes_manager(self, tmp_path: Path) -> None:
        summary = FactorCombinationGenerationSummary(
            manager="testmgr",
            panels=3,
            rows=100,
            successful_tasks=2,
            failed_tasks=1,
            skipped_tasks=0,
            duration_seconds=1.5,
            output_directory=tmp_path / "comb",
            failed_task_labels=("2026/1h",),
        )
        text = format_summary(summary)
        assert "testmgr" in text
        assert "Failed Tasks" in text
        assert "2026/1h" in text

    def test_no_failed_section_when_clean(self, tmp_path: Path) -> None:
        summary = FactorCombinationGenerationSummary(
            manager="simple",
            panels=1,
            rows=10,
            successful_tasks=1,
            failed_tasks=0,
            skipped_tasks=0,
            duration_seconds=0.1,
            output_directory=tmp_path,
            failed_task_labels=(),
        )
        text = format_summary(summary)
        assert "Failed Tasks" not in text


# ---------------------------------------------------------------------------
# main (smoke test)
# ---------------------------------------------------------------------------


class TestMain:
    def test_returns_failure_on_missing_manager(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            _run(main([]))
        assert exc_info.value.code == 2

    def test_exits_0_when_no_fta_work(self, tmp_path: Path) -> None:
        exit_code = _run(main(["--manager", _MANAGER, "--storage-root", str(tmp_path)]))
        assert exit_code == 0

    def test_succeeds_with_fta_partitions(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        _seed_fta_with_two_factors(layout=layout, datastore=datastore)
        exit_code = _run(main(["--manager", _MANAGER, "--storage-root", str(tmp_path)]))
        assert exit_code == 0
