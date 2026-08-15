"""Unit tests for CQROS factor timeframe analysis generation CLI."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from pathlib import Path

import polars as pl
import pytest

from cqros.cli.generate_factor_timeframe_analysis import (
    FactorTimeframeAnalysisGenerationOptions,
    FactorTimeframeAnalysisGenerationSummary,
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
from cqros.factor_selection import FactorSelectionRepository
from cqros.factor_selection.schema import FACTOR_SELECTION_SCHEMA, FactorSelectionStatus
from cqros.factor_timeframe_analysis import (
    FactorTimeframeAnalysisRepository,
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
    engine: str = "simple",
    timeframes: tuple[str, ...] | None = None,
    years: tuple[int, ...] | None = None,
    overwrite: bool = False,
    export_detailed_csv: bool = False,
    workers: int = ResearchConfig().worker_count,
    verbose: bool = False,
    debug: bool = False,
) -> FactorTimeframeAnalysisGenerationOptions:
    """Build FactorTimeframeAnalysisGenerationOptions against a temporary storage root."""
    return FactorTimeframeAnalysisGenerationOptions(
        storage_root=storage_root,
        manager=manager,
        engine=engine,
        timeframes=timeframes,
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
    selected: bool = True,
) -> pl.DataFrame:
    """Return a minimal canonical Factor Selection frame for FTA generation tests."""
    return pl.DataFrame(
        {
            "factor_name": [factor_name],
            "factor_version": [factor_version],
            "timeframe": [timeframe],
            "selection_time": [_SELECTION_TIME],
            "factor_category": ["price"],
            "selected": [selected],
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


def _seed_factor_selection(
    *,
    layout: StorageLayout,
    datastore: ParquetStore,
    factor_name: str = "momentum",
    timeframe: str = "1h",
    year: int = _YEAR,
    selected: bool = True,
) -> None:
    """Persist a Factor Selection partition for FTA generation tests."""
    FactorSelectionRepository(layout, datastore).save(
        _factor_selection_frame(factor_name=factor_name, timeframe=timeframe, selected=selected),
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
        assert args.engine == "simple"
        assert args.overwrite is False
        assert args.export_detailed_csv is False
        assert args.years is None
        assert args.timeframes is None
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

    def test_timeframes_allowlist(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--manager", "simple", "--timeframes", "1h", "4h"])
        assert args.timeframes == ["1h", "4h"]

    def test_storage_root_parsed(self, tmp_path: Path) -> None:
        parser = build_parser()
        args = parser.parse_args(["--manager", "simple", "--storage-root", str(tmp_path)])
        assert args.storage_root == tmp_path


# ---------------------------------------------------------------------------
# build_options
# ---------------------------------------------------------------------------


class TestBuildOptions:
    def _parse(self, *argv: str) -> FactorTimeframeAnalysisGenerationOptions:
        parser = build_parser()
        args = parser.parse_args(list(argv))
        return build_options(args)

    def test_valid_minimal(self) -> None:
        opts = self._parse("--manager", "simple")
        assert opts.manager == "simple"
        assert opts.engine == "simple"
        assert opts.overwrite is False
        assert opts.export_detailed_csv is False

    def test_invalid_workers(self) -> None:
        with pytest.raises(ValidationError):
            self._parse("--manager", "simple", "--workers", "0")

    def test_empty_manager_rejected(self) -> None:
        with pytest.raises(ValidationError):
            parser = build_parser()
            args = parser.parse_args(["--manager", "   "])
            build_options(args)

    def test_invalid_year_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._parse("--manager", "simple", "--years", "notayear")

    def test_invalid_timeframe_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._parse("--manager", "simple", "--timeframes", "99x")

    def test_years_normalized(self) -> None:
        opts = self._parse("--manager", "simple", "--years", "2026", "2025")
        assert opts.years == (2025, 2026)

    def test_storage_root_default(self) -> None:
        opts = self._parse("--manager", "simple")
        assert opts.storage_root == Path(DEFAULT_STORAGE_ROOT)


# ---------------------------------------------------------------------------
# discover_work
# ---------------------------------------------------------------------------


class TestDiscoverWork:
    def test_empty_when_no_partitions(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        fs_repo = FactorSelectionRepository(layout, datastore)
        opts = _options(storage_root=tmp_path)
        work = discover_work(fs_repo, opts)
        assert work == ()

    def test_discovers_seeded_partition(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        _seed_factor_selection(layout=layout, datastore=datastore)
        fs_repo = FactorSelectionRepository(layout, datastore)
        opts = _options(storage_root=tmp_path)
        work = discover_work(fs_repo, opts)
        assert len(work) == 1
        assert work[0].manager == _MANAGER
        assert work[0].year == _YEAR

    def test_year_filter_excludes(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        _seed_factor_selection(layout=layout, datastore=datastore, year=2024)
        _seed_factor_selection(layout=layout, datastore=datastore, year=2025)
        fs_repo = FactorSelectionRepository(layout, datastore)
        opts = _options(storage_root=tmp_path, years=(2025,))
        work = discover_work(fs_repo, opts)
        assert len(work) == 1
        assert work[0].year == 2025

    def test_multiple_timeframes_merged_into_one_year_item(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        _seed_factor_selection(layout=layout, datastore=datastore, timeframe="1h")
        _seed_factor_selection(layout=layout, datastore=datastore, timeframe="4h")
        fs_repo = FactorSelectionRepository(layout, datastore)
        opts = _options(storage_root=tmp_path)
        work = discover_work(fs_repo, opts)
        assert len(work) == 1
        assert work[0].year == _YEAR


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
                layout=layout,
                factor_selection_repository=FactorSelectionRepository(layout, datastore),
                fta_repository=FactorTimeframeAnalysisRepository(layout, datastore),
                options=opts,
                work=(),
            )
        )
        assert summary.panels == 0
        assert summary.successful_tasks == 0
        assert summary.failed_tasks == 0

    def test_succeeds_with_seeded_selection(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        _seed_factor_selection(layout=layout, datastore=datastore, timeframe="1h")
        _seed_factor_selection(
            layout=layout, datastore=datastore, factor_name="rsi", timeframe="4h"
        )
        fs_repo = FactorSelectionRepository(layout, datastore)
        fta_repo = FactorTimeframeAnalysisRepository(layout, datastore)
        opts = _options(storage_root=tmp_path)
        work = discover_work(fs_repo, opts)
        summary = _run(
            run_generation(
                layout=layout,
                factor_selection_repository=fs_repo,
                fta_repository=fta_repo,
                options=opts,
                work=work,
            )
        )
        assert summary.successful_tasks == 1
        assert summary.failed_tasks == 0
        assert summary.rows > 0
        assert fta_repo.exists(
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            year=_YEAR,
        )

    def test_skip_when_partition_exists(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        _seed_factor_selection(layout=layout, datastore=datastore)
        fs_repo = FactorSelectionRepository(layout, datastore)
        fta_repo = FactorTimeframeAnalysisRepository(layout, datastore)
        opts = _options(storage_root=tmp_path)
        work = discover_work(fs_repo, opts)
        _run(
            run_generation(
                layout=layout,
                factor_selection_repository=fs_repo,
                fta_repository=fta_repo,
                options=opts,
                work=work,
            )
        )
        summary2 = _run(
            run_generation(
                layout=layout,
                factor_selection_repository=fs_repo,
                fta_repository=fta_repo,
                options=opts,
                work=work,
            )
        )
        assert summary2.skipped_tasks == 1
        assert summary2.successful_tasks == 0

    def test_overwrite_regenerates(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        _seed_factor_selection(layout=layout, datastore=datastore)
        fs_repo = FactorSelectionRepository(layout, datastore)
        fta_repo = FactorTimeframeAnalysisRepository(layout, datastore)
        opts = _options(storage_root=tmp_path)
        work = discover_work(fs_repo, opts)
        _run(
            run_generation(
                layout=layout,
                factor_selection_repository=fs_repo,
                fta_repository=fta_repo,
                options=opts,
                work=work,
            )
        )
        opts_overwrite = _options(storage_root=tmp_path, overwrite=True)
        summary2 = _run(
            run_generation(
                layout=layout,
                factor_selection_repository=fs_repo,
                fta_repository=fta_repo,
                options=opts_overwrite,
                work=work,
            )
        )
        assert summary2.successful_tasks == 1
        assert summary2.skipped_tasks == 0

    def test_fta_output_has_correct_source_version(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        _seed_factor_selection(layout=layout, datastore=datastore)
        fs_repo = FactorSelectionRepository(layout, datastore)
        fta_repo = FactorTimeframeAnalysisRepository(layout, datastore)
        opts = _options(storage_root=tmp_path)
        work = discover_work(fs_repo, opts)
        _run(
            run_generation(
                layout=layout,
                factor_selection_repository=fs_repo,
                fta_repository=fta_repo,
                options=opts,
                work=work,
            )
        )
        frame = fta_repo.load(
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            year=_YEAR,
        )
        versions = frame["source_selection_version"].unique().to_list()
        assert str(_YEAR) in versions

    def test_export_detailed_csv(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        _seed_factor_selection(layout=layout, datastore=datastore)
        fs_repo = FactorSelectionRepository(layout, datastore)
        fta_repo = FactorTimeframeAnalysisRepository(layout, datastore)
        opts = _options(storage_root=tmp_path, export_detailed_csv=True)
        work = discover_work(fs_repo, opts)
        _run(
            run_generation(
                layout=layout,
                factor_selection_repository=fs_repo,
                fta_repository=fta_repo,
                options=opts,
                work=work,
            )
        )
        from cqros.factor_timeframe_analysis import detailed_csv_path

        csv_path = detailed_csv_path(
            tmp_path,
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            year=_YEAR,
        )
        assert csv_path.exists()


# ---------------------------------------------------------------------------
# format_summary
# ---------------------------------------------------------------------------


class TestFormatSummary:
    def test_includes_manager(self, tmp_path: Path) -> None:
        summary = FactorTimeframeAnalysisGenerationSummary(
            manager="testmgr",
            engine="simple",
            panels=3,
            rows=100,
            selected_rows=60,
            successful_tasks=2,
            failed_tasks=1,
            skipped_tasks=0,
            duration_seconds=1.5,
            output_directory=tmp_path / "fta",
            failed_task_labels=("2026",),
        )
        text = format_summary(summary)
        assert "testmgr" in text
        assert "Failed Tasks" in text
        assert "2026" in text

    def test_no_failed_tasks_section(self, tmp_path: Path) -> None:
        summary = FactorTimeframeAnalysisGenerationSummary(
            manager="simple",
            engine="simple",
            panels=1,
            rows=10,
            selected_rows=5,
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
# main (integration smoke test)
# ---------------------------------------------------------------------------


class TestMain:
    def test_returns_failure_on_missing_manager(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            _run(main([]))
        assert exc_info.value.code == 2

    def test_succeeds_with_seeded_selection(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        _seed_factor_selection(layout=layout, datastore=datastore)
        exit_code = _run(main(["--manager", _MANAGER, "--storage-root", str(tmp_path)]))
        assert exit_code == 0

    def test_exits_0_when_no_work(self, tmp_path: Path) -> None:
        exit_code = _run(main(["--manager", _MANAGER, "--storage-root", str(tmp_path)]))
        assert exit_code == 0
