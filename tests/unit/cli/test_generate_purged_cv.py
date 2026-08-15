"""Unit tests for CQROS purged-CV generation CLI."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

import cqros.cli.generate_purged_cv as generate_purged_cv_module
from cqros.cli.generate_purged_cv import (
    DiscoveredWorkItem,
    PurgedCVGenerationOptions,
    PurgedCVGenerationSummary,
    build_default_engine,
    build_options,
    build_parser,
    build_purged_cv_pipeline,
    build_registry,
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
    STORAGE_DIR_PURGED_CV,
)
from cqros.core.exceptions import ValidationError
from cqros.purged_cv import (
    PurgedCVEngineRegistry,
    PurgedCVPipeline,
    PurgedCVRepository,
    SimplePurgedCVEngine,
)
from cqros.storage import ParquetStore, StorageLayout
from cqros.walk_forward import WalkForwardRepository, WalkForwardStatus
from cqros.walk_forward.schema import (
    CANONICAL_COLUMN_ORDER as WF_CANONICAL_COLUMN_ORDER,
)
from cqros.walk_forward.schema import (
    COLUMN_DTYPES as WF_COLUMN_DTYPES,
)

_MANAGER = "simple"
_ENGINE = "simple"
_TIMEFRAME = "1h"
_YEAR = 2026
_BASE_TIME = 1_700_000_000_000
_HOUR_MS = 3_600_000


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def _options(
    *,
    storage_root: Path,
    manager: str = _MANAGER,
    engine: str = _ENGINE,
    timeframes: tuple[str, ...] | None = None,
    years: tuple[int, ...] | None = None,
    overwrite: bool = False,
    workers: int = ResearchConfig().worker_count,
    verbose: bool = False,
    debug: bool = False,
) -> PurgedCVGenerationOptions:
    """Build PurgedCVGenerationOptions against a temporary storage root."""
    return PurgedCVGenerationOptions(
        storage_root=storage_root,
        manager=manager,
        engine=engine,
        timeframes=timeframes,
        years=years,
        overwrite=overwrite,
        workers=workers,
        verbose=verbose,
        debug=debug,
    )


def _walk_forward_frame(*, row_count: int = 10) -> pl.DataFrame:
    """Return a canonical Walk-Forward frame for purged-CV generation tests."""
    times = [_BASE_TIME + (index * _HOUR_MS) for index in range(row_count)]
    return pl.DataFrame(
        {
            "strategy_name": ["default_strategy"] * row_count,
            "strategy_version": ["v1"] * row_count,
            "timeframe": [_TIMEFRAME] * row_count,
            "fold_id": list(range(1, row_count + 1)),
            "train_start": times,
            "train_end": times,
            "test_start": times,
            "test_end": times,
            "train_rows": [3] * row_count,
            "test_rows": [2] * row_count,
            "selected_factors": [1] * row_count,
            "model_version": ["v1"] * row_count,
            "train_score": [0.10 + (0.01 * index) for index in range(row_count)],
            "test_score": [0.05 + (0.01 * index) for index in range(row_count)],
            "overfit_gap": [None] * row_count,
            "status": [WalkForwardStatus.PASS.value] * row_count,
        },
        schema=dict(WF_COLUMN_DTYPES),
    ).select(list(WF_CANONICAL_COLUMN_ORDER))


def _seed_generation_inputs(
    *,
    layout: StorageLayout,
    datastore: ParquetStore,
    year: int = _YEAR,
    timeframe: str = _TIMEFRAME,
) -> None:
    """Persist a Walk-Forward panel needed for purged-CV generation."""
    WalkForwardRepository(layout, datastore).save(
        _walk_forward_frame(),
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe=timeframe,
        year=year,
    )


def _compact_registry() -> PurgedCVEngineRegistry:
    """Build a registry with a compact SimplePurgedCVEngine for deterministic tests."""
    registry = PurgedCVEngineRegistry()
    registry.register(
        _ENGINE,
        SimplePurgedCVEngine(n_folds=2, purge_size=0, embargo_size=0),
    )
    return registry


# ---------------------------------------------------------------------------
# Parser and options
# ---------------------------------------------------------------------------


def test_build_parser_requires_manager() -> None:
    """Parser requires --manager and defaults engine to simple."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(["--manager", "simple"])
    assert args.manager == "simple"
    assert args.engine == "simple"


def test_build_parser_accepts_all_flags(tmp_path: Path) -> None:
    """Parser correctly maps all supported CLI flags."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "--manager",
            "ledger",
            "--engine",
            "custom",
            "--workers",
            "4",
            "--overwrite",
            "--storage-root",
            str(tmp_path),
        ]
    )
    assert args.manager == "ledger"
    assert args.engine == "custom"
    assert args.workers == 4
    assert args.overwrite is True
    assert args.storage_root == tmp_path
    assert not hasattr(args, "symbols") or getattr(args, "symbols", None) is None


def test_build_parser_has_no_symbols_flag() -> None:
    """Panel-based purged-CV generation does not expose --symbols."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--manager", "simple", "--symbols", "BTCUSDT"])


def test_build_options_rejects_non_positive_workers() -> None:
    """build_options rejects workers <= 0 with CLI-GENERATE-PURGED-CV-001."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "simple", "--workers", "0"])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-PURGED-CV-001"


def test_build_options_rejects_blank_manager() -> None:
    """build_options rejects a blank manager identity with CLI-GENERATE-PURGED-CV-004."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "   "])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-PURGED-CV-004"


def test_build_options_rejects_blank_engine() -> None:
    """build_options rejects a blank engine identity with CLI-GENERATE-PURGED-CV-005."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "simple", "--engine", "   "])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-PURGED-CV-005"


def test_build_options_uses_storage_root(tmp_path: Path) -> None:
    """build_options honors an explicit storage root override."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "simple", "--storage-root", str(tmp_path)])
    options = build_options(args)
    assert options.storage_root == tmp_path


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def test_build_default_engine_returns_simple_purged_cv_engine() -> None:
    """build_default_engine returns a SimplePurgedCVEngine instance."""
    assert isinstance(build_default_engine(), SimplePurgedCVEngine)


def test_build_registry_contains_simple_engine() -> None:
    """Default registry contains SimplePurgedCVEngine under 'simple'."""
    registry = build_registry()
    assert registry.exists("simple")
    assert isinstance(registry.get("simple"), SimplePurgedCVEngine)


def test_build_purged_cv_pipeline_wires_registry() -> None:
    """build_purged_cv_pipeline returns a fully wired PurgedCVPipeline."""
    options = _options(storage_root=Path(DEFAULT_STORAGE_ROOT))
    pipeline = build_purged_cv_pipeline(options)
    assert isinstance(pipeline, PurgedCVPipeline)


# ---------------------------------------------------------------------------
# discover_work
# ---------------------------------------------------------------------------


def test_discover_work_groups_walk_forward_panels(tmp_path: Path) -> None:
    """discover_work returns sorted panel work items from the walk-forward tier."""
    layout = StorageLayout(tmp_path)
    walk_forward_repository = WalkForwardRepository(layout, ParquetStore())
    for timeframe, year in (("4h", 2025), ("1h", 2026), ("1h", 2025)):
        walk_forward_repository.save(
            _walk_forward_frame(),
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            timeframe=timeframe,
            year=year,
        )
    work = discover_work(
        walk_forward_repository,
        _options(storage_root=tmp_path),
    )
    assert work == (
        DiscoveredWorkItem(
            manager=_MANAGER,
            timeframe="1h",
            years=(2025, 2026),
        ),
        DiscoveredWorkItem(
            manager=_MANAGER,
            timeframe="4h",
            years=(2025,),
        ),
    )


def test_discover_work_returns_empty_when_no_walk_forward_partitions(
    tmp_path: Path,
) -> None:
    """discover_work returns empty tuple when no Walk-Forward partitions exist."""
    layout = StorageLayout(tmp_path)
    walk_forward_repository = WalkForwardRepository(layout, ParquetStore())
    work = discover_work(walk_forward_repository, _options(storage_root=tmp_path))
    assert work == ()


def test_discover_work_filters_by_timeframe(tmp_path: Path) -> None:
    """discover_work respects timeframe allowlist filters."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    _seed_generation_inputs(layout=layout, datastore=datastore, timeframe="1h")
    _seed_generation_inputs(layout=layout, datastore=datastore, timeframe="4h")
    work = discover_work(
        WalkForwardRepository(layout, datastore),
        _options(storage_root=tmp_path, timeframes=("1h",)),
    )
    assert len(work) == 1
    assert work[0].timeframe == "1h"


# ---------------------------------------------------------------------------
# format_summary
# ---------------------------------------------------------------------------


def test_format_summary_includes_core_fields() -> None:
    """format_summary renders manager, engine, and purged-CV aggregates."""
    summary = PurgedCVGenerationSummary(
        manager=_MANAGER,
        engine=_ENGINE,
        panels=2,
        rows=10,
        pass_rows=8,
        fail_rows=2,
        status="FAILED",
        successful_tasks=2,
        failed_tasks=1,
        skipped_tasks=0,
        duration_seconds=2.5,
        output_directory=Path("data") / STORAGE_DIR_PURGED_CV,
        failed_task_labels=("1h 2026",),
    )
    text = format_summary(summary)
    assert "CQROS Purged-CV Generation Summary" in text
    assert f"Manager: {_MANAGER}" in text
    assert f"Engine: {_ENGINE}" in text
    assert "Panels: 2" in text
    assert "Rows: 10" in text
    assert "Passed: 8" in text
    assert "Failed: 2" in text
    assert "Status: FAILED" in text
    assert "Failed Tasks" in text
    assert "1h 2026" in text
    assert STORAGE_DIR_PURGED_CV in text


# ---------------------------------------------------------------------------
# run_generation — empty work
# ---------------------------------------------------------------------------


def test_run_generation_empty_work(tmp_path: Path) -> None:
    """Empty work produces a zeroed success summary."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    options = _options(storage_root=tmp_path)
    registry = _compact_registry()
    summary = _run(
        run_generation(
            pipeline=PurgedCVPipeline(registry),
            walk_forward_repository=WalkForwardRepository(layout, datastore),
            purged_cv_repository=PurgedCVRepository(layout, datastore),
            options=options,
            work=(),
        )
    )
    assert summary.successful_tasks == 0
    assert summary.failed_tasks == 0
    assert summary.skipped_tasks == 0
    assert summary.panels == 0
    assert summary.rows == 0
    assert summary.pass_rows == 0
    assert summary.fail_rows == 0
    assert summary.status == "SUCCESS"
    assert summary.output_directory == tmp_path / STORAGE_DIR_PURGED_CV


# ---------------------------------------------------------------------------
# run_generation — persists purged-CV ledgers
# ---------------------------------------------------------------------------


def test_run_generation_persists_purged_cv_partitions(tmp_path: Path) -> None:
    """Generation loads Walk-Forward panels, runs the pipeline, and persists output."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()

    _seed_generation_inputs(layout=layout, datastore=datastore, timeframe="1h", year=2025)
    _seed_generation_inputs(layout=layout, datastore=datastore, timeframe="1h", year=2026)

    options = _options(storage_root=tmp_path)
    walk_forward_repository = WalkForwardRepository(layout, datastore)
    work = discover_work(walk_forward_repository, options)

    registry = _compact_registry()
    summary = _run(
        run_generation(
            pipeline=PurgedCVPipeline(registry),
            walk_forward_repository=walk_forward_repository,
            purged_cv_repository=PurgedCVRepository(layout, datastore),
            options=options,
            work=work,
        )
    )
    assert summary.successful_tasks == 2
    assert summary.failed_tasks == 0
    assert summary.panels == 2
    assert summary.rows == 4
    assert summary.pass_rows == 4
    assert summary.fail_rows == 0
    assert summary.status == "SUCCESS"

    purged_cv_repo = PurgedCVRepository(layout, datastore)
    assert purged_cv_repo.exists(
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe=_TIMEFRAME,
        year=2025,
    )
    assert purged_cv_repo.exists(
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe=_TIMEFRAME,
        year=2026,
    )


def test_run_generation_is_deterministic(tmp_path: Path) -> None:
    """Identical Walk-Forward inputs produce identical purged-CV outputs."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    _seed_generation_inputs(layout=layout, datastore=datastore)

    options = _options(storage_root=tmp_path, overwrite=True)
    walk_forward_repository = WalkForwardRepository(layout, datastore)
    work = discover_work(walk_forward_repository, options)
    purged_cv_repository = PurgedCVRepository(layout, datastore)

    first = _run(
        run_generation(
            pipeline=PurgedCVPipeline(_compact_registry()),
            walk_forward_repository=walk_forward_repository,
            purged_cv_repository=purged_cv_repository,
            options=options,
            work=work,
        )
    )
    first_frame = purged_cv_repository.load(
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    second = _run(
        run_generation(
            pipeline=PurgedCVPipeline(_compact_registry()),
            walk_forward_repository=walk_forward_repository,
            purged_cv_repository=purged_cv_repository,
            options=options,
            work=work,
        )
    )
    second_frame = purged_cv_repository.load(
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert first.rows == second.rows
    assert first_frame.equals(second_frame)


# ---------------------------------------------------------------------------
# run_generation — skip / overwrite
# ---------------------------------------------------------------------------


def test_run_generation_skips_existing_without_overwrite(tmp_path: Path) -> None:
    """Existing purged-CV partitions are skipped when overwrite is False."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    _seed_generation_inputs(layout=layout, datastore=datastore)

    options = _options(storage_root=tmp_path, overwrite=False)
    walk_forward_repository = WalkForwardRepository(layout, datastore)
    work = discover_work(walk_forward_repository, options)

    first_summary = _run(
        run_generation(
            pipeline=PurgedCVPipeline(_compact_registry()),
            walk_forward_repository=walk_forward_repository,
            purged_cv_repository=PurgedCVRepository(layout, datastore),
            options=options,
            work=work,
        )
    )
    assert first_summary.successful_tasks == 1

    second_summary = _run(
        run_generation(
            pipeline=PurgedCVPipeline(_compact_registry()),
            walk_forward_repository=walk_forward_repository,
            purged_cv_repository=PurgedCVRepository(layout, datastore),
            options=options,
            work=work,
        )
    )
    assert second_summary.skipped_tasks == 1
    assert second_summary.successful_tasks == 0
    assert second_summary.failed_tasks == 0


def test_run_generation_overwrites_existing(tmp_path: Path) -> None:
    """Existing purged-CV partitions are regenerated when overwrite is True."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    _seed_generation_inputs(layout=layout, datastore=datastore)

    options = _options(storage_root=tmp_path, overwrite=True)
    walk_forward_repository = WalkForwardRepository(layout, datastore)
    work = discover_work(walk_forward_repository, options)

    first_summary = _run(
        run_generation(
            pipeline=PurgedCVPipeline(_compact_registry()),
            walk_forward_repository=walk_forward_repository,
            purged_cv_repository=PurgedCVRepository(layout, datastore),
            options=options,
            work=work,
        )
    )
    assert first_summary.successful_tasks == 1

    second_summary = _run(
        run_generation(
            pipeline=PurgedCVPipeline(_compact_registry()),
            walk_forward_repository=walk_forward_repository,
            purged_cv_repository=PurgedCVRepository(layout, datastore),
            options=options,
            work=work,
        )
    )
    assert second_summary.successful_tasks == 1
    assert second_summary.skipped_tasks == 0


def test_run_generation_unknown_engine_fails_tasks(tmp_path: Path) -> None:
    """Unknown engine names mark partition tasks as failed."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    _seed_generation_inputs(layout=layout, datastore=datastore)

    options = _options(storage_root=tmp_path, engine="missing-engine")
    walk_forward_repository = WalkForwardRepository(layout, datastore)
    work = discover_work(walk_forward_repository, options)

    summary = _run(
        run_generation(
            pipeline=PurgedCVPipeline(_compact_registry()),
            walk_forward_repository=walk_forward_repository,
            purged_cv_repository=PurgedCVRepository(layout, datastore),
            options=options,
            work=work,
        )
    )
    assert summary.failed_tasks == 1
    assert summary.successful_tasks == 0
    assert summary.status == "FAILED"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_returns_success_for_empty_universe(tmp_path: Path) -> None:
    """main returns 0 when no Walk-Forward partitions are discovered."""
    with patch.object(generate_purged_cv_module, "DEFAULT_STORAGE_ROOT", tmp_path):
        code = _run(main(["--manager", "simple", "--storage-root", str(tmp_path)]))
    assert code == 0


def test_main_validation_error_exit_code() -> None:
    """main returns 1 when build_options raises ValidationError."""
    code = _run(main(["--manager", "simple", "--workers", "0"]))
    assert code == 1


def test_main_returns_success_after_generation(tmp_path: Path) -> None:
    """main returns 0 after successful purged-CV generation."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    _seed_generation_inputs(layout=layout, datastore=datastore)
    with patch.object(generate_purged_cv_module, "DEFAULT_STORAGE_ROOT", tmp_path):
        code = _run(main(["--manager", "simple", "--storage-root", str(tmp_path)]))
    assert code == 0
