"""Unit tests for CQROS walk-forward generation CLI."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

import cqros.cli.generate_walk_forward as generate_walk_forward_module
from cqros.cli.generate_walk_forward import (
    DiscoveredWorkItem,
    WalkForwardGenerationOptions,
    WalkForwardGenerationSummary,
    build_default_engine,
    build_options,
    build_parser,
    build_registry,
    build_walk_forward_pipeline,
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
    STORAGE_DIR_WALK_FORWARD,
)
from cqros.core.exceptions import ValidationError
from cqros.factor_selection import FactorSelectionRepository, FactorSelectionStatus
from cqros.factor_selection.schema import CANONICAL_COLUMN_ORDER, COLUMN_DTYPES
from cqros.storage import ParquetStore, StorageLayout
from cqros.walk_forward import (
    SimpleWalkForwardEngine,
    WalkForwardEngineRegistry,
    WalkForwardPipeline,
    WalkForwardRepository,
)
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
_SELECTION_TIME = 1_700_000_000_000
_FACTOR_NAME = "momentum"
_FACTOR_VERSION = "1.0.0"
_FACTOR_CATEGORY = "price"


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
) -> WalkForwardGenerationOptions:
    """Build WalkForwardGenerationOptions against a temporary storage root."""
    return WalkForwardGenerationOptions(
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


def _factor_selection_frame(
    *,
    factor_name: str = _FACTOR_NAME,
    selection_score: float = 0.12,
    timeframe: str = _TIMEFRAME,
) -> pl.DataFrame:
    """Return a canonical Factor Selection frame for walk-forward generation tests."""
    from cqros.factor_selection.orientation import FACTOR_ORIENTATION_POLICY

    return pl.DataFrame(
        {
            "factor_name": [factor_name],
            "factor_version": [_FACTOR_VERSION],
            "timeframe": [timeframe],
            "selection_time": [_SELECTION_TIME],
            "factor_category": [_FACTOR_CATEGORY],
            "selected": [True],
            "selection_score": [selection_score],
            "selection_rank": [1],
            "selection_reason": ["v1_default_selection"],
            "selection_ic": [0.08],
            "selected_direction": [1],
            "orientation_policy": [FACTOR_ORIENTATION_POLICY],
            "status": [FactorSelectionStatus.SELECTED.value],
        },
        schema=dict(COLUMN_DTYPES),
    ).select(list(CANONICAL_COLUMN_ORDER))


def _seed_generation_inputs(
    *,
    layout: StorageLayout,
    datastore: ParquetStore,
    factor_name: str = _FACTOR_NAME,
    timeframe: str = _TIMEFRAME,
    year: int = _YEAR,
) -> None:
    """Persist a Factor Selection partition needed for walk-forward generation."""
    FactorSelectionRepository(layout, datastore).save(
        _factor_selection_frame(factor_name=factor_name, timeframe=timeframe),
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe=timeframe,
        year=year,
    )


def _canonical_walk_forward_row(
    *,
    selected_factors: int = 1,
    status: str = "PASS",
) -> pl.DataFrame:
    """Build one canonical walk-forward row for stub-engine CLI tests."""
    return pl.DataFrame(
        {
            "strategy_name": ["default_strategy"],
            "strategy_version": ["v1"],
            "timeframe": [_TIMEFRAME],
            "fold_id": [1],
            "train_start": [_SELECTION_TIME],
            "train_end": [_SELECTION_TIME],
            "test_start": [_SELECTION_TIME],
            "test_end": [_SELECTION_TIME],
            "train_rows": [1],
            "test_rows": [1],
            "selected_factors": [selected_factors],
            "model_version": ["v1"],
            "train_score": [0.0],
            "test_score": [0.0],
            "overfit_gap": [0.0],
            "status": [status],
        },
        schema=dict(WF_COLUMN_DTYPES),
    ).select(list(WF_CANONICAL_COLUMN_ORDER))


class _StubWalkForwardEngine:
    """Deterministic walk-forward engine stub for CLI orchestration tests.

    Avoids ``SimpleWalkForwardEngine`` fold mathematics so tests can exercise
    panel discovery/persistence against evaluation-enriched Factor Selection
    inputs.
    """

    def build(self, factor_selection: pl.DataFrame) -> pl.DataFrame:
        """Return one PASS walk-forward row per selected factor."""
        selected = int(factor_selection.filter(pl.col("selected")).height)
        count = selected if selected > 0 else 1
        return _canonical_walk_forward_row(selected_factors=count)


class _StubWalkForwardInputBuilder:
    """Pass-through evaluation adapter for CLI orchestration tests.

    Adds a deterministic ``future_return_1`` column without loading Factors
    or Labels partitions so orchestration tests remain focused on discovery
    and persistence wiring.
    """

    def build(
        self,
        factor_selection: pl.DataFrame,
        **_: object,
    ) -> pl.DataFrame:
        """Return Factor Selection rows enriched with evaluation returns."""
        return factor_selection.with_columns(pl.lit(0.01).alias("future_return_1"))


def _stub_registry() -> WalkForwardEngineRegistry:
    """Build a registry with ``_StubWalkForwardEngine`` under ``simple``."""
    registry = WalkForwardEngineRegistry()
    registry.register(_ENGINE, _StubWalkForwardEngine())
    return registry


def _stub_input_builder() -> _StubWalkForwardInputBuilder:
    """Build the CLI orchestration stub evaluation-input adapter."""
    return _StubWalkForwardInputBuilder()


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


def test_build_options_rejects_non_positive_workers() -> None:
    """build_options rejects workers <= 0 with CLI-GENERATE-WALK-FORWARD-001."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "simple", "--workers", "0"])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-WALK-FORWARD-001"


def test_build_options_rejects_blank_manager() -> None:
    """build_options rejects a blank manager identity with CLI-GENERATE-WALK-FORWARD-004."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "   "])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-WALK-FORWARD-004"


def test_build_options_rejects_blank_engine() -> None:
    """build_options rejects a blank engine identity with CLI-GENERATE-WALK-FORWARD-005."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "simple", "--engine", "   "])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-WALK-FORWARD-005"


def test_build_options_uses_storage_root(tmp_path: Path) -> None:
    """build_options honors an explicit storage root override."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "simple", "--storage-root", str(tmp_path)])
    options = build_options(args)
    assert options.storage_root == tmp_path


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def test_build_default_engine_returns_simple_walk_forward_engine() -> None:
    """build_default_engine returns a SimpleWalkForwardEngine instance."""
    assert isinstance(build_default_engine(), SimpleWalkForwardEngine)


def test_build_registry_contains_simple_engine() -> None:
    """Default registry contains SimpleWalkForwardEngine under 'simple'."""
    registry = build_registry()
    assert registry.exists("simple")
    assert isinstance(registry.get("simple"), SimpleWalkForwardEngine)


def test_build_walk_forward_pipeline_wires_registry() -> None:
    """build_walk_forward_pipeline returns a fully wired WalkForwardPipeline."""
    options = _options(storage_root=Path(DEFAULT_STORAGE_ROOT))
    pipeline = build_walk_forward_pipeline(options)
    assert isinstance(pipeline, WalkForwardPipeline)


# ---------------------------------------------------------------------------
# discover_work
# ---------------------------------------------------------------------------


def test_discover_work_groups_factor_selection_partitions(tmp_path: Path) -> None:
    """discover_work returns sorted work items from the factor selection tier."""
    layout = StorageLayout(tmp_path)
    factor_selection_repository = FactorSelectionRepository(layout, ParquetStore())
    for timeframe, year in (("4h", 2025), ("1h", 2026), ("1h", 2025)):
        factor_selection_repository.save(
            _factor_selection_frame(timeframe=timeframe),
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            timeframe=timeframe,
            year=year,
        )
    work = discover_work(
        factor_selection_repository,
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


def test_discover_work_returns_empty_when_no_factor_selection_partitions(
    tmp_path: Path,
) -> None:
    """discover_work returns empty tuple when no Factor Selection partitions exist."""
    layout = StorageLayout(tmp_path)
    factor_selection_repository = FactorSelectionRepository(layout, ParquetStore())
    work = discover_work(factor_selection_repository, _options(storage_root=tmp_path))
    assert work == ()


# ---------------------------------------------------------------------------
# format_summary
# ---------------------------------------------------------------------------


def test_format_summary_includes_core_fields() -> None:
    """format_summary renders manager, engine, and walk-forward aggregates."""
    summary = WalkForwardGenerationSummary(
        manager=_MANAGER,
        engine=_ENGINE,
        panels=2,
        rows=10,
        selected_factors=7,
        pass_rows=8,
        fail_rows=2,
        successful_tasks=2,
        failed_tasks=1,
        skipped_tasks=0,
        duration_seconds=2.5,
        output_directory=Path("data") / STORAGE_DIR_WALK_FORWARD,
        failed_task_labels=("1h 2026",),
    )
    text = format_summary(summary)
    assert "CQROS Walk-Forward Generation Summary" in text
    assert f"Manager: {_MANAGER}" in text
    assert f"Engine: {_ENGINE}" in text
    assert "Panels: 2" in text
    assert "Rows: 10" in text
    assert "Selected Factors: 7" in text
    assert "Pass Rows: 8" in text
    assert "Fail Rows: 2" in text
    assert "Failed Tasks" in text
    assert "1h 2026" in text
    assert STORAGE_DIR_WALK_FORWARD in text


# ---------------------------------------------------------------------------
# run_generation — empty work
# ---------------------------------------------------------------------------


def test_run_generation_empty_work(tmp_path: Path) -> None:
    """Empty work produces a zeroed success summary."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    options = _options(storage_root=tmp_path)
    registry = build_registry()
    summary = _run(
        run_generation(
            pipeline=WalkForwardPipeline(registry),
            factor_selection_repository=FactorSelectionRepository(layout, datastore),
            walk_forward_repository=WalkForwardRepository(layout, datastore),
            walk_forward_input_builder=_stub_input_builder(),
            options=options,
            work=(),
        )
    )
    assert summary.successful_tasks == 0
    assert summary.failed_tasks == 0
    assert summary.skipped_tasks == 0
    assert summary.rows == 0
    assert summary.selected_factors == 0
    assert summary.pass_rows == 0
    assert summary.fail_rows == 0
    assert summary.panels == 0
    assert summary.output_directory == tmp_path / STORAGE_DIR_WALK_FORWARD


# ---------------------------------------------------------------------------
# run_generation — persists walk-forward ledgers
# ---------------------------------------------------------------------------


def test_run_generation_persists_walk_forward_partitions(tmp_path: Path) -> None:
    """Generation loads Factor Selection inputs, runs the pipeline, and persists output."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()

    _seed_generation_inputs(layout=layout, datastore=datastore, year=2025)
    _seed_generation_inputs(
        layout=layout,
        datastore=datastore,
        factor_name="rsi",
        year=2026,
    )

    options = _options(storage_root=tmp_path)
    factor_selection_repository = FactorSelectionRepository(layout, datastore)
    work = discover_work(factor_selection_repository, options)

    summary = _run(
        run_generation(
            pipeline=WalkForwardPipeline(_stub_registry()),
            factor_selection_repository=factor_selection_repository,
            walk_forward_repository=WalkForwardRepository(layout, datastore),
            walk_forward_input_builder=_stub_input_builder(),
            options=options,
            work=work,
        )
    )
    assert summary.successful_tasks == 2
    assert summary.failed_tasks == 0
    assert summary.rows == 2
    assert summary.selected_factors == 2
    assert summary.pass_rows == 2
    assert summary.fail_rows == 0
    assert summary.panels == 2

    walk_forward_repo = WalkForwardRepository(layout, datastore)
    assert walk_forward_repo.exists(
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe=_TIMEFRAME,
        year=2025,
    )
    assert walk_forward_repo.exists(
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe=_TIMEFRAME,
        year=2026,
    )


# ---------------------------------------------------------------------------
# run_generation — skip existing without overwrite
# ---------------------------------------------------------------------------


def test_run_generation_skips_existing_without_overwrite(tmp_path: Path) -> None:
    """Existing walk-forward partitions are skipped when overwrite is False."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    _seed_generation_inputs(layout=layout, datastore=datastore)

    options = _options(storage_root=tmp_path, overwrite=False)
    factor_selection_repository = FactorSelectionRepository(layout, datastore)
    work = discover_work(factor_selection_repository, options)

    first_summary = _run(
        run_generation(
            pipeline=WalkForwardPipeline(_stub_registry()),
            factor_selection_repository=factor_selection_repository,
            walk_forward_repository=WalkForwardRepository(layout, datastore),
            walk_forward_input_builder=_stub_input_builder(),
            options=options,
            work=work,
        )
    )
    assert first_summary.successful_tasks == 1

    second_summary = _run(
        run_generation(
            pipeline=WalkForwardPipeline(_stub_registry()),
            factor_selection_repository=factor_selection_repository,
            walk_forward_repository=WalkForwardRepository(layout, datastore),
            walk_forward_input_builder=_stub_input_builder(),
            options=options,
            work=work,
        )
    )
    assert second_summary.skipped_tasks == 1
    assert second_summary.successful_tasks == 0
    assert second_summary.failed_tasks == 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_returns_success_for_empty_universe(tmp_path: Path) -> None:
    """main returns 0 when no Factor Selection partitions are discovered."""
    with patch.object(generate_walk_forward_module, "DEFAULT_STORAGE_ROOT", tmp_path):
        code = _run(main(["--manager", "simple", "--storage-root", str(tmp_path)]))
    assert code == 0


def test_main_validation_error_exit_code() -> None:
    """main returns 1 when build_options raises ValidationError."""
    code = _run(main(["--manager", "simple", "--workers", "0"]))
    assert code == 1


def test_main_returns_success_after_generation(tmp_path: Path) -> None:
    """main returns 0 after successful walk-forward generation."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    _seed_generation_inputs(layout=layout, datastore=datastore)
    with (
        patch.object(generate_walk_forward_module, "DEFAULT_STORAGE_ROOT", tmp_path),
        patch.object(
            generate_walk_forward_module,
            "build_default_engine",
            return_value=_StubWalkForwardEngine(),
        ),
        patch.object(
            generate_walk_forward_module,
            "WalkForwardInputBuilder",
            return_value=_stub_input_builder(),
        ),
    ):
        code = _run(main(["--manager", "simple", "--storage-root", str(tmp_path)]))
    assert code == 0


def test_run_generation_real_engine_with_labels_enrichment(tmp_path: Path) -> None:
    """Real SimpleWalkForwardEngine succeeds after Labels enrichment."""
    from cqros.factors import FactorsRepository, FactorStatus
    from cqros.factors.schema import (
        CANONICAL_COLUMN_ORDER as FACTOR_COLUMNS,
    )
    from cqros.factors.schema import (
        COLUMN_DTYPES as FACTOR_DTYPES,
    )
    from cqros.labels.schema import (
        CANONICAL_COLUMN_ORDER as LABEL_COLUMNS,
    )
    from cqros.labels.schema import (
        COLUMN_DTYPES as LABEL_DTYPES,
    )
    from cqros.storage import LabelRepository
    from cqros.walk_forward import WalkForwardInputBuilder

    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    _seed_generation_inputs(layout=layout, datastore=datastore)

    open_times = [1_700_000_000_000 + (index * 3_600_000) for index in range(5)]
    factors = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 5,
            "timeframe": [_TIMEFRAME] * 5,
            "open_time": open_times,
            "factor_name": [_FACTOR_NAME] * 5,
            "factor_version": [_FACTOR_VERSION] * 5,
            "factor_category": [_FACTOR_CATEGORY] * 5,
            "factor_group": ["alpha"] * 5,
            "factor_value": [0.1, 0.2, 0.3, 0.4, 0.5],
            "lookback": [20] * 5,
            "prediction_horizon": [1] * 5,
            "enabled": [True] * 5,
            "status": [FactorStatus.ACTIVE.value] * 5,
        },
        schema=dict(FACTOR_DTYPES),
    ).select(list(FACTOR_COLUMNS))
    labels = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 5,
            "timeframe": [_TIMEFRAME] * 5,
            "open_time": open_times,
            "future_return_1": [0.01, 0.02, -0.01, 0.03, 0.04],
            "future_return_5": [0.05] * 5,
            "future_return_10": [0.10] * 5,
            "future_return_20": [0.20] * 5,
            "direction_1": [1] * 5,
            "direction_5": [1] * 5,
            "direction_10": [0] * 5,
            "direction_20": [0] * 5,
        },
        schema=dict(LABEL_DTYPES),
    ).select(list(LABEL_COLUMNS))
    FactorsRepository(layout, datastore).save(
        factors,
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol="BTCUSDT",
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    LabelRepository(layout, datastore).save(
        labels,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol="BTCUSDT",
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )

    options = _options(storage_root=tmp_path, overwrite=True)
    factor_selection_repository = FactorSelectionRepository(layout, datastore)
    work = discover_work(factor_selection_repository, options)
    summary = _run(
        run_generation(
            pipeline=WalkForwardPipeline(build_registry()),
            factor_selection_repository=factor_selection_repository,
            walk_forward_repository=WalkForwardRepository(layout, datastore),
            walk_forward_input_builder=WalkForwardInputBuilder(
                FactorsRepository(layout, datastore),
                LabelRepository(layout, datastore),
            ),
            options=options,
            work=work,
        )
    )
    assert summary.failed_tasks == 0
    assert summary.successful_tasks == 1
    assert summary.rows >= 1
    assert WalkForwardRepository(layout, datastore).exists(
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )


def test_build_parser_defaults_to_full_panel_and_accepts_memory_efficient() -> None:
    """The bounded path is explicit opt-in and does not change the default."""
    parser = build_parser()
    default_args = parser.parse_args(["--manager", _MANAGER])
    bounded_args = parser.parse_args(
        [
            "--manager",
            _MANAGER,
            "--execution-mode",
            "memory_efficient",
            "--workers",
            "1",
        ]
    )
    assert default_args.execution_mode == "full_panel"
    assert build_options(bounded_args).execution_mode == "memory_efficient"


def test_memory_efficient_rejects_parallel_workers() -> None:
    """Unsupported bounded-mode concurrency fails before execution."""
    args = build_parser().parse_args(
        [
            "--manager",
            _MANAGER,
            "--execution-mode",
            "memory_efficient",
            "--workers",
            "2",
        ]
    )
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-WALK-FORWARD-008"


def test_memory_efficient_failure_has_no_fallback_and_preserves_output(
    tmp_path: Path,
) -> None:
    """A bounded-path failure neither invokes full-panel input nor replaces output."""

    class FailingMemoryExecutor:
        """Memory executor stub that fails before persistence."""

        def execute(self, *_: object, **__: object) -> pl.DataFrame:
            """Raise the controlled bounded-path failure."""
            raise RuntimeError("controlled memory-efficient failure")

    class ForbiddenFullPanelBuilder:
        """Full-panel stub that records any prohibited fallback."""

        def build(self, *_: object, **__: object) -> pl.DataFrame:
            """Fail if the canonical adapter is invoked."""
            raise AssertionError("full-panel fallback was invoked")

    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    _seed_generation_inputs(layout=layout, datastore=datastore)
    repository = WalkForwardRepository(layout, datastore)
    repository.save(
        _canonical_walk_forward_row(selected_factors=7),
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    output_path = layout.walk_forward_path(
        _MANAGER,
        EXCHANGE_BINANCE,
        MARKET_USDT_PERPETUAL,
        _TIMEFRAME,
        _YEAR,
    )
    before = output_path.read_bytes()
    options = _options(storage_root=tmp_path, overwrite=True, workers=1)
    options = WalkForwardGenerationOptions(
        storage_root=options.storage_root,
        manager=options.manager,
        engine=options.engine,
        timeframes=options.timeframes,
        years=options.years,
        overwrite=options.overwrite,
        workers=options.workers,
        verbose=options.verbose,
        debug=options.debug,
        execution_mode="memory_efficient",
    )
    factor_selection_repository = FactorSelectionRepository(layout, datastore)
    summary = _run(
        run_generation(
            pipeline=WalkForwardPipeline(_stub_registry()),
            factor_selection_repository=factor_selection_repository,
            walk_forward_repository=repository,
            walk_forward_input_builder=ForbiddenFullPanelBuilder(),  # type: ignore[arg-type]
            options=options,
            work=discover_work(factor_selection_repository, options),
            memory_efficient_executor=FailingMemoryExecutor(),  # type: ignore[arg-type]
        )
    )
    assert summary.failed_tasks == 1
    assert output_path.read_bytes() == before
