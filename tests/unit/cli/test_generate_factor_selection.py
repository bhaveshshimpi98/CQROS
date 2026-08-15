"""Unit tests for CQROS factor selection generation CLI."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

import cqros.cli.generate_factor_selection as generate_factor_selection_module
from cqros.cli.generate_factor_selection import (
    DiscoveredWorkItem,
    FactorSelectionGenerationOptions,
    FactorSelectionGenerationSummary,
    build_default_engine,
    build_factor_selection_pipeline,
    build_observation_source,
    build_options,
    build_parser,
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
    STORAGE_DIR_FACTOR_SELECTION,
)
from cqros.core.exceptions import ValidationError
from cqros.factor_selection import (
    DETAILED_AUDIT_COLUMNS,
    FactorSelectionEngineRegistry,
    FactorSelectionExecutionMode,
    FactorSelectionPipeline,
    FactorSelectionRepository,
    FactorSelectionStatus,
    FactorsObservationLoader,
    MemoryEfficientFactorsObservationLoader,
    SimpleFactorSelectionEngine,
    combined_detailed_csv_path,
    detailed_csv_path,
)
from cqros.factor_validation import (
    FactorValidationRepository,
    FactorValidationStatus,
)
from cqros.factor_validation.schema import FACTOR_VALIDATION_SCHEMA
from cqros.storage import ParquetStore, StorageLayout

_MANAGER = "simple"
_ENGINE = "simple"
_TIMEFRAME = "1h"
_YEAR = 2026
_VALIDATION_TIME = 1_700_000_000_000
_VALIDATION_START_TIME = 1_699_913_600_000
_VALIDATION_END_TIME = 1_700_000_000_000
_FACTOR_NAME = "momentum"
_FACTOR_VERSION = "1.0.0"
_FACTOR_CATEGORY = "price"
_DATASET_VERSION = "dataset-v1"
_LABEL_VERSION = "label-v1"

_STRONG_IC = 0.12
_STRONG_RANK_IC = 0.10
_STRONG_ICIR = 0.80
_STRONG_IC_STD = 0.15
_STRONG_P_VALUE = 0.01
_STRONG_IC_T_STAT = 3.0
_STRONG_IC_DECAY = 0.70
_STRONG_TURNOVER = 0.20
_STRONG_MONOTONICITY_SCORE = 0.80
_STRONG_QUANTILE_SPREAD = 0.05
_STRONG_OBSERVATIONS = 200
_STRONG_IC_OBSERVATIONS = 150


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def _options(
    *,
    storage_root: Path,
    manager: str = _MANAGER,
    engine: str = _ENGINE,
    top_n: int = 20,
    candidate_n: int = 40,
    max_factor_correlation: float = 0.90,
    min_overlap: int = 500,
    timeframes: tuple[str, ...] | None = None,
    years: tuple[int, ...] | None = None,
    overwrite: bool = False,
    export_detailed_csv: bool = False,
    workers: int = ResearchConfig().worker_count,
    execution_mode: FactorSelectionExecutionMode = FactorSelectionExecutionMode.MEMORY_EFFICIENT,
    factor_batch_size: int = 1,
    verbose: bool = False,
    debug: bool = False,
) -> FactorSelectionGenerationOptions:
    """Build FactorSelectionGenerationOptions against a temporary storage root."""
    return FactorSelectionGenerationOptions(
        storage_root=storage_root,
        manager=manager,
        engine=engine,
        top_n=top_n,
        candidate_n=candidate_n,
        max_factor_correlation=max_factor_correlation,
        min_overlap=min_overlap,
        timeframes=timeframes,
        years=years,
        overwrite=overwrite,
        export_detailed_csv=export_detailed_csv,
        workers=workers,
        execution_mode=execution_mode,
        factor_batch_size=factor_batch_size,
        verbose=verbose,
        debug=debug,
    )


def _factor_validation_frame(
    *,
    factor_name: str = _FACTOR_NAME,
    information_coefficient: float = _STRONG_IC,
    status: str = FactorValidationStatus.PASS.value,
    timeframe: str = _TIMEFRAME,
) -> pl.DataFrame:
    """Return a canonical Factor Validation frame for selection generation tests."""
    return pl.DataFrame(
        {
            "factor_name": [factor_name],
            "factor_version": [_FACTOR_VERSION],
            "timeframe": [timeframe],
            "validation_time": [_VALIDATION_TIME],
            "factor_category": [_FACTOR_CATEGORY],
            "dataset_version": [_DATASET_VERSION],
            "label_version": [_LABEL_VERSION],
            "validation_start_time": [_VALIDATION_START_TIME],
            "validation_end_time": [_VALIDATION_END_TIME],
            "information_coefficient": [information_coefficient],
            "rank_information_coefficient": [_STRONG_RANK_IC],
            "ic_information_ratio": [_STRONG_ICIR],
            "ic_std": [_STRONG_IC_STD],
            "ic_p_value": [_STRONG_P_VALUE],
            "ic_t_stat": [_STRONG_IC_T_STAT],
            "ic_decay": [_STRONG_IC_DECAY],
            "turnover": [_STRONG_TURNOVER],
            "monotonicity_score": [_STRONG_MONOTONICITY_SCORE],
            "quantile_spread": [_STRONG_QUANTILE_SPREAD],
            "observations": [_STRONG_OBSERVATIONS],
            "ic_observations": [_STRONG_IC_OBSERVATIONS],
            "status": [status],
        },
        schema=FACTOR_VALIDATION_SCHEMA,
    )


def _seed_generation_inputs(
    *,
    layout: StorageLayout,
    datastore: ParquetStore,
    factor_name: str = _FACTOR_NAME,
    status: str = FactorValidationStatus.PASS.value,
    timeframe: str = _TIMEFRAME,
    year: int = _YEAR,
) -> None:
    """Persist a Factor Validation partition needed for selection generation."""
    FactorValidationRepository(layout, datastore).save(
        _factor_validation_frame(factor_name=factor_name, status=status, timeframe=timeframe),
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe=timeframe,
        year=year,
    )


def _load_selection_partition(
    *,
    layout: StorageLayout,
    datastore: ParquetStore,
    timeframe: str = _TIMEFRAME,
    year: int = _YEAR,
) -> pl.DataFrame:
    """Load the generated factor selection panel partition."""
    return FactorSelectionRepository(layout, datastore).load(
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe=timeframe,
        year=year,
    )


def _run_generation_for_seeded_inputs(
    *,
    tmp_path: Path,
    status: str,
) -> tuple[FactorSelectionGenerationSummary, pl.DataFrame]:
    """Seed one validation partition, run generation, and return summary plus output."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    _seed_generation_inputs(layout=layout, datastore=datastore, status=status)

    options = _options(storage_root=tmp_path)
    factor_validation_repository = FactorValidationRepository(layout, datastore)
    work = discover_work(factor_validation_repository, options)
    summary = _run(
        run_generation(
            layout=layout,
            factor_validation_repository=factor_validation_repository,
            factor_selection_repository=FactorSelectionRepository(layout, datastore),
            options=options,
            work=work,
        )
    )
    output = _load_selection_partition(layout=layout, datastore=datastore)
    return summary, output


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
            "--top-n",
            "10",
            "--candidate-n",
            "40",
            "--max-factor-correlation",
            "0.90",
            "--min-overlap",
            "500",
            "--workers",
            "4",
            "--overwrite",
            "--export-detailed-csv",
            "--execution-mode",
            "full_panel",
            "--factor-batch-size",
            "4",
            "--storage-root",
            str(tmp_path),
        ]
    )
    assert args.manager == "ledger"
    assert args.engine == "custom"
    assert args.top_n == 10
    assert args.candidate_n == 40
    assert args.max_factor_correlation == 0.90
    assert args.min_overlap == 500
    assert args.workers == 4
    assert args.overwrite is True
    assert args.export_detailed_csv is True
    assert args.execution_mode == "full_panel"
    assert args.factor_batch_size == 4
    assert args.storage_root == tmp_path


def test_build_parser_defaults_redundancy_parameters() -> None:
    """Parser defaults redundancy parameters to Phase 3B locked defaults."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "simple"])
    assert args.top_n == 20
    assert args.candidate_n == 40
    assert args.max_factor_correlation == 0.90
    assert args.min_overlap == 500


def test_build_parser_defaults_execution_mode_memory_efficient() -> None:
    """Parser defaults --execution-mode to memory_efficient and batch size 1."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "simple"])
    assert args.execution_mode == FactorSelectionExecutionMode.MEMORY_EFFICIENT.value
    assert args.factor_batch_size == 1
    options = build_options(args)
    assert options.execution_mode is FactorSelectionExecutionMode.MEMORY_EFFICIENT
    assert options.factor_batch_size == 1


def test_build_parser_accepts_full_panel_execution_mode(tmp_path: Path) -> None:
    """Parser and options accept legacy full_panel execution mode."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "--manager",
            "simple",
            "--execution-mode",
            "full_panel",
            "--factor-batch-size",
            "8",
            "--storage-root",
            str(tmp_path),
        ]
    )
    assert args.execution_mode == FactorSelectionExecutionMode.FULL_PANEL.value
    options = build_options(args)
    assert options.execution_mode is FactorSelectionExecutionMode.FULL_PANEL
    assert options.factor_batch_size == 8


def test_build_options_rejects_non_positive_factor_batch_size() -> None:
    """build_options rejects factor_batch_size < 1."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "simple", "--factor-batch-size", "0"])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-FACTOR-SELECTION-012"


def test_build_observation_source_memory_efficient_and_full_panel(tmp_path: Path) -> None:
    """CLI observation-source factory remains callable for both execution modes."""
    layout = StorageLayout(tmp_path / "data")
    memory_source = build_observation_source(
        layout,
        manager=_MANAGER,
        year=_YEAR,
        execution_mode=FactorSelectionExecutionMode.MEMORY_EFFICIENT,
        factor_batch_size=2,
        storage_root=tmp_path,
    )
    full_source = build_observation_source(
        layout,
        manager=_MANAGER,
        year=_YEAR,
        execution_mode=FactorSelectionExecutionMode.FULL_PANEL,
        factor_batch_size=2,
        storage_root=tmp_path,
    )
    assert isinstance(memory_source, MemoryEfficientFactorsObservationLoader)
    assert memory_source.factor_batch_size == 2
    assert isinstance(full_source, FactorsObservationLoader)


def test_build_options_rejects_candidate_n_less_than_top_n() -> None:
    """build_options rejects candidate_n < top_n."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "simple", "--top-n", "20", "--candidate-n", "10"])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert "candidate_n" in str(exc_info.value).lower() or exc_info.value.error_code is not None


def test_build_options_defaults_export_detailed_csv_false(tmp_path: Path) -> None:
    """build_options defaults export_detailed_csv to False when the flag is omitted."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "simple", "--storage-root", str(tmp_path)])
    options = build_options(args)
    assert options.export_detailed_csv is False


def test_build_parser_defaults_top_n_to_twenty() -> None:
    """Parser defaults --top-n to DEFAULT_TOP_N when omitted."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "simple"])
    assert args.top_n == 20


def test_build_options_rejects_non_positive_top_n() -> None:
    """build_options rejects top_n <= 0 with CLI-GENERATE-FACTOR-SELECTION-007."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "simple", "--top-n", "0"])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-FACTOR-SELECTION-007"


def test_build_options_rejects_non_positive_workers() -> None:
    """build_options rejects workers <= 0 with CLI-GENERATE-FACTOR-SELECTION-001."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "simple", "--workers", "0"])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-FACTOR-SELECTION-001"


def test_build_options_rejects_blank_manager() -> None:
    """build_options rejects a blank manager with CLI-GENERATE-FACTOR-SELECTION-004."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "   "])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-FACTOR-SELECTION-004"


def test_build_options_rejects_blank_engine() -> None:
    """build_options rejects a blank engine with CLI-GENERATE-FACTOR-SELECTION-005."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "simple", "--engine", "   "])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-FACTOR-SELECTION-005"


def test_build_options_uses_storage_root(tmp_path: Path) -> None:
    """build_options honors an explicit storage root override."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "simple", "--storage-root", str(tmp_path)])
    options = build_options(args)
    assert options.storage_root == tmp_path


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def test_build_default_engine_returns_simple_factor_selection_engine() -> None:
    """build_default_engine returns a SimpleFactorSelectionEngine instance."""
    engine = build_default_engine()
    assert isinstance(engine, SimpleFactorSelectionEngine)
    assert engine.top_n == 20


def test_build_default_engine_honors_top_n() -> None:
    """build_default_engine passes top_n through to SimpleFactorSelectionEngine."""
    engine = build_default_engine(top_n=10)
    assert isinstance(engine, SimpleFactorSelectionEngine)
    assert engine.top_n == 10


def test_build_registry_contains_simple_engine() -> None:
    """Default registry contains SimpleFactorSelectionEngine under 'simple'."""
    registry = build_registry()
    assert registry.exists("simple")
    assert isinstance(registry.get("simple"), SimpleFactorSelectionEngine)


def test_build_factor_selection_pipeline_wires_registry() -> None:
    """build_factor_selection_pipeline returns a wired FactorSelectionPipeline."""
    options = _options(storage_root=Path(DEFAULT_STORAGE_ROOT))
    pipeline = build_factor_selection_pipeline(options)
    assert isinstance(pipeline, FactorSelectionPipeline)


# ---------------------------------------------------------------------------
# discover_work
# ---------------------------------------------------------------------------


def test_discover_work_groups_factor_validation_partitions(tmp_path: Path) -> None:
    """discover_work returns sorted work items from the factor_validation tier."""
    layout = StorageLayout(tmp_path)
    factor_validation_repository = FactorValidationRepository(layout, ParquetStore())
    for timeframe, year in (("4h", 2025), ("1h", 2026), ("1h", 2025)):
        factor_validation_repository.save(
            _factor_validation_frame(),
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            timeframe=timeframe,
            year=year,
        )
    work = discover_work(
        factor_validation_repository,
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


def test_discover_work_returns_empty_when_no_factor_validation_partitions(
    tmp_path: Path,
) -> None:
    """discover_work returns empty tuple when no Factor Validation partitions exist."""
    layout = StorageLayout(tmp_path)
    factor_validation_repository = FactorValidationRepository(layout, ParquetStore())
    work = discover_work(factor_validation_repository, _options(storage_root=tmp_path))
    assert work == ()


# ---------------------------------------------------------------------------
# format_summary
# ---------------------------------------------------------------------------


def test_format_summary_includes_core_fields() -> None:
    """format_summary renders manager, engine, and factor selection aggregates."""
    summary = FactorSelectionGenerationSummary(
        manager=_MANAGER,
        engine=_ENGINE,
        panels=2,
        rows=10,
        selected_rows=8,
        rejected_status_rows=2,
        successful_tasks=2,
        failed_tasks=1,
        skipped_tasks=0,
        duration_seconds=2.5,
        output_directory=Path("data") / STORAGE_DIR_FACTOR_SELECTION,
        failed_task_labels=("1h 2026",),
    )
    text = format_summary(summary)
    assert "CQROS Factor Selection Generation Summary" in text
    assert f"Manager: {_MANAGER}" in text
    assert f"Engine: {_ENGINE}" in text
    assert "Panels: 2" in text
    assert "Rows: 10" in text
    assert "Selected: 8" in text
    assert "Rejected Status: 2" in text
    assert "Failed Tasks" in text
    assert "1h 2026" in text
    assert STORAGE_DIR_FACTOR_SELECTION in text


# ---------------------------------------------------------------------------
# run_generation — empty work
# ---------------------------------------------------------------------------


def test_run_generation_empty_work(tmp_path: Path) -> None:
    """Empty work produces a zeroed success summary."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    options = _options(storage_root=tmp_path)
    summary = _run(
        run_generation(
            layout=layout,
            factor_validation_repository=FactorValidationRepository(layout, datastore),
            factor_selection_repository=FactorSelectionRepository(layout, datastore),
            options=options,
            work=(),
        )
    )
    assert summary.successful_tasks == 0
    assert summary.failed_tasks == 0
    assert summary.skipped_tasks == 0
    assert summary.rows == 0
    assert summary.selected_rows == 0
    assert summary.rejected_status_rows == 0
    assert summary.panels == 0
    assert summary.output_directory == tmp_path / STORAGE_DIR_FACTOR_SELECTION


# ---------------------------------------------------------------------------
# run_generation — persists factor selection ledgers
# ---------------------------------------------------------------------------


def test_run_generation_persists_factor_selection_partitions(tmp_path: Path) -> None:
    """Generation loads Factor Validation inputs, runs the pipeline, and persists output."""
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
    factor_validation_repository = FactorValidationRepository(layout, datastore)
    work = discover_work(factor_validation_repository, options)

    registry = FactorSelectionEngineRegistry()
    registry.register(_ENGINE, SimpleFactorSelectionEngine())
    summary = _run(
        run_generation(
            layout=layout,
            factor_validation_repository=factor_validation_repository,
            factor_selection_repository=FactorSelectionRepository(layout, datastore),
            options=options,
            work=work,
        )
    )
    assert summary.successful_tasks == 2
    assert summary.failed_tasks == 0
    assert summary.rows == 2
    assert summary.selected_rows == 2
    assert summary.rejected_status_rows == 0
    assert summary.panels == 2

    factor_selection_repo = FactorSelectionRepository(layout, datastore)
    assert factor_selection_repo.exists(
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe=_TIMEFRAME,
        year=2025,
    )
    assert factor_selection_repo.exists(
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe=_TIMEFRAME,
        year=2026,
    )


# ---------------------------------------------------------------------------
# run_generation — validation status eligibility
# ---------------------------------------------------------------------------


def test_run_generation_pass_status_generates_selected_selection_file(
    tmp_path: Path,
) -> None:
    """PASS validation rows generate a selection file with SELECTED status."""
    summary, output = _run_generation_for_seeded_inputs(
        tmp_path=tmp_path,
        status=FactorValidationStatus.PASS.value,
    )
    assert summary.successful_tasks == 1
    assert summary.failed_tasks == 0
    assert summary.rows == 1
    assert summary.selected_rows == 1
    assert summary.rejected_status_rows == 0
    assert output.height == 1
    assert output["selected"].to_list() == [True]
    assert output["status"].to_list() == [FactorSelectionStatus.SELECTED.value]
    assert "SKIPPED" not in output["status"].to_list()


def test_run_generation_fail_status_is_scored_by_rank(tmp_path: Path) -> None:
    """FAIL validation rows are scored/ranked and selected when inside Top-N."""
    summary, output = _run_generation_for_seeded_inputs(
        tmp_path=tmp_path,
        status=FactorValidationStatus.FAIL.value,
    )
    assert summary.successful_tasks == 1
    assert summary.failed_tasks == 0
    assert summary.rows == 1
    assert summary.selected_rows == 1
    assert summary.rejected_status_rows == 0
    assert output.height == 1
    assert output["selected"].to_list() == [True]
    assert output["status"].to_list() == [FactorSelectionStatus.SELECTED.value]
    assert output["selection_reason"].to_list() == ["top_n"]
    assert "SKIPPED" not in output["status"].to_list()


def test_run_generation_skipped_status_is_scored_by_rank(tmp_path: Path) -> None:
    """SKIPPED validation rows are scored/ranked and selected when inside Top-N."""
    summary, output = _run_generation_for_seeded_inputs(
        tmp_path=tmp_path,
        status=FactorValidationStatus.SKIPPED.value,
    )
    assert summary.successful_tasks == 1
    assert summary.failed_tasks == 0
    assert summary.rows == 1
    assert summary.selected_rows == 1
    assert summary.rejected_status_rows == 0
    assert output.height == 1
    assert output["selected"].to_list() == [True]
    assert output["status"].to_list() == [FactorSelectionStatus.SELECTED.value]
    assert output["selection_reason"].to_list() == ["top_n"]
    assert "SKIPPED" not in output["status"].to_list()


def test_run_generation_selection_status_is_selected_or_rejected_only(
    tmp_path: Path,
) -> None:
    """Generated selection status values are SELECTED or REJECTED, never SKIPPED."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    FactorValidationRepository(layout, datastore).save(
        pl.concat(
            [
                _factor_validation_frame(
                    factor_name="pass_factor",
                    status=FactorValidationStatus.PASS.value,
                ),
                _factor_validation_frame(
                    factor_name="fail_factor",
                    status=FactorValidationStatus.FAIL.value,
                ),
                _factor_validation_frame(
                    factor_name="skipped_factor",
                    status=FactorValidationStatus.SKIPPED.value,
                ),
            ]
        ),
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )

    options = _options(storage_root=tmp_path)
    factor_validation_repository = FactorValidationRepository(layout, datastore)
    work = discover_work(factor_validation_repository, options)
    summary = _run(
        run_generation(
            layout=layout,
            factor_validation_repository=factor_validation_repository,
            factor_selection_repository=FactorSelectionRepository(layout, datastore),
            options=options,
            work=work,
        )
    )
    output = _load_selection_partition(layout=layout, datastore=datastore)

    assert summary.successful_tasks == 1
    assert summary.selected_rows == 3
    assert summary.rejected_status_rows == 0
    assert set(output["status"].to_list()) == {
        FactorSelectionStatus.SELECTED.value,
    }
    assert "SKIPPED" not in output["status"].to_list()


# ---------------------------------------------------------------------------
# run_generation — skip existing without overwrite
# ---------------------------------------------------------------------------


def test_run_generation_skips_existing_without_overwrite(tmp_path: Path) -> None:
    """Existing factor selection partitions are skipped when overwrite is False."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    _seed_generation_inputs(layout=layout, datastore=datastore)

    options = _options(storage_root=tmp_path, overwrite=False)
    factor_validation_repository = FactorValidationRepository(layout, datastore)
    work = discover_work(factor_validation_repository, options)

    first_summary = _run(
        run_generation(
            layout=layout,
            factor_validation_repository=factor_validation_repository,
            factor_selection_repository=FactorSelectionRepository(layout, datastore),
            options=options,
            work=work,
        )
    )
    assert first_summary.successful_tasks == 1

    second_summary = _run(
        run_generation(
            layout=layout,
            factor_validation_repository=factor_validation_repository,
            factor_selection_repository=FactorSelectionRepository(layout, datastore),
            options=options,
            work=work,
        )
    )
    assert second_summary.skipped_tasks == 1
    assert second_summary.successful_tasks == 0
    assert second_summary.failed_tasks == 0


def test_run_generation_exports_detailed_csv(tmp_path: Path) -> None:
    """--export-detailed-csv writes per-timeframe and combined audit CSVs."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    _seed_generation_inputs(layout=layout, datastore=datastore, timeframe="1h")
    _seed_generation_inputs(layout=layout, datastore=datastore, timeframe="4h")

    options = _options(storage_root=tmp_path, export_detailed_csv=True, top_n=20)
    factor_validation_repository = FactorValidationRepository(layout, datastore)
    work = discover_work(factor_validation_repository, options)
    summary = _run(
        run_generation(
            layout=layout,
            factor_validation_repository=factor_validation_repository,
            factor_selection_repository=FactorSelectionRepository(layout, datastore),
            options=options,
            work=work,
        )
    )
    assert summary.successful_tasks == 2

    path_1h = detailed_csv_path(
        tmp_path,
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe="1h",
        year=_YEAR,
    )
    path_4h = detailed_csv_path(
        tmp_path,
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe="4h",
        year=_YEAR,
    )
    combined = combined_detailed_csv_path(
        tmp_path,
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
    )
    assert path_1h.is_file()
    assert path_4h.is_file()
    assert combined.is_file()

    loaded_1h = pl.read_csv(path_1h)
    loaded_combined = pl.read_csv(combined)
    assert list(loaded_1h.columns) == list(DETAILED_AUDIT_COLUMNS)
    assert loaded_1h.height == 1
    assert loaded_combined.height == 2
    assert set(loaded_combined["timeframe"].to_list()) == {"1h", "4h"}
    # Parquet canonical partition still exists.
    assert _load_selection_partition(layout=layout, datastore=datastore, timeframe="1h").height == 1


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_returns_success_for_empty_universe(tmp_path: Path) -> None:
    """main returns 0 when no Factor Validation partitions are discovered."""
    with patch.object(generate_factor_selection_module, "DEFAULT_STORAGE_ROOT", tmp_path):
        code = _run(main(["--manager", "simple", "--storage-root", str(tmp_path)]))
    assert code == 0


def test_main_validation_error_exit_code() -> None:
    """main returns 1 when build_options raises ValidationError."""
    code = _run(main(["--manager", "simple", "--workers", "0"]))
    assert code == 1


def test_main_returns_success_after_generation(tmp_path: Path) -> None:
    """main returns 0 after successful factor selection generation."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    _seed_generation_inputs(layout=layout, datastore=datastore)
    with patch.object(generate_factor_selection_module, "DEFAULT_STORAGE_ROOT", tmp_path):
        code = _run(main(["--manager", "simple", "--storage-root", str(tmp_path)]))
    assert code == 0


# ---------------------------------------------------------------------------
# Eligibility policy injection — CLI / builder wiring
# ---------------------------------------------------------------------------


def test_build_default_engine_injects_eligibility_policy_by_default() -> None:
    """build_default_engine must produce an engine that carries a FactorEligibilityPolicy."""
    from cqros.factor_selection import FactorEligibilityPolicy

    engine = build_default_engine()
    assert isinstance(engine, SimpleFactorSelectionEngine)
    assert isinstance(engine.eligibility_policy, FactorEligibilityPolicy)


def test_build_default_engine_accepts_explicit_policy() -> None:
    """build_default_engine forwards an explicitly supplied policy to the engine."""
    from cqros.factor_selection import FactorEligibilityPolicy

    policy = FactorEligibilityPolicy()
    engine = build_default_engine(eligibility_policy=policy)
    assert engine.eligibility_policy is policy


def test_build_registry_engine_carries_eligibility_policy() -> None:
    """Engines created through build_registry must embed an eligibility policy."""
    from cqros.factor_selection import FactorEligibilityPolicy

    registry = build_registry()
    engine = registry.get("simple")
    assert isinstance(engine, SimpleFactorSelectionEngine)
    assert isinstance(engine.eligibility_policy, FactorEligibilityPolicy)


def test_build_factor_selection_pipeline_engine_carries_eligibility_policy(
    tmp_path: Path,
) -> None:
    """Engines in the pipeline produced by build_factor_selection_pipeline carry a policy."""
    from cqros.factor_selection import FactorEligibilityPolicy

    options = _options(storage_root=tmp_path)
    pipeline = build_factor_selection_pipeline(options)
    engine = pipeline._registry.get("simple")  # noqa: SLF001
    assert isinstance(engine, SimpleFactorSelectionEngine)
    assert isinstance(engine.eligibility_policy, FactorEligibilityPolicy)
