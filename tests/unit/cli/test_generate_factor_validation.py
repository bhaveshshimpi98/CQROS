"""Unit tests for CQROS factor validation generation CLI."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

import cqros.cli.generate_factor_validation as generate_factor_validation_module
from cqros.cli.generate_factor_validation import (
    DiscoveredWorkItem,
    FactorValidationGenerationOptions,
    FactorValidationGenerationSummary,
    FactorValidationTaskResult,
    build_default_engine,
    build_factor_validation_pipeline,
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
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
    STORAGE_DIR_FACTOR_VALIDATION,
)
from cqros.core.exceptions import ValidationError
from cqros.factor_validation import (
    FactorValidationEngineRegistry,
    FactorValidationExecutionMode,
    FactorValidationPipeline,
    FactorValidationRepository,
    SimpleFactorValidationEngine,
    ValidationDatasetBuilder,
)
from cqros.factors import FactorsRepository, FactorStatus
from cqros.factors.schema import CANONICAL_COLUMN_ORDER, COLUMN_DTYPES
from cqros.labels.schema import (
    CANONICAL_COLUMN_ORDER as LABEL_CANONICAL_COLUMN_ORDER,
)
from cqros.labels.schema import COLUMN_DTYPES as LABEL_COLUMN_DTYPES
from cqros.labels.schema import LABEL_COLUMNS
from cqros.storage import LabelRepository, ParquetStore, StorageLayout

_MANAGER = "simple"
_ENGINE = "simple"
_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1h"
_YEAR = 2026
_OPEN_TIMES = (1_700_000_000_000, 1_700_003_600_000)
_OPEN_TIME = _OPEN_TIMES[0]
_FACTOR_NAME = "momentum"
_FACTOR_VERSION = "1.0.0"
_FACTOR_CATEGORY = "price"
_FACTOR_GROUP = "alpha"


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def _options(
    *,
    storage_root: Path,
    manager: str = _MANAGER,
    engine: str = _ENGINE,
    symbols: tuple[str, ...] | None = None,
    timeframes: tuple[str, ...] | None = None,
    years: tuple[int, ...] | None = None,
    overwrite: bool = False,
    workers: int = ResearchConfig().worker_count,
    execution_mode: FactorValidationExecutionMode = (
        FactorValidationExecutionMode.MEMORY_EFFICIENT
    ),
    factor_batch_size: int = 1,
    verbose: bool = False,
    debug: bool = False,
) -> FactorValidationGenerationOptions:
    """Build FactorValidationGenerationOptions against a temporary storage root."""
    return FactorValidationGenerationOptions(
        storage_root=storage_root,
        manager=manager,
        engine=engine,
        symbols=symbols,
        timeframes=timeframes,
        years=years,
        overwrite=overwrite,
        workers=workers,
        execution_mode=execution_mode,
        factor_batch_size=factor_batch_size,
        verbose=verbose,
        debug=debug,
    )


def _factors_frame(
    *,
    symbol: str = _SYMBOL,
    factor_name: str = _FACTOR_NAME,
    factor_value: float = 0.1,
) -> pl.DataFrame:
    """Return a canonical Factors frame for factor validation generation tests."""
    rows = len(_OPEN_TIMES)
    return pl.DataFrame(
        {
            "symbol": [symbol] * rows,
            "timeframe": [_TIMEFRAME] * rows,
            "open_time": list(_OPEN_TIMES),
            "factor_name": [factor_name] * rows,
            "factor_version": [_FACTOR_VERSION] * rows,
            "factor_category": [_FACTOR_CATEGORY] * rows,
            "factor_group": [_FACTOR_GROUP] * rows,
            "factor_value": [factor_value + float(index) * 0.01 for index in range(rows)],
            "lookback": [20] * rows,
            "prediction_horizon": [1] * rows,
            "enabled": [True] * rows,
            "status": [FactorStatus.ACTIVE.value] * rows,
        },
        schema=dict(COLUMN_DTYPES),
    ).select(list(CANONICAL_COLUMN_ORDER))


def _labels_frame(*, symbol: str = _SYMBOL) -> pl.DataFrame:
    """Return a canonical Labels frame containing future_return_1."""
    rows = len(_OPEN_TIMES)
    data: dict[str, list[object]] = {
        "symbol": [symbol] * rows,
        "timeframe": [_TIMEFRAME] * rows,
        "open_time": list(_OPEN_TIMES),
    }
    for index, column in enumerate(LABEL_COLUMNS):
        if column.startswith("direction_"):
            data[column] = [1 if offset % 2 == 0 else 0 for offset in range(rows)]
        else:
            data[column] = [0.01 * float(index + offset + 1) for offset in range(rows)]
    return pl.DataFrame(data, schema=dict(LABEL_COLUMN_DTYPES)).select(
        list(LABEL_CANONICAL_COLUMN_ORDER)
    )


def _seed_generation_inputs(
    *,
    layout: StorageLayout,
    datastore: ParquetStore,
    symbol: str = _SYMBOL,
    factor_name: str = _FACTOR_NAME,
) -> None:
    """Persist Factors and Labels partitions needed for generation tests."""
    FactorsRepository(layout, datastore).save(
        _factors_frame(symbol=symbol, factor_name=factor_name),
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=symbol,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    LabelRepository(layout, datastore).save(
        _labels_frame(symbol=symbol),
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=symbol,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )


def _wired_pipeline(
    *,
    layout: StorageLayout,
    datastore: ParquetStore,
    registry: FactorValidationEngineRegistry | None = None,
) -> FactorValidationPipeline:
    """Compose a FactorValidationPipeline with a real ValidationDatasetBuilder."""
    factors_repository = FactorsRepository(layout, datastore)
    label_repository = LabelRepository(layout, datastore)
    builder = ValidationDatasetBuilder(factors_repository, label_repository)
    options = _options(storage_root=layout.root)
    return build_factor_validation_pipeline(
        options,
        builder=builder,
        engine_registry=registry,
    )


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
    """build_options rejects workers <= 0 with CLI-GENERATE-FACTOR-VALIDATION-001."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "simple", "--workers", "0"])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-FACTOR-VALIDATION-001"


def test_build_options_rejects_blank_manager() -> None:
    """build_options rejects a blank manager with CLI-GENERATE-FACTOR-VALIDATION-004."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "   "])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-FACTOR-VALIDATION-004"


def test_build_options_rejects_blank_engine() -> None:
    """build_options rejects a blank engine with CLI-GENERATE-FACTOR-VALIDATION-005."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "simple", "--engine", "   "])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-FACTOR-VALIDATION-005"


def test_build_options_uses_storage_root(tmp_path: Path) -> None:
    """build_options honors an explicit storage root override."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "simple", "--storage-root", str(tmp_path)])
    options = build_options(args)
    assert options.storage_root == tmp_path


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def test_build_default_engine_returns_simple_factor_validation_engine() -> None:
    """build_default_engine returns a SimpleFactorValidationEngine instance."""
    assert isinstance(build_default_engine(), SimpleFactorValidationEngine)


def test_build_registry_contains_simple_engine() -> None:
    """Default registry contains SimpleFactorValidationEngine under 'simple'."""
    registry = build_registry()
    assert registry.exists("simple")
    assert isinstance(registry.get("simple"), SimpleFactorValidationEngine)


def test_build_factor_validation_pipeline_wires_registry(tmp_path: Path) -> None:
    """build_factor_validation_pipeline returns a wired FactorValidationPipeline."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    options = _options(storage_root=tmp_path)
    builder = ValidationDatasetBuilder(
        FactorsRepository(layout, datastore),
        LabelRepository(layout, datastore),
    )
    pipeline = build_factor_validation_pipeline(options, builder=builder)
    assert isinstance(pipeline, FactorValidationPipeline)


# ---------------------------------------------------------------------------
# discover_work
# ---------------------------------------------------------------------------


def test_discover_work_groups_factors_partitions(tmp_path: Path) -> None:
    """discover_work returns sorted timeframe/year panels from the factors tier."""
    layout = StorageLayout(tmp_path)
    factors_repository = FactorsRepository(layout, ParquetStore())
    for symbol, year in (("ETHUSDT", 2025), ("BTCUSDT", 2026), ("BTCUSDT", 2025)):
        factors_repository.save(
            _factors_frame(symbol=symbol),
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            symbol=symbol,
            timeframe=_TIMEFRAME,
            year=year,
        )
    work = discover_work(
        factors_repository,
        _options(storage_root=tmp_path),
    )
    assert work == (
        DiscoveredWorkItem(
            manager=_MANAGER,
            timeframe=_TIMEFRAME,
            year=2025,
            symbols=("BTCUSDT", "ETHUSDT"),
        ),
        DiscoveredWorkItem(
            manager=_MANAGER,
            timeframe=_TIMEFRAME,
            year=2026,
            symbols=("BTCUSDT",),
        ),
    )


def test_discover_work_returns_empty_when_no_factors_partitions(tmp_path: Path) -> None:
    """discover_work returns empty tuple when no Factors partitions exist."""
    layout = StorageLayout(tmp_path)
    factors_repository = FactorsRepository(layout, ParquetStore())
    work = discover_work(factors_repository, _options(storage_root=tmp_path))
    assert work == ()


# ---------------------------------------------------------------------------
# format_summary
# ---------------------------------------------------------------------------


def test_format_summary_includes_core_fields() -> None:
    """format_summary renders manager, engine, and factor validation aggregates."""
    summary = FactorValidationGenerationSummary(
        manager=_MANAGER,
        engine=_ENGINE,
        symbols=2,
        rows=10,
        passed_rows=8,
        failed_status_rows=2,
        successful_tasks=2,
        failed_tasks=1,
        skipped_tasks=0,
        duration_seconds=2.5,
        output_directory=Path("data") / STORAGE_DIR_FACTOR_VALIDATION,
        failed_task_labels=("1h 2026",),
    )
    text = format_summary(summary)
    assert "CQROS Factor Validation Generation Summary" in text
    assert f"Manager: {_MANAGER}" in text
    assert f"Engine: {_ENGINE}" in text
    assert "Symbols: 2" in text
    assert "Rows: 10" in text
    assert "Passed: 8" in text
    assert "Failed Status: 2" in text
    assert "Failed Tasks" in text
    assert "1h 2026" in text
    assert STORAGE_DIR_FACTOR_VALIDATION in text


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
            pipeline=_wired_pipeline(layout=layout, datastore=datastore),
            factors_repository=FactorsRepository(layout, datastore),
            factor_validation_repository=FactorValidationRepository(layout, datastore),
            options=options,
            work=(),
        )
    )
    assert summary.successful_tasks == 0
    assert summary.failed_tasks == 0
    assert summary.skipped_tasks == 0
    assert summary.rows == 0
    assert summary.passed_rows == 0
    assert summary.failed_status_rows == 0
    assert summary.output_directory == tmp_path / STORAGE_DIR_FACTOR_VALIDATION


# ---------------------------------------------------------------------------
# run_generation — persists factor validation ledgers
# ---------------------------------------------------------------------------


def test_run_generation_persists_factor_validation_partitions(tmp_path: Path) -> None:
    """Generation loads Factors panels, runs the pipeline, and persists output."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()

    _seed_generation_inputs(layout=layout, datastore=datastore, symbol="BTCUSDT")
    _seed_generation_inputs(
        layout=layout,
        datastore=datastore,
        symbol="ETHUSDT",
        factor_name="rsi",
    )

    options = _options(storage_root=tmp_path)
    factors_repository = FactorsRepository(layout, datastore)
    work = discover_work(factors_repository, options)

    registry = FactorValidationEngineRegistry()
    registry.register(_ENGINE, SimpleFactorValidationEngine())
    summary = _run(
        run_generation(
            pipeline=_wired_pipeline(
                layout=layout,
                datastore=datastore,
                registry=registry,
            ),
            factors_repository=factors_repository,
            factor_validation_repository=FactorValidationRepository(layout, datastore),
            options=options,
            work=work,
        )
    )
    assert summary.successful_tasks == 1
    assert summary.failed_tasks == 0
    assert summary.symbols == 2
    assert summary.rows == 2
    assert summary.passed_rows == 2
    assert summary.failed_status_rows == 0

    factor_validation_repo = FactorValidationRepository(layout, datastore)
    assert factor_validation_repo.exists(
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )


# ---------------------------------------------------------------------------
# run_generation — skip existing without overwrite
# ---------------------------------------------------------------------------


def test_run_generation_skips_existing_without_overwrite(tmp_path: Path) -> None:
    """Existing factor validation partitions are skipped when overwrite is False."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    _seed_generation_inputs(layout=layout, datastore=datastore)

    options = _options(storage_root=tmp_path, overwrite=False)
    factors_repository = FactorsRepository(layout, datastore)
    work = discover_work(factors_repository, options)

    registry = build_registry()
    first_summary = _run(
        run_generation(
            pipeline=_wired_pipeline(
                layout=layout,
                datastore=datastore,
                registry=registry,
            ),
            factors_repository=factors_repository,
            factor_validation_repository=FactorValidationRepository(layout, datastore),
            options=options,
            work=work,
        )
    )
    assert first_summary.successful_tasks == 1

    second_summary = _run(
        run_generation(
            pipeline=_wired_pipeline(layout=layout, datastore=datastore),
            factors_repository=factors_repository,
            factor_validation_repository=FactorValidationRepository(layout, datastore),
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
    """main returns 0 when no Factors partitions are discovered."""
    with patch.object(generate_factor_validation_module, "DEFAULT_STORAGE_ROOT", tmp_path):
        code = _run(main(["--manager", "simple", "--storage-root", str(tmp_path)]))
    assert code == 0


def test_main_validation_error_exit_code() -> None:
    """main returns 1 when build_options raises ValidationError."""
    code = _run(main(["--manager", "simple", "--workers", "0"]))
    assert code == 1


def test_main_returns_success_after_generation(tmp_path: Path) -> None:
    """main returns 0 after successful factor validation generation."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    _seed_generation_inputs(layout=layout, datastore=datastore)
    with patch.object(generate_factor_validation_module, "DEFAULT_STORAGE_ROOT", tmp_path):
        code = _run(main(["--manager", "simple", "--storage-root", str(tmp_path)]))
    assert code == 0


# ---------------------------------------------------------------------------
# Console isolation — progress must never abort generation
# ---------------------------------------------------------------------------


def test_configure_stdio_utf8_reconfigures_when_supported() -> None:
    """_configure_stdio_utf8 requests UTF-8 replace mode on stdout/stderr."""
    stdout_calls: list[dict[str, str]] = []
    stderr_calls: list[dict[str, str]] = []

    class _Stream:
        def __init__(self, sink: list[dict[str, str]]) -> None:
            self._sink = sink

        def reconfigure(self, *, encoding: str, errors: str) -> None:
            self._sink.append({"encoding": encoding, "errors": errors})

    with (
        patch.object(
            generate_factor_validation_module.sys,
            "stdout",
            _Stream(stdout_calls),
        ),
        patch.object(
            generate_factor_validation_module.sys,
            "stderr",
            _Stream(stderr_calls),
        ),
    ):
        generate_factor_validation_module._configure_stdio_utf8()  # pyright: ignore[reportPrivateUsage]

    assert stdout_calls == [{"encoding": "utf-8", "errors": "replace"}]
    assert stderr_calls == [{"encoding": "utf-8", "errors": "replace"}]


def test_configure_stdio_utf8_ignores_missing_reconfigure() -> None:
    """_configure_stdio_utf8 is a no-op when reconfigure is unavailable."""

    class _Stream:
        pass

    with (
        patch.object(generate_factor_validation_module.sys, "stdout", _Stream()),
        patch.object(generate_factor_validation_module.sys, "stderr", _Stream()),
    ):
        generate_factor_validation_module._configure_stdio_utf8()  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    "error",
    [
        UnicodeEncodeError("charmap", "abc", 0, 1, "ordinal not in range"),
        OSError("console unavailable"),
        BrokenPipeError("broken pipe"),
    ],
)
def test_print_progress_swallows_console_errors(error: Exception) -> None:
    """_print_progress never propagates console encoding or pipe failures."""
    result = FactorValidationTaskResult(
        timeframe=_TIMEFRAME,
        year=_YEAR,
        symbols=1,
        status="succeeded",
        rows_generated=3,
        passed_rows=2,
        failed_status_rows=1,
    )
    with patch.object(
        generate_factor_validation_module,
        "print",
        side_effect=error,
    ):
        generate_factor_validation_module._print_progress(
            result
        )  # pyright: ignore[reportPrivateUsage]


def test_emit_text_logs_debug_on_console_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """_emit_text logs at DEBUG and continues when console writing fails."""
    with (
        caplog.at_level(
            generate_factor_validation_module.logging.DEBUG,
            logger=generate_factor_validation_module.__name__,
        ),
        patch.object(
            generate_factor_validation_module,
            "print",
            side_effect=UnicodeEncodeError("charmap", "x", 0, 1, "fail"),
        ),
    ):
        generate_factor_validation_module._emit_text(  # pyright: ignore[reportPrivateUsage]
            "OK BTCUSDT 1h 2026 rows=1",
            flush=True,
        )

    assert any("console write failed" in record.message for record in caplog.records)


def test_run_generation_continues_when_progress_print_fails(tmp_path: Path) -> None:
    """Progress UnicodeEncodeError must not drop results or fail the run."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    _seed_generation_inputs(layout=layout, datastore=datastore, symbol="BTCUSDT")

    options = _options(storage_root=tmp_path)
    factors_repository = FactorsRepository(layout, datastore)
    work = discover_work(factors_repository, options)

    with patch.object(
        generate_factor_validation_module,
        "print",
        side_effect=UnicodeEncodeError("charmap", "ok", 0, 1, "fail"),
    ):
        summary = _run(
            run_generation(
                pipeline=_wired_pipeline(layout=layout, datastore=datastore),
                factors_repository=factors_repository,
                factor_validation_repository=FactorValidationRepository(layout, datastore),
                options=options,
                work=work,
            )
        )

    assert summary.failed_tasks == 0
    assert summary.successful_tasks == 1
    assert summary.rows > 0
    assert FactorValidationRepository(layout, datastore).exists(
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )


def test_main_returns_success_when_summary_print_fails(tmp_path: Path) -> None:
    """Summary console failures after successful generation still return 0."""
    layout = StorageLayout(tmp_path)
    datastore = ParquetStore()
    _seed_generation_inputs(layout=layout, datastore=datastore)

    real_print = __import__("builtins").print

    def _print_sometimes_fails(*args: object, **kwargs: object) -> None:
        text = str(args[0]) if args else ""
        if "CQROS Factor Validation Generation Summary" in text:
            raise UnicodeEncodeError("charmap", text, 3, 7, "fail")
        return real_print(*args, **kwargs)

    with (
        patch.object(generate_factor_validation_module, "DEFAULT_STORAGE_ROOT", tmp_path),
        patch.object(
            generate_factor_validation_module,
            "print",
            side_effect=_print_sometimes_fails,
        ),
    ):
        code = _run(main(["--manager", "simple", "--storage-root", str(tmp_path)]))

    assert code == 0
