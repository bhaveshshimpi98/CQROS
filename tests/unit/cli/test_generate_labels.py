"""Unit tests for CQROS label-generation CLI."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from cqros.cli.generate_labels import (
    DiscoveredWorkItem,
    LabelGenerationOptions,
    LabelGenerationSummary,
    LabelTaskResult,
    build_label_pipeline,
    build_options,
    build_parser,
    discover_work,
    format_summary,
    load_label_input_frame,
    main,
    run_generation,
)
from cqros.config.models import ResearchConfig
from cqros.core.constants import DEFAULT_STORAGE_ROOT, STORAGE_DIR_LABELS, STORAGE_DIR_PROCESSED
from cqros.core.exceptions import ValidationError
from cqros.labels import LabelPipeline
from cqros.storage import (
    LabelRepository,
    ParquetStore,
    ProcessedMarketDataRepository,
    StorageLayout,
)


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def _options(
    *,
    storage_root: Path,
    symbols: tuple[str, ...] | None = None,
    timeframes: tuple[str, ...] | None = None,
    years: tuple[int, ...] | None = None,
    overwrite: bool = False,
    workers: int = ResearchConfig().worker_count,
    verbose: bool = False,
    debug: bool = False,
) -> LabelGenerationOptions:
    """Build options for tests against a temporary storage root."""
    return LabelGenerationOptions(
        storage_root=storage_root,
        symbols=symbols,
        timeframes=timeframes,
        years=years,
        overwrite=overwrite,
        workers=workers,
        verbose=verbose,
        debug=debug,
    )


def _touch_processed_ohlcv(
    root: Path,
    *,
    symbol: str,
    timeframe: str,
    year: int,
) -> Path:
    """Create an empty processed OHLCV year partition path on disk."""
    path = (
        root
        / STORAGE_DIR_PROCESSED
        / "ohlcv"
        / "binance"
        / "usdt_perpetual"
        / symbol
        / timeframe
        / f"{year}.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _touch_label(
    root: Path,
    *,
    symbol: str,
    timeframe: str,
    year: int,
) -> Path:
    """Create an empty label year partition path on disk."""
    path = (
        root
        / STORAGE_DIR_LABELS
        / "binance"
        / "usdt_perpetual"
        / symbol
        / timeframe
        / f"{year}.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_build_parser_defaults() -> None:
    """Omitted optional flags keep discovery defaults."""
    args = build_parser().parse_args([])
    assert args.symbols is None
    assert args.timeframes is None
    assert args.years is None
    assert args.overwrite is False
    assert args.workers == ResearchConfig().worker_count
    assert args.verbose is False
    assert args.debug is False


def test_build_parser_accepts_all_flags() -> None:
    """Parser accepts every documented label-generation flag."""
    args = build_parser().parse_args(
        [
            "--symbols",
            "BTCUSDT",
            "ETHUSDT",
            "--timeframes",
            "1h",
            "4h",
            "--years",
            "2024",
            "2025",
            "--overwrite",
            "--workers",
            "2",
            "--verbose",
            "--debug",
        ]
    )
    assert args.symbols == ["BTCUSDT", "ETHUSDT"]
    assert args.timeframes == ["1h", "4h"]
    assert args.years == ["2024", "2025"]
    assert args.overwrite is True
    assert args.workers == 2
    assert args.verbose is True
    assert args.debug is True


def test_build_options_defaults() -> None:
    """Omitted filters map to discovery-all options."""
    options = build_options(build_parser().parse_args([]))
    assert options.storage_root == Path(DEFAULT_STORAGE_ROOT)
    assert options.symbols is None
    assert options.timeframes is None
    assert options.years is None
    assert options.overwrite is False
    assert options.workers == ResearchConfig().worker_count


def test_build_options_maps_filters() -> None:
    """Explicit CLI flags map onto LabelGenerationOptions."""
    options = build_options(
        build_parser().parse_args(
            [
                "--symbols",
                "ETHUSDT",
                "--timeframes",
                "1d",
                "--years",
                "2023",
                "--overwrite",
                "--workers",
                "8",
                "--debug",
            ]
        )
    )
    assert options.symbols == ("ETHUSDT",)
    assert options.timeframes == ("1d",)
    assert options.years == (2023,)
    assert options.overwrite is True
    assert options.workers == 8
    assert options.debug is True


def test_build_options_rejects_non_positive_workers() -> None:
    """Non-positive --workers fails validation."""
    args = build_parser().parse_args(["--workers", "0"])
    with pytest.raises(ValidationError, match="workers must be greater than 0"):
        build_options(args)


def test_build_options_rejects_unsupported_timeframe() -> None:
    """Unsupported --timeframes values fail validation."""
    args = build_parser().parse_args(["--timeframes", "2x"])
    with pytest.raises(ValidationError, match="unsupported timeframe"):
        build_options(args)


def test_build_options_rejects_invalid_year() -> None:
    """Non-integer --years values fail validation."""
    args = build_parser().parse_args(["--years", "abc"])
    with pytest.raises(ValidationError, match="invalid year"):
        build_options(args)


def test_discover_work_finds_symbols_and_years(tmp_path: Path) -> None:
    """Discovery walks processed OHLCV partitions without hardcoding symbols."""
    _touch_processed_ohlcv(tmp_path, symbol="BTCUSDT", timeframe="1h", year=2024)
    _touch_processed_ohlcv(tmp_path, symbol="ETHUSDT", timeframe="1h", year=2023)
    _touch_processed_ohlcv(tmp_path, symbol="ETHUSDT", timeframe="1h", year=2024)

    repository = ProcessedMarketDataRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(repository, _options(storage_root=tmp_path))

    assert len(work) == 2
    assert work[0].symbol == "BTCUSDT"
    assert work[0].years == (2024,)
    assert work[1].symbol == "ETHUSDT"
    assert work[1].years == (2023, 2024)


def test_discover_work_filters_symbols_timeframes_years(tmp_path: Path) -> None:
    """Symbol, timeframe, and year filters apply to discovery."""
    _touch_processed_ohlcv(tmp_path, symbol="BTCUSDT", timeframe="1h", year=2024)
    _touch_processed_ohlcv(tmp_path, symbol="BTCUSDT", timeframe="4h", year=2024)
    _touch_processed_ohlcv(tmp_path, symbol="ETHUSDT", timeframe="1h", year=2024)
    _touch_processed_ohlcv(tmp_path, symbol="BTCUSDT", timeframe="1h", year=2025)

    repository = ProcessedMarketDataRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(
        repository,
        _options(
            storage_root=tmp_path,
            symbols=("BTCUSDT",),
            timeframes=("1h",),
            years=(2024,),
        ),
    )

    assert work == (DiscoveredWorkItem(symbol="BTCUSDT", timeframe="1h", years=(2024,)),)


def test_build_label_pipeline_wires_dependencies(tmp_path: Path) -> None:
    """Pipeline composition wires LabelRepository persistence."""
    pipeline = build_label_pipeline(_options(storage_root=tmp_path))
    assert isinstance(pipeline, LabelPipeline)


def test_load_label_input_frame_loads_ohlcv(tmp_path: Path) -> None:
    """Label input loading selects required OHLCV columns only."""
    layout = StorageLayout(tmp_path)
    repository = ProcessedMarketDataRepository(layout, ParquetStore())
    frame = pl.DataFrame(
        {
            "open_time": [1, 2, 3],
            "open": [1.0, 1.1, 1.2],
            "high": [1.2, 1.3, 1.4],
            "low": [0.9, 1.0, 1.1],
            "close": [1.0, 1.1, 1.2],
            "volume": [10.0, 11.0, 12.0],
        }
    )
    repository.save_ohlcv(
        frame,
        exchange="binance",
        market="usdt_perpetual",
        symbol="BTCUSDT",
        timeframe="1h",
        year=2024,
    )

    loaded = load_label_input_frame(
        repository,
        symbol="BTCUSDT",
        timeframe="1h",
        year=2024,
    )
    assert loaded.columns == ["symbol", "timeframe", "open_time", "close"]
    assert loaded.get_column("symbol").to_list() == ["BTCUSDT"] * 3
    assert loaded.get_column("timeframe").to_list() == ["1h"] * 3


def test_run_generation_empty_work(tmp_path: Path) -> None:
    """Empty discovery produces a zeroed summary without invoking the pipeline."""
    pipeline = MagicMock(spec=LabelPipeline)
    processed = MagicMock(spec=ProcessedMarketDataRepository)
    labels = MagicMock(spec=LabelRepository)
    summary = _run(
        run_generation(
            pipeline=pipeline,
            processed_repository=processed,
            label_repository=labels,
            options=_options(storage_root=tmp_path),
            work=(),
        )
    )
    assert summary.successful_tasks == 0
    assert summary.failed_tasks == 0
    assert summary.skipped_tasks == 0
    pipeline.run.assert_not_called()


def test_run_generation_skips_existing_without_overwrite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Existing label partitions are skipped unless --overwrite is set."""
    _touch_label(tmp_path, symbol="BTCUSDT", timeframe="1h", year=2024)
    work = (DiscoveredWorkItem(symbol="BTCUSDT", timeframe="1h", years=(2024,)),)

    pipeline = MagicMock(spec=LabelPipeline)
    processed = MagicMock(spec=ProcessedMarketDataRepository)
    labels = MagicMock(spec=LabelRepository)
    labels.exists.return_value = True

    summary = _run(
        run_generation(
            pipeline=pipeline,
            processed_repository=processed,
            label_repository=labels,
            options=_options(storage_root=tmp_path, workers=1),
            work=work,
        )
    )

    captured = capsys.readouterr().out
    assert summary.skipped_tasks == 1
    assert summary.successful_tasks == 0
    pipeline.run.assert_not_called()
    assert "SKIP BTCUSDT 1h 2024" in captured


def test_run_generation_overwrite_existing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--overwrite regenerates partitions that already exist."""
    work = (DiscoveredWorkItem(symbol="BTCUSDT", timeframe="1h", years=(2024,)),)
    pipeline = MagicMock(spec=LabelPipeline)
    pipeline.run.return_value = pl.DataFrame({"open_time": [1, 2]})
    processed = MagicMock(spec=ProcessedMarketDataRepository)
    processed.load_ohlcv.return_value = pl.DataFrame(
        {
            "open_time": [1, 2],
            "close": [1.0, 1.1],
        }
    )
    labels = MagicMock(spec=LabelRepository)
    labels.exists.return_value = True

    summary = _run(
        run_generation(
            pipeline=pipeline,
            processed_repository=processed,
            label_repository=labels,
            options=_options(storage_root=tmp_path, workers=1, overwrite=True),
            work=work,
        )
    )

    captured = capsys.readouterr().out
    assert summary.successful_tasks == 1
    assert summary.skipped_tasks == 0
    assert summary.rows_generated == 2
    pipeline.run.assert_called_once()
    assert "OK BTCUSDT 1h 2024 rows=2" in captured


def test_run_generation_success_and_progress(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Successful generation increments counters and prints progress."""
    work = (DiscoveredWorkItem(symbol="BTCUSDT", timeframe="1h", years=(2024,)),)
    pipeline = MagicMock(spec=LabelPipeline)
    pipeline.run.return_value = pl.DataFrame({"open_time": [1, 2, 3]})
    processed = MagicMock(spec=ProcessedMarketDataRepository)
    processed.load_ohlcv.return_value = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 3,
            "timeframe": ["1h"] * 3,
            "open_time": [1, 2, 3],
            "close": [1.0, 1.1, 1.2],
        }
    )
    labels = MagicMock(spec=LabelRepository)
    labels.exists.return_value = False

    summary = _run(
        run_generation(
            pipeline=pipeline,
            processed_repository=processed,
            label_repository=labels,
            options=_options(storage_root=tmp_path, workers=1, overwrite=True),
            work=work,
        )
    )

    captured = capsys.readouterr().out
    assert summary.successful_tasks == 1
    assert summary.rows_generated == 3
    assert "OK BTCUSDT 1h 2024 rows=3" in captured
    pipeline.run.assert_called_once()


def test_run_generation_pipeline_failure_isolation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failed year does not prevent later years from running."""
    work = (DiscoveredWorkItem(symbol="BTCUSDT", timeframe="1h", years=(2024, 2025)),)
    pipeline = MagicMock(spec=LabelPipeline)
    pipeline.run.side_effect = [RuntimeError("boom"), pl.DataFrame({"open_time": [1]})]
    processed = MagicMock(spec=ProcessedMarketDataRepository)
    processed.load_ohlcv.return_value = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timeframe": ["1h"],
            "open_time": [1],
            "close": [1.0],
        }
    )
    labels = MagicMock(spec=LabelRepository)
    labels.exists.return_value = False

    summary = _run(
        run_generation(
            pipeline=pipeline,
            processed_repository=processed,
            label_repository=labels,
            options=_options(storage_root=tmp_path, workers=1, overwrite=True),
            work=work,
        )
    )

    captured = capsys.readouterr().out
    assert summary.failed_tasks == 1
    assert summary.successful_tasks == 1
    assert "FAIL BTCUSDT 1h 2024 RuntimeError" in captured
    assert "OK BTCUSDT 1h 2025 rows=1" in captured
    assert summary.failed_task_labels == ("BTCUSDT 1h 2024",)


def test_format_summary_includes_failed_tasks() -> None:
    """Summary rendering includes failed-task labels when present."""
    text = format_summary(
        LabelGenerationSummary(
            symbols_discovered=1,
            symbols_processed=1,
            timeframes_processed=1,
            successful_tasks=0,
            failed_tasks=1,
            skipped_tasks=0,
            rows_generated=0,
            duration_seconds=1.5,
            output_directory=Path("data/labels"),
            failed_task_labels=("BTCUSDT 1h 2024",),
        )
    )
    assert "CQROS Label Generation Summary" in text
    assert "Failed tasks: 1" in text
    assert "- BTCUSDT 1h 2024" in text
    assert "Output directory: data/labels" in text


def test_main_exit_code_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """main returns 0 when no tasks fail."""
    with (
        patch("cqros.cli.generate_labels.DEFAULT_STORAGE_ROOT", str(tmp_path)),
        patch("cqros.cli.generate_labels.build_label_pipeline") as build_pipeline,
        patch("cqros.cli.generate_labels.discover_work", return_value=()),
    ):
        build_pipeline.return_value = MagicMock(spec=LabelPipeline)
        code = _run(main(["--workers", "1"]))

    captured = capsys.readouterr()
    assert code == 0
    assert "CQROS Label Generation Summary" in captured.out


def test_main_validation_error_exit_code() -> None:
    """Fatal CLI validation errors return exit code 1."""
    code = _run(main(["--workers", "0"]))
    assert code == 1


def test_configure_logging_debug(tmp_path: Path) -> None:
    """--debug enables DEBUG logging for the cqros logger."""
    with (
        patch("cqros.cli.generate_labels.DEFAULT_STORAGE_ROOT", str(tmp_path)),
        patch("cqros.cli.generate_labels.build_label_pipeline") as build_pipeline,
        patch("cqros.cli.generate_labels.discover_work", return_value=()),
    ):
        build_pipeline.return_value = MagicMock(spec=LabelPipeline)
        _run(main(["--debug"]))
    assert logging.getLogger("cqros").level == logging.DEBUG


def test_label_task_result_fields() -> None:
    """LabelTaskResult stores status metadata immutably."""
    result = LabelTaskResult(
        symbol="BTCUSDT",
        timeframe="1h",
        year=2024,
        status="failed",
        error_type="RuntimeError",
        error_message="boom",
    )
    assert result.status == "failed"
    assert result.error_type == "RuntimeError"
