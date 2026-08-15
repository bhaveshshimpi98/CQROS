"""Unit tests for CQROS training-dataset build CLI."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from cqros.cli.build_training_dataset import (
    DiscoveredWorkItem,
    TrainingDatasetOptions,
    TrainingDatasetSummary,
    TrainingTaskResult,
    build_options,
    build_parser,
    build_training_pipeline,
    discover_work,
    format_summary,
    main,
    run_generation,
)
from cqros.config.models import ResearchConfig
from cqros.core.constants import (
    DEFAULT_STORAGE_ROOT,
    STORAGE_DIR_FEATURES,
    STORAGE_DIR_LABELS,
    STORAGE_DIR_TRAINING,
)
from cqros.core.exceptions import ValidationError
from cqros.storage import (
    FeatureRepository,
    LabelRepository,
    ParquetStore,
    StorageLayout,
    TrainingRepository,
)
from cqros.training import TrainingPipeline


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
) -> TrainingDatasetOptions:
    """Build options for tests against a temporary storage root."""
    return TrainingDatasetOptions(
        storage_root=storage_root,
        symbols=symbols,
        timeframes=timeframes,
        years=years,
        overwrite=overwrite,
        workers=workers,
        verbose=verbose,
        debug=debug,
    )


def _touch_feature(
    root: Path,
    *,
    symbol: str,
    timeframe: str,
    year: int,
) -> Path:
    """Create an empty feature year partition path on disk."""
    path = (
        root
        / STORAGE_DIR_FEATURES
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


def _touch_training(
    root: Path,
    *,
    symbol: str,
    timeframe: str,
    year: int,
) -> Path:
    """Create an empty training year partition path on disk."""
    path = (
        root
        / STORAGE_DIR_TRAINING
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
    """Parser accepts every documented training-dataset build flag."""
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
    """Explicit CLI flags map onto TrainingDatasetOptions."""
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
    """Discovery walks feature and label partitions without hardcoding symbols."""
    _touch_feature(tmp_path, symbol="BTCUSDT", timeframe="1h", year=2024)
    _touch_label(tmp_path, symbol="BTCUSDT", timeframe="1h", year=2024)
    _touch_feature(tmp_path, symbol="ETHUSDT", timeframe="1h", year=2023)
    _touch_label(tmp_path, symbol="ETHUSDT", timeframe="1h", year=2023)
    _touch_feature(tmp_path, symbol="ETHUSDT", timeframe="1h", year=2024)
    _touch_label(tmp_path, symbol="ETHUSDT", timeframe="1h", year=2024)

    layout = StorageLayout(tmp_path)
    store = ParquetStore()
    features = FeatureRepository(layout, store)
    labels = LabelRepository(layout, store)
    work = discover_work(features, labels, _options(storage_root=tmp_path))

    assert len(work) == 2
    assert work[0].symbol == "BTCUSDT"
    assert work[0].years == (2024,)
    assert work[1].symbol == "ETHUSDT"
    assert work[1].years == (2023, 2024)


def test_discover_work_skips_feature_only_or_label_only(tmp_path: Path) -> None:
    """Feature-only or label-only partitions are never discovered."""
    _touch_feature(tmp_path, symbol="BTCUSDT", timeframe="1h", year=2024)
    _touch_label(tmp_path, symbol="ETHUSDT", timeframe="1h", year=2024)

    layout = StorageLayout(tmp_path)
    store = ParquetStore()
    features = FeatureRepository(layout, store)
    labels = LabelRepository(layout, store)
    work = discover_work(features, labels, _options(storage_root=tmp_path))

    assert work == ()


def test_discover_work_filters_symbols_timeframes_years(tmp_path: Path) -> None:
    """Symbol, timeframe, and year filters apply to discovery."""
    _touch_feature(tmp_path, symbol="BTCUSDT", timeframe="1h", year=2024)
    _touch_label(tmp_path, symbol="BTCUSDT", timeframe="1h", year=2024)
    _touch_feature(tmp_path, symbol="BTCUSDT", timeframe="4h", year=2024)
    _touch_label(tmp_path, symbol="BTCUSDT", timeframe="4h", year=2024)
    _touch_feature(tmp_path, symbol="ETHUSDT", timeframe="1h", year=2024)
    _touch_label(tmp_path, symbol="ETHUSDT", timeframe="1h", year=2024)
    _touch_feature(tmp_path, symbol="BTCUSDT", timeframe="1h", year=2025)
    _touch_label(tmp_path, symbol="BTCUSDT", timeframe="1h", year=2025)

    layout = StorageLayout(tmp_path)
    store = ParquetStore()
    features = FeatureRepository(layout, store)
    labels = LabelRepository(layout, store)
    work = discover_work(
        features,
        labels,
        _options(
            storage_root=tmp_path,
            symbols=("BTCUSDT",),
            timeframes=("1h",),
            years=(2024,),
        ),
    )

    assert work == (DiscoveredWorkItem(symbol="BTCUSDT", timeframe="1h", years=(2024,)),)


def test_build_training_pipeline_wires_dependencies(tmp_path: Path) -> None:
    """Pipeline composition wires TrainingRepository persistence."""
    pipeline = build_training_pipeline(_options(storage_root=tmp_path))
    assert isinstance(pipeline, TrainingPipeline)


def test_run_generation_empty_work(tmp_path: Path) -> None:
    """Empty discovery produces a zeroed summary without invoking the pipeline."""
    pipeline = MagicMock(spec=TrainingPipeline)
    features = MagicMock(spec=FeatureRepository)
    labels = MagicMock(spec=LabelRepository)
    training = MagicMock(spec=TrainingRepository)
    summary = _run(
        run_generation(
            pipeline=pipeline,
            feature_repository=features,
            label_repository=labels,
            training_repository=training,
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
    """Existing training partitions are skipped unless --overwrite is set."""
    _touch_training(tmp_path, symbol="BTCUSDT", timeframe="1h", year=2024)
    work = (DiscoveredWorkItem(symbol="BTCUSDT", timeframe="1h", years=(2024,)),)

    pipeline = MagicMock(spec=TrainingPipeline)
    features = MagicMock(spec=FeatureRepository)
    labels = MagicMock(spec=LabelRepository)
    training = MagicMock(spec=TrainingRepository)
    training.exists.return_value = True

    summary = _run(
        run_generation(
            pipeline=pipeline,
            feature_repository=features,
            label_repository=labels,
            training_repository=training,
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
    pipeline = MagicMock(spec=TrainingPipeline)
    pipeline.run.return_value = pl.DataFrame({"open_time": [1, 2]})
    features = MagicMock(spec=FeatureRepository)
    features.load.return_value = pl.DataFrame({"open_time": [1, 2]})
    labels = MagicMock(spec=LabelRepository)
    labels.load.return_value = pl.DataFrame({"open_time": [1, 2]})
    training = MagicMock(spec=TrainingRepository)
    training.exists.return_value = True

    summary = _run(
        run_generation(
            pipeline=pipeline,
            feature_repository=features,
            label_repository=labels,
            training_repository=training,
            options=_options(storage_root=tmp_path, workers=1, overwrite=True),
            work=work,
        )
    )

    captured = capsys.readouterr().out
    assert summary.successful_tasks == 1
    assert summary.skipped_tasks == 0
    assert summary.rows_generated == 2
    pipeline.run.assert_called_once()
    call_args = pipeline.run.call_args
    assert call_args.args[0].equals(features.load.return_value)
    assert call_args.args[1].equals(labels.load.return_value)
    assert "OK BTCUSDT 1h 2024 rows=2" in captured


def test_run_generation_success_and_progress(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Successful generation increments counters and prints progress."""
    work = (DiscoveredWorkItem(symbol="BTCUSDT", timeframe="1h", years=(2024,)),)
    pipeline = MagicMock(spec=TrainingPipeline)
    pipeline.run.return_value = pl.DataFrame({"open_time": [1, 2, 3]})
    features = MagicMock(spec=FeatureRepository)
    features.load.return_value = pl.DataFrame({"open_time": [1, 2, 3]})
    labels = MagicMock(spec=LabelRepository)
    labels.load.return_value = pl.DataFrame({"open_time": [1, 2, 3]})
    training = MagicMock(spec=TrainingRepository)
    training.exists.return_value = False

    summary = _run(
        run_generation(
            pipeline=pipeline,
            feature_repository=features,
            label_repository=labels,
            training_repository=training,
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
    pipeline = MagicMock(spec=TrainingPipeline)
    pipeline.run.side_effect = [RuntimeError("boom"), pl.DataFrame({"open_time": [1]})]
    features = MagicMock(spec=FeatureRepository)
    features.load.return_value = pl.DataFrame({"open_time": [1]})
    labels = MagicMock(spec=LabelRepository)
    labels.load.return_value = pl.DataFrame({"open_time": [1]})
    training = MagicMock(spec=TrainingRepository)
    training.exists.return_value = False

    summary = _run(
        run_generation(
            pipeline=pipeline,
            feature_repository=features,
            label_repository=labels,
            training_repository=training,
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
        TrainingDatasetSummary(
            symbols_discovered=1,
            symbols_processed=1,
            timeframes_processed=1,
            successful_tasks=0,
            failed_tasks=1,
            skipped_tasks=0,
            rows_generated=0,
            duration_seconds=1.5,
            output_directory=Path("data/training"),
            failed_task_labels=("BTCUSDT 1h 2024",),
        )
    )
    assert "CQROS Training Dataset Summary" in text
    assert "Failed tasks: 1" in text
    assert "- BTCUSDT 1h 2024" in text
    assert "Output directory: data/training" in text


def test_main_exit_code_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """main returns 0 when no tasks fail."""
    with (
        patch("cqros.cli.build_training_dataset.DEFAULT_STORAGE_ROOT", str(tmp_path)),
        patch("cqros.cli.build_training_dataset.build_training_pipeline") as build_pipeline,
        patch("cqros.cli.build_training_dataset.discover_work", return_value=()),
    ):
        build_pipeline.return_value = MagicMock(spec=TrainingPipeline)
        code = _run(main(["--workers", "1"]))

    captured = capsys.readouterr()
    assert code == 0
    assert "CQROS Training Dataset Summary" in captured.out


def test_main_validation_error_exit_code() -> None:
    """Fatal CLI validation errors return exit code 1."""
    code = _run(main(["--workers", "0"]))
    assert code == 1


def test_configure_logging_debug(tmp_path: Path) -> None:
    """--debug enables DEBUG logging for the cqros logger."""
    with (
        patch("cqros.cli.build_training_dataset.DEFAULT_STORAGE_ROOT", str(tmp_path)),
        patch("cqros.cli.build_training_dataset.build_training_pipeline") as build_pipeline,
        patch("cqros.cli.build_training_dataset.discover_work", return_value=()),
    ):
        build_pipeline.return_value = MagicMock(spec=TrainingPipeline)
        _run(main(["--debug"]))
    assert logging.getLogger("cqros").level == logging.DEBUG


def test_training_task_result_fields() -> None:
    """TrainingTaskResult stores status metadata immutably."""
    result = TrainingTaskResult(
        symbol="BTCUSDT",
        timeframe="1h",
        year=2024,
        status="failed",
        error_type="RuntimeError",
        error_message="boom",
    )
    assert result.status == "failed"
    assert result.error_type == "RuntimeError"
