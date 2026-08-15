"""Unit tests for CQROS prediction-generation CLI."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from cqros.cli.generate_predictions import (
    DiscoveredWorkItem,
    PredictionGenerationOptions,
    PredictionGenerationSummary,
    PredictionTaskResult,
    build_options,
    build_parser,
    build_prediction_pipeline,
    discover_work,
    format_summary,
    main,
    resolve_model_artifact,
    run_generation,
)
from cqros.config.models import ResearchConfig
from cqros.core.constants import (
    DEFAULT_STORAGE_ROOT,
    STORAGE_DIR_FEATURES,
    STORAGE_DIR_MODELS,
    STORAGE_DIR_PREDICTIONS,
)
from cqros.core.exceptions import ValidationError
from cqros.ml.models import ModelArtifactRef, ModelArtifactRepository
from cqros.predictions import PredictionPipeline
from cqros.storage import (
    FeatureRepository,
    ParquetStore,
    PredictionRepository,
    StorageLayout,
)

_MODEL = "alpha-lgbm"
_VERSION = "1.0.0"
_FRAMEWORK = "lightgbm"


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def _options(
    *,
    storage_root: Path,
    model: str = _MODEL,
    version: str = _VERSION,
    symbols: tuple[str, ...] | None = None,
    timeframes: tuple[str, ...] | None = None,
    years: tuple[int, ...] | None = None,
    overwrite: bool = False,
    workers: int = ResearchConfig().worker_count,
    verbose: bool = False,
    debug: bool = False,
) -> PredictionGenerationOptions:
    """Build options for tests against a temporary storage root."""
    return PredictionGenerationOptions(
        storage_root=storage_root,
        model=model,
        version=version,
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


def _touch_prediction(
    root: Path,
    *,
    symbol: str,
    timeframe: str,
    year: int,
    framework: str = _FRAMEWORK,
    model: str = _MODEL,
    version: str = _VERSION,
) -> Path:
    """Create an empty prediction year partition path on disk."""
    path = (
        root
        / STORAGE_DIR_PREDICTIONS
        / framework
        / model
        / version
        / "binance"
        / "usdt_perpetual"
        / symbol
        / timeframe
        / f"{year}.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _touch_model_artifact(
    root: Path,
    *,
    framework: str = _FRAMEWORK,
    model: str = _MODEL,
    version: str = _VERSION,
) -> Path:
    """Create a discoverable model artifact directory on disk."""
    directory = root / STORAGE_DIR_MODELS / framework / model / version
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.bin").write_bytes(b"")
    (directory / "metadata.json").write_text("{}", encoding="utf-8")
    return directory


def test_build_parser_requires_model_and_version() -> None:
    """--model and --version are required flags."""
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_build_parser_accepts_all_flags() -> None:
    """Parser accepts every documented prediction-generation flag."""
    args = build_parser().parse_args(
        [
            "--model",
            _MODEL,
            "--version",
            _VERSION,
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
    assert args.model == _MODEL
    assert args.version == _VERSION
    assert args.symbols == ["BTCUSDT", "ETHUSDT"]
    assert args.timeframes == ["1h", "4h"]
    assert args.years == ["2024", "2025"]
    assert args.overwrite is True
    assert args.workers == 2
    assert args.verbose is True
    assert args.debug is True


def test_build_options_maps_filters() -> None:
    """Explicit CLI flags map onto PredictionGenerationOptions."""
    options = build_options(
        build_parser().parse_args(
            [
                "--model",
                "beta",
                "--version",
                "2.0.0",
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
    assert options.storage_root == Path(DEFAULT_STORAGE_ROOT)
    assert options.model == "beta"
    assert options.version == "2.0.0"
    assert options.symbols == ("ETHUSDT",)
    assert options.timeframes == ("1d",)
    assert options.years == (2023,)
    assert options.overwrite is True
    assert options.workers == 8
    assert options.debug is True


def test_build_options_rejects_non_positive_workers() -> None:
    """Non-positive --workers fails validation."""
    args = build_parser().parse_args(["--model", _MODEL, "--version", _VERSION, "--workers", "0"])
    with pytest.raises(ValidationError, match="workers must be greater than 0"):
        build_options(args)


def test_build_options_rejects_unsupported_timeframe() -> None:
    """Unsupported --timeframes values fail validation."""
    args = build_parser().parse_args(
        ["--model", _MODEL, "--version", _VERSION, "--timeframes", "2x"]
    )
    with pytest.raises(ValidationError, match="unsupported timeframe"):
        build_options(args)


def test_build_options_rejects_invalid_year() -> None:
    """Non-integer --years values fail validation."""
    args = build_parser().parse_args(["--model", _MODEL, "--version", _VERSION, "--years", "abc"])
    with pytest.raises(ValidationError, match="invalid year"):
        build_options(args)


def test_resolve_model_artifact_finds_unique_match(tmp_path: Path) -> None:
    """resolve_model_artifact returns the unique matching artifact."""
    _touch_model_artifact(tmp_path)
    repository = ModelArtifactRepository(StorageLayout(tmp_path), MagicMock())
    artifact = resolve_model_artifact(
        repository,
        model_name=_MODEL,
        version=_VERSION,
    )
    assert artifact == ModelArtifactRef(
        framework=_FRAMEWORK,
        model_name=_MODEL,
        version=_VERSION,
    )


def test_resolve_model_artifact_missing_raises(tmp_path: Path) -> None:
    """Missing model artifacts raise ValidationError."""
    repository = ModelArtifactRepository(StorageLayout(tmp_path), MagicMock())
    with pytest.raises(ValidationError, match="model artifact not found"):
        resolve_model_artifact(repository, model_name=_MODEL, version=_VERSION)


def test_discover_work_finds_feature_partitions(tmp_path: Path) -> None:
    """Discovery walks feature partitions without hardcoding symbols."""
    _touch_feature(tmp_path, symbol="BTCUSDT", timeframe="1h", year=2024)
    _touch_feature(tmp_path, symbol="ETHUSDT", timeframe="1h", year=2023)
    _touch_feature(tmp_path, symbol="ETHUSDT", timeframe="1h", year=2024)

    features = FeatureRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(features, _options(storage_root=tmp_path))

    assert len(work) == 2
    assert work[0].symbol == "BTCUSDT"
    assert work[0].years == (2024,)
    assert work[1].symbol == "ETHUSDT"
    assert work[1].years == (2023, 2024)


def test_discover_work_filters_symbols_timeframes_years(tmp_path: Path) -> None:
    """Symbol, timeframe, and year filters apply to discovery."""
    _touch_feature(tmp_path, symbol="BTCUSDT", timeframe="1h", year=2024)
    _touch_feature(tmp_path, symbol="BTCUSDT", timeframe="4h", year=2024)
    _touch_feature(tmp_path, symbol="ETHUSDT", timeframe="1h", year=2024)
    _touch_feature(tmp_path, symbol="BTCUSDT", timeframe="1h", year=2025)

    features = FeatureRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(
        features,
        _options(
            storage_root=tmp_path,
            symbols=("BTCUSDT",),
            timeframes=("1h",),
            years=(2024,),
        ),
    )

    assert work == (DiscoveredWorkItem(symbol="BTCUSDT", timeframe="1h", years=(2024,)),)


def test_run_generation_empty_work(tmp_path: Path) -> None:
    """Empty discovery produces a zeroed summary without invoking the pipeline."""
    pipeline = MagicMock(spec=PredictionPipeline)
    features = MagicMock(spec=FeatureRepository)
    predictions = MagicMock(spec=PredictionRepository)
    summary = _run(
        run_generation(
            pipeline=pipeline,
            feature_repository=features,
            prediction_repository=predictions,
            options=_options(storage_root=tmp_path),
            work=(),
            framework=_FRAMEWORK,
        )
    )
    assert summary.successful_tasks == 0
    assert summary.failed_tasks == 0
    assert summary.skipped_tasks == 0
    assert summary.model == _MODEL
    assert summary.version == _VERSION
    pipeline.run.assert_not_called()


def test_run_generation_skips_existing_without_overwrite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Existing prediction partitions are skipped unless --overwrite is set."""
    _touch_prediction(tmp_path, symbol="BTCUSDT", timeframe="1h", year=2024)
    work = (DiscoveredWorkItem(symbol="BTCUSDT", timeframe="1h", years=(2024,)),)

    pipeline = MagicMock(spec=PredictionPipeline)
    features = MagicMock(spec=FeatureRepository)
    predictions = MagicMock(spec=PredictionRepository)
    predictions.exists.return_value = True

    summary = _run(
        run_generation(
            pipeline=pipeline,
            feature_repository=features,
            prediction_repository=predictions,
            options=_options(storage_root=tmp_path, workers=1),
            work=work,
            framework=_FRAMEWORK,
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
    pipeline = MagicMock(spec=PredictionPipeline)
    pipeline.run.return_value = pl.DataFrame({"open_time": [1, 2]})
    features = MagicMock(spec=FeatureRepository)
    features.load.return_value = pl.DataFrame({"open_time": [1, 2]})
    predictions = MagicMock(spec=PredictionRepository)
    predictions.exists.return_value = True

    summary = _run(
        run_generation(
            pipeline=pipeline,
            feature_repository=features,
            prediction_repository=predictions,
            options=_options(storage_root=tmp_path, workers=1, overwrite=True),
            work=work,
            framework=_FRAMEWORK,
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
    pipeline = MagicMock(spec=PredictionPipeline)
    pipeline.run.return_value = pl.DataFrame({"open_time": [1, 2, 3]})
    features = MagicMock(spec=FeatureRepository)
    features.load.return_value = pl.DataFrame({"open_time": [1, 2, 3]})
    predictions = MagicMock(spec=PredictionRepository)
    predictions.exists.return_value = False

    summary = _run(
        run_generation(
            pipeline=pipeline,
            feature_repository=features,
            prediction_repository=predictions,
            options=_options(storage_root=tmp_path, workers=1, overwrite=True),
            work=work,
            framework=_FRAMEWORK,
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
    pipeline = MagicMock(spec=PredictionPipeline)
    pipeline.run.side_effect = [RuntimeError("boom"), pl.DataFrame({"open_time": [1]})]
    features = MagicMock(spec=FeatureRepository)
    features.load.return_value = pl.DataFrame({"open_time": [1]})
    predictions = MagicMock(spec=PredictionRepository)
    predictions.exists.return_value = False

    summary = _run(
        run_generation(
            pipeline=pipeline,
            feature_repository=features,
            prediction_repository=predictions,
            options=_options(storage_root=tmp_path, workers=1, overwrite=True),
            work=work,
            framework=_FRAMEWORK,
        )
    )

    captured = capsys.readouterr().out
    assert summary.failed_tasks == 1
    assert summary.successful_tasks == 1
    assert "FAIL BTCUSDT 1h 2024 RuntimeError" in captured
    assert "OK BTCUSDT 1h 2025 rows=1" in captured
    assert summary.failed_task_labels == ("BTCUSDT 1h 2024",)


def test_format_summary_includes_model_and_failed_tasks() -> None:
    """Summary rendering includes model identity and failed-task labels."""
    text = format_summary(
        PredictionGenerationSummary(
            model=_MODEL,
            version=_VERSION,
            symbols_discovered=1,
            symbols_processed=1,
            timeframes_processed=1,
            successful_tasks=0,
            failed_tasks=1,
            skipped_tasks=0,
            rows_generated=0,
            duration_seconds=1.5,
            output_directory=Path("data/predictions"),
            failed_task_labels=("BTCUSDT 1h 2024",),
        )
    )
    assert "CQROS Prediction Generation Summary" in text
    assert f"Models: {_MODEL}" in text
    assert f"Version: {_VERSION}" in text
    assert "Failed tasks: 1" in text
    assert "- BTCUSDT 1h 2024" in text
    assert "Output directory: data/predictions" in text


def test_main_exit_code_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """main returns 0 when no tasks fail."""
    artifact = ModelArtifactRef(
        framework=_FRAMEWORK,
        model_name=_MODEL,
        version=_VERSION,
    )
    with (
        patch("cqros.cli.generate_predictions.DEFAULT_STORAGE_ROOT", str(tmp_path)),
        patch(
            "cqros.cli.generate_predictions.resolve_model_artifact",
            return_value=artifact,
        ),
        patch("cqros.cli.generate_predictions.build_prediction_pipeline") as build_pipeline,
        patch("cqros.cli.generate_predictions.discover_work", return_value=()),
    ):
        build_pipeline.return_value = MagicMock(spec=PredictionPipeline)
        code = _run(main(["--model", _MODEL, "--version", _VERSION, "--workers", "1"]))

    captured = capsys.readouterr()
    assert code == 0
    assert "CQROS Prediction Generation Summary" in captured.out


def test_main_validation_error_exit_code() -> None:
    """Fatal CLI validation errors return exit code 1."""
    code = _run(main(["--model", _MODEL, "--version", _VERSION, "--workers", "0"]))
    assert code == 1


def test_configure_logging_debug(tmp_path: Path) -> None:
    """--debug enables DEBUG logging for the cqros logger."""
    artifact = ModelArtifactRef(
        framework=_FRAMEWORK,
        model_name=_MODEL,
        version=_VERSION,
    )
    with (
        patch("cqros.cli.generate_predictions.DEFAULT_STORAGE_ROOT", str(tmp_path)),
        patch(
            "cqros.cli.generate_predictions.resolve_model_artifact",
            return_value=artifact,
        ),
        patch("cqros.cli.generate_predictions.build_prediction_pipeline") as build_pipeline,
        patch("cqros.cli.generate_predictions.discover_work", return_value=()),
    ):
        build_pipeline.return_value = MagicMock(spec=PredictionPipeline)
        _run(main(["--model", _MODEL, "--version", _VERSION, "--debug"]))
    assert logging.getLogger("cqros").level == logging.DEBUG


def test_prediction_task_result_fields() -> None:
    """PredictionTaskResult stores status metadata immutably."""
    result = PredictionTaskResult(
        symbol="BTCUSDT",
        timeframe="1h",
        year=2024,
        status="failed",
        error_type="RuntimeError",
        error_message="boom",
    )
    assert result.status == "failed"
    assert result.error_type == "RuntimeError"


def test_build_prediction_pipeline_wires_loaded_model(tmp_path: Path) -> None:
    """Pipeline composition loads the resolved model into inference wiring."""
    model = MagicMock()
    model.metadata.return_value = MagicMock(name=_MODEL)
    model_repository = MagicMock(spec=ModelArtifactRepository)
    model_repository.load.return_value = model
    artifact = ModelArtifactRef(
        framework=_FRAMEWORK,
        model_name=_MODEL,
        version=_VERSION,
    )

    with patch("cqros.cli.generate_predictions.ModelRegistry") as registry_cls:
        registry = MagicMock()
        registry_cls.return_value = registry
        with patch("cqros.cli.generate_predictions.InferencePredictionPipeline") as inference_cls:
            inference = MagicMock()
            inference_cls.return_value = inference
            with patch("cqros.cli.generate_predictions.PredictionPipeline") as pipeline_cls:
                pipeline_cls.return_value = MagicMock(spec=PredictionPipeline)
                build_prediction_pipeline(
                    _options(storage_root=tmp_path),
                    model_repository=model_repository,
                    model_artifact=artifact,
                )

    model_repository.load.assert_called_once_with(
        framework=_FRAMEWORK,
        model_name=_MODEL,
        version=_VERSION,
    )
    registry.register.assert_called_once_with(model)
