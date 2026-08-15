"""Unit tests for CQROS signal-generation CLI."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

import cqros.cli.generate_signals as generate_signals_module
from cqros.cli.generate_signals import (
    DiscoveredWorkItem,
    SignalGenerationOptions,
    SignalGenerationSummary,
    SignalTaskResult,
    build_adaptive_regression_policy,
    build_default_policy,
    build_options,
    build_parser,
    build_policy_registry,
    build_regression_policy,
    build_signal_pipeline,
    discover_work,
    format_summary,
    main,
    resolve_model_artifact,
    run_generation,
)
from cqros.config.models import ResearchConfig
from cqros.core.constants import (
    DEFAULT_STORAGE_ROOT,
    STORAGE_DIR_MODELS,
    STORAGE_DIR_PREDICTIONS,
    STORAGE_DIR_SIGNALS,
)
from cqros.core.exceptions import ValidationError
from cqros.ml.models import ModelArtifactRef, ModelArtifactRepository
from cqros.signals import (
    AdaptiveRegressionSignalPolicy,
    ClassificationSignalPolicy,
    RegressionSignalPolicy,
    SignalPipeline,
    SignalPolicyRegistry,
)
from cqros.storage import (
    ParquetStore,
    PredictionRepository,
    SignalRepository,
    StorageLayout,
)

_MODEL = "alpha-lgbm"
_VERSION = "1.0.0"
_FRAMEWORK = "lightgbm"
_POLICY = "classification"
_POLICY_REGRESSION = "regression"
_POLICY_ADAPTIVE_REGRESSION = "adaptive_regression"


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def _options(
    *,
    storage_root: Path,
    policy: str = _POLICY,
    model: str = _MODEL,
    version: str = _VERSION,
    symbols: tuple[str, ...] | None = None,
    timeframes: tuple[str, ...] | None = None,
    years: tuple[int, ...] | None = None,
    overwrite: bool = False,
    workers: int = ResearchConfig().worker_count,
    verbose: bool = False,
    debug: bool = False,
) -> SignalGenerationOptions:
    """Build options for tests against a temporary storage root."""
    return SignalGenerationOptions(
        storage_root=storage_root,
        policy=policy,
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


def _touch_signal(
    root: Path,
    *,
    symbol: str,
    timeframe: str,
    year: int,
) -> Path:
    """Create an empty signal year partition path on disk."""
    path = (
        root
        / STORAGE_DIR_SIGNALS
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


def test_package_exports() -> None:
    """Public CLI symbols are exported through module ``__all__``."""
    expected = {
        "DiscoveredWorkItem",
        "SignalGenerationOptions",
        "SignalGenerationSummary",
        "SignalTaskResult",
        "build_default_policy",
        "build_options",
        "build_parser",
        "build_policy_registry",
        "build_regression_policy",
        "build_signal_pipeline",
        "discover_work",
        "format_summary",
        "main",
        "resolve_model_artifact",
        "run_generation",
    }
    assert expected.issubset(set(generate_signals_module.__all__))
    assert generate_signals_module.build_parser is build_parser
    assert generate_signals_module.main is main


def test_build_parser_requires_policy_model_and_version() -> None:
    """--policy, --model, and --version are required flags."""
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--model", _MODEL, "--version", _VERSION])


def test_build_parser_accepts_all_flags() -> None:
    """Parser accepts every documented signal-generation flag."""
    args = build_parser().parse_args(
        [
            "--policy",
            _POLICY,
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
    assert args.policy == _POLICY
    assert args.model == _MODEL
    assert args.version == _VERSION
    assert args.symbols == ["BTCUSDT", "ETHUSDT"]
    assert args.timeframes == ["1h", "4h"]
    assert args.years == ["2024", "2025"]
    assert args.overwrite is True
    assert args.workers == 2
    assert args.verbose is True
    assert args.debug is True


def test_build_parser_accepts_regression_policy() -> None:
    """Parser accepts --policy regression."""
    args = build_parser().parse_args(
        [
            "--policy",
            _POLICY_REGRESSION,
            "--model",
            _MODEL,
            "--version",
            _VERSION,
        ]
    )
    assert args.policy == _POLICY_REGRESSION


def test_build_parser_accepts_adaptive_regression_policy() -> None:
    """Parser accepts --policy adaptive_regression."""
    args = build_parser().parse_args(
        [
            "--policy",
            _POLICY_ADAPTIVE_REGRESSION,
            "--model",
            _MODEL,
            "--version",
            _VERSION,
        ]
    )
    assert args.policy == _POLICY_ADAPTIVE_REGRESSION


def test_build_options_maps_adaptive_regression_policy() -> None:
    """--policy adaptive_regression maps onto SignalGenerationOptions."""
    options = build_options(
        build_parser().parse_args(
            [
                "--policy",
                _POLICY_ADAPTIVE_REGRESSION,
                "--model",
                _MODEL,
                "--version",
                _VERSION,
            ]
        )
    )
    assert options.policy == _POLICY_ADAPTIVE_REGRESSION


def test_build_options_maps_filters() -> None:
    """Explicit CLI flags map onto SignalGenerationOptions."""
    options = build_options(
        build_parser().parse_args(
            [
                "--policy",
                _POLICY_REGRESSION,
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
    assert options.policy == _POLICY_REGRESSION
    assert options.model == "beta"
    assert options.version == "2.0.0"
    assert options.symbols == ("ETHUSDT",)
    assert options.timeframes == ("1d",)
    assert options.years == (2023,)
    assert options.overwrite is True
    assert options.workers == 8
    assert options.debug is True


def test_build_options_rejects_blank_policy() -> None:
    """Blank --policy fails validation."""
    args = build_parser().parse_args(["--policy", "   ", "--model", _MODEL, "--version", _VERSION])
    with pytest.raises(ValidationError, match="policy must be a non-empty string"):
        build_options(args)


def test_build_options_rejects_non_positive_workers() -> None:
    """Non-positive --workers fails validation."""
    args = build_parser().parse_args(
        ["--policy", _POLICY, "--model", _MODEL, "--version", _VERSION, "--workers", "0"]
    )
    with pytest.raises(ValidationError, match="workers must be greater than 0"):
        build_options(args)


def test_build_options_rejects_unsupported_timeframe() -> None:
    """Unsupported --timeframes values fail validation."""
    args = build_parser().parse_args(
        [
            "--policy",
            _POLICY,
            "--model",
            _MODEL,
            "--version",
            _VERSION,
            "--timeframes",
            "2x",
        ]
    )
    with pytest.raises(ValidationError, match="unsupported timeframe"):
        build_options(args)


def test_build_options_rejects_invalid_year() -> None:
    """Non-integer --years values fail validation."""
    args = build_parser().parse_args(
        [
            "--policy",
            _POLICY,
            "--model",
            _MODEL,
            "--version",
            _VERSION,
            "--years",
            "abc",
        ]
    )
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


def test_discover_work_finds_prediction_partitions(tmp_path: Path) -> None:
    """Discovery walks prediction partitions without hardcoding symbols."""
    _touch_prediction(tmp_path, symbol="BTCUSDT", timeframe="1h", year=2024)
    _touch_prediction(tmp_path, symbol="ETHUSDT", timeframe="1h", year=2023)
    _touch_prediction(tmp_path, symbol="ETHUSDT", timeframe="1h", year=2024)

    predictions = PredictionRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(
        predictions,
        _options(storage_root=tmp_path),
        framework=_FRAMEWORK,
    )

    assert len(work) == 2
    assert work[0].symbol == "BTCUSDT"
    assert work[0].years == (2024,)
    assert work[1].symbol == "ETHUSDT"
    assert work[1].years == (2023, 2024)


def test_discover_work_filters_symbols_timeframes_years(tmp_path: Path) -> None:
    """Symbol, timeframe, and year filters apply to discovery."""
    _touch_prediction(tmp_path, symbol="BTCUSDT", timeframe="1h", year=2024)
    _touch_prediction(tmp_path, symbol="BTCUSDT", timeframe="4h", year=2024)
    _touch_prediction(tmp_path, symbol="ETHUSDT", timeframe="1h", year=2024)
    _touch_prediction(tmp_path, symbol="BTCUSDT", timeframe="1h", year=2025)

    predictions = PredictionRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(
        predictions,
        _options(
            storage_root=tmp_path,
            symbols=("BTCUSDT",),
            timeframes=("1h",),
            years=(2024,),
        ),
        framework=_FRAMEWORK,
    )

    assert work == (DiscoveredWorkItem(symbol="BTCUSDT", timeframe="1h", years=(2024,)),)


def test_discover_work_skips_missing_prediction_partitions(tmp_path: Path) -> None:
    """Missing prediction trees yield empty discovery instead of partial work."""
    predictions = PredictionRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(
        predictions,
        _options(storage_root=tmp_path),
        framework=_FRAMEWORK,
    )
    assert work == ()


def test_run_generation_empty_work(tmp_path: Path) -> None:
    """Empty discovery produces a zeroed summary without invoking the pipeline."""
    pipeline = MagicMock(spec=SignalPipeline)
    predictions = MagicMock(spec=PredictionRepository)
    signals = MagicMock(spec=SignalRepository)
    summary = _run(
        run_generation(
            pipeline=pipeline,
            prediction_repository=predictions,
            signal_repository=signals,
            options=_options(storage_root=tmp_path),
            work=(),
            framework=_FRAMEWORK,
        )
    )
    assert summary.successful_tasks == 0
    assert summary.failed_tasks == 0
    assert summary.skipped_tasks == 0
    assert summary.policy == _POLICY
    assert summary.model == _MODEL
    assert summary.version == _VERSION
    pipeline.run.assert_not_called()


def test_run_generation_skips_existing_without_overwrite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Existing signal partitions are skipped unless --overwrite is set."""
    _touch_signal(tmp_path, symbol="BTCUSDT", timeframe="1h", year=2024)
    work = (DiscoveredWorkItem(symbol="BTCUSDT", timeframe="1h", years=(2024,)),)

    pipeline = MagicMock(spec=SignalPipeline)
    predictions = MagicMock(spec=PredictionRepository)
    signals = MagicMock(spec=SignalRepository)
    signals.exists.return_value = True

    summary = _run(
        run_generation(
            pipeline=pipeline,
            prediction_repository=predictions,
            signal_repository=signals,
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
    pipeline = MagicMock(spec=SignalPipeline)
    pipeline.run.return_value = pl.DataFrame({"open_time": [1, 2]})
    predictions = MagicMock(spec=PredictionRepository)
    predictions.load.return_value = pl.DataFrame({"open_time": [1, 2]})
    signals = MagicMock(spec=SignalRepository)
    signals.exists.return_value = True

    summary = _run(
        run_generation(
            pipeline=pipeline,
            prediction_repository=predictions,
            signal_repository=signals,
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
    assert pipeline.run.call_args.args[0] == _POLICY
    assert "OK BTCUSDT 1h 2024 rows=2" in captured


def test_run_generation_passes_regression_policy_name(
    tmp_path: Path,
) -> None:
    """run_generation forwards the selected regression policy to the pipeline."""
    work = (DiscoveredWorkItem(symbol="BTCUSDT", timeframe="1h", years=(2024,)),)
    pipeline = MagicMock(spec=SignalPipeline)
    pipeline.run.return_value = pl.DataFrame({"open_time": [1]})
    predictions = MagicMock(spec=PredictionRepository)
    predictions.load.return_value = pl.DataFrame({"open_time": [1]})
    signals = MagicMock(spec=SignalRepository)
    signals.exists.return_value = False

    summary = _run(
        run_generation(
            pipeline=pipeline,
            prediction_repository=predictions,
            signal_repository=signals,
            options=_options(
                storage_root=tmp_path,
                policy=_POLICY_REGRESSION,
                workers=1,
                overwrite=True,
            ),
            work=work,
            framework=_FRAMEWORK,
        )
    )

    assert summary.policy == _POLICY_REGRESSION
    assert summary.successful_tasks == 1
    assert pipeline.run.call_args.args[0] == _POLICY_REGRESSION


def test_run_generation_success_and_progress(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Successful generation increments counters and prints progress."""
    work = (DiscoveredWorkItem(symbol="BTCUSDT", timeframe="1h", years=(2024,)),)
    pipeline = MagicMock(spec=SignalPipeline)
    pipeline.run.return_value = pl.DataFrame({"open_time": [1, 2, 3]})
    predictions = MagicMock(spec=PredictionRepository)
    predictions.load.return_value = pl.DataFrame({"open_time": [1, 2, 3]})
    signals = MagicMock(spec=SignalRepository)
    signals.exists.return_value = False

    summary = _run(
        run_generation(
            pipeline=pipeline,
            prediction_repository=predictions,
            signal_repository=signals,
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
    pipeline = MagicMock(spec=SignalPipeline)
    pipeline.run.side_effect = [RuntimeError("boom"), pl.DataFrame({"open_time": [1]})]
    predictions = MagicMock(spec=PredictionRepository)
    predictions.load.return_value = pl.DataFrame({"open_time": [1]})
    signals = MagicMock(spec=SignalRepository)
    signals.exists.return_value = False

    summary = _run(
        run_generation(
            pipeline=pipeline,
            prediction_repository=predictions,
            signal_repository=signals,
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


def test_format_summary_includes_policy_model_and_failed_tasks() -> None:
    """Summary rendering includes policy, model identity, and failed-task labels."""
    text = format_summary(
        SignalGenerationSummary(
            policy=_POLICY_REGRESSION,
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
            output_directory=Path("data/signals"),
            failed_task_labels=("BTCUSDT 1h 2024",),
        )
    )
    assert "CQROS Signal Generation Summary" in text
    assert f"Policy: {_POLICY_REGRESSION}" in text
    assert f"Model: {_MODEL}" in text
    assert f"Version: {_VERSION}" in text
    assert "Failed tasks: 1" in text
    assert "- BTCUSDT 1h 2024" in text
    assert "Output directory: data/signals" in text


def test_main_exit_code_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """main returns 0 when no tasks fail."""
    artifact = ModelArtifactRef(
        framework=_FRAMEWORK,
        model_name=_MODEL,
        version=_VERSION,
    )
    with (
        patch("cqros.cli.generate_signals.DEFAULT_STORAGE_ROOT", str(tmp_path)),
        patch(
            "cqros.cli.generate_signals.resolve_model_artifact",
            return_value=artifact,
        ),
        patch("cqros.cli.generate_signals.build_signal_pipeline") as build_pipeline,
        patch("cqros.cli.generate_signals.discover_work", return_value=()),
    ):
        build_pipeline.return_value = MagicMock(spec=SignalPipeline)
        code = _run(
            main(
                [
                    "--policy",
                    _POLICY,
                    "--model",
                    _MODEL,
                    "--version",
                    _VERSION,
                    "--workers",
                    "1",
                ]
            )
        )

    captured = capsys.readouterr()
    assert code == 0
    assert "CQROS Signal Generation Summary" in captured.out
    assert f"Policy: {_POLICY}" in captured.out


def test_main_unknown_policy_fails(tmp_path: Path) -> None:
    """Unknown --policy fails before generation with a non-zero exit code."""
    artifact = ModelArtifactRef(
        framework=_FRAMEWORK,
        model_name=_MODEL,
        version=_VERSION,
    )
    with (
        patch("cqros.cli.generate_signals.DEFAULT_STORAGE_ROOT", str(tmp_path)),
        patch(
            "cqros.cli.generate_signals.resolve_model_artifact",
            return_value=artifact,
        ),
    ):
        code = _run(
            main(
                [
                    "--policy",
                    "unknown_policy",
                    "--model",
                    _MODEL,
                    "--version",
                    _VERSION,
                    "--workers",
                    "1",
                ]
            )
        )
    assert code == 1


def test_main_validation_error_exit_code() -> None:
    """Fatal CLI validation errors return exit code 1."""
    code = _run(
        main(
            [
                "--policy",
                _POLICY,
                "--model",
                _MODEL,
                "--version",
                _VERSION,
                "--workers",
                "0",
            ]
        )
    )
    assert code == 1


def test_configure_logging_verbose(tmp_path: Path) -> None:
    """--verbose enables INFO logging for the cqros logger."""
    artifact = ModelArtifactRef(
        framework=_FRAMEWORK,
        model_name=_MODEL,
        version=_VERSION,
    )
    with (
        patch("cqros.cli.generate_signals.DEFAULT_STORAGE_ROOT", str(tmp_path)),
        patch(
            "cqros.cli.generate_signals.resolve_model_artifact",
            return_value=artifact,
        ),
        patch("cqros.cli.generate_signals.build_signal_pipeline") as build_pipeline,
        patch("cqros.cli.generate_signals.discover_work", return_value=()),
    ):
        build_pipeline.return_value = MagicMock(spec=SignalPipeline)
        _run(
            main(
                [
                    "--policy",
                    _POLICY,
                    "--model",
                    _MODEL,
                    "--version",
                    _VERSION,
                    "--verbose",
                ]
            )
        )
    assert logging.getLogger("cqros").level == logging.INFO


def test_configure_logging_debug(tmp_path: Path) -> None:
    """--debug enables DEBUG logging for the cqros logger."""
    artifact = ModelArtifactRef(
        framework=_FRAMEWORK,
        model_name=_MODEL,
        version=_VERSION,
    )
    with (
        patch("cqros.cli.generate_signals.DEFAULT_STORAGE_ROOT", str(tmp_path)),
        patch(
            "cqros.cli.generate_signals.resolve_model_artifact",
            return_value=artifact,
        ),
        patch("cqros.cli.generate_signals.build_signal_pipeline") as build_pipeline,
        patch("cqros.cli.generate_signals.discover_work", return_value=()),
    ):
        build_pipeline.return_value = MagicMock(spec=SignalPipeline)
        _run(
            main(
                [
                    "--policy",
                    _POLICY,
                    "--model",
                    _MODEL,
                    "--version",
                    _VERSION,
                    "--debug",
                ]
            )
        )
    assert logging.getLogger("cqros").level == logging.DEBUG


def test_signal_task_result_fields() -> None:
    """SignalTaskResult stores status metadata immutably."""
    result = SignalTaskResult(
        symbol="BTCUSDT",
        timeframe="1h",
        year=2024,
        status="failed",
        error_type="RuntimeError",
        error_message="boom",
    )
    assert result.status == "failed"
    assert result.error_type == "RuntimeError"


def test_build_default_policy_returns_classification_policy() -> None:
    """Default composition-root policy is ClassificationSignalPolicy."""
    policy = build_default_policy()
    assert isinstance(policy, ClassificationSignalPolicy)


def test_build_regression_policy_returns_regression_policy() -> None:
    """Regression composition-root policy is RegressionSignalPolicy."""
    policy = build_regression_policy()
    assert isinstance(policy, RegressionSignalPolicy)


def test_build_adaptive_regression_policy_returns_adaptive_policy() -> None:
    """Adaptive regression composition-root policy is AdaptiveRegressionSignalPolicy."""
    policy = build_adaptive_regression_policy()
    assert isinstance(policy, AdaptiveRegressionSignalPolicy)


def test_build_policy_registry_registers_builtin_policies() -> None:
    """Default registry exposes classification, regression, and adaptive policies."""
    registry = build_policy_registry()
    assert registry.exists(_POLICY)
    assert registry.exists(_POLICY_REGRESSION)
    assert registry.exists(_POLICY_ADAPTIVE_REGRESSION)
    assert isinstance(registry.get(_POLICY), ClassificationSignalPolicy)
    assert isinstance(registry.get(_POLICY_REGRESSION), RegressionSignalPolicy)
    assert isinstance(
        registry.get(_POLICY_ADAPTIVE_REGRESSION),
        AdaptiveRegressionSignalPolicy,
    )
    assert registry.list() == (
        _POLICY,
        _POLICY_REGRESSION,
        _POLICY_ADAPTIVE_REGRESSION,
    )


def test_build_signal_pipeline_wires_registry(tmp_path: Path) -> None:
    """Pipeline composition injects the supplied policy registry and repository."""
    registry = SignalPolicyRegistry()
    registry.register(_POLICY, ClassificationSignalPolicy())
    signal_repository = MagicMock(spec=SignalRepository)

    with patch("cqros.cli.generate_signals.SignalPipeline") as pipeline_cls:
        pipeline_cls.return_value = MagicMock(spec=SignalPipeline)
        build_signal_pipeline(
            _options(storage_root=tmp_path),
            policy_registry=registry,
            signal_repository=signal_repository,
        )

    pipeline_cls.assert_called_once_with(
        signal_repository,
        registry,
        logger=generate_signals_module._logger,
    )


def test_regression_policy_actually_used_through_pipeline(tmp_path: Path) -> None:
    """CLI-composed registry resolves and wires RegressionSignalPolicy."""
    registry = build_policy_registry()
    policy = registry.get(_POLICY_REGRESSION)
    assert isinstance(policy, RegressionSignalPolicy)

    with patch("cqros.cli.generate_signals.SignalPipeline") as pipeline_cls:
        pipeline_cls.return_value = MagicMock(spec=SignalPipeline)
        build_signal_pipeline(
            _options(storage_root=tmp_path, policy=_POLICY_REGRESSION),
            policy_registry=registry,
            signal_repository=MagicMock(spec=SignalRepository),
        )

    wired_registry = pipeline_cls.call_args.args[1]
    assert wired_registry.get(_POLICY_REGRESSION) is policy
    assert isinstance(wired_registry.get(_POLICY_REGRESSION), RegressionSignalPolicy)


def test_adaptive_regression_policy_actually_used_through_pipeline(
    tmp_path: Path,
) -> None:
    """CLI-composed registry resolves and wires AdaptiveRegressionSignalPolicy."""
    registry = build_policy_registry()
    policy = registry.get(_POLICY_ADAPTIVE_REGRESSION)
    assert isinstance(policy, AdaptiveRegressionSignalPolicy)

    with patch("cqros.cli.generate_signals.SignalPipeline") as pipeline_cls:
        pipeline_cls.return_value = MagicMock(spec=SignalPipeline)
        build_signal_pipeline(
            _options(storage_root=tmp_path, policy=_POLICY_ADAPTIVE_REGRESSION),
            policy_registry=registry,
            signal_repository=MagicMock(spec=SignalRepository),
        )

    wired_registry = pipeline_cls.call_args.args[1]
    assert wired_registry.get(_POLICY_ADAPTIVE_REGRESSION) is policy
    assert isinstance(
        wired_registry.get(_POLICY_ADAPTIVE_REGRESSION),
        AdaptiveRegressionSignalPolicy,
    )
