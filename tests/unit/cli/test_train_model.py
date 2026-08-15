"""Unit tests for CQROS model-training CLI."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cqros.cli import train_model as train_model_module
from cqros.cli.train_model import (
    ModelTrainingOptions,
    ModelTrainingSummary,
    build_options,
    build_parser,
    build_training_workflow,
    format_summary,
    main,
    run_training,
)
from cqros.core.constants import DEFAULT_STORAGE_ROOT, STORAGE_DIR_MODELS
from cqros.core.exceptions import ValidationError
from cqros.ml.evaluation.interfaces import EvaluationResult
from cqros.ml.experiments import ExperimentTracker
from cqros.ml.models import (
    ModelArtifactRepository,
    ModelFramework,
    ModelMetadata,
    ModelTaskType,
    ModelValidationError,
)
from cqros.ml.training.interfaces import TrainerResult
from cqros.ml.workflow import TrainingWorkflow, WorkflowResult
from cqros.ml.workflow.exceptions import ModelValidationError as WorkflowValidationError

_FRAMEWORK = "lightgbm"
_MODEL = "alpha-lgbm"
_VERSION = "1.0.0"
_TASK = "regression"
_LABEL = "future_return_1"


def _options(
    *,
    storage_root: Path,
    framework: str = _FRAMEWORK,
    model: str = _MODEL,
    version: str = _VERSION,
    task: str = _TASK,
    label: str = _LABEL,
    scaler: str = "identity",
    symbols: tuple[str, ...] | None = None,
    timeframes: tuple[str, ...] | None = None,
    years: tuple[int, ...] | None = None,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    cross_validation_folds: int | None = None,
    hyperparameter_search: bool = False,
    overwrite: bool = False,
    verbose: bool = False,
    debug: bool = False,
) -> ModelTrainingOptions:
    """Build options for tests against a temporary storage root."""
    return ModelTrainingOptions(
        storage_root=storage_root,
        framework=framework,
        model=model,
        version=version,
        task=task,
        label=label,
        scaler=scaler,
        symbols=symbols,
        timeframes=timeframes,
        years=years,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        cross_validation_folds=cross_validation_folds,
        hyperparameter_search=hyperparameter_search,
        overwrite=overwrite,
        verbose=verbose,
        debug=debug,
    )


def _required_argv(
    *,
    framework: str = _FRAMEWORK,
    model: str = _MODEL,
    version: str = _VERSION,
    task: str = _TASK,
    label: str = _LABEL,
) -> list[str]:
    """Build the required CLI argument vector."""
    return [
        "--framework",
        framework,
        "--model",
        model,
        "--version",
        version,
        "--task",
        task,
        "--label",
        label,
    ]


def _metadata() -> ModelMetadata:
    """Build model metadata for workflow result stubs."""
    return ModelMetadata(
        name=_MODEL,
        version=_VERSION,
        framework=ModelFramework.LIGHTGBM,
        task_type=ModelTaskType.REGRESSION,
        feature_columns=("returns", "log_returns", "atr"),
        label_column=_LABEL,
        description="test model",
    )


def _workflow_result(
    *,
    experiment_id: str = f"{_FRAMEWORK}:{_MODEL}:{_VERSION}",
    train_rows: int = 70,
    validation_rows: int = 15,
    test_rows: int = 15,
) -> WorkflowResult:
    """Build a minimal WorkflowResult for CLI orchestration tests."""
    metadata = _metadata()
    fitted_model = MagicMock()
    fitted_model.metadata.return_value = metadata
    train_result = TrainerResult(
        model_metadata=metadata,
        train_rows=train_rows,
        validation_rows=validation_rows,
        test_rows=test_rows,
        feature_count=len(metadata.feature_columns),
        label_column=metadata.label_column,
        training_duration=0.25,
        fitted_model=fitted_model,
    )
    validation_result = EvaluationResult(
        model_metadata=metadata,
        task_type=ModelTaskType.REGRESSION,
        dataset_rows=validation_rows,
        metrics={"mae": 0.1, "mse": 0.2, "rmse": 0.447214, "r2": 0.9},
        evaluation_duration=0.01,
    )
    test_result = EvaluationResult(
        model_metadata=metadata,
        task_type=ModelTaskType.REGRESSION,
        dataset_rows=test_rows,
        metrics={"mae": 0.11, "mse": 0.21, "rmse": 0.458258, "r2": 0.88},
        evaluation_duration=0.01,
    )
    return WorkflowResult(
        experiment_id=experiment_id,
        model_metadata=metadata,
        scaler=MagicMock(),
        train_result=train_result,
        validation_result=validation_result,
        test_result=test_result,
        cross_validation_result=None,
        optimization_result=None,
        duration=1.234,
    )


def test_package_exports() -> None:
    """Public CLI symbols are exported through module ``__all__``."""
    expected = {
        "ModelTrainingOptions",
        "ModelTrainingSummary",
        "build_options",
        "build_parser",
        "build_training_workflow",
        "format_summary",
        "main",
        "run_training",
    }
    assert expected.issubset(set(train_model_module.__all__))
    assert train_model_module.build_parser is build_parser
    assert train_model_module.main is main


def test_build_parser_requires_identity_flags() -> None:
    """Required training identity flags are enforced by argparse."""
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_build_parser_accepts_all_flags() -> None:
    """Parser accepts every documented model-training flag."""
    args = build_parser().parse_args(
        [
            *_required_argv(),
            "--scaler",
            "standard",
            "--symbols",
            "BTCUSDT",
            "ETHUSDT",
            "--timeframes",
            "1h",
            "4h",
            "--years",
            "2024",
            "2025",
            "--train-ratio",
            "0.6",
            "--validation-ratio",
            "0.2",
            "--test-ratio",
            "0.2",
            "--cross-validation",
            "3",
            "--hyperparameter-search",
            "--overwrite",
            "--verbose",
            "--debug",
        ]
    )
    assert args.framework == _FRAMEWORK
    assert args.model == _MODEL
    assert args.version == _VERSION
    assert args.task == _TASK
    assert args.label == _LABEL
    assert args.scaler == "standard"
    assert args.symbols == ["BTCUSDT", "ETHUSDT"]
    assert args.timeframes == ["1h", "4h"]
    assert args.years == ["2024", "2025"]
    assert args.train_ratio == 0.6
    assert args.validation_ratio == 0.2
    assert args.test_ratio == 0.2
    assert args.cross_validation == 3
    assert args.hyperparameter_search is True
    assert args.overwrite is True
    assert args.verbose is True
    assert args.debug is True


def test_build_options_maps_flags() -> None:
    """Explicit CLI flags map onto ModelTrainingOptions."""
    options = build_options(
        build_parser().parse_args(
            [
                *_required_argv(model="beta", version="2.0.0"),
                "--scaler",
                "minmax",
                "--symbols",
                "ETHUSDT",
                "--timeframes",
                "1d",
                "--years",
                "2023",
                "--train-ratio",
                "0.8",
                "--validation-ratio",
                "0.1",
                "--test-ratio",
                "0.1",
                "--cross-validation",
                "4",
                "--hyperparameter-search",
                "--overwrite",
                "--debug",
            ]
        )
    )
    assert options.storage_root == Path(DEFAULT_STORAGE_ROOT)
    assert options.framework == _FRAMEWORK
    assert options.model == "beta"
    assert options.version == "2.0.0"
    assert options.scaler == "minmax"
    assert options.symbols == ("ETHUSDT",)
    assert options.timeframes == ("1d",)
    assert options.years == (2023,)
    assert options.train_ratio == 0.8
    assert options.validation_ratio == 0.1
    assert options.test_ratio == 0.1
    assert options.cross_validation_folds == 4
    assert options.hyperparameter_search is True
    assert options.overwrite is True
    assert options.debug is True


def test_build_options_rejects_invalid_framework() -> None:
    """Unsupported --framework values fail validation."""
    args = build_parser().parse_args([*_required_argv(framework="tensorflow")])
    with pytest.raises(ValidationError, match="invalid framework"):
        build_options(args)


def test_build_options_rejects_invalid_model() -> None:
    """Blank --model values fail validation."""
    args = build_parser().parse_args([*_required_argv(model="   ")])
    with pytest.raises(ValidationError, match="model must be a non-empty string"):
        build_options(args)


def test_build_options_rejects_invalid_label() -> None:
    """Unknown --label values fail validation."""
    args = build_parser().parse_args([*_required_argv(label="not_a_label")])
    with pytest.raises(ValidationError, match="invalid label"):
        build_options(args)


def test_build_options_rejects_label_task_mismatch() -> None:
    """Classification labels are rejected for regression tasks."""
    args = build_parser().parse_args([*_required_argv(label="direction_1")])
    with pytest.raises(ValidationError, match="invalid label for regression"):
        build_options(args)


def test_build_options_rejects_ratio_sum() -> None:
    """Split ratios that do not sum to 1.0 fail validation."""
    args = build_parser().parse_args(
        [
            *_required_argv(),
            "--train-ratio",
            "0.5",
            "--validation-ratio",
            "0.5",
            "--test-ratio",
            "0.5",
        ]
    )
    with pytest.raises(ValidationError, match="must sum to 1.0"):
        build_options(args)


def test_format_summary_includes_identity_and_metrics() -> None:
    """Summary rendering includes identity, metrics, and persistence flags."""
    text = format_summary(
        ModelTrainingSummary(
            framework=_FRAMEWORK,
            model=_MODEL,
            version=_VERSION,
            task=_TASK,
            label=_LABEL,
            training_rows=70,
            validation_rows=15,
            test_rows=15,
            duration_seconds=1.5,
            validation_metrics={"mae": 0.1, "mse": 0.2},
            test_metrics={"mae": 0.11, "mse": 0.21},
            model_saved=True,
            experiment_recorded=True,
            experiment_id=f"{_FRAMEWORK}:{_MODEL}:{_VERSION}",
            output_directory=Path("data/models"),
        )
    )
    assert "CQROS Model Training Summary" in text
    assert f"Framework: {_FRAMEWORK}" in text
    assert f"Model: {_MODEL}" in text
    assert f"Version: {_VERSION}" in text
    assert f"Task: {_TASK}" in text
    assert f"Label: {_LABEL}" in text
    assert "Training rows: 70" in text
    assert "Validation rows: 15" in text
    assert "Test rows: 15" in text
    assert "Training duration: 1.500s" in text
    assert "Validation metrics: mae=0.100000 mse=0.200000" in text
    assert "Test metrics: mae=0.110000 mse=0.210000" in text
    assert "Model saved: True" in text
    assert "Experiment recorded: True" in text
    assert "Output directory: data/models" in text


def test_run_training_successful(
    tmp_path: Path,
) -> None:
    """Successful training persists the model and records the experiment."""
    options = _options(storage_root=tmp_path)
    experiment_id = f"{_FRAMEWORK}:{_MODEL}:{_VERSION}"
    result = _workflow_result(experiment_id=experiment_id)

    workflow = MagicMock(spec=TrainingWorkflow)
    workflow.run.return_value = result
    model_repository = MagicMock(spec=ModelArtifactRepository)
    model_repository.exists.return_value = False
    tracker = MagicMock(spec=ExperimentTracker)
    tracker.exists.return_value = True

    summary = run_training(
        workflow=workflow,
        model_repository=model_repository,
        experiment_tracker=tracker,
        options=options,
    )

    workflow.run.assert_called_once()
    call_kwargs = workflow.run.call_args.kwargs
    assert call_kwargs["model_name"] == _MODEL
    assert call_kwargs["experiment_id"] == experiment_id
    assert call_kwargs["train_ratio"] == 0.70
    assert call_kwargs["parameter_grid"] is None
    model_repository.save.assert_called_once_with(result.train_result.fitted_model)
    assert summary.model_saved is True
    assert summary.experiment_recorded is True
    assert summary.training_rows == 70
    assert summary.validation_rows == 15
    assert summary.test_rows == 15
    assert summary.output_directory == tmp_path / STORAGE_DIR_MODELS


def test_run_training_empty_dataset(tmp_path: Path) -> None:
    """Empty-dataset workflow failures are reported as dataset not found."""
    options = _options(storage_root=tmp_path)
    workflow = MagicMock(spec=TrainingWorkflow)
    workflow.run.side_effect = WorkflowValidationError(
        "dataset must contain at least one row",
        error_code="ML-WORKFLOW-011",
        details={"rows": 0},
    )
    model_repository = MagicMock(spec=ModelArtifactRepository)
    model_repository.exists.return_value = False
    tracker = MagicMock(spec=ExperimentTracker)

    with pytest.raises(ValidationError, match="dataset not found or empty"):
        run_training(
            workflow=workflow,
            model_repository=model_repository,
            experiment_tracker=tracker,
            options=options,
        )
    model_repository.save.assert_not_called()


def test_run_training_workflow_failure(tmp_path: Path) -> None:
    """Non-empty-dataset workflow failures are reported as workflow failure."""
    options = _options(storage_root=tmp_path)
    workflow = MagicMock(spec=TrainingWorkflow)
    workflow.run.side_effect = WorkflowValidationError(
        "model not registered: missing",
        error_code="ML-WORKFLOW-010",
        details={"model_name": "missing"},
    )
    model_repository = MagicMock(spec=ModelArtifactRepository)
    model_repository.exists.return_value = False
    tracker = MagicMock(spec=ExperimentTracker)

    with pytest.raises(ValidationError, match="workflow failure"):
        run_training(
            workflow=workflow,
            model_repository=model_repository,
            experiment_tracker=tracker,
            options=options,
        )


def test_run_training_duplicate_version_without_overwrite(tmp_path: Path) -> None:
    """Existing model versions are rejected unless --overwrite is set."""
    options = _options(storage_root=tmp_path, overwrite=False)
    workflow = MagicMock(spec=TrainingWorkflow)
    model_repository = MagicMock(spec=ModelArtifactRepository)
    model_repository.exists.return_value = True
    tracker = MagicMock(spec=ExperimentTracker)

    with pytest.raises(ValidationError, match="duplicate model version"):
        run_training(
            workflow=workflow,
            model_repository=model_repository,
            experiment_tracker=tracker,
            options=options,
        )
    workflow.run.assert_not_called()
    model_repository.save.assert_not_called()


def test_run_training_overwrite_existing(tmp_path: Path) -> None:
    """--overwrite allows replacing an existing model version."""
    options = _options(storage_root=tmp_path, overwrite=True)
    result = _workflow_result()
    workflow = MagicMock(spec=TrainingWorkflow)
    workflow.run.return_value = result
    model_repository = MagicMock(spec=ModelArtifactRepository)
    model_repository.exists.return_value = True
    tracker = MagicMock(spec=ExperimentTracker)
    tracker.exists.return_value = True

    summary = run_training(
        workflow=workflow,
        model_repository=model_repository,
        experiment_tracker=tracker,
        options=options,
    )

    workflow.run.assert_called_once()
    model_repository.save.assert_called_once()
    assert summary.model_saved is True


def test_run_training_repository_save(tmp_path: Path) -> None:
    """run_training persists the fitted model through ModelArtifactRepository."""
    options = _options(storage_root=tmp_path)
    result = _workflow_result()
    workflow = MagicMock(spec=TrainingWorkflow)
    workflow.run.return_value = result
    model_repository = MagicMock(spec=ModelArtifactRepository)
    model_repository.exists.return_value = False
    tracker = MagicMock(spec=ExperimentTracker)
    tracker.exists.return_value = True

    run_training(
        workflow=workflow,
        model_repository=model_repository,
        experiment_tracker=tracker,
        options=options,
    )

    model_repository.save.assert_called_once_with(result.train_result.fitted_model)


def test_run_training_persistence_failure(tmp_path: Path) -> None:
    """Persistence exceptions are reported as persistence failures."""
    options = _options(storage_root=tmp_path)
    result = _workflow_result()
    workflow = MagicMock(spec=TrainingWorkflow)
    workflow.run.return_value = result
    model_repository = MagicMock(spec=ModelArtifactRepository)
    model_repository.exists.return_value = False
    model_repository.save.side_effect = RuntimeError("disk full")
    tracker = MagicMock(spec=ExperimentTracker)

    with pytest.raises(ValidationError, match="persistence failure"):
        run_training(
            workflow=workflow,
            model_repository=model_repository,
            experiment_tracker=tracker,
            options=options,
        )


def test_run_training_experiment_recording(tmp_path: Path) -> None:
    """Summary reflects whether the experiment tracker recorded the run."""
    options = _options(storage_root=tmp_path)
    experiment_id = f"{_FRAMEWORK}:{_MODEL}:{_VERSION}"
    result = _workflow_result(experiment_id=experiment_id)
    workflow = MagicMock(spec=TrainingWorkflow)
    workflow.run.return_value = result
    model_repository = MagicMock(spec=ModelArtifactRepository)
    model_repository.exists.return_value = False
    tracker = MagicMock(spec=ExperimentTracker)
    tracker.exists.return_value = True

    summary = run_training(
        workflow=workflow,
        model_repository=model_repository,
        experiment_tracker=tracker,
        options=options,
    )

    tracker.exists.assert_called_once_with(experiment_id)
    assert summary.experiment_recorded is True
    assert summary.experiment_id == experiment_id


def test_run_training_hyperparameter_search_kwargs(tmp_path: Path) -> None:
    """--hyperparameter-search forwards HPO arguments to TrainingWorkflow."""
    options = _options(storage_root=tmp_path, hyperparameter_search=True)
    result = _workflow_result()
    workflow = MagicMock(spec=TrainingWorkflow)
    workflow.run.return_value = result
    model_repository = MagicMock(spec=ModelArtifactRepository)
    model_repository.exists.return_value = False
    tracker = MagicMock(spec=ExperimentTracker)
    tracker.exists.return_value = True

    run_training(
        workflow=workflow,
        model_repository=model_repository,
        experiment_tracker=tracker,
        options=options,
    )

    call_kwargs = workflow.run.call_args.kwargs
    assert call_kwargs["parameter_grid"] == {"num_boost_round": (50, 100)}
    assert call_kwargs["optimization_metric"] == "mae"
    assert call_kwargs["optimization_folds"] == 3


def test_build_training_workflow_registers_model(tmp_path: Path) -> None:
    """Workflow composition registers a framework model for the CLI options."""
    options = _options(storage_root=tmp_path)
    training_repository = MagicMock()
    workflow, tracker, registry = build_training_workflow(
        options,
        training_repository=training_repository,
    )
    assert isinstance(workflow, TrainingWorkflow)
    assert isinstance(tracker, ExperimentTracker)
    assert registry.exists(_MODEL) is True
    metadata = registry.get(_MODEL).metadata()
    assert metadata.framework is ModelFramework.LIGHTGBM
    assert metadata.version == _VERSION
    assert metadata.label_column == _LABEL


def test_main_successful_training(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """main returns 0 and prints the training summary on success."""
    result = _workflow_result()
    workflow = MagicMock(spec=TrainingWorkflow)
    workflow.run.return_value = result
    tracker = MagicMock(spec=ExperimentTracker)
    tracker.exists.return_value = True
    registry = MagicMock()

    with (
        patch("cqros.cli.train_model.DEFAULT_STORAGE_ROOT", str(tmp_path)),
        patch(
            "cqros.cli.train_model.build_training_workflow",
            return_value=(workflow, tracker, registry),
        ),
        patch("cqros.cli.train_model.ModelArtifactRepository") as repository_cls,
    ):
        repository = MagicMock(spec=ModelArtifactRepository)
        repository.exists.return_value = False
        repository_cls.return_value = repository
        code = main([*_required_argv(), "--verbose"])

    captured = capsys.readouterr()
    assert code == 0
    assert "CQROS Model Training Summary" in captured.out
    repository.save.assert_called_once()


def test_main_invalid_framework_exit_code(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Invalid framework values return exit code 1."""
    code = main([*_required_argv(framework="pytorch")])
    captured = capsys.readouterr()
    assert code == 1
    assert "invalid framework" in captured.err


def test_main_verbose_logging(tmp_path: Path) -> None:
    """--verbose enables INFO logging for the cqros logger."""
    result = _workflow_result()
    workflow = MagicMock(spec=TrainingWorkflow)
    workflow.run.return_value = result
    tracker = MagicMock(spec=ExperimentTracker)
    tracker.exists.return_value = True
    registry = MagicMock()

    with (
        patch("cqros.cli.train_model.DEFAULT_STORAGE_ROOT", str(tmp_path)),
        patch(
            "cqros.cli.train_model.build_training_workflow",
            return_value=(workflow, tracker, registry),
        ),
        patch("cqros.cli.train_model.ModelArtifactRepository") as repository_cls,
    ):
        repository = MagicMock(spec=ModelArtifactRepository)
        repository.exists.return_value = False
        repository_cls.return_value = repository
        main([*_required_argv(), "--verbose"])

    assert logging.getLogger("cqros").level == logging.INFO


def test_main_debug_logging(tmp_path: Path) -> None:
    """--debug enables DEBUG logging for the cqros logger."""
    result = _workflow_result()
    workflow = MagicMock(spec=TrainingWorkflow)
    workflow.run.return_value = result
    tracker = MagicMock(spec=ExperimentTracker)
    tracker.exists.return_value = True
    registry = MagicMock()

    with (
        patch("cqros.cli.train_model.DEFAULT_STORAGE_ROOT", str(tmp_path)),
        patch(
            "cqros.cli.train_model.build_training_workflow",
            return_value=(workflow, tracker, registry),
        ),
        patch("cqros.cli.train_model.ModelArtifactRepository") as repository_cls,
    ):
        repository = MagicMock(spec=ModelArtifactRepository)
        repository.exists.return_value = False
        repository_cls.return_value = repository
        main([*_required_argv(), "--debug"])

    assert logging.getLogger("cqros").level == logging.DEBUG


def test_main_workflow_failure_exit_code(tmp_path: Path) -> None:
    """Workflow failures return exit code 1."""
    workflow = MagicMock(spec=TrainingWorkflow)
    workflow.run.side_effect = WorkflowValidationError(
        "boom",
        error_code="ML-WORKFLOW-010",
        details={},
    )
    tracker = MagicMock(spec=ExperimentTracker)
    registry = MagicMock()

    with (
        patch("cqros.cli.train_model.DEFAULT_STORAGE_ROOT", str(tmp_path)),
        patch(
            "cqros.cli.train_model.build_training_workflow",
            return_value=(workflow, tracker, registry),
        ),
        patch("cqros.cli.train_model.ModelArtifactRepository") as repository_cls,
    ):
        repository = MagicMock(spec=ModelArtifactRepository)
        repository.exists.return_value = False
        repository_cls.return_value = repository
        code = main(_required_argv())

    assert code == 1


def test_model_validation_error_alias() -> None:
    """Workflow ModelValidationError remains the shared CQROS model error."""
    assert WorkflowValidationError is ModelValidationError
