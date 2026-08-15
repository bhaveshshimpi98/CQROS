"""CQROS model-training CLI.

Purpose:
    Provide an argparse-based production entry point that wires existing CQROS
    ML components and executes ``TrainingWorkflow`` end-to-end, then persists
    the trained model through ``ModelArtifactRepository``.

Responsibilities:
    - Parse CLI arguments for model training
    - Compose ``DatasetLoader``, ``DatasetSplitter``, ``DatasetScaler``,
      ``ModelRegistry``, ``ModelTrainer``, ``ModelEvaluator``,
      ``TrainingWorkflow``, ``ModelArtifactRepository``, and
      ``ExperimentTracker``
    - Execute ``TrainingWorkflow.run`` without duplicating ML logic
    - Persist the fitted model via ``ModelArtifactRepository.save``
    - Print a deterministic training summary
    - Honor ``--overwrite``, verbose, and debug logging

Dependencies:
    ``argparse``, ``logging``, ``cqros.core``, ``cqros.ml``, and
    ``cqros.storage``.

Public API:
    ``build_parser``, ``build_options``, ``build_training_workflow``,
    ``format_summary``, ``run_training``, and ``main``.

Notes:
    This module is a thin composition root. It does not implement dataset
    loading, splitting, scaling, training, evaluation, cross-validation,
    hyperparameter search, experiment-tracker internals, or persistence
    internals beyond calling existing CQROS component APIs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from cqros.core.constants import (
    DEFAULT_STORAGE_ROOT,
    STORAGE_DIR_MODELS,
    SUPPORTED_TIMEFRAMES,
)
from cqros.core.exceptions import CQROSError, ValidationError
from cqros.core.types import Symbol, Timeframe
from cqros.ml.dataset import (
    CLASSIFICATION_LABEL_COLUMNS,
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    REGRESSION_LABEL_COLUMNS,
    DatasetLoader,
    DatasetScaler,
    DatasetSplitter,
    IdentityScaler,
    MinMaxScaler,
    StandardScaler,
)
from cqros.ml.evaluation import ModelEvaluator, TimeSeriesCrossValidator
from cqros.ml.experiments import ExperimentTracker
from cqros.ml.models import (
    CatBoostModel,
    LightGBMModel,
    Model,
    ModelArtifactRepository,
    ModelFramework,
    ModelMetadata,
    ModelPersistence,
    ModelRegistry,
    ModelTaskType,
    ModelValidationError,
    XGBoostModel,
)
from cqros.ml.optimization import HyperparameterOptimizer
from cqros.ml.training import ModelTrainer
from cqros.ml.workflow import TrainingWorkflow, WorkflowResult
from cqros.storage import ParquetStore, StorageLayout, TrainingRepository

__all__ = [
    "ModelTrainingOptions",
    "ModelTrainingSummary",
    "build_options",
    "build_parser",
    "build_training_workflow",
    "format_summary",
    "main",
    "run_training",
]

_logger = logging.getLogger(__name__)

_EXIT_SUCCESS: Final[int] = 0
_EXIT_FAILURE: Final[int] = 1

_DEFAULT_TRAIN_RATIO: Final[float] = 0.70
_DEFAULT_VALIDATION_RATIO: Final[float] = 0.15
_DEFAULT_TEST_RATIO: Final[float] = 0.15
_DEFAULT_SCALER: Final[str] = "identity"
_DEFAULT_HPO_FOLDS: Final[int] = 3
_DEFAULT_HPO_NUM_BOOST_ROUNDS: Final[tuple[int, ...]] = (50, 100)
_RATIO_SUM_TOLERANCE: Final[float] = 1e-9

_SUPPORTED_FRAMEWORKS: Final[frozenset[str]] = frozenset(
    {
        str(ModelFramework.LIGHTGBM),
        str(ModelFramework.XGBOOST),
        str(ModelFramework.CATBOOST),
    }
)
_SUPPORTED_TASKS: Final[frozenset[str]] = frozenset(
    {
        str(ModelTaskType.REGRESSION),
        str(ModelTaskType.CLASSIFICATION),
    }
)
_SUPPORTED_SCALERS: Final[frozenset[str]] = frozenset(
    {
        "identity",
        "standard",
        "minmax",
    }
)

_ERROR_FRAMEWORK: Final[str] = "CLI-TRAIN-MODEL-001"
_ERROR_MODEL: Final[str] = "CLI-TRAIN-MODEL-002"
_ERROR_VERSION: Final[str] = "CLI-TRAIN-MODEL-003"
_ERROR_TASK: Final[str] = "CLI-TRAIN-MODEL-004"
_ERROR_LABEL: Final[str] = "CLI-TRAIN-MODEL-005"
_ERROR_SCALER: Final[str] = "CLI-TRAIN-MODEL-006"
_ERROR_TIMEFRAME: Final[str] = "CLI-TRAIN-MODEL-007"
_ERROR_YEAR: Final[str] = "CLI-TRAIN-MODEL-008"
_ERROR_RATIO: Final[str] = "CLI-TRAIN-MODEL-009"
_ERROR_RATIO_SUM: Final[str] = "CLI-TRAIN-MODEL-010"
_ERROR_CROSS_VALIDATION: Final[str] = "CLI-TRAIN-MODEL-011"
_ERROR_DUPLICATE_VERSION: Final[str] = "CLI-TRAIN-MODEL-012"
_ERROR_DATASET_EMPTY: Final[str] = "CLI-TRAIN-MODEL-013"
_ERROR_PERSISTENCE: Final[str] = "CLI-TRAIN-MODEL-014"
_ERROR_WORKFLOW: Final[str] = "CLI-TRAIN-MODEL-015"

_METADATA_FILENAME: Final[str] = "metadata.json"


@dataclass(frozen=True, slots=True)
class ModelTrainingOptions:
    """Immutable CLI options for model training.

    Attributes:
        storage_root: Storage root containing ``training`` and ``models``.
        framework: Machine-learning framework identifier.
        model: Stable model identifier.
        version: Model version identifier.
        task: Supervised learning task type (``regression`` or
            ``classification``).
        label: Target label column name.
        scaler: Feature scaler name (``identity``, ``standard``, or
            ``minmax``).
        symbols: Optional symbol allowlist. ``None`` loads all.
        timeframes: Optional timeframe allowlist. ``None`` loads all.
        years: Optional year allowlist. ``None`` loads all.
        train_ratio: Fraction of rows assigned to the train split.
        validation_ratio: Fraction of rows assigned to the validation split.
        test_ratio: Fraction of rows assigned to the test split.
        cross_validation_folds: Optional walk-forward fold count.
        hyperparameter_search: When ``True``, run hyperparameter optimization.
        overwrite: When ``True``, replace an existing model version.
        verbose: When ``True``, enable INFO logging.
        debug: When ``True``, enable DEBUG logging and log full failure
            tracebacks.
    """

    storage_root: Path
    framework: str
    model: str
    version: str
    task: str
    label: str
    scaler: str
    symbols: tuple[Symbol, ...] | None
    timeframes: tuple[Timeframe, ...] | None
    years: tuple[int, ...] | None
    train_ratio: float
    validation_ratio: float
    test_ratio: float
    cross_validation_folds: int | None
    hyperparameter_search: bool
    overwrite: bool
    verbose: bool
    debug: bool


@dataclass(frozen=True, slots=True)
class ModelTrainingSummary:
    """Immutable summary for one model-training CLI run.

    Attributes:
        framework: Machine-learning framework used for training.
        model: Stable model identifier.
        version: Model version identifier.
        task: Supervised learning task type.
        label: Target label column name.
        training_rows: Rows used for training.
        validation_rows: Rows used for validation.
        test_rows: Rows used for testing.
        duration_seconds: Wall-clock training workflow duration.
        validation_metrics: Validation-split metric mapping.
        test_metrics: Test-split metric mapping.
        model_saved: Whether the model artifact was persisted.
        experiment_recorded: Whether the experiment was recorded.
        experiment_id: Identifier recorded for the experiment.
        output_directory: Models-tier output directory.
    """

    framework: str
    model: str
    version: str
    task: str
    label: str
    training_rows: int
    validation_rows: int
    test_rows: int
    duration_seconds: float
    validation_metrics: Mapping[str, float]
    test_metrics: Mapping[str, float]
    model_saved: bool
    experiment_recorded: bool
    experiment_id: str
    output_directory: Path


def build_parser() -> argparse.ArgumentParser:
    """Create the model-training argument parser.

    Returns:
        Configured ``ArgumentParser`` for model-training flags.
    """
    parser = argparse.ArgumentParser(
        prog="cqros-train-model",
        description=(
            "Train, evaluate, and persist a versioned CQROS ML model using "
            "existing TrainingWorkflow orchestration."
        ),
    )
    parser.add_argument(
        "--framework",
        dest="framework",
        required=True,
        metavar="NAME",
        help=("Machine-learning framework " f"({', '.join(sorted(_SUPPORTED_FRAMEWORKS))})."),
    )
    parser.add_argument(
        "--model",
        dest="model",
        required=True,
        metavar="NAME",
        help="Stable model identifier used for registry and artifact identity.",
    )
    parser.add_argument(
        "--version",
        dest="version",
        required=True,
        metavar="VERSION",
        help="Model version identifier used for artifact persistence.",
    )
    parser.add_argument(
        "--task",
        dest="task",
        required=True,
        metavar="TASK",
        help=("Supervised task type " f"({', '.join(sorted(_SUPPORTED_TASKS))})."),
    )
    parser.add_argument(
        "--label",
        dest="label",
        required=True,
        metavar="COLUMN",
        help="Target label column from the canonical training schema.",
    )
    parser.add_argument(
        "--scaler",
        dest="scaler",
        default=_DEFAULT_SCALER,
        metavar="NAME",
        help=(
            "Feature scaler "
            f"({', '.join(sorted(_SUPPORTED_SCALERS))}; "
            f"default: {_DEFAULT_SCALER})."
        ),
    )
    parser.add_argument(
        "--symbols",
        dest="symbols",
        nargs="*",
        default=None,
        metavar="SYMBOL",
        help="Optional symbol allowlist (0..N values). Omit to load all.",
    )
    parser.add_argument(
        "--timeframes",
        dest="timeframes",
        nargs="*",
        default=None,
        metavar="TIMEFRAME",
        help="Optional timeframe allowlist (0..N values). Omit to load all.",
    )
    parser.add_argument(
        "--years",
        dest="years",
        nargs="*",
        default=None,
        metavar="YEAR",
        help="Optional calendar-year allowlist (0..N values). Omit to load all.",
    )
    parser.add_argument(
        "--train-ratio",
        dest="train_ratio",
        type=float,
        default=_DEFAULT_TRAIN_RATIO,
        metavar="FLOAT",
        help=f"Train split ratio (default: {_DEFAULT_TRAIN_RATIO}).",
    )
    parser.add_argument(
        "--validation-ratio",
        dest="validation_ratio",
        type=float,
        default=_DEFAULT_VALIDATION_RATIO,
        metavar="FLOAT",
        help=f"Validation split ratio (default: {_DEFAULT_VALIDATION_RATIO}).",
    )
    parser.add_argument(
        "--test-ratio",
        dest="test_ratio",
        type=float,
        default=_DEFAULT_TEST_RATIO,
        metavar="FLOAT",
        help=f"Test split ratio (default: {_DEFAULT_TEST_RATIO}).",
    )
    parser.add_argument(
        "--cross-validation",
        dest="cross_validation",
        type=int,
        default=None,
        metavar="FOLDS",
        help="Optional walk-forward cross-validation fold count (>= 2).",
    )
    parser.add_argument(
        "--hyperparameter-search",
        dest="hyperparameter_search",
        action="store_true",
        help="Enable hyperparameter optimization through TrainingWorkflow.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing model artifact at the same version.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO logging.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=("Enable DEBUG logging and log complete failure tracebacks " "with logger.exception."),
    )
    return parser


def build_options(args: argparse.Namespace) -> ModelTrainingOptions:
    """Map parsed CLI arguments onto ``ModelTrainingOptions``.

    Args:
        args: Namespace produced by ``build_parser().parse_args(...)``.

    Returns:
        Immutable training options.

    Raises:
        ValidationError: If any argument fails validation.
    """
    framework = _normalize_framework(args.framework)
    model = _normalize_required_string(
        args.model,
        parameter="model",
        error_code=_ERROR_MODEL,
    )
    version = _normalize_required_string(
        args.version,
        parameter="version",
        error_code=_ERROR_VERSION,
    )
    task = _normalize_task(args.task)
    label = _normalize_label(args.label, task=task)
    scaler = _normalize_scaler(args.scaler)
    train_ratio = _normalize_ratio(args.train_ratio, parameter="train_ratio")
    validation_ratio = _normalize_ratio(
        args.validation_ratio,
        parameter="validation_ratio",
    )
    test_ratio = _normalize_ratio(args.test_ratio, parameter="test_ratio")
    _validate_ratio_sum(
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )
    cross_validation_folds = _normalize_cross_validation(args.cross_validation)

    return ModelTrainingOptions(
        storage_root=Path(DEFAULT_STORAGE_ROOT),
        framework=framework,
        model=model,
        version=version,
        task=task,
        label=label,
        scaler=scaler,
        symbols=_normalize_symbols(args.symbols),
        timeframes=_normalize_timeframes(args.timeframes),
        years=_normalize_years(args.years),
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        cross_validation_folds=cross_validation_folds,
        hyperparameter_search=bool(args.hyperparameter_search),
        overwrite=bool(args.overwrite),
        verbose=bool(args.verbose),
        debug=bool(args.debug),
    )


def build_training_workflow(
    options: ModelTrainingOptions,
    *,
    training_repository: TrainingRepository | None = None,
    experiment_tracker: ExperimentTracker | None = None,
    logger: logging.Logger | None = None,
) -> tuple[TrainingWorkflow, ExperimentTracker, ModelRegistry]:
    """Compose ``TrainingWorkflow`` and supporting ML dependencies.

    Args:
        options: Immutable training options providing identity and scaler.
        training_repository: Optional training repository. When ``None``, one
            is constructed from ``options.storage_root``.
        experiment_tracker: Optional experiment tracker. When ``None``, a new
            in-memory tracker is created.
        logger: Optional logger forwarded to composed components.

    Returns:
        Tuple of ``(workflow, experiment_tracker, model_registry)``.
    """
    active_logger = logger if logger is not None else _logger
    if training_repository is None:
        layout = StorageLayout(options.storage_root)
        training_repository = TrainingRepository(layout, ParquetStore())
    tracker = experiment_tracker if experiment_tracker is not None else ExperimentTracker()

    registry = ModelRegistry()
    registry.register(_construct_model(options))

    trainer = ModelTrainer(model_registry=registry, logger=active_logger)
    evaluator = ModelEvaluator(logger=active_logger)
    cross_validator = TimeSeriesCrossValidator(
        model_registry=registry,
        model_trainer=trainer,
        model_evaluator=evaluator,
        logger=active_logger,
    )
    optimizer = HyperparameterOptimizer(
        cross_validator=cross_validator,
        logger=active_logger,
    )
    workflow = TrainingWorkflow(
        dataset_loader=DatasetLoader(training_repository, logger=active_logger),
        dataset_splitter=DatasetSplitter(logger=active_logger),
        dataset_scaler=_construct_scaler(options.scaler, logger=active_logger),
        model_registry=registry,
        model_trainer=trainer,
        model_evaluator=evaluator,
        cross_validator=cross_validator,
        hyperparameter_optimizer=optimizer,
        experiment_tracker=tracker,
        logger=active_logger,
    )
    return workflow, tracker, registry


def format_summary(summary: ModelTrainingSummary) -> str:
    """Render a deterministic model-training summary report.

    Args:
        summary: Aggregate training summary.

    Returns:
        Multi-line summary text suitable for stdout.
    """
    lines = [
        "=====================================",
        "CQROS Model Training Summary",
        "=====================================",
        "",
        f"Framework: {summary.framework}",
        f"Model: {summary.model}",
        f"Version: {summary.version}",
        f"Task: {summary.task}",
        f"Label: {summary.label}",
        "",
        f"Training rows: {summary.training_rows}",
        f"Validation rows: {summary.validation_rows}",
        f"Test rows: {summary.test_rows}",
        "",
        f"Training duration: {_format_duration(summary.duration_seconds)}",
        "",
        f"Validation metrics: {_format_metrics(summary.validation_metrics)}",
        "",
        f"Test metrics: {_format_metrics(summary.test_metrics)}",
        "",
        f"Model saved: {summary.model_saved}",
        "",
        f"Experiment recorded: {summary.experiment_recorded}",
        "",
        f"Output directory: {_format_output_directory(summary.output_directory)}",
        "",
        "=====================================",
    ]
    return "\n".join(lines) + "\n"


def run_training(
    *,
    workflow: TrainingWorkflow,
    model_repository: ModelArtifactRepository,
    experiment_tracker: ExperimentTracker,
    options: ModelTrainingOptions,
) -> ModelTrainingSummary:
    """Execute training workflow orchestration and persist the fitted model.

    Args:
        workflow: Injected training workflow.
        model_repository: Repository used to persist the trained artifact.
        experiment_tracker: Tracker used to confirm experiment recording.
        options: Immutable training options.

    Returns:
        Immutable training summary.

    Raises:
        ValidationError: If a duplicate version exists without overwrite, the
            dataset is empty, or persistence fails.
        CQROSError: If the workflow or repository raises a CQROS error.
    """
    if not options.overwrite and model_repository.exists(
        framework=options.framework,
        model_name=options.model,
        version=options.version,
    ):
        raise ValidationError(
            (f"duplicate model version: {options.framework}/" f"{options.model}/{options.version}"),
            error_code=_ERROR_DUPLICATE_VERSION,
            details={
                "framework": options.framework,
                "model": options.model,
                "version": options.version,
            },
        )

    experiment_id = _experiment_id(options)
    output_directory = options.storage_root / STORAGE_DIR_MODELS

    hpo = _hyperparameter_search_kwargs(options)
    try:
        result = workflow.run(
            model_name=options.model,
            experiment_id=experiment_id,
            train_ratio=options.train_ratio,
            validation_ratio=options.validation_ratio,
            test_ratio=options.test_ratio,
            symbols=options.symbols,
            timeframes=options.timeframes,
            years=options.years,
            cross_validation_folds=options.cross_validation_folds,
            parameter_grid=hpo.parameter_grid,
            optimization_metric=hpo.optimization_metric,
            optimization_folds=hpo.optimization_folds,
            artifact_path=str(output_directory),
            notes="cqros-train-model",
        )
    except ModelValidationError as exc:
        if exc.error_code == "ML-WORKFLOW-011":
            raise ValidationError(
                "dataset not found or empty",
                error_code=_ERROR_DATASET_EMPTY,
                details={
                    "symbols": options.symbols,
                    "timeframes": options.timeframes,
                    "years": options.years,
                    "reason": str(exc),
                },
            ) from exc
        raise ValidationError(
            f"workflow failure: {exc.message}",
            error_code=_ERROR_WORKFLOW,
            details={
                "workflow_error_code": exc.error_code,
                "workflow_details": dict(exc.details),
            },
        ) from exc
    except CQROSError:
        raise
    except Exception as exc:
        raise ValidationError(
            f"workflow failure: {exc}",
            error_code=_ERROR_WORKFLOW,
            details={"error_type": type(exc).__name__},
        ) from exc

    fitted_model = result.train_result.fitted_model
    try:
        model_repository.save(fitted_model)
    except CQROSError:
        raise
    except Exception as exc:
        raise ValidationError(
            f"persistence failure: {exc}",
            error_code=_ERROR_PERSISTENCE,
            details={
                "framework": options.framework,
                "model": options.model,
                "version": options.version,
                "error_type": type(exc).__name__,
            },
        ) from exc

    return _build_summary(
        options=options,
        result=result,
        experiment_tracker=experiment_tracker,
        experiment_id=experiment_id,
        model_saved=True,
        output_directory=output_directory,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the model-training CLI.

    Args:
        argv: Optional argument vector. When ``None``, ``sys.argv[1:]`` is used.

    Returns:
        ``0`` on success; ``1`` when a fatal CLI error occurs.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        options = build_options(args)
        _configure_logging(verbose=options.verbose, debug=options.debug)
        layout = StorageLayout(options.storage_root)
        model_repository = ModelArtifactRepository(layout, _CliModelPersistence())
        workflow, tracker, _registry = build_training_workflow(options)
        summary = run_training(
            workflow=workflow,
            model_repository=model_repository,
            experiment_tracker=tracker,
            options=options,
        )
    except CQROSError as exc:
        _report_failure(exc, debug=bool(getattr(args, "debug", False)))
        return _EXIT_FAILURE
    except Exception as exc:
        _report_failure(exc, debug=bool(getattr(args, "debug", False)))
        return _EXIT_FAILURE

    print(format_summary(summary), end="")
    return _EXIT_SUCCESS


def _configure_logging(*, verbose: bool, debug: bool) -> None:
    """Configure process logging for the CLI entry point."""
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger("cqros").setLevel(level)


def _report_failure(exc: BaseException, *, debug: bool) -> None:
    """Print a CLI failure and optionally log a full traceback."""
    print(str(exc), file=sys.stderr)
    if debug:
        _logger.exception("Model training CLI failed", exc_info=exc)


def _normalize_framework(value: object) -> str:
    """Validate and normalize ``--framework``."""
    framework = _normalize_required_string(
        value,
        parameter="framework",
        error_code=_ERROR_FRAMEWORK,
    )
    if framework not in _SUPPORTED_FRAMEWORKS:
        raise ValidationError(
            f"invalid framework: {framework}",
            error_code=_ERROR_FRAMEWORK,
            details={
                "parameter": "framework",
                "value": framework,
                "supported": tuple(sorted(_SUPPORTED_FRAMEWORKS)),
            },
        )
    return framework


def _normalize_task(value: object) -> str:
    """Validate and normalize ``--task``."""
    task = _normalize_required_string(
        value,
        parameter="task",
        error_code=_ERROR_TASK,
    )
    if task not in _SUPPORTED_TASKS:
        raise ValidationError(
            f"invalid task: {task}",
            error_code=_ERROR_TASK,
            details={
                "parameter": "task",
                "value": task,
                "supported": tuple(sorted(_SUPPORTED_TASKS)),
            },
        )
    return task


def _normalize_label(value: object, *, task: str) -> str:
    """Validate and normalize ``--label`` against the task type."""
    label = _normalize_required_string(
        value,
        parameter="label",
        error_code=_ERROR_LABEL,
    )
    if label not in LABEL_COLUMNS:
        raise ValidationError(
            f"invalid label: {label}",
            error_code=_ERROR_LABEL,
            details={
                "parameter": "label",
                "value": label,
                "supported": LABEL_COLUMNS,
            },
        )
    if task == str(ModelTaskType.REGRESSION) and label not in REGRESSION_LABEL_COLUMNS:
        raise ValidationError(
            f"invalid label for regression task: {label}",
            error_code=_ERROR_LABEL,
            details={
                "parameter": "label",
                "value": label,
                "task": task,
                "supported": REGRESSION_LABEL_COLUMNS,
            },
        )
    if task == str(ModelTaskType.CLASSIFICATION) and label not in CLASSIFICATION_LABEL_COLUMNS:
        raise ValidationError(
            f"invalid label for classification task: {label}",
            error_code=_ERROR_LABEL,
            details={
                "parameter": "label",
                "value": label,
                "task": task,
                "supported": CLASSIFICATION_LABEL_COLUMNS,
            },
        )
    return label


def _normalize_scaler(value: object) -> str:
    """Validate and normalize ``--scaler``."""
    scaler = _normalize_required_string(
        value,
        parameter="scaler",
        error_code=_ERROR_SCALER,
    )
    if scaler not in _SUPPORTED_SCALERS:
        raise ValidationError(
            f"invalid scaler: {scaler}",
            error_code=_ERROR_SCALER,
            details={
                "parameter": "scaler",
                "value": scaler,
                "supported": tuple(sorted(_SUPPORTED_SCALERS)),
            },
        )
    return scaler


def _normalize_required_string(
    value: object,
    *,
    parameter: str,
    error_code: str,
) -> str:
    """Require a non-empty stripped string argument."""
    if not isinstance(value, str) or value.strip() == "":
        raise ValidationError(
            f"{parameter} must be a non-empty string",
            error_code=error_code,
            details={"parameter": parameter, "value": value},
        )
    return value.strip()


def _normalize_ratio(value: object, *, parameter: str) -> float:
    """Validate one split ratio in ``[0, 1]``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(
            f"{parameter} must be a number",
            error_code=_ERROR_RATIO,
            details={"parameter": parameter, "value": value},
        )
    ratio = float(value)
    if ratio < 0.0 or ratio > 1.0:
        raise ValidationError(
            f"{parameter} must be between 0 and 1 inclusive",
            error_code=_ERROR_RATIO,
            details={"parameter": parameter, "value": ratio},
        )
    return ratio


def _validate_ratio_sum(
    *,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> None:
    """Require train/validation/test ratios to sum to 1.0."""
    total = train_ratio + validation_ratio + test_ratio
    if abs(total - 1.0) > _RATIO_SUM_TOLERANCE:
        raise ValidationError(
            "train_ratio, validation_ratio, and test_ratio must sum to 1.0",
            error_code=_ERROR_RATIO_SUM,
            details={
                "train_ratio": train_ratio,
                "validation_ratio": validation_ratio,
                "test_ratio": test_ratio,
                "total": total,
            },
        )


def _normalize_cross_validation(value: object) -> int | None:
    """Validate optional ``--cross-validation`` fold count."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValidationError(
            "cross_validation must be an integer greater than or equal to 2",
            error_code=_ERROR_CROSS_VALIDATION,
            details={"parameter": "cross_validation", "value": value},
        )
    return value


def _normalize_symbols(values: Sequence[str] | None) -> tuple[Symbol, ...] | None:
    """Validate and freeze optional symbol filters."""
    if values is None:
        return None
    normalized: list[Symbol] = []
    for symbol in values:
        stripped = symbol.strip()
        if stripped == "":
            continue
        if stripped not in normalized:
            normalized.append(stripped)
    return tuple(normalized) if normalized else None


def _normalize_timeframes(
    values: Sequence[str] | None,
) -> tuple[Timeframe, ...] | None:
    """Validate and freeze optional timeframe filters."""
    if values is None:
        return None
    normalized: list[Timeframe] = []
    for timeframe in values:
        if timeframe not in SUPPORTED_TIMEFRAMES:
            raise ValidationError(
                f"unsupported timeframe: {timeframe}",
                error_code=_ERROR_TIMEFRAME,
                details={"parameter": "timeframes", "value": timeframe},
            )
        if timeframe not in normalized:
            normalized.append(timeframe)
    return tuple(normalized) if normalized else None


def _normalize_years(values: Sequence[str] | None) -> tuple[int, ...] | None:
    """Validate and freeze optional year filters."""
    if values is None:
        return None
    normalized: list[int] = []
    for raw in values:
        try:
            year = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"invalid year: {raw}",
                error_code=_ERROR_YEAR,
                details={"parameter": "years", "value": raw},
            ) from exc
        if year < 1:
            raise ValidationError(
                f"invalid year: {raw}",
                error_code=_ERROR_YEAR,
                details={"parameter": "years", "value": raw},
            )
        if year not in normalized:
            normalized.append(year)
    return tuple(sorted(normalized)) if normalized else None


def _construct_scaler(
    scaler_name: str,
    *,
    logger: logging.Logger,
) -> DatasetScaler:
    """Construct the requested ``DatasetScaler`` implementation."""
    match scaler_name:
        case "identity":
            return IdentityScaler(logger=logger)
        case "standard":
            return StandardScaler(logger=logger)
        case "minmax":
            return MinMaxScaler(logger=logger)
        case _:
            raise ValidationError(
                f"invalid scaler: {scaler_name}",
                error_code=_ERROR_SCALER,
                details={"parameter": "scaler", "value": scaler_name},
            )


def _construct_model(options: ModelTrainingOptions) -> Model:
    """Construct an unfitted framework model shell for ``options``."""
    metadata = ModelMetadata(
        name=options.model,
        version=options.version,
        framework=ModelFramework(options.framework),
        task_type=ModelTaskType(options.task),
        feature_columns=FEATURE_COLUMNS,
        label_column=options.label,
        description=(
            f"CQROS {options.framework} {options.task} model targeting " f"{options.label}"
        ),
    )
    match options.framework:
        case "lightgbm":
            return LightGBMModel(model_metadata=metadata)
        case "xgboost":
            return XGBoostModel(model_metadata=metadata)
        case "catboost":
            return CatBoostModel(model_metadata=metadata)
        case _:
            raise ValidationError(
                f"invalid framework: {options.framework}",
                error_code=_ERROR_FRAMEWORK,
                details={"framework": options.framework},
            )


@dataclass(frozen=True, slots=True)
class _HyperparameterSearchArgs:
    """Optional HPO arguments forwarded to ``TrainingWorkflow.run``."""

    parameter_grid: Mapping[str, Sequence[object]] | None
    optimization_metric: str | None
    optimization_folds: int | None


def _hyperparameter_search_kwargs(
    options: ModelTrainingOptions,
) -> _HyperparameterSearchArgs:
    """Build optional HPO keyword arguments for ``TrainingWorkflow.run``."""
    if not options.hyperparameter_search:
        return _HyperparameterSearchArgs(
            parameter_grid=None,
            optimization_metric=None,
            optimization_folds=None,
        )
    metric = "accuracy" if options.task == str(ModelTaskType.CLASSIFICATION) else "mae"
    folds = (
        options.cross_validation_folds
        if options.cross_validation_folds is not None
        else _DEFAULT_HPO_FOLDS
    )
    return _HyperparameterSearchArgs(
        parameter_grid={"num_boost_round": _DEFAULT_HPO_NUM_BOOST_ROUNDS},
        optimization_metric=metric,
        optimization_folds=folds,
    )


def _experiment_id(options: ModelTrainingOptions) -> str:
    """Build a deterministic experiment identifier for one training run."""
    return f"{options.framework}:{options.model}:{options.version}"


def _build_summary(
    *,
    options: ModelTrainingOptions,
    result: WorkflowResult,
    experiment_tracker: ExperimentTracker,
    experiment_id: str,
    model_saved: bool,
    output_directory: Path,
) -> ModelTrainingSummary:
    """Aggregate workflow and persistence outcomes into a CLI summary."""
    return ModelTrainingSummary(
        framework=options.framework,
        model=options.model,
        version=options.version,
        task=options.task,
        label=options.label,
        training_rows=result.train_result.train_rows,
        validation_rows=result.train_result.validation_rows,
        test_rows=result.test_result.dataset_rows,
        duration_seconds=result.duration,
        validation_metrics=dict(result.validation_result.metrics),
        test_metrics=dict(result.test_result.metrics),
        model_saved=model_saved,
        experiment_recorded=experiment_tracker.exists(experiment_id),
        experiment_id=experiment_id,
        output_directory=output_directory,
    )


def _format_duration(duration_seconds: float) -> str:
    """Format duration seconds for the summary report."""
    return f"{duration_seconds:.3f}s"


def _format_metrics(metrics: Mapping[str, float]) -> str:
    """Format metric mappings deterministically for the summary report."""
    if len(metrics) == 0:
        return "(none)"
    return " ".join(f"{name}={metrics[name]:.6f}" for name in sorted(metrics))


def _format_output_directory(path: Path) -> str:
    """Format the output directory using POSIX separators."""
    return path.as_posix()


class _CliModelPersistence(ModelPersistence):
    """Composition-root model persistence used by the training CLI.

    Delegates serialization to framework model implementations. This class is
    not a general ML persistence backend and exists only so
    ``ModelArtifactRepository`` can save and load artifacts for CLI wiring.
    """

    def save(self, model: object, path: Path | str) -> None:
        """Persist ``model`` through its framework ``save`` implementation."""
        typed = self._require_model(model)
        typed.save(self._require_path(path))

    def load(self, path: Path | str) -> Model:
        """Load a model by reconstructing it from sibling metadata."""
        model_path = self._require_path(path)
        metadata = _load_sibling_metadata(model_path)
        model = _construct_framework_model(metadata)
        return model.load(model_path)

    def exists(self, path: Path | str) -> bool:
        """Return whether a model binary exists at ``path``."""
        return self._require_path(path).is_file()

    def delete(self, path: Path | str) -> None:
        """Delete the model binary at ``path``."""
        target = self._require_path(path)
        if not target.is_file():
            raise ModelValidationError(
                "model artifact not found",
                error_code=_ERROR_PERSISTENCE,
                details={"path": str(target)},
            )
        target.unlink()


def _load_sibling_metadata(model_path: Path) -> ModelMetadata:
    """Load ``ModelMetadata`` from ``metadata.json`` beside ``model_path``."""
    metadata_path = model_path.parent / _METADATA_FILENAME
    try:
        raw_payload: object = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelValidationError(
            "model metadata is missing or invalid",
            error_code=_ERROR_PERSISTENCE,
            details={"path": str(metadata_path), "reason": str(exc)},
        ) from exc
    if not isinstance(raw_payload, dict):
        raise ModelValidationError(
            "model metadata must be a JSON object",
            error_code=_ERROR_PERSISTENCE,
            details={"path": str(metadata_path)},
        )
    payload = cast(dict[str, object], raw_payload)
    feature_columns_raw = payload.get("feature_columns")
    if not isinstance(feature_columns_raw, list):
        raise ModelValidationError(
            "model metadata contents are invalid",
            error_code=_ERROR_PERSISTENCE,
            details={"path": str(metadata_path)},
        )
    feature_items = cast(list[object], feature_columns_raw)
    try:
        return ModelMetadata(
            name=str(payload["name"]),
            version=str(payload["version"]),
            framework=ModelFramework(str(payload["framework"])),
            task_type=ModelTaskType(str(payload["task_type"])),
            feature_columns=tuple(str(item) for item in feature_items),
            label_column=str(payload["label_column"]),
            description=str(payload["description"]),
        )
    except (KeyError, TypeError, ValueError, ModelValidationError) as exc:
        raise ModelValidationError(
            "model metadata contents are invalid",
            error_code=_ERROR_PERSISTENCE,
            details={"path": str(metadata_path), "reason": str(exc)},
        ) from exc


def _construct_framework_model(metadata: ModelMetadata) -> Model:
    """Construct an empty framework model shell for ``metadata``."""
    match metadata.framework:
        case ModelFramework.LIGHTGBM:
            return LightGBMModel(model_metadata=metadata)
        case ModelFramework.XGBOOST:
            return XGBoostModel(model_metadata=metadata)
        case ModelFramework.CATBOOST:
            return CatBoostModel(model_metadata=metadata)
        case _:
            raise ModelValidationError(
                f"unsupported model framework for CLI loading: {metadata.framework}",
                error_code=_ERROR_PERSISTENCE,
                details={"framework": str(metadata.framework)},
            )


if __name__ == "__main__":
    raise SystemExit(main())
