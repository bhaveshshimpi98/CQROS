"""Unit tests for CQROS ``HyperparameterOptimizer``."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.ml.evaluation import ModelEvaluator, TimeSeriesCrossValidator
from cqros.ml.models import (
    LightGBMModel,
    ModelFramework,
    ModelMetadata,
    ModelRegistry,
    ModelTaskType,
    ModelValidationError,
)
from cqros.ml.optimization import (
    HyperparameterOptimizer,
    OptimizationDirection,
    OptimizationResult,
    OptimizationTrial,
)
from cqros.ml.optimization.optimizer import (
    HyperparameterOptimizer as HyperparameterOptimizerDirect,
)
from cqros.ml.training import ModelTrainer


def _metadata(
    *,
    name: str = "hpo-lgbm",
    task_type: ModelTaskType = ModelTaskType.REGRESSION,
) -> ModelMetadata:
    """Build ModelMetadata for optimizer unit tests."""
    return ModelMetadata(
        name=name,
        version="1.0.0",
        framework=ModelFramework.LIGHTGBM,
        task_type=task_type,
        feature_columns=("f1", "f2"),
        label_column="label",
        description="LightGBM hyperparameter optimization test model",
    )


def _regression_frame(*, rows: int = 45) -> pl.DataFrame:
    """Build a deterministic chronological regression frame."""
    return pl.DataFrame(
        {
            "f1": [float(index) for index in range(rows)],
            "f2": [float(index) * 0.5 for index in range(rows)],
            "label": [float(index) * 1.5 + 0.1 for index in range(rows)],
        }
    )


def _classification_frame(*, rows: int = 45) -> pl.DataFrame:
    """Build a deterministic chronological classification frame."""
    return pl.DataFrame(
        {
            "f1": [float(index) for index in range(rows)],
            "f2": [float(index % 3) for index in range(rows)],
            "label": [index % 2 for index in range(rows)],
        }
    )


def _build_optimizer(
    *,
    task_type: ModelTaskType = ModelTaskType.REGRESSION,
    model_name: str = "hpo-lgbm",
    num_boost_round: int = 10,
) -> tuple[HyperparameterOptimizer, ModelRegistry]:
    """Build an optimizer wired to a LightGBM model and walk-forward CV."""
    registry = ModelRegistry()
    registry.register(
        LightGBMModel(
            model_metadata=_metadata(name=model_name, task_type=task_type),
            num_boost_round=num_boost_round,
        )
    )
    trainer = ModelTrainer(model_registry=registry)
    evaluator = ModelEvaluator()
    cross_validator = TimeSeriesCrossValidator(
        model_registry=registry,
        model_trainer=trainer,
        model_evaluator=evaluator,
    )
    optimizer = HyperparameterOptimizer(cross_validator=cross_validator)
    return optimizer, registry


def test_package_exports_hyperparameter_optimizer() -> None:
    """Optimizer types are exported from the optimization package."""
    import cqros.ml.optimization as optimization_package

    assert "HyperparameterOptimizer" in optimization_package.__all__
    assert "OptimizationResult" in optimization_package.__all__
    assert "OptimizationTrial" in optimization_package.__all__
    assert optimization_package.HyperparameterOptimizer is HyperparameterOptimizer
    assert HyperparameterOptimizer is HyperparameterOptimizerDirect


def test_successful_regression_optimization() -> None:
    """Grid search completes for a regression model and metric."""
    optimizer, _registry = _build_optimizer()
    frame = _regression_frame()

    result = optimizer.optimize(
        "hpo-lgbm",
        frame,
        parameter_grid={"num_boost_round": [8, 12]},
        metric="mae",
        folds=2,
    )

    assert isinstance(result, OptimizationResult)
    assert result.model_name == "hpo-lgbm"
    assert result.metric_name == "mae"
    assert result.optimization_direction is OptimizationDirection.MINIMIZE
    assert len(result.trials) == 2
    assert result.best_trial in result.trials
    assert result.best_parameters == result.best_trial.parameters
    assert result.best_score == result.best_trial.score
    assert result.duration >= 0.0


def test_successful_classification_optimization() -> None:
    """Grid search completes for a classification model and metric."""
    optimizer, _registry = _build_optimizer(
        task_type=ModelTaskType.CLASSIFICATION,
        model_name="hpo-clf",
    )
    frame = _classification_frame()

    result = optimizer.optimize(
        "hpo-clf",
        frame,
        parameter_grid={"num_boost_round": [8, 12]},
        metric="accuracy",
        folds=2,
    )

    assert result.optimization_direction is OptimizationDirection.MAXIMIZE
    assert len(result.trials) == 2
    assert set(result.best_trial.cross_validation_result.mean_metrics) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
    }


def test_multiple_parameter_combinations() -> None:
    """Every grid combination produces an independent trial."""
    optimizer, _registry = _build_optimizer()
    frame = _regression_frame()

    result = optimizer.optimize(
        "hpo-lgbm",
        frame,
        parameter_grid={"num_boost_round": [6, 9, 12]},
        metric="rmse",
        folds=2,
    )

    assert len(result.trials) == 3
    assert [trial.trial_number for trial in result.trials] == [1, 2, 3]
    assert [trial.parameters["num_boost_round"] for trial in result.trials] == [6, 9, 12]


def test_best_trial_selection_minimize() -> None:
    """Best trial is the minimum-score candidate for minimize metrics."""
    optimizer, _registry = _build_optimizer()
    frame = _regression_frame()

    result = optimizer.optimize(
        "hpo-lgbm",
        frame,
        parameter_grid={"num_boost_round": [8, 16]},
        metric="mae",
        folds=2,
    )

    expected = min(result.trials, key=lambda trial: trial.score)
    assert result.best_trial.trial_number == expected.trial_number
    assert result.best_score == expected.score
    assert all(isinstance(trial, OptimizationTrial) for trial in result.trials)


def test_best_trial_selection_maximize() -> None:
    """Best trial is the maximum-score candidate for maximize metrics."""
    optimizer, _registry = _build_optimizer(
        task_type=ModelTaskType.CLASSIFICATION,
        model_name="hpo-clf",
    )
    frame = _classification_frame()

    result = optimizer.optimize(
        "hpo-clf",
        frame,
        parameter_grid={"num_boost_round": [8, 16]},
        metric="f1",
        folds=2,
    )

    expected = max(result.trials, key=lambda trial: trial.score)
    assert result.best_trial.trial_number == expected.trial_number
    assert result.best_score == expected.score


def test_restores_original_registry_model() -> None:
    """Template model remains registered after optimization completes."""
    optimizer, registry = _build_optimizer(num_boost_round=10)
    frame = _regression_frame()

    optimizer.optimize(
        "hpo-lgbm",
        frame,
        parameter_grid={"num_boost_round": [8, 12]},
        metric="mse",
        folds=2,
    )

    restored = registry.get("hpo-lgbm")
    assert isinstance(restored, LightGBMModel)
    assert restored.num_boost_round == 10


def test_unsupported_metric_rejected() -> None:
    """Unsupported metrics raise ModelValidationError."""
    optimizer, _registry = _build_optimizer()

    with pytest.raises(ModelValidationError, match="unsupported metric"):
        optimizer.optimize(
            "hpo-lgbm",
            _regression_frame(),
            parameter_grid={"num_boost_round": [8]},
            metric="roc_auc",
            folds=2,
        )


def test_empty_grid_rejected() -> None:
    """Empty parameter grids raise ModelValidationError."""
    optimizer, _registry = _build_optimizer()

    with pytest.raises(ModelValidationError, match="must not be empty"):
        optimizer.optimize(
            "hpo-lgbm",
            _regression_frame(),
            parameter_grid={},
            metric="mae",
            folds=2,
        )


def test_invalid_folds_rejected() -> None:
    """Fold counts below 2 raise ModelValidationError."""
    optimizer, _registry = _build_optimizer()

    with pytest.raises(ModelValidationError, match="folds must be an integer"):
        optimizer.optimize(
            "hpo-lgbm",
            _regression_frame(),
            parameter_grid={"num_boost_round": [8]},
            metric="mae",
            folds=1,
        )


def test_unknown_model_rejected() -> None:
    """Unknown model names raise ModelValidationError."""
    optimizer, _registry = _build_optimizer()

    with pytest.raises(ModelValidationError, match="not registered"):
        optimizer.optimize(
            "missing-model",
            _regression_frame(),
            parameter_grid={"num_boost_round": [8]},
            metric="mae",
            folds=2,
        )


def test_empty_dataset_rejected() -> None:
    """Empty datasets raise ModelValidationError."""
    optimizer, _registry = _build_optimizer()
    empty = pl.DataFrame(schema={"f1": pl.Float64, "f2": pl.Float64, "label": pl.Float64})

    with pytest.raises(ModelValidationError, match="at least one row"):
        optimizer.optimize(
            "hpo-lgbm",
            empty,
            parameter_grid={"num_boost_round": [8]},
            metric="mae",
            folds=2,
        )


def test_unknown_parameter_rejected() -> None:
    """Unknown model constructor parameters raise ModelValidationError."""
    optimizer, _registry = _build_optimizer()

    with pytest.raises(ModelValidationError, match="unknown model parameters"):
        optimizer.optimize(
            "hpo-lgbm",
            _regression_frame(),
            parameter_grid={"learning_rate": [0.1]},
            metric="mae",
            folds=2,
        )


def test_rejects_invalid_cross_validator() -> None:
    """Constructor rejects non-TimeSeriesCrossValidator values."""
    with pytest.raises(ModelValidationError, match="TimeSeriesCrossValidator"):
        HyperparameterOptimizer(cross_validator="bad")  # type: ignore[arg-type]
