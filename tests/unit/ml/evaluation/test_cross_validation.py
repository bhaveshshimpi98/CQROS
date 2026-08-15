"""Unit tests for CQROS ``TimeSeriesCrossValidator``."""

from __future__ import annotations

import math
import statistics

import polars as pl
import pytest

from cqros.ml.evaluation import (
    CrossValidationFold,
    CrossValidationResult,
    ModelEvaluator,
    TimeSeriesCrossValidator,
)
from cqros.ml.evaluation.cross_validation import (
    TimeSeriesCrossValidator as TimeSeriesCrossValidatorDirect,
)
from cqros.ml.models import (
    LightGBMModel,
    ModelFramework,
    ModelMetadata,
    ModelRegistry,
    ModelTaskType,
    ModelValidationError,
)
from cqros.ml.training import ModelTrainer


def _metadata(
    *,
    name: str = "cv-lgbm",
    task_type: ModelTaskType = ModelTaskType.REGRESSION,
) -> ModelMetadata:
    """Build ModelMetadata for cross-validation unit tests."""
    return ModelMetadata(
        name=name,
        version="1.0.0",
        framework=ModelFramework.LIGHTGBM,
        task_type=task_type,
        feature_columns=("f1", "f2"),
        label_column="label",
        description="LightGBM cross-validation test model",
    )


def _regression_frame(*, rows: int = 60) -> pl.DataFrame:
    """Build a deterministic chronological regression frame."""
    return pl.DataFrame(
        {
            "f1": [float(index) for index in range(rows)],
            "f2": [float(index) * 0.5 for index in range(rows)],
            "label": [float(index) * 1.5 + 0.1 for index in range(rows)],
        }
    )


def _classification_frame(*, rows: int = 60) -> pl.DataFrame:
    """Build a deterministic chronological classification frame."""
    return pl.DataFrame(
        {
            "f1": [float(index) for index in range(rows)],
            "f2": [float(index % 3) for index in range(rows)],
            "label": [index % 2 for index in range(rows)],
        }
    )


def _build_validator(
    *,
    task_type: ModelTaskType = ModelTaskType.REGRESSION,
    model_name: str = "cv-lgbm",
) -> tuple[TimeSeriesCrossValidator, ModelRegistry]:
    """Build a validator with a single registered LightGBM model."""
    registry = ModelRegistry()
    registry.register(
        LightGBMModel(
            model_metadata=_metadata(name=model_name, task_type=task_type),
            num_boost_round=15,
        )
    )
    trainer = ModelTrainer(model_registry=registry)
    evaluator = ModelEvaluator()
    validator = TimeSeriesCrossValidator(
        model_registry=registry,
        model_trainer=trainer,
        model_evaluator=evaluator,
    )
    return validator, registry


def test_package_exports_time_series_cross_validator() -> None:
    """TimeSeriesCrossValidator and result types are package exports."""
    import cqros.ml.evaluation as evaluation_package

    assert "TimeSeriesCrossValidator" in evaluation_package.__all__
    assert "CrossValidationFold" in evaluation_package.__all__
    assert "CrossValidationResult" in evaluation_package.__all__
    assert evaluation_package.TimeSeriesCrossValidator is TimeSeriesCrossValidator
    assert TimeSeriesCrossValidator is TimeSeriesCrossValidatorDirect


def test_successful_regression_cv() -> None:
    """Walk-forward CV completes for a regression model."""
    validator, _registry = _build_validator()
    frame = _regression_frame(rows=60)

    result = validator.evaluate("cv-lgbm", frame, folds=3)

    assert isinstance(result, CrossValidationResult)
    assert result.fold_count == 3
    assert result.total_rows == 60
    assert len(result.folds) == 3
    assert set(result.mean_metrics) == {"mae", "mse", "rmse", "r2"}
    assert set(result.std_metrics) == {"mae", "mse", "rmse", "r2"}
    for fold in result.folds:
        assert isinstance(fold, CrossValidationFold)
        assert set(fold.evaluation_result.metrics) == {"mae", "mse", "rmse", "r2"}


def test_successful_classification_cv() -> None:
    """Walk-forward CV completes for a classification model."""
    validator, _registry = _build_validator(
        task_type=ModelTaskType.CLASSIFICATION,
        model_name="cv-clf",
    )
    frame = _classification_frame(rows=60)

    result = validator.evaluate("cv-clf", frame, folds=3)

    assert result.fold_count == 3
    assert set(result.mean_metrics) == {"accuracy", "precision", "recall", "f1"}
    assert set(result.std_metrics) == {"accuracy", "precision", "recall", "f1"}
    for fold in result.folds:
        metrics = fold.evaluation_result.metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0


def test_multiple_folds() -> None:
    """Requested fold count is honored with one result per fold."""
    validator, _registry = _build_validator()
    frame = _regression_frame(rows=80)

    result = validator.evaluate("cv-lgbm", frame, folds=4)

    assert result.fold_count == 4
    assert [fold.fold_number for fold in result.folds] == [1, 2, 3, 4]


def test_expanding_windows() -> None:
    """Training windows expand while validation stays chronological."""
    validator, _registry = _build_validator()
    frame = _regression_frame(rows=60)

    result = validator.evaluate("cv-lgbm", frame, folds=3)

    # sklearn TimeSeriesSplit layout: validation_size = 60 // 4 = 15
    # Fold1 train=15 val=15, Fold2 train=30 val=15, Fold3 train=45 val=15
    assert [fold.train_rows for fold in result.folds] == [15, 30, 45]
    assert [fold.validation_rows for fold in result.folds] == [15, 15, 15]
    assert result.folds[0].train_rows < result.folds[1].train_rows
    assert result.folds[1].train_rows < result.folds[2].train_rows


def test_metric_aggregation() -> None:
    """Mean and std metrics match fold-level population aggregates."""
    validator, _registry = _build_validator()
    frame = _regression_frame(rows=60)

    result = validator.evaluate("cv-lgbm", frame, folds=3)

    for metric_name in result.mean_metrics:
        values = [float(fold.evaluation_result.metrics[metric_name]) for fold in result.folds]
        assert math.isclose(result.mean_metrics[metric_name], statistics.fmean(values))
        assert math.isclose(result.std_metrics[metric_name], statistics.pstdev(values))


def test_duration_recorded() -> None:
    """CrossValidationResult records a non-negative wall-clock duration."""
    validator, _registry = _build_validator()
    frame = _regression_frame(rows=45)

    result = validator.evaluate("cv-lgbm", frame, folds=2)

    assert isinstance(result.duration, float)
    assert result.duration >= 0.0


def test_unknown_model_rejected() -> None:
    """Unknown model names raise ModelValidationError."""
    validator, _registry = _build_validator()

    with pytest.raises(ModelValidationError, match="not registered"):
        validator.evaluate("missing-model", _regression_frame(), folds=2)


def test_invalid_fold_count_rejected() -> None:
    """Fold counts below 2 raise ModelValidationError."""
    validator, _registry = _build_validator()
    frame = _regression_frame()

    with pytest.raises(ModelValidationError, match="folds must be an integer"):
        validator.evaluate("cv-lgbm", frame, folds=1)
    with pytest.raises(ModelValidationError, match="folds must be an integer"):
        validator.evaluate("cv-lgbm", frame, folds=0)


def test_insufficient_rows_rejected() -> None:
    """Frames too short for the requested folds raise ModelValidationError."""
    validator, _registry = _build_validator()
    frame = _regression_frame(rows=3)

    with pytest.raises(ModelValidationError, match="insufficient rows"):
        validator.evaluate("cv-lgbm", frame, folds=3)


def test_empty_frame_rejected() -> None:
    """Empty frames raise ModelValidationError."""
    validator, _registry = _build_validator()
    empty = pl.DataFrame(schema={"f1": pl.Float64, "f2": pl.Float64, "label": pl.Float64})

    with pytest.raises(ModelValidationError, match="at least one row"):
        validator.evaluate("cv-lgbm", empty, folds=2)


def test_rejects_invalid_dependencies() -> None:
    """Constructor rejects dependencies with invalid types."""
    registry = ModelRegistry()
    trainer = ModelTrainer(model_registry=registry)
    evaluator = ModelEvaluator()

    with pytest.raises(ModelValidationError, match="ModelRegistry"):
        TimeSeriesCrossValidator(
            model_registry="bad",  # type: ignore[arg-type]
            model_trainer=trainer,
            model_evaluator=evaluator,
        )
    with pytest.raises(ModelValidationError, match="ModelTrainer"):
        TimeSeriesCrossValidator(
            model_registry=registry,
            model_trainer="bad",  # type: ignore[arg-type]
            model_evaluator=evaluator,
        )
    with pytest.raises(ModelValidationError, match="ModelEvaluator"):
        TimeSeriesCrossValidator(
            model_registry=registry,
            model_trainer=trainer,
            model_evaluator="bad",  # type: ignore[arg-type]
        )


def test_does_not_mutate_input_frame() -> None:
    """Cross-validation leaves the caller-supplied frame unchanged."""
    validator, _registry = _build_validator()
    frame = _regression_frame(rows=45)
    before = frame.clone()

    validator.evaluate("cv-lgbm", frame, folds=2)

    assert frame.equals(before)
