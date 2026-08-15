"""Unit tests for CQROS ``ModelEvaluator``."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Self

import polars as pl
import pytest
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)

from cqros.ml.evaluation import EvaluationResult, ModelEvaluator
from cqros.ml.evaluation.evaluator import ModelEvaluator as ModelEvaluatorDirect
from cqros.ml.models import (
    LightGBMModel,
    Model,
    ModelFramework,
    ModelMetadata,
    ModelTaskType,
    ModelValidationError,
)


def _metadata(*, task_type: ModelTaskType = ModelTaskType.REGRESSION) -> ModelMetadata:
    """Build ModelMetadata for evaluator unit tests."""
    return ModelMetadata(
        name="eval-lgbm",
        version="1.0.0",
        framework=ModelFramework.LIGHTGBM,
        task_type=task_type,
        feature_columns=("f1", "f2"),
        label_column="label",
        description="LightGBM evaluator test model",
    )


def _regression_frame(*, rows: int = 40) -> pl.DataFrame:
    """Build a deterministic regression frame."""
    return pl.DataFrame(
        {
            "f1": [float(index) for index in range(rows)],
            "f2": [float(index) * 0.5 for index in range(rows)],
            "label": [float(index) * 1.5 + 0.1 for index in range(rows)],
        }
    )


def _classification_frame(*, rows: int = 40) -> pl.DataFrame:
    """Build a deterministic binary classification frame."""
    return pl.DataFrame(
        {
            "f1": [float(index) for index in range(rows)],
            "f2": [float(index % 3) for index in range(rows)],
            "label": [index % 2 for index in range(rows)],
        }
    )


def test_package_exports_model_evaluator() -> None:
    """ModelEvaluator and EvaluationResult are exported from the package."""
    import cqros.ml.evaluation as evaluation_package

    assert "ModelEvaluator" in evaluation_package.__all__
    assert "EvaluationResult" in evaluation_package.__all__
    assert evaluation_package.ModelEvaluator is ModelEvaluator
    assert ModelEvaluator is ModelEvaluatorDirect


def test_regression_evaluation() -> None:
    """ModelEvaluator computes regression metrics for a fitted model."""
    model = LightGBMModel(model_metadata=_metadata(), num_boost_round=20)
    frame = _regression_frame()
    fitted = model.fit(frame)
    evaluator = ModelEvaluator()

    result = evaluator.evaluate(fitted, frame)

    assert isinstance(result, EvaluationResult)
    assert result.model_metadata is fitted.metadata()
    assert result.task_type is ModelTaskType.REGRESSION
    assert result.dataset_rows == frame.height
    assert set(result.metrics) == {"mae", "mse", "rmse", "r2"}
    assert all(isinstance(value, float) for value in result.metrics.values())
    assert result.metrics["mae"] >= 0.0
    assert result.metrics["mse"] >= 0.0
    assert result.metrics["rmse"] >= 0.0
    assert math.isclose(result.metrics["rmse"], math.sqrt(result.metrics["mse"]))


def test_classification_evaluation() -> None:
    """ModelEvaluator computes classification metrics for a fitted model."""
    model = LightGBMModel(
        model_metadata=_metadata(task_type=ModelTaskType.CLASSIFICATION),
        num_boost_round=20,
    )
    frame = _classification_frame()
    fitted = model.fit(frame)
    evaluator = ModelEvaluator()

    result = evaluator.evaluate(fitted, frame)

    assert result.task_type is ModelTaskType.CLASSIFICATION
    assert result.dataset_rows == frame.height
    assert set(result.metrics) == {"accuracy", "precision", "recall", "f1"}
    assert 0.0 <= result.metrics["accuracy"] <= 1.0
    assert 0.0 <= result.metrics["precision"] <= 1.0
    assert 0.0 <= result.metrics["recall"] <= 1.0
    assert 0.0 <= result.metrics["f1"] <= 1.0


def test_metric_correctness_against_sklearn() -> None:
    """Evaluator metrics match scikit-learn for known predictions."""
    metadata = _metadata()
    frame = _regression_frame(rows=8)
    predictions = pl.Series(
        "prediction",
        [float(index) * 1.5 for index in range(frame.height)],
    )
    model = _StubModel(metadata=metadata, predictions=predictions)
    evaluator = ModelEvaluator()

    result = evaluator.evaluate(model, frame)

    y_true = frame.get_column("label").to_numpy()
    y_pred = predictions.to_numpy()
    assert result.metrics["mae"] == mean_absolute_error(y_true, y_pred)
    assert result.metrics["mse"] == mean_squared_error(y_true, y_pred)
    assert result.metrics["rmse"] == root_mean_squared_error(y_true, y_pred)
    assert result.metrics["r2"] == r2_score(y_true, y_pred)


def test_classification_metric_correctness_against_sklearn() -> None:
    """Evaluator classification metrics match scikit-learn for known labels."""
    metadata = _metadata(task_type=ModelTaskType.CLASSIFICATION)
    frame = _classification_frame(rows=8)
    predictions = pl.Series("prediction", [0, 1, 0, 1, 0, 1, 0, 1], dtype=pl.Int64)
    model = _StubModel(metadata=metadata, predictions=predictions)
    evaluator = ModelEvaluator()

    result = evaluator.evaluate(model, frame)

    y_true = frame.get_column("label").to_numpy()
    y_pred = predictions.to_numpy()
    assert result.metrics["accuracy"] == accuracy_score(y_true, y_pred)
    assert result.metrics["precision"] == precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0.0,
    )
    assert result.metrics["recall"] == recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0.0,
    )
    assert result.metrics["f1"] == f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0.0,
    )


def test_empty_frame_rejected() -> None:
    """Empty evaluation frames raise ModelValidationError."""
    model = LightGBMModel(model_metadata=_metadata(), num_boost_round=10)
    fitted = model.fit(_regression_frame(rows=20))
    empty = pl.DataFrame(schema={"f1": pl.Float64, "f2": pl.Float64, "label": pl.Float64})
    evaluator = ModelEvaluator()

    with pytest.raises(ModelValidationError, match="at least one row"):
        evaluator.evaluate(fitted, empty)


def test_unfitted_model_rejected() -> None:
    """Unfitted models raise ModelValidationError during evaluation."""
    model = LightGBMModel(model_metadata=_metadata(), num_boost_round=10)
    evaluator = ModelEvaluator()

    with pytest.raises(ModelValidationError, match="must be fitted"):
        evaluator.evaluate(model, _regression_frame())


def test_missing_label_column_rejected() -> None:
    """Frames missing the label column raise ModelValidationError."""
    model = LightGBMModel(model_metadata=_metadata(), num_boost_round=10)
    fitted = model.fit(_regression_frame(rows=20))
    frame = pl.DataFrame({"f1": [1.0], "f2": [2.0]})
    evaluator = ModelEvaluator()

    with pytest.raises(ModelValidationError, match="required model columns"):
        evaluator.evaluate(fitted, frame)


def test_duration_recorded() -> None:
    """EvaluationResult records a non-negative wall-clock duration."""
    model = LightGBMModel(model_metadata=_metadata(), num_boost_round=15)
    frame = _regression_frame(rows=25)
    fitted = model.fit(frame)
    evaluator = ModelEvaluator()

    result = evaluator.evaluate(fitted, frame)

    assert isinstance(result.evaluation_duration, float)
    assert result.evaluation_duration >= 0.0


def test_rejects_non_model() -> None:
    """Non-Model values raise ModelValidationError."""
    evaluator = ModelEvaluator()
    with pytest.raises(ModelValidationError, match="Model protocol"):
        evaluator.evaluate("not-a-model", _regression_frame())  # type: ignore[arg-type]


class _StubModel:
    """Minimal fitted Model used for deterministic metric assertions."""

    def __init__(self, *, metadata: ModelMetadata, predictions: pl.Series) -> None:
        self._metadata = metadata
        self._predictions = predictions

    def fit(self, frame: pl.DataFrame) -> Self:
        return self

    def predict(self, frame: pl.DataFrame) -> pl.Series:
        if frame.height != self._predictions.len():
            raise ModelValidationError(
                "prediction length mismatch",
                error_code="TEST-STUB-001",
                details={
                    "frame_rows": frame.height,
                    "prediction_rows": self._predictions.len(),
                },
            )
        return self._predictions

    def save(self, path: Path | str) -> None:
        return None

    def load(self, path: Path | str) -> Self:
        return self

    def metadata(self) -> ModelMetadata:
        return self._metadata


def test_stub_model_satisfies_protocol() -> None:
    """Stub model used by metric-correctness tests satisfies Model."""
    metadata = _metadata()
    predictions = pl.Series("prediction", [1.0, 2.0], dtype=pl.Float64)
    model = _StubModel(metadata=metadata, predictions=predictions)
    assert isinstance(model, Model)
