"""Unit tests for CQROS ``XGBoostModel``."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from cqros.ml.models import (
    Model,
    ModelFramework,
    ModelMetadata,
    ModelTaskType,
    ModelValidationError,
    XGBoostModel,
)
from cqros.ml.models.xgboost import XGBoostModel as XGBoostModelDirect


def _metadata(**overrides: object) -> ModelMetadata:
    """Build ModelMetadata with optional field overrides."""
    values: dict[str, object] = {
        "name": "alpha-xgb",
        "version": "1.0.0",
        "framework": ModelFramework.XGBOOST,
        "task_type": ModelTaskType.REGRESSION,
        "feature_columns": ("f1", "f2"),
        "label_column": "label",
        "description": "XGBoost unit-test model",
    }
    values.update(overrides)
    return ModelMetadata(**values)  # type: ignore[arg-type]


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


def test_package_exports_xgboost_model() -> None:
    """XGBoostModel is exported from the ml.models package."""
    import cqros.ml.models as models_package

    assert "XGBoostModel" in models_package.__all__
    assert models_package.XGBoostModel is XGBoostModel
    assert XGBoostModel is XGBoostModelDirect


def test_metadata_and_model_protocol() -> None:
    """Constructor stores metadata and satisfies the Model protocol."""
    meta = _metadata()
    model = XGBoostModel(model_metadata=meta, num_boost_round=10)

    assert model.metadata() is meta
    assert model.metadata().framework is ModelFramework.XGBOOST
    assert model.metadata().task_type is ModelTaskType.REGRESSION
    assert isinstance(model, Model)


def test_rejects_non_xgboost_framework() -> None:
    """Non-XGBoost framework metadata is rejected."""
    with pytest.raises(ModelValidationError, match="XGBOOST"):
        XGBoostModel(
            model_metadata=_metadata(framework=ModelFramework.LIGHTGBM),
            num_boost_round=5,
        )


def test_regression_training_and_prediction() -> None:
    """Regression fit stores a booster and predict returns Float64 scores."""
    model = XGBoostModel(model_metadata=_metadata(), num_boost_round=20)
    frame = _regression_frame()
    fitted = model.fit(frame)
    predictions = fitted.predict(frame)

    assert fitted is model
    assert predictions.name == "prediction"
    assert predictions.dtype == pl.Float64
    assert predictions.len() == frame.height


def test_classification_training_and_prediction() -> None:
    """Classification fit predicts integer class labels."""
    model = XGBoostModel(
        model_metadata=_metadata(task_type=ModelTaskType.CLASSIFICATION),
        num_boost_round=20,
    )
    frame = _classification_frame()
    predictions = model.fit(frame).predict(frame)

    assert predictions.dtype == pl.Int64
    assert predictions.len() == frame.height
    assert set(predictions.to_list()).issubset({0, 1})


def test_fit_accepts_optional_validation_frame() -> None:
    """Optional validation_frame is accepted without mutating inputs."""
    model = XGBoostModel(model_metadata=_metadata(), num_boost_round=10)
    train = _regression_frame(rows=30)
    valid = _regression_frame(rows=10)
    before_train = train.clone()
    before_valid = valid.clone()

    model.fit(train, validation_frame=valid)
    predictions = model.predict(valid)

    assert train.equals(before_train)
    assert valid.equals(before_valid)
    assert predictions.len() == valid.height


def test_predict_before_fit_raises() -> None:
    """predict before fit raises ModelValidationError."""
    model = XGBoostModel(model_metadata=_metadata(), num_boost_round=5)
    with pytest.raises(ModelValidationError, match="fitted"):
        model.predict(_regression_frame())


def test_save_before_fit_raises() -> None:
    """save before fit raises ModelValidationError."""
    model = XGBoostModel(model_metadata=_metadata(), num_boost_round=5)
    with pytest.raises(ModelValidationError, match="fitted"):
        model.save("unused.bin")


def test_invalid_inputs_are_rejected() -> None:
    """Empty frames and missing columns raise ModelValidationError."""
    model = XGBoostModel(model_metadata=_metadata(), num_boost_round=5)
    empty = pl.DataFrame(schema={"f1": pl.Float64, "f2": pl.Float64, "label": pl.Float64})
    missing_features = pl.DataFrame({"f1": [1.0], "label": [1.0]})
    missing_label = pl.DataFrame({"f1": [1.0], "f2": [2.0]})

    with pytest.raises(ModelValidationError, match="at least one row"):
        model.fit(empty)
    with pytest.raises(ModelValidationError, match="feature columns"):
        model.fit(missing_features)
    with pytest.raises(ModelValidationError, match="label column"):
        model.fit(missing_label)


def test_classification_rejects_single_class_labels() -> None:
    """Classification with a single label class is rejected."""
    model = XGBoostModel(
        model_metadata=_metadata(task_type=ModelTaskType.CLASSIFICATION),
        num_boost_round=5,
    )
    frame = pl.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0],
            "f2": [0.5, 1.5, 2.5, 3.5],
            "label": [0, 0, 0, 0],
        }
    )
    with pytest.raises(ModelValidationError, match="at least two label classes"):
        model.fit(frame)


def test_save_load_round_trip(tmp_path: Path) -> None:
    """Native XGBoost serialization round-trips predictions."""
    model = XGBoostModel(model_metadata=_metadata(), num_boost_round=25)
    frame = _regression_frame()
    model.fit(frame)
    expected = model.predict(frame)

    path = tmp_path / "alpha-xgb.json"
    model.save(path)

    loaded = XGBoostModel(model_metadata=_metadata(), num_boost_round=25).load(path)
    restored = loaded.predict(frame)

    assert path.is_file()
    assert restored.dtype == pl.Float64
    assert restored.to_list() == pytest.approx(expected.to_list())


def test_model_persistence_path_validation(tmp_path: Path) -> None:
    """save and load reject invalid path types."""
    model = XGBoostModel(model_metadata=_metadata(), num_boost_round=10)
    model.fit(_regression_frame())
    path = tmp_path / "model.json"
    model.save(path)

    with pytest.raises(ModelValidationError, match="Path or str"):
        model.save(123)  # type: ignore[arg-type]
    with pytest.raises(ModelValidationError, match="Path or str"):
        XGBoostModel(model_metadata=_metadata()).load(123)  # type: ignore[arg-type]
