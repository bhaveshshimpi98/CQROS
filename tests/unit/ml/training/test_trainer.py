"""Unit tests for CQROS ``ModelTrainer``."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.ml.models import (
    CatBoostModel,
    LightGBMModel,
    ModelFramework,
    ModelMetadata,
    ModelRegistry,
    ModelTaskType,
    ModelValidationError,
    XGBoostModel,
)
from cqros.ml.training import ModelTrainer, TrainerResult
from cqros.ml.training.trainer import ModelTrainer as ModelTrainerDirect


def _metadata(*, name: str, framework: ModelFramework) -> ModelMetadata:
    """Build ModelMetadata for trainer unit tests."""
    return ModelMetadata(
        name=name,
        version="1.0.0",
        framework=framework,
        task_type=ModelTaskType.REGRESSION,
        feature_columns=("f1", "f2"),
        label_column="label",
        description=f"{framework.value} trainer test model",
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


def _registry_with_all_frameworks() -> ModelRegistry:
    """Build a registry containing one model per supported framework."""
    registry = ModelRegistry()
    registry.register_many(
        (
            LightGBMModel(
                model_metadata=_metadata(name="alpha-lgbm", framework=ModelFramework.LIGHTGBM),
                num_boost_round=15,
            ),
            XGBoostModel(
                model_metadata=_metadata(name="alpha-xgb", framework=ModelFramework.XGBOOST),
                num_boost_round=15,
            ),
            CatBoostModel(
                model_metadata=_metadata(name="alpha-cat", framework=ModelFramework.CATBOOST),
                num_boost_round=15,
            ),
        )
    )
    return registry


def test_package_exports_model_trainer() -> None:
    """ModelTrainer and TrainerResult are exported from the training package."""
    import cqros.ml.training as training_package

    assert "ModelTrainer" in training_package.__all__
    assert "TrainerResult" in training_package.__all__
    assert training_package.ModelTrainer is ModelTrainer
    assert ModelTrainer is ModelTrainerDirect


def test_rejects_invalid_registry() -> None:
    """Constructor rejects non-ModelRegistry values."""
    with pytest.raises(ModelValidationError, match="ModelRegistry"):
        ModelTrainer(model_registry="not-a-registry")  # type: ignore[arg-type]


def test_registry_lookup_unknown_model() -> None:
    """Unknown model names raise ModelValidationError."""
    trainer = ModelTrainer(model_registry=ModelRegistry())
    with pytest.raises(ModelValidationError, match="not registered"):
        trainer.train("missing", _regression_frame())


def test_empty_training_frame_rejected() -> None:
    """Empty training frames raise ModelValidationError."""
    registry = _registry_with_all_frameworks()
    trainer = ModelTrainer(model_registry=registry)
    empty = pl.DataFrame(schema={"f1": pl.Float64, "f2": pl.Float64, "label": pl.Float64})
    with pytest.raises(ModelValidationError, match="at least one row"):
        trainer.train("alpha-lgbm", empty)


def test_missing_required_columns_rejected() -> None:
    """Training frames missing model columns raise ModelValidationError."""
    registry = _registry_with_all_frameworks()
    trainer = ModelTrainer(model_registry=registry)
    frame = pl.DataFrame({"f1": [1.0], "label": [1.0]})
    with pytest.raises(ModelValidationError, match="required model columns"):
        trainer.train("alpha-lgbm", frame)


def test_successful_lightgbm_training() -> None:
    """ModelTrainer fits a registered LightGBM model."""
    registry = _registry_with_all_frameworks()
    trainer = ModelTrainer(model_registry=registry)
    train = _regression_frame(rows=30)
    valid = _regression_frame(rows=10)

    result = trainer.train("alpha-lgbm", train, validation_frame=valid)

    assert isinstance(result, TrainerResult)
    assert result.fitted_model.metadata().name == "alpha-lgbm"
    assert result.model_metadata.framework is ModelFramework.LIGHTGBM
    assert result.train_rows == 30
    assert result.validation_rows == 10
    assert result.test_rows == 0
    assert result.feature_count == 2
    assert result.label_column == "label"
    assert result.training_duration >= 0.0
    predictions = result.fitted_model.predict(train)
    assert predictions.dtype == pl.Float64
    assert predictions.len() == train.height


def test_successful_xgboost_training() -> None:
    """ModelTrainer fits a registered XGBoost model."""
    registry = _registry_with_all_frameworks()
    trainer = ModelTrainer(model_registry=registry)
    train = _regression_frame()

    result = trainer.train("alpha-xgb", train)

    assert result.model_metadata.framework is ModelFramework.XGBOOST
    assert result.fitted_model.metadata().name == "alpha-xgb"
    assert result.train_rows == train.height
    assert result.validation_rows == 0
    assert result.training_duration >= 0.0


def test_successful_catboost_training() -> None:
    """ModelTrainer fits a registered CatBoost model."""
    registry = _registry_with_all_frameworks()
    trainer = ModelTrainer(model_registry=registry)
    train = _regression_frame()

    result = trainer.train("alpha-cat", train)

    assert result.model_metadata.framework is ModelFramework.CATBOOST
    assert result.fitted_model.metadata().name == "alpha-cat"
    assert result.train_rows == train.height
    assert result.training_duration >= 0.0


def test_trainer_result_contents_and_duration() -> None:
    """TrainerResult captures metadata, row counts, and a recorded duration."""
    registry = _registry_with_all_frameworks()
    trainer = ModelTrainer(model_registry=registry)
    train = _regression_frame(rows=25)
    valid = _regression_frame(rows=5)

    result = trainer.train("alpha-lgbm", train, validation_frame=valid)

    assert result.model_metadata is result.fitted_model.metadata()
    assert result.train_rows == 25
    assert result.validation_rows == 5
    assert result.test_rows == 0
    assert result.feature_count == 2
    assert result.label_column == "label"
    assert isinstance(result.training_duration, float)
    assert result.training_duration > 0.0


def test_registry_lookup_uses_injected_registry() -> None:
    """Trainer resolves models only from the injected registry."""
    registry = ModelRegistry()
    registry.register(
        LightGBMModel(
            model_metadata=_metadata(name="only-lgbm", framework=ModelFramework.LIGHTGBM),
            num_boost_round=10,
        )
    )
    trainer = ModelTrainer(model_registry=registry)

    result = trainer.train("only-lgbm", _regression_frame())
    assert result.fitted_model.metadata().name == "only-lgbm"
    with pytest.raises(ModelValidationError, match="not registered"):
        trainer.train("alpha-xgb", _regression_frame())
