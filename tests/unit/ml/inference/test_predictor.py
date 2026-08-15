"""Unit tests for CQROS ``PredictionPipeline``."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.ml.inference import PredictionPipeline, PredictionResult
from cqros.ml.inference.predictor import PredictionPipeline as PredictionPipelineDirect
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


def _metadata(*, name: str, framework: ModelFramework) -> ModelMetadata:
    """Build ModelMetadata for inference unit tests."""
    return ModelMetadata(
        name=name,
        version="1.0.0",
        framework=framework,
        task_type=ModelTaskType.REGRESSION,
        feature_columns=("f1", "f2"),
        label_column="label",
        description=f"{framework.value} inference test model",
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


def _registry_with_fitted_models() -> ModelRegistry:
    """Build a registry containing one fitted model per supported framework."""
    frame = _regression_frame(rows=30)
    registry = ModelRegistry()
    registry.register_many(
        (
            LightGBMModel(
                model_metadata=_metadata(name="infer-lgbm", framework=ModelFramework.LIGHTGBM),
                num_boost_round=15,
            ).fit(frame),
            XGBoostModel(
                model_metadata=_metadata(name="infer-xgb", framework=ModelFramework.XGBOOST),
                num_boost_round=15,
            ).fit(frame),
            CatBoostModel(
                model_metadata=_metadata(name="infer-cat", framework=ModelFramework.CATBOOST),
                num_boost_round=15,
            ).fit(frame),
        )
    )
    return registry


def test_package_exports_prediction_pipeline() -> None:
    """PredictionPipeline and PredictionResult are package exports."""
    import cqros.ml.inference as inference_package

    assert "PredictionPipeline" in inference_package.__all__
    assert "PredictionResult" in inference_package.__all__
    assert inference_package.PredictionPipeline is PredictionPipeline
    assert PredictionPipeline is PredictionPipelineDirect


def test_successful_lightgbm_prediction() -> None:
    """PredictionPipeline generates predictions from a fitted LightGBM model."""
    registry = _registry_with_fitted_models()
    pipeline = PredictionPipeline(model_registry=registry)
    frame = _regression_frame(rows=12)

    result = pipeline.predict("infer-lgbm", frame)

    assert isinstance(result, PredictionResult)
    assert result.model_metadata.name == "infer-lgbm"
    assert result.model_metadata.framework is ModelFramework.LIGHTGBM
    assert result.prediction_count == frame.height
    assert result.predictions.len() == frame.height
    assert result.predictions.dtype == pl.Float64
    assert result.prediction_time >= 0.0


def test_successful_xgboost_prediction() -> None:
    """PredictionPipeline generates predictions from a fitted XGBoost model."""
    registry = _registry_with_fitted_models()
    pipeline = PredictionPipeline(model_registry=registry)
    frame = _regression_frame(rows=12)

    result = pipeline.predict("infer-xgb", frame)

    assert result.model_metadata.framework is ModelFramework.XGBOOST
    assert result.prediction_count == frame.height
    assert result.predictions.len() == frame.height


def test_successful_catboost_prediction() -> None:
    """PredictionPipeline generates predictions from a fitted CatBoost model."""
    registry = _registry_with_fitted_models()
    pipeline = PredictionPipeline(model_registry=registry)
    frame = _regression_frame(rows=12)

    result = pipeline.predict("infer-cat", frame)

    assert result.model_metadata.framework is ModelFramework.CATBOOST
    assert result.prediction_count == frame.height
    assert result.predictions.len() == frame.height


def test_prediction_metadata_and_row_order() -> None:
    """PredictionResult captures metadata and preserves input row count/order."""
    registry = _registry_with_fitted_models()
    pipeline = PredictionPipeline(model_registry=registry)
    frame = _regression_frame(rows=8)

    result = pipeline.predict("infer-lgbm", frame)

    assert result.model_metadata is registry.get("infer-lgbm").metadata()
    assert result.prediction_count == 8
    assert result.predictions.len() == 8
    assert isinstance(result.prediction_time, float)


def test_does_not_mutate_input_frame() -> None:
    """Inference leaves the caller-supplied frame unchanged."""
    registry = _registry_with_fitted_models()
    pipeline = PredictionPipeline(model_registry=registry)
    frame = _regression_frame(rows=10)
    before = frame.clone()

    pipeline.predict("infer-lgbm", frame)

    assert frame.equals(before)


def test_unknown_model_rejected() -> None:
    """Unknown model names raise ModelValidationError."""
    pipeline = PredictionPipeline(model_registry=ModelRegistry())
    with pytest.raises(ModelValidationError, match="not registered"):
        pipeline.predict("missing", _regression_frame())


def test_unfitted_model_rejected() -> None:
    """Unfitted models raise ModelValidationError during inference."""
    registry = ModelRegistry()
    registry.register(
        LightGBMModel(
            model_metadata=_metadata(name="unfitted-lgbm", framework=ModelFramework.LIGHTGBM),
            num_boost_round=10,
        )
    )
    pipeline = PredictionPipeline(model_registry=registry)

    with pytest.raises(ModelValidationError, match="must be fitted"):
        pipeline.predict("unfitted-lgbm", _regression_frame())


def test_empty_frame_rejected() -> None:
    """Empty inference frames raise ModelValidationError."""
    registry = _registry_with_fitted_models()
    pipeline = PredictionPipeline(model_registry=registry)
    empty = pl.DataFrame(schema={"f1": pl.Float64, "f2": pl.Float64, "label": pl.Float64})

    with pytest.raises(ModelValidationError, match="at least one row"):
        pipeline.predict("infer-lgbm", empty)


def test_missing_features_rejected() -> None:
    """Frames missing required feature columns raise ModelValidationError."""
    registry = _registry_with_fitted_models()
    pipeline = PredictionPipeline(model_registry=registry)
    frame = pl.DataFrame({"f1": [1.0], "label": [1.0]})

    with pytest.raises(ModelValidationError, match="required feature columns"):
        pipeline.predict("infer-lgbm", frame)


def test_rejects_invalid_registry() -> None:
    """Constructor rejects non-ModelRegistry values."""
    with pytest.raises(ModelValidationError, match="ModelRegistry"):
        PredictionPipeline(model_registry="not-a-registry")  # type: ignore[arg-type]
