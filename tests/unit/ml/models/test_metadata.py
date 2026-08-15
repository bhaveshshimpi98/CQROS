"""Unit tests for CQROS ML ``ModelMetadata`` and enumerations."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass

import pytest

from cqros.core.exceptions import ModelError
from cqros.ml.models import (
    ModelFramework,
    ModelMetadata,
    ModelTaskType,
    ModelValidationError,
)
from cqros.ml.models.metadata import ModelFramework as ModelFrameworkDirect
from cqros.ml.models.metadata import ModelMetadata as ModelMetadataDirect
from cqros.ml.models.metadata import ModelTaskType as ModelTaskTypeDirect


def _metadata(**overrides: object) -> ModelMetadata:
    """Build ModelMetadata with optional field overrides."""
    values: dict[str, object] = {
        "name": "alpha-lgbm",
        "version": "1.0.0",
        "framework": ModelFramework.LIGHTGBM,
        "task_type": ModelTaskType.REGRESSION,
        "feature_columns": ("returns", "log_returns"),
        "label_column": "future_return_1",
        "description": "Baseline LightGBM regressor",
    }
    values.update(overrides)
    return ModelMetadata(**values)  # type: ignore[arg-type]


def test_metadata_types_are_exported_from_package() -> None:
    """Package exports match the metadata module symbols."""
    assert ModelMetadata is ModelMetadataDirect
    assert ModelFramework is ModelFrameworkDirect
    assert ModelTaskType is ModelTaskTypeDirect


def test_model_metadata_is_frozen_slotted_dataclass() -> None:
    """ModelMetadata is an immutable slotted dataclass."""
    meta = _metadata()
    assert is_dataclass(meta)
    assert is_dataclass(ModelMetadata)
    with pytest.raises(FrozenInstanceError):
        meta.name = "other"  # type: ignore[misc]


def test_model_metadata_creation() -> None:
    """Constructor arguments are exposed as immutable attributes."""
    meta = _metadata(
        name="direction-xgb",
        version="2.1.0",
        framework=ModelFramework.XGBOOST,
        task_type=ModelTaskType.CLASSIFICATION,
        feature_columns=("returns", "atr"),
        label_column="direction_1",
        description="Direction classifier",
    )
    assert meta.name == "direction-xgb"
    assert meta.version == "2.1.0"
    assert meta.framework is ModelFramework.XGBOOST
    assert meta.task_type is ModelTaskType.CLASSIFICATION
    assert meta.feature_columns == ("returns", "atr")
    assert meta.label_column == "direction_1"
    assert meta.description == "Direction classifier"


def test_feature_columns_are_frozen_copies() -> None:
    """feature_columns is stored as an independent immutable tuple."""
    columns = ["returns", "atr"]
    meta = _metadata(feature_columns=columns)
    assert meta.feature_columns == ("returns", "atr")
    assert meta.feature_columns is not columns
    columns.append("oi_change")
    assert meta.feature_columns == ("returns", "atr")


def test_model_framework_enum_values() -> None:
    """ModelFramework exposes the supported framework identifiers."""
    assert ModelFramework.LIGHTGBM == "lightgbm"
    assert ModelFramework.XGBOOST == "xgboost"
    assert ModelFramework.CATBOOST == "catboost"
    assert ModelFramework.RANDOM_FOREST == "random_forest"
    assert set(ModelFramework) == {
        ModelFramework.LIGHTGBM,
        ModelFramework.XGBOOST,
        ModelFramework.CATBOOST,
        ModelFramework.RANDOM_FOREST,
    }


def test_model_task_type_enum_values() -> None:
    """ModelTaskType exposes regression and classification."""
    assert ModelTaskType.REGRESSION == "regression"
    assert ModelTaskType.CLASSIFICATION == "classification"
    assert set(ModelTaskType) == {
        ModelTaskType.REGRESSION,
        ModelTaskType.CLASSIFICATION,
    }


def test_metadata_validation_rejects_invalid_fields() -> None:
    """Invalid metadata fields raise ModelValidationError."""
    with pytest.raises(ModelValidationError, match="name"):
        _metadata(name="")
    with pytest.raises(ModelValidationError, match="version"):
        _metadata(version="   ")
    with pytest.raises(ModelValidationError, match="framework"):
        _metadata(framework="lightgbm")
    with pytest.raises(ModelValidationError, match="task_type"):
        _metadata(task_type="regression")
    with pytest.raises(ModelValidationError, match="feature_columns"):
        _metadata(feature_columns=())
    with pytest.raises(ModelValidationError, match="feature_columns"):
        _metadata(feature_columns="returns")
    with pytest.raises(ModelValidationError, match="label_column"):
        _metadata(label_column="")
    with pytest.raises(ModelValidationError, match="description"):
        _metadata(description=123)


def test_model_validation_error_inherits_model_error() -> None:
    """ModelValidationError is part of the ModelError hierarchy."""
    assert issubclass(ModelValidationError, ModelError)
    error = ModelValidationError("invalid metadata")
    assert isinstance(error, ModelError)
