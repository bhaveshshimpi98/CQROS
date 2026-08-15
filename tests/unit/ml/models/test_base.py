"""Unit tests for CQROS ML ``BaseModel``."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, is_dataclass
from pathlib import Path

import polars as pl
import pytest

from cqros.ml.models import (
    BaseModel,
    Model,
    ModelError,
    ModelFramework,
    ModelMetadata,
    ModelTaskType,
    ModelValidationError,
)
from cqros.ml.models.base import BaseModel as BaseModelDirect
from cqros.ml.models.interfaces import Model as ModelDirect


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


@dataclass(frozen=True, slots=True)
class _ConcreteModel(BaseModel):
    """Minimal concrete model used only for unit tests."""

    def fit(self, frame: pl.DataFrame) -> _ConcreteModel:
        """Validate inputs and return self without training."""
        validated = self._require_dataframe(frame)
        self._require_feature_columns(validated)
        self._require_label_column(validated)
        return self

    def predict(self, frame: pl.DataFrame) -> pl.Series:
        """Return a zero prediction series for abstract-base coverage."""
        validated = self._require_dataframe(frame)
        self._require_feature_columns(validated)
        return pl.Series("prediction", [0.0] * validated.height)

    def save(self, path: Path | str) -> None:
        """Validate path without writing artifacts."""
        self._require_path(path)

    def load(self, path: Path | str) -> _ConcreteModel:
        """Validate path and return self without loading artifacts."""
        self._require_path(path)
        return self


def test_base_model_is_exported_from_package() -> None:
    """Package exports match the base and interface modules."""
    assert BaseModel is BaseModelDirect
    assert Model is ModelDirect


def test_base_model_is_frozen_slotted_dataclass() -> None:
    """BaseModel is an immutable slotted dataclass."""
    model = _ConcreteModel(model_metadata=_metadata())
    assert is_dataclass(model)
    assert is_dataclass(BaseModel)
    with pytest.raises(FrozenInstanceError):
        model.model_metadata = _metadata(name="other")  # type: ignore[misc]


def test_base_model_initialization_exposes_metadata() -> None:
    """Constructor stores metadata and exposes it through metadata()."""
    meta = _metadata()
    model = _ConcreteModel(model_metadata=meta)

    assert model.metadata() is meta
    assert model.metadata().name == "alpha-lgbm"
    assert model.metadata().framework is ModelFramework.LIGHTGBM


def test_base_model_rejects_invalid_metadata_type() -> None:
    """Non-ModelMetadata constructor values raise ModelValidationError."""
    with pytest.raises(ModelValidationError, match="model_metadata"):
        _ConcreteModel(model_metadata="not-metadata")  # type: ignore[arg-type]


def test_abstract_methods_require_concrete_implementation() -> None:
    """BaseModel cannot be instantiated without abstract method overrides."""
    with pytest.raises(TypeError, match="abstract"):
        BaseModel(model_metadata=_metadata())  # type: ignore[abstract]


def test_concrete_model_satisfies_model_protocol() -> None:
    """Concrete BaseModel subclasses satisfy the Model protocol."""
    model = _ConcreteModel(model_metadata=_metadata())
    assert isinstance(model, Model)


def test_validation_helpers_reject_invalid_frames() -> None:
    """Shared helpers reject empty frames and missing columns."""
    model = _ConcreteModel(model_metadata=_metadata())
    empty = pl.DataFrame(
        schema={
            "returns": pl.Float64,
            "log_returns": pl.Float64,
            "future_return_1": pl.Float64,
        }
    )
    missing_features = pl.DataFrame(
        {
            "returns": [0.1],
            "future_return_1": [0.2],
        }
    )
    missing_label = pl.DataFrame(
        {
            "returns": [0.1],
            "log_returns": [0.2],
        }
    )

    with pytest.raises(ModelValidationError, match="at least one row"):
        model.fit(empty)
    with pytest.raises(ModelValidationError, match="feature columns"):
        model.fit(missing_features)
    with pytest.raises(ModelValidationError, match="label column"):
        model.fit(missing_label)


def test_validation_helpers_accept_valid_frame() -> None:
    """fit and predict succeed for frames with required columns."""
    model = _ConcreteModel(model_metadata=_metadata())
    frame = pl.DataFrame(
        {
            "returns": [0.1, 0.2],
            "log_returns": [0.01, 0.02],
            "future_return_1": [0.3, 0.4],
        }
    )

    fitted = model.fit(frame)
    predictions = fitted.predict(frame)

    assert fitted is model
    assert predictions.to_list() == [0.0, 0.0]


def test_path_helper_accepts_path_and_str() -> None:
    """_require_path accepts Path and str values."""
    model = _ConcreteModel(model_metadata=_metadata())
    model.save(Path("model.bin"))
    loaded = model.load("model.bin")
    assert loaded is model
    with pytest.raises(ModelValidationError, match="path"):
        model.save(123)  # type: ignore[arg-type]


def test_exception_hierarchy_and_package_exports() -> None:
    """Model exceptions and architecture symbols are exported correctly."""
    assert issubclass(ModelValidationError, ModelError)
    assert BaseModel.__name__ in __import__("cqros.ml.models", fromlist=["__all__"]).__all__
    assert "Model" in __import__("cqros.ml.models", fromlist=["__all__"]).__all__
    assert str(_ConcreteModel(model_metadata=_metadata())) == "alpha-lgbm@1.0.0"
