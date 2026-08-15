"""Unit tests for CQROS ML ``ModelPersistence``."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pytest

from cqros.ml.models import (
    BaseModel,
    Model,
    ModelError,
    ModelFramework,
    ModelMetadata,
    ModelPersistence,
    ModelTaskType,
    ModelValidationError,
)
from cqros.ml.models.persistence import ModelPersistence as ModelPersistenceDirect


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
class _StubModel(BaseModel):
    """Minimal concrete model used only for persistence unit tests."""

    def fit(self, frame: pl.DataFrame) -> _StubModel:
        """Return self without training."""
        return self

    def predict(self, frame: pl.DataFrame) -> pl.Series:
        """Return an empty prediction series."""
        return pl.Series("prediction", [], dtype=pl.Float64)

    def save(self, path: Path | str) -> None:
        """No-op persistence stub."""

    def load(self, path: Path | str) -> _StubModel:
        """Return self without loading artifacts."""
        return self


class _StubPersistence(ModelPersistence):
    """Minimal concrete persistence backend used only for unit tests.

    Performs argument validation through the base helpers and records calls
    without touching the filesystem.
    """

    def __init__(self) -> None:
        """Initialize an empty in-memory call recorder."""
        self.saved: tuple[Model, Path] | None = None
        self.loaded_path: Path | None = None
        self.exists_path: Path | None = None
        self.deleted_path: Path | None = None
        self._model = _StubModel(model_metadata=_metadata())

    def save(self, model: Model, path: Path | str) -> None:
        """Validate arguments and record the save call."""
        validated_model = self._require_model(model)
        validated_path = self._require_path(path)
        self.saved = (validated_model, validated_path)

    def load(self, path: Path | str) -> Model:
        """Validate path and return the stub model."""
        self.loaded_path = self._require_path(path)
        return self._model

    def exists(self, path: Path | str) -> bool:
        """Validate path and report that artifacts are absent."""
        self.exists_path = self._require_path(path)
        return False

    def delete(self, path: Path | str) -> None:
        """Validate path and record the delete call."""
        self.deleted_path = self._require_path(path)


def test_model_persistence_is_abstract_interface() -> None:
    """ModelPersistence is an ABC and cannot be instantiated directly."""
    assert issubclass(ModelPersistence, ABC)
    with pytest.raises(TypeError, match="abstract"):
        ModelPersistence()  # type: ignore[abstract]


def test_abstract_methods_are_declared() -> None:
    """All persistence operations are abstract on the base contract."""
    for method_name in ("save", "load", "exists", "delete"):
        method = getattr(ModelPersistence, method_name)
        assert getattr(method, "__isabstractmethod__", False) is True


def test_concrete_persistence_can_be_instantiated() -> None:
    """Concrete subclasses that implement all methods can be constructed."""
    persistence = _StubPersistence()
    assert isinstance(persistence, ModelPersistence)


def test_invalid_path_types_are_rejected() -> None:
    """Non-path arguments raise ModelValidationError."""
    persistence = _StubPersistence()
    model = _StubModel(model_metadata=_metadata())
    for invalid in (None, 123, 1.5, ["models", "a.bin"], {"path": "a.bin"}):
        with pytest.raises(ModelValidationError, match="Path or str"):
            persistence.save(model, invalid)  # type: ignore[arg-type]
        with pytest.raises(ModelValidationError, match="Path or str"):
            persistence.load(invalid)  # type: ignore[arg-type]
        with pytest.raises(ModelValidationError, match="Path or str"):
            persistence.exists(invalid)  # type: ignore[arg-type]
        with pytest.raises(ModelValidationError, match="Path or str"):
            persistence.delete(invalid)  # type: ignore[arg-type]


def test_empty_paths_are_rejected() -> None:
    """Empty and blank string paths raise ModelValidationError."""
    persistence = _StubPersistence()
    model = _StubModel(model_metadata=_metadata())
    for empty in ("", "   "):
        with pytest.raises(ModelValidationError, match="non-empty path"):
            persistence.save(model, empty)
        with pytest.raises(ModelValidationError, match="non-empty path"):
            persistence.load(empty)
        with pytest.raises(ModelValidationError, match="non-empty path"):
            persistence.exists(empty)
        with pytest.raises(ModelValidationError, match="non-empty path"):
            persistence.delete(empty)


def test_valid_paths_are_accepted() -> None:
    """Path and non-empty str values pass validation."""
    persistence = _StubPersistence()
    model = _StubModel(model_metadata=_metadata())

    persistence.save(model, "models/alpha.bin")
    assert persistence.saved is not None
    assert persistence.saved[0] is model
    assert persistence.saved[1] == Path("models/alpha.bin")

    loaded = persistence.load(Path("models/alpha.bin"))
    assert loaded is persistence._model
    assert persistence.loaded_path == Path("models/alpha.bin")

    assert persistence.exists("models/alpha.bin") is False
    assert persistence.exists_path == Path("models/alpha.bin")

    persistence.delete(Path("models/alpha.bin"))
    assert persistence.deleted_path == Path("models/alpha.bin")


def test_invalid_model_objects_are_rejected() -> None:
    """Objects that do not implement Model raise ModelValidationError."""
    persistence = _StubPersistence()
    for invalid in (None, "alpha-lgbm", 123, object(), {"name": "alpha-lgbm"}):
        with pytest.raises(ModelValidationError, match="Model protocol"):
            persistence.save(invalid, "models/alpha.bin")  # type: ignore[arg-type]


def test_valid_model_is_accepted() -> None:
    """Model protocol implementations pass model validation."""
    persistence = _StubPersistence()
    model = _StubModel(model_metadata=_metadata())
    assert isinstance(model, Model)
    persistence.save(model, "models/alpha.bin")
    assert persistence.saved is not None
    assert persistence.saved[0] is model


def test_exception_hierarchy() -> None:
    """ModelValidationError remains under ModelError."""
    assert issubclass(ModelValidationError, ModelError)
    assert issubclass(ModelError, Exception)


def test_package_exports_model_persistence() -> None:
    """ModelPersistence is exported from the ml.models package."""
    import cqros.ml.models as models_package

    assert "ModelPersistence" in models_package.__all__
    assert models_package.ModelPersistence is ModelPersistence
    assert ModelPersistence is ModelPersistenceDirect
