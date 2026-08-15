"""Unit tests for CQROS ML ``ModelRegistry``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pytest

from cqros.ml.models.base import BaseModel
from cqros.ml.models.exceptions import ModelValidationError
from cqros.ml.models.metadata import ModelFramework, ModelMetadata, ModelTaskType
from cqros.ml.models.registry import ModelRegistry


def _metadata(name: str, **overrides: object) -> ModelMetadata:
    """Build ModelMetadata with a given name and optional field overrides."""
    values: dict[str, object] = {
        "name": name,
        "version": "1.0.0",
        "framework": ModelFramework.LIGHTGBM,
        "task_type": ModelTaskType.REGRESSION,
        "feature_columns": ("returns", "log_returns"),
        "label_column": "future_return_1",
        "description": "stub model",
    }
    values.update(overrides)
    return ModelMetadata(**values)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class _StubModel(BaseModel):
    """Minimal concrete model used only for registry unit tests."""

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


def _model(name: str, **overrides: object) -> _StubModel:
    """Build a stub model with the given metadata name."""
    return _StubModel(model_metadata=_metadata(name, **overrides))


def test_empty_registry() -> None:
    """A new registry contains no models."""
    registry = ModelRegistry()
    assert registry.count() == 0
    assert registry.list() == ()
    assert registry.metadata() == ()
    assert registry.exists("alpha-lgbm") is False


def test_register_and_get() -> None:
    """register stores a model that get can retrieve by metadata name."""
    registry = ModelRegistry()
    model = _model("alpha-lgbm")
    registry.register(model)
    assert registry.get("alpha-lgbm") is model


def test_register_rejects_duplicates() -> None:
    """Duplicate model names raise ModelValidationError."""
    registry = ModelRegistry()
    registry.register(_model("alpha-lgbm"))
    with pytest.raises(ModelValidationError, match="already registered"):
        registry.register(_model("alpha-lgbm", version="2.0.0"))
    assert registry.get("alpha-lgbm").metadata().version == "1.0.0"
    assert registry.count() == 1


def test_register_many_registers_all() -> None:
    """register_many stores every provided model."""
    registry = ModelRegistry()
    alpha = _model("alpha-lgbm")
    beta = _model("beta-xgb", framework=ModelFramework.XGBOOST)
    registry.register_many((alpha, beta))
    assert registry.get("alpha-lgbm") is alpha
    assert registry.get("beta-xgb") is beta
    assert registry.count() == 2


def test_register_many_is_atomic_on_duplicate_existing() -> None:
    """register_many leaves the registry unchanged when a name already exists."""
    registry = ModelRegistry()
    registry.register(_model("alpha-lgbm"))
    with pytest.raises(ModelValidationError, match="already registered"):
        registry.register_many((_model("beta-xgb"), _model("alpha-lgbm")))
    assert registry.count() == 1
    assert registry.exists("alpha-lgbm") is True
    assert registry.exists("beta-xgb") is False


def test_register_many_is_atomic_on_duplicate_within_batch() -> None:
    """register_many rejects duplicate names within the same batch."""
    registry = ModelRegistry()
    with pytest.raises(ModelValidationError, match="already registered"):
        registry.register_many((_model("alpha-lgbm"), _model("alpha-lgbm", version="2.0.0")))
    assert registry.count() == 0
    assert registry.list() == ()


def test_register_many_is_atomic_on_invalid_object() -> None:
    """register_many leaves the registry unchanged when an invalid object appears."""
    registry = ModelRegistry()
    with pytest.raises(ModelValidationError, match="Model protocol"):
        registry.register_many((_model("alpha-lgbm"), object()))  # type: ignore[arg-type]
    assert registry.count() == 0
    assert registry.list() == ()


def test_get_unknown_raises() -> None:
    """get raises ModelValidationError for missing names."""
    registry = ModelRegistry()
    with pytest.raises(ModelValidationError, match="not registered"):
        registry.get("missing")


def test_exists() -> None:
    """exists reports registration presence without raising."""
    registry = ModelRegistry()
    assert registry.exists("alpha-lgbm") is False
    registry.register(_model("alpha-lgbm"))
    assert registry.exists("alpha-lgbm") is True
    assert registry.exists("beta-xgb") is False


def test_remove() -> None:
    """remove deletes a registered model and rejects missing names."""
    registry = ModelRegistry()
    registry.register(_model("alpha-lgbm"))
    registry.remove("alpha-lgbm")
    assert registry.exists("alpha-lgbm") is False
    assert registry.count() == 0
    with pytest.raises(ModelValidationError, match="not registered"):
        registry.remove("alpha-lgbm")


def test_clear() -> None:
    """clear removes all registered models."""
    registry = ModelRegistry()
    registry.register_many((_model("alpha-lgbm"), _model("beta-xgb")))
    registry.clear()
    assert registry.count() == 0
    assert registry.list() == ()
    assert registry.metadata() == ()


def test_list_preserves_insertion_order() -> None:
    """list returns models in registration insertion order."""
    registry = ModelRegistry()
    registry.register_many((_model("zeta"), _model("alpha"), _model("mu")))
    assert tuple(model.metadata().name for model in registry.list()) == (
        "zeta",
        "alpha",
        "mu",
    )


def test_metadata() -> None:
    """metadata returns ModelMetadata for every registered model."""
    registry = ModelRegistry()
    registry.register_many(
        (
            _model(
                "alpha-lgbm",
                version="1.2.0",
                framework=ModelFramework.LIGHTGBM,
                description="LightGBM baseline",
            ),
            _model(
                "beta-xgb",
                version="2.0.0",
                framework=ModelFramework.XGBOOST,
                description="XGBoost baseline",
            ),
        )
    )
    metadata = registry.metadata()
    assert isinstance(metadata, tuple)
    assert len(metadata) == 2
    assert all(isinstance(item, ModelMetadata) for item in metadata)
    assert metadata[0].name == "alpha-lgbm"
    assert metadata[0].version == "1.2.0"
    assert metadata[0].framework is ModelFramework.LIGHTGBM
    assert metadata[0].description == "LightGBM baseline"
    assert metadata[1].name == "beta-xgb"
    assert metadata[1].version == "2.0.0"
    assert metadata[1].framework is ModelFramework.XGBOOST


def test_count() -> None:
    """count tracks the number of registered models."""
    registry = ModelRegistry()
    assert registry.count() == 0
    registry.register(_model("alpha-lgbm"))
    assert registry.count() == 1
    registry.register(_model("beta-xgb"))
    assert registry.count() == 2
    registry.remove("alpha-lgbm")
    assert registry.count() == 1


def test_invalid_objects_rejected() -> None:
    """Objects that do not implement Model raise ModelValidationError."""
    registry = ModelRegistry()
    for invalid in (None, "alpha-lgbm", 123, object(), {"name": "alpha-lgbm"}):
        with pytest.raises(ModelValidationError, match="Model protocol"):
            registry.register(invalid)  # type: ignore[arg-type]
    assert registry.count() == 0


def test_returned_collections_are_immutable_snapshots() -> None:
    """Returned tuples are snapshots unaffected by later registry mutation."""
    registry = ModelRegistry()
    registry.register_many((_model("alpha-lgbm"), _model("beta-xgb")))
    models = registry.list()
    metadata = registry.metadata()
    assert isinstance(models, tuple)
    assert isinstance(metadata, tuple)
    registry.clear()
    assert tuple(model.metadata().name for model in models) == ("alpha-lgbm", "beta-xgb")
    assert tuple(item.name for item in metadata) == ("alpha-lgbm", "beta-xgb")
    assert registry.list() == ()
    assert registry.count() == 0


def test_register_does_not_mutate_model() -> None:
    """Registry stores the model reference without altering its metadata."""
    registry = ModelRegistry()
    model = _model("alpha-lgbm", version="1.0.0")
    registry.register(model)
    assert model.metadata().name == "alpha-lgbm"
    assert model.metadata().version == "1.0.0"
    assert registry.get("alpha-lgbm") is model


def test_package_exports_model_registry() -> None:
    """ModelRegistry is exported from the ml.models package."""
    import cqros.ml.models as models_package

    assert "ModelRegistry" in models_package.__all__
    assert models_package.ModelRegistry is ModelRegistry
