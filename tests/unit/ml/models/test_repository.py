"""Unit tests for CQROS ``ModelArtifactRepository``."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path
from typing import Self

import polars as pl
import pytest

from cqros.core.constants import STORAGE_DIR_MODELS
from cqros.ml.models import (
    ModelArtifactRef,
    ModelArtifactRepository,
    ModelFramework,
    ModelMetadata,
    ModelPersistence,
    ModelTaskType,
    ModelValidationError,
)
from cqros.ml.models.repository import ModelArtifactRepository as ModelArtifactRepositoryDirect
from cqros.storage import StorageLayout

_FRAMEWORK = "lightgbm"
_MODEL_NAME = "alpha-lgbm"
_VERSION = "1.0.0"


class _StubModel:
    """Minimal Model-protocol stub used by repository tests."""

    def __init__(self, metadata: ModelMetadata, *, marker: str = "v1") -> None:
        self._metadata = metadata
        self.marker = marker

    def fit(self, frame: pl.DataFrame) -> Self:
        return self

    def predict(self, frame: pl.DataFrame) -> pl.Series:
        return pl.Series("prediction", [0.0] * frame.height)

    def save(self, path: Path | str) -> None:
        Path(path).write_text(self.marker, encoding="utf-8")

    def load(self, path: Path | str) -> Self:
        self.marker = Path(path).read_text(encoding="utf-8")
        return self

    def metadata(self) -> ModelMetadata:
        return self._metadata


class _InMemoryModelPersistence(ModelPersistence):
    """ModelPersistence stub that stores markers at filesystem paths."""

    def __init__(self) -> None:
        self.save_paths: list[Path] = []
        self.load_paths: list[Path] = []
        self.exists_paths: list[Path] = []
        self.delete_paths: list[Path] = []
        self._models: dict[Path, _StubModel] = {}

    def save(self, model: object, path: Path | str) -> None:
        target = self._require_path(path)
        typed = self._require_model(model)
        assert isinstance(typed, _StubModel)
        self.save_paths.append(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(typed.marker, encoding="utf-8")
        self._models[target] = typed

    def load(self, path: Path | str) -> _StubModel:
        target = self._require_path(path)
        self.load_paths.append(target)
        try:
            stored = self._models[target]
        except KeyError as exc:
            raise ModelValidationError(
                "model artifact not found",
                error_code="ML-MODEL-PERS-TEST-001",
                details={"path": str(target)},
            ) from exc
        loaded = _StubModel(stored.metadata(), marker=target.read_text(encoding="utf-8"))
        return loaded

    def exists(self, path: Path | str) -> bool:
        target = self._require_path(path)
        self.exists_paths.append(target)
        return target.is_file()

    def delete(self, path: Path | str) -> None:
        target = self._require_path(path)
        self.delete_paths.append(target)
        if not target.is_file():
            raise ModelValidationError(
                "model artifact not found",
                error_code="ML-MODEL-PERS-TEST-001",
                details={"path": str(target)},
            )
        target.unlink()
        self._models.pop(target, None)


def _metadata(
    *,
    name: str = _MODEL_NAME,
    version: str = _VERSION,
    framework: ModelFramework = ModelFramework.LIGHTGBM,
) -> ModelMetadata:
    """Build ModelMetadata for repository tests."""
    return ModelMetadata(
        name=name,
        version=version,
        framework=framework,
        task_type=ModelTaskType.REGRESSION,
        feature_columns=("returns", "atr"),
        label_column="future_return_1",
        description="repository test model",
    )


def _touch_artifact(
    root: Path,
    *,
    framework: str,
    model_name: str,
    version: str,
) -> Path:
    """Create a minimal on-disk artifact directory with both required files."""
    directory = root / STORAGE_DIR_MODELS / framework / model_name / version
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.bin").write_bytes(b"model")
    (directory / "metadata.json").write_text(
        json.dumps(
            {
                "name": model_name,
                "version": version,
                "framework": framework,
                "task_type": "regression",
                "feature_columns": ["returns"],
                "label_column": "future_return_1",
                "description": "fixture",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return directory


@pytest.fixture
def layout(tmp_path: Path) -> StorageLayout:
    """Return a layout rooted at a temporary directory."""
    return StorageLayout(tmp_path)


@pytest.fixture
def persistence() -> _InMemoryModelPersistence:
    """Return an in-memory model persistence stub."""
    return _InMemoryModelPersistence()


@pytest.fixture
def repository(
    layout: StorageLayout,
    persistence: _InMemoryModelPersistence,
) -> ModelArtifactRepository:
    """Return a model artifact repository wired for tests."""
    return ModelArtifactRepository(layout, persistence)


def test_model_artifact_repository_is_exported_from_package() -> None:
    """Package export matches the repository module class."""
    assert ModelArtifactRepository is ModelArtifactRepositoryDirect


def test_model_artifact_ref_is_frozen_dataclass() -> None:
    """ModelArtifactRef is an immutable slotted dataclass."""
    ref = ModelArtifactRef(
        framework=_FRAMEWORK,
        model_name=_MODEL_NAME,
        version=_VERSION,
    )
    assert is_dataclass(ref)
    assert ref.framework == _FRAMEWORK
    assert ref.model_name == _MODEL_NAME
    assert ref.version == _VERSION
    assert ref == ModelArtifactRef(
        framework=_FRAMEWORK,
        model_name=_MODEL_NAME,
        version=_VERSION,
    )
    with pytest.raises(FrozenInstanceError):
        ref.version = "2.0.0"  # type: ignore[misc]


def test_model_path_partitioning_matches_layout_contract(layout: StorageLayout) -> None:
    """Model directories follow models/framework/model_name/version."""
    path = layout.model_path(_FRAMEWORK, _MODEL_NAME, _VERSION)
    assert path.name == _VERSION
    assert path.parent.name == _MODEL_NAME
    assert path.parent.parent.name == _FRAMEWORK
    assert path.parent.parent.parent.name == STORAGE_DIR_MODELS


def test_save_and_load_uses_layout_and_persistence(
    repository: ModelArtifactRepository,
    layout: StorageLayout,
    persistence: _InMemoryModelPersistence,
) -> None:
    """save/load use StorageLayout.model_path and ModelPersistence."""
    model = _StubModel(_metadata(), marker="payload-v1")
    expected_dir = layout.model_path(_FRAMEWORK, _MODEL_NAME, _VERSION)
    expected_model = expected_dir / "model.bin"
    expected_metadata = expected_dir / "metadata.json"

    repository.save(model)
    loaded = repository.load(
        framework=_FRAMEWORK,
        model_name=_MODEL_NAME,
        version=_VERSION,
    )

    assert persistence.save_paths == [expected_model]
    assert persistence.load_paths == [expected_model]
    assert expected_model.is_file()
    assert expected_metadata.is_file()
    assert isinstance(loaded, _StubModel)
    assert loaded.marker == "payload-v1"
    assert loaded.metadata() == model.metadata()


def test_save_persists_metadata_json(
    repository: ModelArtifactRepository,
    layout: StorageLayout,
) -> None:
    """save writes metadata.json beside the model binary."""
    model = _StubModel(_metadata())
    repository.save(model)

    metadata_path = layout.model_path(_FRAMEWORK, _MODEL_NAME, _VERSION) / "metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["name"] == _MODEL_NAME
    assert payload["version"] == _VERSION
    assert payload["framework"] == _FRAMEWORK
    assert payload["feature_columns"] == ["returns", "atr"]


def test_save_overwrites_existing_artifact(
    repository: ModelArtifactRepository,
) -> None:
    """Saving the same identity twice replaces the stored model."""
    repository.save(_StubModel(_metadata(), marker="first"))
    repository.save(_StubModel(_metadata(), marker="second"))
    loaded = repository.load(
        framework=_FRAMEWORK,
        model_name=_MODEL_NAME,
        version=_VERSION,
    )
    assert isinstance(loaded, _StubModel)
    assert loaded.marker == "second"


def test_exists_false_when_missing(repository: ModelArtifactRepository) -> None:
    """exists returns False when the artifact is absent."""
    assert (
        repository.exists(
            framework=_FRAMEWORK,
            model_name=_MODEL_NAME,
            version=_VERSION,
        )
        is False
    )


def test_exists_true_when_artifact_saved(
    repository: ModelArtifactRepository,
) -> None:
    """exists returns True after save."""
    repository.save(_StubModel(_metadata()))
    assert (
        repository.exists(
            framework=_FRAMEWORK,
            model_name=_MODEL_NAME,
            version=_VERSION,
        )
        is True
    )


def test_delete_removes_artifact(
    repository: ModelArtifactRepository,
    layout: StorageLayout,
    persistence: _InMemoryModelPersistence,
) -> None:
    """delete removes model binary and metadata through persistence."""
    repository.save(_StubModel(_metadata()))
    expected = layout.model_path(_FRAMEWORK, _MODEL_NAME, _VERSION) / "model.bin"

    repository.delete(
        framework=_FRAMEWORK,
        model_name=_MODEL_NAME,
        version=_VERSION,
    )

    assert persistence.delete_paths == [expected]
    assert (
        repository.exists(
            framework=_FRAMEWORK,
            model_name=_MODEL_NAME,
            version=_VERSION,
        )
        is False
    )


def test_delete_missing_raises(repository: ModelArtifactRepository) -> None:
    """delete raises ModelValidationError when the artifact is absent."""
    with pytest.raises(ModelValidationError) as exc_info:
        repository.delete(
            framework=_FRAMEWORK,
            model_name=_MODEL_NAME,
            version=_VERSION,
        )
    assert exc_info.value.error_code == "ML-MODEL-REPO-005"


def test_load_missing_raises(repository: ModelArtifactRepository) -> None:
    """load raises ModelValidationError when the artifact is absent."""
    with pytest.raises(ModelValidationError) as exc_info:
        repository.load(
            framework=_FRAMEWORK,
            model_name=_MODEL_NAME,
            version=_VERSION,
        )
    assert exc_info.value.error_code == "ML-MODEL-REPO-005"


def test_list_versions_returns_sorted_versions(tmp_path: Path) -> None:
    """list_versions discovers existing version directories in sorted order."""
    _touch_artifact(tmp_path, framework=_FRAMEWORK, model_name=_MODEL_NAME, version="1.0.0")
    _touch_artifact(tmp_path, framework=_FRAMEWORK, model_name=_MODEL_NAME, version="2.0.0")
    _touch_artifact(tmp_path, framework=_FRAMEWORK, model_name=_MODEL_NAME, version="1.5.0")
    incomplete = tmp_path / STORAGE_DIR_MODELS / _FRAMEWORK / _MODEL_NAME / "0.9.0"
    incomplete.mkdir(parents=True, exist_ok=True)
    (incomplete / "model.bin").write_bytes(b"missing-metadata")

    repository = ModelArtifactRepository(StorageLayout(tmp_path), _InMemoryModelPersistence())
    versions = repository.list_versions(framework=_FRAMEWORK, model_name=_MODEL_NAME)
    assert versions == ("1.0.0", "1.5.0", "2.0.0")


def test_list_versions_empty_when_missing(tmp_path: Path) -> None:
    """list_versions returns an empty tuple when no versions exist."""
    repository = ModelArtifactRepository(StorageLayout(tmp_path), _InMemoryModelPersistence())
    assert repository.list_versions(framework=_FRAMEWORK, model_name=_MODEL_NAME) == ()


def test_discover_frameworks_and_models(tmp_path: Path) -> None:
    """Discovery walks model trees without returning filesystem paths."""
    _touch_artifact(tmp_path, framework="lightgbm", model_name="alpha-lgbm", version="1.0.0")
    _touch_artifact(tmp_path, framework="xgboost", model_name="alpha-xgb", version="1.0.0")
    _touch_artifact(tmp_path, framework="lightgbm", model_name="beta-lgbm", version="2.0.0")

    repository = ModelArtifactRepository(StorageLayout(tmp_path), _InMemoryModelPersistence())
    assert repository.discover_frameworks() == ("lightgbm", "xgboost")
    assert repository.discover_models(framework="lightgbm") == ("alpha-lgbm", "beta-lgbm")


def test_discover_artifacts_finds_complete_versions(tmp_path: Path) -> None:
    """discover_artifacts returns deterministic ModelArtifactRef values."""
    _touch_artifact(tmp_path, framework="lightgbm", model_name="alpha-lgbm", version="1.0.0")
    _touch_artifact(tmp_path, framework="xgboost", model_name="alpha-xgb", version="2.0.0")

    repository = ModelArtifactRepository(StorageLayout(tmp_path), _InMemoryModelPersistence())
    artifacts = repository.discover_artifacts()

    assert artifacts == (
        ModelArtifactRef(framework="lightgbm", model_name="alpha-lgbm", version="1.0.0"),
        ModelArtifactRef(framework="xgboost", model_name="alpha-xgb", version="2.0.0"),
    )


def test_discover_artifacts_applies_filters(tmp_path: Path) -> None:
    """Discovery filters by framework and model-name allowlists."""
    _touch_artifact(tmp_path, framework="lightgbm", model_name="alpha-lgbm", version="1.0.0")
    _touch_artifact(tmp_path, framework="lightgbm", model_name="beta-lgbm", version="1.0.0")
    _touch_artifact(tmp_path, framework="xgboost", model_name="alpha-xgb", version="1.0.0")

    repository = ModelArtifactRepository(StorageLayout(tmp_path), _InMemoryModelPersistence())
    artifacts = repository.discover_artifacts(
        frameworks=("lightgbm",),
        model_names=("alpha-lgbm",),
    )
    assert artifacts == (
        ModelArtifactRef(framework="lightgbm", model_name="alpha-lgbm", version="1.0.0"),
    )


def test_rejects_empty_model_name(repository: ModelArtifactRepository) -> None:
    """Empty model names raise ModelValidationError."""
    with pytest.raises(ModelValidationError) as exc_info:
        repository.exists(framework=_FRAMEWORK, model_name="  ", version=_VERSION)
    assert exc_info.value.error_code == "ML-MODEL-REPO-002"


def test_rejects_empty_version(repository: ModelArtifactRepository) -> None:
    """Empty versions raise ModelValidationError."""
    with pytest.raises(ModelValidationError) as exc_info:
        repository.exists(framework=_FRAMEWORK, model_name=_MODEL_NAME, version="")
    assert exc_info.value.error_code == "ML-MODEL-REPO-003"


def test_rejects_invalid_reference_components(
    repository: ModelArtifactRepository,
) -> None:
    """Path separators in identity fields raise ModelValidationError."""
    with pytest.raises(ModelValidationError) as exc_info:
        repository.exists(
            framework=_FRAMEWORK,
            model_name="alpha/lgbm",
            version=_VERSION,
        )
    assert exc_info.value.error_code == "ML-MODEL-REPO-004"


def test_public_api_does_not_return_filesystem_paths(
    repository: ModelArtifactRepository,
) -> None:
    """save returns None and load returns a Model, never a Path."""
    model = _StubModel(_metadata())
    result = repository.save(model)
    loaded = repository.load(
        framework=_FRAMEWORK,
        model_name=_MODEL_NAME,
        version=_VERSION,
    )
    assert result is None
    assert isinstance(loaded, _StubModel)
    assert not isinstance(loaded, Path)
