"""CQROS ML model artifact repository.

Purpose:
    Provide a path-free facade for persisting and retrieving versioned trained
    model artifacts and their metadata.

Responsibilities:
    - Resolve storage locations for model version directories via
      ``StorageLayout.model_path``
    - Persist, load, check existence, and delete model artifacts
    - Discover frameworks, model names, and versioned artifacts under the
      models tier
    - Delegate model serialization exclusively to an injected
      ``ModelPersistence`` backend
    - Persist ``ModelMetadata`` alongside each model artifact
    - Keep filesystem paths out of the public API
    - Remain free of training, prediction, evaluation, and framework-specific
      serialization logic

Dependencies:
    ``json``, ``logging``, ``pathlib``, ``cqros.core``, ``cqros.ml.models``,
    and ``cqros.storage.layout``.

Public API:
    ``ModelArtifactRef``, ``ModelArtifactRepository``
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from cqros.core.constants import STORAGE_DIR_MODELS
from cqros.ml.models.exceptions import ModelValidationError
from cqros.ml.models.interfaces import Model
from cqros.ml.models.metadata import ModelFramework, ModelMetadata, ModelTaskType
from cqros.ml.models.persistence import ModelPersistence
from cqros.storage.layout import StorageLayout

__all__ = [
    "ModelArtifactRef",
    "ModelArtifactRepository",
]

_logger = logging.getLogger(__name__)

_MODEL_FILENAME: Final[str] = "model.bin"
_METADATA_FILENAME: Final[str] = "metadata.json"

_ERROR_FRAMEWORK_EMPTY: Final[str] = "ML-MODEL-REPO-001"
_ERROR_MODEL_NAME_EMPTY: Final[str] = "ML-MODEL-REPO-002"
_ERROR_VERSION_EMPTY: Final[str] = "ML-MODEL-REPO-003"
_ERROR_INVALID_REFERENCE: Final[str] = "ML-MODEL-REPO-004"
_ERROR_ARTIFACT_NOT_FOUND: Final[str] = "ML-MODEL-REPO-005"
_ERROR_METADATA_INVALID: Final[str] = "ML-MODEL-REPO-006"

_INVALID_PATH_CHARS: Final[frozenset[str]] = frozenset({"/", "\\", "\0"})


@dataclass(frozen=True, slots=True)
class ModelArtifactRef:
    """Identity of one discovered versioned model artifact.

    Attributes:
        framework: Machine-learning framework identifier.
        model_name: Stable model identifier.
        version: Model version identifier.
    """

    framework: str
    model_name: str
    version: str


class ModelArtifactRepository:
    """Repository facade for versioned trained model artifacts.

    Callers identify artifacts by framework, model name, and version. Paths
    are composed privately via ``StorageLayout.model_path`` and never
    returned. Model serialization is delegated entirely to the injected
    ``ModelPersistence`` backend. Metadata is stored beside the model binary.

    Artifact layout::

        models/{framework}/{model_name}/{version}/model.bin
        models/{framework}/{model_name}/{version}/metadata.json

    Args:
        layout: Canonical path composer for the data lake.
        persistence: Model serialization backend implementing
            ``ModelPersistence``.
        logger: Optional logger instance. Defaults to the module logger.
    """

    __slots__ = ("_layout", "_logger", "_persistence")

    _layout: StorageLayout
    _persistence: ModelPersistence
    _logger: logging.Logger

    def __init__(
        self,
        layout: StorageLayout,
        persistence: ModelPersistence,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the repository with injected layout and persistence.

        Args:
            layout: Canonical path composer for the data lake.
            persistence: Backend used for model binary save/load/delete.
            logger: Optional logger instance.
        """
        self._layout = layout
        self._persistence = persistence
        self._logger = logger if logger is not None else _logger

    def discover_frameworks(self) -> tuple[str, ...]:
        """Return sorted frameworks with at least one model artifact.

        Returns:
            Deterministically ordered framework identifiers discovered on
            disk.
        """
        base = self._models_root()
        if not base.is_dir():
            return ()
        return tuple(sorted(path.name for path in base.iterdir() if path.is_dir()))

    def discover_models(self, *, framework: str) -> tuple[str, ...]:
        """Return sorted model names with at least one version for ``framework``.

        Args:
            framework: Machine-learning framework identifier.

        Returns:
            Deterministically ordered model names discovered on disk.

        Raises:
            ModelValidationError: If ``framework`` is empty or invalid.
        """
        normalized_framework = _require_identity_component(
            framework,
            parameter="framework",
            empty_error_code=_ERROR_FRAMEWORK_EMPTY,
        )
        base = self._models_root() / normalized_framework
        if not base.is_dir():
            return ()
        return tuple(sorted(path.name for path in base.iterdir() if path.is_dir()))

    def list_versions(self, *, framework: str, model_name: str) -> tuple[str, ...]:
        """Return sorted versions present for ``framework`` / ``model_name``.

        Args:
            framework: Machine-learning framework identifier.
            model_name: Stable model identifier.

        Returns:
            Deterministically ordered version identifiers for existing
            artifacts.

        Raises:
            ModelValidationError: If identity fields are empty or invalid.
        """
        normalized_framework = _require_identity_component(
            framework,
            parameter="framework",
            empty_error_code=_ERROR_FRAMEWORK_EMPTY,
        )
        normalized_model_name = _require_identity_component(
            model_name,
            parameter="model_name",
            empty_error_code=_ERROR_MODEL_NAME_EMPTY,
        )
        return self._discover_versions(
            framework=normalized_framework,
            model_name=normalized_model_name,
        )

    def discover_artifacts(
        self,
        *,
        frameworks: Sequence[str] | None = None,
        model_names: Sequence[str] | None = None,
    ) -> tuple[ModelArtifactRef, ...]:
        """Discover versioned model artifacts matching optional filters.

        Missing model trees are skipped. Only version directories that contain
        both ``model.bin`` and ``metadata.json`` are included. Paths are
        never returned.

        Args:
            frameworks: Optional framework allowlist. ``None`` discovers every
                framework present under the models tier.
            model_names: Optional model-name allowlist. ``None`` discovers
                every model name present for each framework.

        Returns:
            Deterministically ordered artifact references.
        """
        framework_filter = set(frameworks) if frameworks is not None else None
        model_filter = set(model_names) if model_names is not None else None

        items: list[ModelArtifactRef] = []
        for framework in self.discover_frameworks():
            if framework_filter is not None and framework not in framework_filter:
                continue
            for model_name in self.discover_models(framework=framework):
                if model_filter is not None and model_name not in model_filter:
                    continue
                for version in self._discover_versions(
                    framework=framework,
                    model_name=model_name,
                ):
                    items.append(
                        ModelArtifactRef(
                            framework=framework,
                            model_name=model_name,
                            version=version,
                        )
                    )

        return tuple(
            sorted(
                items,
                key=lambda item: (item.framework, item.model_name, item.version),
            )
        )

    def exists(
        self,
        *,
        framework: str,
        model_name: str,
        version: str,
    ) -> bool:
        """Return whether a versioned model artifact exists.

        Existence requires both the model binary (via ``ModelPersistence``)
        and the companion metadata file.

        Args:
            framework: Machine-learning framework identifier.
            model_name: Stable model identifier.
            version: Model version identifier.

        Returns:
            ``True`` when the artifact exists; otherwise ``False``.

        Raises:
            ModelValidationError: If identity fields are empty or invalid.
        """
        identity = _normalize_identity(
            framework=framework,
            model_name=model_name,
            version=version,
        )
        directory = self._layout.model_path(
            identity.framework,
            identity.model_name,
            identity.version,
        )
        model_file = directory / _MODEL_FILENAME
        metadata_file = directory / _METADATA_FILENAME
        present = self._persistence.exists(model_file) and metadata_file.is_file()
        self._logger.debug(
            "Model artifact exists" if present else "Model artifact does not exist",
            extra=_model_log_extra(
                framework=identity.framework,
                model_name=identity.model_name,
                version=identity.version,
            ),
        )
        return present

    def save(self, model: Model) -> None:
        """Persist a trained model artifact and its metadata.

        Identity is taken from ``model.metadata()``. Existing artifacts at the
        same framework/name/version location are overwritten.

        Args:
            model: Fitted model implementing the ``Model`` protocol.

        Raises:
            ModelValidationError: If identity fields derived from metadata are
                empty or invalid.
        """
        metadata = model.metadata()
        identity = _normalize_identity(
            framework=str(metadata.framework),
            model_name=metadata.name,
            version=metadata.version,
        )
        directory = self._layout.model_path(
            identity.framework,
            identity.model_name,
            identity.version,
        )
        directory.mkdir(parents=True, exist_ok=True)
        model_file = directory / _MODEL_FILENAME
        metadata_file = directory / _METADATA_FILENAME

        self._logger.debug(
            "Saving model artifact",
            extra=_model_log_extra(
                framework=identity.framework,
                model_name=identity.model_name,
                version=identity.version,
            ),
        )
        self._persistence.save(model, model_file)
        _write_metadata(metadata_file, metadata)
        self._logger.info(
            "Saved model artifact",
            extra=_model_log_extra(
                framework=identity.framework,
                model_name=identity.model_name,
                version=identity.version,
            ),
        )

    def load(
        self,
        *,
        framework: str,
        model_name: str,
        version: str,
    ) -> Model:
        """Load a versioned model artifact.

        Args:
            framework: Machine-learning framework identifier.
            model_name: Stable model identifier.
            version: Model version identifier.

        Returns:
            Deserialized model instance.

        Raises:
            ModelValidationError: If identity fields are empty/invalid or the
                artifact does not exist.
        """
        identity = _normalize_identity(
            framework=framework,
            model_name=model_name,
            version=version,
        )
        directory = self._layout.model_path(
            identity.framework,
            identity.model_name,
            identity.version,
        )
        model_file = directory / _MODEL_FILENAME
        metadata_file = directory / _METADATA_FILENAME
        if not self._persistence.exists(model_file) or not metadata_file.is_file():
            raise ModelValidationError(
                "model artifact not found",
                error_code=_ERROR_ARTIFACT_NOT_FOUND,
                details={
                    "framework": identity.framework,
                    "model_name": identity.model_name,
                    "version": identity.version,
                },
            )

        self._logger.debug(
            "Loading model artifact",
            extra=_model_log_extra(
                framework=identity.framework,
                model_name=identity.model_name,
                version=identity.version,
            ),
        )
        # Validate companion metadata without mutating the loaded model.
        _read_metadata(metadata_file)
        model = self._persistence.load(model_file)
        self._logger.info(
            "Loaded model artifact",
            extra=_model_log_extra(
                framework=identity.framework,
                model_name=identity.model_name,
                version=identity.version,
            ),
        )
        return model

    def delete(
        self,
        *,
        framework: str,
        model_name: str,
        version: str,
    ) -> None:
        """Delete a versioned model artifact and its metadata.

        Args:
            framework: Machine-learning framework identifier.
            model_name: Stable model identifier.
            version: Model version identifier.

        Raises:
            ModelValidationError: If identity fields are empty/invalid or the
                artifact does not exist.
        """
        identity = _normalize_identity(
            framework=framework,
            model_name=model_name,
            version=version,
        )
        directory = self._layout.model_path(
            identity.framework,
            identity.model_name,
            identity.version,
        )
        model_file = directory / _MODEL_FILENAME
        metadata_file = directory / _METADATA_FILENAME
        if not self._persistence.exists(model_file) and not metadata_file.is_file():
            raise ModelValidationError(
                "model artifact not found",
                error_code=_ERROR_ARTIFACT_NOT_FOUND,
                details={
                    "framework": identity.framework,
                    "model_name": identity.model_name,
                    "version": identity.version,
                },
            )

        self._logger.debug(
            "Deleting model artifact",
            extra=_model_log_extra(
                framework=identity.framework,
                model_name=identity.model_name,
                version=identity.version,
            ),
        )
        if self._persistence.exists(model_file):
            self._persistence.delete(model_file)
        if metadata_file.is_file():
            metadata_file.unlink()
        self._logger.info(
            "Deleted model artifact",
            extra=_model_log_extra(
                framework=identity.framework,
                model_name=identity.model_name,
                version=identity.version,
            ),
        )

    def _models_root(self) -> Path:
        """Return the models tier directory."""
        return self._layout.root / STORAGE_DIR_MODELS

    def _discover_versions(self, *, framework: str, model_name: str) -> tuple[str, ...]:
        """Return sorted versions that contain both model and metadata files."""
        base = self._models_root() / framework / model_name
        if not base.is_dir():
            return ()
        versions: list[str] = []
        for path in sorted(base.iterdir()):
            if not path.is_dir():
                continue
            model_file = path / _MODEL_FILENAME
            metadata_file = path / _METADATA_FILENAME
            if model_file.is_file() and metadata_file.is_file():
                versions.append(path.name)
        return tuple(versions)


def _normalize_identity(
    *,
    framework: str,
    model_name: str,
    version: str,
) -> ModelArtifactRef:
    """Validate and normalize artifact identity fields."""
    return ModelArtifactRef(
        framework=_require_identity_component(
            framework,
            parameter="framework",
            empty_error_code=_ERROR_FRAMEWORK_EMPTY,
        ),
        model_name=_require_identity_component(
            model_name,
            parameter="model_name",
            empty_error_code=_ERROR_MODEL_NAME_EMPTY,
        ),
        version=_require_identity_component(
            version,
            parameter="version",
            empty_error_code=_ERROR_VERSION_EMPTY,
        ),
    )


def _require_identity_component(
    value: object,
    *,
    parameter: str,
    empty_error_code: str,
) -> str:
    """Validate a non-empty identity path component."""
    if not isinstance(value, str) or value.strip() == "":
        raise ModelValidationError(
            f"{parameter} must be a non-empty string",
            error_code=empty_error_code,
            details={"parameter": parameter, "value": value},
        )
    normalized = value.strip()
    if normalized in {".", ".."} or any(char in normalized for char in _INVALID_PATH_CHARS):
        raise ModelValidationError(
            f"{parameter} is not a valid artifact reference component",
            error_code=_ERROR_INVALID_REFERENCE,
            details={"parameter": parameter, "value": value},
        )
    return normalized


def _write_metadata(path: Path, metadata: ModelMetadata) -> None:
    """Serialize ``metadata`` to JSON at ``path``."""
    payload: Mapping[str, object] = {
        "name": metadata.name,
        "version": metadata.version,
        "framework": str(metadata.framework),
        "task_type": str(metadata.task_type),
        "feature_columns": list(metadata.feature_columns),
        "label_column": metadata.label_column,
        "description": metadata.description,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_metadata(path: Path) -> ModelMetadata:
    """Deserialize ``ModelMetadata`` from JSON at ``path``."""
    try:
        raw_payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelValidationError(
            "model metadata is missing or invalid",
            error_code=_ERROR_METADATA_INVALID,
            details={"path": str(path), "reason": str(exc)},
        ) from exc
    if not isinstance(raw_payload, dict):
        raise ModelValidationError(
            "model metadata must be a JSON object",
            error_code=_ERROR_METADATA_INVALID,
            details={"path": str(path), "value_type": type(raw_payload).__name__},
        )
    payload_mapping = cast(dict[object, object], raw_payload)
    payload: dict[str, object] = {str(key): value for key, value in payload_mapping.items()}
    try:
        feature_columns_raw = payload["feature_columns"]
        if not isinstance(feature_columns_raw, list):
            raise TypeError("feature_columns must be a list")
        feature_items = cast(list[object], feature_columns_raw)
        return ModelMetadata(
            name=str(payload["name"]),
            version=str(payload["version"]),
            framework=ModelFramework(str(payload["framework"])),
            task_type=ModelTaskType(str(payload["task_type"])),
            feature_columns=tuple(str(item) for item in feature_items),
            label_column=str(payload["label_column"]),
            description=str(payload["description"]),
        )
    except (KeyError, TypeError, ValueError, ModelValidationError) as exc:
        raise ModelValidationError(
            "model metadata contents are invalid",
            error_code=_ERROR_METADATA_INVALID,
            details={"path": str(path), "reason": str(exc)},
        ) from exc


def _model_log_extra(
    *,
    framework: str,
    model_name: str,
    version: str,
) -> dict[str, object]:
    """Build structured log fields for a model artifact operation."""
    return {
        "tier": "models",
        "framework": framework,
        "model_name": model_name,
        "version": version,
    }
