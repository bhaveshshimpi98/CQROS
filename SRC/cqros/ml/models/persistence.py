"""CQROS ML Model persistence contract.

Purpose:
    Define the framework-independent serialization contract for saving and
    loading trained model artifacts.

Responsibilities:
    - Expose ``ModelPersistence`` as the shared persistence ABC
    - Validate paths and model arguments for concrete implementations
    - Remain free of filesystem I/O, framework serializers, and training logic

Dependencies:
    ``pathlib``, ``cqros.ml.models.exceptions``, and
    ``cqros.ml.models.interfaces.Model``.

Public API:
    ``ModelPersistence``

Notes:
    Concrete backends implement ``save``, ``load``, ``exists``, and
    ``delete``. This module never touches the filesystem or any ML framework
    serializer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Final

from cqros.ml.models.exceptions import ModelValidationError
from cqros.ml.models.interfaces import Model

__all__ = ["ModelPersistence"]

_ERROR_PATH_TYPE: Final[str] = "ML-MODEL-PERS-001"
_ERROR_PATH_EMPTY: Final[str] = "ML-MODEL-PERS-002"
_ERROR_NOT_MODEL: Final[str] = "ML-MODEL-PERS-003"


class ModelPersistence(ABC):
    """Abstract persistence contract for CQROS ML model artifacts.

    Concrete implementations serialize and deserialize models through a
    storage backend. This base class performs argument validation only and
    never reads, writes, or deletes files. It contains no LightGBM, XGBoost,
    CatBoost, pickle, or joblib logic.
    """

    __slots__ = ()

    @abstractmethod
    def save(self, model: Model, path: Path | str) -> None:
        """Persist ``model`` artifacts to ``path``.

        Args:
            model: Model instance to serialize. Must implement ``Model``.
            path: Destination path for the serialized model.

        Raises:
            ModelValidationError: If ``model`` or ``path`` is invalid.
        """

    @abstractmethod
    def load(self, path: Path | str) -> Model:
        """Load model artifacts from ``path``.

        Args:
            path: Source path of the serialized model.

        Returns:
            The deserialized model instance.

        Raises:
            ModelValidationError: If ``path`` is invalid.
        """

    @abstractmethod
    def exists(self, path: Path | str) -> bool:
        """Return whether model artifacts exist at ``path``.

        Args:
            path: Filesystem path to check.

        Returns:
            ``True`` when artifacts exist at ``path``, otherwise ``False``.

        Raises:
            ModelValidationError: If ``path`` is invalid.
        """

    @abstractmethod
    def delete(self, path: Path | str) -> None:
        """Delete model artifacts at ``path``.

        Args:
            path: Filesystem path to delete.

        Raises:
            ModelValidationError: If ``path`` is invalid.
        """

    def _require_path(self, path: object, *, parameter: str = "path") -> Path:
        """Validate that ``path`` is a non-empty ``Path`` or ``str``.

        Args:
            path: Candidate filesystem path.
            parameter: Parameter name used in error messages.

        Returns:
            ``path`` as a ``Path``.

        Raises:
            ModelValidationError: If ``path`` has an invalid type or is empty.
        """
        if isinstance(path, Path):
            return path
        if isinstance(path, str):
            if path.strip() == "":
                raise ModelValidationError(
                    f"{parameter} must be a non-empty path",
                    error_code=_ERROR_PATH_EMPTY,
                    details={"parameter": parameter, "value": path},
                )
            return Path(path)
        raise ModelValidationError(
            f"{parameter} must be a Path or str",
            error_code=_ERROR_PATH_TYPE,
            details={"parameter": parameter, "value_type": type(path).__name__},
        )

    def _require_model(self, model: object, *, parameter: str = "model") -> Model:
        """Validate that ``model`` implements the ``Model`` protocol.

        Args:
            model: Candidate model instance.
            parameter: Parameter name used in error messages.

        Returns:
            ``model`` narrowed as a ``Model``.

        Raises:
            ModelValidationError: If ``model`` does not implement ``Model``.
        """
        if not isinstance(model, Model):
            raise ModelValidationError(
                f"{parameter} must implement the Model protocol",
                error_code=_ERROR_NOT_MODEL,
                details={"parameter": parameter, "value_type": type(model).__name__},
            )
        return model
