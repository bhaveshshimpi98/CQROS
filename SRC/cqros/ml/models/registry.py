"""CQROS ML Model registry.

Purpose:
    Provide the authoritative in-memory catalog of available model
    implementations for registration and lookup.

Responsibilities:
    - Register ``Model`` instances by unique ``metadata().name``
    - Provide deterministic lookup, listing, and metadata projection
    - Reject duplicates and objects that do not implement ``Model``
    - Remain free of training, prediction, serialization, and framework logic

Dependencies:
    ``cqros.ml.models.exceptions``, ``cqros.ml.models.interfaces.Model``, and
    ``cqros.ml.models.metadata.ModelMetadata``.

Public API:
    ``ModelRegistry``

Notes:
    This registry is not thread-safe. Callers that share one instance across
    threads must provide their own synchronization.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final, cast

from cqros.ml.models.exceptions import ModelValidationError
from cqros.ml.models.interfaces import Model
from cqros.ml.models.metadata import ModelMetadata

__all__ = ["ModelRegistry"]

_ERROR_NOT_MODEL: Final[str] = "ML-MODEL-REG-001"
_ERROR_METADATA_TYPE: Final[str] = "ML-MODEL-REG-002"
_ERROR_NAME_BLANK: Final[str] = "ML-MODEL-REG-003"
_ERROR_DUPLICATE: Final[str] = "ML-MODEL-REG-004"
_ERROR_UNKNOWN: Final[str] = "ML-MODEL-REG-005"


class ModelRegistry:
    """Authoritative catalog of registered CQROS ML model implementations.

    Models are indexed by ``metadata().name``. The registry stores references
    to the supplied ``Model`` instances and never mutates, instantiates,
    trains, or predicts with them. Returned collections are new tuples and
    do not expose the internal mapping. Insertion order is preserved.

    Notes:
        This registry is not thread-safe. Concurrent mutation from multiple
        threads requires external synchronization.
    """

    __slots__ = ("_models",)

    def __init__(self) -> None:
        """Initialize an empty model registry."""
        self._models: dict[str, Model] = {}

    def register(self, model: Model) -> None:
        """Register one model by metadata name.

        Args:
            model: Model instance to register. Must not be mutated by the
                registry after registration.

        Raises:
            ModelValidationError: If ``model`` does not implement ``Model``,
                exposes invalid metadata, or a model with the same name
                already exists.
        """
        name = _require_model_name(model)
        if name in self._models:
            raise ModelValidationError(
                f"model already registered: {name}",
                error_code=_ERROR_DUPLICATE,
                details={"name": name},
            )
        self._models[name] = model

    def register_many(self, models: Iterable[Model]) -> None:
        """Register multiple models atomically.

        Either every model in ``models`` is registered, or the registry
        remains unchanged when any registration would fail.

        Args:
            models: Models to register.

        Raises:
            ModelValidationError: If any entry is invalid, already
                registered, or duplicated within ``models``.
        """
        pending: dict[str, Model] = {}
        for model in models:
            name = _require_model_name(model)
            if name in self._models or name in pending:
                raise ModelValidationError(
                    f"model already registered: {name}",
                    error_code=_ERROR_DUPLICATE,
                    details={"name": name},
                )
            pending[name] = model
        self._models.update(pending)

    def get(self, name: str) -> Model:
        """Return the registered model for ``name``.

        Args:
            name: Model name to look up.

        Returns:
            The registered model instance.

        Raises:
            ModelValidationError: If no model is registered under ``name``.
        """
        model = self._models.get(name)
        if model is None:
            raise ModelValidationError(
                f"model not registered: {name}",
                error_code=_ERROR_UNKNOWN,
                details={"name": name},
            )
        return model

    def exists(self, name: str) -> bool:
        """Return whether a model is registered under ``name``.

        Args:
            name: Model name to check.

        Returns:
            ``True`` when the name is registered, otherwise ``False``.
        """
        return name in self._models

    def remove(self, name: str) -> None:
        """Remove a registered model by name.

        Args:
            name: Model name to remove.

        Raises:
            ModelValidationError: If no model is registered under ``name``.
        """
        if name not in self._models:
            raise ModelValidationError(
                f"model not registered: {name}",
                error_code=_ERROR_UNKNOWN,
                details={"name": name},
            )
        del self._models[name]

    def clear(self) -> None:
        """Remove all registered models."""
        self._models.clear()

    def list(self) -> tuple[Model, ...]:
        """Return registered models in insertion order.

        Returns:
            A new tuple of registered model instances.
        """
        return tuple(self._models.values())

    def metadata(self) -> tuple[ModelMetadata, ...]:
        """Return metadata snapshots for all registered models.

        Returns:
            A new tuple of ``ModelMetadata`` objects in insertion order.
        """
        return tuple(model.metadata() for model in self._models.values())

    def count(self) -> int:
        """Return the number of registered models.

        Returns:
            Count of models currently stored in the registry.
        """
        return len(self._models)


def _require_model_name(model: object) -> str:
    """Validate ``model`` and return its non-blank metadata name.

    Args:
        model: Candidate model instance.

    Returns:
        The validated model name from ``model.metadata().name``.

    Raises:
        ModelValidationError: If ``model`` is not a ``Model``, does not
            expose ``ModelMetadata``, or has a blank name.
    """
    if not isinstance(model, Model):
        raise ModelValidationError(
            "model must implement the Model protocol",
            error_code=_ERROR_NOT_MODEL,
            details={"value_type": type(model).__name__},
        )

    metadata = model.metadata()
    if not isinstance(cast(object, metadata), ModelMetadata):
        raise ModelValidationError(
            "model.metadata() must return a ModelMetadata instance",
            error_code=_ERROR_METADATA_TYPE,
            details={"value_type": type(metadata).__name__},
        )

    name = metadata.name
    if name.strip() == "":
        raise ModelValidationError(
            "model name must be a non-blank string",
            error_code=_ERROR_NAME_BLANK,
            details={"name": name},
        )
    return name
