"""CQROS Feature Engine registry.

Purpose:
    Provide the authoritative in-memory catalog of available features for
    registration and lookup.

Responsibilities:
    - Register immutable ``Feature`` instances by unique name
    - Provide deterministic lookup, listing, and metadata projection
    - Reject duplicate and blank feature names
    - Remain free of execution, dependency resolution, storage, validation,
      pipeline, and dataframe logic

Dependencies:
    ``cqros.features.exceptions``, ``cqros.features.interfaces.Feature``, and
    ``cqros.features.metadata.FeatureMetadata``.

Public API:
    ``FeatureRegistry``

Notes:
    This registry is not thread-safe. Callers that share one instance across
    threads must provide their own synchronization.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

from cqros.features.exceptions import (
    DuplicateFeatureError,
    FeatureRegistrationError,
    UnknownFeatureError,
)
from cqros.features.interfaces import Feature
from cqros.features.metadata import FeatureMetadata

__all__ = ["FeatureRegistry"]

_ERROR_NAME_BLANK: Final[str] = "FEATURE-REG-001"
_ERROR_DUPLICATE: Final[str] = "FEATURE-REG-002"
_ERROR_UNKNOWN: Final[str] = "FEATURE-REG-003"
_DEFAULT_AUTHOR: Final[str] = ""


class FeatureRegistry:
    """Authoritative catalog of registered CQROS features.

    Features are indexed by name. The registry stores references to the
    supplied ``Feature`` instances and never mutates them. Returned
    collections are new tuples and do not expose the internal mapping.

    Notes:
        This registry is not thread-safe. Concurrent mutation from multiple
        threads requires external synchronization.
    """

    __slots__ = ("_features",)

    def __init__(self) -> None:
        """Initialize an empty feature registry."""
        self._features: dict[str, Feature] = {}

    def register(self, feature: Feature) -> None:
        """Register one feature by name.

        Args:
            feature: Feature instance to register. Must not be mutated by the
                registry after registration.

        Raises:
            FeatureRegistrationError: If ``feature.name`` is blank.
            DuplicateFeatureError: If a feature with the same name exists.
        """
        name = _require_feature_name(feature.name)
        if name in self._features:
            raise DuplicateFeatureError(
                f"feature already registered: {name}",
                error_code=_ERROR_DUPLICATE,
                details={"name": name},
            )
        self._features[name] = feature

    def register_many(self, features: Iterable[Feature]) -> None:
        """Register multiple features atomically.

        Either every feature in ``features`` is registered, or the registry
        remains unchanged when any registration would fail.

        Args:
            features: Features to register.

        Raises:
            FeatureRegistrationError: If any feature name is blank.
            DuplicateFeatureError: If any name is already registered or
                duplicated within ``features``.
        """
        pending: dict[str, Feature] = {}
        for feature in features:
            name = _require_feature_name(feature.name)
            if name in self._features or name in pending:
                raise DuplicateFeatureError(
                    f"feature already registered: {name}",
                    error_code=_ERROR_DUPLICATE,
                    details={"name": name},
                )
            pending[name] = feature
        self._features.update(pending)

    def get(self, name: str) -> Feature:
        """Return the registered feature for ``name``.

        Args:
            name: Feature name to look up.

        Returns:
            The registered feature instance.

        Raises:
            UnknownFeatureError: If no feature is registered under ``name``.
        """
        feature = self._features.get(name)
        if feature is None:
            raise UnknownFeatureError(
                f"feature not registered: {name}",
                error_code=_ERROR_UNKNOWN,
                details={"name": name},
            )
        return feature

    def exists(self, name: str) -> bool:
        """Return whether a feature is registered under ``name``.

        Args:
            name: Feature name to check.

        Returns:
            ``True`` when the name is registered, otherwise ``False``.
        """
        return name in self._features

    def remove(self, name: str) -> None:
        """Remove a registered feature by name.

        Args:
            name: Feature name to remove.

        Raises:
            UnknownFeatureError: If no feature is registered under ``name``.
        """
        if name not in self._features:
            raise UnknownFeatureError(
                f"feature not registered: {name}",
                error_code=_ERROR_UNKNOWN,
                details={"name": name},
            )
        del self._features[name]

    def clear(self) -> None:
        """Remove all registered features."""
        self._features.clear()

    def list(self) -> tuple[Feature, ...]:
        """Return registered features sorted alphabetically by name.

        Returns:
            A new tuple of registered feature instances.
        """
        return tuple(self._features[name] for name in sorted(self._features))

    def names(self) -> tuple[str, ...]:
        """Return registered feature names in alphabetical order.

        Returns:
            A new tuple of feature names.
        """
        return tuple(sorted(self._features))

    def metadata(self) -> tuple[FeatureMetadata, ...]:
        """Return metadata snapshots for all registered features.

        Metadata is projected from each feature's public attributes.
        Fields not present on ``Feature`` use safe defaults.

        Returns:
            A new tuple of ``FeatureMetadata`` objects sorted alphabetically
            by feature name.
        """
        return tuple(_to_feature_metadata(feature) for feature in self.list())


def _require_feature_name(name: object) -> str:
    """Validate and return a non-blank feature name.

    Args:
        name: Candidate feature name.

    Returns:
        The validated feature name.

    Raises:
        FeatureRegistrationError: If ``name`` is not a non-blank string.
    """
    if not isinstance(name, str) or name.strip() == "":
        raise FeatureRegistrationError(
            "feature name must be a non-blank string",
            error_code=_ERROR_NAME_BLANK,
            details={"name": name},
        )
    return name


def _to_feature_metadata(feature: Feature) -> FeatureMetadata:
    """Project a registered feature into immutable metadata."""
    return FeatureMetadata(
        name=feature.name,
        version=feature.version,
        category=feature.category,
        description=feature.description,
        author=_DEFAULT_AUTHOR,
        lookback=feature.lookback,
        warmup_rows=feature.warmup_rows,
        required_columns=tuple(feature.required_columns),
        produced_columns=tuple(feature.produced_columns),
        dependencies=tuple(feature.dependencies),
    )
