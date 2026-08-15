"""CQROS Feature Engine metadata models.

Purpose:
    Provide immutable value objects that describe features, feature
    categories, feature groups, and feature manifests—not feature values.

Responsibilities:
    - Define metadata structures used by registry, pipeline, lineage,
      reporting, experiment tracking, feature store, and documentation
    - Remain free of execution, validation, serialization, and I/O logic

Dependencies:
    Python standard library and ``cqros.core.types``.

Public API:
    ``FeatureMetadata``, ``FeatureCategory``, ``FeatureGroup``,
    ``FeatureManifest``

Notes:
    Collections that form part of an immutable value object use ``tuple``
    rather than ``list``. Arbitrary extension fields use
    ``Mapping[str, object]`` so future feature workflows can attach
    structured context without changing the core models.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from cqros.core.types import Timestamp

__all__ = [
    "FeatureMetadata",
    "FeatureCategory",
    "FeatureGroup",
    "FeatureManifest",
]


def _empty_extension_data() -> dict[str, object]:
    """Return a new empty extension-data mapping."""
    return {}


def _freeze_str_tuple(value: Sequence[str]) -> tuple[str, ...]:
    """Return an immutable tuple copy of a string sequence."""
    return tuple(value)


@dataclass(frozen=True, slots=True)
class FeatureMetadata:
    """Immutable metadata describing a single CQROS feature.

    Captures identity, classification, column contracts, dependency names,
    and lifecycle flags for one feature definition. This model does not
    compute features, resolve dependencies, or validate schemas.

    Attributes:
        name: Stable feature identifier.
        version: Semantic version of the feature formula and parameters.
        category: Feature group classification (for example ``trend``).
        description: Human-readable summary of what the feature computes.
        author: Researcher or system that owns the feature definition.
        tags: Immutable classification tags.
        required_columns: Input column names required before transform.
        produced_columns: Output column names produced by transform.
        dependencies: Names of other features that must be computed first.
        lookback: Minimum historical row count for a fully defined window.
        warmup_rows: Leading undefined rows before the first fully defined
            value. When set to a negative sentinel (default ``-1``), resolves
            to ``max(0, lookback - 1)`` during initialization.
        created_at: Creation timestamp (UTC), if recorded.
        deprecated: Whether the feature is deprecated.
        experimental: Whether the feature is experimental.
        extension_data: Additional structured feature context.
    """

    name: str
    version: str
    category: str
    description: str
    author: str
    lookback: int
    warmup_rows: int = -1
    tags: tuple[str, ...] = ()
    required_columns: tuple[str, ...] = ()
    produced_columns: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    created_at: Timestamp | None = None
    deprecated: bool = False
    experimental: bool = False
    extension_data: Mapping[str, object] = field(
        default_factory=_empty_extension_data,
        hash=False,
    )

    def __post_init__(self) -> None:
        """Freeze collection and mapping fields into immutable containers."""
        if self.warmup_rows < 0:
            object.__setattr__(self, "warmup_rows", max(0, self.lookback - 1))
        object.__setattr__(self, "tags", _freeze_str_tuple(self.tags))
        object.__setattr__(
            self,
            "required_columns",
            _freeze_str_tuple(self.required_columns),
        )
        object.__setattr__(
            self,
            "produced_columns",
            _freeze_str_tuple(self.produced_columns),
        )
        object.__setattr__(self, "dependencies", _freeze_str_tuple(self.dependencies))
        object.__setattr__(
            self,
            "extension_data",
            MappingProxyType(dict(self.extension_data)),
        )


@dataclass(frozen=True, slots=True)
class FeatureCategory:
    """Immutable metadata describing a feature category.

    Attributes:
        name: Stable category identifier (for example ``momentum``).
        description: Human-readable category summary.
        display_name: Presentation label for documentation and reports.
    """

    name: str
    description: str
    display_name: str


@dataclass(frozen=True, slots=True)
class FeatureGroup:
    """Immutable logical grouping of feature names.

    Attributes:
        name: Stable group identifier.
        description: Human-readable group summary.
        features: Immutable sequence of feature names in the group.
    """

    name: str
    description: str
    features: tuple[str, ...]

    def __post_init__(self) -> None:
        """Freeze the feature-name collection into an immutable tuple."""
        object.__setattr__(self, "features", _freeze_str_tuple(self.features))


@dataclass(frozen=True, slots=True)
class FeatureManifest:
    """Immutable collection of feature metadata records.

    A manifest is a metadata snapshot of registered features. It does not
    perform registry lookups, dependency resolution, or persistence.

    Attributes:
        version: Manifest version string.
        created_at: Manifest creation timestamp (UTC).
        features: Immutable sequence of feature metadata records.
    """

    version: str
    created_at: Timestamp
    features: tuple[FeatureMetadata, ...]

    def __post_init__(self) -> None:
        """Freeze the feature metadata collection into an immutable tuple."""
        object.__setattr__(self, "features", tuple(self.features))
