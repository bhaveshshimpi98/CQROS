"""CQROS Feature Engine exception hierarchy.

Purpose:
    Provide feature-specific exception types used by registry, pipeline,
    validation, feature store, dataset building, and research workflows.

Responsibilities:
    - Re-export the shared ``FeatureError`` root from the core taxonomy
    - Expose specialized feature failures for registration, dependencies,
      validation, execution, metadata, configuration, and storage
    - Remain free of logging, validation, and business logic

Dependencies:
    ``cqros.core.exceptions.FeatureError``.

Public API:
    The exception types listed in ``__all__``.
"""

from __future__ import annotations

from cqros.core.exceptions import FeatureError

__all__ = [
    "FeatureError",
    "FeatureRegistrationError",
    "DuplicateFeatureError",
    "UnknownFeatureError",
    "FeatureDependencyError",
    "FeatureValidationError",
    "FeatureExecutionError",
    "FeatureMetadataError",
    "FeatureConfigurationError",
    "FeatureStoreError",
]


class FeatureRegistrationError(FeatureError):
    """Raised when registering a feature with the Feature Engine fails."""

    __slots__ = ()


class DuplicateFeatureError(FeatureError):
    """Raised when a feature name or version is already registered."""

    __slots__ = ()


class UnknownFeatureError(FeatureError):
    """Raised when a requested feature cannot be found."""

    __slots__ = ()


class FeatureDependencyError(FeatureError):
    """Raised when feature dependency resolution or ordering fails."""

    __slots__ = ()


class FeatureValidationError(FeatureError):
    """Raised when feature inputs, outputs, or contracts fail validation."""

    __slots__ = ()


class FeatureExecutionError(FeatureError):
    """Raised when feature transform execution fails."""

    __slots__ = ()


class FeatureMetadataError(FeatureError):
    """Raised when feature metadata is missing, inconsistent, or unusable."""

    __slots__ = ()


class FeatureConfigurationError(FeatureError):
    """Raised when feature configuration is invalid or incomplete."""

    __slots__ = ()


class FeatureStoreError(FeatureError):
    """Raised when feature store persistence or retrieval fails."""

    __slots__ = ()
