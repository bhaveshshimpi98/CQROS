"""CQROS Feature Engine pipeline.

Purpose:
    Orchestrate deterministic execution of registered features over a Polars
    DataFrame, including recursive dependency resolution, merged-schema
    finalization, warm-up trimming, and persistence through
    ``FeatureRepository``.

Responsibilities:
    - Resolve feature dependencies into a deterministic topological order
    - Execute each required feature exactly once
    - Finalize outputs against the canonical merged feature schema
    - Trim warm-up rows using the maximum executed-feature ``warmup_rows``
    - Persist the merged partition through an injected ``FeatureRepository``
    - Preserve input DataFrame immutability
    - Remain free of registration, caching, multiprocessing, and dataset
      verification logic

Dependencies:
    ``polars``, ``cqros.core.types``, ``cqros.features.exceptions``,
    ``cqros.features.interfaces``, ``cqros.features.metadata``,
    ``cqros.features.registry``, ``cqros.features.schema``, and
    ``cqros.storage.feature_repository``.

Public API:
    ``FeaturePipeline``
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Final

import polars as pl

from cqros.core.types import Exchange, Market, Symbol, Timeframe
from cqros.features.exceptions import (
    FeatureDependencyError,
    FeatureExecutionError,
    FeatureValidationError,
)
from cqros.features.interfaces import Feature
from cqros.features.metadata import FeatureMetadata
from cqros.features.registry import FeatureRegistry
from cqros.features.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    REQUIRED_COLUMNS,
)
from cqros.storage.feature_repository import FeatureRepository

__all__ = ["FeaturePipeline"]

_ERROR_NAME_BLANK: Final[str] = "FEATURE-PIPE-001"
_ERROR_CYCLE: Final[str] = "FEATURE-PIPE-002"
_ERROR_EXECUTION: Final[str] = "FEATURE-PIPE-003"
_ERROR_MISSING_COLUMNS: Final[str] = "FEATURE-PIPE-004"

_logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Deterministic orchestrator for registered feature transforms.

    The pipeline resolves dependencies, executes each feature once, finalizes
    the result to the canonical merged feature schema, trims warm-up rows, and
    persists the partition through ``FeatureRepository``. The caller-supplied
    input frame is never mutated.

    Args:
        registry: Feature catalog used to resolve feature names.
        repository: Persistence facade for merged feature partitions.
        logger: Optional logger instance. Defaults to the module logger.
    """

    __slots__ = ("_logger", "_registry", "_repository")

    _registry: FeatureRegistry
    _repository: FeatureRepository
    _logger: logging.Logger

    def __init__(
        self,
        registry: FeatureRegistry,
        repository: FeatureRepository,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the pipeline with registry and feature repository.

        Args:
            registry: Registry providing feature lookup by name.
            repository: Repository used to persist finalized partitions.
            logger: Optional logger instance.
        """
        self._registry = registry
        self._repository = repository
        self._logger = logger if logger is not None else _logger

    def run(
        self,
        frame: pl.DataFrame,
        features: Sequence[str],
        *,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> pl.DataFrame:
        """Execute, finalize, trim, and persist named features.

        Dependencies are resolved recursively and executed before dependents.
        Duplicate names in ``features`` are executed once. After execution the
        frame is checked against ``REQUIRED_COLUMNS``, reordered to
        ``CANONICAL_COLUMN_ORDER``, cast to ``COLUMN_DTYPES``, trimmed by the
        maximum executed-feature ``warmup_rows``, and saved through
        ``FeatureRepository``. The original ``frame`` is never mutated.

        Args:
            frame: Input market or research DataFrame.
            features: Feature names to apply, in caller preference order.
            exchange: Exchange identifier for the persisted partition.
            market: Market segment for the persisted partition.
            symbol: Tradeable symbol for the persisted partition.
            timeframe: Feature bar interval for the persisted partition.
            year: Calendar year of the persisted partition.

        Returns:
            A new DataFrame containing the finalized merged feature matrix.

        Raises:
            FeatureValidationError: If any requested feature name is blank or
                a required merged-schema column is missing.
            UnknownFeatureError: If a requested feature or dependency is
                missing from the registry.
            FeatureDependencyError: If a dependency cycle is detected or a
                dependency name is blank.
            FeatureExecutionError: If a feature ``transform`` raises.
        """
        order = self._topological_order(features)
        executed = self._execute(frame, order)
        metadata = self._executed_metadata(order)
        finalized = self._finalize(executed, metadata)
        self._logger.debug(
            "Persisting merged feature partition",
            extra={
                "exchange": exchange,
                "market": market,
                "symbol": symbol,
                "timeframe": timeframe,
                "year": year,
                "rows": finalized.height,
                "columns": finalized.width,
                "features": order,
            },
        )
        self._repository.save(
            finalized,
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        )
        self._logger.info(
            "Persisted merged feature partition",
            extra={
                "exchange": exchange,
                "market": market,
                "symbol": symbol,
                "timeframe": timeframe,
                "year": year,
                "rows": finalized.height,
                "columns": finalized.width,
                "features": order,
            },
        )
        return finalized

    def _topological_order(self, features: Sequence[str]) -> tuple[str, ...]:
        """Return a deterministic execution order for ``features``.

        Args:
            features: Requested feature names.

        Returns:
            Feature names in dependency-respecting order. Each name appears
            at most once.

        Raises:
            FeatureValidationError: If a requested name is blank.
            UnknownFeatureError: If a feature or dependency is unregistered.
            FeatureDependencyError: If a cycle or blank dependency is found.
        """
        requested = _dedupe_preserve_order(features)
        ordered: list[str] = []
        visited: set[str] = set()
        visiting: list[str] = []

        for name in requested:
            self._resolve_dependencies(name, visited=visited, visiting=visiting, ordered=ordered)
        return tuple(ordered)

    def _resolve_dependencies(
        self,
        name: str,
        *,
        visited: set[str],
        visiting: list[str],
        ordered: list[str],
    ) -> None:
        """Depth-first resolve ``name`` and append it after its dependencies.

        Args:
            name: Feature name to resolve.
            visited: Names whose full dependency subtree is complete.
            visiting: Ordered recursion stack used for cycle detection.
            ordered: Output accumulator receiving post-order names.

        Raises:
            UnknownFeatureError: If ``name`` is not registered.
            FeatureDependencyError: If a cycle or blank dependency is found.
        """
        if name in visited:
            return
        if name in visiting:
            cycle = (*visiting, name)
            raise FeatureDependencyError(
                f"circular feature dependency detected: {' -> '.join(cycle)}",
                error_code=_ERROR_CYCLE,
                details={"feature": name, "cycle": cycle},
            )

        feature = self._registry.get(name)
        visiting.append(name)
        for dependency in feature.dependencies:
            _require_dependency_name(dependency, dependent=name)
            self._resolve_dependencies(
                dependency,
                visited=visited,
                visiting=visiting,
                ordered=ordered,
            )
        visiting.pop()
        visited.add(name)
        ordered.append(name)

    def _execute(self, frame: pl.DataFrame, order: Sequence[str]) -> pl.DataFrame:
        """Execute features in ``order`` without mutating ``frame``.

        Args:
            frame: Caller-supplied input DataFrame.
            order: Deterministic feature execution order.

        Returns:
            A new DataFrame produced by sequential transforms.

        Raises:
            FeatureExecutionError: If any feature ``transform`` fails.
            UnknownFeatureError: If a name in ``order`` is no longer
                registered.
        """
        current = frame.clone()
        for name in order:
            feature = self._registry.get(name)
            current = self._transform_feature(feature, current)
        return current

    def _transform_feature(self, feature: Feature, frame: pl.DataFrame) -> pl.DataFrame:
        """Execute one feature transform and wrap unexpected failures.

        Args:
            feature: Feature to execute.
            frame: Current pipeline DataFrame.

        Returns:
            DataFrame returned by ``feature.transform``.

        Raises:
            FeatureExecutionError: If ``transform`` raises any exception.
        """
        try:
            return feature.transform(frame)
        except Exception as exc:
            raise FeatureExecutionError(
                f"feature transform failed: {feature.name}",
                error_code=_ERROR_EXECUTION,
                details={
                    "feature": feature.name,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            ) from exc

    def _executed_metadata(self, order: Sequence[str]) -> tuple[FeatureMetadata, ...]:
        """Return ``FeatureMetadata`` snapshots for executed features.

        Metadata is projected from the registry in execution order so warm-up
        aggregation remains deterministic.

        Args:
            order: Deterministic feature execution order.

        Returns:
            Immutable metadata records for each executed feature.
        """
        by_name = {item.name: item for item in self._registry.metadata()}
        return tuple(by_name[name] for name in order)

    def _finalize(
        self,
        frame: pl.DataFrame,
        metadata: Sequence[FeatureMetadata],
    ) -> pl.DataFrame:
        """Apply merged-schema checks, ordering, casting, and warm-up trim.

        Args:
            frame: Frame produced by feature execution.
            metadata: Metadata for features included in the execution plan.

        Returns:
            Finalized merged feature DataFrame.

        Raises:
            FeatureValidationError: If any required schema column is missing.
        """
        _require_schema_columns(frame)
        ordered = frame.select(list(CANONICAL_COLUMN_ORDER))
        casted = ordered.cast(COLUMN_DTYPES)
        warmup_rows = _maximum_warmup_rows(metadata)
        if warmup_rows == 0:
            return casted
        return casted.slice(warmup_rows)


def _maximum_warmup_rows(metadata: Sequence[FeatureMetadata]) -> int:
    """Return the maximum explicit warm-up row count across executed features."""
    if not metadata:
        return 0
    return max(item.warmup_rows for item in metadata)


def _require_schema_columns(frame: pl.DataFrame) -> None:
    """Raise when any required merged-schema column is missing.

    Raises:
        FeatureValidationError: If one or more ``REQUIRED_COLUMNS`` are absent.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise FeatureValidationError(
            "merged feature schema is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "missing_columns": tuple(missing),
                "required_columns": REQUIRED_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )


def _dedupe_preserve_order(features: Sequence[str]) -> tuple[str, ...]:
    """Return unique feature names preserving first-seen order.

    Args:
        features: Requested feature names, possibly with duplicates.

    Returns:
        Deduplicated feature names.

    Raises:
        FeatureValidationError: If any name is blank.
    """
    ordered: list[str] = []
    seen: set[str] = set()
    for name in features:
        _require_requested_name(name)
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return tuple(ordered)


def _require_requested_name(name: object) -> str:
    """Validate a caller-requested feature name.

    Raises:
        FeatureValidationError: If ``name`` is not a non-blank string.
    """
    if not isinstance(name, str) or name.strip() == "":
        raise FeatureValidationError(
            "feature name must be a non-blank string",
            error_code=_ERROR_NAME_BLANK,
            details={"name": name},
        )
    return name


def _require_dependency_name(name: object, *, dependent: str) -> str:
    """Validate a dependency name declared by ``dependent``.

    Raises:
        FeatureDependencyError: If ``name`` is not a non-blank string.
    """
    if not isinstance(name, str) or name.strip() == "":
        raise FeatureDependencyError(
            f"feature {dependent!r} declares a blank dependency name",
            error_code=_ERROR_NAME_BLANK,
            details={"feature": dependent, "dependency": name},
        )
    return name
