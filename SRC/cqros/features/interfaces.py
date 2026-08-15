"""CQROS Feature Engine public interfaces.

Purpose:
    Define structural contracts for feature computation, validation, and
    pipeline execution so every feature implementation shares one public
    surface.

Responsibilities:
    - Expose ``Feature``, ``FeatureValidator``, and ``FeaturePipeline`` as
      the shared Feature Engine contracts
    - Remain free of calculation, storage, validation, registry, and
      concrete orchestration logic

Dependencies:
    ``polars`` and the Python standard library.

Public API:
    ``Feature``, ``FeatureValidator``, ``FeaturePipeline``
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import polars as pl

__all__ = [
    "Feature",
    "FeatureValidator",
    "FeaturePipeline",
]


@runtime_checkable
class Feature(Protocol):
    """Structural contract for a single deterministic feature transform.

    Implementations must be immutable and deterministic: identical inputs
    must always produce identical outputs. ``transform`` must never mutate
    the caller-supplied DataFrame; it returns a new frame that includes the
    produced columns.

    Attributes:
        name: Stable feature identifier used by registries and pipelines.
        version: Semantic version of the feature formula and parameters.
        category: Feature group classification (for example ``trend`` or
            ``momentum``).
        description: Human-readable summary of what the feature computes.
        required_columns: Input column names that must be present before
            ``transform`` runs.
        produced_columns: Output column names added by ``transform``.
        lookback: Minimum historical row count required for a fully defined
            warm-up window.
        warmup_rows: Leading undefined rows before the first fully defined
            value. Defaults to rolling-window semantics
            ``max(0, lookback - 1)`` unless a feature overrides it.
        dependencies: Names of other features that must be computed first.
    """

    @property
    def name(self) -> str:
        """Stable feature identifier used by registries and pipelines."""
        ...

    @property
    def version(self) -> str:
        """Semantic version of the feature formula and parameters."""
        ...

    @property
    def category(self) -> str:
        """Feature group classification."""
        ...

    @property
    def description(self) -> str:
        """Human-readable summary of what the feature computes."""
        ...

    @property
    def required_columns(self) -> Sequence[str]:
        """Input column names required before ``transform`` runs."""
        ...

    @property
    def produced_columns(self) -> Sequence[str]:
        """Output column names added by ``transform``."""
        ...

    @property
    def lookback(self) -> int:
        """Minimum historical rows required for a fully defined warm-up."""
        ...

    @property
    def warmup_rows(self) -> int:
        """Leading undefined rows before the first fully defined value."""
        ...

    @property
    def dependencies(self) -> Sequence[str]:
        """Names of other features that must be computed first."""
        ...

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Compute feature columns from ``frame`` without mutating it.

        Args:
            frame: Input market or research DataFrame. Must not be mutated.

        Returns:
            A new DataFrame containing the original columns plus
            ``produced_columns``.
        """
        ...


@runtime_checkable
class FeatureValidator(Protocol):
    """Structural contract for validating feature transform outputs.

    Validators inspect a DataFrame against a ``Feature`` contract and fail
    fast when the result violates schema, dtype, or quality expectations.
    """

    def validate(self, frame: pl.DataFrame, feature: Feature) -> None:
        """Validate ``frame`` against the contract of ``feature``.

        Args:
            frame: DataFrame produced by or prepared for ``feature``.
            feature: Feature whose schema and output constraints apply.

        Raises:
            Exception: When validation fails. Concrete CQROS validators
                should raise a project feature or validation error type.
        """
        ...


@runtime_checkable
class FeaturePipeline(Protocol):
    """Structural contract for ordered multi-feature execution.

    Pipelines apply a named sequence of features to an input DataFrame,
    finalize outputs to the merged feature schema, persist the partition, and
    return a new frame. Implementations own dependency ordering and must not
    mutate the caller-supplied input.
    """

    def run(
        self,
        frame: pl.DataFrame,
        features: Sequence[str],
        *,
        exchange: str,
        market: str,
        symbol: str,
        timeframe: str,
        year: int,
    ) -> pl.DataFrame:
        """Execute, finalize, and persist named features against ``frame``.

        Args:
            frame: Input DataFrame supplied to the feature sequence. Must not
                be mutated.
            features: Ordered feature names to apply.
            exchange: Exchange identifier for the persisted partition.
            market: Market segment for the persisted partition.
            symbol: Tradeable symbol for the persisted partition.
            timeframe: Feature bar interval for the persisted partition.
            year: Calendar year of the persisted partition.

        Returns:
            A new DataFrame containing the finalized merged feature matrix.
        """
        ...
