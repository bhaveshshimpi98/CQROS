"""CQROS Factor Research Engine public interfaces.

Purpose:
    Define structural contracts for factor computation, validation, and
    pipeline execution so every factor implementation shares one public
    surface.

Responsibilities:
    - Expose ``Factor``, ``FactorValidator``, and ``FactorPipeline`` as
      the shared Factor Research Engine contracts
    - Remain free of calculation, storage, validation, registry, and
      research logic

Dependencies:
    ``polars`` and the Python standard library.

Public API:
    ``Factor``, ``FactorValidator``, ``FactorPipeline``

Notes:
    Factors are research objects that generate cross-sectional alpha
    signals. They are not trading strategies, models, or indicators.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import polars as pl

from cqros.factors.schema import FactorStatus

__all__ = [
    "Factor",
    "FactorPipeline",
    "FactorValidator",
]


@runtime_checkable
class Factor(Protocol):
    """Structural contract for a single deterministic factor transform.

    Implementations must be immutable and deterministic: identical inputs
    must always produce identical outputs. ``compute`` must never mutate
    the caller-supplied DataFrame; it returns a new frame that includes the
    produced columns.

    Factors consume research features and emit cross-sectional alpha signal
    columns. They are not trading strategies, models, or indicators.

    Attributes:
        name: Stable factor identifier used by registries and pipelines.
        version: Semantic version of the factor formula and parameters.
        description: Human-readable summary of what the factor computes.
        category: Factor group classification (for example ``momentum`` or
            ``value``).
        required_features: Feature names that must be present before
            ``compute`` runs.
        produced_columns: Output column names added by ``compute``.
        lookback: Minimum historical row count required for a fully defined
            warm-up window.
        factor_group: Research group classification for ``FACTOR_SCHEMA``.
        prediction_horizon: Forward horizon associated with the factor.
        enabled: Whether the factor is enabled for research use.
        status: Lifecycle status stored as ``FactorStatus``.
    """

    @property
    def name(self) -> str:
        """Stable factor identifier used by registries and pipelines."""
        ...

    @property
    def version(self) -> str:
        """Semantic version of the factor formula and parameters."""
        ...

    @property
    def description(self) -> str:
        """Human-readable summary of what the factor computes."""
        ...

    @property
    def category(self) -> str:
        """Factor category classification."""
        ...

    @property
    def required_features(self) -> Sequence[str]:
        """Feature names required before ``compute`` runs."""
        ...

    @property
    def produced_columns(self) -> Sequence[str]:
        """Output column names added by ``compute``."""
        ...

    @property
    def lookback(self) -> int:
        """Minimum historical rows required for a fully defined warm-up."""
        ...

    @property
    def factor_group(self) -> str:
        """Research group classification for ``FACTOR_SCHEMA``."""
        ...

    @property
    def prediction_horizon(self) -> int:
        """Forward horizon associated with the factor."""
        ...

    @property
    def enabled(self) -> bool:
        """Whether the factor is enabled for research use."""
        ...

    @property
    def status(self) -> FactorStatus:
        """Lifecycle status stored as ``FactorStatus``."""
        ...

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Compute factor signal columns from ``frame`` without mutating it.

        Args:
            frame: Input research DataFrame containing required features.
                Must not be mutated.

        Returns:
            A new DataFrame containing the original columns plus
            ``produced_columns``.
        """
        ...


@runtime_checkable
class FactorValidator(Protocol):
    """Structural contract for validating factor compute outputs.

    Validators inspect a DataFrame against a ``Factor`` contract and fail
    fast when the result violates schema, dtype, or quality expectations.
    """

    def validate(self, frame: pl.DataFrame, factor: Factor) -> None:
        """Validate ``frame`` against the contract of ``factor``.

        Args:
            frame: DataFrame produced by or prepared for ``factor``.
            factor: Factor whose schema and output constraints apply.

        Raises:
            Exception: When validation fails. Concrete CQROS validators
                should raise a project factor or validation error type.
        """
        ...


@runtime_checkable
class FactorPipeline(Protocol):
    """Structural contract for production-catalog factor execution.

    Pipelines execute every registered factor independently against the same
    input DataFrame, merge produced columns into one wide matrix, and return
    a new frame. Implementations must not mutate the caller-supplied input.
    """

    def run(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Execute registered factors against ``frame`` and return the matrix.

        Args:
            frame: Input training DataFrame. Must not be mutated.

        Returns:
            A new wide DataFrame with primary-key columns and every generated
            factor column.
        """
        ...
