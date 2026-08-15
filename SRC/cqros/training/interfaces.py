"""CQROS Training package public interfaces.

Purpose:
    Define structural contracts for training pipeline execution so every
    Training package implementation shares one public surface.

Responsibilities:
    - Expose ``TrainingPipeline`` as the shared Training package orchestration
      contract
    - Remain free of joins, storage, validation, and concrete orchestration
      logic

Dependencies:
    ``polars`` and the Python standard library.

Public API:
    ``TrainingPipeline``
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import polars as pl

__all__ = [
    "TrainingPipeline",
]


@runtime_checkable
class TrainingPipeline(Protocol):
    """Structural contract for merged training dataset assembly.

    Pipelines join feature and label frames on the shared primary key,
    finalize outputs to the merged training schema, persist the partition,
    and return a new frame. Implementations must not mutate the
    caller-supplied inputs.
    """

    def run(
        self,
        features: pl.DataFrame,
        labels: pl.DataFrame,
        *,
        exchange: str,
        market: str,
        symbol: str,
        timeframe: str,
        year: int,
    ) -> pl.DataFrame:
        """Join, finalize, and persist training data from ``features`` and ``labels``.

        Args:
            features: Feature DataFrame. Must not be mutated.
            labels: Label DataFrame. Must not be mutated.
            exchange: Exchange identifier for the persisted partition.
            market: Market segment for the persisted partition.
            symbol: Tradeable symbol for the persisted partition.
            timeframe: Training bar interval for the persisted partition.
            year: Calendar year of the persisted partition.

        Returns:
            A new DataFrame containing the finalized merged training matrix.
        """
        ...
