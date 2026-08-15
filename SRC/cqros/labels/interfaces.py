"""CQROS Label Engine public interfaces.

Purpose:
    Define structural contracts for label pipeline execution so every Label
    Engine implementation shares one public surface.

Responsibilities:
    - Expose ``LabelPipeline`` as the shared Label Engine orchestration
      contract
    - Remain free of calculation, storage, validation, and concrete
      orchestration logic

Dependencies:
    ``polars`` and the Python standard library.

Public API:
    ``LabelPipeline``
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import polars as pl

__all__ = [
    "LabelPipeline",
]


@runtime_checkable
class LabelPipeline(Protocol):
    """Structural contract for merged label generation.

    Pipelines compute regression and classification labels from processed
    OHLCV input, finalize outputs to the merged label schema, persist the
    partition, and return a new frame. Implementations must not mutate the
    caller-supplied input.
    """

    def run(
        self,
        frame: pl.DataFrame,
        *,
        exchange: str,
        market: str,
        symbol: str,
        timeframe: str,
        year: int,
    ) -> pl.DataFrame:
        """Generate, finalize, and persist labels against ``frame``.

        Args:
            frame: Processed OHLCV DataFrame. Must not be mutated.
            exchange: Exchange identifier for the persisted partition.
            market: Market segment for the persisted partition.
            symbol: Tradeable symbol for the persisted partition.
            timeframe: Label bar interval for the persisted partition.
            year: Calendar year of the persisted partition.

        Returns:
            A new DataFrame containing the finalized merged label matrix.
        """
        ...
