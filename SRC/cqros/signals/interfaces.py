"""CQROS Signals public interfaces.

Purpose:
    Define structural contracts for signal-generation policies so every
    signal pipeline implementation shares one public surface.

Responsibilities:
    - Expose ``SignalPolicy`` as the shared signal-generation contract
    - Remain free of threshold logic, trading strategy, and persistence

Dependencies:
    ``polars``.

Public API:
    ``SignalPolicy``
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import polars as pl

__all__ = [
    "SignalPolicy",
]


@runtime_checkable
class SignalPolicy(Protocol):
    """Structural contract for converting prediction frames into signal frames.

    Implementations own signal semantics (thresholds, class mapping, and
    related strategy choices). ``SignalPipeline`` delegates generation
    exclusively through this contract. Implementations must return a new
    DataFrame and must not mutate the input prediction frame.
    """

    def generate(self, predictions: pl.DataFrame) -> pl.DataFrame:
        """Convert a canonical prediction DataFrame into a signal DataFrame.

        Args:
            predictions: Canonical prediction dataset. Must not be mutated.

        Returns:
            A new DataFrame containing the columns required by the Signal
            schema contract.
        """
        ...
