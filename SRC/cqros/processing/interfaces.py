"""CQROS Data Processing Framework public interfaces.

Purpose:
    Define structural contracts for processing steps so every processing
    implementation shares one public surface.

Responsibilities:
    - Expose ``ProcessingStep`` as the shared processing contract
    - Remain free of calculation, storage, validation, registry, and
      orchestration logic

Dependencies:
    ``polars`` and the Python standard library.

Public API:
    ``ProcessingStep``
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import polars as pl

__all__ = [
    "ProcessingStep",
]


@runtime_checkable
class ProcessingStep(Protocol):
    """Structural contract for a single deterministic processing transform.

    Implementations must be immutable and deterministic: identical inputs
    must always produce identical outputs. ``process`` must never mutate
    the caller-supplied DataFrame; it returns a new frame.

    Processing steps transform raw market datasets into cleaned,
    research-ready frames. They are not feature calculations, storage
    adapters, or research workflows.

    Attributes:
        name: Stable processing-step identifier used by registries and
            pipelines.
        version: Semantic version of the step formula and parameters.
        description: Human-readable summary of what the step does.
    """

    @property
    def name(self) -> str:
        """Stable processing-step identifier used by registries and pipelines."""
        ...

    @property
    def version(self) -> str:
        """Semantic version of the step formula and parameters."""
        ...

    @property
    def description(self) -> str:
        """Human-readable summary of what the step does."""
        ...

    def process(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Transform ``frame`` without mutating it.

        Args:
            frame: Input market DataFrame. Must not be mutated.

        Returns:
            A new DataFrame produced by this processing step.
        """
        ...
