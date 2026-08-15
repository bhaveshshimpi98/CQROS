"""CQROS ML Model public interfaces.

Purpose:
    Define structural contracts for machine-learning models so every model
    implementation shares one public surface.

Responsibilities:
    - Expose ``Model`` as the shared model contract
    - Remain free of training, inference, serialization, and concrete
      framework logic

Dependencies:
    ``polars``, ``pathlib``, and ``cqros.ml.models.metadata``.

Public API:
    ``Model``
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Self, runtime_checkable

import polars as pl

from cqros.ml.models.metadata import ModelMetadata

__all__ = [
    "Model",
]


@runtime_checkable
class Model(Protocol):
    """Structural contract for a CQROS machine-learning model.

    Implementations fit on training frames, produce predictions, and persist
    artifacts through ``save`` / ``load``. Implementations must not mutate
    caller-supplied DataFrames.
    """

    def fit(self, frame: pl.DataFrame) -> Self:
        """Fit the model on ``frame`` and return ``self``.

        Args:
            frame: Training dataset. Must not be mutated.

        Returns:
            The fitted model instance.
        """
        ...

    def predict(self, frame: pl.DataFrame) -> pl.Series:
        """Generate predictions for ``frame``.

        Args:
            frame: Feature dataset. Must not be mutated.

        Returns:
            A new prediction series.
        """
        ...

    def save(self, path: Path | str) -> None:
        """Persist model artifacts to ``path``.

        Args:
            path: Destination path for the serialized model.
        """
        ...

    def load(self, path: Path | str) -> Self:
        """Load model artifacts from ``path`` and return the loaded model.

        Args:
            path: Source path of the serialized model.

        Returns:
            The loaded model instance.
        """
        ...

    def metadata(self) -> ModelMetadata:
        """Return immutable model metadata."""
        ...
