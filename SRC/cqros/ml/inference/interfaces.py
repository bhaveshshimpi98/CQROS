"""CQROS ML Inference public interfaces.

Purpose:
    Define structural contracts for inference orchestration so every
    prediction pipeline implementation shares one public surface.

Responsibilities:
    - Expose ``PredictionPipeline`` as the shared inference-orchestration contract
    - Remain free of registry lookup, timing, and concrete predict logic

Dependencies:
    ``polars`` and ``cqros.ml.inference.result.PredictionResult``.

Public API:
    ``PredictionPipeline``
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import polars as pl

from cqros.ml.inference.result import PredictionResult

__all__ = [
    "PredictionPipeline",
]


@runtime_checkable
class PredictionPipeline(Protocol):
    """Structural contract for framework-independent model inference.

    Implementations resolve fitted models from a registry, validate inference
    frames against feature-column contracts, invoke ``predict``, and return an
    immutable ``PredictionResult``. They must not train, evaluate, tune, or
    persist models.
    """

    def predict(
        self,
        model_name: str,
        frame: pl.DataFrame,
    ) -> PredictionResult:
        """Generate predictions for ``model_name`` on ``frame``.

        Args:
            model_name: Registry key of the fitted model to use.
            frame: Feature dataset. Must not be mutated.

        Returns:
            Immutable ``PredictionResult`` describing the completed inference.
        """
        ...
