"""CQROS Predictions public interfaces.

Purpose:
    Define structural contracts for inference delegation so every prediction
    persistence pipeline shares one injected inference surface.

Responsibilities:
    - Expose ``InferencePipeline`` as the shared inference-delegation contract
    - Remain free of persistence, schema finalization, and trading logic

Dependencies:
    ``polars`` and ``cqros.ml.inference.result.PredictionResult``.

Public API:
    ``InferencePipeline``
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import polars as pl

from cqros.ml.inference.result import PredictionResult

__all__ = [
    "InferencePipeline",
]


@runtime_checkable
class InferencePipeline(Protocol):
    """Structural contract for model inference used by PredictionPipeline.

    Implementations resolve fitted models, validate inference frames against
    feature-column contracts, invoke ``predict``, and return an immutable
    ``PredictionResult``. They must not train, evaluate, tune, persist models,
    or mutate the caller-supplied frame.
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
