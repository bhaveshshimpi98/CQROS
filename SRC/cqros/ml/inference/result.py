"""CQROS ML Inference result models.

Purpose:
    Provide immutable value objects that describe inference outcomes without
    coupling to training, evaluation, or framework-specific model internals.

Responsibilities:
    - Define ``PredictionResult`` as the inference outcome contract
    - Remain free of prediction execution and registry lookup logic

Dependencies:
    ``polars`` and ``cqros.ml.models.metadata.ModelMetadata``.

Public API:
    ``PredictionResult``
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from cqros.ml.models.metadata import ModelMetadata

__all__ = [
    "PredictionResult",
]


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """Immutable outcome of one ``PredictionPipeline.predict`` orchestration.

    Attributes:
        model_metadata: Metadata for the model that produced the predictions.
        prediction_count: Number of prediction rows generated.
        prediction_time: Wall-clock inference duration in seconds.
        predictions: Prediction series preserving input row order.
    """

    model_metadata: ModelMetadata
    prediction_count: int
    prediction_time: float
    predictions: pl.Series
