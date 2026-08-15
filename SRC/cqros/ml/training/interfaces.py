"""CQROS ML Training public interfaces.

Purpose:
    Define structural contracts for model-training orchestration so every
    trainer implementation shares one public surface.

Responsibilities:
    - Expose ``ModelTrainer`` as the shared training-orchestration contract
    - Expose ``TrainerResult`` as the immutable training outcome contract
    - Remain free of registry lookup, timing, and concrete fit logic

Dependencies:
    ``polars``, ``cqros.ml.models.interfaces.Model``, and
    ``cqros.ml.models.metadata.ModelMetadata``.

Public API:
    ``ModelTrainer``, ``TrainerResult``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import polars as pl

from cqros.ml.models.interfaces import Model
from cqros.ml.models.metadata import ModelMetadata

__all__ = [
    "ModelTrainer",
    "TrainerResult",
]


@dataclass(frozen=True, slots=True)
class TrainerResult:
    """Immutable outcome of one ``ModelTrainer.train`` orchestration.

    Attributes:
        model_metadata: Metadata for the trained model.
        train_rows: Number of rows in the training frame.
        validation_rows: Number of rows in the validation frame, or ``0``.
        test_rows: Number of rows in the test frame, or ``0`` when unused.
        feature_count: Number of feature columns from model metadata.
        label_column: Label column name from model metadata.
        training_duration: Wall-clock fit duration in seconds.
        fitted_model: Model instance returned by ``fit``.
    """

    model_metadata: ModelMetadata
    train_rows: int
    validation_rows: int
    test_rows: int
    feature_count: int
    label_column: str
    training_duration: float
    fitted_model: Model


@runtime_checkable
class ModelTrainer(Protocol):
    """Structural contract for framework-independent model training.

    Implementations resolve models from a registry, orchestrate ``fit``, and
    return an immutable ``TrainerResult``. They must not split, scale,
    evaluate, tune, or persist models.
    """

    def train(
        self,
        model_name: str,
        train_frame: pl.DataFrame,
        validation_frame: pl.DataFrame | None = None,
    ) -> TrainerResult:
        """Train the registered model named ``model_name``.

        Args:
            model_name: Registry key of the model to train.
            train_frame: Training dataset. Must not be mutated.
            validation_frame: Optional validation dataset. Must not be mutated.

        Returns:
            Immutable ``TrainerResult`` describing the completed fit.
        """
        ...
