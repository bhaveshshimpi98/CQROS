"""CQROS ML Model abstract base.

Purpose:
    Provide a training-agnostic abstract base class that every concrete model
    inherits from, eliminating metadata boilerplate while remaining free of
    framework-specific training and inference logic.

Responsibilities:
    - Hold immutable model metadata
    - Validate constructor metadata
    - Define abstract ``fit``, ``predict``, ``save``, and ``load`` contracts
    - Expose shared validation helpers for subclasses
    - Remain free of training algorithms and serialization implementations

Dependencies:
    ``polars``, ``pathlib``, ``cqros.ml.models.exceptions``,
    ``cqros.ml.models.metadata``, and structural compatibility with
    ``cqros.ml.models.interfaces.Model``.

Public API:
    ``BaseModel``
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Self, cast

import polars as pl

from cqros.ml.models.exceptions import ModelValidationError
from cqros.ml.models.metadata import ModelMetadata

__all__ = ["BaseModel"]

_ERROR_METADATA_TYPE: Final[str] = "ML-MODEL-BASE-001"
_ERROR_FRAME_TYPE: Final[str] = "ML-MODEL-BASE-002"
_ERROR_FRAME_EMPTY: Final[str] = "ML-MODEL-BASE-003"
_ERROR_MISSING_FEATURES: Final[str] = "ML-MODEL-BASE-004"
_ERROR_MISSING_LABEL: Final[str] = "ML-MODEL-BASE-005"
_ERROR_PATH_TYPE: Final[str] = "ML-MODEL-BASE-006"


@dataclass(frozen=True, slots=True)
class BaseModel(ABC):
    """Abstract immutable base for every CQROS ML model implementation.

    Concrete models supply ``fit``, ``predict``, ``save``, and ``load``.
    Metadata is provided at construction time and remains fixed for the
    lifetime of the instance. This class intentionally does not train models,
    compute predictions, or serialize framework artifacts.

    Attributes:
        model_metadata: Immutable descriptive metadata for the model.
    """

    model_metadata: ModelMetadata

    def __post_init__(self) -> None:
        """Validate that ``model_metadata`` is a ``ModelMetadata`` instance.

        Raises:
            ModelValidationError: If ``model_metadata`` has an invalid type.
        """
        if not isinstance(cast(object, self.model_metadata), ModelMetadata):
            raise ModelValidationError(
                "model_metadata must be a ModelMetadata instance",
                error_code=_ERROR_METADATA_TYPE,
                details={
                    "parameter": "model_metadata",
                    "value_type": type(self.model_metadata).__name__,
                },
            )

    def metadata(self) -> ModelMetadata:
        """Return the immutable model metadata."""
        return self.model_metadata

    @abstractmethod
    def fit(self, frame: pl.DataFrame) -> Self:
        """Fit the model on ``frame`` and return ``self``.

        Args:
            frame: Training dataset. Must not be mutated.

        Returns:
            The fitted model instance.
        """

    @abstractmethod
    def predict(self, frame: pl.DataFrame) -> pl.Series:
        """Generate predictions for ``frame``.

        Args:
            frame: Feature dataset. Must not be mutated.

        Returns:
            A new prediction series.
        """

    @abstractmethod
    def save(self, path: Path | str) -> None:
        """Persist model artifacts to ``path``.

        Args:
            path: Destination path for the serialized model.
        """

    @abstractmethod
    def load(self, path: Path | str) -> Self:
        """Load model artifacts from ``path`` and return the loaded model.

        Args:
            path: Source path of the serialized model.

        Returns:
            The loaded model instance.
        """

    def _require_dataframe(self, frame: object, *, parameter: str = "frame") -> pl.DataFrame:
        """Validate that ``frame`` is a non-empty Polars DataFrame.

        Args:
            frame: Candidate training or inference frame.
            parameter: Parameter name used in error messages.

        Returns:
            ``frame`` cast as a DataFrame.

        Raises:
            ModelValidationError: If ``frame`` is not a non-empty DataFrame.
        """
        if not isinstance(frame, pl.DataFrame):
            raise ModelValidationError(
                f"{parameter} must be a polars DataFrame",
                error_code=_ERROR_FRAME_TYPE,
                details={"parameter": parameter, "value_type": type(frame).__name__},
            )
        if frame.height == 0:
            raise ModelValidationError(
                f"{parameter} must contain at least one row",
                error_code=_ERROR_FRAME_EMPTY,
                details={"parameter": parameter, "rows": frame.height},
            )
        return frame

    def _require_feature_columns(self, frame: pl.DataFrame) -> None:
        """Validate that all metadata feature columns are present.

        Args:
            frame: Candidate frame.

        Raises:
            ModelValidationError: If one or more feature columns are missing.
        """
        missing = [
            column for column in self.model_metadata.feature_columns if column not in frame.columns
        ]
        if missing:
            raise ModelValidationError(
                "frame is missing required feature columns",
                error_code=_ERROR_MISSING_FEATURES,
                details={
                    "missing_columns": tuple(missing),
                    "required_feature_columns": self.model_metadata.feature_columns,
                    "available_columns": tuple(frame.columns),
                },
            )

    def _require_label_column(self, frame: pl.DataFrame) -> None:
        """Validate that the metadata label column is present.

        Args:
            frame: Candidate training frame.

        Raises:
            ModelValidationError: If the label column is missing.
        """
        label_column = self.model_metadata.label_column
        if label_column not in frame.columns:
            raise ModelValidationError(
                "frame is missing required label column",
                error_code=_ERROR_MISSING_LABEL,
                details={
                    "missing_columns": (label_column,),
                    "required_label_column": label_column,
                    "available_columns": tuple(frame.columns),
                },
            )

    def _require_path(self, path: object, *, parameter: str = "path") -> Path:
        """Validate that ``path`` is a ``Path`` instance.

        Args:
            path: Candidate filesystem path.
            parameter: Parameter name used in error messages.

        Returns:
            ``path`` as a ``Path``.

        Raises:
            ModelValidationError: If ``path`` is not a ``Path``.
        """
        if isinstance(path, Path):
            return path
        if isinstance(path, str):
            return Path(path)
        raise ModelValidationError(
            f"{parameter} must be a Path or str",
            error_code=_ERROR_PATH_TYPE,
            details={"parameter": parameter, "value_type": type(path).__name__},
        )

    def _require_columns(
        self,
        frame: pl.DataFrame,
        columns: Sequence[str],
        *,
        parameter: str = "columns",
    ) -> None:
        """Validate that every column in ``columns`` is present on ``frame``.

        Args:
            frame: Candidate frame.
            columns: Required column names.
            parameter: Parameter name used in error messages.

        Raises:
            ModelValidationError: If one or more columns are missing.
        """
        required = tuple(columns)
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ModelValidationError(
                f"frame is missing required {parameter}",
                error_code=_ERROR_MISSING_FEATURES,
                details={
                    "missing_columns": tuple(missing),
                    "required_columns": required,
                    "available_columns": tuple(frame.columns),
                },
            )

    def __str__(self) -> str:
        """Return a compact human-readable model identity."""
        meta = self.model_metadata
        return f"{meta.name}@{meta.version}"

    def __repr__(self) -> str:
        """Return an unambiguous representation including metadata."""
        return f"{type(self).__name__}(" f"model_metadata={self.model_metadata!r})"
