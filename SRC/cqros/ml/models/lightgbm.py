"""CQROS LightGBM model implementation.

Purpose:
    Provide the first framework-specific ``BaseModel`` implementation backed
    by the official LightGBM Python API.

Responsibilities:
    - Fit regression and classification boosters from research frames
    - Produce Float64 regression predictions or class-label predictions
    - Persist and restore boosters through LightGBM native serialization
    - Remain free of trainers, evaluators, cross-validation, and HPO

Dependencies:
    ``lightgbm``, ``numpy``, ``polars``, ``pathlib``,
    ``cqros.ml.models.base.BaseModel``, ``cqros.ml.models.exceptions``,
    ``cqros.ml.models.metadata``, and ``cqros.ml.models.interfaces.Model``.

Public API:
    ``LightGBMModel``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Self, cast

import lightgbm as lgb
import numpy as np
import polars as pl

from cqros.ml.models.base import BaseModel
from cqros.ml.models.exceptions import ModelValidationError
from cqros.ml.models.metadata import ModelFramework, ModelTaskType

__all__ = ["LightGBMModel"]

_logger = logging.getLogger(__name__)

_ERROR_FRAMEWORK: Final[str] = "ML-MODEL-LGBM-001"
_ERROR_TASK_TYPE: Final[str] = "ML-MODEL-LGBM-002"
_ERROR_NOT_FITTED: Final[str] = "ML-MODEL-LGBM-003"
_ERROR_XY_MISMATCH: Final[str] = "ML-MODEL-LGBM-004"
_ERROR_EMPTY_FEATURES: Final[str] = "ML-MODEL-LGBM-005"
_ERROR_LABEL_CLASSES: Final[str] = "ML-MODEL-LGBM-006"

_DEFAULT_NUM_BOOST_ROUND: Final[int] = 100
_DEFAULT_VERBOSITY: Final[int] = -1
_BINARY_THRESHOLD: Final[float] = 0.5
_PREDICTION_NAME: Final[str] = "prediction"
_STATE_BOOSTER: Final[str] = "booster"


def _new_state() -> dict[str, object]:
    """Return an empty mutable model-state mapping."""
    return {}


@dataclass(frozen=True, slots=True)
class LightGBMModel(BaseModel):
    """LightGBM-backed CQROS model for regression and classification.

    Training consumes a Polars frame using ``feature_columns`` and
    ``label_column`` from ``model_metadata``. Optional validation data may be
    supplied as a keyword-only frame for LightGBM ``valid_sets``. The trained
    booster is stored internally and never exposes framework-specific APIs
    through the public CQROS surface.

    Attributes:
        model_metadata: Immutable model identity and column contract.
        num_boost_round: Number of boosting iterations used by ``fit``.
    """

    num_boost_round: int = _DEFAULT_NUM_BOOST_ROUND
    _state: dict[str, object] = field(
        default_factory=_new_state,
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        """Validate metadata framework and task type for LightGBM.

        Raises:
            ModelValidationError: If metadata is incompatible with LightGBM.
        """
        BaseModel.__post_init__(self)
        if self.model_metadata.framework is not ModelFramework.LIGHTGBM:
            raise ModelValidationError(
                "LightGBMModel requires framework=ModelFramework.LIGHTGBM",
                error_code=_ERROR_FRAMEWORK,
                details={
                    "framework": str(self.model_metadata.framework),
                    "expected": str(ModelFramework.LIGHTGBM),
                },
            )
        if self.model_metadata.task_type not in (
            ModelTaskType.REGRESSION,
            ModelTaskType.CLASSIFICATION,
        ):
            raise ModelValidationError(
                "task_type must be REGRESSION or CLASSIFICATION",
                error_code=_ERROR_TASK_TYPE,
                details={"task_type": str(self.model_metadata.task_type)},
            )

    def fit(
        self,
        frame: pl.DataFrame,
        *,
        validation_frame: pl.DataFrame | None = None,
    ) -> Self:
        """Fit the LightGBM booster on ``frame``.

        Feature matrix ``X_train`` and label vector ``y_train`` are extracted
        from ``frame`` using metadata column contracts. When
        ``validation_frame`` is provided it is passed to LightGBM as a
        validation set.

        Args:
            frame: Training dataset. Must not be mutated.
            validation_frame: Optional validation dataset with the same
                feature and label columns.

        Returns:
            ``self`` for fluent chaining.

        Raises:
            ModelValidationError: If inputs are empty, missing columns, or
                have mismatched feature/label lengths.
        """
        train_frame = self._require_dataframe(frame, parameter="frame")
        self._require_feature_columns(train_frame)
        self._require_label_column(train_frame)

        x_train, y_train = self._extract_xy(train_frame, parameter="frame")
        train_set = lgb.Dataset(x_train, label=y_train)

        valid_sets: list[lgb.Dataset] = [train_set]
        if validation_frame is not None:
            valid_frame = self._require_dataframe(
                validation_frame,
                parameter="validation_frame",
            )
            self._require_feature_columns(valid_frame)
            self._require_label_column(valid_frame)
            x_valid, y_valid = self._extract_xy(
                valid_frame,
                parameter="validation_frame",
            )
            valid_sets.append(lgb.Dataset(x_valid, label=y_valid, reference=train_set))

        params = self._build_params(y_train)
        booster = _train_booster(
            params,
            train_set,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
        )
        self._state[_STATE_BOOSTER] = booster

        _logger.info(
            "Fitted LightGBM model",
            extra={
                "model": self.model_metadata.name,
                "task_type": str(self.model_metadata.task_type),
                "rows": train_frame.height,
                "num_boost_round": self.num_boost_round,
            },
        )
        return self

    def predict(self, frame: pl.DataFrame) -> pl.Series:
        """Generate predictions for ``frame``.

        Regression returns Float64 scores. Classification returns predicted
        class labels.

        Args:
            frame: Feature dataset. Must not be mutated.

        Returns:
            A new prediction series named ``prediction``.

        Raises:
            ModelValidationError: If the model is not fitted or ``frame`` is
                invalid.
        """
        booster = self._require_booster()
        predict_frame = self._require_dataframe(frame, parameter="frame")
        self._require_feature_columns(predict_frame)
        features = self._extract_features(predict_frame, parameter="frame")
        raw = _predict_booster(booster, features)
        return self._format_predictions(raw, row_count=predict_frame.height)

    def save(self, path: Path | str) -> None:
        """Persist the fitted booster with LightGBM native serialization.

        Args:
            path: Destination path for the serialized model.

        Raises:
            ModelValidationError: If the model is not fitted or ``path`` is
                invalid.
        """
        booster = self._require_booster()
        destination = self._require_path(path)
        booster.save_model(str(destination))
        _logger.info(
            "Saved LightGBM model",
            extra={"model": self.model_metadata.name, "path": str(destination)},
        )

    def load(self, path: Path | str) -> Self:
        """Load a booster from ``path`` using LightGBM native serialization.

        Args:
            path: Source path of the serialized model.

        Returns:
            ``self`` with the loaded booster installed.

        Raises:
            ModelValidationError: If ``path`` is invalid.
        """
        source = self._require_path(path)
        booster = _load_booster(source)
        self._state[_STATE_BOOSTER] = booster
        _logger.info(
            "Loaded LightGBM model",
            extra={"model": self.model_metadata.name, "path": str(source)},
        )
        return self

    def _build_params(self, y_train: np.ndarray[Any, np.dtype[Any]]) -> dict[str, object]:
        """Build LightGBM parameters from metadata and training labels."""
        params: dict[str, object] = {"verbosity": _DEFAULT_VERBOSITY}
        task_type = self.model_metadata.task_type

        if task_type is ModelTaskType.REGRESSION:
            params["objective"] = "regression"
            return params

        if task_type is ModelTaskType.CLASSIFICATION:
            classes = np.unique(y_train)
            class_count = int(classes.size)
            if class_count < 2:
                raise ModelValidationError(
                    "classification requires at least two label classes",
                    error_code=_ERROR_LABEL_CLASSES,
                    details={"class_count": class_count},
                )
            if class_count == 2:
                params["objective"] = "binary"
                return params
            params["objective"] = "multiclass"
            params["num_class"] = class_count
            return params

        raise ModelValidationError(
            "task_type must be REGRESSION or CLASSIFICATION",
            error_code=_ERROR_TASK_TYPE,
            details={"task_type": str(task_type)},
        )

    def _extract_xy(
        self,
        frame: pl.DataFrame,
        *,
        parameter: str,
    ) -> tuple[np.ndarray[Any, np.dtype[Any]], np.ndarray[Any, np.dtype[Any]]]:
        """Extract feature matrix and label vector from ``frame``."""
        features = self._extract_features(frame, parameter=parameter)
        labels = cast(
            np.ndarray[Any, np.dtype[Any]],
            frame.get_column(self.model_metadata.label_column).to_numpy(),
        )
        if features.shape[0] != labels.shape[0]:
            raise ModelValidationError(
                "feature and label row counts must match",
                error_code=_ERROR_XY_MISMATCH,
                details={
                    "parameter": parameter,
                    "feature_rows": int(features.shape[0]),
                    "label_rows": int(labels.shape[0]),
                },
            )
        return features, labels

    def _extract_features(
        self,
        frame: pl.DataFrame,
        *,
        parameter: str,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Extract the feature matrix for LightGBM."""
        features = cast(
            np.ndarray[Any, np.dtype[Any]],
            frame.select(list(self.model_metadata.feature_columns)).to_numpy(),
        )
        if features.size == 0:
            raise ModelValidationError(
                f"{parameter} feature matrix must not be empty",
                error_code=_ERROR_EMPTY_FEATURES,
                details={"parameter": parameter},
            )
        return features

    def _require_booster(self) -> lgb.Booster:
        """Return the fitted booster or raise when missing."""
        booster = self._state.get(_STATE_BOOSTER)
        if booster is None:
            raise ModelValidationError(
                "model must be fitted before this operation",
                error_code=_ERROR_NOT_FITTED,
                details={"model": self.model_metadata.name},
            )
        return cast(lgb.Booster, booster)

    def _format_predictions(
        self,
        raw: np.ndarray[Any, np.dtype[Any]],
        *,
        row_count: int,
    ) -> pl.Series:
        """Convert LightGBM raw output into a CQROS prediction series."""
        values = np.asarray(raw)
        if self.model_metadata.task_type is ModelTaskType.REGRESSION:
            flat = values.reshape(row_count).astype(np.float64, copy=False)
            return pl.Series(_PREDICTION_NAME, flat, dtype=pl.Float64)

        if values.ndim == 2:
            labels = values.argmax(axis=1).astype(np.int64, copy=False)
        else:
            labels = (values >= _BINARY_THRESHOLD).astype(np.int64, copy=False)
        return pl.Series(_PREDICTION_NAME, labels, dtype=pl.Int64)


def _train_booster(
    params: dict[str, object],
    train_set: lgb.Dataset,
    *,
    num_boost_round: int,
    valid_sets: list[lgb.Dataset],
) -> lgb.Booster:
    """Train a LightGBM booster with an explicitly typed return value."""
    return lgb.train(  # pyright: ignore[reportUnknownMemberType]
        cast(dict[str, Any], params),
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
    )


def _predict_booster(
    booster: lgb.Booster,
    features: np.ndarray[Any, np.dtype[Any]],
) -> np.ndarray[Any, np.dtype[Any]]:
    """Generate raw LightGBM predictions with an explicitly typed array."""
    raw = booster.predict(features)  # pyright: ignore[reportUnknownMemberType]
    return cast(np.ndarray[Any, np.dtype[Any]], np.asarray(raw))


def _load_booster(path: Path) -> lgb.Booster:
    """Load a LightGBM booster from ``path``."""
    return lgb.Booster(model_file=str(path))
