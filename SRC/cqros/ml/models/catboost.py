"""CQROS CatBoost model implementation.

Purpose:
    Provide the third framework-specific ``BaseModel`` implementation backed
    by the official CatBoost Python API.

Responsibilities:
    - Fit regression and classification models from research frames
    - Produce Float64 regression predictions or class-label predictions
    - Persist and restore models through CatBoost native serialization
    - Remain free of trainers, evaluators, cross-validation, and HPO

Dependencies:
    ``catboost``, ``numpy``, ``polars``, ``pathlib``,
    ``cqros.ml.models.base.BaseModel``, ``cqros.ml.models.exceptions``,
    ``cqros.ml.models.metadata``, and ``cqros.ml.models.interfaces.Model``.

Public API:
    ``CatBoostModel``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Self, cast

import numpy as np
import polars as pl
from catboost import CatBoost, Pool  # pyright: ignore[reportMissingTypeStubs]

from cqros.ml.models.base import BaseModel
from cqros.ml.models.exceptions import ModelValidationError
from cqros.ml.models.metadata import ModelFramework, ModelTaskType

__all__ = ["CatBoostModel"]

_logger = logging.getLogger(__name__)

_ERROR_FRAMEWORK: Final[str] = "ML-MODEL-CAT-001"
_ERROR_TASK_TYPE: Final[str] = "ML-MODEL-CAT-002"
_ERROR_NOT_FITTED: Final[str] = "ML-MODEL-CAT-003"
_ERROR_XY_MISMATCH: Final[str] = "ML-MODEL-CAT-004"
_ERROR_EMPTY_FEATURES: Final[str] = "ML-MODEL-CAT-005"
_ERROR_LABEL_CLASSES: Final[str] = "ML-MODEL-CAT-006"

_DEFAULT_NUM_BOOST_ROUND: Final[int] = 100
_PREDICTION_NAME: Final[str] = "prediction"
_STATE_MODEL: Final[str] = "model"


def _new_state() -> dict[str, object]:
    """Return an empty mutable model-state mapping."""
    return {}


@dataclass(frozen=True, slots=True)
class CatBoostModel(BaseModel):
    """CatBoost-backed CQROS model for regression and classification.

    Training consumes a Polars frame using ``feature_columns`` and
    ``label_column`` from ``model_metadata``. Optional validation data may be
    supplied as a keyword-only frame for CatBoost ``eval_set``. Feature and
    label arrays are converted to ``Pool`` objects internally. The trained
    model is stored in mutable state and never exposes framework-specific APIs
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
        """Validate metadata framework and task type for CatBoost.

        Raises:
            ModelValidationError: If metadata is incompatible with CatBoost.
        """
        BaseModel.__post_init__(self)
        if self.model_metadata.framework is not ModelFramework.CATBOOST:
            raise ModelValidationError(
                "CatBoostModel requires framework=ModelFramework.CATBOOST",
                error_code=_ERROR_FRAMEWORK,
                details={
                    "framework": str(self.model_metadata.framework),
                    "expected": str(ModelFramework.CATBOOST),
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
        """Fit the CatBoost model on ``frame``.

        Feature matrix ``X_train`` and label vector ``y_train`` are extracted
        from ``frame`` using metadata column contracts and converted into
        ``Pool`` objects. When ``validation_frame`` is provided it is passed
        to CatBoost as an evaluation set.

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
        train_pool = Pool(x_train, label=y_train)

        eval_set: Pool | None = None
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
            eval_set = Pool(x_valid, label=y_valid)

        params = self._build_params(y_train)
        model = _train_model(
            params,
            train_pool,
            eval_set=eval_set,
        )
        self._state[_STATE_MODEL] = model

        _logger.info(
            "Fitted CatBoost model",
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
        model = self._require_model()
        predict_frame = self._require_dataframe(frame, parameter="frame")
        self._require_feature_columns(predict_frame)
        features = self._extract_features(predict_frame, parameter="frame")
        raw = _predict_model(
            model,
            features,
            task_type=self.model_metadata.task_type,
        )
        return self._format_predictions(raw, row_count=predict_frame.height)

    def save(self, path: Path | str) -> None:
        """Persist the fitted model with CatBoost native serialization.

        Args:
            path: Destination path for the serialized model.

        Raises:
            ModelValidationError: If the model is not fitted or ``path`` is
                invalid.
        """
        model = self._require_model()
        destination = self._require_path(path)
        model.save_model(str(destination))  # pyright: ignore[reportUnknownMemberType]
        _logger.info(
            "Saved CatBoost model",
            extra={"model": self.model_metadata.name, "path": str(destination)},
        )

    def load(self, path: Path | str) -> Self:
        """Load a model from ``path`` using CatBoost native serialization.

        Args:
            path: Source path of the serialized model.

        Returns:
            ``self`` with the loaded model installed.

        Raises:
            ModelValidationError: If ``path`` is invalid.
        """
        source = self._require_path(path)
        model = _load_model(source)
        self._state[_STATE_MODEL] = model
        _logger.info(
            "Loaded CatBoost model",
            extra={"model": self.model_metadata.name, "path": str(source)},
        )
        return self

    def _build_params(self, y_train: np.ndarray[Any, np.dtype[Any]]) -> dict[str, object]:
        """Build CatBoost parameters from metadata and training labels."""
        params: dict[str, object] = {
            "iterations": self.num_boost_round,
            "verbose": False,
            "allow_writing_files": False,
        }
        task_type = self.model_metadata.task_type

        if task_type is ModelTaskType.REGRESSION:
            params["loss_function"] = "RMSE"
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
                params["loss_function"] = "Logloss"
                return params
            params["loss_function"] = "MultiClass"
            params["classes_count"] = class_count
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
        """Extract the feature matrix for CatBoost."""
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

    def _require_model(self) -> CatBoost:
        """Return the fitted CatBoost model or raise when missing."""
        model = self._state.get(_STATE_MODEL)
        if model is None:
            raise ModelValidationError(
                "model must be fitted before this operation",
                error_code=_ERROR_NOT_FITTED,
                details={"model": self.model_metadata.name},
            )
        return cast(CatBoost, model)

    def _format_predictions(
        self,
        raw: np.ndarray[Any, np.dtype[Any]],
        *,
        row_count: int,
    ) -> pl.Series:
        """Convert CatBoost raw output into a CQROS prediction series."""
        values = np.asarray(raw)
        if self.model_metadata.task_type is ModelTaskType.REGRESSION:
            flat = values.reshape(row_count).astype(np.float64, copy=False)
            return pl.Series(_PREDICTION_NAME, flat, dtype=pl.Float64)

        flat = values.reshape(row_count).astype(np.int64, copy=False)
        return pl.Series(_PREDICTION_NAME, flat, dtype=pl.Int64)


def _train_model(
    params: dict[str, object],
    train_pool: Pool,
    *,
    eval_set: Pool | None,
) -> CatBoost:
    """Train a CatBoost model with an explicitly typed return value."""
    model = CatBoost(cast(dict[str, Any], params))
    if eval_set is None:
        model.fit(train_pool)  # pyright: ignore[reportUnknownMemberType]
    else:
        model.fit(train_pool, eval_set=eval_set)  # pyright: ignore[reportUnknownMemberType]
    return model


def _predict_model(
    model: CatBoost,
    features: np.ndarray[Any, np.dtype[Any]],
    *,
    task_type: ModelTaskType,
) -> np.ndarray[Any, np.dtype[Any]]:
    """Generate CatBoost predictions with an explicitly typed array."""
    pool = Pool(features)
    if task_type is ModelTaskType.CLASSIFICATION:
        raw = cast(
            object,
            model.predict(  # pyright: ignore[reportUnknownMemberType]
                pool,
                prediction_type="Class",
            ),
        )
    else:
        raw = cast(
            object,
            model.predict(pool),  # pyright: ignore[reportUnknownMemberType]
        )
    return cast(np.ndarray[Any, np.dtype[Any]], np.asarray(raw))


def _load_model(path: Path) -> CatBoost:
    """Load a CatBoost model from ``path``."""
    model = CatBoost()
    model.load_model(str(path))  # pyright: ignore[reportUnknownMemberType]
    return model
