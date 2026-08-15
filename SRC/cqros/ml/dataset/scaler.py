"""CQROS ML DatasetScaler.

Purpose:
    Fit feature scaling parameters on the training dataset and transform
    train, validation, and test datasets consistently.

Responsibilities:
    - Scale ``FEATURE_COLUMNS`` only
    - Preserve primary-key and label columns unchanged
    - Fit on training data and reuse parameters for later transforms
    - Return new DataFrames without mutating inputs
    - Preserve canonical column ordering
    - Remain free of loading, splitting, statistics, and repository access

Dependencies:
    ``polars``, ``cqros.ml.dataset.exceptions``, and ``cqros.ml.dataset.schema``.

Public API:
    ``DatasetScaler``, ``IdentityScaler``, ``StandardScaler``, ``MinMaxScaler``
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Final, Self

import polars as pl

from cqros.ml.dataset.exceptions import DatasetScalerError
from cqros.ml.dataset.schema import (
    CANONICAL_COLUMN_ORDER,
    FEATURE_COLUMNS,
)

__all__ = [
    "DatasetScaler",
    "IdentityScaler",
    "MinMaxScaler",
    "StandardScaler",
]

_logger = logging.getLogger(__name__)

_ERROR_EMPTY_FRAME: Final[str] = "ML-DATASET-SCALE-001"
_ERROR_MISSING_FEATURES: Final[str] = "ML-DATASET-SCALE-002"
_ERROR_NOT_FITTED: Final[str] = "ML-DATASET-SCALE-003"

_COLUMN_ORDER: Final[list[str]] = list(CANONICAL_COLUMN_ORDER)
_FEATURE_COLUMNS: Final[tuple[str, ...]] = FEATURE_COLUMNS
_STD_DDOF: Final[int] = 0


class DatasetScaler(ABC):
    """Common interface for deterministic ML feature scalers.

    Scaling applies only to ``FEATURE_COLUMNS``. Primary-key and label columns
    are preserved exactly. Parameters are fit on training data and reused for
    validation and test transforms. Caller-supplied frames are never mutated.

    Args:
        logger: Optional logger instance. Defaults to the module logger.
    """

    __slots__ = ("_fitted", "_logger")

    _fitted: bool
    _logger: logging.Logger

    def __init__(self, *, logger: logging.Logger | None = None) -> None:
        """Initialize an unfitted scaler.

        Args:
            logger: Optional logger instance.
        """
        self._fitted = False
        self._logger = logger if logger is not None else _logger

    def fit(self, train_frame: pl.DataFrame) -> Self:
        """Fit scaling parameters from ``train_frame`` feature columns.

        Args:
            train_frame: Training dataset used to estimate parameters.

        Returns:
            ``self`` for fluent chaining.

        Raises:
            DatasetScalerError: If ``train_frame`` is empty or missing feature
                columns.
        """
        _validate_frame(train_frame)
        self._fit_parameters(train_frame)
        self._fitted = True
        self._logger.info(
            "Fitted dataset scaler",
            extra={
                "scaler": type(self).__name__,
                "rows": train_frame.height,
                "feature_count": len(_FEATURE_COLUMNS),
            },
        )
        return self

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Transform feature columns using fitted parameters.

        Args:
            frame: Dataset to transform.

        Returns:
            A new DataFrame with scaled features and unchanged non-feature
            columns in canonical order.

        Raises:
            DatasetScalerError: If the scaler is not fitted, ``frame`` is empty,
                or feature columns are missing.
        """
        _ensure_fitted(self._fitted)
        _validate_frame(frame)
        return self._transform_frame(frame, inverse=False)

    def fit_transform(self, train_frame: pl.DataFrame) -> pl.DataFrame:
        """Fit on ``train_frame`` and return its scaled transform.

        Args:
            train_frame: Training dataset used for fit and transform.

        Returns:
            A new scaled training DataFrame.

        Raises:
            DatasetScalerError: If ``train_frame`` is empty or missing feature
                columns.
        """
        self.fit(train_frame)
        return self.transform(train_frame)

    def inverse_transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Invert feature scaling using fitted parameters.

        Args:
            frame: Previously scaled dataset.

        Returns:
            A new DataFrame with features restored to the original scale.

        Raises:
            DatasetScalerError: If the scaler is not fitted, ``frame`` is empty,
                or feature columns are missing.
        """
        _ensure_fitted(self._fitted)
        _validate_frame(frame)
        return self._transform_frame(frame, inverse=True)

    @abstractmethod
    def _fit_parameters(self, train_frame: pl.DataFrame) -> None:
        """Estimate scaler parameters from ``train_frame``."""

    @abstractmethod
    def _feature_expression(self, column: str, *, inverse: bool) -> pl.Expr:
        """Return the forward or inverse scaling expression for ``column``."""

    def _transform_frame(self, frame: pl.DataFrame, *, inverse: bool) -> pl.DataFrame:
        """Apply forward or inverse feature scaling and restore column order."""
        expressions = [
            self._feature_expression(column, inverse=inverse).alias(column)
            for column in _FEATURE_COLUMNS
        ]
        transformed = frame.with_columns(expressions)
        return _finalize(transformed)


class IdentityScaler(DatasetScaler):
    """No-op scaler that leaves feature values unchanged."""

    __slots__ = ()

    def _fit_parameters(self, train_frame: pl.DataFrame) -> None:
        """Identity fitting records no parameters."""
        del train_frame

    def _feature_expression(self, column: str, *, inverse: bool) -> pl.Expr:
        """Return the unmodified feature column."""
        del inverse
        return pl.col(column)


class StandardScaler(DatasetScaler):
    """Scale features to zero mean and unit variance.

    Uses population standard deviation (``ddof=0``). Constant features with
    ``std == 0`` transform to ``0.0`` and inverse-transform back to the fitted
    mean.
    """

    __slots__ = ("_means", "_stds")

    _means: tuple[float, ...]
    _stds: tuple[float, ...]

    def __init__(self, *, logger: logging.Logger | None = None) -> None:
        """Initialize an unfitted standard scaler."""
        super().__init__(logger=logger)
        self._means = ()
        self._stds = ()

    def _fit_parameters(self, train_frame: pl.DataFrame) -> None:
        """Store per-feature mean and population standard deviation."""
        means: list[float] = []
        stds: list[float] = []
        for column in _FEATURE_COLUMNS:
            series = train_frame.get_column(column)
            mean_value = series.mean()
            std_value = series.std(ddof=_STD_DDOF)
            means.append(0.0 if mean_value is None else _as_float(mean_value))
            stds.append(0.0 if std_value is None else _as_float(std_value))
        self._means = tuple(means)
        self._stds = tuple(stds)

    def _feature_expression(self, column: str, *, inverse: bool) -> pl.Expr:
        """Apply or invert standard scaling for ``column``."""
        index = _FEATURE_COLUMNS.index(column)
        mean = self._means[index]
        std = self._stds[index]
        if inverse:
            if std == 0.0:
                return pl.lit(mean).cast(pl.Float64)
            return pl.col(column) * std + mean
        if std == 0.0:
            return pl.when(pl.col(column).is_null()).then(None).otherwise(0.0).cast(pl.Float64)
        return ((pl.col(column) - mean) / std).cast(pl.Float64)


class MinMaxScaler(DatasetScaler):
    """Scale features into the unit interval ``[0, 1]``.

    Constant features with ``max == min`` transform to ``0.0`` and
    inverse-transform back to the fitted minimum.
    """

    __slots__ = ("_maxima", "_minima")

    _minima: tuple[float, ...]
    _maxima: tuple[float, ...]

    def __init__(self, *, logger: logging.Logger | None = None) -> None:
        """Initialize an unfitted min-max scaler."""
        super().__init__(logger=logger)
        self._minima = ()
        self._maxima = ()

    def _fit_parameters(self, train_frame: pl.DataFrame) -> None:
        """Store per-feature minimum and maximum."""
        minima: list[float] = []
        maxima: list[float] = []
        for column in _FEATURE_COLUMNS:
            series = train_frame.get_column(column)
            min_value = series.min()
            max_value = series.max()
            minima.append(0.0 if min_value is None else _as_float(min_value))
            maxima.append(0.0 if max_value is None else _as_float(max_value))
        self._minima = tuple(minima)
        self._maxima = tuple(maxima)

    def _feature_expression(self, column: str, *, inverse: bool) -> pl.Expr:
        """Apply or invert min-max scaling for ``column``."""
        index = _FEATURE_COLUMNS.index(column)
        minimum = self._minima[index]
        maximum = self._maxima[index]
        span = maximum - minimum
        if inverse:
            if span == 0.0:
                return pl.lit(minimum).cast(pl.Float64)
            return pl.col(column) * span + minimum
        if span == 0.0:
            return pl.when(pl.col(column).is_null()).then(None).otherwise(0.0).cast(pl.Float64)
        return ((pl.col(column) - minimum) / span).cast(pl.Float64)


def _validate_frame(frame: pl.DataFrame) -> None:
    """Reject empty frames and frames missing feature columns.

    Args:
        frame: Candidate ML dataset.

    Raises:
        DatasetScalerError: If ``frame`` is empty or incomplete.
    """
    if frame.height == 0:
        raise DatasetScalerError(
            "dataset must contain at least one row",
            error_code=_ERROR_EMPTY_FRAME,
            details={"rows": frame.height},
        )

    missing = [column for column in _FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise DatasetScalerError(
            "dataset is missing required feature columns",
            error_code=_ERROR_MISSING_FEATURES,
            details={
                "missing_columns": tuple(missing),
                "required_feature_columns": _FEATURE_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )


def _ensure_fitted(fitted: bool) -> None:
    """Reject transform calls before fit.

    Args:
        fitted: Whether the scaler has been fitted.

    Raises:
        DatasetScalerError: If the scaler has not been fitted.
    """
    if not fitted:
        raise DatasetScalerError(
            "scaler must be fitted before transform",
            error_code=_ERROR_NOT_FITTED,
        )


def _finalize(frame: pl.DataFrame) -> pl.DataFrame:
    """Return a new frame with canonical column order when possible."""
    if set(CANONICAL_COLUMN_ORDER).issubset(frame.columns):
        return frame.select(_COLUMN_ORDER)
    return frame.select(list(frame.columns))


def _as_float(value: object) -> float:
    """Convert a Polars aggregate value to ``float``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float(str(value))
    return float(value)
