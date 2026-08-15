"""CQROS ML DatasetSplitter.

Purpose:
    Produce chronological train, validation, and test frames from a canonical
    ML dataset without look-ahead bias.

Responsibilities:
    - Validate input frames and split ratios
    - Split strictly in chronological row order without shuffling
    - Guarantee disjoint splits with complete row coverage
    - Preserve canonical column order and schema dtypes
    - Return new DataFrames without mutating the input
    - Remain free of loading, scaling, training, and repository access

Dependencies:
    ``polars``, ``cqros.ml.dataset.exceptions``, and ``cqros.ml.dataset.schema``.

Public API:
    ``DatasetSplitter``
"""

from __future__ import annotations

import logging
import math
from typing import Final

import polars as pl

from cqros.ml.dataset.exceptions import DatasetSplitterError
from cqros.ml.dataset.schema import (
    CANONICAL_COLUMN_ORDER,
    MERGED_TRAINING_SCHEMA,
    REQUIRED_COLUMNS,
)

__all__ = ["DatasetSplitter"]

_logger = logging.getLogger(__name__)

_ERROR_EMPTY_FRAME: Final[str] = "ML-DATASET-SPLIT-001"
_ERROR_MISSING_COLUMNS: Final[str] = "ML-DATASET-SPLIT-002"
_ERROR_RATIO_TYPE: Final[str] = "ML-DATASET-SPLIT-003"
_ERROR_RATIO_RANGE: Final[str] = "ML-DATASET-SPLIT-004"
_ERROR_RATIO_SUM: Final[str] = "ML-DATASET-SPLIT-005"

_COLUMN_ORDER: Final[list[str]] = list(CANONICAL_COLUMN_ORDER)
_RATIO_SUM_TOLERANCE: Final[float] = 1e-9


class DatasetSplitter:
    """Deterministic chronological train/validation/test splitter.

    The splitter validates the assembled ML dataset and ratios, then partitions
    rows in existing chronological order. Rows are never shuffled. Validation
    rows always follow train rows, and test rows always follow validation rows.
    Caller-supplied frames are never mutated.

    Args:
        logger: Optional logger instance. Defaults to the module logger.
    """

    __slots__ = ("_logger",)

    _logger: logging.Logger

    def __init__(self, *, logger: logging.Logger | None = None) -> None:
        """Initialize the splitter.

        Args:
            logger: Optional logger instance.
        """
        self._logger = logger if logger is not None else _logger

    def split(
        self,
        frame: pl.DataFrame,
        *,
        train_ratio: float,
        validation_ratio: float,
        test_ratio: float,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Split ``frame`` into chronological train, validation, and test sets.

        Args:
            frame: Canonical ML dataset ordered by symbol, timeframe, open_time.
            train_ratio: Fraction of rows assigned to the train split.
            validation_ratio: Fraction of rows assigned to the validation split.
            test_ratio: Fraction of rows assigned to the test split.

        Returns:
            ``(train_frame, validation_frame, test_frame)`` as new DataFrames
            with canonical column order and dtypes.

        Raises:
            DatasetSplitterError: If the frame is empty, required columns are
                missing, or ratios are invalid.
        """
        _validate_frame(frame)
        train = _validate_ratio(train_ratio, parameter="train_ratio")
        validation = _validate_ratio(validation_ratio, parameter="validation_ratio")
        test = _validate_ratio(test_ratio, parameter="test_ratio")
        _validate_ratio_sum(train=train, validation=validation, test=test)

        row_count = frame.height
        train_end = int(row_count * train)
        validation_end = train_end + int(row_count * validation)

        self._logger.debug(
            "Splitting ML dataset",
            extra={
                "rows": row_count,
                "train_ratio": train,
                "validation_ratio": validation,
                "test_ratio": test,
                "train_end": train_end,
                "validation_end": validation_end,
            },
        )

        train_frame = _finalize(frame.slice(0, train_end))
        validation_frame = _finalize(frame.slice(train_end, validation_end - train_end))
        test_frame = _finalize(frame.slice(validation_end, row_count - validation_end))

        self._logger.info(
            "Split ML dataset",
            extra={
                "rows": row_count,
                "train_rows": train_frame.height,
                "validation_rows": validation_frame.height,
                "test_rows": test_frame.height,
            },
        )
        return train_frame, validation_frame, test_frame


def _finalize(frame: pl.DataFrame) -> pl.DataFrame:
    """Return a new frame with canonical column order and dtypes."""
    return frame.select(_COLUMN_ORDER).cast(MERGED_TRAINING_SCHEMA)


def _validate_frame(frame: pl.DataFrame) -> None:
    """Reject empty frames and frames missing required columns.

    Args:
        frame: Candidate ML dataset.

    Raises:
        DatasetSplitterError: If ``frame`` is empty or incomplete.
    """
    if frame.height == 0:
        raise DatasetSplitterError(
            "dataset must contain at least one row",
            error_code=_ERROR_EMPTY_FRAME,
            details={"rows": frame.height},
        )

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DatasetSplitterError(
            "dataset is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "missing_columns": tuple(missing),
                "required_columns": REQUIRED_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )


def _validate_ratio(value: object, *, parameter: str) -> float:
    """Validate one split ratio.

    Args:
        value: Candidate ratio.
        parameter: Parameter name used in error messages.

    Returns:
        The ratio as ``float``.

    Raises:
        DatasetSplitterError: If ``value`` is not a finite ratio in ``[0, 1]``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DatasetSplitterError(
            f"{parameter} must be a number",
            error_code=_ERROR_RATIO_TYPE,
            details={"parameter": parameter, "value_type": type(value).__name__},
        )
    ratio = float(value)
    if not math.isfinite(ratio) or ratio < 0.0 or ratio > 1.0:
        raise DatasetSplitterError(
            f"{parameter} must be between 0 and 1 inclusive",
            error_code=_ERROR_RATIO_RANGE,
            details={"parameter": parameter, "value": ratio},
        )
    return ratio


def _validate_ratio_sum(*, train: float, validation: float, test: float) -> None:
    """Require train, validation, and test ratios to sum to 1.0.

    Args:
        train: Validated train ratio.
        validation: Validated validation ratio.
        test: Validated test ratio.

    Raises:
        DatasetSplitterError: If the ratios do not sum to 1.0.
    """
    total = train + validation + test
    if not math.isclose(total, 1.0, abs_tol=_RATIO_SUM_TOLERANCE):
        raise DatasetSplitterError(
            "train_ratio, validation_ratio, and test_ratio must sum to 1.0",
            error_code=_ERROR_RATIO_SUM,
            details={
                "train_ratio": train,
                "validation_ratio": validation,
                "test_ratio": test,
                "sum": total,
            },
        )
