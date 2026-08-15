"""CQROS production signal policies.

Purpose:
    Provide deterministic ``SignalPolicy`` implementations that convert
    canonical prediction datasets into canonical signal datasets.

Responsibilities:
    - Validate policy construction parameters
    - Validate canonical prediction DataFrame structure and dtypes
    - Map continuous or discrete predictions to ``Signal`` values
    - Return newly constructed signal DataFrames
    - Remain free of persistence, repositories, CLI, and trading execution

Dependencies:
    ``polars``, ``math``, ``cqros.predictions.schema``,
    ``cqros.signals.enums``, ``cqros.signals.exceptions``, and
    ``cqros.signals.schema``.

Public API:
    ``RegressionSignalPolicy``, ``ClassificationSignalPolicy``
"""

from __future__ import annotations

import math
from typing import Final

import polars as pl

from cqros.predictions.schema import (
    COLUMN_DTYPES as PREDICTION_COLUMN_DTYPES,
)
from cqros.predictions.schema import (
    REQUIRED_COLUMNS as PREDICTION_REQUIRED_COLUMNS,
)
from cqros.signals.enums import Signal
from cqros.signals.exceptions import SignalValidationError
from cqros.signals.schema import (
    CANONICAL_COLUMN_ORDER,
    MERGED_SIGNAL_SCHEMA,
    METADATA_COLUMNS,
    PRIMARY_KEY_COLUMNS,
)

__all__ = [
    "ClassificationSignalPolicy",
    "RegressionSignalPolicy",
]

_ERROR_THRESHOLD_TYPE: Final[str] = "SIGNAL-POL-001"
_ERROR_THRESHOLD_FINITE: Final[str] = "SIGNAL-POL-002"
_ERROR_THRESHOLD_ORDER: Final[str] = "SIGNAL-POL-003"
_ERROR_CLASS_TYPE: Final[str] = "SIGNAL-POL-004"
_ERROR_CLASS_IDENTICAL: Final[str] = "SIGNAL-POL-005"
_ERROR_FRAME_TYPE: Final[str] = "SIGNAL-POL-006"
_ERROR_FRAME_EMPTY: Final[str] = "SIGNAL-POL-007"
_ERROR_MISSING_COLUMNS: Final[str] = "SIGNAL-POL-008"
_ERROR_DTYPE_MISMATCH: Final[str] = "SIGNAL-POL-009"
_ERROR_PREDICTION_INVALID: Final[str] = "SIGNAL-POL-010"

_OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    *PRIMARY_KEY_COLUMNS,
    *METADATA_COLUMNS,
)


class RegressionSignalPolicy:
    """Map continuous predictions to discrete signals via thresholds.

    Rules:
        - ``prediction >= buy_threshold`` → ``BUY``
        - ``prediction <= sell_threshold`` → ``SELL``
        - ``sell_threshold < prediction < buy_threshold`` → ``HOLD``

    Args:
        buy_threshold: Inclusive lower bound for ``BUY`` signals.
        sell_threshold: Inclusive upper bound for ``SELL`` signals.

    Raises:
        SignalValidationError: If thresholds are non-numeric, non-finite, or
            ``buy_threshold <= sell_threshold``.
    """

    __slots__ = ("_buy_threshold", "_sell_threshold")

    _buy_threshold: float
    _sell_threshold: float

    def __init__(self, buy_threshold: float, sell_threshold: float) -> None:
        """Initialize thresholds after validating numeric ordering constraints.

        Args:
            buy_threshold: Inclusive lower bound for ``BUY`` signals.
            sell_threshold: Inclusive upper bound for ``SELL`` signals.

        Raises:
            SignalValidationError: If thresholds fail validation.
        """
        buy = _require_finite_number(buy_threshold, parameter="buy_threshold")
        sell = _require_finite_number(sell_threshold, parameter="sell_threshold")
        if buy <= sell:
            raise SignalValidationError(
                "buy_threshold must be greater than sell_threshold",
                error_code=_ERROR_THRESHOLD_ORDER,
                details={
                    "buy_threshold": buy,
                    "sell_threshold": sell,
                },
            )
        self._buy_threshold = buy
        self._sell_threshold = sell

    def generate(self, predictions: pl.DataFrame) -> pl.DataFrame:
        """Convert a prediction frame into a canonical signal frame.

        Args:
            predictions: Canonical prediction dataset. Must not be mutated.

        Returns:
            A new DataFrame matching ``MERGED_SIGNAL_SCHEMA``, ordered to
            ``CANONICAL_COLUMN_ORDER``. Primary-key and model metadata columns
            are preserved unchanged; only ``signal`` is generated.

        Raises:
            SignalValidationError: If ``predictions`` fails structural
                validation, or any ``prediction`` value is null, NaN, or
                non-finite.
        """
        frame = _require_prediction_frame(predictions)
        _require_finite_predictions(frame)
        signal_expr = (
            pl.when(pl.col("prediction") >= self._buy_threshold)
            .then(pl.lit(Signal.BUY.value))
            .when(pl.col("prediction") <= self._sell_threshold)
            .then(pl.lit(Signal.SELL.value))
            .otherwise(pl.lit(Signal.HOLD.value))
            .alias("signal")
        )
        return _build_signal_frame(frame, signal_expr)


class ClassificationSignalPolicy:
    """Map discrete class predictions to discrete trading signals.

    Rules:
        - ``prediction == positive_class`` → ``BUY``
        - ``prediction == negative_class`` → ``SELL``
        - otherwise → ``HOLD``

    Args:
        positive_class: Class label mapped to ``BUY``. Defaults to ``1``.
        negative_class: Class label mapped to ``SELL``. Defaults to ``0``.

    Raises:
        SignalValidationError: If class labels are non-integer or identical.
    """

    __slots__ = ("_negative_class", "_positive_class")

    _positive_class: int
    _negative_class: int

    def __init__(
        self,
        positive_class: int = 1,
        negative_class: int = 0,
    ) -> None:
        """Initialize class labels after validating integer uniqueness.

        Args:
            positive_class: Class label mapped to ``BUY``.
            negative_class: Class label mapped to ``SELL``.

        Raises:
            SignalValidationError: If class labels fail validation.
        """
        positive = _require_integer(positive_class, parameter="positive_class")
        negative = _require_integer(negative_class, parameter="negative_class")
        if positive == negative:
            raise SignalValidationError(
                "positive_class and negative_class must be distinct",
                error_code=_ERROR_CLASS_IDENTICAL,
                details={
                    "positive_class": positive,
                    "negative_class": negative,
                },
            )
        self._positive_class = positive
        self._negative_class = negative

    def generate(self, predictions: pl.DataFrame) -> pl.DataFrame:
        """Convert a prediction frame into a canonical signal frame.

        Args:
            predictions: Canonical prediction dataset. Must not be mutated.

        Returns:
            A new DataFrame matching the merged signal schema.

        Raises:
            SignalValidationError: If ``predictions`` fails structural
                validation.
        """
        frame = _require_prediction_frame(predictions)
        signal_expr = (
            pl.when(pl.col("prediction") == self._positive_class)
            .then(pl.lit(Signal.BUY.value))
            .when(pl.col("prediction") == self._negative_class)
            .then(pl.lit(Signal.SELL.value))
            .otherwise(pl.lit(Signal.HOLD.value))
            .alias("signal")
        )
        return _build_signal_frame(frame, signal_expr)


def _build_signal_frame(frame: pl.DataFrame, signal_expr: pl.Expr) -> pl.DataFrame:
    """Assemble a canonical signal DataFrame from prediction columns.

    Args:
        frame: Validated prediction DataFrame.
        signal_expr: Expression producing the ``signal`` column.

    Returns:
        A new DataFrame ordered and cast to ``MERGED_SIGNAL_SCHEMA``.
    """
    assembled = frame.select(
        *[pl.col(column) for column in _OUTPUT_COLUMNS],
        signal_expr,
    )
    return assembled.select(list(CANONICAL_COLUMN_ORDER)).cast(MERGED_SIGNAL_SCHEMA)


def _require_prediction_frame(predictions: object) -> pl.DataFrame:
    """Validate that ``predictions`` is a non-empty canonical prediction frame.

    Args:
        predictions: Candidate prediction dataset.

    Returns:
        ``predictions`` as a DataFrame after structural checks.

    Raises:
        SignalValidationError: If the frame is empty, missing required
            columns, or has incorrect dtypes.
    """
    if not isinstance(predictions, pl.DataFrame):
        raise SignalValidationError(
            "predictions must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={"actual_type": type(predictions).__name__},
        )
    if predictions.height == 0:
        raise SignalValidationError(
            "predictions must contain at least one row",
            error_code=_ERROR_FRAME_EMPTY,
            details={"rows": predictions.height},
        )

    missing = [
        column for column in PREDICTION_REQUIRED_COLUMNS if column not in predictions.columns
    ]
    if missing:
        raise SignalValidationError(
            "prediction frame is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "missing_columns": tuple(missing),
                "required_columns": PREDICTION_REQUIRED_COLUMNS,
                "available_columns": tuple(predictions.columns),
            },
        )

    mismatched: list[dict[str, object]] = []
    for column in PREDICTION_REQUIRED_COLUMNS:
        expected = PREDICTION_COLUMN_DTYPES[column]
        actual = predictions.schema[column]
        if actual != expected:
            mismatched.append(
                {
                    "column": column,
                    "expected": str(expected),
                    "actual": str(actual),
                }
            )
    if mismatched:
        raise SignalValidationError(
            "prediction frame dtype mismatch",
            error_code=_ERROR_DTYPE_MISMATCH,
            details={
                "mismatched_columns": tuple(item["column"] for item in mismatched),
                "mismatches": tuple(mismatched),
            },
        )
    return predictions


def _require_finite_predictions(frame: pl.DataFrame) -> None:
    """Reject null, NaN, or non-finite values in the ``prediction`` column.

    Args:
        frame: Structurally validated prediction DataFrame.

    Raises:
        SignalValidationError: If any ``prediction`` value is null, NaN, or
            infinite.
    """
    invalid_count = int(
        frame.select(
            (
                pl.col("prediction").is_null()
                | pl.col("prediction").is_nan()
                | pl.col("prediction").is_infinite()
            )
            .sum()
            .alias("invalid_count")
        ).item()
    )
    if invalid_count > 0:
        raise SignalValidationError(
            "prediction column contains null, NaN, or non-finite values",
            error_code=_ERROR_PREDICTION_INVALID,
            details={
                "invalid_prediction_rows": invalid_count,
                "row_count": frame.height,
            },
        )


def _require_finite_number(value: object, *, parameter: str) -> float:
    """Require a finite numeric threshold.

    Args:
        value: Candidate threshold.
        parameter: Parameter name used in error messages.

    Returns:
        ``value`` as ``float``.

    Raises:
        SignalValidationError: If ``value`` is non-numeric or non-finite.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SignalValidationError(
            f"{parameter} must be a number",
            error_code=_ERROR_THRESHOLD_TYPE,
            details={
                "parameter": parameter,
                "value_type": type(value).__name__,
            },
        )
    number = float(value)
    if not math.isfinite(number):
        raise SignalValidationError(
            f"{parameter} must be a finite number",
            error_code=_ERROR_THRESHOLD_FINITE,
            details={
                "parameter": parameter,
                "value": number,
            },
        )
    return number


def _require_integer(value: object, *, parameter: str) -> int:
    """Require an integer class label.

    Args:
        value: Candidate class label.
        parameter: Parameter name used in error messages.

    Returns:
        ``value`` as ``int``.

    Raises:
        SignalValidationError: If ``value`` is not an integer.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise SignalValidationError(
            f"{parameter} must be an integer",
            error_code=_ERROR_CLASS_TYPE,
            details={
                "parameter": parameter,
                "value_type": type(value).__name__,
            },
        )
    return value
