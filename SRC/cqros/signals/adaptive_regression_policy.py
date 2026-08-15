"""CQROS adaptive regression signal policy.

Purpose:
    Convert canonical prediction datasets into canonical signal datasets using
    per-partition BUY/SELL thresholds supplied by a ``ThresholdProvider``.

Responsibilities:
    - Resolve thresholds for each prediction partition identity
    - Validate threshold finiteness, non-nullness, and ``buy > sell`` ordering
    - Map continuous predictions to ``Signal`` values using regression rules
    - Preserve primary-key and model metadata columns
    - Return frames matching ``MERGED_SIGNAL_SCHEMA``
    - Remain free of threshold estimation, calibration, persistence, and trading

Dependencies:
    ``polars``, ``math``, ``cqros.predictions.schema``, ``cqros.signals.enums``,
    ``cqros.signals.exceptions``, ``cqros.signals.schema``, and
    ``cqros.signals.threshold_provider``.

Public API:
    ``AdaptiveRegressionSignalPolicy``
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
from cqros.signals.threshold_provider import (
    RegressionThresholds,
    ThresholdProvider,
)

__all__ = ["AdaptiveRegressionSignalPolicy"]

_ERROR_PROVIDER_TYPE: Final[str] = "SIGNAL-ADAPT-001"
_ERROR_THRESHOLDS_TYPE: Final[str] = "SIGNAL-ADAPT-002"
_ERROR_THRESHOLD_NULL: Final[str] = "SIGNAL-ADAPT-003"
_ERROR_THRESHOLD_FINITE: Final[str] = "SIGNAL-ADAPT-004"
_ERROR_THRESHOLD_ORDER: Final[str] = "SIGNAL-ADAPT-005"
_ERROR_PARTITION_KEYS: Final[str] = "SIGNAL-ADAPT-006"
_ERROR_FRAME_TYPE: Final[str] = "SIGNAL-POL-006"
_ERROR_FRAME_EMPTY: Final[str] = "SIGNAL-POL-007"
_ERROR_MISSING_COLUMNS: Final[str] = "SIGNAL-POL-008"
_ERROR_DTYPE_MISMATCH: Final[str] = "SIGNAL-POL-009"
_ERROR_PREDICTION_INVALID: Final[str] = "SIGNAL-POL-010"

_PARTITION_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "model_name",
    "model_version",
)

_OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    *PRIMARY_KEY_COLUMNS,
    *METADATA_COLUMNS,
)

_ROW_INDEX_COLUMN: Final[str] = "__cqros_adaptive_row_idx"


class AdaptiveRegressionSignalPolicy:
    """Map continuous predictions to signals using per-partition thresholds.

    For each unique ``(symbol, timeframe, model_name, model_version)`` group in
    the prediction frame, thresholds are requested from the injected
    ``ThresholdProvider`` and applied with the same rules as
    ``RegressionSignalPolicy``:

    - ``prediction >= buy_threshold`` → ``BUY``
    - ``prediction <= sell_threshold`` → ``SELL``
    - otherwise → ``HOLD``

    Args:
        threshold_provider: Read-only provider supplying partition thresholds.

    Raises:
        SignalValidationError: If ``threshold_provider`` does not implement
            ``ThresholdProvider``.
    """

    __slots__ = ("_threshold_provider",)

    _threshold_provider: ThresholdProvider

    def __init__(self, threshold_provider: ThresholdProvider) -> None:
        """Initialize with an injected threshold provider.

        Args:
            threshold_provider: Read-only provider supplying partition
                thresholds. Must not be mutated by this policy.

        Raises:
            SignalValidationError: If ``threshold_provider`` does not implement
                ``ThresholdProvider``.
        """
        if not isinstance(threshold_provider, ThresholdProvider):
            raise SignalValidationError(
                "threshold_provider must implement ThresholdProvider",
                error_code=_ERROR_PROVIDER_TYPE,
                details={"value_type": type(threshold_provider).__name__},
            )
        self._threshold_provider = threshold_provider

    def generate(self, predictions: pl.DataFrame) -> pl.DataFrame:
        """Convert a prediction frame into a canonical signal frame.

        Args:
            predictions: Canonical prediction dataset. Must not be mutated.

        Returns:
            A new DataFrame matching ``MERGED_SIGNAL_SCHEMA``, ordered to
            ``CANONICAL_COLUMN_ORDER``. Primary-key and model metadata columns
            are preserved unchanged; only ``signal`` is generated. Row order
            matches the input frame.

        Raises:
            SignalValidationError: If ``predictions`` fails structural
                validation, any ``prediction`` value is null/NaN/non-finite,
                partition keys are null, thresholds are missing/invalid, or
                ``buy_threshold <= sell_threshold``.
        """
        frame = _require_prediction_frame(predictions)
        _require_finite_predictions(frame)
        _require_non_null_partition_keys(frame)

        partitions = frame.select(list(_PARTITION_KEY_COLUMNS)).unique(maintain_order=True)
        if partitions.height == 1:
            thresholds = _resolve_thresholds(
                self._threshold_provider,
                partitions.row(0, named=True),
            )
            return _build_signal_frame(
                frame,
                buy_threshold=thresholds.buy_threshold,
                sell_threshold=thresholds.sell_threshold,
            )

        indexed = frame.with_row_index(_ROW_INDEX_COLUMN)
        pieces: list[pl.DataFrame] = []
        for partition in partitions.iter_rows(named=True):
            thresholds = _resolve_thresholds(self._threshold_provider, partition)
            subset = indexed.filter(
                (pl.col("symbol") == partition["symbol"])
                & (pl.col("timeframe") == partition["timeframe"])
                & (pl.col("model_name") == partition["model_name"])
                & (pl.col("model_version") == partition["model_version"])
            )
            row_index = subset.get_column(_ROW_INDEX_COLUMN)
            generated = _build_signal_frame(
                subset.drop(_ROW_INDEX_COLUMN),
                buy_threshold=thresholds.buy_threshold,
                sell_threshold=thresholds.sell_threshold,
            )
            pieces.append(generated.with_columns(row_index))

        combined = pl.concat(pieces).sort(_ROW_INDEX_COLUMN).drop(_ROW_INDEX_COLUMN)
        return combined.select(list(CANONICAL_COLUMN_ORDER)).cast(MERGED_SIGNAL_SCHEMA)


def _build_signal_frame(
    frame: pl.DataFrame,
    *,
    buy_threshold: float,
    sell_threshold: float,
) -> pl.DataFrame:
    """Assemble a canonical signal DataFrame using regression threshold rules.

    Args:
        frame: Validated prediction DataFrame for one threshold partition.
        buy_threshold: Inclusive lower bound for ``BUY``.
        sell_threshold: Inclusive upper bound for ``SELL``.

    Returns:
        A new DataFrame ordered and cast to ``MERGED_SIGNAL_SCHEMA``.
    """
    signal_expr = (
        pl.when(pl.col("prediction") >= buy_threshold)
        .then(pl.lit(Signal.BUY.value))
        .when(pl.col("prediction") <= sell_threshold)
        .then(pl.lit(Signal.SELL.value))
        .otherwise(pl.lit(Signal.HOLD.value))
        .alias("signal")
    )
    assembled = frame.select(
        *[pl.col(column) for column in _OUTPUT_COLUMNS],
        signal_expr,
    )
    return assembled.select(list(CANONICAL_COLUMN_ORDER)).cast(MERGED_SIGNAL_SCHEMA)


def _resolve_thresholds(
    provider: ThresholdProvider,
    partition: dict[str, object],
) -> RegressionThresholds:
    """Request and validate thresholds for one partition identity.

    Args:
        provider: Threshold provider.
        partition: Mapping containing partition key columns.

    Returns:
        Validated ``RegressionThresholds``.

    Raises:
        SignalValidationError: If the provider returns an invalid threshold
            object or thresholds fail numeric validation.
    """
    symbol = str(partition["symbol"])
    timeframe = str(partition["timeframe"])
    model_name = str(partition["model_name"])
    model_version = str(partition["model_version"])
    try:
        thresholds = provider.get_thresholds(
            symbol,
            timeframe,
            model_name,
            model_version,
        )
    except SignalValidationError:
        raise
    except Exception as exc:
        raise SignalValidationError(
            "threshold provider failed to supply thresholds",
            error_code=_ERROR_THRESHOLDS_TYPE,
            details={
                "symbol": symbol,
                "timeframe": timeframe,
                "model_name": model_name,
                "model_version": model_version,
                "cause_type": type(exc).__name__,
                "cause": str(exc),
            },
        ) from exc

    return _validate_thresholds(
        thresholds,
        symbol=symbol,
        timeframe=timeframe,
        model_name=model_name,
        model_version=model_version,
    )


def _validate_thresholds(
    thresholds: object,
    *,
    symbol: str,
    timeframe: str,
    model_name: str,
    model_version: str,
) -> RegressionThresholds:
    """Validate provider thresholds before signal generation.

    Args:
        thresholds: Candidate thresholds object from the provider.
        symbol: Partition symbol used in error details.
        timeframe: Partition timeframe used in error details.
        model_name: Partition model name used in error details.
        model_version: Partition model version used in error details.

    Returns:
        Validated ``RegressionThresholds`` with finite ``buy > sell``.

    Raises:
        SignalValidationError: If thresholds are missing, non-finite, null, or
            incorrectly ordered.
    """
    if thresholds is None:
        raise SignalValidationError(
            "threshold provider returned no thresholds",
            error_code=_ERROR_THRESHOLD_NULL,
            details={
                "symbol": symbol,
                "timeframe": timeframe,
                "model_name": model_name,
                "model_version": model_version,
            },
        )
    if not isinstance(thresholds, RegressionThresholds):
        raise SignalValidationError(
            "threshold provider must return RegressionThresholds",
            error_code=_ERROR_THRESHOLDS_TYPE,
            details={
                "symbol": symbol,
                "timeframe": timeframe,
                "model_name": model_name,
                "model_version": model_version,
                "value_type": type(thresholds).__name__,
            },
        )

    buy = _require_finite_number(
        thresholds.buy_threshold,
        parameter="buy_threshold",
        symbol=symbol,
        timeframe=timeframe,
        model_name=model_name,
        model_version=model_version,
    )
    sell = _require_finite_number(
        thresholds.sell_threshold,
        parameter="sell_threshold",
        symbol=symbol,
        timeframe=timeframe,
        model_name=model_name,
        model_version=model_version,
    )
    if buy <= sell:
        raise SignalValidationError(
            "buy_threshold must be greater than sell_threshold",
            error_code=_ERROR_THRESHOLD_ORDER,
            details={
                "symbol": symbol,
                "timeframe": timeframe,
                "model_name": model_name,
                "model_version": model_version,
                "buy_threshold": buy,
                "sell_threshold": sell,
            },
        )
    return RegressionThresholds(buy_threshold=buy, sell_threshold=sell)


def _require_finite_number(
    value: object,
    *,
    parameter: str,
    symbol: str,
    timeframe: str,
    model_name: str,
    model_version: str,
) -> float:
    """Require a finite non-null numeric threshold.

    Args:
        value: Candidate threshold.
        parameter: Parameter name used in error messages.
        symbol: Partition symbol used in error details.
        timeframe: Partition timeframe used in error details.
        model_name: Partition model name used in error details.
        model_version: Partition model version used in error details.

    Returns:
        ``value`` as ``float``.

    Raises:
        SignalValidationError: If ``value`` is null, non-numeric, or non-finite.
    """
    if value is None:
        raise SignalValidationError(
            f"{parameter} must not be null",
            error_code=_ERROR_THRESHOLD_NULL,
            details={
                "parameter": parameter,
                "symbol": symbol,
                "timeframe": timeframe,
                "model_name": model_name,
                "model_version": model_version,
            },
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SignalValidationError(
            f"{parameter} must be a finite number",
            error_code=_ERROR_THRESHOLD_FINITE,
            details={
                "parameter": parameter,
                "value_type": type(value).__name__,
                "symbol": symbol,
                "timeframe": timeframe,
                "model_name": model_name,
                "model_version": model_version,
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
                "symbol": symbol,
                "timeframe": timeframe,
                "model_name": model_name,
                "model_version": model_version,
            },
        )
    return number


def _require_non_null_partition_keys(frame: pl.DataFrame) -> None:
    """Reject null values in partition identity columns.

    Args:
        frame: Prediction DataFrame under evaluation.

    Raises:
        SignalValidationError: If any partition key column contains nulls.
    """
    null_counts = {
        column: int(frame.get_column(column).null_count()) for column in _PARTITION_KEY_COLUMNS
    }
    null_columns = tuple(column for column, count in null_counts.items() if count > 0)
    if null_columns:
        raise SignalValidationError(
            "prediction partition keys contain null values",
            error_code=_ERROR_PARTITION_KEYS,
            details={
                "null_columns": null_columns,
                "null_counts": null_counts,
            },
        )


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
