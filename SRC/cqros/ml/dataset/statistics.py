"""CQROS ML DatasetStatistics.

Purpose:
    Compute deterministic descriptive statistics for canonical ML datasets
    before model training.

Responsibilities:
    - Validate input frames against the ML dataset schema
    - Compute dataset summary, timestamp range, and missing-value counts
    - Compute per-feature and per-label descriptive statistics
    - Return a structured immutable statistics report
    - Never mutate the input frame
    - Remain free of loading, splitting, scaling, and repository access

Dependencies:
    ``polars``, ``cqros.ml.dataset.exceptions``, and ``cqros.ml.dataset.schema``.

Public API:
    ``ClassCount``, ``ClassificationLabelStatistics``, ``DatasetStatistics``,
    ``DatasetStatisticsReport``, ``NumericColumnStatistics``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.ml.dataset.exceptions import DatasetStatisticsError
from cqros.ml.dataset.schema import (
    CLASSIFICATION_LABEL_COLUMNS,
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    REGRESSION_LABEL_COLUMNS,
    REQUIRED_COLUMNS,
)

__all__ = [
    "ClassCount",
    "ClassificationLabelStatistics",
    "DatasetStatistics",
    "DatasetStatisticsReport",
    "NumericColumnStatistics",
]

_logger = logging.getLogger(__name__)

_ERROR_EMPTY_FRAME: Final[str] = "ML-DATASET-STATS-001"
_ERROR_MISSING_COLUMNS: Final[str] = "ML-DATASET-STATS-002"

_COL_OPEN_TIME: Final[str] = "open_time"

# Floating columns inspected for NaN and infinite values.
_FLOATING_VALUE_COLUMNS: Final[tuple[str, ...]] = (
    *FEATURE_COLUMNS,
    *REGRESSION_LABEL_COLUMNS,
)


@dataclass(frozen=True, slots=True)
class NumericColumnStatistics:
    """Immutable descriptive statistics for one numeric column.

    Attributes:
        column: Column name.
        minimum: Minimum finite observed value, or ``None`` when undefined.
        maximum: Maximum finite observed value, or ``None`` when undefined.
        mean: Arithmetic mean, or ``None`` when undefined.
        std: Sample standard deviation, or ``None`` when undefined.
    """

    column: str
    minimum: float | None
    maximum: float | None
    mean: float | None
    std: float | None


@dataclass(frozen=True, slots=True)
class ClassCount:
    """Immutable class distribution entry for one classification label.

    Attributes:
        label: Observed class value.
        count: Number of rows with ``label``.
        percentage: Percentage of total rows occupied by ``label``.
    """

    label: int
    count: int
    percentage: float


@dataclass(frozen=True, slots=True)
class ClassificationLabelStatistics:
    """Immutable class distribution for one classification label column.

    Attributes:
        column: Classification label column name.
        classes: Deterministically ordered class counts and percentages.
    """

    column: str
    classes: tuple[ClassCount, ...]


@dataclass(frozen=True, slots=True)
class DatasetStatisticsReport:
    """Immutable descriptive statistics for one canonical ML dataset.

    Attributes:
        total_rows: Number of rows in the analyzed frame.
        total_columns: Number of columns in the analyzed frame.
        feature_count: Number of canonical feature columns.
        label_count: Number of canonical label columns.
        earliest_open_time: Minimum ``open_time`` value.
        latest_open_time: Maximum ``open_time`` value.
        null_count: Total null cell count across all columns.
        nan_count: Total NaN count across floating feature/label columns.
        infinite_count: Total infinite count across floating feature/label
            columns.
        feature_statistics: Per-feature numeric statistics in canonical order.
        regression_label_statistics: Per-regression-label numeric statistics
            in canonical order.
        classification_label_statistics: Per-classification-label class
            distributions in canonical order.
    """

    total_rows: int
    total_columns: int
    feature_count: int
    label_count: int
    earliest_open_time: int
    latest_open_time: int
    null_count: int
    nan_count: int
    infinite_count: int
    feature_statistics: tuple[NumericColumnStatistics, ...]
    regression_label_statistics: tuple[NumericColumnStatistics, ...]
    classification_label_statistics: tuple[ClassificationLabelStatistics, ...]


class DatasetStatistics:
    """Deterministic descriptive analyzer for canonical ML datasets.

    Computes summary, timestamp range, missing-value counts, and per-column
    statistics without mutating the input frame or accessing repositories.

    Args:
        logger: Optional logger instance. Defaults to the module logger.
    """

    __slots__ = ("_logger",)

    _logger: logging.Logger

    def __init__(self, *, logger: logging.Logger | None = None) -> None:
        """Initialize the statistics analyzer.

        Args:
            logger: Optional logger instance.
        """
        self._logger = logger if logger is not None else _logger

    def analyze(self, frame: pl.DataFrame) -> DatasetStatisticsReport:
        """Analyze ``frame`` and return an immutable statistics report.

        Args:
            frame: Canonical ML dataset. Must not be mutated.

        Returns:
            Structured descriptive statistics for ``frame``.

        Raises:
            DatasetStatisticsError: If ``frame`` is empty or missing required
                columns.
        """
        _validate_frame(frame)

        self._logger.debug(
            "Analyzing ML dataset",
            extra={"rows": frame.height, "columns": frame.width},
        )

        open_times = frame.get_column(_COL_OPEN_TIME)
        earliest = open_times.min()
        latest = open_times.max()
        if earliest is None or latest is None:
            raise DatasetStatisticsError(
                "open_time range is undefined",
                error_code=_ERROR_EMPTY_FRAME,
                details={"rows": frame.height},
            )

        report = DatasetStatisticsReport(
            total_rows=frame.height,
            total_columns=frame.width,
            feature_count=len(FEATURE_COLUMNS),
            label_count=len(LABEL_COLUMNS),
            earliest_open_time=_as_int(earliest),
            latest_open_time=_as_int(latest),
            null_count=_count_nulls(frame),
            nan_count=_count_nans(frame),
            infinite_count=_count_infinites(frame),
            feature_statistics=tuple(
                _numeric_column_statistics(frame, column) for column in FEATURE_COLUMNS
            ),
            regression_label_statistics=tuple(
                _numeric_column_statistics(frame, column) for column in REGRESSION_LABEL_COLUMNS
            ),
            classification_label_statistics=tuple(
                _classification_label_statistics(frame, column)
                for column in CLASSIFICATION_LABEL_COLUMNS
            ),
        )

        self._logger.info(
            "Analyzed ML dataset",
            extra={
                "rows": report.total_rows,
                "columns": report.total_columns,
                "null_count": report.null_count,
                "nan_count": report.nan_count,
                "infinite_count": report.infinite_count,
            },
        )
        return report


def _validate_frame(frame: pl.DataFrame) -> None:
    """Reject empty frames and frames missing required columns.

    Args:
        frame: Candidate ML dataset.

    Raises:
        DatasetStatisticsError: If ``frame`` is empty or incomplete.
    """
    if frame.height == 0:
        raise DatasetStatisticsError(
            "dataset must contain at least one row",
            error_code=_ERROR_EMPTY_FRAME,
            details={"rows": frame.height},
        )

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DatasetStatisticsError(
            "dataset is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "missing_columns": tuple(missing),
                "required_columns": REQUIRED_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )


def _count_nulls(frame: pl.DataFrame) -> int:
    """Return the total null cell count across all columns."""
    if frame.width == 0:
        return 0
    return int(frame.null_count().sum_horizontal().item())


def _count_nans(frame: pl.DataFrame) -> int:
    """Return the total NaN count across floating feature and label columns."""
    return _count_floating_predicate(frame, predicate="is_nan")


def _count_infinites(frame: pl.DataFrame) -> int:
    """Return the total infinite count across floating feature/label columns."""
    return _count_floating_predicate(frame, predicate="is_infinite")


def _count_floating_predicate(frame: pl.DataFrame, *, predicate: str) -> int:
    """Sum True values of a floating-column predicate across the frame."""
    expressions: list[pl.Expr] = []
    for column in _FLOATING_VALUE_COLUMNS:
        series_expr = pl.col(column)
        if predicate == "is_nan":
            expressions.append(series_expr.is_nan().cast(pl.UInt64))
        else:
            expressions.append(series_expr.is_infinite().cast(pl.UInt64))
    if not expressions:
        return 0
    return int(frame.select(pl.sum_horizontal(*expressions).sum()).item())


def _numeric_column_statistics(
    frame: pl.DataFrame,
    column: str,
) -> NumericColumnStatistics:
    """Compute min/max/mean/std for one numeric column."""
    summary = frame.select(
        pl.col(column).min().alias("minimum"),
        pl.col(column).max().alias("maximum"),
        pl.col(column).mean().alias("mean"),
        pl.col(column).std().alias("std"),
    ).row(0)
    return NumericColumnStatistics(
        column=column,
        minimum=_as_optional_float(summary[0]),
        maximum=_as_optional_float(summary[1]),
        mean=_as_optional_float(summary[2]),
        std=_as_optional_float(summary[3]),
    )


def _classification_label_statistics(
    frame: pl.DataFrame,
    column: str,
) -> ClassificationLabelStatistics:
    """Compute class counts and percentages for one classification column."""
    row_count = frame.height
    counts = frame.group_by(column).len().sort(column).rename({column: "label", "len": "count"})
    classes: list[ClassCount] = []
    for label, count in counts.iter_rows():
        if label is None:
            continue
        percentage = 0.0 if row_count == 0 else (float(count) / float(row_count)) * 100.0
        classes.append(
            ClassCount(
                label=int(label),
                count=int(count),
                percentage=percentage,
            )
        )
    return ClassificationLabelStatistics(column=column, classes=tuple(classes))


def _as_optional_float(value: object) -> float | None:
    """Convert a Polars aggregate value to ``float`` or ``None``."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float(str(value))
    return float(value)


def _as_int(value: object) -> int:
    """Convert a Polars aggregate value to ``int``."""
    if isinstance(value, bool) or not isinstance(value, int):
        return int(str(value))
    return int(value)
