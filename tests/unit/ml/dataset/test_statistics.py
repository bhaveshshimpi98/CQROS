"""Unit tests for the CQROS ML ``DatasetStatistics`` analyzer."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.ml.dataset import (
    CANONICAL_COLUMN_ORDER,
    CLASSIFICATION_LABEL_COLUMNS,
    COLUMN_DTYPES,
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    MERGED_TRAINING_SCHEMA,
    REGRESSION_LABEL_COLUMNS,
    ClassCount,
    DatasetStatistics,
    DatasetStatisticsError,
    DatasetStatisticsReport,
    NumericColumnStatistics,
)
from cqros.ml.dataset.statistics import DatasetStatistics as DatasetStatisticsDirect

_START = 1_700_000_000_000
_INTERVAL = 3_600_000


def _feature_values(row_count: int, *, value: float = 0.01) -> dict[str, list[float]]:
    """Build default float values for every feature column."""
    values: dict[str, list[float]] = {}
    for column in FEATURE_COLUMNS:
        values[column] = [value + float(index) for index in range(row_count)]
    return values


def _label_values(row_count: int) -> dict[str, list[float] | list[int]]:
    """Build default values for every label column."""
    values: dict[str, list[float] | list[int]] = {}
    for column in LABEL_COLUMNS:
        if column.startswith("direction_"):
            values[column] = [1 if index % 2 == 0 else 0 for index in range(row_count)]
        else:
            values[column] = [0.01 * float(index + 1) for index in range(row_count)]
    return values


def _training_frame(*, row_count: int = 10, symbol: str = "BTCUSDT") -> pl.DataFrame:
    """Build a chronologically ordered canonical ML dataset."""
    open_times = [_START + index * _INTERVAL for index in range(row_count)]
    data: dict[str, object] = {
        "symbol": [symbol] * row_count,
        "timeframe": ["1h"] * row_count,
        "open_time": open_times,
    }
    data.update(_feature_values(row_count))
    data.update(_label_values(row_count))
    frame = pl.DataFrame(data, schema=COLUMN_DTYPES)
    return frame.select(list(CANONICAL_COLUMN_ORDER))


def test_dataset_statistics_is_exported_from_package() -> None:
    """Package export matches the statistics module class."""
    assert DatasetStatistics is DatasetStatisticsDirect


def test_analyze_dataset_summary() -> None:
    """Summary fields report rows, columns, features, and labels."""
    frame = _training_frame(row_count=10)
    report = DatasetStatistics().analyze(frame)

    assert isinstance(report, DatasetStatisticsReport)
    assert report.total_rows == 10
    assert report.total_columns == len(CANONICAL_COLUMN_ORDER)
    assert report.feature_count == len(FEATURE_COLUMNS)
    assert report.label_count == len(LABEL_COLUMNS)


def test_analyze_earliest_and_latest_timestamps() -> None:
    """Timestamp range matches min/max open_time."""
    frame = _training_frame(row_count=5)
    report = DatasetStatistics().analyze(frame)

    assert report.earliest_open_time == _START
    assert report.latest_open_time == _START + 4 * _INTERVAL


def test_analyze_feature_statistics() -> None:
    """Each feature reports min/max/mean/std in canonical order."""
    frame = _training_frame(row_count=4)
    report = DatasetStatistics().analyze(frame)

    assert tuple(item.column for item in report.feature_statistics) == FEATURE_COLUMNS
    returns = report.feature_statistics[0]
    assert returns.column == "returns"
    assert isinstance(returns, NumericColumnStatistics)
    series = frame.get_column("returns")
    assert returns.minimum == pytest.approx(series.min())
    assert returns.maximum == pytest.approx(series.max())
    assert returns.mean == pytest.approx(series.mean())
    assert returns.std is not None
    assert returns.std == pytest.approx(series.std())


def test_analyze_regression_label_statistics() -> None:
    """Each regression label reports min/max/mean/std in canonical order."""
    frame = _training_frame(row_count=4)
    report = DatasetStatistics().analyze(frame)

    assert (
        tuple(item.column for item in report.regression_label_statistics)
        == REGRESSION_LABEL_COLUMNS
    )
    first = report.regression_label_statistics[0]
    series = frame.get_column("future_return_1")
    assert first.column == "future_return_1"
    assert first.minimum == pytest.approx(series.min())
    assert first.maximum == pytest.approx(series.max())
    assert first.mean == pytest.approx(series.mean())
    assert first.std == pytest.approx(series.std())


def test_analyze_classification_label_statistics() -> None:
    """Each classification label reports class counts and percentages."""
    frame = _training_frame(row_count=4)
    report = DatasetStatistics().analyze(frame)

    assert (
        tuple(item.column for item in report.classification_label_statistics)
        == CLASSIFICATION_LABEL_COLUMNS
    )
    direction = report.classification_label_statistics[0]
    assert direction.column == "direction_1"
    assert direction.classes == (
        ClassCount(label=0, count=2, percentage=50.0),
        ClassCount(label=1, count=2, percentage=50.0),
    )


def test_analyze_missing_value_counting() -> None:
    """Null cells across the frame are counted exactly."""
    frame = _training_frame(row_count=3).with_columns(
        pl.when(pl.arange(0, 3) == 1).then(None).otherwise(pl.col("returns")).alias("returns")
    )
    report = DatasetStatistics().analyze(frame)

    assert report.null_count == 1


def test_analyze_nan_counting() -> None:
    """NaN cells in floating feature/label columns are counted exactly."""
    frame = _training_frame(row_count=3).with_columns(
        pl.Series("returns", [0.01, float("nan"), 0.03])
    )
    report = DatasetStatistics().analyze(frame)

    assert report.nan_count == 1
    assert report.null_count == 0


def test_analyze_infinite_value_counting() -> None:
    """Infinite cells in floating feature/label columns are counted exactly."""
    frame = _training_frame(row_count=3).with_columns(
        pl.Series("returns", [0.01, float("inf"), float("-inf")])
    )
    report = DatasetStatistics().analyze(frame)

    assert report.infinite_count == 2
    assert report.nan_count == 0


def test_analyze_rejects_empty_dataset() -> None:
    """Empty frames raise DatasetStatisticsError."""
    empty = pl.DataFrame(schema=MERGED_TRAINING_SCHEMA)
    with pytest.raises(DatasetStatisticsError, match="at least one row"):
        DatasetStatistics().analyze(empty)


def test_analyze_rejects_missing_required_columns() -> None:
    """Frames missing required schema columns are rejected."""
    frame = _training_frame(row_count=3).drop("returns")
    with pytest.raises(DatasetStatisticsError, match="missing required columns"):
        DatasetStatistics().analyze(frame)


def test_analyze_result_is_immutable() -> None:
    """The statistics report and nested collections cannot be mutated."""
    report = DatasetStatistics().analyze(_training_frame(row_count=3))

    with pytest.raises(FrozenInstanceError):
        report.total_rows = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        report.feature_statistics[0].mean = 0.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        report.feature_statistics += ()  # type: ignore[misc]


def test_analyze_does_not_mutate_input() -> None:
    """The caller-supplied frame remains unchanged after analysis."""
    frame = _training_frame(row_count=5)
    snapshot = frame.clone()

    DatasetStatistics().analyze(frame)

    assert_frame_equal(frame, snapshot)


def test_analyze_clean_frame_has_zero_missing_markers() -> None:
    """A clean canonical frame reports zero null/NaN/infinite counts."""
    report = DatasetStatistics().analyze(_training_frame(row_count=8))

    assert report.null_count == 0
    assert report.nan_count == 0
    assert report.infinite_count == 0
    assert math.isfinite(report.feature_statistics[0].mean or 0.0)
