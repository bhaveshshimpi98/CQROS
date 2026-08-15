"""Unit tests for CQROS Training package ``TrainingPipeline``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.core.types import FilePath
from cqros.storage import DatasetNotFoundError, StorageLayout, TrainingRepository
from cqros.training import TrainingPipeline, TrainingValidationError
from cqros.training.pipeline import TrainingPipeline as TrainingPipelineDirect
from cqros.training.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    MERGED_TRAINING_SCHEMA,
    PRIMARY_KEY_COLUMNS,
)

_EXCHANGE = "binance"
_MARKET = "perpetual"
_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1h"
_YEAR = 2026
_PARTITION_KWARGS: dict[str, Any] = {
    "exchange": _EXCHANGE,
    "market": _MARKET,
    "symbol": _SYMBOL,
    "timeframe": _TIMEFRAME,
    "year": _YEAR,
}


class _RecordingRepository:
    """Minimal training repository stub that records save calls."""

    def __init__(self) -> None:
        self.saved: list[pl.DataFrame] = []
        self.save_kwargs: list[dict[str, object]] = []

    def save(
        self,
        dataframe: pl.DataFrame,
        *,
        exchange: str,
        market: str,
        symbol: str,
        timeframe: str,
        year: int,
    ) -> None:
        self.saved.append(dataframe)
        self.save_kwargs.append(
            {
                "exchange": exchange,
                "market": market,
                "symbol": symbol,
                "timeframe": timeframe,
                "year": year,
            }
        )


class _InMemoryDataStore:
    """Minimal ``IDataStore`` stub for real ``TrainingRepository`` wiring."""

    def __init__(self) -> None:
        self.frames: dict[Path, pl.DataFrame] = {}

    def write(self, path: FilePath, dataframe: pl.DataFrame) -> None:
        self.frames[Path(path)] = dataframe

    def read(self, path: FilePath) -> pl.DataFrame:
        target = Path(path)
        try:
            return self.frames[target]
        except KeyError as exc:
            raise DatasetNotFoundError(
                "Dataset not found",
                error_code="STORAGE-TEST-001",
                details={"path": str(target)},
            ) from exc

    def scan(self, path: FilePath) -> pl.LazyFrame:
        return self.read(path).lazy()

    def exists(self, path: FilePath) -> bool:
        return Path(path) in self.frames

    def delete(self, path: FilePath) -> None:
        del self.frames[Path(path)]

    def schema(self, path: FilePath) -> pl.Schema:
        return self.read(path).schema

    def row_count(self, path: FilePath) -> int:
        return self.read(path).height


def _features_frame(*, open_times: list[int]) -> pl.DataFrame:
    """Build a feature input frame covering FEATURE_COLUMNS."""
    rows = len(open_times)
    data: dict[str, list[object]] = {
        "symbol": [_SYMBOL] * rows,
        "timeframe": [_TIMEFRAME] * rows,
        "open_time": open_times,
    }
    for index, column in enumerate(FEATURE_COLUMNS):
        data[column] = [float(index + offset) for offset in range(rows)]
    return pl.DataFrame(data)


def _labels_frame(*, open_times: list[int]) -> pl.DataFrame:
    """Build a label input frame covering LABEL_COLUMNS."""
    rows = len(open_times)
    data: dict[str, list[object]] = {
        "symbol": [_SYMBOL] * rows,
        "timeframe": [_TIMEFRAME] * rows,
        "open_time": open_times,
    }
    for index, column in enumerate(LABEL_COLUMNS):
        if column.startswith("direction_"):
            data[column] = [1 if offset % 2 == 0 else 0 for offset in range(rows)]
        else:
            data[column] = [0.01 * float(index + offset) for offset in range(rows)]
    return pl.DataFrame(data)


def test_training_pipeline_is_exported_from_package() -> None:
    """Package export matches the pipeline module class."""
    assert TrainingPipeline is TrainingPipelineDirect


def test_successful_join_assembles_canonical_training_frame() -> None:
    """Matching feature and label rows produce a finalized training frame."""
    open_times = [0, 1, 2]
    features = _features_frame(open_times=open_times)
    labels = _labels_frame(open_times=open_times)
    repository = _RecordingRepository()
    pipeline = TrainingPipeline(repository)

    result = pipeline.run(features, labels, **_PARTITION_KWARGS)

    assert result.height == 3
    assert result.columns == list(CANONICAL_COLUMN_ORDER)
    assert result.get_column("open_time").to_list() == open_times
    assert_frame_equal(result, repository.saved[0])


def test_inner_join_removes_unmatched_rows() -> None:
    """Rows present on only one side are discarded by the inner join."""
    features = _features_frame(open_times=[0, 1, 2, 3])
    labels = _labels_frame(open_times=[1, 2, 4])
    pipeline = TrainingPipeline(_RecordingRepository())

    result = pipeline.run(features, labels, **_PARTITION_KWARGS)

    assert result.get_column("open_time").to_list() == [1, 2]
    assert result.height == 2


def test_empty_join_result_returns_empty_canonical_frame() -> None:
    """Disjoint keys yield an empty frame with the canonical schema."""
    features = _features_frame(open_times=[0, 1])
    labels = _labels_frame(open_times=[10, 11])
    repository = _RecordingRepository()
    pipeline = TrainingPipeline(repository)

    result = pipeline.run(features, labels, **_PARTITION_KWARGS)

    assert result.height == 0
    assert result.columns == list(CANONICAL_COLUMN_ORDER)
    assert result.schema == MERGED_TRAINING_SCHEMA
    assert_frame_equal(repository.saved[0], result)


def test_schema_ordering_matches_canonical_column_order() -> None:
    """Finalized output columns follow CANONICAL_COLUMN_ORDER exactly."""
    features = _features_frame(open_times=[0, 1])
    labels = _labels_frame(open_times=[0, 1])
    # Extra non-canonical columns must be dropped during finalization.
    features = features.with_columns(pl.lit(1.0).alias("extra_feature_noise"))
    labels = labels.with_columns(pl.lit(2.0).alias("extra_label_noise"))
    pipeline = TrainingPipeline(_RecordingRepository())

    result = pipeline.run(features, labels, **_PARTITION_KWARGS)

    assert result.columns == list(CANONICAL_COLUMN_ORDER)
    assert "extra_feature_noise" not in result.columns
    assert "extra_label_noise" not in result.columns


def test_dtype_casting_matches_column_dtypes() -> None:
    """Finalized columns are cast to COLUMN_DTYPES / MERGED_TRAINING_SCHEMA."""
    features = _features_frame(open_times=[0, 1])
    labels = _labels_frame(open_times=[0, 1])
    features = features.with_columns(pl.col("open_time").cast(pl.Int32))
    labels = labels.with_columns(pl.col("open_time").cast(pl.Int32))
    pipeline = TrainingPipeline(_RecordingRepository())

    result = pipeline.run(features, labels, **_PARTITION_KWARGS)

    assert result.schema == MERGED_TRAINING_SCHEMA
    for column in CANONICAL_COLUMN_ORDER:
        assert result.schema[column] == COLUMN_DTYPES[column]


def test_repository_save_invoked_with_partition_identity() -> None:
    """Pipeline invokes TrainingRepository.save with partition identity."""
    repository = _RecordingRepository()
    pipeline = TrainingPipeline(repository)
    features = _features_frame(open_times=[0, 1])
    labels = _labels_frame(open_times=[0, 1])

    result = pipeline.run(features, labels, **_PARTITION_KWARGS)

    assert len(repository.saved) == 1
    assert repository.save_kwargs == [_PARTITION_KWARGS]
    assert_frame_equal(repository.saved[0], result)


def test_repository_save_with_real_training_repository() -> None:
    """Pipeline persists through a real TrainingRepository + in-memory store."""
    layout = StorageLayout(Path("unused-root"))
    datastore = _InMemoryDataStore()
    repository = TrainingRepository(layout, datastore)
    pipeline = TrainingPipeline(repository)
    features = _features_frame(open_times=[0, 1])
    labels = _labels_frame(open_times=[0, 1])

    result = pipeline.run(features, labels, **_PARTITION_KWARGS)

    expected_path = layout.training_path(
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    assert expected_path in datastore.frames
    assert_frame_equal(datastore.frames[expected_path], result)


def test_missing_primary_key_on_features_raises() -> None:
    """Missing primary-key columns on features raise TrainingValidationError."""
    features = _features_frame(open_times=[0]).drop("open_time")
    labels = _labels_frame(open_times=[0])
    pipeline = TrainingPipeline(_RecordingRepository())

    with pytest.raises(
        TrainingValidationError,
        match="features input is missing required primary-key columns",
    ) as exc_info:
        pipeline.run(features, labels, **_PARTITION_KWARGS)

    assert exc_info.value.error_code == "TRAINING-PIPE-001"
    assert exc_info.value.details["side"] == "features"
    assert exc_info.value.details["missing_columns"] == ("open_time",)


def test_missing_primary_key_on_labels_raises() -> None:
    """Missing primary-key columns on labels raise TrainingValidationError."""
    features = _features_frame(open_times=[0])
    labels = _labels_frame(open_times=[0]).drop("symbol")
    pipeline = TrainingPipeline(_RecordingRepository())

    with pytest.raises(
        TrainingValidationError,
        match="labels input is missing required primary-key columns",
    ) as exc_info:
        pipeline.run(features, labels, **_PARTITION_KWARGS)

    assert exc_info.value.error_code == "TRAINING-PIPE-001"
    assert exc_info.value.details["side"] == "labels"
    assert exc_info.value.details["missing_columns"] == ("symbol",)


def test_required_column_validation_after_join() -> None:
    """Missing required feature columns after join raise TrainingValidationError."""
    features = _features_frame(open_times=[0, 1]).drop(FEATURE_COLUMNS[0])
    labels = _labels_frame(open_times=[0, 1])
    pipeline = TrainingPipeline(_RecordingRepository())

    with pytest.raises(
        TrainingValidationError,
        match="merged training schema is missing required columns",
    ) as exc_info:
        pipeline.run(features, labels, **_PARTITION_KWARGS)

    assert exc_info.value.error_code == "TRAINING-PIPE-003"
    assert FEATURE_COLUMNS[0] in exc_info.value.details["missing_columns"]


def test_duplicate_join_keys_on_features_raises() -> None:
    """Duplicate primary keys on features raise TrainingValidationError."""
    features = _features_frame(open_times=[0, 0])
    labels = _labels_frame(open_times=[0])
    pipeline = TrainingPipeline(_RecordingRepository())

    with pytest.raises(
        TrainingValidationError,
        match="features input contains duplicate join keys",
    ) as exc_info:
        pipeline.run(features, labels, **_PARTITION_KWARGS)

    assert exc_info.value.error_code == "TRAINING-PIPE-002"
    assert exc_info.value.details["side"] == "features"


def test_duplicate_join_keys_on_labels_raises() -> None:
    """Duplicate primary keys on labels raise TrainingValidationError."""
    features = _features_frame(open_times=[0])
    labels = _labels_frame(open_times=[1, 1])
    pipeline = TrainingPipeline(_RecordingRepository())

    with pytest.raises(
        TrainingValidationError,
        match="labels input contains duplicate join keys",
    ) as exc_info:
        pipeline.run(features, labels, **_PARTITION_KWARGS)

    assert exc_info.value.error_code == "TRAINING-PIPE-002"
    assert exc_info.value.details["side"] == "labels"


def test_input_frames_unchanged() -> None:
    """Pipeline never mutates the caller-supplied feature or label frames."""
    features = _features_frame(open_times=[0, 1, 2])
    labels = _labels_frame(open_times=[0, 1, 2])
    original_features = features.clone()
    original_labels = labels.clone()
    pipeline = TrainingPipeline(_RecordingRepository())

    pipeline.run(features, labels, **_PARTITION_KWARGS)

    assert_frame_equal(features, original_features)
    assert_frame_equal(labels, original_labels)


def test_returned_frame_is_new_and_matches_saved_frame() -> None:
    """Returned finalized frame is a new DataFrame identical to the saved one."""
    repository = _RecordingRepository()
    pipeline = TrainingPipeline(repository)
    features = _features_frame(open_times=[0, 1])
    labels = _labels_frame(open_times=[0, 1])

    result = pipeline.run(features, labels, **_PARTITION_KWARGS)

    assert result is not features
    assert result is not labels
    assert_frame_equal(result, repository.saved[0])
    assert set(PRIMARY_KEY_COLUMNS).issubset(set(result.columns))
    assert set(FEATURE_COLUMNS).issubset(set(result.columns))
    assert set(LABEL_COLUMNS).issubset(set(result.columns))


def test_no_duplicate_columns_after_join() -> None:
    """Finalized training frame contains unique column names only."""
    features = _features_frame(open_times=[0, 1])
    labels = _labels_frame(open_times=[0, 1])
    pipeline = TrainingPipeline(_RecordingRepository())

    result = pipeline.run(features, labels, **_PARTITION_KWARGS)

    assert len(result.columns) == len(set(result.columns))
    assert len(result.columns) == len(CANONICAL_COLUMN_ORDER)
