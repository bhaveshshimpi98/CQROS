"""Unit tests for CQROS Label Engine ``LabelPipeline``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.core.types import FilePath
from cqros.labels import LabelPipeline, LabelValidationError
from cqros.labels.pipeline import LabelPipeline as LabelPipelineDirect
from cqros.labels.schema import (
    CANONICAL_COLUMN_ORDER,
    CLASSIFICATION_LABEL_COLUMNS,
    COLUMN_DTYPES,
    MERGED_LABEL_SCHEMA,
    REGRESSION_LABEL_COLUMNS,
)
from cqros.storage import DatasetNotFoundError, LabelRepository, StorageLayout

_EXCHANGE = "binance"
_MARKET = "perpetual"
_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1h"
_YEAR = 2026
_MAX_HORIZON = 20
_PARTITION_KWARGS: dict[str, Any] = {
    "exchange": _EXCHANGE,
    "market": _MARKET,
    "symbol": _SYMBOL,
    "timeframe": _TIMEFRAME,
    "year": _YEAR,
}


class _RecordingRepository:
    """Minimal label repository stub that records save calls."""

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
    """Minimal ``IDataStore`` stub for real ``LabelRepository`` wiring."""

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


def _ohlcv_frame(*, closes: list[float], start_open_time: int = 0) -> pl.DataFrame:
    """Build a processed OHLCV input frame from close prices."""
    rows = len(closes)
    return pl.DataFrame(
        {
            "symbol": [_SYMBOL] * rows,
            "timeframe": [_TIMEFRAME] * rows,
            "open_time": list(range(start_open_time, start_open_time + rows)),
            "close": closes,
        }
    )


def _monotonic_closes(*, rows: int) -> list[float]:
    """Return strictly increasing closes for deterministic return checks."""
    return [100.0 + float(index) for index in range(rows)]


def test_label_pipeline_is_exported_from_package() -> None:
    """Package export matches the pipeline module class."""
    assert LabelPipeline is LabelPipelineDirect


def test_future_return_1_calculation() -> None:
    """future_return_1 uses a one-bar forward close ratio."""
    closes = _monotonic_closes(rows=_MAX_HORIZON + 3)
    frame = _ohlcv_frame(closes=closes)
    pipeline = LabelPipeline(_RecordingRepository())
    result = pipeline.run(frame, **_PARTITION_KWARGS)

    expected = [
        (closes[index + 1] - closes[index]) / closes[index] for index in range(result.height)
    ]
    assert result.get_column("future_return_1").to_list() == pytest.approx(expected)


def test_future_return_5_calculation() -> None:
    """future_return_5 uses a five-bar forward close ratio."""
    closes = _monotonic_closes(rows=_MAX_HORIZON + 5)
    frame = _ohlcv_frame(closes=closes)
    pipeline = LabelPipeline(_RecordingRepository())
    result = pipeline.run(frame, **_PARTITION_KWARGS)

    expected = [
        (closes[index + 5] - closes[index]) / closes[index] for index in range(result.height)
    ]
    assert result.get_column("future_return_5").to_list() == pytest.approx(expected)


def test_future_return_10_calculation() -> None:
    """future_return_10 uses a ten-bar forward close ratio."""
    closes = _monotonic_closes(rows=_MAX_HORIZON + 4)
    frame = _ohlcv_frame(closes=closes)
    pipeline = LabelPipeline(_RecordingRepository())
    result = pipeline.run(frame, **_PARTITION_KWARGS)

    expected = [
        (closes[index + 10] - closes[index]) / closes[index] for index in range(result.height)
    ]
    assert result.get_column("future_return_10").to_list() == pytest.approx(expected)


def test_future_return_20_calculation() -> None:
    """future_return_20 uses a twenty-bar forward close ratio."""
    closes = _monotonic_closes(rows=_MAX_HORIZON + 2)
    frame = _ohlcv_frame(closes=closes)
    pipeline = LabelPipeline(_RecordingRepository())
    result = pipeline.run(frame, **_PARTITION_KWARGS)

    expected = [
        (closes[index + 20] - closes[index]) / closes[index] for index in range(result.height)
    ]
    assert result.get_column("future_return_20").to_list() == pytest.approx(expected)


def test_direction_positive_return_is_one() -> None:
    """Positive future returns map to direction label 1."""
    # Rising closes produce positive returns for every horizon.
    frame = _ohlcv_frame(closes=_monotonic_closes(rows=_MAX_HORIZON + 3))
    pipeline = LabelPipeline(_RecordingRepository())
    result = pipeline.run(frame, **_PARTITION_KWARGS)

    for column in CLASSIFICATION_LABEL_COLUMNS:
        assert set(result.get_column(column).to_list()) == {1}


def test_direction_zero_return_is_zero() -> None:
    """Zero future returns map to direction label 0."""
    rows = _MAX_HORIZON + 3
    frame = _ohlcv_frame(closes=[100.0] * rows)
    pipeline = LabelPipeline(_RecordingRepository())
    result = pipeline.run(frame, **_PARTITION_KWARGS)

    assert result.get_column("future_return_1").to_list() == pytest.approx([0.0] * result.height)
    assert result.get_column("direction_1").to_list() == [0] * result.height


def test_direction_negative_return_is_zero() -> None:
    """Negative future returns map to direction label 0."""
    # Falling closes produce negative returns for every horizon.
    closes = [200.0 - float(index) for index in range(_MAX_HORIZON + 3)]
    frame = _ohlcv_frame(closes=closes)
    pipeline = LabelPipeline(_RecordingRepository())
    result = pipeline.run(frame, **_PARTITION_KWARGS)

    for column in CLASSIFICATION_LABEL_COLUMNS:
        assert set(result.get_column(column).to_list()) == {0}
    assert all(value < 0.0 for value in result.get_column("future_return_1").to_list())


def test_trailing_horizon_trim_removes_final_max_horizon_rows() -> None:
    """Final max-horizon rows are removed so no incomplete labels remain."""
    rows = _MAX_HORIZON + 7
    frame = _ohlcv_frame(closes=_monotonic_closes(rows=rows))
    repository = _RecordingRepository()
    pipeline = LabelPipeline(repository)
    result = pipeline.run(frame, **_PARTITION_KWARGS)

    assert result.height == rows - _MAX_HORIZON
    assert result.get_column("open_time").to_list() == list(range(result.height))
    assert repository.saved[0].height == result.height
    for column in REGRESSION_LABEL_COLUMNS:
        assert result.get_column(column).null_count() == 0
    for column in CLASSIFICATION_LABEL_COLUMNS:
        assert result.get_column(column).null_count() == 0


def test_schema_ordering_matches_canonical_column_order() -> None:
    """Finalized output columns follow CANONICAL_COLUMN_ORDER exactly."""
    frame = _ohlcv_frame(closes=_monotonic_closes(rows=_MAX_HORIZON + 2))
    pipeline = LabelPipeline(_RecordingRepository())
    result = pipeline.run(frame, **_PARTITION_KWARGS)

    assert result.columns == list(CANONICAL_COLUMN_ORDER)
    assert "close" not in result.columns


def test_dtype_casting_matches_column_dtypes() -> None:
    """Finalized columns are cast to COLUMN_DTYPES / MERGED_LABEL_SCHEMA."""
    frame = _ohlcv_frame(closes=_monotonic_closes(rows=_MAX_HORIZON + 2))
    # Force open_time as a narrower integer to verify casting.
    frame = frame.with_columns(pl.col("open_time").cast(pl.Int32))
    pipeline = LabelPipeline(_RecordingRepository())
    result = pipeline.run(frame, **_PARTITION_KWARGS)

    assert result.schema == MERGED_LABEL_SCHEMA
    for column in CANONICAL_COLUMN_ORDER:
        assert result.schema[column] == COLUMN_DTYPES[column]


def test_repository_save_invoked_with_partition_identity() -> None:
    """Pipeline invokes LabelRepository.save with partition identity."""
    repository = _RecordingRepository()
    pipeline = LabelPipeline(repository)
    frame = _ohlcv_frame(closes=_monotonic_closes(rows=_MAX_HORIZON + 2))
    result = pipeline.run(frame, **_PARTITION_KWARGS)

    assert len(repository.saved) == 1
    assert repository.save_kwargs == [_PARTITION_KWARGS]
    assert_frame_equal(repository.saved[0], result)


def test_repository_save_with_real_label_repository() -> None:
    """Pipeline persists through a real LabelRepository + in-memory store."""
    layout = StorageLayout(Path("unused-root"))
    datastore = _InMemoryDataStore()
    repository = LabelRepository(layout, datastore)
    pipeline = LabelPipeline(repository)
    frame = _ohlcv_frame(closes=_monotonic_closes(rows=_MAX_HORIZON + 2))
    result = pipeline.run(frame, **_PARTITION_KWARGS)

    expected_path = layout.label_path(
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    assert expected_path in datastore.frames
    assert_frame_equal(datastore.frames[expected_path], result)


def test_required_column_validation() -> None:
    """Missing required input columns raise LabelValidationError."""
    frame = pl.DataFrame(
        {
            "symbol": [_SYMBOL],
            "timeframe": [_TIMEFRAME],
            "open_time": [0],
        }
    )
    pipeline = LabelPipeline(_RecordingRepository())

    with pytest.raises(LabelValidationError, match="missing required columns") as exc_info:
        pipeline.run(frame, **_PARTITION_KWARGS)

    assert exc_info.value.error_code == "LABEL-PIPE-001"
    assert exc_info.value.details["missing_columns"] == ("close",)


def test_input_frame_unchanged() -> None:
    """Pipeline never mutates the caller-supplied input frame."""
    frame = _ohlcv_frame(closes=_monotonic_closes(rows=_MAX_HORIZON + 3))
    original = frame.clone()
    pipeline = LabelPipeline(_RecordingRepository())
    pipeline.run(frame, **_PARTITION_KWARGS)

    assert_frame_equal(frame, original)
    assert frame.columns == ["symbol", "timeframe", "open_time", "close"]


def test_returned_frame_matches_saved_frame() -> None:
    """Returned finalized frame is identical to the persisted frame."""
    repository = _RecordingRepository()
    pipeline = LabelPipeline(repository)
    frame = _ohlcv_frame(closes=_monotonic_closes(rows=_MAX_HORIZON + 4))
    result = pipeline.run(frame, **_PARTITION_KWARGS)

    assert_frame_equal(result, repository.saved[0])
