"""Unit tests for CQROS Predictions package ``PredictionPipeline``."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

from cqros.core.types import FilePath
from cqros.ml.inference.result import PredictionResult
from cqros.ml.models import ModelFramework, ModelMetadata, ModelTaskType
from cqros.predictions import (
    InferencePipeline,
    PredictionPipeline,
    PredictionValidationError,
)
from cqros.predictions.pipeline import PredictionPipeline as PredictionPipelineDirect
from cqros.predictions.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    MERGED_PREDICTION_SCHEMA,
    PRIMARY_KEY_COLUMNS,
)
from cqros.storage import (
    DatasetNotFoundError,
    PredictionPartitionRef,
    PredictionRepository,
    StorageLayout,
)

_FRAMEWORK = "lightgbm"
_MODEL_NAME = "alpha-lgbm"
_MODEL_VERSION = "1.0.0"
_EXCHANGE = "binance"
_MARKET = "perpetual"
_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1h"
_YEAR = 2026

_PARTITION_REF = PredictionPartitionRef(
    exchange=_EXCHANGE,
    market=_MARKET,
    symbol=_SYMBOL,
    timeframe=_TIMEFRAME,
    year=_YEAR,
    model_name=_MODEL_NAME,
    model_version=_MODEL_VERSION,
)


class _RecordingRepository:
    """Minimal prediction repository stub that records save calls."""

    def __init__(self) -> None:
        self.saved: list[pl.DataFrame] = []
        self.save_kwargs: list[dict[str, object]] = []

    def save(
        self,
        dataframe: pl.DataFrame,
        *,
        framework: str,
        model_name: str,
        model_version: str,
        exchange: str,
        market: str,
        symbol: str,
        timeframe: str,
        year: int,
    ) -> None:
        self.saved.append(dataframe)
        self.save_kwargs.append(
            {
                "framework": framework,
                "model_name": model_name,
                "model_version": model_version,
                "exchange": exchange,
                "market": market,
                "symbol": symbol,
                "timeframe": timeframe,
                "year": year,
            }
        )


class _InMemoryDataStore:
    """Minimal ``IDataStore`` stub for real ``PredictionRepository`` wiring."""

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


class _StubInferencePipeline:
    """Test inference pipeline that returns a prebuilt PredictionResult."""

    def __init__(self, result: PredictionResult) -> None:
        self.result = result
        self.calls: list[tuple[str, pl.DataFrame]] = []

    def predict(self, model_name: str, frame: pl.DataFrame) -> PredictionResult:
        self.calls.append((model_name, frame))
        return self.result


class _InvalidInferencePipeline:
    """Test inference pipeline that returns a non-PredictionResult value."""

    def predict(self, model_name: str, frame: pl.DataFrame) -> object:
        return "not-a-result"


def _metadata() -> ModelMetadata:
    """Build ModelMetadata for prediction pipeline unit tests."""
    return ModelMetadata(
        name=_MODEL_NAME,
        version=_MODEL_VERSION,
        framework=ModelFramework.LIGHTGBM,
        task_type=ModelTaskType.REGRESSION,
        feature_columns=("f1", "f2"),
        label_column="label",
        description="prediction pipeline test model",
    )


def _prediction_result(*, values: list[float]) -> PredictionResult:
    """Build a valid PredictionResult for the given prediction values."""
    predictions = pl.Series("prediction", values, dtype=pl.Float64)
    return PredictionResult(
        model_metadata=_metadata(),
        prediction_count=len(values),
        prediction_time=0.01,
        predictions=predictions,
    )


def _feature_frame(*, open_times: list[int]) -> pl.DataFrame:
    """Build a feature frame covering primary keys and model features."""
    rows = len(open_times)
    return pl.DataFrame(
        {
            "symbol": [_SYMBOL] * rows,
            "timeframe": [_TIMEFRAME] * rows,
            "open_time": open_times,
            "f1": [float(index) for index in range(rows)],
            "f2": [float(index) * 0.5 for index in range(rows)],
        }
    )


def test_prediction_pipeline_is_exported_from_package() -> None:
    """Package export matches the pipeline module class."""
    assert PredictionPipeline is PredictionPipelineDirect


def test_inference_pipeline_protocol_is_exported() -> None:
    """InferencePipeline protocol is exported and runtime-checkable."""
    stub = _StubInferencePipeline(_prediction_result(values=[0.1]))
    assert isinstance(stub, InferencePipeline)


def test_successful_run_assembles_canonical_prediction_frame() -> None:
    """Valid inference output produces a finalized prediction frame."""
    open_times = [0, 1, 2]
    predictions = [0.2, -0.1, 0.05]
    feature_frame = _feature_frame(open_times=open_times)
    inference = _StubInferencePipeline(_prediction_result(values=predictions))
    repository = _RecordingRepository()
    pipeline = PredictionPipeline(inference, repository)

    result = pipeline.run(
        _MODEL_NAME,
        _MODEL_VERSION,
        feature_frame,
        _PARTITION_REF,
    )

    assert result.height == 3
    assert result.columns == list(CANONICAL_COLUMN_ORDER)
    assert result.get_column("open_time").to_list() == open_times
    assert result.get_column("model_name").to_list() == [_MODEL_NAME] * 3
    assert result.get_column("model_version").to_list() == [_MODEL_VERSION] * 3
    assert result.get_column("prediction").to_list() == predictions
    assert_frame_equal(result, repository.saved[0])
    assert len(inference.calls) == 1
    assert inference.calls[0][0] == _MODEL_NAME
    assert inference.calls[0][1] is feature_frame


def test_schema_ordering_matches_canonical_column_order() -> None:
    """Finalized output columns follow CANONICAL_COLUMN_ORDER exactly."""
    feature_frame = _feature_frame(open_times=[0, 1]).with_columns(pl.lit(1.0).alias("extra_noise"))
    inference = _StubInferencePipeline(_prediction_result(values=[0.1, -0.2]))
    pipeline = PredictionPipeline(inference, _RecordingRepository())

    result = pipeline.run(
        _MODEL_NAME,
        _MODEL_VERSION,
        feature_frame,
        _PARTITION_REF,
    )

    assert result.columns == list(CANONICAL_COLUMN_ORDER)
    assert "extra_noise" not in result.columns
    assert "f1" not in result.columns


def test_dtype_casting_matches_column_dtypes() -> None:
    """Finalized columns are cast to COLUMN_DTYPES / MERGED_PREDICTION_SCHEMA."""
    feature_frame = _feature_frame(open_times=[0, 1]).with_columns(
        pl.col("open_time").cast(pl.Int32),
    )
    inference = _StubInferencePipeline(_prediction_result(values=[0.1, -0.2]))
    pipeline = PredictionPipeline(inference, _RecordingRepository())

    result = pipeline.run(
        _MODEL_NAME,
        _MODEL_VERSION,
        feature_frame,
        _PARTITION_REF,
    )

    assert result.schema == MERGED_PREDICTION_SCHEMA
    for column in CANONICAL_COLUMN_ORDER:
        assert result.schema[column] == COLUMN_DTYPES[column]


def test_repository_save_invoked_with_partition_identity() -> None:
    """Pipeline invokes PredictionRepository.save with partition identity."""
    repository = _RecordingRepository()
    feature_frame = _feature_frame(open_times=[0, 1])
    inference = _StubInferencePipeline(_prediction_result(values=[0.1, -0.2]))
    pipeline = PredictionPipeline(inference, repository)

    result = pipeline.run(
        _MODEL_NAME,
        _MODEL_VERSION,
        feature_frame,
        _PARTITION_REF,
    )

    assert len(repository.saved) == 1
    assert repository.save_kwargs == [
        {
            "framework": _FRAMEWORK,
            "model_name": _MODEL_NAME,
            "model_version": _MODEL_VERSION,
            "exchange": _EXCHANGE,
            "market": _MARKET,
            "symbol": _SYMBOL,
            "timeframe": _TIMEFRAME,
            "year": _YEAR,
        }
    ]
    assert_frame_equal(repository.saved[0], result)


def test_repository_save_with_real_prediction_repository() -> None:
    """Pipeline persists through a real PredictionRepository + in-memory store."""
    layout = StorageLayout(Path("unused-root"))
    datastore = _InMemoryDataStore()
    repository = PredictionRepository(layout, datastore)
    feature_frame = _feature_frame(open_times=[0, 1])
    inference = _StubInferencePipeline(_prediction_result(values=[0.1, -0.2]))
    pipeline = PredictionPipeline(inference, repository)

    result = pipeline.run(
        _MODEL_NAME,
        _MODEL_VERSION,
        feature_frame,
        _PARTITION_REF,
    )

    expected_path = layout.prediction_path(
        _FRAMEWORK,
        _MODEL_NAME,
        _MODEL_VERSION,
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    assert expected_path in datastore.frames
    assert_frame_equal(datastore.frames[expected_path], result)


def test_missing_primary_key_columns_raises() -> None:
    """Missing required primary-key columns raise PredictionValidationError."""
    feature_frame = _feature_frame(open_times=[0, 1]).drop("open_time")
    inference = _StubInferencePipeline(_prediction_result(values=[0.1, -0.2]))
    pipeline = PredictionPipeline(inference, _RecordingRepository())

    with pytest.raises(
        PredictionValidationError,
        match="feature_frame is missing required primary-key columns",
    ) as exc_info:
        pipeline.run(
            _MODEL_NAME,
            _MODEL_VERSION,
            feature_frame,
            _PARTITION_REF,
        )

    assert exc_info.value.details["missing_columns"] == ("open_time",)
    assert inference.calls == []


def test_duplicate_primary_keys_raises() -> None:
    """Duplicate primary-key combinations raise PredictionValidationError."""
    feature_frame = _feature_frame(open_times=[0, 0])
    inference = _StubInferencePipeline(_prediction_result(values=[0.1, -0.2]))
    pipeline = PredictionPipeline(inference, _RecordingRepository())

    with pytest.raises(
        PredictionValidationError,
        match="prediction frame contains duplicate primary keys",
    ) as exc_info:
        pipeline.run(
            _MODEL_NAME,
            _MODEL_VERSION,
            feature_frame,
            _PARTITION_REF,
        )

    assert exc_info.value.details["primary_key_columns"] == PRIMARY_KEY_COLUMNS
    assert exc_info.value.details["row_count"] == 2
    assert exc_info.value.details["unique_key_count"] == 1
    assert inference.calls == []


def test_invalid_prediction_result_type_raises() -> None:
    """Non-PredictionResult inference outputs raise PredictionValidationError."""
    feature_frame = _feature_frame(open_times=[0])
    pipeline = PredictionPipeline(
        _InvalidInferencePipeline(),  # type: ignore[arg-type]
        _RecordingRepository(),
    )

    with pytest.raises(
        PredictionValidationError,
        match="prediction_result must be a PredictionResult instance",
    ):
        pipeline.run(
            _MODEL_NAME,
            _MODEL_VERSION,
            feature_frame,
            _PARTITION_REF,
        )


def test_mismatched_prediction_count_raises() -> None:
    """Prediction count mismatches raise PredictionValidationError."""
    feature_frame = _feature_frame(open_times=[0])
    invalid = PredictionResult(
        model_metadata=_metadata(),
        prediction_count=2,
        prediction_time=0.01,
        predictions=pl.Series("prediction", [0.1], dtype=pl.Float64),
    )
    pipeline = PredictionPipeline(
        _StubInferencePipeline(invalid),
        _RecordingRepository(),
    )

    with pytest.raises(
        PredictionValidationError,
        match="prediction_count does not match predictions length",
    ):
        pipeline.run(
            _MODEL_NAME,
            _MODEL_VERSION,
            feature_frame,
            _PARTITION_REF,
        )


def test_prediction_length_mismatch_raises() -> None:
    """Prediction series length mismatches raise PredictionValidationError."""
    feature_frame = _feature_frame(open_times=[0, 1])
    pipeline = PredictionPipeline(
        _StubInferencePipeline(_prediction_result(values=[0.1])),
        _RecordingRepository(),
    )

    with pytest.raises(
        PredictionValidationError,
        match="prediction_result length does not match feature_frame row count",
    ):
        pipeline.run(
            _MODEL_NAME,
            _MODEL_VERSION,
            feature_frame,
            _PARTITION_REF,
        )


def test_feature_frame_unchanged() -> None:
    """Pipeline never mutates the caller-supplied feature frame."""
    feature_frame = _feature_frame(open_times=[0, 1, 2])
    original = feature_frame.clone()
    inference = _StubInferencePipeline(
        _prediction_result(values=[0.1, -0.2, 0.3]),
    )
    pipeline = PredictionPipeline(inference, _RecordingRepository())

    pipeline.run(
        _MODEL_NAME,
        _MODEL_VERSION,
        feature_frame,
        _PARTITION_REF,
    )

    assert_frame_equal(feature_frame, original)


def test_prediction_result_unchanged() -> None:
    """Pipeline never mutates the inference PredictionResult."""
    predictions = [0.1, -0.2, 0.3]
    prediction_result = _prediction_result(values=predictions)
    original_series = prediction_result.predictions.clone()
    feature_frame = _feature_frame(open_times=[0, 1, 2])
    pipeline = PredictionPipeline(
        _StubInferencePipeline(prediction_result),
        _RecordingRepository(),
    )

    pipeline.run(
        _MODEL_NAME,
        _MODEL_VERSION,
        feature_frame,
        _PARTITION_REF,
    )

    assert_series_equal(prediction_result.predictions, original_series)
    assert prediction_result.prediction_count == 3
    assert prediction_result.prediction_time == 0.01


def test_returned_frame_is_new_and_matches_saved_frame() -> None:
    """Returned finalized frame is a new DataFrame identical to the saved one."""
    feature_frame = _feature_frame(open_times=[0, 1])
    repository = _RecordingRepository()
    pipeline = PredictionPipeline(
        _StubInferencePipeline(_prediction_result(values=[0.1, -0.2])),
        repository,
    )

    result = pipeline.run(
        _MODEL_NAME,
        _MODEL_VERSION,
        feature_frame,
        _PARTITION_REF,
    )

    assert result is not feature_frame
    assert_frame_equal(result, repository.saved[0])
    assert set(PRIMARY_KEY_COLUMNS).issubset(set(result.columns))
