"""Unit tests for CQROS Signals package ``SignalPipeline``."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.core.types import FilePath
from cqros.signals import (
    Signal,
    SignalPipeline,
    SignalPolicy,
    SignalPolicyRegistry,
    SignalValidationError,
)
from cqros.signals.pipeline import SignalPipeline as SignalPipelineDirect
from cqros.signals.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    MERGED_SIGNAL_SCHEMA,
    PRIMARY_KEY_COLUMNS,
)
from cqros.storage import DatasetNotFoundError, SignalPartitionRef, SignalRepository, StorageLayout

_EXCHANGE = "binance"
_MARKET = "perpetual"
_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1h"
_YEAR = 2026
_MODEL_NAME = "alpha-lgbm"
_MODEL_VERSION = "1.0.0"
_POLICY_NAME = "test_policy"

_PARTITION_REF = SignalPartitionRef(
    exchange=_EXCHANGE,
    market=_MARKET,
    symbol=_SYMBOL,
    timeframe=_TIMEFRAME,
    year=_YEAR,
)


class _RecordingRepository:
    """Minimal signal repository stub that records save calls."""

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
    """Minimal ``IDataStore`` stub for real ``SignalRepository`` wiring."""

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


class _PassthroughPolicy:
    """Test policy that returns a prebuilt signal frame."""

    def __init__(self, frame: pl.DataFrame) -> None:
        self.frame = frame
        self.calls: list[pl.DataFrame] = []

    def generate(self, predictions: pl.DataFrame) -> pl.DataFrame:
        self.calls.append(predictions)
        return self.frame


class _FailingPolicy:
    """Test policy that always raises ``SignalValidationError``."""

    def generate(self, predictions: pl.DataFrame) -> pl.DataFrame:
        raise SignalValidationError(
            "policy refused to generate signals",
            error_code="SIGNAL-POL-TEST",
            details={"rows": predictions.height},
        )


def _prediction_frame(
    *,
    predictions: list[float],
    open_times: list[int] | None = None,
) -> pl.DataFrame:
    """Build a canonical prediction DataFrame for pipeline tests."""
    row_count = len(predictions)
    times = open_times if open_times is not None else list(range(row_count))
    return pl.DataFrame(
        {
            "symbol": [_SYMBOL] * row_count,
            "timeframe": [_TIMEFRAME] * row_count,
            "open_time": times,
            "model_name": [_MODEL_NAME] * row_count,
            "model_version": [_MODEL_VERSION] * row_count,
            "prediction": predictions,
        },
        schema={
            "symbol": pl.String,
            "timeframe": pl.String,
            "open_time": pl.Int64,
            "model_name": pl.String,
            "model_version": pl.String,
            "prediction": pl.Float64,
        },
    )


def _signal_frame(
    *,
    open_times: list[int],
    signals: list[str] | None = None,
) -> pl.DataFrame:
    """Build a policy output frame covering the Signal schema columns."""
    rows = len(open_times)
    signal_values = (
        signals
        if signals is not None
        else [Signal.BUY.value, Signal.SELL.value, Signal.HOLD.value][:rows]
    )
    if len(signal_values) != rows:
        signal_values = [Signal.HOLD.value] * rows
    return pl.DataFrame(
        {
            "symbol": [_SYMBOL] * rows,
            "timeframe": [_TIMEFRAME] * rows,
            "open_time": open_times,
            "model_name": [_MODEL_NAME] * rows,
            "model_version": [_MODEL_VERSION] * rows,
            "signal": signal_values,
        },
        schema={
            "symbol": pl.String,
            "timeframe": pl.String,
            "open_time": pl.Int64,
            "model_name": pl.String,
            "model_version": pl.String,
            "signal": pl.String,
        },
    )


def _make_pipeline(
    policy: SignalPolicy,
    repository: _RecordingRepository | SignalRepository | None = None,
    *,
    policy_name: str = _POLICY_NAME,
) -> tuple[SignalPipeline, _RecordingRepository | SignalRepository]:
    """Build a SignalPipeline with an optional repository stub."""
    repo: _RecordingRepository | SignalRepository
    if repository is None:
        repo = _RecordingRepository()
    else:
        repo = repository
    registry = SignalPolicyRegistry()
    registry.register(policy_name, policy)
    return SignalPipeline(cast(SignalRepository, repo), registry), repo


def test_signal_pipeline_is_exported_from_package() -> None:
    """Package export matches the pipeline module class."""
    assert SignalPipeline is SignalPipelineDirect


def test_successful_run_assembles_canonical_signal_frame() -> None:
    """Valid prediction and policy output produce a finalized signal frame."""
    open_times = [0, 1, 2]
    signals = [Signal.BUY.value, Signal.SELL.value, Signal.HOLD.value]
    policy = _PassthroughPolicy(_signal_frame(open_times=open_times, signals=signals))
    pipeline, repository = _make_pipeline(policy)
    predictions = _prediction_frame(predictions=[0.2, -0.1, 0.05], open_times=open_times)

    result = pipeline.run(_POLICY_NAME, predictions, _PARTITION_REF)

    assert isinstance(repository, _RecordingRepository)
    assert result.height == 3
    assert result.columns == list(CANONICAL_COLUMN_ORDER)
    assert result.get_column("open_time").to_list() == open_times
    assert result.get_column("signal").to_list() == signals
    assert_frame_equal(result, repository.saved[0])
    assert len(policy.calls) == 1
    assert policy.calls[0] is predictions


def test_policy_delegation() -> None:
    """Pipeline delegates generation exclusively to SignalPolicy.generate."""
    predictions = _prediction_frame(predictions=[0.1, -0.2])
    signal_frame = _signal_frame(
        open_times=[0, 1],
        signals=[Signal.BUY.value, Signal.SELL.value],
    )
    policy = _PassthroughPolicy(signal_frame)
    pipeline, _repository = _make_pipeline(policy)

    pipeline.run(_POLICY_NAME, predictions, _PARTITION_REF)

    assert len(policy.calls) == 1
    assert_frame_equal(policy.calls[0], predictions)


def test_unknown_policy_raises() -> None:
    """Unknown policy names raise SignalValidationError before generation."""
    policy = _PassthroughPolicy(_signal_frame(open_times=[0], signals=[Signal.BUY.value]))
    pipeline, _repository = _make_pipeline(policy)

    with pytest.raises(SignalValidationError, match="not registered") as exc_info:
        pipeline.run("missing", _prediction_frame(predictions=[0.1]), _PARTITION_REF)

    assert exc_info.value.error_code == "SIGNAL_REG_UNKNOWN"
    assert policy.calls == []


def test_blank_policy_name_raises() -> None:
    """Blank policy names raise SignalValidationError before registry lookup."""
    policy = _PassthroughPolicy(_signal_frame(open_times=[0], signals=[Signal.BUY.value]))
    pipeline, _repository = _make_pipeline(policy)

    with pytest.raises(SignalValidationError, match="non-blank") as exc_info:
        pipeline.run("   ", _prediction_frame(predictions=[0.1]), _PARTITION_REF)

    assert exc_info.value.error_code == "SIGNAL-PIPE-005"
    assert policy.calls == []


def test_schema_ordering_matches_canonical_column_order() -> None:
    """Finalized output columns follow CANONICAL_COLUMN_ORDER exactly."""
    frame = _signal_frame(
        open_times=[0, 1],
        signals=[Signal.BUY.value, Signal.SELL.value],
    )
    frame = frame.with_columns(pl.lit(1.0).alias("extra_noise"))
    pipeline, _repository = _make_pipeline(_PassthroughPolicy(frame))

    result = pipeline.run(
        _POLICY_NAME,
        _prediction_frame(predictions=[0.1, -0.2]),
        _PARTITION_REF,
    )

    assert result.columns == list(CANONICAL_COLUMN_ORDER)
    assert "extra_noise" not in result.columns
    assert "prediction" not in result.columns
    assert "confidence" not in result.columns


def test_dtype_casting_matches_column_dtypes() -> None:
    """Finalized columns are cast to COLUMN_DTYPES / MERGED_SIGNAL_SCHEMA."""
    frame = _signal_frame(
        open_times=[0, 1],
        signals=[Signal.BUY.value, Signal.SELL.value],
    )
    frame = frame.with_columns(pl.col("open_time").cast(pl.Int32))
    pipeline, _repository = _make_pipeline(_PassthroughPolicy(frame))

    result = pipeline.run(
        _POLICY_NAME,
        _prediction_frame(predictions=[0.1, -0.2]),
        _PARTITION_REF,
    )

    assert result.schema == MERGED_SIGNAL_SCHEMA
    for column in CANONICAL_COLUMN_ORDER:
        assert result.schema[column] == COLUMN_DTYPES[column]


def test_repository_save_invoked_with_partition_identity() -> None:
    """Pipeline invokes SignalRepository.save with partition identity."""
    frame = _signal_frame(
        open_times=[0, 1],
        signals=[Signal.BUY.value, Signal.SELL.value],
    )
    pipeline, repository = _make_pipeline(_PassthroughPolicy(frame))

    result = pipeline.run(
        _POLICY_NAME,
        _prediction_frame(predictions=[0.1, -0.2]),
        _PARTITION_REF,
    )

    assert isinstance(repository, _RecordingRepository)
    assert len(repository.saved) == 1
    assert repository.save_kwargs == [
        {
            "exchange": _EXCHANGE,
            "market": _MARKET,
            "symbol": _SYMBOL,
            "timeframe": _TIMEFRAME,
            "year": _YEAR,
        }
    ]
    assert_frame_equal(repository.saved[0], result)


def test_repository_save_with_real_signal_repository() -> None:
    """Pipeline persists through a real SignalRepository + in-memory store."""
    layout = StorageLayout(Path("unused-root"))
    datastore = _InMemoryDataStore()
    repository = SignalRepository(layout, datastore)
    frame = _signal_frame(
        open_times=[0, 1],
        signals=[Signal.BUY.value, Signal.SELL.value],
    )
    pipeline, _repository = _make_pipeline(_PassthroughPolicy(frame), repository)

    result = pipeline.run(
        _POLICY_NAME,
        _prediction_frame(predictions=[0.1, -0.2]),
        _PARTITION_REF,
    )

    expected_path = layout.signal_path(
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    assert expected_path in datastore.frames
    assert_frame_equal(datastore.frames[expected_path], result)


def test_missing_prediction_columns_raises() -> None:
    """Missing prediction-schema columns raise SignalValidationError."""
    frame = _signal_frame(open_times=[0], signals=[Signal.BUY.value])
    pipeline, _repository = _make_pipeline(_PassthroughPolicy(frame))
    predictions = _prediction_frame(predictions=[0.1]).drop("prediction")

    with pytest.raises(
        SignalValidationError,
        match="prediction frame is missing required columns",
    ) as exc_info:
        pipeline.run(_POLICY_NAME, predictions, _PARTITION_REF)

    missing = cast(tuple[str, ...], dict(exc_info.value.details)["missing_columns"])
    assert missing == ("prediction",)
    assert exc_info.value.error_code == "SIGNAL-PIPE-004"


def test_missing_required_signal_columns_raises() -> None:
    """Missing required signal schema columns raise SignalValidationError."""
    frame = _signal_frame(
        open_times=[0, 1],
        signals=[Signal.BUY.value, Signal.SELL.value],
    ).drop("signal")
    pipeline, _repository = _make_pipeline(_PassthroughPolicy(frame))

    with pytest.raises(
        SignalValidationError,
        match="merged signal schema is missing required columns",
    ) as exc_info:
        pipeline.run(
            _POLICY_NAME,
            _prediction_frame(predictions=[0.1, -0.2]),
            _PARTITION_REF,
        )

    missing = cast(tuple[str, ...], dict(exc_info.value.details)["missing_columns"])
    assert missing == ("signal",)
    assert exc_info.value.error_code == "SIGNAL-PIPE-002"


def test_duplicate_primary_keys_raises() -> None:
    """Duplicate primary-key combinations raise SignalValidationError."""
    frame = _signal_frame(
        open_times=[0, 0],
        signals=[Signal.BUY.value, Signal.SELL.value],
    )
    pipeline, _repository = _make_pipeline(_PassthroughPolicy(frame))

    with pytest.raises(
        SignalValidationError,
        match="signal frame contains duplicate primary keys",
    ) as exc_info:
        pipeline.run(
            _POLICY_NAME,
            _prediction_frame(predictions=[0.1, -0.2], open_times=[0, 1]),
            _PARTITION_REF,
        )

    assert exc_info.value.details["primary_key_columns"] == PRIMARY_KEY_COLUMNS
    assert exc_info.value.details["row_count"] == 2
    assert exc_info.value.details["unique_key_count"] == 1


def test_duplicate_prediction_primary_keys_raises() -> None:
    """Duplicate keys on the prediction input raise before policy delegation."""
    policy = _PassthroughPolicy(
        _signal_frame(open_times=[0, 1], signals=[Signal.BUY.value, Signal.SELL.value])
    )
    pipeline, _repository = _make_pipeline(policy)
    predictions = _prediction_frame(predictions=[0.1, -0.2], open_times=[0, 0])

    with pytest.raises(
        SignalValidationError,
        match="prediction frame contains duplicate primary keys",
    ):
        pipeline.run(_POLICY_NAME, predictions, _PARTITION_REF)

    assert policy.calls == []


def test_invalid_prediction_frame_type_raises() -> None:
    """Non-DataFrame prediction inputs raise SignalValidationError."""
    frame = _signal_frame(open_times=[0], signals=[Signal.BUY.value])
    pipeline, _repository = _make_pipeline(_PassthroughPolicy(frame))

    with pytest.raises(
        SignalValidationError,
        match="predictions must be a polars DataFrame",
    ) as exc_info:
        pipeline.run(
            _POLICY_NAME,
            "not-a-frame",  # type: ignore[arg-type]
            _PARTITION_REF,
        )

    assert exc_info.value.error_code == "SIGNAL-PIPE-001"


def test_empty_prediction_frame_raises() -> None:
    """Empty prediction frames raise SignalValidationError."""
    frame = _signal_frame(open_times=[0], signals=[Signal.BUY.value])
    pipeline, _repository = _make_pipeline(_PassthroughPolicy(frame))
    empty = _prediction_frame(predictions=[]).clear()

    with pytest.raises(
        SignalValidationError,
        match="predictions must contain at least one row",
    ) as exc_info:
        pipeline.run(_POLICY_NAME, empty, _PARTITION_REF)

    assert exc_info.value.error_code == "SIGNAL-PIPE-001"


def test_policy_failure_propagates() -> None:
    """SignalValidationError raised by the policy propagates unchanged."""
    pipeline, _repository = _make_pipeline(_FailingPolicy())

    with pytest.raises(
        SignalValidationError,
        match="policy refused to generate signals",
    ) as exc_info:
        pipeline.run(
            _POLICY_NAME,
            _prediction_frame(predictions=[0.1, -0.2]),
            _PARTITION_REF,
        )

    assert exc_info.value.error_code == "SIGNAL-POL-TEST"


def test_prediction_frame_unchanged() -> None:
    """Pipeline never mutates the caller-supplied prediction DataFrame."""
    predictions = _prediction_frame(predictions=[0.1, -0.2, 0.3])
    original = predictions.clone()
    frame = _signal_frame(
        open_times=[0, 1, 2],
        signals=[Signal.BUY.value, Signal.SELL.value, Signal.HOLD.value],
    )
    pipeline, _repository = _make_pipeline(_PassthroughPolicy(frame))

    pipeline.run(_POLICY_NAME, predictions, _PARTITION_REF)

    assert_frame_equal(predictions, original)
    assert "signal" not in predictions.columns


def test_returned_frame_is_new_and_matches_saved_frame() -> None:
    """Returned finalized frame is a new DataFrame identical to the saved one."""
    source = _signal_frame(
        open_times=[0, 1],
        signals=[Signal.BUY.value, Signal.SELL.value],
    )
    pipeline, repository = _make_pipeline(_PassthroughPolicy(source))

    result = pipeline.run(
        _POLICY_NAME,
        _prediction_frame(predictions=[0.1, -0.2]),
        _PARTITION_REF,
    )

    assert isinstance(repository, _RecordingRepository)
    assert result is not source
    assert_frame_equal(result, repository.saved[0])
    assert set(PRIMARY_KEY_COLUMNS).issubset(set(result.columns))
