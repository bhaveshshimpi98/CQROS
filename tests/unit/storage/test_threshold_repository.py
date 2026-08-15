"""Unit tests for CQROS threshold repository and repository provider."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.core.constants import FILE_EXTENSION_PARQUET, STORAGE_DIR_THRESHOLDS
from cqros.core.types import FilePath
from cqros.signals import (
    AdaptiveRegressionSignalPolicy,
    RegressionThresholds,
    RepositoryThresholdProvider,
    Signal,
    ThresholdProvider,
)
from cqros.storage import (
    DatasetNotFoundError,
    ParquetStore,
    StorageError,
    StorageLayout,
    ThresholdPartitionRef,
    ThresholdRepository,
)
from cqros.storage.threshold_repository import (
    ThresholdRepository as ThresholdRepositoryDirect,
)
from cqros.storage.threshold_schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    THRESHOLD_SCHEMA,
)

_MODEL_NAME = "alpha-lgbm"
_MODEL_VERSION = "1.0.0"
_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1h"
_CREATED_AT = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
_GLOBAL = RegressionThresholds(buy_threshold=0.01, sell_threshold=-0.01)


class _InMemoryDataStore:
    """Minimal ``IDataStore`` stub that records paths and frames."""

    def __init__(self) -> None:
        self.frames: dict[Path, pl.DataFrame] = {}
        self.write_paths: list[Path] = []
        self.read_paths: list[Path] = []
        self.exists_paths: list[Path] = []

    def write(self, path: FilePath, dataframe: pl.DataFrame) -> None:
        target = Path(path)
        self.write_paths.append(target)
        self.frames[target] = dataframe

    def read(self, path: FilePath) -> pl.DataFrame:
        target = Path(path)
        self.read_paths.append(target)
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
        target = Path(path)
        self.exists_paths.append(target)
        return target in self.frames

    def delete(self, path: FilePath) -> None:
        target = Path(path)
        del self.frames[target]

    def schema(self, path: FilePath) -> pl.Schema:
        return self.read(path).schema

    def row_count(self, path: FilePath) -> int:
        return self.read(path).height


def _threshold_frame(
    *,
    profiles: list[str] | None = None,
    buy_thresholds: list[float] | None = None,
    sell_thresholds: list[float] | None = None,
    symbol: str = _SYMBOL,
    timeframe: str = _TIMEFRAME,
    model_name: str = _MODEL_NAME,
    model_version: str = _MODEL_VERSION,
) -> pl.DataFrame:
    """Build a canonical threshold DataFrame for repository tests."""
    profiles = profiles if profiles is not None else ["Conservative", "Balanced", "Active"]
    buy_thresholds = buy_thresholds if buy_thresholds is not None else [0.05, 0.03, 0.02]
    sell_thresholds = sell_thresholds if sell_thresholds is not None else [-0.05, -0.03, -0.02]
    row_count = len(profiles)
    return pl.DataFrame(
        {
            "symbol": [symbol] * row_count,
            "timeframe": [timeframe] * row_count,
            "model_name": [model_name] * row_count,
            "model_version": [model_version] * row_count,
            "buy_threshold": buy_thresholds,
            "sell_threshold": sell_thresholds,
            "profile": profiles,
            "created_at": [_CREATED_AT] * row_count,
        },
        schema=dict(COLUMN_DTYPES),
    )


def _threshold_path(layout: StorageLayout) -> Path:
    """Compose the expected threshold partition path."""
    return layout.threshold_path(_MODEL_NAME, _MODEL_VERSION, _SYMBOL, _TIMEFRAME)


@pytest.fixture
def layout(tmp_path: Path) -> StorageLayout:
    """Return a storage layout rooted at a temporary directory."""
    return StorageLayout(tmp_path)


@pytest.fixture
def datastore() -> _InMemoryDataStore:
    """Return an in-memory datastore stub."""
    return _InMemoryDataStore()


@pytest.fixture
def repository(layout: StorageLayout, datastore: _InMemoryDataStore) -> ThresholdRepository:
    """Return a threshold repository over the in-memory datastore."""
    return ThresholdRepository(layout, datastore)


def test_exported_from_package() -> None:
    """Package export matches the repository module by identity."""
    assert ThresholdRepository is ThresholdRepositoryDirect


def test_save_and_load_round_trip(
    repository: ThresholdRepository,
    layout: StorageLayout,
    datastore: _InMemoryDataStore,
) -> None:
    """save persists a canonical frame that load returns unchanged."""
    frame = _threshold_frame()
    repository.save(
        frame,
        model_name=_MODEL_NAME,
        model_version=_MODEL_VERSION,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
    )
    loaded = repository.load(
        model_name=_MODEL_NAME,
        model_version=_MODEL_VERSION,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
    )
    assert_frame_equal(loaded, frame.select(list(CANONICAL_COLUMN_ORDER)).cast(THRESHOLD_SCHEMA))
    assert datastore.write_paths == [_threshold_path(layout)]
    assert loaded.schema == THRESHOLD_SCHEMA


def test_exists(repository: ThresholdRepository) -> None:
    """exists reflects whether a partition has been saved."""
    assert (
        repository.exists(
            model_name=_MODEL_NAME,
            model_version=_MODEL_VERSION,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
        )
        is False
    )
    repository.save(
        _threshold_frame(),
        model_name=_MODEL_NAME,
        model_version=_MODEL_VERSION,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
    )
    assert (
        repository.exists(
            model_name=_MODEL_NAME,
            model_version=_MODEL_VERSION,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
        )
        is True
    )


def test_load_profile_filter(repository: ThresholdRepository) -> None:
    """load can filter rows by profile."""
    repository.save(
        _threshold_frame(),
        model_name=_MODEL_NAME,
        model_version=_MODEL_VERSION,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
    )
    loaded = repository.load(
        model_name=_MODEL_NAME,
        model_version=_MODEL_VERSION,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        profile="Balanced",
    )
    assert loaded.height == 1
    assert loaded.get_column("profile").to_list() == ["Balanced"]
    assert loaded.get_column("buy_threshold").to_list() == [0.03]


def test_discover_and_filters(tmp_path: Path) -> None:
    """discover returns sorted partitions and honors identity filters."""
    layout = StorageLayout(tmp_path)
    repository = ThresholdRepository(layout, ParquetStore())
    repository.save(
        _threshold_frame(),
        model_name=_MODEL_NAME,
        model_version=_MODEL_VERSION,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
    )
    repository.save(
        _threshold_frame(symbol="ETHUSDT", buy_thresholds=[0.04, 0.03, 0.02]),
        model_name=_MODEL_NAME,
        model_version=_MODEL_VERSION,
        symbol="ETHUSDT",
        timeframe=_TIMEFRAME,
    )
    repository.save(
        _threshold_frame(timeframe="4h", buy_thresholds=[0.06, 0.04, 0.03]),
        model_name=_MODEL_NAME,
        model_version=_MODEL_VERSION,
        symbol=_SYMBOL,
        timeframe="4h",
    )

    all_partitions = repository.discover()
    assert all_partitions == (
        ThresholdPartitionRef(_MODEL_NAME, _MODEL_VERSION, _SYMBOL, _TIMEFRAME),
        ThresholdPartitionRef(_MODEL_NAME, _MODEL_VERSION, _SYMBOL, "4h"),
        ThresholdPartitionRef(_MODEL_NAME, _MODEL_VERSION, "ETHUSDT", _TIMEFRAME),
    )

    filtered = repository.discover(symbols=(_SYMBOL,), timeframes=(_TIMEFRAME,))
    assert filtered == (ThresholdPartitionRef(_MODEL_NAME, _MODEL_VERSION, _SYMBOL, _TIMEFRAME),)

    by_profile = repository.discover(profiles=("Balanced",))
    assert len(by_profile) == 3

    missing_profile = repository.discover(profiles=("MissingProfile",))
    assert missing_profile == ()


def test_duplicates_rejected(repository: ThresholdRepository) -> None:
    """Duplicate primary keys raise StorageError on save."""
    frame = _threshold_frame(
        profiles=["Balanced", "Balanced"],
        buy_thresholds=[0.03, 0.04],
        sell_thresholds=[-0.03, -0.04],
    )
    with pytest.raises(StorageError) as exc_info:
        repository.save(
            frame,
            model_name=_MODEL_NAME,
            model_version=_MODEL_VERSION,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
        )
    assert exc_info.value.error_code == "STORAGE-THR-007"


def test_schema_and_ordering(repository: ThresholdRepository) -> None:
    """Saved frames are reordered and cast to THRESHOLD_SCHEMA."""
    unordered = _threshold_frame().select(
        [
            "created_at",
            "profile",
            "sell_threshold",
            "buy_threshold",
            "model_version",
            "model_name",
            "timeframe",
            "symbol",
        ]
    )
    repository.save(
        unordered,
        model_name=_MODEL_NAME,
        model_version=_MODEL_VERSION,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
    )
    loaded = repository.load(
        model_name=_MODEL_NAME,
        model_version=_MODEL_VERSION,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
    )
    assert loaded.columns == list(CANONICAL_COLUMN_ORDER)
    assert loaded.schema == THRESHOLD_SCHEMA
    for column in CANONICAL_COLUMN_ORDER:
        assert loaded.schema[column] == COLUMN_DTYPES[column]


def test_dtype_mismatch_rejected(repository: ThresholdRepository) -> None:
    """Incorrect dtypes raise StorageError on save."""
    frame = _threshold_frame().with_columns(pl.col("buy_threshold").cast(pl.Float32))
    with pytest.raises(StorageError) as exc_info:
        repository.save(
            frame,
            model_name=_MODEL_NAME,
            model_version=_MODEL_VERSION,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
        )
    assert exc_info.value.error_code == "STORAGE-THR-004"


def test_invalid_threshold_order_rejected(repository: ThresholdRepository) -> None:
    """buy_threshold <= sell_threshold raises StorageError."""
    frame = _threshold_frame(
        profiles=["Balanced"],
        buy_thresholds=[-0.01],
        sell_thresholds=[0.01],
    )
    with pytest.raises(StorageError) as exc_info:
        repository.save(
            frame,
            model_name=_MODEL_NAME,
            model_version=_MODEL_VERSION,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
        )
    assert exc_info.value.error_code == "STORAGE-THR-005"


def test_non_finite_thresholds_rejected(repository: ThresholdRepository) -> None:
    """Non-finite threshold values raise StorageError."""
    frame = _threshold_frame(
        profiles=["Balanced"],
        buy_thresholds=[float("nan")],
        sell_thresholds=[-0.01],
    )
    with pytest.raises(StorageError) as exc_info:
        repository.save(
            frame,
            model_name=_MODEL_NAME,
            model_version=_MODEL_VERSION,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
        )
    assert exc_info.value.error_code == "STORAGE-THR-006"


def test_partition_identity_mismatch_rejected(repository: ThresholdRepository) -> None:
    """Rows that disagree with path identity raise StorageError."""
    frame = _threshold_frame(symbol="ETHUSDT")
    with pytest.raises(StorageError) as exc_info:
        repository.save(
            frame,
            model_name=_MODEL_NAME,
            model_version=_MODEL_VERSION,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
        )
    assert exc_info.value.error_code == "STORAGE-THR-008"


def test_load_missing_raises(repository: ThresholdRepository) -> None:
    """load raises DatasetNotFoundError for missing partitions."""
    with pytest.raises(DatasetNotFoundError):
        repository.load(
            model_name=_MODEL_NAME,
            model_version=_MODEL_VERSION,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
        )


def test_repository_threshold_provider_protocol(
    repository: ThresholdRepository,
) -> None:
    """RepositoryThresholdProvider satisfies ThresholdProvider."""
    provider = RepositoryThresholdProvider(repository, global_thresholds=_GLOBAL)
    assert isinstance(provider, ThresholdProvider)


def test_fallback_when_missing(repository: ThresholdRepository) -> None:
    """Missing calibrated partitions fall back to configured global defaults."""
    provider = RepositoryThresholdProvider(repository, global_thresholds=_GLOBAL)
    result = provider.get_thresholds(_SYMBOL, _TIMEFRAME, _MODEL_NAME, _MODEL_VERSION)
    assert result == _GLOBAL


def test_fallback_when_profile_missing(repository: ThresholdRepository) -> None:
    """Missing profile rows fall back to configured global defaults."""
    repository.save(
        _threshold_frame(profiles=["Conservative"], buy_thresholds=[0.05], sell_thresholds=[-0.05]),
        model_name=_MODEL_NAME,
        model_version=_MODEL_VERSION,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
    )
    provider = RepositoryThresholdProvider(
        repository,
        global_thresholds=_GLOBAL,
        profile="Balanced",
    )
    result = provider.get_thresholds(_SYMBOL, _TIMEFRAME, _MODEL_NAME, _MODEL_VERSION)
    assert result == _GLOBAL


def test_repository_integration_loads_approved_thresholds(
    repository: ThresholdRepository,
) -> None:
    """Provider returns stored Balanced thresholds for a saved partition."""
    repository.save(
        _threshold_frame(),
        model_name=_MODEL_NAME,
        model_version=_MODEL_VERSION,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
    )
    provider = RepositoryThresholdProvider(repository, global_thresholds=_GLOBAL)
    result = provider.get_thresholds(_SYMBOL, _TIMEFRAME, _MODEL_NAME, _MODEL_VERSION)
    assert result == RegressionThresholds(buy_threshold=0.03, sell_threshold=-0.03)


def test_repository_integration_with_adaptive_policy(
    repository: ThresholdRepository,
) -> None:
    """AdaptiveRegressionSignalPolicy consumes repository-backed thresholds."""
    repository.save(
        _threshold_frame(),
        model_name=_MODEL_NAME,
        model_version=_MODEL_VERSION,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
    )
    provider = RepositoryThresholdProvider(repository, global_thresholds=_GLOBAL)
    policy = AdaptiveRegressionSignalPolicy(provider)
    frame = pl.DataFrame(
        {
            "symbol": [_SYMBOL, _SYMBOL, _SYMBOL],
            "timeframe": [_TIMEFRAME, _TIMEFRAME, _TIMEFRAME],
            "open_time": [1, 2, 3],
            "model_name": [_MODEL_NAME, _MODEL_NAME, _MODEL_NAME],
            "model_version": [_MODEL_VERSION, _MODEL_VERSION, _MODEL_VERSION],
            "prediction": [0.04, -0.04, 0.0],
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
    result = policy.generate(frame)
    assert result.get_column("signal").to_list() == [
        Signal.BUY.value,
        Signal.SELL.value,
        Signal.HOLD.value,
    ]


def test_parquet_store_round_trip(tmp_path: Path) -> None:
    """Threshold repository round-trips through a real ParquetStore."""
    layout = StorageLayout(tmp_path)
    repository = ThresholdRepository(layout, ParquetStore())
    frame = _threshold_frame()
    repository.save(
        frame,
        model_name=_MODEL_NAME,
        model_version=_MODEL_VERSION,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
    )
    path = layout.threshold_path(_MODEL_NAME, _MODEL_VERSION, _SYMBOL, _TIMEFRAME)
    assert path.exists()
    assert path.name == f"thresholds{FILE_EXTENSION_PARQUET}"
    assert STORAGE_DIR_THRESHOLDS in path.parts
    loaded = repository.load(
        model_name=_MODEL_NAME,
        model_version=_MODEL_VERSION,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
    )
    assert_frame_equal(loaded, frame.select(list(CANONICAL_COLUMN_ORDER)).cast(THRESHOLD_SCHEMA))
