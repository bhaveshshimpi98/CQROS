"""Unit tests for CQROS prediction dataset repository."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.core.constants import (
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
    STORAGE_DIR_FEATURES,
    STORAGE_DIR_LABELS,
    STORAGE_DIR_MODELS,
    STORAGE_DIR_PREDICTIONS,
    STORAGE_DIR_SIGNALS,
    STORAGE_DIR_TRAINING,
)
from cqros.core.types import FilePath
from cqros.storage import (
    DatasetNotFoundError,
    ParquetStore,
    PredictionPartitionRef,
    PredictionRepository,
    StorageLayout,
)
from cqros.storage.prediction_repository import (
    PredictionRepository as PredictionRepositoryDirect,
)

_FRAMEWORK = "lightgbm"
_MODEL_NAME = "alpha-lgbm"
_MODEL_VERSION = "1.0.0"
_EXCHANGE = "binance"
_MARKET = "perpetual"
_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1h"
_YEAR = 2026


class _InMemoryDataStore:
    """Minimal ``IDataStore`` stub that records paths and frames."""

    def __init__(self) -> None:
        self.frames: dict[Path, pl.DataFrame] = {}
        self.write_paths: list[Path] = []
        self.read_paths: list[Path] = []
        self.exists_paths: list[Path] = []
        self.delete_paths: list[Path] = []
        self.scan_paths: list[Path] = []

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
        target = Path(path)
        self.scan_paths.append(target)
        return self.read(path).lazy()

    def exists(self, path: FilePath) -> bool:
        target = Path(path)
        self.exists_paths.append(target)
        return target in self.frames

    def delete(self, path: FilePath) -> None:
        target = Path(path)
        self.delete_paths.append(target)
        try:
            del self.frames[target]
        except KeyError as exc:
            raise DatasetNotFoundError(
                "Dataset not found",
                error_code="STORAGE-TEST-001",
                details={"path": str(target)},
            ) from exc

    def schema(self, path: FilePath) -> pl.Schema:
        return self.read(path).schema

    def row_count(self, path: FilePath) -> int:
        return self.read(path).height


@pytest.fixture
def sample_frame() -> pl.DataFrame:
    """Return a deterministic sample prediction DataFrame."""
    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "timeframe": ["1h", "1h"],
            "open_time": [1_700_000_000_000, 1_700_000_060_000],
            "model_name": [_MODEL_NAME, _MODEL_NAME],
            "model_version": [_MODEL_VERSION, _MODEL_VERSION],
            "prediction": [0.12, -0.08],
        }
    )


@pytest.fixture
def layout(tmp_path: Path) -> StorageLayout:
    """Return a layout rooted at a temporary directory."""
    return StorageLayout(tmp_path)


@pytest.fixture
def datastore() -> _InMemoryDataStore:
    """Return an in-memory datastore stub."""
    return _InMemoryDataStore()


@pytest.fixture
def repository(
    layout: StorageLayout,
    datastore: _InMemoryDataStore,
) -> PredictionRepository:
    """Return a prediction repository wired to the test layout and datastore."""
    return PredictionRepository(layout, datastore)


def _prediction_path(layout: StorageLayout) -> Path:
    """Compose the canonical sample prediction partition path."""
    return layout.prediction_path(
        _FRAMEWORK,
        _MODEL_NAME,
        _MODEL_VERSION,
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )


def test_prediction_repository_is_exported_from_package() -> None:
    """Package export matches the prediction repository module class."""
    assert PredictionRepository is PredictionRepositoryDirect


def test_prediction_partition_ref_is_frozen_dataclass() -> None:
    """PredictionPartitionRef is an immutable slotted dataclass."""
    ref = PredictionPartitionRef(
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
        model_name=_MODEL_NAME,
        model_version=_MODEL_VERSION,
    )
    assert is_dataclass(ref)
    assert ref.exchange == _EXCHANGE
    assert ref.market == _MARKET
    assert ref.symbol == _SYMBOL
    assert ref.timeframe == _TIMEFRAME
    assert ref.year == _YEAR
    assert ref.model_name == _MODEL_NAME
    assert ref.model_version == _MODEL_VERSION
    assert ref == PredictionPartitionRef(
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
        model_name=_MODEL_NAME,
        model_version=_MODEL_VERSION,
    )
    with pytest.raises(FrozenInstanceError):
        ref.year = 2025  # type: ignore[misc]


def test_save_and_load_uses_prediction_layout_path(
    repository: PredictionRepository,
    layout: StorageLayout,
    datastore: _InMemoryDataStore,
    sample_frame: pl.DataFrame,
) -> None:
    """Prediction save/load uses StorageLayout.prediction_path."""
    expected = _prediction_path(layout)
    assert STORAGE_DIR_PREDICTIONS in expected.parts
    assert STORAGE_DIR_FEATURES not in expected.parts
    assert STORAGE_DIR_LABELS not in expected.parts
    assert STORAGE_DIR_TRAINING not in expected.parts
    assert STORAGE_DIR_SIGNALS not in expected.parts
    assert STORAGE_DIR_MODELS not in expected.parts

    repository.save(
        sample_frame,
        framework=_FRAMEWORK,
        model_name=_MODEL_NAME,
        model_version=_MODEL_VERSION,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    loaded = repository.load(
        framework=_FRAMEWORK,
        model_name=_MODEL_NAME,
        model_version=_MODEL_VERSION,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )

    assert datastore.write_paths == [expected]
    assert datastore.read_paths == [expected]
    assert_frame_equal(loaded, sample_frame)


def test_save_overwrites_existing_partition(
    repository: PredictionRepository,
    sample_frame: pl.DataFrame,
) -> None:
    """Saving the same partition twice replaces the stored frame."""
    replacement = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timeframe": ["1h"],
            "open_time": [1_800_000_000_000],
            "model_name": [_MODEL_NAME],
            "model_version": [_MODEL_VERSION],
            "prediction": [0.05],
        }
    )
    repository.save(
        sample_frame,
        framework=_FRAMEWORK,
        model_name=_MODEL_NAME,
        model_version=_MODEL_VERSION,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    repository.save(
        replacement,
        framework=_FRAMEWORK,
        model_name=_MODEL_NAME,
        model_version=_MODEL_VERSION,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    loaded = repository.load(
        framework=_FRAMEWORK,
        model_name=_MODEL_NAME,
        model_version=_MODEL_VERSION,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert_frame_equal(loaded, replacement)


def test_prediction_path_partitioning_matches_layout_contract(
    layout: StorageLayout,
) -> None:
    """Prediction partitions follow framework/model/version/exchange/.../year."""
    path = _prediction_path(layout)
    assert path.name == f"{_YEAR}.parquet"
    assert path.parent.name == _TIMEFRAME
    assert path.parent.parent.name == _SYMBOL
    assert path.parent.parent.parent.name == _MARKET
    assert path.parent.parent.parent.parent.name == _EXCHANGE
    assert path.parent.parent.parent.parent.parent.name == _MODEL_VERSION
    assert path.parent.parent.parent.parent.parent.parent.name == _MODEL_NAME
    assert path.parent.parent.parent.parent.parent.parent.parent.name == _FRAMEWORK
    assert (
        path.parent.parent.parent.parent.parent.parent.parent.parent.name == STORAGE_DIR_PREDICTIONS
    )
    assert STORAGE_DIR_PREDICTIONS in path.parts


def test_public_api_does_not_return_filesystem_paths(
    repository: PredictionRepository,
    sample_frame: pl.DataFrame,
) -> None:
    """Save returns None and load returns a DataFrame, never a Path."""
    result = repository.save(
        sample_frame,
        framework=_FRAMEWORK,
        model_name=_MODEL_NAME,
        model_version=_MODEL_VERSION,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    loaded = repository.load(
        framework=_FRAMEWORK,
        model_name=_MODEL_NAME,
        model_version=_MODEL_VERSION,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert result is None
    assert isinstance(loaded, pl.DataFrame)
    assert not isinstance(loaded, Path)


def test_load_propagates_datastore_not_found(
    repository: PredictionRepository,
) -> None:
    """Missing datasets surface the datastore ``DatasetNotFoundError``."""
    with pytest.raises(DatasetNotFoundError):
        repository.load(
            framework=_FRAMEWORK,
            model_name=_MODEL_NAME,
            model_version=_MODEL_VERSION,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )


def test_exists_false_when_missing(
    repository: PredictionRepository,
    layout: StorageLayout,
    datastore: _InMemoryDataStore,
) -> None:
    """exists returns False and never reads Parquet contents."""
    assert (
        repository.exists(
            framework=_FRAMEWORK,
            model_name=_MODEL_NAME,
            model_version=_MODEL_VERSION,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )
        is False
    )
    assert datastore.read_paths == []
    assert datastore.exists_paths == [_prediction_path(layout)]


def test_exists_true_when_partition_saved(
    repository: PredictionRepository,
    layout: StorageLayout,
    datastore: _InMemoryDataStore,
    sample_frame: pl.DataFrame,
) -> None:
    """exists returns True after a partition is saved."""
    repository.save(
        sample_frame,
        framework=_FRAMEWORK,
        model_name=_MODEL_NAME,
        model_version=_MODEL_VERSION,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    datastore.read_paths.clear()
    datastore.exists_paths.clear()

    assert (
        repository.exists(
            framework=_FRAMEWORK,
            model_name=_MODEL_NAME,
            model_version=_MODEL_VERSION,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )
        is True
    )
    assert datastore.read_paths == []
    assert _prediction_path(layout) in datastore.exists_paths


def test_delete_removes_partition(
    repository: PredictionRepository,
    layout: StorageLayout,
    datastore: _InMemoryDataStore,
    sample_frame: pl.DataFrame,
) -> None:
    """delete removes a saved partition through the datastore."""
    repository.save(
        sample_frame,
        framework=_FRAMEWORK,
        model_name=_MODEL_NAME,
        model_version=_MODEL_VERSION,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    expected = _prediction_path(layout)

    repository.delete(
        framework=_FRAMEWORK,
        model_name=_MODEL_NAME,
        model_version=_MODEL_VERSION,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )

    assert datastore.delete_paths == [expected]
    assert (
        repository.exists(
            framework=_FRAMEWORK,
            model_name=_MODEL_NAME,
            model_version=_MODEL_VERSION,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )
        is False
    )


def test_delete_missing_propagates_not_found(
    repository: PredictionRepository,
) -> None:
    """delete surfaces DatasetNotFoundError when the partition is absent."""
    with pytest.raises(DatasetNotFoundError):
        repository.delete(
            framework=_FRAMEWORK,
            model_name=_MODEL_NAME,
            model_version=_MODEL_VERSION,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )


def test_discover_models_and_list_versions(tmp_path: Path) -> None:
    """Model and version discovery walks the predictions tree."""
    for model_name, version in (
        ("alpha-lgbm", "1.0.0"),
        ("alpha-lgbm", "1.1.0"),
        ("beta-lgbm", "2.0.0"),
    ):
        path = (
            tmp_path
            / STORAGE_DIR_PREDICTIONS
            / _FRAMEWORK
            / model_name
            / version
            / EXCHANGE_BINANCE
            / MARKET_USDT_PERPETUAL
            / "BTCUSDT"
            / "1h"
        )
        path.mkdir(parents=True, exist_ok=True)
        (path / "2024.parquet").write_bytes(b"")

    repository = PredictionRepository(StorageLayout(tmp_path), ParquetStore())
    assert repository.discover_models(framework=_FRAMEWORK) == (
        "alpha-lgbm",
        "beta-lgbm",
    )
    assert repository.list_versions(framework=_FRAMEWORK, model_name="alpha-lgbm") == (
        "1.0.0",
        "1.1.0",
    )
    assert repository.discover_models(framework="missing") == ()
    assert repository.list_versions(framework=_FRAMEWORK, model_name="missing") == ()


def test_discover_partitions_finds_year_files(tmp_path: Path) -> None:
    """Discovery walks prediction trees without returning filesystem paths."""
    for model_name, version, symbol, year in (
        ("alpha-lgbm", "1.0.0", "BTCUSDT", 2024),
        ("alpha-lgbm", "1.0.0", "ETHUSDT", 2023),
        ("beta-lgbm", "2.0.0", "BTCUSDT", 2025),
    ):
        path = (
            tmp_path
            / STORAGE_DIR_PREDICTIONS
            / _FRAMEWORK
            / model_name
            / version
            / EXCHANGE_BINANCE
            / MARKET_USDT_PERPETUAL
            / symbol
            / "1h"
            / f"{year}.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

    repository = PredictionRepository(StorageLayout(tmp_path), ParquetStore())
    partitions = repository.discover_partitions(framework=_FRAMEWORK)

    assert partitions == (
        PredictionPartitionRef(
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            symbol="BTCUSDT",
            timeframe="1h",
            year=2024,
            model_name="alpha-lgbm",
            model_version="1.0.0",
        ),
        PredictionPartitionRef(
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            symbol="ETHUSDT",
            timeframe="1h",
            year=2023,
            model_name="alpha-lgbm",
            model_version="1.0.0",
        ),
        PredictionPartitionRef(
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            symbol="BTCUSDT",
            timeframe="1h",
            year=2025,
            model_name="beta-lgbm",
            model_version="2.0.0",
        ),
    )


def test_discover_partitions_applies_filters(tmp_path: Path) -> None:
    """Discovery filters by model, version, symbol, and timeframe allowlists."""
    for model_name, version, symbol, timeframe, year in (
        ("alpha-lgbm", "1.0.0", "BTCUSDT", "1h", 2024),
        ("alpha-lgbm", "1.0.0", "BTCUSDT", "4h", 2024),
        ("alpha-lgbm", "1.1.0", "BTCUSDT", "1h", 2024),
        ("beta-lgbm", "2.0.0", "ETHUSDT", "1h", 2024),
    ):
        path = (
            tmp_path
            / STORAGE_DIR_PREDICTIONS
            / _FRAMEWORK
            / model_name
            / version
            / EXCHANGE_BINANCE
            / MARKET_USDT_PERPETUAL
            / symbol
            / timeframe
            / f"{year}.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

    repository = PredictionRepository(StorageLayout(tmp_path), ParquetStore())
    partitions = repository.discover_partitions(
        framework=_FRAMEWORK,
        model_names=("alpha-lgbm",),
        versions=("1.0.0",),
        symbols=("BTCUSDT",),
        timeframes=("1h",),
    )

    assert partitions == (
        PredictionPartitionRef(
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            symbol="BTCUSDT",
            timeframe="1h",
            year=2024,
            model_name="alpha-lgbm",
            model_version="1.0.0",
        ),
    )


def test_round_trip_with_parquet_store(
    layout: StorageLayout,
    sample_frame: pl.DataFrame,
) -> None:
    """Prediction repository round-trips through a real ``ParquetStore``."""
    repository = PredictionRepository(layout, ParquetStore())
    repository.save(
        sample_frame,
        framework=_FRAMEWORK,
        model_name=_MODEL_NAME,
        model_version=_MODEL_VERSION,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    loaded = repository.load(
        framework=_FRAMEWORK,
        model_name=_MODEL_NAME,
        model_version=_MODEL_VERSION,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert_frame_equal(loaded, sample_frame)
    assert _prediction_path(layout).is_file()


def test_prediction_paths_differ_from_other_dataset_paths(layout: StorageLayout) -> None:
    """Prediction partitions resolve to a location distinct from other tiers."""
    prediction = _prediction_path(layout)
    signal = layout.signal_path(_EXCHANGE, _MARKET, _SYMBOL, _TIMEFRAME, _YEAR)
    training = layout.training_path(_EXCHANGE, _MARKET, _SYMBOL, _TIMEFRAME, _YEAR)
    feature = layout.feature_path(_EXCHANGE, _MARKET, _SYMBOL, _TIMEFRAME, _YEAR)
    label = layout.label_path(_EXCHANGE, _MARKET, _SYMBOL, _TIMEFRAME, _YEAR)
    model = layout.model_path(_FRAMEWORK, _MODEL_NAME, _MODEL_VERSION)
    assert prediction != signal
    assert prediction != training
    assert prediction != feature
    assert prediction != label
    assert prediction != model
    assert STORAGE_DIR_PREDICTIONS in prediction.parts
    assert STORAGE_DIR_SIGNALS in signal.parts
    assert STORAGE_DIR_TRAINING in training.parts
    assert STORAGE_DIR_FEATURES in feature.parts
    assert STORAGE_DIR_LABELS in label.parts
    assert STORAGE_DIR_MODELS in model.parts
