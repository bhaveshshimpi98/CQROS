"""Unit tests for CQROS market-data repository."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.core.constants import EXCHANGE_BINANCE, MARKET_USDT_PERPETUAL
from cqros.core.types import FilePath
from cqros.storage import (
    DatasetNotFoundError,
    MarketDataRepository,
    ParquetStore,
    StorageLayout,
)
from cqros.storage.repository import MarketDataRepository as MarketDataRepositoryDirect

_EXCHANGE = "binance"
_MARKET = "perpetual"
_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1m"
_YEAR = 2026
_OHLCV_EXCHANGE = EXCHANGE_BINANCE
_OHLCV_MARKET = MARKET_USDT_PERPETUAL


class _InMemoryDataStore:
    """Minimal ``IDataStore`` stub that records paths and frames."""

    def __init__(self) -> None:
        self.frames: dict[Path, pl.DataFrame] = {}
        self.write_paths: list[Path] = []
        self.read_paths: list[Path] = []
        self.exists_paths: list[Path] = []
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
        del self.frames[Path(path)]

    def schema(self, path: FilePath) -> pl.Schema:
        return self.read(path).schema

    def row_count(self, path: FilePath) -> int:
        return self.read(path).height


@pytest.fixture
def sample_frame() -> pl.DataFrame:
    """Return a deterministic sample DataFrame."""
    return pl.DataFrame(
        {
            "timestamp_ms": [1_700_000_000_000, 1_700_000_060_000],
            "value": [0.01, 0.02],
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
) -> MarketDataRepository:
    """Return a repository wired to the test layout and datastore."""
    return MarketDataRepository(layout, datastore)


def test_market_data_repository_is_exported_from_package() -> None:
    """Package export matches the repository module class."""
    assert MarketDataRepository is MarketDataRepositoryDirect


def test_save_and_load_ohlcv_delegates_to_layout_path(
    repository: MarketDataRepository,
    layout: StorageLayout,
    datastore: _InMemoryDataStore,
    sample_frame: pl.DataFrame,
) -> None:
    """OHLCV save/load uses the canonical layout path without exposing it."""
    expected = layout.raw_ohlcv_path(
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )

    repository.save_ohlcv(
        sample_frame,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    loaded = repository.load_ohlcv(
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )

    assert datastore.write_paths == [expected]
    assert datastore.read_paths == [expected]
    assert_frame_equal(loaded, sample_frame)


def test_save_and_load_funding_delegates_to_layout_path(
    repository: MarketDataRepository,
    layout: StorageLayout,
    datastore: _InMemoryDataStore,
    sample_frame: pl.DataFrame,
) -> None:
    """Funding save/load uses the canonical funding layout path."""
    expected = layout.raw_funding_path(
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )

    repository.save_funding(
        sample_frame,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    loaded = repository.load_funding(
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )

    assert datastore.write_paths == [expected]
    assert datastore.read_paths == [expected]
    assert_frame_equal(loaded, sample_frame)


def test_save_and_load_open_interest_delegates_to_layout_path(
    repository: MarketDataRepository,
    layout: StorageLayout,
    datastore: _InMemoryDataStore,
    sample_frame: pl.DataFrame,
) -> None:
    """Open-interest save/load uses the canonical open-interest layout path."""
    expected = layout.raw_open_interest_path(
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )

    repository.save_open_interest(
        sample_frame,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    loaded = repository.load_open_interest(
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )

    assert datastore.write_paths == [expected]
    assert datastore.read_paths == [expected]
    assert_frame_equal(loaded, sample_frame)


def test_save_and_load_taker_volume_delegates_to_layout_path(
    repository: MarketDataRepository,
    layout: StorageLayout,
    datastore: _InMemoryDataStore,
    sample_frame: pl.DataFrame,
) -> None:
    """Taker-volume save/load uses the canonical taker-volume layout path."""
    expected = layout.raw_taker_volume_path(
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )

    repository.save_taker_volume(
        sample_frame,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    loaded = repository.load_taker_volume(
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )

    assert datastore.write_paths == [expected]
    assert datastore.read_paths == [expected]
    assert_frame_equal(loaded, sample_frame)


@pytest.mark.parametrize(
    ("save_name", "load_name", "path_method"),
    [
        (
            "save_global_long_short_account_ratio",
            "load_global_long_short_account_ratio",
            "raw_global_long_short_account_ratio_path",
        ),
        (
            "save_top_long_short_account_ratio",
            "load_top_long_short_account_ratio",
            "raw_top_long_short_account_ratio_path",
        ),
        (
            "save_top_long_short_position_ratio",
            "load_top_long_short_position_ratio",
            "raw_top_long_short_position_ratio_path",
        ),
    ],
)
def test_save_and_load_long_short_ratio_delegates_to_layout_path(
    repository: MarketDataRepository,
    layout: StorageLayout,
    datastore: _InMemoryDataStore,
    sample_frame: pl.DataFrame,
    save_name: str,
    load_name: str,
    path_method: str,
) -> None:
    """Each long/short ratio dataset uses its own layout namespace."""
    expected = getattr(layout, path_method)(
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    save = getattr(repository, save_name)
    load = getattr(repository, load_name)

    save(
        sample_frame,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    loaded = load(
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )

    assert datastore.write_paths == [expected]
    assert datastore.read_paths == [expected]
    assert_frame_equal(loaded, sample_frame)


def test_save_and_load_liquidations_delegates_to_layout_path(
    repository: MarketDataRepository,
    layout: StorageLayout,
    datastore: _InMemoryDataStore,
    sample_frame: pl.DataFrame,
) -> None:
    """Liquidation save/load uses the canonical liquidation layout path."""
    expected = layout.raw_liquidation_path(
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )

    repository.save_liquidations(
        sample_frame,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    loaded = repository.load_liquidations(
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )

    assert datastore.write_paths == [expected]
    assert datastore.read_paths == [expected]
    assert_frame_equal(loaded, sample_frame)


def test_public_api_does_not_return_filesystem_paths(
    repository: MarketDataRepository,
    sample_frame: pl.DataFrame,
) -> None:
    """Save returns None and load returns a DataFrame, never a Path."""
    result = repository.save_ohlcv(
        sample_frame,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    loaded = repository.load_ohlcv(
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
    repository: MarketDataRepository,
) -> None:
    """Missing datasets surface the datastore ``DatasetNotFoundError``."""
    with pytest.raises(DatasetNotFoundError):
        repository.load_ohlcv(
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )


def test_round_trip_with_parquet_store(
    layout: StorageLayout,
    sample_frame: pl.DataFrame,
) -> None:
    """Repository round-trips through a real ``ParquetStore`` backend."""
    repository = MarketDataRepository(layout, ParquetStore())

    repository.save_ohlcv(
        sample_frame,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    loaded = repository.load_ohlcv(
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )

    assert_frame_equal(loaded, sample_frame)
    assert layout.raw_ohlcv_path(
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    ).is_file()


@pytest.mark.parametrize(
    ("save_name", "load_name", "path_method"),
    [
        ("save_ohlcv", "load_ohlcv", "raw_ohlcv_path"),
        ("save_funding", "load_funding", "raw_funding_path"),
        ("save_open_interest", "load_open_interest", "raw_open_interest_path"),
        ("save_taker_volume", "load_taker_volume", "raw_taker_volume_path"),
        (
            "save_global_long_short_account_ratio",
            "load_global_long_short_account_ratio",
            "raw_global_long_short_account_ratio_path",
        ),
        (
            "save_top_long_short_account_ratio",
            "load_top_long_short_account_ratio",
            "raw_top_long_short_account_ratio_path",
        ),
        (
            "save_top_long_short_position_ratio",
            "load_top_long_short_position_ratio",
            "raw_top_long_short_position_ratio_path",
        ),
        ("save_liquidations", "load_liquidations", "raw_liquidation_path"),
    ],
)
def test_each_dataset_type_uses_distinct_path(
    repository: MarketDataRepository,
    layout: StorageLayout,
    datastore: _InMemoryDataStore,
    sample_frame: pl.DataFrame,
    save_name: str,
    load_name: str,
    path_method: str,
) -> None:
    """Each dataset type resolves a distinct layout path."""
    expected = getattr(layout, path_method)(
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    save = getattr(repository, save_name)
    load = getattr(repository, load_name)

    save(
        sample_frame,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    load(
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )

    assert datastore.write_paths[-1] == expected
    assert datastore.read_paths[-1] == expected


def test_has_ohlcv_empty_returns_false(
    repository: MarketDataRepository,
    datastore: _InMemoryDataStore,
) -> None:
    """has_ohlcv returns False when no yearly partitions exist."""
    assert repository.has_ohlcv(_SYMBOL, _TIMEFRAME) is False
    assert datastore.read_paths == []
    assert datastore.scan_paths == []
    assert datastore.exists_paths


def test_has_ohlcv_single_year_returns_true(
    repository: MarketDataRepository,
    layout: StorageLayout,
    datastore: _InMemoryDataStore,
    sample_frame: pl.DataFrame,
) -> None:
    """has_ohlcv returns True when a single year partition exists."""
    repository.save_ohlcv(
        sample_frame,
        exchange=_OHLCV_EXCHANGE,
        market=_OHLCV_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=2023,
    )
    datastore.read_paths.clear()
    datastore.exists_paths.clear()

    assert repository.has_ohlcv(_SYMBOL, _TIMEFRAME) is True
    assert datastore.read_paths == []
    assert datastore.scan_paths == []
    assert (
        layout.raw_ohlcv_path(
            _OHLCV_EXCHANGE,
            _OHLCV_MARKET,
            _SYMBOL,
            _TIMEFRAME,
            2023,
        )
        in datastore.exists_paths
    )


def test_has_ohlcv_multiple_years_returns_true(
    repository: MarketDataRepository,
    datastore: _InMemoryDataStore,
    sample_frame: pl.DataFrame,
) -> None:
    """has_ohlcv returns True when multiple year partitions exist."""
    for year in (2022, 2023, 2024):
        repository.save_ohlcv(
            sample_frame,
            exchange=_OHLCV_EXCHANGE,
            market=_OHLCV_MARKET,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            year=year,
        )
    datastore.read_paths.clear()

    assert repository.has_ohlcv(_SYMBOL, _TIMEFRAME) is True
    assert datastore.read_paths == []
    assert datastore.scan_paths == []


def test_has_ohlcv_missing_symbol_returns_false(
    repository: MarketDataRepository,
    datastore: _InMemoryDataStore,
    sample_frame: pl.DataFrame,
) -> None:
    """has_ohlcv returns False for a symbol with no stored partitions."""
    repository.save_ohlcv(
        sample_frame,
        exchange=_OHLCV_EXCHANGE,
        market=_OHLCV_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=2023,
    )
    datastore.read_paths.clear()

    assert repository.has_ohlcv("ETHUSDT", _TIMEFRAME) is False
    assert datastore.read_paths == []
    assert datastore.scan_paths == []
