"""Unit tests for resumable MarketDataRepository timestamp helpers."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from cqros.core.types import FilePath
from cqros.storage import (
    CorruptedDatasetError,
    MarketDataRepository,
    StorageLayout,
)

_EXCHANGE = "binance"
_MARKET = "usdt_perpetual"
_SYMBOL = "BTCUSDT"


class _RecordingStore:
    """In-memory datastore that can mark paths as corrupt on read/scan."""

    def __init__(self) -> None:
        self.frames: dict[Path, pl.DataFrame] = {}
        self.corrupt: set[Path] = set()

    def write(self, path: FilePath, dataframe: pl.DataFrame) -> None:
        target = Path(path)
        self.frames[target] = dataframe
        self.corrupt.discard(target)

    def read(self, path: FilePath) -> pl.DataFrame:
        target = Path(path)
        if target in self.corrupt:
            raise CorruptedDatasetError(
                "corrupt partition",
                error_code="STORAGE-TEST-CORRUPT",
                details={"path": str(target)},
            )
        return self.frames[target]

    def scan(self, path: FilePath) -> pl.LazyFrame:
        return self.read(path).lazy()

    def exists(self, path: FilePath) -> bool:
        target = Path(path)
        return target in self.frames or target in self.corrupt

    def delete(self, path: FilePath) -> None:
        del self.frames[Path(path)]

    def schema(self, path: FilePath) -> pl.Schema:
        return self.read(path).schema

    def row_count(self, path: FilePath) -> int:
        return self.read(path).height


@pytest.fixture
def repo(tmp_path: Path) -> tuple[MarketDataRepository, _RecordingStore]:
    """Compose a repository backed by an in-memory datastore."""
    store = _RecordingStore()
    repository = MarketDataRepository(StorageLayout(tmp_path), store)  # type: ignore[arg-type]
    return repository, store


def test_get_latest_funding_timestamp_empty(
    repo: tuple[MarketDataRepository, _RecordingStore],
) -> None:
    """Empty repository returns None."""
    repository, _ = repo
    assert (
        repository.get_latest_funding_timestamp(
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=_SYMBOL,
            timeframe="8h",
        )
        is None
    )


def test_get_latest_ohlcv_timestamp_reads_newest_partition(
    repo: tuple[MarketDataRepository, _RecordingStore],
) -> None:
    """Latest timestamp comes from the newest readable year partition."""
    repository, _ = repo
    repository.save_ohlcv(
        pl.DataFrame(
            {
                "symbol": ["BTCUSDT", "BTCUSDT"],
                "timeframe": ["1h", "1h"],
                "open_time": [1_700_000_000_000, 1_700_000_360_000],
                "close_time": [1_700_000_359_999, 1_700_000_719_999],
                "open": [1.0, 2.0],
                "high": [1.0, 2.0],
                "low": [1.0, 2.0],
                "close": [1.0, 2.0],
                "volume": [1.0, 1.0],
                "quote_volume": [1.0, 1.0],
                "trade_count": [1, 1],
            }
        ),
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe="1h",
        year=2023,
    )
    repository.save_ohlcv(
        pl.DataFrame(
            {
                "symbol": ["BTCUSDT"],
                "timeframe": ["1h"],
                "open_time": [1_735_689_600_000],
                "close_time": [1_735_689_959_999],
                "open": [3.0],
                "high": [3.0],
                "low": [3.0],
                "close": [3.0],
                "volume": [1.0],
                "quote_volume": [1.0],
                "trade_count": [1],
            }
        ),
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe="1h",
        year=2025,
    )
    latest = repository.get_latest_ohlcv_timestamp(
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe="1h",
    )
    assert latest == 1_735_689_600_000


def test_get_latest_skips_corrupt_newest_partition(
    repo: tuple[MarketDataRepository, _RecordingStore],
    tmp_path: Path,
) -> None:
    """Corrupt newest partitions are ignored in favor of older readable ones."""
    repository, store = repo
    older = pl.DataFrame(
        {"timestamp": [100, 200], "open_interest": [1.0, 2.0], "symbol": ["BTCUSDT", "BTCUSDT"]}
    )
    repository.save_open_interest(
        older,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe="1h",
        year=2024,
    )
    newer_path = StorageLayout(tmp_path).raw_open_interest_path(
        _EXCHANGE, _MARKET, _SYMBOL, "1h", 2025
    )
    store.frames[newer_path] = pl.DataFrame({"timestamp": [300]})
    store.corrupt.add(newer_path)

    latest = repository.get_latest_open_interest_timestamp(
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe="1h",
    )
    assert latest == 200


def test_save_funding_merges_and_deduplicates(
    repo: tuple[MarketDataRepository, _RecordingStore],
) -> None:
    """Saving overlapping funding rows preserves unique funding_time values."""
    repository, _ = repo
    first = pl.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "funding_time": [1000, 2000],
            "funding_rate": [0.01, 0.02],
            "mark_price": [1.0, 2.0],
        }
    )
    second = pl.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "funding_time": [2000, 3000],
            "funding_rate": [0.025, 0.03],
            "mark_price": [2.5, 3.0],
        }
    )
    repository.save_funding(
        first,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe="8h",
        year=2024,
    )
    repository.save_funding(
        second,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe="8h",
        year=2024,
    )
    loaded = repository.load_funding(
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe="8h",
        year=2024,
    )
    assert loaded.height == 3
    assert loaded.get_column("funding_time").to_list() == [1000, 2000, 3000]
    assert loaded.filter(pl.col("funding_time") == 2000)["funding_rate"][0] == 0.025


def test_get_latest_long_short_timestamp_by_dataset(
    repo: tuple[MarketDataRepository, _RecordingStore],
) -> None:
    """Long/short latest helper routes by storage dataset name."""
    repository, _ = repo
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timestamp": [9_000],
            "long_account": [0.6],
            "short_account": [0.4],
            "long_short_ratio": [1.5],
        }
    )
    repository.save_global_long_short_account_ratio(
        frame,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe="1h",
        year=2024,
    )
    latest = repository.get_latest_long_short_timestamp(
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe="1h",
        dataset="global_long_short_account_ratio",
    )
    assert latest == 9_000
