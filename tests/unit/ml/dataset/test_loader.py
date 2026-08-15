"""Unit tests for the CQROS ML ``DatasetLoader``."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.core.constants import (
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
    STORAGE_DIR_TRAINING,
)
from cqros.core.types import FilePath
from cqros.ml.dataset import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    MERGED_TRAINING_SCHEMA,
    PRIMARY_KEY_COLUMNS,
    DatasetLoader,
    DatasetLoaderError,
)
from cqros.ml.dataset.loader import DatasetLoader as DatasetLoaderDirect
from cqros.storage import (
    DatasetNotFoundError,
    StorageLayout,
    TrainingPartitionRef,
    TrainingRepository,
)

_EXCHANGE = EXCHANGE_BINANCE
_MARKET = MARKET_USDT_PERPETUAL
_START = 1_700_000_000_000
_INTERVAL = 3_600_000


class _RecordingRepository:
    """Training repository stub that records discovery and load calls."""

    def __init__(self, partitions: dict[tuple[str, str, int], pl.DataFrame]) -> None:
        self.partitions = partitions
        self.discover_calls: list[dict[str, object]] = []
        self.load_calls: list[dict[str, object]] = []

    def discover_partitions(
        self,
        *,
        symbols: tuple[str, ...] | list[str] | None = None,
        timeframes: tuple[str, ...] | list[str] | None = None,
        exchange: str = _EXCHANGE,
        market: str = _MARKET,
    ) -> tuple[TrainingPartitionRef, ...]:
        self.discover_calls.append(
            {
                "symbols": symbols,
                "timeframes": timeframes,
                "exchange": exchange,
                "market": market,
            }
        )
        symbol_filter = set(symbols) if symbols is not None else None
        timeframe_filter = set(timeframes) if timeframes is not None else None
        items: list[TrainingPartitionRef] = []
        for symbol, timeframe, year in self.partitions:
            if symbol_filter is not None and symbol not in symbol_filter:
                continue
            if timeframe_filter is not None and timeframe not in timeframe_filter:
                continue
            items.append(TrainingPartitionRef(symbol=symbol, timeframe=timeframe, year=year))
        return tuple(sorted(items, key=lambda item: (item.symbol, item.timeframe, item.year)))

    def load(
        self,
        *,
        exchange: str,
        market: str,
        symbol: str,
        timeframe: str,
        year: int,
    ) -> pl.DataFrame:
        self.load_calls.append(
            {
                "exchange": exchange,
                "market": market,
                "symbol": symbol,
                "timeframe": timeframe,
                "year": year,
            }
        )
        return self.partitions[(symbol, timeframe, year)]


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


def _training_frame(
    *,
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    open_times: list[int] | None = None,
    feature_value: float = 0.01,
) -> pl.DataFrame:
    """Build a canonical merged training frame for one partition."""
    if open_times is None:
        open_times = [_START, _START + _INTERVAL]
    row_count = len(open_times)
    data: dict[str, object] = {
        "symbol": [symbol] * row_count,
        "timeframe": [timeframe] * row_count,
        "open_time": open_times,
    }
    data.update(_feature_values(row_count, value=feature_value))
    data.update(_label_values(row_count))
    frame = pl.DataFrame(data, schema=COLUMN_DTYPES)
    return frame.select(list(CANONICAL_COLUMN_ORDER))


def _save_partition_files(
    root: Path,
    *,
    symbol: str,
    timeframe: str,
    year: int,
) -> None:
    """Create an empty parquet marker so repository discovery can see it."""
    path = (
        root / STORAGE_DIR_TRAINING / _EXCHANGE / _MARKET / symbol / timeframe / f"{year}.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_dataset_loader_is_exported_from_package() -> None:
    """Package export matches the loader module class."""
    assert DatasetLoader is DatasetLoaderDirect


def test_load_single_partition() -> None:
    """A single matching partition is returned with canonical schema."""
    frame = _training_frame()
    repository = _RecordingRepository({("BTCUSDT", "1h", 2024): frame})
    loader = DatasetLoader(repository)  # type: ignore[arg-type]

    result = loader.load(symbols=("BTCUSDT",), timeframes=("1h",), years=(2024,))

    assert result.height == 2
    assert result.columns == list(CANONICAL_COLUMN_ORDER)
    assert result.schema == MERGED_TRAINING_SCHEMA
    assert_frame_equal(result, frame.sort(list(PRIMARY_KEY_COLUMNS)))


def test_load_multiple_partitions() -> None:
    """Multiple partitions are concatenated into one frame."""
    btc = _training_frame(symbol="BTCUSDT", open_times=[_START])
    eth = _training_frame(
        symbol="ETHUSDT",
        open_times=[_START + _INTERVAL],
        feature_value=0.5,
    )
    repository = _RecordingRepository(
        {
            ("BTCUSDT", "1h", 2024): btc,
            ("ETHUSDT", "1h", 2024): eth,
        }
    )
    loader = DatasetLoader(repository)  # type: ignore[arg-type]

    result = loader.load()

    assert result.height == 2
    assert result.get_column("symbol").to_list() == ["BTCUSDT", "ETHUSDT"]
    assert len(repository.load_calls) == 2


def test_load_multiple_symbols() -> None:
    """Symbol filter selects only requested symbols."""
    repository = _RecordingRepository(
        {
            ("BTCUSDT", "1h", 2024): _training_frame(symbol="BTCUSDT"),
            ("ETHUSDT", "1h", 2024): _training_frame(symbol="ETHUSDT"),
            ("SOLUSDT", "1h", 2024): _training_frame(symbol="SOLUSDT"),
        }
    )
    loader = DatasetLoader(repository)  # type: ignore[arg-type]

    result = loader.load(symbols=("BTCUSDT", "SOLUSDT"))

    assert set(result.get_column("symbol").to_list()) == {"BTCUSDT", "SOLUSDT"}
    assert repository.discover_calls[0]["symbols"] == ("BTCUSDT", "SOLUSDT")


def test_load_multiple_timeframes() -> None:
    """Timeframe filter selects only requested timeframes."""
    repository = _RecordingRepository(
        {
            ("BTCUSDT", "1h", 2024): _training_frame(timeframe="1h"),
            ("BTCUSDT", "4h", 2024): _training_frame(timeframe="4h"),
            ("BTCUSDT", "1d", 2024): _training_frame(timeframe="1d"),
        }
    )
    loader = DatasetLoader(repository)  # type: ignore[arg-type]

    result = loader.load(timeframes=("1h", "1d"))

    assert set(result.get_column("timeframe").to_list()) == {"1h", "1d"}
    assert repository.discover_calls[0]["timeframes"] == ("1h", "1d")


def test_load_multiple_years() -> None:
    """Year filter keeps only requested calendar years."""
    repository = _RecordingRepository(
        {
            ("BTCUSDT", "1h", 2023): _training_frame(open_times=[_START]),
            ("BTCUSDT", "1h", 2024): _training_frame(open_times=[_START + _INTERVAL]),
            ("BTCUSDT", "1h", 2025): _training_frame(open_times=[_START + 2 * _INTERVAL]),
        }
    )
    loader = DatasetLoader(repository)  # type: ignore[arg-type]

    result = loader.load(years=(2023, 2025))

    assert result.height == 2
    assert {call["year"] for call in repository.load_calls} == {2023, 2025}


def test_load_combined_filters() -> None:
    """Symbol, timeframe, and year filters compose."""
    repository = _RecordingRepository(
        {
            ("BTCUSDT", "1h", 2024): _training_frame(symbol="BTCUSDT", timeframe="1h"),
            ("BTCUSDT", "4h", 2024): _training_frame(symbol="BTCUSDT", timeframe="4h"),
            ("ETHUSDT", "1h", 2024): _training_frame(symbol="ETHUSDT", timeframe="1h"),
            ("BTCUSDT", "1h", 2025): _training_frame(
                symbol="BTCUSDT",
                timeframe="1h",
                open_times=[_START + _INTERVAL],
            ),
        }
    )
    loader = DatasetLoader(repository)  # type: ignore[arg-type]

    result = loader.load(
        symbols=("BTCUSDT",),
        timeframes=("1h",),
        years=(2024,),
    )

    assert result.height == 2
    assert result.get_column("symbol").to_list() == ["BTCUSDT", "BTCUSDT"]
    assert result.get_column("timeframe").to_list() == ["1h", "1h"]
    assert repository.load_calls == [
        {
            "exchange": _EXCHANGE,
            "market": _MARKET,
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "year": 2024,
        }
    ]


def test_load_preserves_primary_key_ordering() -> None:
    """Returned rows are sorted by symbol, timeframe, open_time."""
    later = _training_frame(
        symbol="BTCUSDT",
        timeframe="1h",
        open_times=[_START + 2 * _INTERVAL, _START + 3 * _INTERVAL],
    )
    earlier = _training_frame(
        symbol="AAVEUSDT",
        timeframe="4h",
        open_times=[_START + _INTERVAL],
    )
    middle = _training_frame(
        symbol="BTCUSDT",
        timeframe="1h",
        open_times=[_START],
        feature_value=0.2,
    )
    repository = _RecordingRepository(
        {
            ("BTCUSDT", "1h", 2025): later,
            ("AAVEUSDT", "4h", 2024): earlier,
            ("BTCUSDT", "1h", 2024): middle,
        }
    )
    loader = DatasetLoader(repository)  # type: ignore[arg-type]

    result = loader.load()

    assert result.select(list(PRIMARY_KEY_COLUMNS)).to_dicts() == [
        {"symbol": "AAVEUSDT", "timeframe": "4h", "open_time": _START + _INTERVAL},
        {"symbol": "BTCUSDT", "timeframe": "1h", "open_time": _START},
        {
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "open_time": _START + 2 * _INTERVAL,
        },
        {
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "open_time": _START + 3 * _INTERVAL,
        },
    ]


def test_load_preserves_dtypes_and_canonical_column_order() -> None:
    """Output schema matches MERGED_TRAINING_SCHEMA column order and dtypes."""
    disordered = _training_frame().select(
        [
            *CANONICAL_COLUMN_ORDER[1:],
            CANONICAL_COLUMN_ORDER[0],
        ]
    )
    repository = _RecordingRepository({("BTCUSDT", "1h", 2024): disordered})
    loader = DatasetLoader(repository)  # type: ignore[arg-type]

    result = loader.load()

    assert result.columns == list(CANONICAL_COLUMN_ORDER)
    assert result.schema == MERGED_TRAINING_SCHEMA
    for column in CANONICAL_COLUMN_ORDER:
        assert result.schema[column] == COLUMN_DTYPES[column]


def test_load_empty_repository_returns_empty_canonical_frame() -> None:
    """An empty repository yields an empty canonical schema frame."""
    loader = DatasetLoader(_RecordingRepository({}))  # type: ignore[arg-type]

    result = loader.load()

    assert result.height == 0
    assert result.columns == list(CANONICAL_COLUMN_ORDER)
    assert result.schema == MERGED_TRAINING_SCHEMA
    assert result is not None


def test_load_empty_filter_result_returns_empty_canonical_frame() -> None:
    """Filters that match nothing return an empty canonical schema frame."""
    repository = _RecordingRepository({("BTCUSDT", "1h", 2024): _training_frame()})
    loader = DatasetLoader(repository)  # type: ignore[arg-type]

    result = loader.load(symbols=("ETHUSDT",))

    assert result.height == 0
    assert result.schema == MERGED_TRAINING_SCHEMA
    assert repository.load_calls == []


def test_invalid_parameter_types_raise_dataset_loader_error() -> None:
    """Non-string symbols/timeframes and non-integer years are rejected."""
    loader = DatasetLoader(_RecordingRepository({}))  # type: ignore[arg-type]

    with pytest.raises(DatasetLoaderError, match="symbols"):
        loader.load(symbols="BTCUSDT")  # type: ignore[arg-type]
    with pytest.raises(DatasetLoaderError, match="symbols"):
        loader.load(symbols=("BTCUSDT", 1))  # type: ignore[arg-type]
    with pytest.raises(DatasetLoaderError, match="timeframes"):
        loader.load(timeframes=["1h", 4])  # type: ignore[arg-type]
    with pytest.raises(DatasetLoaderError, match="years"):
        loader.load(years=2024)  # type: ignore[arg-type]
    with pytest.raises(DatasetLoaderError, match="years"):
        loader.load(years=(2024, True))  # type: ignore[arg-type]
    with pytest.raises(DatasetLoaderError, match="years"):
        loader.load(years=("2024",))  # type: ignore[arg-type]


def test_repository_interaction_uses_discovery_and_load_only() -> None:
    """Loader discovers partitions then loads each match through the repository."""
    frame = _training_frame()
    repository = _RecordingRepository(
        {
            ("BTCUSDT", "1h", 2024): frame,
            ("ETHUSDT", "1h", 2024): _training_frame(symbol="ETHUSDT"),
        }
    )
    loader = DatasetLoader(repository)  # type: ignore[arg-type]

    loader.load(symbols=["BTCUSDT"], timeframes=["1h"], years=[2024])

    assert len(repository.discover_calls) == 1
    assert repository.discover_calls[0] == {
        "symbols": ("BTCUSDT",),
        "timeframes": ("1h",),
        "exchange": _EXCHANGE,
        "market": _MARKET,
    }
    assert repository.load_calls == [
        {
            "exchange": _EXCHANGE,
            "market": _MARKET,
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "year": 2024,
        }
    ]


def test_filter_inputs_are_never_mutated() -> None:
    """Caller-supplied filter collections are left unchanged."""
    symbols = ["BTCUSDT", "ETHUSDT"]
    timeframes = ["1h"]
    years = [2024]
    symbols_copy = list(symbols)
    timeframes_copy = list(timeframes)
    years_copy = list(years)
    repository = _RecordingRepository(
        {
            ("BTCUSDT", "1h", 2024): _training_frame(symbol="BTCUSDT"),
            ("ETHUSDT", "1h", 2024): _training_frame(symbol="ETHUSDT"),
        }
    )
    loader = DatasetLoader(repository)  # type: ignore[arg-type]

    loader.load(symbols=symbols, timeframes=timeframes, years=years)

    assert symbols == symbols_copy
    assert timeframes == timeframes_copy
    assert years == years_copy


def test_loaded_frames_are_not_mutated() -> None:
    """Repository-owned frames remain unchanged after loading."""
    original = _training_frame()
    snapshot = original.clone()
    repository = _RecordingRepository({("BTCUSDT", "1h", 2024): original})
    loader = DatasetLoader(repository)  # type: ignore[arg-type]

    result = loader.load()
    result = result.with_columns(pl.lit("MUTATED").alias("symbol"))

    assert_frame_equal(repository.partitions[("BTCUSDT", "1h", 2024)], snapshot)
    assert result.get_column("symbol").to_list() == ["MUTATED", "MUTATED"]


def test_load_through_real_training_repository(tmp_path: Path) -> None:
    """Loader works against a real TrainingRepository and ParquetStore."""
    layout = StorageLayout(tmp_path)
    datastore = _InMemoryDataStore()
    repository = TrainingRepository(layout, datastore)
    btc = _training_frame(symbol="BTCUSDT", open_times=[_START + _INTERVAL])
    eth = _training_frame(symbol="ETHUSDT", open_times=[_START])

    for symbol, year, frame in (
        ("BTCUSDT", 2024, btc),
        ("ETHUSDT", 2023, eth),
    ):
        _save_partition_files(tmp_path, symbol=symbol, timeframe="1h", year=year)
        repository.save(
            frame,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=symbol,
            timeframe="1h",
            year=year,
        )

    loader = DatasetLoader(repository)
    result = loader.load(years=(2023, 2024))

    assert result.height == 2
    assert result.columns == list(CANONICAL_COLUMN_ORDER)
    assert result.schema == MERGED_TRAINING_SCHEMA
    assert result.select(list(PRIMARY_KEY_COLUMNS)).to_dicts() == [
        {
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "open_time": _START + _INTERVAL,
        },
        {"symbol": "ETHUSDT", "timeframe": "1h", "open_time": _START},
    ]
