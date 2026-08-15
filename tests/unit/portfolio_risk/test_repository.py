"""Unit tests for CQROS portfolio-risk dataset repository."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.core.constants import (
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
    STORAGE_DIR_PORTFOLIO_RISK,
)
from cqros.core.types import FilePath
from cqros.portfolio_risk import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    PortfolioRiskPartitionRef,
    PortfolioRiskRepository,
    PortfolioRiskState,
    ShutdownReason,
)
from cqros.portfolio_risk.repository import (
    PortfolioRiskRepository as PortfolioRiskRepositoryDirect,
)
from cqros.storage import DatasetNotFoundError, ParquetStore, StorageLayout

_MANAGER = "simple"
_EXCHANGE = EXCHANGE_BINANCE
_MARKET = MARKET_USDT_PERPETUAL
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


def _portfolio_risk_frame(*, symbol: str = _SYMBOL) -> pl.DataFrame:
    """Return a deterministic canonical portfolio-risk DataFrame."""
    open_time = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
    return pl.DataFrame(
        {
            "symbol": [symbol],
            "timeframe": [_TIMEFRAME],
            "open_time": [open_time],
            "manager": [_MANAGER],
            "position_id": ["pos-00000001"],
            "equity": [1000.0],
            "gross_exposure": [500.0],
            "net_exposure": [500.0],
            "daily_realized_pnl": [0.0],
            "daily_unrealized_pnl": [0.0],
            "daily_total_pnl": [0.0],
            "daily_return_pct": [0.0],
            "daily_drawdown_pct": [0.0],
            "portfolio_risk_state": [PortfolioRiskState.NORMAL.value],
            "allow_new_entries": [True],
            "shutdown_reason": [ShutdownReason.NONE.value],
            "cooldown_until": [None],
            "model_name": ["alpha-lgbm"],
            "model_version": ["1.0.0"],
            "optimizer": ["equal_weight"],
            "policy": ["fixed_risk"],
        },
        schema=dict(COLUMN_DTYPES),
    ).select(list(CANONICAL_COLUMN_ORDER))


@pytest.fixture
def sample_frame() -> pl.DataFrame:
    """Return a deterministic canonical sample portfolio-risk DataFrame."""
    return _portfolio_risk_frame()


def test_portfolio_risk_repository_is_exported_from_package() -> None:
    """Package export matches the repository module by identity."""
    assert PortfolioRiskRepository is PortfolioRiskRepositoryDirect


def test_portfolio_risk_partition_ref_is_frozen_dataclass() -> None:
    """PortfolioRiskPartitionRef is an immutable slotted dataclass."""
    assert is_dataclass(PortfolioRiskPartitionRef)
    ref = PortfolioRiskPartitionRef(
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    with pytest.raises(FrozenInstanceError):
        ref.year = 2025  # type: ignore[misc]


def test_save_load_exists_delete_round_trip(
    tmp_path: Path,
    sample_frame: pl.DataFrame,
) -> None:
    """Repository persists and reloads frames through the portfolio-risk path."""
    layout = StorageLayout(tmp_path)
    datastore = _InMemoryDataStore()
    repository = PortfolioRiskRepository(layout, datastore)
    assert (
        repository.exists(
            manager=_MANAGER,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )
        is False
    )
    repository.save(
        sample_frame,
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    expected_path = layout.portfolio_risk_path(
        _MANAGER,
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    assert datastore.write_paths == [expected_path]
    assert STORAGE_DIR_PORTFOLIO_RISK in expected_path.parts
    loaded = repository.load(
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert_frame_equal(loaded, sample_frame)
    repository.delete(
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert datastore.delete_paths == [expected_path]


def test_save_overwrites_existing_partition(tmp_path: Path) -> None:
    """Saving an existing partition overwrites the stored frame."""
    layout = StorageLayout(tmp_path)
    datastore = _InMemoryDataStore()
    repository = PortfolioRiskRepository(layout, datastore)
    first = _portfolio_risk_frame()
    second = _portfolio_risk_frame().with_columns(pl.lit(999.0).alias("equity"))
    kwargs = {
        "manager": _MANAGER,
        "exchange": _EXCHANGE,
        "market": _MARKET,
        "symbol": _SYMBOL,
        "timeframe": _TIMEFRAME,
        "year": _YEAR,
    }
    repository.save(first, **kwargs)
    repository.save(second, **kwargs)
    loaded = repository.load(**kwargs)
    assert_frame_equal(loaded, second)
    assert len(datastore.write_paths) == 2


def test_discover_partitions_multiple_symbols(tmp_path: Path) -> None:
    """Discovery walks the portfolio-risk tier and returns sorted partition refs."""
    layout = StorageLayout(tmp_path)
    repository = PortfolioRiskRepository(layout, ParquetStore())
    for symbol in ("BTCUSDT", "ETHUSDT"):
        repository.save(
            _portfolio_risk_frame(symbol=symbol),
            manager=_MANAGER,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=symbol,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )
    assert repository.discover_managers() == (_MANAGER,)
    assert repository.discover_symbols(manager=_MANAGER) == ("BTCUSDT", "ETHUSDT")
    partitions = repository.discover_partitions(managers=(_MANAGER,))
    assert partitions == (
        PortfolioRiskPartitionRef(
            manager=_MANAGER,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol="BTCUSDT",
            timeframe=_TIMEFRAME,
            year=_YEAR,
        ),
        PortfolioRiskPartitionRef(
            manager=_MANAGER,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol="ETHUSDT",
            timeframe=_TIMEFRAME,
            year=_YEAR,
        ),
    )


def test_discover_partitions_multiple_years_and_timeframes(tmp_path: Path) -> None:
    """Discovery returns sorted years and timeframes for a symbol."""
    layout = StorageLayout(tmp_path)
    repository = PortfolioRiskRepository(layout, ParquetStore())
    for timeframe, year in (("1h", 2025), ("1h", 2026), ("4h", 2026)):
        repository.save(
            _portfolio_risk_frame(),
            manager=_MANAGER,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=_SYMBOL,
            timeframe=timeframe,
            year=year,
        )
    assert repository.discover_timeframes(manager=_MANAGER, symbol=_SYMBOL) == ("1h", "4h")
    assert repository.list_years(
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe="1h",
    ) == (2025, 2026)
    partitions = repository.discover_partitions(managers=(_MANAGER,), timeframes=("1h",))
    assert tuple(ref.year for ref in partitions) == (2025, 2026)


def test_discover_managers_empty_when_missing(tmp_path: Path) -> None:
    """Discovery returns empty tuples when the portfolio-risk tier is absent."""
    repository = PortfolioRiskRepository(StorageLayout(tmp_path), ParquetStore())
    assert repository.discover_managers() == ()
    assert repository.discover_symbols(manager=_MANAGER) == ()
    assert repository.discover_timeframes(manager=_MANAGER, symbol=_SYMBOL) == ()
    assert repository.discover_partitions() == ()
