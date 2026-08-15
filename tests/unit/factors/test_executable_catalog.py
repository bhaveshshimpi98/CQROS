"""Unit tests for CQROS ``ExecutableFactorCatalog`` and generation filtering."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import polars as pl

from cqros.core.constants import EXCHANGE_BINANCE, MARKET_USDT_PERPETUAL
from cqros.factors.base import BaseFactor
from cqros.factors.default_registry import build_default_registry
from cqros.factors.executable_catalog import ExecutableFactorCatalog
from cqros.factors.generation_pipeline import (
    FactorGenerationPipeline,
    FactorGenerationStatistics,
)
from cqros.factors.registry import FactorRegistry
from cqros.factors.wide_to_long import WideToLongFactorTransformer

_MANAGER = "simple"
_EXCHANGE = EXCHANGE_BINANCE
_MARKET = MARKET_USDT_PERPETUAL
_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1h"
_YEAR = 2024
_OPEN_TIME = 1_704_067_200_000

# Columns currently assembled by generate_factors.load_factor_input_frame.
_PROCESSED_FACTOR_INPUT_COLUMNS: tuple[str, ...] = (
    "symbol",
    "timeframe",
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "funding_rate",
    "mark_price",
    "open_interest",
    "taker_buy_volume",
    "taker_sell_volume",
    "long_short_ratio",
)

_EXPECTED_EXECUTABLE_COUNT = 73
_EXPECTED_REGISTERED_COUNT = 111


@dataclass(frozen=True, slots=True)
class _CloseOnlyFactor(BaseFactor):
    """Executable when ``close`` is present."""

    name: str = "close_only"
    version: str = "1.0.0"
    description: str = "Close-only stub"
    category: str = "test"
    required_features: tuple[str, ...] = ("close",)
    produced_columns: tuple[str, ...] = ("close_only",)
    lookback: int = 0

    def __post_init__(self) -> None:
        BaseFactor.__post_init__(self)

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(pl.col("close").alias("close_only"))


@dataclass(frozen=True, slots=True)
class _VwapFactor(BaseFactor):
    """Non-executable until ``vwap`` is available."""

    name: str = "needs_vwap"
    version: str = "1.0.0"
    description: str = "VWAP-dependent stub"
    category: str = "test"
    required_features: tuple[str, ...] = ("close", "vwap")
    produced_columns: tuple[str, ...] = ("needs_vwap",)
    lookback: int = 0

    def __post_init__(self) -> None:
        BaseFactor.__post_init__(self)

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(
            ((pl.col("close") - pl.col("vwap")) / pl.col("vwap")).alias("needs_vwap")
        )


@dataclass(frozen=True, slots=True)
class _RelativeStubFactor(BaseFactor):
    """Relative-style stub requiring benchmark returns."""

    name: str = "needs_benchmark"
    version: str = "1.0.0"
    description: str = "Benchmark-dependent stub"
    category: str = "test"
    required_features: tuple[str, ...] = ("asset_return", "btc_return")
    produced_columns: tuple[str, ...] = ("needs_benchmark",)
    lookback: int = 0

    def __post_init__(self) -> None:
        BaseFactor.__post_init__(self)

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(
            (pl.col("asset_return") - pl.col("btc_return")).alias("needs_benchmark")
        )


def _registry_with(*factors: BaseFactor) -> FactorRegistry:
    registry = FactorRegistry()
    registry.register_many(factors)
    return registry


def _partition_kwargs() -> dict[str, object]:
    return {
        "manager": _MANAGER,
        "exchange": _EXCHANGE,
        "market": _MARKET,
        "symbol": _SYMBOL,
        "timeframe": _TIMEFRAME,
        "year": _YEAR,
    }


def _mock_repository() -> MagicMock:
    repository = MagicMock()
    repository.save = MagicMock(return_value=None)
    return repository


def _frame_with(*columns: str, row_count: int = 3) -> pl.DataFrame:
    """Build a synthetic frame containing the requested columns."""
    data: dict[str, object] = {
        "symbol": [_SYMBOL] * row_count,
        "timeframe": [_TIMEFRAME] * row_count,
        "open_time": [_OPEN_TIME + index for index in range(row_count)],
    }
    defaults: dict[str, list[float]] = {
        "open": [100.0 + index for index in range(row_count)],
        "high": [110.0 + index for index in range(row_count)],
        "low": [90.0 + index for index in range(row_count)],
        "close": [105.0 + index for index in range(row_count)],
        "volume": [1_000.0 + index for index in range(row_count)],
        "trade_count": [10.0 + index for index in range(row_count)],
        "funding_rate": [0.0001 for _ in range(row_count)],
        "mark_price": [105.0 + index for index in range(row_count)],
        "open_interest": [50_000.0 + index for index in range(row_count)],
        "taker_buy_volume": [600.0 + index for index in range(row_count)],
        "taker_sell_volume": [400.0 + index for index in range(row_count)],
        "long_short_ratio": [1.1 for _ in range(row_count)],
        "vwap": [104.0 + index for index in range(row_count)],
    }
    for column in columns:
        if column in data:
            continue
        if column in defaults:
            data[column] = defaults[column]
        else:
            data[column] = [float(index + 1) for index in range(row_count)]
    return pl.DataFrame(data)


class TestExecutableFactorCatalog:
    """Unit tests for automatic executable-factor detection."""

    def test_selects_factors_whose_required_features_are_present(self) -> None:
        """Only factors with fully satisfied requirements are returned."""
        registry = _registry_with(_CloseOnlyFactor(), _VwapFactor(), _RelativeStubFactor())
        catalog = ExecutableFactorCatalog(registry)

        executable = catalog.get_executable_factors(("symbol", "close"))

        assert tuple(factor.name for factor in executable) == ("close_only",)

    def test_skips_factors_with_missing_required_features(self) -> None:
        """Skipped factors are the complement of the executable set."""
        registry = _registry_with(_CloseOnlyFactor(), _VwapFactor(), _RelativeStubFactor())
        catalog = ExecutableFactorCatalog(registry)
        available = ("symbol", "close")

        skipped = catalog.get_skipped_factors(available)

        assert tuple(factor.name for factor in skipped) == (
            "needs_benchmark",
            "needs_vwap",
        )

    def test_production_catalog_selects_seventy_three_with_processed_columns(
        self,
    ) -> None:
        """Current processed factor-input columns unlock exactly 73 factors."""
        registry = build_default_registry()
        catalog = ExecutableFactorCatalog(registry)

        executable = catalog.get_executable_factors(_PROCESSED_FACTOR_INPUT_COLUMNS)
        skipped = catalog.get_skipped_factors(_PROCESSED_FACTOR_INPUT_COLUMNS)

        assert len(registry.list()) == _EXPECTED_REGISTERED_COUNT
        assert len(executable) == _EXPECTED_EXECUTABLE_COUNT
        assert len(skipped) == _EXPECTED_REGISTERED_COUNT - _EXPECTED_EXECUTABLE_COUNT
        assert {factor.name for factor in executable}.isdisjoint(
            {factor.name for factor in skipped}
        )

    def test_future_columns_unlock_additional_factors_automatically(self) -> None:
        """Adding previously missing columns expands the executable set."""
        registry = build_default_registry()
        catalog = ExecutableFactorCatalog(registry)
        baseline = catalog.get_executable_factors(_PROCESSED_FACTOR_INPUT_COLUMNS)

        with_vwap = catalog.get_executable_factors((*_PROCESSED_FACTOR_INPUT_COLUMNS, "vwap"))
        unlocked = {factor.name for factor in with_vwap} - {factor.name for factor in baseline}

        assert len(with_vwap) > len(baseline)
        assert unlocked == {
            "micro_price_pressure",
            "vwap_distance",
            "vwap_zscore",
        }

    def test_preserves_registry_listing_order(self) -> None:
        """Executable results follow ``FactorRegistry.list`` order."""
        registry = _registry_with(_VwapFactor(), _CloseOnlyFactor())
        catalog = ExecutableFactorCatalog(registry)
        available = ("close", "vwap")

        executable = catalog.get_executable_factors(available)

        assert tuple(factor.name for factor in executable) == tuple(
            factor.name for factor in registry.list()
        )


class TestGenerationPipelineExecutableFiltering:
    """Generation pipeline executes only catalog-selected factors."""

    def test_skips_non_executable_factors_without_failure(self) -> None:
        """Missing-column factors are skipped; generation still succeeds."""
        registry = _registry_with(_CloseOnlyFactor(), _VwapFactor(), _RelativeStubFactor())
        repository = _mock_repository()
        generation = FactorGenerationPipeline(
            registry,
            None,
            WideToLongFactorTransformer(),
            repository,
            executable_catalog=ExecutableFactorCatalog(registry),
        )
        frame = _frame_with("close")

        statistics = generation.run(frame, **_partition_kwargs())  # type: ignore[arg-type]

        assert statistics.total_registered_factors == 3
        assert statistics.executable_factors == 1
        assert statistics.skipped_factors == 2
        assert statistics.factors_generated == 1
        assert statistics.rows_generated == frame.height
        assert statistics.failed_symbols == ()
        repository.save.assert_called_once()
        saved = repository.save.call_args.args[0]
        assert set(saved.get_column("factor_name").to_list()) == {"close_only"}

    def test_statistics_report_registered_executable_and_skipped(self) -> None:
        """Statistics expose registered / executable / skipped counts."""
        registry = build_default_registry()
        repository = _mock_repository()
        generation = FactorGenerationPipeline(
            registry,
            None,
            WideToLongFactorTransformer(),
            repository,
            executable_catalog=ExecutableFactorCatalog(registry),
        )
        # Minimal OHLCV-only frame: many factors skip, price factors run.
        frame = _frame_with("open", "high", "low", "close", "volume", row_count=40)

        statistics = generation.run(frame, **_partition_kwargs())  # type: ignore[arg-type]

        assert isinstance(statistics, FactorGenerationStatistics)
        assert statistics.total_registered_factors == _EXPECTED_REGISTERED_COUNT
        assert statistics.executable_factors < _EXPECTED_REGISTERED_COUNT
        assert statistics.skipped_factors == (
            statistics.total_registered_factors - statistics.executable_factors
        )
        assert statistics.executable_factors == statistics.factors_generated
        assert statistics.skipped_factors > 0
        assert statistics.rows_generated == frame.height * statistics.factors_generated
        repository.save.assert_called_once()

    def test_processed_columns_execute_exactly_seventy_three_factors(self) -> None:
        """Full processed input columns select and generate 73 factors."""
        registry = build_default_registry()
        repository = _mock_repository()
        generation = FactorGenerationPipeline(
            registry,
            None,
            WideToLongFactorTransformer(),
            repository,
            executable_catalog=ExecutableFactorCatalog(registry),
        )
        frame = _frame_with(
            *[
                column
                for column in _PROCESSED_FACTOR_INPUT_COLUMNS
                if column not in {"symbol", "timeframe", "open_time"}
            ],
            row_count=60,
        )

        statistics = generation.run(frame, **_partition_kwargs())  # type: ignore[arg-type]

        assert statistics.total_registered_factors == _EXPECTED_REGISTERED_COUNT
        assert statistics.executable_factors == _EXPECTED_EXECUTABLE_COUNT
        assert statistics.skipped_factors == (
            _EXPECTED_REGISTERED_COUNT - _EXPECTED_EXECUTABLE_COUNT
        )
        assert statistics.factors_generated == _EXPECTED_EXECUTABLE_COUNT
        assert statistics.rows_generated == frame.height * _EXPECTED_EXECUTABLE_COUNT
