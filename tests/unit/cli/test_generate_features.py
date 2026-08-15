"""Unit tests for CQROS feature-generation CLI."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from cqros.cli.generate_features import (
    DiscoveredWorkItem,
    FeatureGenerationOptions,
    FeatureGenerationSummary,
    FeatureTaskResult,
    align_feature_input_frame,
    build_default_registry,
    build_feature_pipeline,
    build_options,
    build_parser,
    discover_work,
    format_summary,
    load_feature_input_frame,
    main,
    run_generation,
)
from cqros.config.models import ResearchConfig
from cqros.core.constants import DEFAULT_STORAGE_ROOT, STORAGE_DIR_FEATURES, STORAGE_DIR_PROCESSED
from cqros.core.exceptions import ValidationError
from cqros.features import FEATURE_NAMES, FeaturePipeline, FeatureRegistry, FeatureVerifier
from cqros.features.schema import FEATURE_COLUMNS
from cqros.storage import (
    DatasetNotFoundError,
    FeatureRepository,
    ParquetStore,
    ProcessedMarketDataRepository,
    StorageLayout,
)


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def _options(
    *,
    storage_root: Path,
    symbols: tuple[str, ...] | None = None,
    timeframes: tuple[str, ...] | None = None,
    years: tuple[int, ...] | None = None,
    overwrite: bool = False,
    workers: int = ResearchConfig().worker_count,
    verbose: bool = False,
    debug: bool = False,
) -> FeatureGenerationOptions:
    """Build options for tests against a temporary storage root."""
    return FeatureGenerationOptions(
        storage_root=storage_root,
        symbols=symbols,
        timeframes=timeframes,
        years=years,
        overwrite=overwrite,
        workers=workers,
        verbose=verbose,
        debug=debug,
    )


def _touch_processed(
    root: Path,
    *,
    dataset: str,
    symbol: str,
    timeframe: str,
    year: int,
) -> Path:
    """Create an empty processed year partition path on disk."""
    path = (
        root
        / STORAGE_DIR_PROCESSED
        / dataset
        / "binance"
        / "usdt_perpetual"
        / symbol
        / timeframe
        / f"{year}.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _touch_feature_ready_year(
    root: Path,
    *,
    symbol: str,
    timeframe: str,
    year: int,
) -> None:
    """Create OHLCV plus every catalog-required companion partition for a year."""
    _touch_processed(root, dataset="ohlcv", symbol=symbol, timeframe=timeframe, year=year)
    _touch_processed(root, dataset="funding", symbol=symbol, timeframe="8h", year=year)
    _touch_processed(root, dataset="open_interest", symbol=symbol, timeframe=timeframe, year=year)
    _touch_processed(root, dataset="taker_volume", symbol=symbol, timeframe=timeframe, year=year)
    _touch_processed(
        root,
        dataset="global_long_short_account_ratio",
        symbol=symbol,
        timeframe=timeframe,
        year=year,
    )


def _touch_feature(
    root: Path,
    *,
    symbol: str,
    timeframe: str,
    year: int,
) -> Path:
    """Create an empty feature year partition path on disk."""
    path = (
        root
        / STORAGE_DIR_FEATURES
        / "binance"
        / "usdt_perpetual"
        / symbol
        / timeframe
        / f"{year}.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_build_parser_defaults() -> None:
    """Omitted optional flags keep discovery defaults."""
    args = build_parser().parse_args([])
    assert args.symbols is None
    assert args.timeframes is None
    assert args.years is None
    assert args.overwrite is False
    assert args.workers == ResearchConfig().worker_count
    assert args.verbose is False
    assert args.debug is False


def test_build_parser_accepts_all_flags() -> None:
    """Parser accepts every documented feature-generation flag."""
    args = build_parser().parse_args(
        [
            "--symbols",
            "BTCUSDT",
            "ETHUSDT",
            "--timeframes",
            "1h",
            "4h",
            "--years",
            "2024",
            "2025",
            "--overwrite",
            "--workers",
            "2",
            "--verbose",
            "--debug",
        ]
    )
    assert args.symbols == ["BTCUSDT", "ETHUSDT"]
    assert args.timeframes == ["1h", "4h"]
    assert args.years == ["2024", "2025"]
    assert args.overwrite is True
    assert args.workers == 2
    assert args.verbose is True
    assert args.debug is True


def test_build_options_defaults() -> None:
    """Omitted filters map to discovery-all options."""
    options = build_options(build_parser().parse_args([]))
    assert options.storage_root == Path(DEFAULT_STORAGE_ROOT)
    assert options.symbols is None
    assert options.timeframes is None
    assert options.years is None
    assert options.overwrite is False
    assert options.workers == ResearchConfig().worker_count


def test_build_options_maps_filters() -> None:
    """Explicit CLI flags map onto FeatureGenerationOptions."""
    options = build_options(
        build_parser().parse_args(
            [
                "--symbols",
                "ETHUSDT",
                "--timeframes",
                "1d",
                "--years",
                "2023",
                "--overwrite",
                "--workers",
                "8",
                "--debug",
            ]
        )
    )
    assert options.symbols == ("ETHUSDT",)
    assert options.timeframes == ("1d",)
    assert options.years == (2023,)
    assert options.overwrite is True
    assert options.workers == 8
    assert options.debug is True


def test_build_options_rejects_non_positive_workers() -> None:
    """Non-positive --workers fails validation."""
    args = build_parser().parse_args(["--workers", "0"])
    with pytest.raises(ValidationError, match="workers must be greater than 0"):
        build_options(args)


def test_build_options_rejects_unsupported_timeframe() -> None:
    """Unsupported --timeframes values fail validation."""
    args = build_parser().parse_args(["--timeframes", "2x"])
    with pytest.raises(ValidationError, match="unsupported timeframe"):
        build_options(args)


def test_build_options_rejects_invalid_year() -> None:
    """Non-integer --years values fail validation."""
    args = build_parser().parse_args(["--years", "abc"])
    with pytest.raises(ValidationError, match="invalid year"):
        build_options(args)


def test_build_default_registry_contains_catalog() -> None:
    """Default registry registers every FEATURE_NAMES entry."""
    registry = build_default_registry()
    assert isinstance(registry, FeatureRegistry)
    assert set(registry.names()) == set(FEATURE_NAMES)
    assert len(registry.names()) == len(FEATURE_COLUMNS)


def test_discover_work_finds_symbols_and_years(tmp_path: Path) -> None:
    """Discovery walks feature-ready processed partitions without hardcoding symbols."""
    _touch_feature_ready_year(tmp_path, symbol="BTCUSDT", timeframe="1h", year=2024)
    _touch_feature_ready_year(tmp_path, symbol="ETHUSDT", timeframe="1h", year=2023)
    _touch_feature_ready_year(tmp_path, symbol="ETHUSDT", timeframe="1h", year=2024)

    repository = ProcessedMarketDataRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(repository, _options(storage_root=tmp_path))

    assert len(work) == 2
    assert work[0].symbol == "BTCUSDT"
    assert work[0].years == (2024,)
    assert work[1].symbol == "ETHUSDT"
    assert work[1].years == (2023, 2024)


def test_discover_work_excludes_years_missing_companions(tmp_path: Path) -> None:
    """OHLCV years without required companions are not scheduled."""
    _touch_feature_ready_year(tmp_path, symbol="BTCUSDT", timeframe="1d", year=2026)
    _touch_processed(tmp_path, dataset="ohlcv", symbol="BTCUSDT", timeframe="1d", year=2025)
    _touch_processed(tmp_path, dataset="funding", symbol="BTCUSDT", timeframe="8h", year=2025)
    # 2025 intentionally lacks open_interest / taker / long_short.

    repository = ProcessedMarketDataRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(repository, _options(storage_root=tmp_path))

    assert len(work) == 1
    assert work[0] == DiscoveredWorkItem(symbol="BTCUSDT", timeframe="1d", years=(2026,))


def test_discover_work_filters_symbols_timeframes_years(tmp_path: Path) -> None:
    """Symbol, timeframe, and year filters apply to discovery."""
    _touch_feature_ready_year(tmp_path, symbol="BTCUSDT", timeframe="1h", year=2024)
    _touch_feature_ready_year(tmp_path, symbol="BTCUSDT", timeframe="4h", year=2024)
    _touch_feature_ready_year(tmp_path, symbol="ETHUSDT", timeframe="1h", year=2024)
    _touch_feature_ready_year(tmp_path, symbol="BTCUSDT", timeframe="1h", year=2025)

    repository = ProcessedMarketDataRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(
        repository,
        _options(
            storage_root=tmp_path,
            symbols=("BTCUSDT",),
            timeframes=("1h",),
            years=(2024,),
        ),
    )

    assert len(work) == 1
    assert work[0] == DiscoveredWorkItem(symbol="BTCUSDT", timeframe="1h", years=(2024,))


def test_build_feature_pipeline_wires_dependencies(tmp_path: Path) -> None:
    """Dependency construction wires registry, repository, and pipeline."""
    options = _options(storage_root=tmp_path)
    with (
        patch("cqros.cli.generate_features.StorageLayout", wraps=StorageLayout) as layout_cls,
        patch("cqros.cli.generate_features.ParquetStore", wraps=ParquetStore) as store_cls,
        patch(
            "cqros.cli.generate_features.FeatureRepository",
            wraps=FeatureRepository,
        ) as feature_cls,
        patch(
            "cqros.cli.generate_features.build_default_registry",
            wraps=build_default_registry,
        ) as registry_fn,
        patch(
            "cqros.cli.generate_features.FeaturePipeline",
            wraps=FeaturePipeline,
        ) as pipeline_cls,
    ):
        pipeline = build_feature_pipeline(options)

    assert isinstance(pipeline, FeaturePipeline)
    layout_cls.assert_called_once_with(tmp_path)
    store_cls.assert_called_once_with()
    feature_cls.assert_called_once()
    registry_fn.assert_called_once_with()
    pipeline_cls.assert_called_once()


def test_load_feature_input_frame_joins_companions(tmp_path: Path) -> None:
    """Companion series are as-of joined onto the OHLCV timeline."""
    layout = StorageLayout(tmp_path)
    store = ParquetStore()
    processed = ProcessedMarketDataRepository(layout, store)

    ohlcv = pl.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "timeframe": ["1h", "1h"],
            "open_time": [1_000, 2_000],
            "open": [1.0, 1.0],
            "high": [2.0, 2.0],
            "low": [0.5, 0.5],
            "close": [1.5, 1.6],
            "volume": [10.0, 11.0],
            "quote_volume": [15.0, 16.0],
            "trade_count": [1, 2],
            "close_time": [1_999, 2_999],
        }
    )
    funding = pl.DataFrame(
        {
            "funding_time": [1_000],
            "funding_rate": [0.0001],
            "mark_price": [1.5],
        }
    )
    open_interest = pl.DataFrame(
        {
            "timestamp": [1_000, 2_000],
            "open_interest": [100.0, 110.0],
        }
    )
    taker = pl.DataFrame(
        {
            "timestamp": [1_000, 2_000],
            "buy_volume": [4.0, 5.0],
            "sell_volume": [6.0, 7.0],
            "buy_sell_ratio": [4 / 6, 5 / 7],
        }
    )
    long_short = pl.DataFrame(
        {
            "timestamp": [1_000, 2_000],
            "long_account": [0.6, 0.55],
            "short_account": [0.4, 0.45],
            "long_short_ratio": [1.5, 1.22],
        }
    )

    base_kwargs: dict[str, Any] = {
        "exchange": "binance",
        "market": "usdt_perpetual",
        "symbol": "BTCUSDT",
        "year": 2024,
    }
    # Funding is stored under the native 8h settlement timeframe, not the
    # OHLCV bar timeframe. Companions that share bar intervals use ``1h``.
    processed.save_ohlcv(ohlcv, timeframe="1h", **base_kwargs)
    processed.save_funding(funding, timeframe="8h", **base_kwargs)
    processed.save_open_interest(open_interest, timeframe="1h", **base_kwargs)
    processed.save_taker_volume(taker, timeframe="1h", **base_kwargs)
    processed.save_global_long_short_account_ratio(long_short, timeframe="1h", **base_kwargs)

    frame = load_feature_input_frame(processed, symbol="BTCUSDT", timeframe="1h", year=2024)
    assert frame.height == 2
    assert "funding_rate" in frame.columns
    assert "open_interest" in frame.columns
    assert "buy_volume" in frame.columns
    assert "sell_volume" in frame.columns
    assert "long_short_ratio" in frame.columns
    assert frame.get_column("funding_rate").to_list() == [0.0001, 0.0001]
    assert frame.get_column("open_interest").to_list() == [100.0, 110.0]


def test_load_feature_input_frame_loads_funding_from_native_timeframe(
    tmp_path: Path,
) -> None:
    """Funding must be loaded from ``8h`` even when OHLCV is ``1h``."""
    layout = StorageLayout(tmp_path)
    store = ParquetStore()
    processed = ProcessedMarketDataRepository(layout, store)

    ohlcv = pl.DataFrame(
        {
            "symbol": ["ETHUSDT", "ETHUSDT", "ETHUSDT"],
            "timeframe": ["1h", "1h", "1h"],
            "open_time": [8_000, 9_000, 10_000],
            "open": [1.0, 1.0, 1.0],
            "high": [2.0, 2.0, 2.0],
            "low": [0.5, 0.5, 0.5],
            "close": [1.5, 1.6, 1.7],
            "volume": [10.0, 11.0, 12.0],
            "quote_volume": [15.0, 16.0, 17.0],
            "trade_count": [1, 2, 3],
            "close_time": [8_999, 9_999, 10_999],
        }
    )
    funding = pl.DataFrame(
        {
            "funding_time": [8_000],
            "funding_rate": [0.0002],
            "mark_price": [1.5],
        }
    )
    open_interest = pl.DataFrame(
        {
            "timestamp": [8_000, 9_000, 10_000],
            "open_interest": [100.0, 110.0, 120.0],
        }
    )
    taker = pl.DataFrame(
        {
            "timestamp": [8_000, 9_000, 10_000],
            "buy_volume": [4.0, 5.0, 6.0],
            "sell_volume": [6.0, 7.0, 8.0],
            "buy_sell_ratio": [4 / 6, 5 / 7, 6 / 8],
        }
    )
    long_short = pl.DataFrame(
        {
            "timestamp": [8_000, 9_000, 10_000],
            "long_account": [0.6, 0.55, 0.5],
            "short_account": [0.4, 0.45, 0.5],
            "long_short_ratio": [1.5, 1.22, 1.0],
        }
    )
    kwargs: dict[str, Any] = {
        "exchange": "binance",
        "market": "usdt_perpetual",
        "symbol": "ETHUSDT",
        "year": 2024,
    }
    processed.save_ohlcv(ohlcv, timeframe="1h", **kwargs)
    processed.save_funding(funding, timeframe="8h", **kwargs)
    processed.save_open_interest(open_interest, timeframe="1h", **kwargs)
    processed.save_taker_volume(taker, timeframe="1h", **kwargs)
    processed.save_global_long_short_account_ratio(long_short, timeframe="1h", **kwargs)

    # No funding partition exists under the OHLCV timeframe.
    with pytest.raises(DatasetNotFoundError):
        processed.load_funding(
            exchange="binance",
            market="usdt_perpetual",
            symbol="ETHUSDT",
            timeframe="1h",
            year=2024,
        )

    frame = load_feature_input_frame(processed, symbol="ETHUSDT", timeframe="1h", year=2024)
    assert "funding_rate" in frame.columns
    assert frame.get_column("funding_rate").to_list() == [0.0002, 0.0002, 0.0002]


def test_load_feature_input_frame_requires_companions(tmp_path: Path) -> None:
    """Missing companion partitions raise DatasetNotFoundError instead of omitting columns."""
    layout = StorageLayout(tmp_path)
    store = ParquetStore()
    processed = ProcessedMarketDataRepository(layout, store)
    ohlcv = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timeframe": ["1d"],
            "open_time": [1_000],
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "volume": [10.0],
            "quote_volume": [15.0],
            "trade_count": [1],
            "close_time": [1_999],
        }
    )
    funding = pl.DataFrame(
        {
            "funding_time": [1_000],
            "funding_rate": [0.0001],
            "mark_price": [1.5],
        }
    )
    kwargs: dict[str, Any] = {
        "exchange": "binance",
        "market": "usdt_perpetual",
        "symbol": "BTCUSDT",
        "year": 2025,
    }
    processed.save_ohlcv(ohlcv, timeframe="1d", **kwargs)
    processed.save_funding(funding, timeframe="8h", **kwargs)

    with pytest.raises(DatasetNotFoundError):
        load_feature_input_frame(processed, symbol="BTCUSDT", timeframe="1d", year=2025)


def test_load_feature_input_frame_requires_ohlcv(tmp_path: Path) -> None:
    """Missing OHLCV partitions raise DatasetNotFoundError."""
    processed = ProcessedMarketDataRepository(StorageLayout(tmp_path), ParquetStore())
    with pytest.raises(DatasetNotFoundError):
        load_feature_input_frame(processed, symbol="BTCUSDT", timeframe="1h", year=2024)


def _joined_feature_input_frame(
    *,
    open_times: list[int],
    funding_times: list[int],
    open_interest_times: list[int],
    taker_times: list[int],
    long_short_times: list[int],
) -> pl.DataFrame:
    """Build a joined feature-input frame with explicit companion timelines."""
    n = len(open_times)
    base = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * n,
            "timeframe": ["1h"] * n,
            "open_time": open_times,
            "close": [100.0 + float(i) for i in range(n)],
            "high": [101.0 + float(i) for i in range(n)],
            "low": [99.0 + float(i) for i in range(n)],
            "volume": [10.0 + float(i) for i in range(n)],
        }
    ).sort("open_time")

    def _asof(
        left: pl.DataFrame,
        times: list[int],
        values: dict[str, list[float]],
    ) -> pl.DataFrame:
        right = pl.DataFrame({"open_time": times, **values}).sort("open_time")
        return left.join_asof(right, on="open_time", strategy="backward")

    frame = _asof(
        base,
        funding_times,
        {"funding_rate": [0.0001 * (i + 1) for i in range(len(funding_times))]},
    )
    frame = _asof(
        frame,
        open_interest_times,
        {"open_interest": [1000.0 + float(i) for i in range(len(open_interest_times))]},
    )
    frame = _asof(
        frame,
        taker_times,
        {
            "buy_volume": [4.0 + float(i) for i in range(len(taker_times))],
            "sell_volume": [6.0 + float(i) for i in range(len(taker_times))],
        },
    )
    frame = _asof(
        frame,
        long_short_times,
        {"long_short_ratio": [1.5 - 0.01 * i for i in range(len(long_short_times))]},
    )
    return frame


def test_align_feature_input_frame_drops_rows_before_companions() -> None:
    """OHLCV bars before the latest companion start are dropped."""
    open_times = [1_000, 2_000, 3_000, 4_000, 5_000]
    frame = _joined_feature_input_frame(
        open_times=open_times,
        funding_times=[1_000, 2_000, 3_000, 4_000, 5_000],
        open_interest_times=[3_000, 4_000, 5_000],
        taker_times=[2_000, 3_000, 4_000, 5_000],
        long_short_times=[3_000, 4_000, 5_000],
    )
    # Before alignment, early rows have null companions.
    assert frame.filter(pl.col("open_interest").is_null()).height == 2

    aligned = align_feature_input_frame(frame)
    assert aligned.height == 3
    assert aligned.get_column("open_time").to_list() == [3_000, 4_000, 5_000]
    for column in (
        "funding_rate",
        "open_interest",
        "buy_volume",
        "sell_volume",
        "long_short_ratio",
    ):
        assert aligned.get_column(column).null_count() == 0


def test_align_feature_input_frame_selects_earliest_complete_row() -> None:
    """Alignment starts at the first row where every companion is non-null."""
    frame = _joined_feature_input_frame(
        open_times=[10, 20, 30, 40],
        funding_times=[10, 20, 30, 40],
        open_interest_times=[30, 40],
        taker_times=[20, 30, 40],
        long_short_times=[40],
    )
    aligned = align_feature_input_frame(frame)
    assert aligned.height == 1
    assert aligned.get_column("open_time").to_list() == [40]


def test_align_feature_input_frame_empty_raises() -> None:
    """No complete companion row raises DatasetNotFoundError."""
    frame = _joined_feature_input_frame(
        open_times=[1_000, 2_000, 3_000],
        funding_times=[1_000, 2_000, 3_000],
        open_interest_times=[10_000],
        taker_times=[1_000, 2_000, 3_000],
        long_short_times=[1_000, 2_000, 3_000],
    )
    with pytest.raises(DatasetNotFoundError, match="complete companion coverage") as exc_info:
        align_feature_input_frame(frame)
    assert exc_info.value.error_code == "CLI-GENERATE-FEATURES-006"


def test_align_feature_input_frame_fully_aligned_unchanged() -> None:
    """Frames that already have complete companions from row 0 are unchanged."""
    frame = _joined_feature_input_frame(
        open_times=[1_000, 2_000, 3_000],
        funding_times=[1_000, 2_000, 3_000],
        open_interest_times=[1_000, 2_000, 3_000],
        taker_times=[1_000, 2_000, 3_000],
        long_short_times=[1_000, 2_000, 3_000],
    )
    aligned = align_feature_input_frame(frame)
    assert aligned.equals(frame)


def test_load_feature_input_frame_aligns_late_companions(tmp_path: Path) -> None:
    """load_feature_input_frame drops OHLCV rows before companion coverage starts."""
    layout = StorageLayout(tmp_path)
    store = ParquetStore()
    processed = ProcessedMarketDataRepository(layout, store)

    ohlcv = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 5,
            "timeframe": ["1h"] * 5,
            "open_time": [1_000, 2_000, 3_000, 4_000, 5_000],
            "open": [1.0] * 5,
            "high": [2.0] * 5,
            "low": [0.5] * 5,
            "close": [1.5, 1.6, 1.7, 1.8, 1.9],
            "volume": [10.0, 11.0, 12.0, 13.0, 14.0],
            "quote_volume": [15.0] * 5,
            "trade_count": [1, 2, 3, 4, 5],
            "close_time": [1_999, 2_999, 3_999, 4_999, 5_999],
        }
    )
    kwargs: dict[str, Any] = {
        "exchange": "binance",
        "market": "usdt_perpetual",
        "symbol": "BTCUSDT",
        "year": 2024,
    }
    processed.save_ohlcv(ohlcv, timeframe="1h", **kwargs)
    processed.save_funding(
        pl.DataFrame(
            {
                "funding_time": [1_000, 3_000, 5_000],
                "funding_rate": [0.0001, 0.0002, 0.0003],
                "mark_price": [1.5, 1.7, 1.9],
            }
        ),
        timeframe="8h",
        **kwargs,
    )
    processed.save_open_interest(
        pl.DataFrame(
            {
                "timestamp": [3_000, 4_000, 5_000],
                "open_interest": [100.0, 110.0, 120.0],
            }
        ),
        timeframe="1h",
        **kwargs,
    )
    processed.save_taker_volume(
        pl.DataFrame(
            {
                "timestamp": [3_000, 4_000, 5_000],
                "buy_volume": [4.0, 5.0, 6.0],
                "sell_volume": [6.0, 7.0, 8.0],
                "buy_sell_ratio": [4 / 6, 5 / 7, 6 / 8],
            }
        ),
        timeframe="1h",
        **kwargs,
    )
    processed.save_global_long_short_account_ratio(
        pl.DataFrame(
            {
                "timestamp": [3_000, 4_000, 5_000],
                "long_account": [0.6, 0.55, 0.5],
                "short_account": [0.4, 0.45, 0.5],
                "long_short_ratio": [1.5, 1.22, 1.0],
            }
        ),
        timeframe="1h",
        **kwargs,
    )

    frame = load_feature_input_frame(processed, symbol="BTCUSDT", timeframe="1h", year=2024)
    assert frame.get_column("open_time").to_list() == [3_000, 4_000, 5_000]
    for column in (
        "funding_rate",
        "open_interest",
        "buy_volume",
        "sell_volume",
        "long_short_ratio",
    ):
        assert frame.get_column(column).null_count() == 0


def test_aligned_feature_generation_verifies_with_zero_null_rows(tmp_path: Path) -> None:
    """Aligned companion inputs produce a FeatureVerifier-passing feature frame."""
    layout = StorageLayout(tmp_path)
    store = ParquetStore()
    processed = ProcessedMarketDataRepository(layout, store)
    features = FeatureRepository(layout, store)

    row_count = 60
    open_times = [1_000_000 + i * 3_600_000 for i in range(row_count)]
    # Companions start 10 bars after OHLCV so alignment must drop leading rows.
    companion_start = 10
    companion_times = open_times[companion_start:]

    ohlcv = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * row_count,
            "timeframe": ["1h"] * row_count,
            "open_time": open_times,
            "open": [100.0 + i for i in range(row_count)],
            "high": [101.0 + i for i in range(row_count)],
            "low": [99.0 + i for i in range(row_count)],
            "close": [100.5 + i * 0.25 for i in range(row_count)],
            "volume": [10.0 + i for i in range(row_count)],
            "quote_volume": [1_000.0 + i for i in range(row_count)],
            "trade_count": list(range(1, row_count + 1)),
            "close_time": [t + 3_599_999 for t in open_times],
        }
    )
    kwargs: dict[str, Any] = {
        "exchange": "binance",
        "market": "usdt_perpetual",
        "symbol": "BTCUSDT",
        "year": 2024,
    }
    processed.save_ohlcv(ohlcv, timeframe="1h", **kwargs)
    processed.save_funding(
        pl.DataFrame(
            {
                "funding_time": companion_times,
                "funding_rate": [0.0001 + 0.00001 * i for i in range(len(companion_times))],
                "mark_price": [100.0 + i for i in range(len(companion_times))],
            }
        ),
        timeframe="8h",
        **kwargs,
    )
    processed.save_open_interest(
        pl.DataFrame(
            {
                "timestamp": companion_times,
                "open_interest": [1_000.0 + i for i in range(len(companion_times))],
            }
        ),
        timeframe="1h",
        **kwargs,
    )
    processed.save_taker_volume(
        pl.DataFrame(
            {
                "timestamp": companion_times,
                "buy_volume": [5.0 + i for i in range(len(companion_times))],
                "sell_volume": [4.0 + i for i in range(len(companion_times))],
                "buy_sell_ratio": [(5.0 + i) / (4.0 + i) for i in range(len(companion_times))],
            }
        ),
        timeframe="1h",
        **kwargs,
    )
    processed.save_global_long_short_account_ratio(
        pl.DataFrame(
            {
                "timestamp": companion_times,
                "long_account": [0.55 + 0.001 * i for i in range(len(companion_times))],
                "short_account": [0.45 - 0.001 * i for i in range(len(companion_times))],
                "long_short_ratio": [1.2 + 0.01 * i for i in range(len(companion_times))],
            }
        ),
        timeframe="1h",
        **kwargs,
    )

    frame = load_feature_input_frame(processed, symbol="BTCUSDT", timeframe="1h", year=2024)
    assert frame.height == row_count - companion_start
    assert frame.get_column("open_time").to_list()[0] == open_times[companion_start]
    for column in (
        "funding_rate",
        "open_interest",
        "buy_volume",
        "sell_volume",
        "long_short_ratio",
    ):
        assert frame.get_column(column).null_count() == 0

    pipeline = FeaturePipeline(build_default_registry(), features)
    output = pipeline.run(
        frame,
        FEATURE_NAMES,
        exchange="binance",
        market="usdt_perpetual",
        symbol="BTCUSDT",
        timeframe="1h",
        year=2024,
    )

    report = FeatureVerifier().verify(output)
    assert report.null_rows == 0
    assert report.passed is True
    for column in FEATURE_COLUMNS:
        assert output.get_column(column).null_count() == 0


def test_run_generation_empty_work(tmp_path: Path) -> None:
    """Empty discovery produces a zeroed summary without invoking the pipeline."""
    pipeline = MagicMock(spec=FeaturePipeline)
    processed = MagicMock(spec=ProcessedMarketDataRepository)
    features = MagicMock(spec=FeatureRepository)
    summary = _run(
        run_generation(
            pipeline=pipeline,
            processed_repository=processed,
            feature_repository=features,
            options=_options(storage_root=tmp_path),
            work=(),
        )
    )
    assert summary.successful_tasks == 0
    assert summary.failed_tasks == 0
    assert summary.skipped_tasks == 0
    pipeline.run.assert_not_called()


def test_run_generation_skips_existing_without_overwrite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Existing feature partitions are skipped unless --overwrite is set."""
    _touch_feature(tmp_path, symbol="BTCUSDT", timeframe="1h", year=2024)
    work = (DiscoveredWorkItem(symbol="BTCUSDT", timeframe="1h", years=(2024,)),)

    pipeline = MagicMock(spec=FeaturePipeline)
    processed = MagicMock(spec=ProcessedMarketDataRepository)
    features = MagicMock(spec=FeatureRepository)
    features.exists.return_value = True

    summary = _run(
        run_generation(
            pipeline=pipeline,
            processed_repository=processed,
            feature_repository=features,
            options=_options(storage_root=tmp_path, workers=1),
            work=work,
        )
    )

    captured = capsys.readouterr().out
    assert summary.skipped_tasks == 1
    assert summary.successful_tasks == 0
    pipeline.run.assert_not_called()
    assert "SKIP BTCUSDT 1h 2024" in captured


def test_run_generation_success_and_progress(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Successful generation increments counters and prints progress."""
    work = (DiscoveredWorkItem(symbol="BTCUSDT", timeframe="1h", years=(2024,)),)
    pipeline = MagicMock(spec=FeaturePipeline)
    pipeline.run.return_value = pl.DataFrame({"open_time": [1, 2, 3]})
    processed = MagicMock(spec=ProcessedMarketDataRepository)
    processed.load_ohlcv.return_value = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 3,
            "timeframe": ["1h"] * 3,
            "open_time": [1, 2, 3],
            "close": [1.0, 1.1, 1.2],
            "high": [1.2, 1.3, 1.4],
            "low": [0.9, 1.0, 1.1],
            "volume": [10.0, 11.0, 12.0],
        }
    )
    processed.load_funding.return_value = pl.DataFrame(
        {
            "funding_time": [1, 2, 3],
            "funding_rate": [0.0001, 0.0002, 0.0003],
        }
    )
    processed.load_open_interest.return_value = pl.DataFrame(
        {
            "timestamp": [1, 2, 3],
            "open_interest": [100.0, 110.0, 120.0],
        }
    )
    processed.load_taker_volume.return_value = pl.DataFrame(
        {
            "timestamp": [1, 2, 3],
            "buy_volume": [4.0, 5.0, 6.0],
            "sell_volume": [6.0, 7.0, 8.0],
        }
    )
    processed.load_global_long_short_account_ratio.return_value = pl.DataFrame(
        {
            "timestamp": [1, 2, 3],
            "long_short_ratio": [1.5, 1.4, 1.3],
        }
    )
    features = MagicMock(spec=FeatureRepository)
    features.exists.return_value = False

    summary = _run(
        run_generation(
            pipeline=pipeline,
            processed_repository=processed,
            feature_repository=features,
            options=_options(storage_root=tmp_path, workers=1, overwrite=True),
            work=work,
        )
    )

    captured = capsys.readouterr().out
    assert summary.successful_tasks == 1
    assert summary.rows_generated == 3
    assert "OK BTCUSDT 1h 2024 rows=3" in captured
    pipeline.run.assert_called_once()


def test_run_generation_failure_isolation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failed year does not prevent later years from running."""
    work = (DiscoveredWorkItem(symbol="BTCUSDT", timeframe="1h", years=(2024, 2025)),)
    pipeline = MagicMock(spec=FeaturePipeline)
    pipeline.run.side_effect = [RuntimeError("boom"), pl.DataFrame({"open_time": [1]})]
    processed = MagicMock(spec=ProcessedMarketDataRepository)
    processed.load_ohlcv.return_value = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timeframe": ["1h"],
            "open_time": [1],
            "close": [1.0],
            "high": [1.0],
            "low": [1.0],
            "volume": [1.0],
        }
    )
    processed.load_funding.return_value = pl.DataFrame(
        {
            "funding_time": [1],
            "funding_rate": [0.0001],
        }
    )
    processed.load_open_interest.return_value = pl.DataFrame(
        {
            "timestamp": [1],
            "open_interest": [100.0],
        }
    )
    processed.load_taker_volume.return_value = pl.DataFrame(
        {
            "timestamp": [1],
            "buy_volume": [4.0],
            "sell_volume": [6.0],
        }
    )
    processed.load_global_long_short_account_ratio.return_value = pl.DataFrame(
        {
            "timestamp": [1],
            "long_short_ratio": [1.5],
        }
    )
    features = MagicMock(spec=FeatureRepository)
    features.exists.return_value = False

    summary = _run(
        run_generation(
            pipeline=pipeline,
            processed_repository=processed,
            feature_repository=features,
            options=_options(storage_root=tmp_path, workers=1, overwrite=True),
            work=work,
        )
    )

    captured = capsys.readouterr().out
    assert summary.failed_tasks == 1
    assert summary.successful_tasks == 1
    assert "FAIL BTCUSDT 1h 2024 RuntimeError" in captured
    assert "OK BTCUSDT 1h 2025 rows=1" in captured
    assert summary.failed_task_labels == ("BTCUSDT 1h 2024",)


def test_format_summary_includes_failed_tasks() -> None:
    """Summary rendering includes failed-task labels when present."""
    text = format_summary(
        FeatureGenerationSummary(
            symbols_discovered=1,
            symbols_processed=1,
            timeframes_processed=1,
            successful_tasks=0,
            failed_tasks=1,
            skipped_tasks=0,
            rows_generated=0,
            duration_seconds=1.5,
            output_directory=Path("data/features"),
            failed_task_labels=("BTCUSDT 1h 2024",),
        )
    )
    assert "CQROS Feature Generation Summary" in text
    assert "Failed tasks: 1" in text
    assert "- BTCUSDT 1h 2024" in text
    assert "Output directory: data/features" in text


def test_main_exit_code_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """main returns 0 when no tasks fail."""
    with (
        patch("cqros.cli.generate_features.DEFAULT_STORAGE_ROOT", str(tmp_path)),
        patch("cqros.cli.generate_features.build_feature_pipeline") as build_pipeline,
        patch("cqros.cli.generate_features.discover_work", return_value=()),
    ):
        build_pipeline.return_value = MagicMock(spec=FeaturePipeline)
        code = _run(main(["--workers", "1"]))

    captured = capsys.readouterr()
    assert code == 0
    assert "CQROS Feature Generation Summary" in captured.out


def test_main_validation_error_exit_code() -> None:
    """Fatal CLI validation errors return exit code 1."""
    code = _run(main(["--workers", "0"]))
    assert code == 1


def test_configure_logging_debug(tmp_path: Path) -> None:
    """--debug enables DEBUG logging for the cqros logger."""
    with (
        patch("cqros.cli.generate_features.DEFAULT_STORAGE_ROOT", str(tmp_path)),
        patch("cqros.cli.generate_features.build_feature_pipeline") as build_pipeline,
        patch("cqros.cli.generate_features.discover_work", return_value=()),
    ):
        build_pipeline.return_value = MagicMock(spec=FeaturePipeline)
        _run(main(["--debug"]))
    assert logging.getLogger("cqros").level == logging.DEBUG


def test_feature_task_result_fields() -> None:
    """FeatureTaskResult stores status metadata immutably."""
    result = FeatureTaskResult(
        symbol="BTCUSDT",
        timeframe="1h",
        year=2024,
        status="failed",
        error_type="RuntimeError",
        error_message="boom",
    )
    assert result.status == "failed"
    assert result.error_type == "RuntimeError"
