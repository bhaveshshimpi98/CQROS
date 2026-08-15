"""Unit tests for CQROS factor generation CLI."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from cqros.cli.generate_factors import (
    DiscoveredWorkItem,
    FactorGenerationOptions,
    FactorGenerationSummary,
    FactorTaskResult,
    align_factor_input_frame,
    build_factor_generation_pipeline,
    build_options,
    build_parser,
    discover_work,
    format_summary,
    load_factor_input_frame,
    main,
    run_generation,
)
from cqros.config.models import ResearchConfig
from cqros.core.constants import (
    DEFAULT_STORAGE_ROOT,
    STORAGE_DIR_FACTORS,
    STORAGE_DIR_PROCESSED,
)
from cqros.core.exceptions import ValidationError
from cqros.factors import (
    FactorGenerationPipeline,
    FactorGenerationStatistics,
    FactorsRepository,
)
from cqros.storage import (
    DatasetNotFoundError,
    ParquetStore,
    ProcessedMarketDataRepository,
    StorageLayout,
)

_MANAGER = "simple"
_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1h"
_YEAR = 2024


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def _options(
    *,
    storage_root: Path,
    manager: str = _MANAGER,
    symbols: tuple[str, ...] | None = None,
    timeframes: tuple[str, ...] | None = None,
    years: tuple[int, ...] | None = None,
    overwrite: bool = False,
    workers: int = ResearchConfig().worker_count,
    verbose: bool = False,
    debug: bool = False,
) -> FactorGenerationOptions:
    """Build FactorGenerationOptions against a temporary storage root."""
    return FactorGenerationOptions(
        storage_root=storage_root,
        manager=manager,
        symbols=symbols,
        timeframes=timeframes,
        years=years,
        overwrite=overwrite,
        workers=workers,
        verbose=verbose,
        debug=debug,
    )


def _statistics(*, rows_generated: int = 2) -> FactorGenerationStatistics:
    """Return immutable pipeline statistics for successful generation."""
    return FactorGenerationStatistics(
        symbols_processed=1,
        rows_generated=rows_generated,
        factors_generated=3,
        total_registered_factors=111,
        executable_factors=73,
        skipped_factors=38,
        generation_duration=0.01,
        failed_symbols=(),
        successful_symbols=(_SYMBOL,),
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


def _touch_factor_ready_year(
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


def _save_factor_ready_frames(
    processed: ProcessedMarketDataRepository,
    *,
    symbol: str = _SYMBOL,
    timeframe: str = _TIMEFRAME,
    year: int = _YEAR,
) -> None:
    """Persist minimal processed partitions suitable for factor input assembly."""
    base_kwargs: dict[str, Any] = {
        "exchange": "binance",
        "market": "usdt_perpetual",
        "symbol": symbol,
        "year": year,
    }
    ohlcv = pl.DataFrame(
        {
            "symbol": [symbol, symbol],
            "timeframe": [timeframe, timeframe],
            "open_time": [1_000, 2_000],
            "open": [1.0, 1.1],
            "high": [2.0, 2.1],
            "low": [0.5, 0.6],
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
    processed.save_ohlcv(ohlcv, timeframe=timeframe, **base_kwargs)
    processed.save_funding(funding, timeframe="8h", **base_kwargs)
    processed.save_open_interest(open_interest, timeframe=timeframe, **base_kwargs)
    processed.save_taker_volume(taker, timeframe=timeframe, **base_kwargs)
    processed.save_global_long_short_account_ratio(long_short, timeframe=timeframe, **base_kwargs)


# ---------------------------------------------------------------------------
# Parser and options
# ---------------------------------------------------------------------------


def test_build_parser_requires_manager() -> None:
    """Parser requires --manager."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(["--manager", "simple"])
    assert args.manager == "simple"
    assert args.overwrite is False


def test_build_parser_accepts_all_flags(tmp_path: Path) -> None:
    """Parser correctly maps all supported CLI flags."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "--manager",
            "ledger",
            "--symbols",
            "BTCUSDT",
            "--timeframes",
            "1h",
            "--years",
            "2024",
            "--workers",
            "4",
            "--overwrite",
            "--storage-root",
            str(tmp_path),
            "--verbose",
            "--debug",
        ]
    )
    assert args.manager == "ledger"
    assert args.symbols == ["BTCUSDT"]
    assert args.timeframes == ["1h"]
    assert args.years == ["2024"]
    assert args.workers == 4
    assert args.overwrite is True
    assert args.storage_root == tmp_path
    assert args.verbose is True
    assert args.debug is True


def test_build_options_defaults() -> None:
    """Omitted filters map to discovery-all options."""
    args = build_parser().parse_args(["--manager", "simple"])
    options = build_options(args)
    assert options.manager == "simple"
    assert options.storage_root == Path(DEFAULT_STORAGE_ROOT)
    assert options.symbols is None
    assert options.timeframes is None
    assert options.years is None
    assert options.overwrite is False
    assert options.workers == ResearchConfig().worker_count


def test_build_options_maps_filters(tmp_path: Path) -> None:
    """build_options maps filters and storage-root overrides."""
    args = build_parser().parse_args(
        [
            "--manager",
            "simple",
            "--symbols",
            "BTCUSDT",
            "ETHUSDT",
            "--timeframes",
            "1h",
            "--years",
            "2024",
            "2025",
            "--overwrite",
            "--storage-root",
            str(tmp_path),
            "--workers",
            "2",
        ]
    )
    options = build_options(args)
    assert options.manager == "simple"
    assert options.symbols == ("BTCUSDT", "ETHUSDT")
    assert options.timeframes == ("1h",)
    assert options.years == (2024, 2025)
    assert options.overwrite is True
    assert options.storage_root == tmp_path
    assert options.workers == 2


def test_build_options_rejects_non_positive_workers() -> None:
    """Non-positive --workers fails validation."""
    args = build_parser().parse_args(["--manager", "simple", "--workers", "0"])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-FACTORS-001"


def test_build_options_rejects_blank_manager() -> None:
    """Blank --manager fails validation."""
    args = build_parser().parse_args(["--manager", "   "])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-FACTORS-004"


def test_build_options_rejects_unsupported_timeframe() -> None:
    """Unsupported --timeframes values fail validation."""
    args = build_parser().parse_args(["--manager", "simple", "--timeframes", "2x"])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-FACTORS-002"


def test_build_options_rejects_invalid_year() -> None:
    """Non-integer --years values fail validation."""
    args = build_parser().parse_args(["--manager", "simple", "--years", "abc"])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-GENERATE-FACTORS-003"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_discover_work_groups_processed_partitions(tmp_path: Path) -> None:
    """discover_work groups processed partitions without inventing missing years."""
    _touch_factor_ready_year(tmp_path, symbol="ETHUSDT", timeframe="1h", year=2025)
    _touch_factor_ready_year(tmp_path, symbol="BTCUSDT", timeframe="1h", year=2024)
    _touch_factor_ready_year(tmp_path, symbol="BTCUSDT", timeframe="1h", year=2025)
    _touch_factor_ready_year(tmp_path, symbol="BTCUSDT", timeframe="4h", year=2024)

    repository = ProcessedMarketDataRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(repository, _options(storage_root=tmp_path))

    assert work == (
        DiscoveredWorkItem(symbol="BTCUSDT", timeframe="1h", years=(2024, 2025)),
        DiscoveredWorkItem(symbol="BTCUSDT", timeframe="4h", years=(2024,)),
        DiscoveredWorkItem(symbol="ETHUSDT", timeframe="1h", years=(2025,)),
    )


def test_discover_work_excludes_years_missing_companions(tmp_path: Path) -> None:
    """OHLCV years without required companions are not scheduled."""
    _touch_factor_ready_year(tmp_path, symbol="BTCUSDT", timeframe="1d", year=2026)
    _touch_processed(tmp_path, dataset="ohlcv", symbol="BTCUSDT", timeframe="1d", year=2025)
    _touch_processed(tmp_path, dataset="funding", symbol="BTCUSDT", timeframe="8h", year=2025)

    repository = ProcessedMarketDataRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(repository, _options(storage_root=tmp_path))

    assert work == (DiscoveredWorkItem(symbol="BTCUSDT", timeframe="1d", years=(2026,)),)


def test_discover_work_filters_symbols_timeframes_years(tmp_path: Path) -> None:
    """Symbol, timeframe, and year filters apply to discovery."""
    _touch_factor_ready_year(tmp_path, symbol="BTCUSDT", timeframe="1h", year=2024)
    _touch_factor_ready_year(tmp_path, symbol="BTCUSDT", timeframe="1h", year=2025)
    _touch_factor_ready_year(tmp_path, symbol="BTCUSDT", timeframe="4h", year=2024)
    _touch_factor_ready_year(tmp_path, symbol="ETHUSDT", timeframe="1h", year=2024)

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

    assert work == (DiscoveredWorkItem(symbol="BTCUSDT", timeframe="1h", years=(2024,)),)


def test_discover_work_empty_when_no_partitions(tmp_path: Path) -> None:
    """Empty processed discovery yields no work items."""
    repository = ProcessedMarketDataRepository(StorageLayout(tmp_path), ParquetStore())
    assert discover_work(repository, _options(storage_root=tmp_path)) == ()


# ---------------------------------------------------------------------------
# Pipeline composition and input assembly
# ---------------------------------------------------------------------------


def test_build_factor_generation_pipeline_wires_dependencies(tmp_path: Path) -> None:
    """Pipeline composition returns a FactorGenerationPipeline instance."""
    from cqros.factors import FactorRegistry

    with patch(
        "cqros.cli.generate_factors.build_default_registry",
        return_value=FactorRegistry(),
    ):
        pipeline = build_factor_generation_pipeline(_options(storage_root=tmp_path))
    assert isinstance(pipeline, FactorGenerationPipeline)


def test_load_factor_input_frame_joins_companions(tmp_path: Path) -> None:
    """Companion series are as-of joined onto the OHLCV timeline."""
    layout = StorageLayout(tmp_path)
    store = ParquetStore()
    processed = ProcessedMarketDataRepository(layout, store)
    _save_factor_ready_frames(processed)

    frame = load_factor_input_frame(processed, symbol=_SYMBOL, timeframe=_TIMEFRAME, year=_YEAR)

    assert frame.height == 2
    assert frame.columns == [
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
    ]
    assert frame.get_column("funding_rate").to_list() == [0.0001, 0.0001]
    assert frame.get_column("mark_price").to_list() == [1.5, 1.5]
    assert frame.get_column("open_interest").to_list() == [100.0, 110.0]
    assert frame.get_column("taker_buy_volume").to_list() == [4.0, 5.0]
    assert frame.get_column("taker_sell_volume").to_list() == [6.0, 7.0]
    assert "buy_volume" not in frame.columns
    assert "sell_volume" not in frame.columns


def test_load_factor_input_frame_loads_funding_from_native_timeframe(
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
        "year": _YEAR,
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
            year=_YEAR,
        )

    frame = load_factor_input_frame(processed, symbol="ETHUSDT", timeframe="1h", year=_YEAR)
    assert "funding_rate" in frame.columns
    assert "mark_price" in frame.columns
    assert frame.get_column("funding_rate").to_list() == [0.0002, 0.0002, 0.0002]
    assert frame.get_column("mark_price").to_list() == [1.5, 1.5, 1.5]


def test_load_factor_input_frame_floors_funding_subsecond_noise(
    tmp_path: Path,
) -> None:
    """Funding settlements a few ms after bar open still as-of match that bar."""
    layout = StorageLayout(tmp_path)
    store = ParquetStore()
    processed = ProcessedMarketDataRepository(layout, store)

    ohlcv = pl.DataFrame(
        {
            "symbol": ["ETHUSDT", "ETHUSDT"],
            "timeframe": ["1h", "1h"],
            "open_time": [8_000_000, 9_000_000],
            "open": [1.0, 1.0],
            "high": [2.0, 2.0],
            "low": [0.5, 0.5],
            "close": [1.5, 1.6],
            "volume": [10.0, 11.0],
            "quote_volume": [15.0, 16.0],
            "trade_count": [1, 2],
            "close_time": [8_999_999, 9_999_999],
        }
    )
    # Settlement is 8ms after the first bar open (Binance exchange noise).
    funding = pl.DataFrame(
        {
            "funding_time": [8_000_008],
            "funding_rate": [0.0003],
            "mark_price": [1.55],
        }
    )
    open_interest = pl.DataFrame(
        {
            "timestamp": [8_000_000, 9_000_000],
            "open_interest": [100.0, 110.0],
        }
    )
    taker = pl.DataFrame(
        {
            "timestamp": [8_000_000, 9_000_000],
            "buy_volume": [4.0, 5.0],
            "sell_volume": [6.0, 7.0],
            "buy_sell_ratio": [4 / 6, 5 / 7],
        }
    )
    long_short = pl.DataFrame(
        {
            "timestamp": [8_000_000, 9_000_000],
            "long_account": [0.6, 0.55],
            "short_account": [0.4, 0.45],
            "long_short_ratio": [1.5, 1.22],
        }
    )
    kwargs: dict[str, Any] = {
        "exchange": "binance",
        "market": "usdt_perpetual",
        "symbol": "ETHUSDT",
        "year": _YEAR,
    }
    processed.save_ohlcv(ohlcv, timeframe="1h", **kwargs)
    processed.save_funding(funding, timeframe="8h", **kwargs)
    processed.save_open_interest(open_interest, timeframe="1h", **kwargs)
    processed.save_taker_volume(taker, timeframe="1h", **kwargs)
    processed.save_global_long_short_account_ratio(long_short, timeframe="1h", **kwargs)

    frame = load_factor_input_frame(processed, symbol="ETHUSDT", timeframe="1h", year=_YEAR)

    assert frame.height == 2
    assert frame.get_column("funding_rate").null_count() == 0
    assert frame.get_column("funding_rate").to_list() == [0.0003, 0.0003]
    assert frame.get_column("open_time").is_sorted()


def test_load_factor_input_frame_requires_companions(tmp_path: Path) -> None:
    """Missing companion partitions raise DatasetNotFoundError."""
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
    processed.save_ohlcv(
        ohlcv,
        exchange="binance",
        market="usdt_perpetual",
        symbol="BTCUSDT",
        timeframe="1d",
        year=2025,
    )

    with pytest.raises(DatasetNotFoundError):
        load_factor_input_frame(processed, symbol="BTCUSDT", timeframe="1d", year=2025)


def test_load_factor_input_frame_requires_ohlcv(tmp_path: Path) -> None:
    """Missing OHLCV partitions raise DatasetNotFoundError."""
    processed = ProcessedMarketDataRepository(StorageLayout(tmp_path), ParquetStore())
    with pytest.raises(DatasetNotFoundError):
        load_factor_input_frame(processed, symbol="BTCUSDT", timeframe="1h", year=2024)


def test_load_factor_input_frame_missing_required_columns(tmp_path: Path) -> None:
    """Companion frames missing required columns raise ValidationError."""
    layout = StorageLayout(tmp_path)
    store = ParquetStore()
    processed = ProcessedMarketDataRepository(layout, store)
    base_kwargs: dict[str, Any] = {
        "exchange": "binance",
        "market": "usdt_perpetual",
        "symbol": _SYMBOL,
        "year": _YEAR,
    }
    ohlcv = pl.DataFrame(
        {
            "symbol": [_SYMBOL],
            "timeframe": [_TIMEFRAME],
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
            # mark_price intentionally omitted
        }
    )
    open_interest = pl.DataFrame({"timestamp": [1_000], "open_interest": [100.0]})
    taker = pl.DataFrame(
        {
            "timestamp": [1_000],
            "buy_volume": [4.0],
            "sell_volume": [6.0],
            "buy_sell_ratio": [4 / 6],
        }
    )
    long_short = pl.DataFrame(
        {
            "timestamp": [1_000],
            "long_account": [0.6],
            "short_account": [0.4],
            "long_short_ratio": [1.5],
        }
    )
    processed.save_ohlcv(ohlcv, timeframe=_TIMEFRAME, **base_kwargs)
    processed.save_funding(funding, timeframe="8h", **base_kwargs)
    processed.save_open_interest(open_interest, timeframe=_TIMEFRAME, **base_kwargs)
    processed.save_taker_volume(taker, timeframe=_TIMEFRAME, **base_kwargs)
    processed.save_global_long_short_account_ratio(long_short, timeframe=_TIMEFRAME, **base_kwargs)

    with pytest.raises(ValidationError) as exc_info:
        load_factor_input_frame(processed, symbol=_SYMBOL, timeframe=_TIMEFRAME, year=_YEAR)
    assert exc_info.value.error_code == "CLI-GENERATE-FACTORS-006"
    assert "mark_price" in str(exc_info.value.details["missing_columns"])


def test_align_factor_input_frame_drops_leading_incomplete_rows() -> None:
    """Leading rows without companion coverage are dropped."""
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "timeframe": ["1h", "1h"],
            "open_time": [1_000, 2_000],
            "open": [1.0, 1.1],
            "high": [2.0, 2.1],
            "low": [0.5, 0.6],
            "close": [1.5, 1.6],
            "volume": [10.0, 11.0],
            "trade_count": [1, 2],
            "funding_rate": [None, 0.0001],
            "mark_price": [None, 1.5],
            "open_interest": [None, 110.0],
            "taker_buy_volume": [None, 5.0],
            "taker_sell_volume": [None, 7.0],
            "long_short_ratio": [None, 1.22],
        }
    )
    aligned = align_factor_input_frame(frame)
    assert aligned.height == 1
    assert aligned.get_column("open_time").to_list() == [2_000]


def test_load_factor_input_frame_retains_leading_companion_nulls(tmp_path: Path) -> None:
    """Joined load keeps OHLCV bars that precede companion availability."""
    layout = StorageLayout(tmp_path)
    store = ParquetStore()
    processed = ProcessedMarketDataRepository(layout, store)
    base_kwargs: dict[str, Any] = {
        "exchange": "binance",
        "market": "usdt_perpetual",
        "symbol": "BTCUSDT",
        "year": _YEAR,
    }
    ohlcv = pl.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "timeframe": ["1h", "1h", "1h"],
            "open_time": [1_000, 2_000, 3_000],
            "open": [1.0, 1.1, 1.2],
            "high": [2.0, 2.1, 2.2],
            "low": [0.5, 0.6, 0.7],
            "close": [1.5, 1.6, 1.7],
            "volume": [10.0, 11.0, 12.0],
            "quote_volume": [15.0, 16.0, 17.0],
            "trade_count": [1, 2, 3],
            "close_time": [1_999, 2_999, 3_999],
        }
    )
    funding = pl.DataFrame(
        {
            "funding_time": [2_000],
            "funding_rate": [0.0001],
            "mark_price": [1.5],
        }
    )
    open_interest = pl.DataFrame(
        {
            "timestamp": [2_000, 3_000],
            "open_interest": [100.0, 110.0],
        }
    )
    taker = pl.DataFrame(
        {
            "timestamp": [2_000, 3_000],
            "buy_volume": [4.0, 5.0],
            "sell_volume": [6.0, 7.0],
            "buy_sell_ratio": [4 / 6, 5 / 7],
        }
    )
    long_short = pl.DataFrame(
        {
            "timestamp": [2_000, 3_000],
            "long_account": [0.6, 0.55],
            "short_account": [0.4, 0.45],
            "long_short_ratio": [1.5, 1.22],
        }
    )
    processed.save_ohlcv(ohlcv, timeframe="1h", **base_kwargs)
    processed.save_funding(funding, timeframe="8h", **base_kwargs)
    processed.save_open_interest(open_interest, timeframe="1h", **base_kwargs)
    processed.save_taker_volume(taker, timeframe="1h", **base_kwargs)
    processed.save_global_long_short_account_ratio(long_short, timeframe="1h", **base_kwargs)

    frame = load_factor_input_frame(processed, symbol="BTCUSDT", timeframe="1h", year=_YEAR)
    assert frame.height == 3
    assert frame.get_column("open_time").to_list() == [1_000, 2_000, 3_000]
    assert frame.get_column("open_interest").to_list()[0] is None
    assert align_factor_input_frame(frame).height == 2



# ---------------------------------------------------------------------------
# run_generation
# ---------------------------------------------------------------------------


def test_run_generation_empty_work(tmp_path: Path) -> None:
    """Empty discovery produces a zeroed summary without invoking the pipeline."""
    pipeline = MagicMock(spec=FactorGenerationPipeline)
    processed = MagicMock(spec=ProcessedMarketDataRepository)
    factors = MagicMock(spec=FactorsRepository)

    summary = _run(
        run_generation(
            pipeline=pipeline,
            processed_repository=processed,
            factors_repository=factors,
            options=_options(storage_root=tmp_path),
            work=(),
        )
    )

    assert summary.manager == _MANAGER
    assert summary.symbols_discovered == 0
    assert summary.successful_tasks == 0
    assert summary.failed_tasks == 0
    assert summary.skipped_tasks == 0
    assert summary.output_directory == tmp_path / STORAGE_DIR_FACTORS
    pipeline.run.assert_not_called()
    factors.exists.assert_not_called()


def test_run_generation_skips_existing_without_overwrite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Existing factor partitions are skipped unless --overwrite is set."""
    work = (DiscoveredWorkItem(symbol=_SYMBOL, timeframe=_TIMEFRAME, years=(_YEAR,)),)
    pipeline = MagicMock(spec=FactorGenerationPipeline)
    processed = MagicMock(spec=ProcessedMarketDataRepository)
    factors = MagicMock(spec=FactorsRepository)
    factors.exists.return_value = True

    summary = _run(
        run_generation(
            pipeline=pipeline,
            processed_repository=processed,
            factors_repository=factors,
            options=_options(storage_root=tmp_path, workers=1),
            work=work,
        )
    )

    captured = capsys.readouterr().out
    assert summary.skipped_tasks == 1
    assert summary.successful_tasks == 0
    pipeline.run.assert_not_called()
    factors.exists.assert_called_once_with(
        manager=_MANAGER,
        exchange="binance",
        market="usdt_perpetual",
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert f"SKIP {_SYMBOL} {_TIMEFRAME} {_YEAR}" in captured


def test_run_generation_overwrite_existing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--overwrite regenerates partitions that already exist."""
    work = (DiscoveredWorkItem(symbol=_SYMBOL, timeframe=_TIMEFRAME, years=(_YEAR,)),)
    frame = pl.DataFrame({"open_time": [1, 2]})
    pipeline = MagicMock(spec=FactorGenerationPipeline)
    pipeline.run.return_value = _statistics(rows_generated=5)
    processed = MagicMock(spec=ProcessedMarketDataRepository)
    factors = MagicMock(spec=FactorsRepository)
    factors.exists.return_value = True

    with patch(
        "cqros.cli.generate_factors.load_factor_input_frame",
        return_value=frame,
    ) as load_frame:
        summary = _run(
            run_generation(
                pipeline=pipeline,
                processed_repository=processed,
                factors_repository=factors,
                options=_options(storage_root=tmp_path, workers=1, overwrite=True),
                work=work,
            )
        )

    captured = capsys.readouterr().out
    assert summary.successful_tasks == 1
    assert summary.skipped_tasks == 0
    assert summary.rows_generated == 5
    factors.exists.assert_not_called()
    load_frame.assert_called_once()
    pipeline.run.assert_called_once()
    assert f"OK {_SYMBOL} {_TIMEFRAME} {_YEAR} rows=5" in captured


def test_run_generation_success_and_repository_interactions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Successful generation loads processed data and runs the pipeline."""
    work = (DiscoveredWorkItem(symbol=_SYMBOL, timeframe=_TIMEFRAME, years=(_YEAR,)),)
    frame = pl.DataFrame({"open_time": [1, 2, 3]})
    pipeline = MagicMock(spec=FactorGenerationPipeline)
    pipeline.run.return_value = _statistics(rows_generated=3)
    processed = MagicMock(spec=ProcessedMarketDataRepository)
    factors = MagicMock(spec=FactorsRepository)
    factors.exists.return_value = False

    with patch(
        "cqros.cli.generate_factors.load_factor_input_frame",
        return_value=frame,
    ) as load_frame:
        summary = _run(
            run_generation(
                pipeline=pipeline,
                processed_repository=processed,
                factors_repository=factors,
                options=_options(storage_root=tmp_path, workers=1),
                work=work,
            )
        )

    captured = capsys.readouterr().out
    assert summary.successful_tasks == 1
    assert summary.symbols_discovered == 1
    assert summary.symbols_processed == 1
    assert summary.rows_generated == 3
    assert f"OK {_SYMBOL} {_TIMEFRAME} {_YEAR} rows=3" in captured

    load_frame.assert_called_once_with(
        processed,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    pipeline.run.assert_called_once_with(
        frame,
        manager=_MANAGER,
        exchange="binance",
        market="usdt_perpetual",
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )


def test_run_generation_pipeline_failure_isolation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failed year does not prevent later years from running."""
    work = (DiscoveredWorkItem(symbol=_SYMBOL, timeframe=_TIMEFRAME, years=(2024, 2025)),)
    pipeline = MagicMock(spec=FactorGenerationPipeline)
    pipeline.run.side_effect = [RuntimeError("boom"), _statistics(rows_generated=1)]
    processed = MagicMock(spec=ProcessedMarketDataRepository)
    factors = MagicMock(spec=FactorsRepository)
    factors.exists.return_value = False

    with patch(
        "cqros.cli.generate_factors.load_factor_input_frame",
        return_value=pl.DataFrame({"open_time": [1]}),
    ):
        summary = _run(
            run_generation(
                pipeline=pipeline,
                processed_repository=processed,
                factors_repository=factors,
                options=_options(storage_root=tmp_path, workers=1),
                work=work,
            )
        )

    captured = capsys.readouterr().out
    assert summary.failed_tasks == 1
    assert summary.successful_tasks == 1
    assert summary.rows_generated == 1
    assert f"FAIL {_SYMBOL} {_TIMEFRAME} 2024 RuntimeError" in captured
    assert f"OK {_SYMBOL} {_TIMEFRAME} 2025 rows=1" in captured
    assert summary.failed_task_labels == (f"{_SYMBOL} {_TIMEFRAME} 2024",)


def test_run_generation_processed_load_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Processed load failures are isolated as failed tasks."""
    work = (DiscoveredWorkItem(symbol=_SYMBOL, timeframe=_TIMEFRAME, years=(_YEAR,)),)
    pipeline = MagicMock(spec=FactorGenerationPipeline)
    processed = MagicMock(spec=ProcessedMarketDataRepository)
    factors = MagicMock(spec=FactorsRepository)
    factors.exists.return_value = False

    with patch(
        "cqros.cli.generate_factors.load_factor_input_frame",
        side_effect=FileNotFoundError("missing processed partition"),
    ):
        summary = _run(
            run_generation(
                pipeline=pipeline,
                processed_repository=processed,
                factors_repository=factors,
                options=_options(storage_root=tmp_path, workers=1),
                work=work,
            )
        )

    captured = capsys.readouterr().out
    assert summary.failed_tasks == 1
    assert summary.successful_tasks == 0
    pipeline.run.assert_not_called()
    assert f"FAIL {_SYMBOL} {_TIMEFRAME} {_YEAR} FileNotFoundError" in captured


def test_run_generation_repository_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Repository failures raised by the pipeline are isolated as failed tasks."""
    work = (DiscoveredWorkItem(symbol=_SYMBOL, timeframe=_TIMEFRAME, years=(_YEAR,)),)
    pipeline = MagicMock(spec=FactorGenerationPipeline)
    pipeline.run.side_effect = OSError("repository write failed")
    processed = MagicMock(spec=ProcessedMarketDataRepository)
    factors = MagicMock(spec=FactorsRepository)
    factors.exists.return_value = False

    with patch(
        "cqros.cli.generate_factors.load_factor_input_frame",
        return_value=pl.DataFrame({"open_time": [1]}),
    ):
        summary = _run(
            run_generation(
                pipeline=pipeline,
                processed_repository=processed,
                factors_repository=factors,
                options=_options(storage_root=tmp_path, workers=1),
                work=work,
            )
        )

    captured = capsys.readouterr().out
    assert summary.failed_tasks == 1
    assert summary.successful_tasks == 0
    assert f"FAIL {_SYMBOL} {_TIMEFRAME} {_YEAR} OSError" in captured


# ---------------------------------------------------------------------------
# Summary and main
# ---------------------------------------------------------------------------


def test_format_summary_includes_failed_tasks() -> None:
    """Summary rendering includes failed-task labels when present."""
    text = format_summary(
        FactorGenerationSummary(
            manager=_MANAGER,
            symbols_discovered=1,
            symbols_processed=1,
            timeframes_processed=1,
            successful_tasks=0,
            failed_tasks=1,
            skipped_tasks=0,
            rows_generated=0,
            total_registered_factors=111,
            executable_factors=73,
            skipped_factors=38,
            duration_seconds=1.5,
            output_directory=Path("data/factors"),
            failed_task_labels=(f"{_SYMBOL} {_TIMEFRAME} {_YEAR}",),
        )
    )
    assert "CQROS Factor Generation Summary" in text
    assert f"Manager: {_MANAGER}" in text
    assert "Symbols discovered: 1" in text
    assert "Symbols processed: 1" in text
    assert "Successful tasks: 0" in text
    assert "Failed tasks: 1" in text
    assert "Skipped tasks: 0" in text
    assert "Total registered factors: 111" in text
    assert "Executable factors: 73" in text
    assert "Skipped factors: 38" in text
    assert "Rows generated: 0" in text
    assert "Generation duration: 1.500s" in text
    assert "Output directory: data/factors" in text
    assert f"- {_SYMBOL} {_TIMEFRAME} {_YEAR}" in text


def test_main_exit_code_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """main returns 0 when no tasks fail."""
    with (
        patch("cqros.cli.generate_factors.build_factor_generation_pipeline") as build_pipeline,
        patch("cqros.cli.generate_factors.discover_work", return_value=()),
    ):
        build_pipeline.return_value = MagicMock(spec=FactorGenerationPipeline)
        code = _run(
            main(["--manager", "simple", "--storage-root", str(tmp_path), "--workers", "1"])
        )

    captured = capsys.readouterr()
    assert code == 0
    assert "CQROS Factor Generation Summary" in captured.out


def test_main_validation_error_exit_code() -> None:
    """Fatal CLI validation errors return exit code 1."""
    code = _run(main(["--manager", "simple", "--workers", "0"]))
    assert code == 1


def test_main_missing_manager_exit_code() -> None:
    """Missing required --manager exits via argparse."""
    with pytest.raises(SystemExit) as exc_info:
        _run(main([]))
    assert exc_info.value.code == 2


def test_configure_logging_debug(tmp_path: Path) -> None:
    """--debug enables DEBUG logging for the cqros logger."""
    with (
        patch("cqros.cli.generate_factors.build_factor_generation_pipeline") as build_pipeline,
        patch("cqros.cli.generate_factors.discover_work", return_value=()),
    ):
        build_pipeline.return_value = MagicMock(spec=FactorGenerationPipeline)
        _run(main(["--manager", "simple", "--storage-root", str(tmp_path), "--debug"]))
    assert logging.getLogger("cqros").level == logging.DEBUG


def test_factor_task_result_fields() -> None:
    """FactorTaskResult stores status metadata immutably."""
    result = FactorTaskResult(
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
        status="failed",
        error_type="RuntimeError",
        error_message="boom",
    )
    assert result.status == "failed"
    assert result.error_type == "RuntimeError"
    assert result.error_message == "boom"
