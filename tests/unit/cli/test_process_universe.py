"""Unit tests for CQROS processing universe CLI."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from cqros.cli.process_universe import (
    CLI_DATASETS,
    DEFAULT_PROCESS_UNIVERSE_WORKERS,
    ProcessUniverseOptions,
    ProcessUniverseSummary,
    build_options,
    build_parser,
    build_processing_runner,
    discover_work,
    format_summary,
    main,
    run_universe,
)
from cqros.core.constants import DEFAULT_STORAGE_ROOT, STORAGE_DIR_RAW
from cqros.core.exceptions import ValidationError
from cqros.processing import (
    CleaningReport,
    FundingCleaner,
    LongShortCleaner,
    OHLCVCleaner,
    OpenInterestCleaner,
    ProcessingPipeline,
    ProcessingRunner,
    ProcessingSummary,
    ProcessingTaskResult,
    TakerVolumeCleaner,
)
from cqros.storage import (
    MarketDataRepository,
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
    symbol: str | None = None,
    timeframes: tuple[str, ...] | None = None,
    datasets: tuple[str, ...] | None = None,
    workers: int = DEFAULT_PROCESS_UNIVERSE_WORKERS,
    verbose: bool = False,
    dry_run: bool = False,
) -> ProcessUniverseOptions:
    """Build options for tests against a temporary storage root."""
    return ProcessUniverseOptions(
        storage_root=storage_root,
        symbol=symbol,
        timeframes=timeframes,
        datasets=datasets,
        workers=workers,
        verbose=verbose,
        dry_run=dry_run,
    )


def _touch_partition(
    root: Path,
    *,
    dataset: str,
    symbol: str,
    timeframe: str,
    year: int,
) -> Path:
    """Create an empty raw year partition path on disk."""
    path = (
        root
        / STORAGE_DIR_RAW
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


def test_build_parser_defaults() -> None:
    """Omitted optional flags keep discovery defaults."""
    args = build_parser().parse_args([])
    assert args.symbol is None
    assert args.timeframes is None
    assert args.datasets is None
    assert args.workers == DEFAULT_PROCESS_UNIVERSE_WORKERS
    assert args.verbose is False
    assert args.dry_run is False


def test_build_parser_accepts_all_flags() -> None:
    """Parser accepts every documented processing-universe flag."""
    args = build_parser().parse_args(
        [
            "--symbol",
            "BTCUSDT",
            "--timeframe",
            "1h",
            "--timeframe",
            "4h",
            "--dataset",
            "ohlcv",
            "--dataset",
            "funding",
            "--workers",
            "2",
            "--verbose",
            "--dry-run",
        ]
    )
    assert args.symbol == "BTCUSDT"
    assert args.timeframes == ["1h", "4h"]
    assert args.datasets == ["ohlcv", "funding"]
    assert args.workers == 2
    assert args.verbose is True
    assert args.dry_run is True


def test_build_parser_rejects_unknown_dataset() -> None:
    """Unsupported --dataset values are rejected by argparse."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--dataset", "liquidation"])


def test_build_options_defaults() -> None:
    """Omitted filters map to discovery-all options."""
    options = build_options(build_parser().parse_args([]))
    assert options.storage_root == Path(DEFAULT_STORAGE_ROOT)
    assert options.symbol is None
    assert options.timeframes is None
    assert options.datasets is None
    assert options.workers == DEFAULT_PROCESS_UNIVERSE_WORKERS
    assert options.verbose is False
    assert options.dry_run is False


def test_build_options_maps_filters() -> None:
    """Explicit CLI flags map onto ProcessUniverseOptions."""
    options = build_options(
        build_parser().parse_args(
            [
                "--symbol",
                "ETHUSDT",
                "--timeframe",
                "1d",
                "--dataset",
                "taker_volume",
                "--workers",
                "8",
                "--verbose",
                "--dry-run",
            ]
        )
    )
    assert options.symbol == "ETHUSDT"
    assert options.timeframes == ("1d",)
    assert options.datasets == ("taker_volume",)
    assert options.workers == 8
    assert options.verbose is True
    assert options.dry_run is True


def test_build_options_rejects_non_positive_workers() -> None:
    """Non-positive --workers fails validation."""
    args = build_parser().parse_args(["--workers", "0"])
    with pytest.raises(ValidationError, match="workers must be greater than 0"):
        build_options(args)


def test_build_options_rejects_unsupported_timeframe() -> None:
    """Unsupported --timeframe values fail validation."""
    args = build_parser().parse_args(["--timeframe", "2x"])
    with pytest.raises(ValidationError, match="unsupported timeframe"):
        build_options(args)


def test_discover_work_finds_symbols_and_years(tmp_path: Path) -> None:
    """Discovery walks raw partitions without hardcoding symbols."""
    _touch_partition(tmp_path, dataset="ohlcv", symbol="BTCUSDT", timeframe="1h", year=2024)
    _touch_partition(tmp_path, dataset="ohlcv", symbol="ETHUSDT", timeframe="1h", year=2023)
    _touch_partition(tmp_path, dataset="ohlcv", symbol="ETHUSDT", timeframe="1h", year=2024)

    work = discover_work(
        StorageLayout(tmp_path), _options(storage_root=tmp_path, datasets=("ohlcv",))
    )

    assert len(work) == 2
    assert work[0].symbol == "BTCUSDT"
    assert work[0].years == (2024,)
    assert work[1].symbol == "ETHUSDT"
    assert work[1].years == (2023, 2024)


def test_discover_work_filters_symbol(tmp_path: Path) -> None:
    """--symbol limits discovery to one symbol."""
    _touch_partition(tmp_path, dataset="ohlcv", symbol="BTCUSDT", timeframe="1h", year=2024)
    _touch_partition(tmp_path, dataset="ohlcv", symbol="ETHUSDT", timeframe="1h", year=2024)

    work = discover_work(
        StorageLayout(tmp_path),
        _options(storage_root=tmp_path, symbol="BTCUSDT", datasets=("ohlcv",)),
    )

    assert len(work) == 1
    assert work[0].symbol == "BTCUSDT"


def test_discover_work_filters_timeframe(tmp_path: Path) -> None:
    """--timeframe limits discovery to requested intervals."""
    _touch_partition(tmp_path, dataset="ohlcv", symbol="BTCUSDT", timeframe="1h", year=2024)
    _touch_partition(tmp_path, dataset="ohlcv", symbol="BTCUSDT", timeframe="4h", year=2024)
    _touch_partition(tmp_path, dataset="ohlcv", symbol="BTCUSDT", timeframe="1d", year=2024)

    work = discover_work(
        StorageLayout(tmp_path),
        _options(
            storage_root=tmp_path,
            datasets=("ohlcv",),
            timeframes=("1h", "1d"),
        ),
    )

    assert {item.timeframe for item in work} == {"1h", "1d"}
    assert len(work) == 2


def test_discover_work_filters_dataset_and_skips_missing(tmp_path: Path) -> None:
    """Dataset filters apply and missing dataset trees are skipped."""
    _touch_partition(tmp_path, dataset="ohlcv", symbol="BTCUSDT", timeframe="1h", year=2024)
    _touch_partition(tmp_path, dataset="funding", symbol="BTCUSDT", timeframe="8h", year=2024)

    work = discover_work(
        StorageLayout(tmp_path),
        _options(storage_root=tmp_path, datasets=("funding", "open_interest")),
    )

    assert len(work) == 1
    assert work[0].cli_dataset == "funding"
    assert work[0].storage_dataset == "funding"


def test_discover_work_expands_long_short(tmp_path: Path) -> None:
    """CLI long_short expands to the three ratio storage datasets."""
    _touch_partition(
        tmp_path,
        dataset="global_long_short_account_ratio",
        symbol="BTCUSDT",
        timeframe="1h",
        year=2024,
    )
    _touch_partition(
        tmp_path,
        dataset="top_long_short_position_ratio",
        symbol="BTCUSDT",
        timeframe="1h",
        year=2024,
    )

    work = discover_work(
        StorageLayout(tmp_path),
        _options(storage_root=tmp_path, datasets=("long_short",)),
    )

    assert {item.storage_dataset for item in work} == {
        "global_long_short_account_ratio",
        "top_long_short_position_ratio",
    }
    assert all(item.cli_dataset == "long_short" for item in work)


def test_build_processing_runner_wires_dependencies(tmp_path: Path) -> None:
    """Dependency construction wires repositories, pipeline, and cleaners once."""
    options = _options(storage_root=tmp_path)
    with (
        patch("cqros.cli.process_universe.StorageLayout", wraps=StorageLayout) as layout_cls,
        patch("cqros.cli.process_universe.ParquetStore", wraps=ParquetStore) as store_cls,
        patch(
            "cqros.cli.process_universe.MarketDataRepository",
            wraps=MarketDataRepository,
        ) as raw_cls,
        patch(
            "cqros.cli.process_universe.ProcessedMarketDataRepository",
            wraps=ProcessedMarketDataRepository,
        ) as processed_cls,
        patch(
            "cqros.cli.process_universe.ProcessingPipeline",
            wraps=ProcessingPipeline,
        ) as pipeline_cls,
        patch("cqros.cli.process_universe.OHLCVCleaner", wraps=OHLCVCleaner) as ohlcv_cls,
        patch("cqros.cli.process_universe.FundingCleaner", wraps=FundingCleaner) as funding_cls,
        patch(
            "cqros.cli.process_universe.OpenInterestCleaner",
            wraps=OpenInterestCleaner,
        ) as oi_cls,
        patch(
            "cqros.cli.process_universe.TakerVolumeCleaner",
            wraps=TakerVolumeCleaner,
        ) as taker_cls,
        patch(
            "cqros.cli.process_universe.LongShortCleaner",
            wraps=LongShortCleaner,
        ) as ls_cls,
        patch(
            "cqros.cli.process_universe.ProcessingRunner",
            wraps=ProcessingRunner,
        ) as runner_cls,
    ):
        runner = build_processing_runner(options)

    assert isinstance(runner, ProcessingRunner)
    layout_cls.assert_called_once_with(tmp_path)
    store_cls.assert_called_once_with()
    assert raw_cls.call_count == 1
    assert processed_cls.call_count == 1
    pipeline_cls.assert_called_once_with(())
    ohlcv_cls.assert_called_once_with()
    funding_cls.assert_called_once_with()
    oi_cls.assert_called_once_with()
    taker_cls.assert_called_once_with()
    ls_cls.assert_called_once_with()
    assert runner_cls.call_count == 1
    kwargs = runner_cls.call_args.kwargs
    assert "ohlcv_cleaner" in kwargs
    assert "funding_cleaner" in kwargs
    assert "open_interest_cleaner" in kwargs
    assert "taker_volume_cleaner" in kwargs
    assert "long_short_cleaner" in kwargs


def test_dry_run_discovers_without_invoking_runner(tmp_path: Path) -> None:
    """Dry-run discovers work and never calls ProcessingRunner methods."""
    _touch_partition(tmp_path, dataset="ohlcv", symbol="BTCUSDT", timeframe="1h", year=2024)
    options = _options(storage_root=tmp_path, datasets=("ohlcv",), dry_run=True)
    work = discover_work(StorageLayout(tmp_path), options)
    runner = MagicMock(spec=ProcessingRunner)

    summary = _run(run_universe(runner=runner, options=options, work=work))

    assert summary.dry_run is True
    assert summary.symbols_discovered == 1
    assert summary.symbols_processed == 0
    assert summary.successful_tasks == 0
    runner.process_ohlcv.assert_not_called()


def test_run_universe_invokes_runner_per_work_item(tmp_path: Path) -> None:
    """Each discovered symbol/dataset/timeframe invokes ProcessingRunner."""
    _touch_partition(tmp_path, dataset="ohlcv", symbol="BTCUSDT", timeframe="1h", year=2024)
    _touch_partition(tmp_path, dataset="funding", symbol="BTCUSDT", timeframe="8h", year=2024)
    options = _options(
        storage_root=tmp_path,
        datasets=("ohlcv", "funding"),
        workers=1,
    )
    work = discover_work(StorageLayout(tmp_path), options)
    runner = MagicMock(spec=ProcessingRunner)
    runner.process_ohlcv.return_value = ProcessingSummary(
        dataset="ohlcv",
        exchange="binance",
        market="usdt_perpetual",
        results=(
            ProcessingTaskResult(
                symbol="BTCUSDT",
                timeframe="1h",
                year=2024,
                status="succeeded",
                rows_loaded=10,
                rows_saved=9,
                cleaning_report=_cleaning_report(rows_before=10, rows_after=9),
            ),
        ),
    )
    runner.process_funding.return_value = ProcessingSummary(
        dataset="funding",
        exchange="binance",
        market="usdt_perpetual",
        results=(
            ProcessingTaskResult(
                symbol="BTCUSDT",
                timeframe="8h",
                year=2024,
                status="succeeded",
                rows_loaded=5,
                rows_saved=5,
                cleaning_report=_cleaning_report(rows_before=5, rows_after=5),
            ),
        ),
    )

    summary = _run(run_universe(runner=runner, options=options, work=work))

    runner.process_ohlcv.assert_called_once_with(
        symbols=("BTCUSDT",),
        timeframes=("1h",),
        years=(2024,),
        exchange="binance",
        market="usdt_perpetual",
    )
    runner.process_funding.assert_called_once_with(
        symbols=("BTCUSDT",),
        timeframes=("8h",),
        years=(2024,),
        exchange="binance",
        market="usdt_perpetual",
    )
    assert summary.successful_tasks == 2
    assert summary.rows_processed == 15
    assert summary.rows_removed == 1
    assert summary.symbols_processed == 1
    assert summary.datasets_processed == 2


def test_failure_isolation_continues_remaining_work(tmp_path: Path) -> None:
    """One failing work item does not stop remaining universe processing."""
    _touch_partition(tmp_path, dataset="ohlcv", symbol="BTCUSDT", timeframe="1h", year=2024)
    _touch_partition(tmp_path, dataset="ohlcv", symbol="ETHUSDT", timeframe="1h", year=2024)
    options = _options(storage_root=tmp_path, datasets=("ohlcv",), workers=1)
    work = discover_work(StorageLayout(tmp_path), options)
    runner = MagicMock(spec=ProcessingRunner)

    def _process_ohlcv(**kwargs: object) -> ProcessingSummary:
        symbols_obj = kwargs["symbols"]
        assert isinstance(symbols_obj, tuple)
        symbols = cast(tuple[str, ...], symbols_obj)
        symbol = symbols[0]
        if symbol == "BTCUSDT":
            raise RuntimeError("boom")
        return ProcessingSummary(
            dataset="ohlcv",
            exchange="binance",
            market="usdt_perpetual",
            results=(
                ProcessingTaskResult(
                    symbol="ETHUSDT",
                    timeframe="1h",
                    year=2024,
                    status="succeeded",
                    rows_loaded=3,
                    rows_saved=3,
                    cleaning_report=_cleaning_report(rows_before=3, rows_after=3),
                ),
            ),
        )

    runner.process_ohlcv.side_effect = _process_ohlcv

    summary = _run(run_universe(runner=runner, options=options, work=work))

    assert summary.successful_tasks == 1
    assert summary.failed_tasks == 1
    assert summary.failed_task_labels == ("BTCUSDT 1h ohlcv",)


def test_worker_limit_spawns_bounded_workers(tmp_path: Path) -> None:
    """Worker pool creates exactly worker_count concurrent workers."""
    for symbol in ("AAAUSDT", "BBBUSDT", "CCCUSDT"):
        _touch_partition(tmp_path, dataset="ohlcv", symbol=symbol, timeframe="1h", year=2024)
    options = _options(storage_root=tmp_path, datasets=("ohlcv",), workers=2)
    work = discover_work(StorageLayout(tmp_path), options)
    runner = MagicMock(spec=ProcessingRunner)
    runner.process_ohlcv.return_value = ProcessingSummary(
        dataset="ohlcv",
        exchange="binance",
        market="usdt_perpetual",
        results=(),
    )

    created: list[asyncio.Task[None]] = []
    real_create_task = asyncio.create_task

    def _track_create_task(
        coro: Coroutine[object, object, None],
        **kwargs: object,
    ) -> asyncio.Task[None]:
        task = real_create_task(coro, **kwargs)  # type: ignore[arg-type]
        created.append(task)
        return task

    with patch("cqros.cli.process_universe.asyncio.create_task", side_effect=_track_create_task):
        _run(run_universe(runner=runner, options=options, work=work))

    assert len(created) == 2


def test_format_summary_deterministic_report() -> None:
    """Summary rendering matches the production report contract."""
    summary = ProcessUniverseSummary(
        symbols_discovered=2,
        symbols_processed=2,
        datasets_processed=1,
        timeframes_processed=2,
        successful_tasks=3,
        failed_tasks=2,
        rows_processed=100,
        rows_removed=4,
        duration_seconds=1.5,
        output_directory=Path("data") / "processed",
        failed_task_labels=("BTCUSDT 4h funding", "ETHUSDT 1d ohlcv"),
        dry_run=False,
    )

    report = format_summary(summary)

    assert report.startswith("=====================================\nCQROS Processing Summary\n")
    assert "Symbols discovered: 2" in report
    assert "Symbols processed: 2" in report
    assert "Datasets processed: 1" in report
    assert "Timeframes processed: 2" in report
    assert "Successful tasks: 3" in report
    assert "Failed tasks: 2" in report
    assert "Rows processed: 100" in report
    assert "Rows removed: 4" in report
    assert "Processing duration: 1.500s" in report
    assert "Output directory: data/processed" in report
    assert "Failed Tasks" in report
    assert "- BTCUSDT 4h funding" in report
    assert "- ETHUSDT 1d ohlcv" in report


def test_cli_datasets_match_contract() -> None:
    """Allowed CLI datasets match the documented production set."""
    assert CLI_DATASETS == (
        "ohlcv",
        "funding",
        "open_interest",
        "taker_volume",
        "long_short",
    )


def test_main_prints_summary_and_returns_success(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """main discovers work, runs the universe, and prints the summary."""
    _touch_partition(tmp_path, dataset="ohlcv", symbol="BTCUSDT", timeframe="1h", year=2024)
    options = _options(storage_root=tmp_path, datasets=("ohlcv",), dry_run=True)

    with (
        patch("cqros.cli.process_universe.build_options", return_value=options),
        patch(
            "cqros.cli.process_universe.build_processing_runner",
            return_value=MagicMock(spec=ProcessingRunner),
        ),
    ):
        exit_code = _run(main(["--dry-run", "--dataset", "ohlcv"]))

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "CQROS Processing Summary" in captured.out
    assert "Symbols discovered: 1" in captured.out


def test_main_returns_failure_on_validation_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """main returns exit code 1 when options validation fails."""
    exit_code = _run(main(["--workers", "0"]))
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "workers must be greater than 0" in captured.err


def test_verbose_enables_info_logging(tmp_path: Path) -> None:
    """--verbose configures INFO logging for the cqros logger tree."""
    options = _options(storage_root=tmp_path, verbose=True, dry_run=True)
    with (
        patch("cqros.cli.process_universe.build_options", return_value=options),
        patch(
            "cqros.cli.process_universe.build_processing_runner",
            return_value=MagicMock(spec=ProcessingRunner),
        ),
        patch("cqros.cli.process_universe.logging.basicConfig") as basic_config,
    ):
        _run(main(["--verbose", "--dry-run"]))

    basic_config.assert_called_once()
    assert basic_config.call_args.kwargs["level"] == logging.INFO
    assert logging.getLogger("cqros").level == logging.INFO


def _cleaning_report(*, rows_before: int, rows_after: int) -> CleaningReport:
    """Build a CleaningReport for summary aggregation tests."""
    return CleaningReport(
        rows_before=rows_before,
        rows_after=rows_after,
        duplicates_removed=0,
        null_rows_removed=0,
        invalid_price_rows_removed=0,
        invalid_volume_rows_removed=0,
        invalid_trade_count_rows_removed=0,
        invalid_timestamp_rows_removed=0,
        warnings=(),
    )
