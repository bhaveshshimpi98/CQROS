"""Unit tests for CQROS portfolio-generation CLI."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

import cqros.cli.generate_portfolios as generate_portfolios_module
from cqros.cli.generate_portfolios import (
    DiscoveredWorkItem,
    PortfolioGenerationOptions,
    PortfolioGenerationSummary,
    PortfolioTaskResult,
    build_default_optimizer,
    build_optimizer_registry,
    build_options,
    build_parser,
    build_portfolio_pipeline,
    discover_work,
    format_summary,
    main,
    run_generation,
)
from cqros.config.models import ResearchConfig
from cqros.core.constants import (
    DEFAULT_STORAGE_ROOT,
    STORAGE_DIR_PORTFOLIOS,
    STORAGE_DIR_SIGNALS,
)
from cqros.core.exceptions import ValidationError
from cqros.portfolio import (
    EqualWeightOptimizer,
    OptimizerStrategy,
    PortfolioOptimizerRegistry,
    PortfolioPipeline,
)
from cqros.storage import (
    ParquetStore,
    PortfolioRepository,
    SignalRepository,
    StorageLayout,
)

_MODEL = "alpha-lgbm"
_VERSION = "1.0.0"
_OPTIMIZER = OptimizerStrategy.EQUAL_WEIGHT.value


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def _options(
    *,
    storage_root: Path,
    optimizer: str = _OPTIMIZER,
    model: str = _MODEL,
    version: str = _VERSION,
    symbols: tuple[str, ...] | None = None,
    timeframes: tuple[str, ...] | None = None,
    years: tuple[int, ...] | None = None,
    overwrite: bool = False,
    workers: int = ResearchConfig().worker_count,
    verbose: bool = False,
    debug: bool = False,
) -> PortfolioGenerationOptions:
    """Build options for tests against a temporary storage root."""
    return PortfolioGenerationOptions(
        storage_root=storage_root,
        optimizer=optimizer,
        model=model,
        version=version,
        symbols=symbols,
        timeframes=timeframes,
        years=years,
        overwrite=overwrite,
        workers=workers,
        verbose=verbose,
        debug=debug,
    )


def _touch_signal(
    root: Path,
    *,
    symbol: str,
    timeframe: str,
    year: int,
) -> Path:
    """Create an empty signal year partition path on disk."""
    path = (
        root
        / STORAGE_DIR_SIGNALS
        / "binance"
        / "usdt_perpetual"
        / symbol
        / timeframe
        / f"{year}.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _touch_portfolio(
    root: Path,
    *,
    symbol: str,
    timeframe: str,
    year: int,
) -> Path:
    """Create an empty portfolio year partition path on disk."""
    path = (
        root
        / STORAGE_DIR_PORTFOLIOS
        / "binance"
        / "usdt_perpetual"
        / symbol
        / timeframe
        / f"{year}.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_package_exports() -> None:
    """Public CLI symbols are exported through module ``__all__``."""
    expected = {
        "DiscoveredWorkItem",
        "PortfolioGenerationOptions",
        "PortfolioGenerationSummary",
        "PortfolioTaskResult",
        "build_default_optimizer",
        "build_optimizer_registry",
        "build_options",
        "build_parser",
        "build_portfolio_pipeline",
        "discover_work",
        "format_summary",
        "main",
        "run_generation",
    }
    assert expected.issubset(set(generate_portfolios_module.__all__))
    assert generate_portfolios_module.build_parser is build_parser
    assert generate_portfolios_module.main is main


def test_build_parser_requires_model_and_version() -> None:
    """--model and --version are required flags."""
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_build_parser_accepts_all_flags() -> None:
    """Parser accepts every documented portfolio-generation flag."""
    args = build_parser().parse_args(
        [
            "--optimizer",
            _OPTIMIZER,
            "--model",
            _MODEL,
            "--version",
            _VERSION,
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
    assert args.optimizer == _OPTIMIZER
    assert args.model == _MODEL
    assert args.version == _VERSION
    assert args.symbols == ["BTCUSDT", "ETHUSDT"]
    assert args.timeframes == ["1h", "4h"]
    assert args.years == ["2024", "2025"]
    assert args.overwrite is True
    assert args.workers == 2
    assert args.verbose is True
    assert args.debug is True


def test_build_options_maps_filters() -> None:
    """Explicit CLI flags map onto PortfolioGenerationOptions."""
    options = build_options(
        build_parser().parse_args(
            [
                "--optimizer",
                "fixed_weight",
                "--model",
                "beta",
                "--version",
                "2.0.0",
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
    assert options.storage_root == Path(DEFAULT_STORAGE_ROOT)
    assert options.optimizer == "fixed_weight"
    assert options.model == "beta"
    assert options.version == "2.0.0"
    assert options.symbols == ("ETHUSDT",)
    assert options.timeframes == ("1d",)
    assert options.years == (2023,)
    assert options.overwrite is True
    assert options.workers == 8
    assert options.debug is True


def test_build_options_defaults_optimizer() -> None:
    """--optimizer defaults to equal_weight when omitted."""
    options = build_options(build_parser().parse_args(["--model", _MODEL, "--version", _VERSION]))
    assert options.optimizer == _OPTIMIZER


def test_build_options_rejects_non_positive_workers() -> None:
    """Non-positive --workers fails validation."""
    args = build_parser().parse_args(["--model", _MODEL, "--version", _VERSION, "--workers", "0"])
    with pytest.raises(ValidationError, match="workers must be greater than 0"):
        build_options(args)


def test_build_options_rejects_unsupported_timeframe() -> None:
    """Unsupported --timeframes values fail validation."""
    args = build_parser().parse_args(
        ["--model", _MODEL, "--version", _VERSION, "--timeframes", "2x"]
    )
    with pytest.raises(ValidationError, match="unsupported timeframe"):
        build_options(args)


def test_build_options_rejects_invalid_year() -> None:
    """Non-integer --years values fail validation."""
    args = build_parser().parse_args(["--model", _MODEL, "--version", _VERSION, "--years", "abc"])
    with pytest.raises(ValidationError, match="invalid year"):
        build_options(args)


def test_discover_work_finds_signal_partitions(tmp_path: Path) -> None:
    """Discovery walks signal partitions without hardcoding symbols."""
    _touch_signal(tmp_path, symbol="BTCUSDT", timeframe="1h", year=2024)
    _touch_signal(tmp_path, symbol="ETHUSDT", timeframe="1h", year=2023)
    _touch_signal(tmp_path, symbol="ETHUSDT", timeframe="1h", year=2024)

    signals = SignalRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(signals, _options(storage_root=tmp_path))

    assert len(work) == 2
    assert work[0].symbol == "BTCUSDT"
    assert work[0].years == (2024,)
    assert work[1].symbol == "ETHUSDT"
    assert work[1].years == (2023, 2024)


def test_discover_work_filters_symbols_timeframes_years(tmp_path: Path) -> None:
    """Symbol, timeframe, and year filters apply to discovery."""
    _touch_signal(tmp_path, symbol="BTCUSDT", timeframe="1h", year=2024)
    _touch_signal(tmp_path, symbol="BTCUSDT", timeframe="4h", year=2024)
    _touch_signal(tmp_path, symbol="ETHUSDT", timeframe="1h", year=2024)
    _touch_signal(tmp_path, symbol="BTCUSDT", timeframe="1h", year=2025)

    signals = SignalRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(
        signals,
        _options(
            storage_root=tmp_path,
            symbols=("BTCUSDT",),
            timeframes=("1h",),
            years=(2024,),
        ),
    )

    assert work == (DiscoveredWorkItem(symbol="BTCUSDT", timeframe="1h", years=(2024,)),)


def test_discover_work_skips_missing_signal_partitions(tmp_path: Path) -> None:
    """Missing signal trees yield empty discovery instead of partial work."""
    signals = SignalRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(signals, _options(storage_root=tmp_path))
    assert work == ()


def test_run_generation_empty_work(tmp_path: Path) -> None:
    """Empty discovery produces a zeroed summary without invoking the pipeline."""
    pipeline = MagicMock(spec=PortfolioPipeline)
    signals = MagicMock(spec=SignalRepository)
    portfolios = MagicMock(spec=PortfolioRepository)
    summary = _run(
        run_generation(
            pipeline=pipeline,
            signal_repository=signals,
            portfolio_repository=portfolios,
            options=_options(storage_root=tmp_path),
            work=(),
        )
    )
    assert summary.successful_tasks == 0
    assert summary.failed_tasks == 0
    assert summary.skipped_tasks == 0
    assert summary.optimizer == _OPTIMIZER
    assert summary.model == _MODEL
    assert summary.version == _VERSION
    pipeline.run.assert_not_called()


def test_run_generation_skips_existing_without_overwrite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Existing portfolio partitions are skipped unless --overwrite is set."""
    _touch_portfolio(tmp_path, symbol="BTCUSDT", timeframe="1h", year=2024)
    work = (DiscoveredWorkItem(symbol="BTCUSDT", timeframe="1h", years=(2024,)),)

    pipeline = MagicMock(spec=PortfolioPipeline)
    signals = MagicMock(spec=SignalRepository)
    portfolios = MagicMock(spec=PortfolioRepository)
    portfolios.exists.return_value = True

    summary = _run(
        run_generation(
            pipeline=pipeline,
            signal_repository=signals,
            portfolio_repository=portfolios,
            options=_options(storage_root=tmp_path, workers=1),
            work=work,
        )
    )

    captured = capsys.readouterr().out
    assert summary.skipped_tasks == 1
    assert summary.successful_tasks == 0
    pipeline.run.assert_not_called()
    portfolios.save.assert_not_called()
    assert "SKIP BTCUSDT 1h 2024" in captured


def test_run_generation_overwrite_existing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--overwrite regenerates partitions that already exist."""
    work = (DiscoveredWorkItem(symbol="BTCUSDT", timeframe="1h", years=(2024,)),)
    pipeline = MagicMock(spec=PortfolioPipeline)
    pipeline.run.return_value = pl.DataFrame({"open_time": [1, 2]})
    signals = MagicMock(spec=SignalRepository)
    signals.load.return_value = pl.DataFrame(
        {
            "model_name": [_MODEL, _MODEL],
            "model_version": [_VERSION, _VERSION],
            "open_time": [1, 2],
        }
    )
    portfolios = MagicMock(spec=PortfolioRepository)
    portfolios.exists.return_value = True

    summary = _run(
        run_generation(
            pipeline=pipeline,
            signal_repository=signals,
            portfolio_repository=portfolios,
            options=_options(storage_root=tmp_path, workers=1, overwrite=True),
            work=work,
        )
    )

    captured = capsys.readouterr().out
    assert summary.successful_tasks == 1
    assert summary.skipped_tasks == 0
    assert summary.rows_generated == 2
    pipeline.run.assert_called_once()
    portfolios.save.assert_called_once()
    assert "OK BTCUSDT 1h 2024 rows=2" in captured


def test_run_generation_success_and_progress(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Successful generation increments counters and prints progress."""
    work = (DiscoveredWorkItem(symbol="BTCUSDT", timeframe="1h", years=(2024,)),)
    pipeline = MagicMock(spec=PortfolioPipeline)
    pipeline.run.return_value = pl.DataFrame({"open_time": [1, 2, 3]})
    signals = MagicMock(spec=SignalRepository)
    signals.load.return_value = pl.DataFrame(
        {
            "model_name": [_MODEL, _MODEL, "other"],
            "model_version": [_VERSION, _VERSION, "9.9.9"],
            "open_time": [1, 2, 3],
        }
    )
    portfolios = MagicMock(spec=PortfolioRepository)
    portfolios.exists.return_value = False

    summary = _run(
        run_generation(
            pipeline=pipeline,
            signal_repository=signals,
            portfolio_repository=portfolios,
            options=_options(storage_root=tmp_path, workers=1, overwrite=True),
            work=work,
        )
    )

    captured = capsys.readouterr().out
    assert summary.successful_tasks == 1
    assert summary.rows_generated == 3
    assert "OK BTCUSDT 1h 2024 rows=3" in captured
    pipeline.run.assert_called_once()
    filtered = pipeline.run.call_args.args[1]
    assert filtered.height == 2
    assert set(filtered.get_column("model_name").to_list()) == {_MODEL}


def test_run_generation_pipeline_failure_isolation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failed year does not prevent later years from running."""
    work = (DiscoveredWorkItem(symbol="BTCUSDT", timeframe="1h", years=(2024, 2025)),)
    pipeline = MagicMock(spec=PortfolioPipeline)
    pipeline.run.side_effect = [RuntimeError("boom"), pl.DataFrame({"open_time": [1]})]
    signals = MagicMock(spec=SignalRepository)
    signals.load.return_value = pl.DataFrame(
        {
            "model_name": [_MODEL],
            "model_version": [_VERSION],
            "open_time": [1],
        }
    )
    portfolios = MagicMock(spec=PortfolioRepository)
    portfolios.exists.return_value = False

    summary = _run(
        run_generation(
            pipeline=pipeline,
            signal_repository=signals,
            portfolio_repository=portfolios,
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


def test_format_summary_includes_optimizer_model_and_failed_tasks() -> None:
    """Summary rendering includes optimizer identity and failed-task labels."""
    text = format_summary(
        PortfolioGenerationSummary(
            optimizer=_OPTIMIZER,
            model=_MODEL,
            version=_VERSION,
            symbols_discovered=1,
            symbols_processed=1,
            timeframes_processed=1,
            successful_tasks=0,
            failed_tasks=1,
            skipped_tasks=0,
            rows_generated=0,
            duration_seconds=1.5,
            output_directory=Path("data/portfolios"),
            failed_task_labels=("BTCUSDT 1h 2024",),
        )
    )
    assert "CQROS Portfolio Generation Summary" in text
    assert f"Optimizer: {_OPTIMIZER}" in text
    assert f"Model: {_MODEL}" in text
    assert f"Version: {_VERSION}" in text
    assert "Failed tasks: 1" in text
    assert "- BTCUSDT 1h 2024" in text
    assert "Output directory: data/portfolios" in text


def test_main_exit_code_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """main returns 0 when no tasks fail."""
    with (
        patch("cqros.cli.generate_portfolios.DEFAULT_STORAGE_ROOT", str(tmp_path)),
        patch("cqros.cli.generate_portfolios.build_portfolio_pipeline") as build_pipeline,
        patch("cqros.cli.generate_portfolios.discover_work", return_value=()),
    ):
        build_pipeline.return_value = MagicMock(spec=PortfolioPipeline)
        code = _run(main(["--model", _MODEL, "--version", _VERSION, "--workers", "1"]))

    captured = capsys.readouterr()
    assert code == 0
    assert "CQROS Portfolio Generation Summary" in captured.out


def test_main_validation_error_exit_code() -> None:
    """Fatal CLI validation errors return exit code 1."""
    code = _run(main(["--model", _MODEL, "--version", _VERSION, "--workers", "0"]))
    assert code == 1


def test_configure_logging_verbose(tmp_path: Path) -> None:
    """--verbose enables INFO logging for the cqros logger."""
    with (
        patch("cqros.cli.generate_portfolios.DEFAULT_STORAGE_ROOT", str(tmp_path)),
        patch("cqros.cli.generate_portfolios.build_portfolio_pipeline") as build_pipeline,
        patch("cqros.cli.generate_portfolios.discover_work", return_value=()),
    ):
        build_pipeline.return_value = MagicMock(spec=PortfolioPipeline)
        _run(main(["--model", _MODEL, "--version", _VERSION, "--verbose"]))
    assert logging.getLogger("cqros").level == logging.INFO


def test_configure_logging_debug(tmp_path: Path) -> None:
    """--debug enables DEBUG logging for the cqros logger."""
    with (
        patch("cqros.cli.generate_portfolios.DEFAULT_STORAGE_ROOT", str(tmp_path)),
        patch("cqros.cli.generate_portfolios.build_portfolio_pipeline") as build_pipeline,
        patch("cqros.cli.generate_portfolios.discover_work", return_value=()),
    ):
        build_pipeline.return_value = MagicMock(spec=PortfolioPipeline)
        _run(main(["--model", _MODEL, "--version", _VERSION, "--debug"]))
    assert logging.getLogger("cqros").level == logging.DEBUG


def test_portfolio_task_result_fields() -> None:
    """PortfolioTaskResult stores status metadata immutably."""
    result = PortfolioTaskResult(
        symbol="BTCUSDT",
        timeframe="1h",
        year=2024,
        status="failed",
        error_type="RuntimeError",
        error_message="boom",
    )
    assert result.status == "failed"
    assert result.error_type == "RuntimeError"


def test_build_default_optimizer_returns_equal_weight() -> None:
    """Default composition-root optimizer is EqualWeightOptimizer."""
    optimizer = build_default_optimizer()
    assert isinstance(optimizer, EqualWeightOptimizer)


def test_build_optimizer_registry_registers_equal_weight() -> None:
    """Default registry exposes EqualWeightOptimizer under equal_weight."""
    registry = build_optimizer_registry()
    assert registry.exists(_OPTIMIZER)
    assert isinstance(registry.get(_OPTIMIZER), EqualWeightOptimizer)


def test_build_portfolio_pipeline_wires_registry(tmp_path: Path) -> None:
    """Pipeline composition injects the supplied optimizer registry."""
    registry = PortfolioOptimizerRegistry()
    registry.register(_OPTIMIZER, EqualWeightOptimizer())

    with patch("cqros.cli.generate_portfolios.PortfolioPipeline") as pipeline_cls:
        pipeline_cls.return_value = MagicMock(spec=PortfolioPipeline)
        build_portfolio_pipeline(
            _options(storage_root=tmp_path),
            optimizer_registry=registry,
        )

    pipeline_cls.assert_called_once_with(registry)
