"""Unit tests for CQROS regression signal threshold calibration CLI."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

import cqros.cli.calibrate_signal_thresholds as calibrate_module
from cqros.cli.calibrate_signal_thresholds import (
    CalibrationOptions,
    CalibrationSummary,
    DiscoveredWorkItem,
    build_options,
    build_parser,
    discover_work,
    format_group_report,
    format_summary,
    main,
    run_calibration,
)
from cqros.config.models import ResearchConfig
from cqros.core.constants import DEFAULT_STORAGE_ROOT, STORAGE_DIR_PREDICTIONS
from cqros.core.exceptions import ValidationError
from cqros.research.signal_threshold_calibrator import (
    PredictionDistributionStatistics,
    SignalThresholdCalibrator,
    SymbolTimeframeCalibration,
    ThresholdCalibrationResult,
    ThresholdRecommendation,
)
from cqros.storage import ParquetStore, PredictionRepository, StorageLayout

_MODEL = "alpha-lgbm"
_VERSION = "1.0.0"
_FRAMEWORK = "lightgbm"


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def _options(
    *,
    storage_root: Path,
    model: str = _MODEL,
    version: str = _VERSION,
    symbols: tuple[str, ...] | None = None,
    timeframes: tuple[str, ...] | None = None,
    years: tuple[int, ...] | None = None,
    workers: int = ResearchConfig().worker_count,
    verbose: bool = False,
    debug: bool = False,
) -> CalibrationOptions:
    """Build options for tests against a temporary storage root."""
    return CalibrationOptions(
        storage_root=storage_root,
        model=model,
        version=version,
        symbols=symbols,
        timeframes=timeframes,
        years=years,
        workers=workers,
        verbose=verbose,
        debug=debug,
    )


def _touch_prediction(
    root: Path,
    *,
    symbol: str,
    timeframe: str,
    year: int,
    framework: str = _FRAMEWORK,
    model: str = _MODEL,
    version: str = _VERSION,
) -> Path:
    """Create an empty prediction year partition path on disk."""
    path = (
        root
        / STORAGE_DIR_PREDICTIONS
        / framework
        / model
        / version
        / "binance"
        / "usdt_perpetual"
        / symbol
        / timeframe
        / f"{year}.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _prediction_frame(values: list[float], *, symbol: str = "BTCUSDT") -> pl.DataFrame:
    """Build a canonical-ish prediction frame for calibration tests."""
    count = len(values)
    return pl.DataFrame(
        {
            "symbol": [symbol] * count,
            "timeframe": ["1h"] * count,
            "open_time": list(range(count)),
            "model_name": [_MODEL] * count,
            "model_version": [_VERSION] * count,
            "prediction": values,
        }
    )


def _statistics(**overrides: object) -> PredictionDistributionStatistics:
    """Build distribution statistics with optional field overrides."""
    payload: dict[str, object] = {
        "count": 100,
        "minimum": -1.0,
        "maximum": 1.0,
        "mean": 0.0,
        "std": 0.5,
        "median": 0.0,
        "percentile_01": -0.9,
        "percentile_025": -0.8,
        "percentile_05": -0.7,
        "percentile_10": -0.5,
        "percentile_90": 0.5,
        "percentile_95": 0.7,
        "percentile_975": 0.8,
        "percentile_99": 0.9,
        "positive_ratio": 0.5,
        "negative_ratio": 0.5,
    }
    payload.update(overrides)
    return PredictionDistributionStatistics(**payload)  # type: ignore[arg-type]


def _recommendations() -> tuple[ThresholdRecommendation, ...]:
    """Build deterministic Conservative / Balanced / Active recommendations."""
    return (
        ThresholdRecommendation(
            profile="Conservative",
            buy_threshold=0.9,
            sell_threshold=-0.9,
            expected_buy_ratio=0.01,
            expected_sell_ratio=0.01,
            expected_hold_ratio=0.98,
        ),
        ThresholdRecommendation(
            profile="Balanced",
            buy_threshold=0.7,
            sell_threshold=-0.7,
            expected_buy_ratio=0.05,
            expected_sell_ratio=0.05,
            expected_hold_ratio=0.90,
        ),
        ThresholdRecommendation(
            profile="Active",
            buy_threshold=0.5,
            sell_threshold=-0.5,
            expected_buy_ratio=0.10,
            expected_sell_ratio=0.10,
            expected_hold_ratio=0.80,
        ),
    )


def _calibration_result() -> ThresholdCalibrationResult:
    """Build a deterministic aggregate calibration result."""
    stats = _statistics()
    recommendations = _recommendations()
    return ThresholdCalibrationResult(
        symbols_analyzed=("BTCUSDT", "ETHUSDT"),
        datasets_analyzed=2,
        rows_analyzed=200,
        global_statistics=stats,
        recommendations=recommendations,
        symbol_timeframe_results=(
            SymbolTimeframeCalibration(
                symbol="BTCUSDT",
                timeframe="1h",
                statistics=stats,
                recommendations=recommendations,
            ),
            SymbolTimeframeCalibration(
                symbol="ETHUSDT",
                timeframe="4h",
                statistics=stats,
                recommendations=recommendations,
            ),
        ),
    )


def test_package_exports() -> None:
    """Public CLI symbols are exported through module ``__all__``."""
    expected = {
        "CalibrationOptions",
        "CalibrationSummary",
        "DiscoveredWorkItem",
        "build_options",
        "build_parser",
        "discover_work",
        "format_group_report",
        "format_summary",
        "main",
        "run_calibration",
    }
    assert expected.issubset(set(calibrate_module.__all__))
    assert calibrate_module.build_parser is build_parser
    assert calibrate_module.main is main


def test_build_parser_requires_model_and_version() -> None:
    """--model and --version are required flags."""
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--model", _MODEL])


def test_build_parser_accepts_all_flags() -> None:
    """Parser accepts every documented calibration flag."""
    args = build_parser().parse_args(
        [
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
            "--workers",
            "2",
            "--verbose",
            "--debug",
        ]
    )
    assert args.model == _MODEL
    assert args.version == _VERSION
    assert args.symbols == ["BTCUSDT", "ETHUSDT"]
    assert args.timeframes == ["1h", "4h"]
    assert args.years == ["2024", "2025"]
    assert args.workers == 2
    assert args.verbose is True
    assert args.debug is True


def test_build_options_maps_filters() -> None:
    """Explicit CLI flags map onto CalibrationOptions."""
    options = build_options(
        build_parser().parse_args(
            [
                "--model",
                _MODEL,
                "--version",
                _VERSION,
                "--symbols",
                "ETHUSDT",
                "--timeframes",
                "4h",
                "--years",
                "2025",
                "--workers",
                "3",
                "--debug",
            ]
        )
    )
    assert options.storage_root == Path(DEFAULT_STORAGE_ROOT)
    assert options.model == _MODEL
    assert options.version == _VERSION
    assert options.symbols == ("ETHUSDT",)
    assert options.timeframes == ("4h",)
    assert options.years == (2025,)
    assert options.workers == 3
    assert options.debug is True


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


def test_build_options_rejects_blank_model() -> None:
    """Blank --model fails validation."""
    args = build_parser().parse_args(["--model", "   ", "--version", _VERSION])
    with pytest.raises(ValidationError, match="model must be a non-empty string"):
        build_options(args)


def test_discover_work_filters_model_version_and_years(tmp_path: Path) -> None:
    """discover_work returns only matching prediction partitions."""
    _touch_prediction(tmp_path, symbol="BTCUSDT", timeframe="1h", year=2024)
    _touch_prediction(tmp_path, symbol="BTCUSDT", timeframe="1h", year=2025)
    _touch_prediction(tmp_path, symbol="ETHUSDT", timeframe="4h", year=2025)
    _touch_prediction(
        tmp_path,
        symbol="BTCUSDT",
        timeframe="1h",
        year=2025,
        model="other-model",
    )
    repository = PredictionRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(
        repository,
        _options(
            storage_root=tmp_path,
            symbols=("BTCUSDT",),
            timeframes=("1h",),
            years=(2025,),
        ),
    )
    assert work == (
        DiscoveredWorkItem(
            framework=_FRAMEWORK,
            model_name=_MODEL,
            model_version=_VERSION,
            symbol="BTCUSDT",
            timeframe="1h",
            years=(2025,),
        ),
    )


def test_format_summary_contains_required_sections() -> None:
    """Global summary includes analysis header and recommended thresholds."""
    text = format_summary(
        CalibrationSummary(
            model=_MODEL,
            version=_VERSION,
            result=_calibration_result(),
            duration_seconds=1.25,
            failed_groups=0,
            failed_group_labels=(),
        )
    )
    assert "CQROS Regression Threshold Analysis" in text
    assert f"Model: {_MODEL}" in text
    assert f"Version: {_VERSION}" in text
    assert "Symbols analyzed: BTCUSDT, ETHUSDT" in text
    assert "Datasets: 2" in text
    assert "Rows analyzed: 200" in text
    assert "Prediction range:" in text
    assert "Global percentiles:" in text
    assert "Recommended thresholds" in text
    assert "Conservative:" in text
    assert "Balanced:" in text
    assert "Active:" in text
    assert "BUY >=" in text
    assert "SELL <=" in text
    assert "Expected BUY:" in text
    assert "Expected SELL:" in text
    assert "Expected HOLD:" in text
    assert "Duration: 1.250s" in text


def test_format_group_report_contains_prediction_summary() -> None:
    """Per-group report includes prediction summary and recommendations."""
    text = format_group_report(
        SymbolTimeframeCalibration(
            symbol="BTCUSDT",
            timeframe="1h",
            statistics=_statistics(count=42),
            recommendations=_recommendations(),
        )
    )
    assert "Symbol: BTCUSDT" in text
    assert "Timeframe: 1h" in text
    assert "Prediction summary" in text
    assert "Rows: 42" in text
    assert "Recommended thresholds" in text
    assert "Conservative:" in text


def test_run_calibration_with_no_work_returns_empty_summary(tmp_path: Path) -> None:
    """Empty discovery yields a summary without a calibration result."""
    repository = PredictionRepository(StorageLayout(tmp_path), ParquetStore())
    summary = _run(
        run_calibration(
            repository=repository,
            calibrator=SignalThresholdCalibrator(),
            options=_options(storage_root=tmp_path),
            work=(),
        )
    )
    assert summary.result is None
    assert summary.failed_groups == 0
    assert summary.model == _MODEL
    assert summary.version == _VERSION


def test_run_calibration_loads_partitions_and_calibrates(tmp_path: Path) -> None:
    """run_calibration loads partitions read-only and returns recommendations."""
    layout = StorageLayout(tmp_path)
    repository = PredictionRepository(layout, ParquetStore())
    values = [-1.0 + (2.0 * index / 99) for index in range(100)]
    frame = _prediction_frame(values)
    repository.save(
        frame,
        framework=_FRAMEWORK,
        model_name=_MODEL,
        model_version=_VERSION,
        exchange="binance",
        market="usdt_perpetual",
        symbol="BTCUSDT",
        timeframe="1h",
        year=2025,
    )
    work = (
        DiscoveredWorkItem(
            framework=_FRAMEWORK,
            model_name=_MODEL,
            model_version=_VERSION,
            symbol="BTCUSDT",
            timeframe="1h",
            years=(2025,),
        ),
    )
    summary = _run(
        run_calibration(
            repository=repository,
            calibrator=SignalThresholdCalibrator(),
            options=_options(storage_root=tmp_path, workers=1),
            work=work,
        )
    )
    assert summary.result is not None
    assert summary.failed_groups == 0
    assert summary.result.rows_analyzed == 100
    assert summary.result.symbols_analyzed == ("BTCUSDT",)
    assert len(summary.result.recommendations) == 3


def test_main_returns_success_for_calibrated_run(tmp_path: Path) -> None:
    """main returns exit code 0 when calibration succeeds."""
    layout = StorageLayout(tmp_path)
    repository = PredictionRepository(layout, ParquetStore())
    values = [-1.0 + (2.0 * index / 99) for index in range(100)]
    repository.save(
        _prediction_frame(values),
        framework=_FRAMEWORK,
        model_name=_MODEL,
        model_version=_VERSION,
        exchange="binance",
        market="usdt_perpetual",
        symbol="BTCUSDT",
        timeframe="1h",
        year=2025,
    )

    original_build_options = calibrate_module.build_options

    def _build_options_with_tmp(args: object) -> CalibrationOptions:
        options = original_build_options(args)
        return CalibrationOptions(
            storage_root=tmp_path,
            model=options.model,
            version=options.version,
            symbols=options.symbols,
            timeframes=options.timeframes,
            years=options.years,
            workers=options.workers,
            verbose=options.verbose,
            debug=options.debug,
        )

    with patch.object(calibrate_module, "build_options", _build_options_with_tmp):
        code = _run(
            main(
                [
                    "--model",
                    _MODEL,
                    "--version",
                    _VERSION,
                    "--symbols",
                    "BTCUSDT",
                    "--timeframes",
                    "1h",
                    "--years",
                    "2025",
                    "--workers",
                    "1",
                ]
            )
        )
    assert code == 0


def test_main_returns_failure_on_validation_error() -> None:
    """main returns exit code 1 for invalid CLI options."""
    code = _run(
        main(
            [
                "--model",
                _MODEL,
                "--version",
                _VERSION,
                "--workers",
                "0",
            ]
        )
    )
    assert code == 1


def test_configure_logging_levels() -> None:
    """Logging configuration honors verbose and debug flags."""
    with patch.object(calibrate_module.logging, "basicConfig") as basic_config:
        calibrate_module._configure_logging(verbose=False, debug=False)
        assert basic_config.call_args.kwargs["level"] == logging.WARNING
        calibrate_module._configure_logging(verbose=True, debug=False)
        assert basic_config.call_args.kwargs["level"] == logging.INFO
        calibrate_module._configure_logging(verbose=True, debug=True)
        assert basic_config.call_args.kwargs["level"] == logging.DEBUG


def test_run_calibration_records_load_failures(tmp_path: Path) -> None:
    """Failed partition loads are recorded without aborting other groups."""
    repository = MagicMock(spec=PredictionRepository)
    repository.load.side_effect = RuntimeError("disk read failed")
    work = (
        DiscoveredWorkItem(
            framework=_FRAMEWORK,
            model_name=_MODEL,
            model_version=_VERSION,
            symbol="BTCUSDT",
            timeframe="1h",
            years=(2025,),
        ),
    )
    summary = _run(
        run_calibration(
            repository=repository,
            calibrator=SignalThresholdCalibrator(),
            options=_options(storage_root=tmp_path, workers=1),
            work=work,
        )
    )
    assert summary.result is None
    assert summary.failed_groups == 1
    assert summary.failed_group_labels == ("BTCUSDT/1h",)
