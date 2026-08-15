"""Unit tests for CQROS risk-decision generation CLI."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

import cqros.cli.generate_risk as generate_risk_module
from cqros.cli.generate_risk import (
    DiscoveredWorkItem,
    RiskGenerationOptions,
    RiskGenerationSummary,
    RiskTaskResult,
    build_default_policy,
    build_options,
    build_parser,
    build_policy_registry,
    build_risk_pipeline,
    discover_work,
    format_summary,
    main,
    run_generation,
)
from cqros.config.models import ResearchConfig
from cqros.core.constants import (
    DEFAULT_STORAGE_ROOT,
    STORAGE_DIR_PORTFOLIOS,
    STORAGE_DIR_RISKS,
)
from cqros.core.exceptions import ValidationError
from cqros.risk import (
    FixedRiskPolicy,
    RiskPipeline,
    RiskPolicy,
    RiskPolicyRegistry,
)
from cqros.risk.schema import CANONICAL_COLUMN_ORDER, MERGED_RISK_SCHEMA
from cqros.storage import (
    ParquetStore,
    PortfolioRepository,
    RiskRepository,
    StorageLayout,
)

_POLICY = RiskPolicy.FIXED_RISK.value
_MODEL = "alpha-lgbm"
_VERSION = "1.0.0"


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def _options(
    *,
    storage_root: Path,
    policy: str = _POLICY,
    model: str | None = _MODEL,
    version: str | None = _VERSION,
    symbols: tuple[str, ...] | None = None,
    timeframes: tuple[str, ...] | None = None,
    years: tuple[int, ...] | None = None,
    overwrite: bool = False,
    workers: int = ResearchConfig().worker_count,
    verbose: bool = False,
    debug: bool = False,
) -> RiskGenerationOptions:
    """Build options for tests against a temporary storage root."""
    return RiskGenerationOptions(
        storage_root=storage_root,
        policy=policy,
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


def _touch_risk(
    root: Path,
    *,
    policy: str,
    symbol: str,
    timeframe: str,
    year: int,
) -> Path:
    """Create an empty risk year partition path on disk."""
    path = (
        root
        / STORAGE_DIR_RISKS
        / policy
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
        "RiskGenerationOptions",
        "RiskGenerationSummary",
        "RiskTaskResult",
        "build_default_policy",
        "build_options",
        "build_parser",
        "build_policy_registry",
        "build_risk_pipeline",
        "discover_work",
        "format_summary",
        "main",
        "run_generation",
    }
    assert expected.issubset(set(generate_risk_module.__all__))
    assert generate_risk_module.build_parser is build_parser
    assert generate_risk_module.main is main


def test_build_parser_requires_policy() -> None:
    """--policy is a required flag."""
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_build_parser_accepts_all_flags() -> None:
    """Parser accepts every documented risk-generation flag."""
    args = build_parser().parse_args(
        [
            "--policy",
            _POLICY,
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
    assert args.policy == _POLICY
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
    """Explicit CLI flags map onto RiskGenerationOptions."""
    options = build_options(
        build_parser().parse_args(
            [
                "--policy",
                "kelly",
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
    assert options.policy == "kelly"
    assert options.model == "beta"
    assert options.version == "2.0.0"
    assert options.symbols == ("ETHUSDT",)
    assert options.timeframes == ("1d",)
    assert options.years == (2023,)
    assert options.overwrite is True
    assert options.workers == 8
    assert options.debug is True


def test_build_options_allows_optional_model_and_version() -> None:
    """--model and --version may be omitted."""
    options = build_options(build_parser().parse_args(["--policy", _POLICY]))
    assert options.policy == _POLICY
    assert options.model is None
    assert options.version is None


def test_build_options_rejects_non_positive_workers() -> None:
    """Non-positive --workers fails validation."""
    args = build_parser().parse_args(["--policy", _POLICY, "--workers", "0"])
    with pytest.raises(ValidationError, match="workers must be greater than 0"):
        build_options(args)


def test_build_options_rejects_unsupported_timeframe() -> None:
    """Unsupported --timeframes values fail validation."""
    args = build_parser().parse_args(["--policy", _POLICY, "--timeframes", "2x"])
    with pytest.raises(ValidationError, match="unsupported timeframe"):
        build_options(args)


def test_build_options_rejects_invalid_year() -> None:
    """Non-integer --years values fail validation."""
    args = build_parser().parse_args(["--policy", _POLICY, "--years", "abc"])
    with pytest.raises(ValidationError, match="invalid year"):
        build_options(args)


def test_discover_work_finds_portfolio_partitions(tmp_path: Path) -> None:
    """Discovery walks portfolio partitions without hardcoding symbols."""
    _touch_portfolio(tmp_path, symbol="BTCUSDT", timeframe="1h", year=2024)
    _touch_portfolio(tmp_path, symbol="ETHUSDT", timeframe="1h", year=2023)
    _touch_portfolio(tmp_path, symbol="ETHUSDT", timeframe="1h", year=2024)

    portfolios = PortfolioRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(portfolios, _options(storage_root=tmp_path))

    assert len(work) == 2
    assert work[0].symbol == "BTCUSDT"
    assert work[0].years == (2024,)
    assert work[1].symbol == "ETHUSDT"
    assert work[1].years == (2023, 2024)


def test_discover_work_filters_symbols_timeframes_years(tmp_path: Path) -> None:
    """Symbol, timeframe, and year filters apply to discovery."""
    _touch_portfolio(tmp_path, symbol="BTCUSDT", timeframe="1h", year=2024)
    _touch_portfolio(tmp_path, symbol="BTCUSDT", timeframe="4h", year=2024)
    _touch_portfolio(tmp_path, symbol="ETHUSDT", timeframe="1h", year=2024)
    _touch_portfolio(tmp_path, symbol="BTCUSDT", timeframe="1h", year=2025)

    portfolios = PortfolioRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(
        portfolios,
        _options(
            storage_root=tmp_path,
            symbols=("BTCUSDT",),
            timeframes=("1h",),
            years=(2024,),
        ),
    )

    assert work == (DiscoveredWorkItem(symbol="BTCUSDT", timeframe="1h", years=(2024,)),)


def test_discover_work_skips_missing_portfolio_partitions(tmp_path: Path) -> None:
    """Missing portfolio trees yield empty discovery instead of partial work."""
    portfolios = PortfolioRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(portfolios, _options(storage_root=tmp_path))
    assert work == ()


def test_run_generation_empty_work(tmp_path: Path) -> None:
    """Empty discovery produces a zeroed summary without invoking the pipeline."""
    pipeline = MagicMock(spec=RiskPipeline)
    portfolios = MagicMock(spec=PortfolioRepository)
    risks = MagicMock(spec=RiskRepository)
    summary = _run(
        run_generation(
            pipeline=pipeline,
            portfolio_repository=portfolios,
            risk_repository=risks,
            options=_options(storage_root=tmp_path),
            work=(),
        )
    )
    assert summary.successful_tasks == 0
    assert summary.failed_tasks == 0
    assert summary.skipped_tasks == 0
    assert summary.policy == _POLICY
    assert summary.version == _VERSION
    pipeline.run.assert_not_called()


def test_run_generation_skips_existing_without_overwrite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Existing risk partitions are skipped unless --overwrite is set."""
    _touch_risk(
        tmp_path,
        policy=_POLICY,
        symbol="BTCUSDT",
        timeframe="1h",
        year=2024,
    )
    work = (DiscoveredWorkItem(symbol="BTCUSDT", timeframe="1h", years=(2024,)),)

    pipeline = MagicMock(spec=RiskPipeline)
    portfolios = MagicMock(spec=PortfolioRepository)
    risks = MagicMock(spec=RiskRepository)
    risks.exists.return_value = True

    summary = _run(
        run_generation(
            pipeline=pipeline,
            portfolio_repository=portfolios,
            risk_repository=risks,
            options=_options(storage_root=tmp_path, workers=1),
            work=work,
        )
    )

    captured = capsys.readouterr().out
    assert summary.skipped_tasks == 1
    assert summary.successful_tasks == 0
    pipeline.run.assert_not_called()
    risks.save.assert_not_called()
    assert "SKIP BTCUSDT 1h 2024" in captured


def test_run_generation_overwrite_existing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--overwrite regenerates partitions that already exist."""
    work = (DiscoveredWorkItem(symbol="BTCUSDT", timeframe="1h", years=(2024,)),)
    pipeline = MagicMock(spec=RiskPipeline)
    pipeline.run.return_value = pl.DataFrame({"open_time": [1, 2]})
    portfolios = MagicMock(spec=PortfolioRepository)
    portfolios.load.return_value = pl.DataFrame(
        {
            "model_name": [_MODEL, _MODEL],
            "model_version": [_VERSION, _VERSION],
            "open_time": [1, 2],
        }
    )
    risks = MagicMock(spec=RiskRepository)
    risks.exists.return_value = True

    summary = _run(
        run_generation(
            pipeline=pipeline,
            portfolio_repository=portfolios,
            risk_repository=risks,
            options=_options(storage_root=tmp_path, workers=1, overwrite=True),
            work=work,
        )
    )

    captured = capsys.readouterr().out
    assert summary.successful_tasks == 1
    assert summary.skipped_tasks == 0
    assert summary.rows_generated == 2
    pipeline.run.assert_called_once()
    risks.save.assert_called_once()
    assert "OK BTCUSDT 1h 2024 rows=2" in captured


def test_run_generation_success_and_progress(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Successful generation increments counters and prints progress."""
    work = (DiscoveredWorkItem(symbol="BTCUSDT", timeframe="1h", years=(2024,)),)
    pipeline = MagicMock(spec=RiskPipeline)
    pipeline.run.return_value = pl.DataFrame({"open_time": [1, 2, 3]})
    portfolios = MagicMock(spec=PortfolioRepository)
    portfolios.load.return_value = pl.DataFrame(
        {
            "model_name": [_MODEL, _MODEL, "other"],
            "model_version": [_VERSION, _VERSION, "9.9.9"],
            "open_time": [1, 2, 3],
        }
    )
    risks = MagicMock(spec=RiskRepository)
    risks.exists.return_value = False

    summary = _run(
        run_generation(
            pipeline=pipeline,
            portfolio_repository=portfolios,
            risk_repository=risks,
            options=_options(storage_root=tmp_path, workers=1, overwrite=True),
            work=work,
        )
    )

    captured = capsys.readouterr().out
    assert summary.successful_tasks == 1
    assert summary.rows_generated == 3
    assert "OK BTCUSDT 1h 2024 rows=3" in captured
    pipeline.run.assert_called_once()
    assert pipeline.run.call_args.args[0] == _POLICY
    filtered = pipeline.run.call_args.args[1]
    assert filtered.height == 2
    assert set(filtered.get_column("model_name").to_list()) == {_MODEL}


def test_run_generation_without_model_filter(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """When model filters are omitted, the full portfolio frame is evaluated."""
    work = (DiscoveredWorkItem(symbol="BTCUSDT", timeframe="1h", years=(2024,)),)
    pipeline = MagicMock(spec=RiskPipeline)
    pipeline.run.return_value = pl.DataFrame({"open_time": [1, 2, 3]})
    portfolios = MagicMock(spec=PortfolioRepository)
    portfolios.load.return_value = pl.DataFrame(
        {
            "model_name": [_MODEL, "other", "other"],
            "model_version": [_VERSION, "9.9.9", "9.9.9"],
            "open_time": [1, 2, 3],
        }
    )
    risks = MagicMock(spec=RiskRepository)
    risks.exists.return_value = False

    summary = _run(
        run_generation(
            pipeline=pipeline,
            portfolio_repository=portfolios,
            risk_repository=risks,
            options=_options(
                storage_root=tmp_path,
                workers=1,
                overwrite=True,
                model=None,
                version=None,
            ),
            work=work,
        )
    )

    assert summary.successful_tasks == 1
    filtered = pipeline.run.call_args.args[1]
    assert filtered.height == 3
    assert "OK BTCUSDT 1h 2024 rows=3" in capsys.readouterr().out


def test_run_generation_pipeline_failure_isolation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failed year does not prevent later years from running."""
    work = (DiscoveredWorkItem(symbol="BTCUSDT", timeframe="1h", years=(2024, 2025)),)
    pipeline = MagicMock(spec=RiskPipeline)
    pipeline.run.side_effect = [RuntimeError("boom"), pl.DataFrame({"open_time": [1]})]
    portfolios = MagicMock(spec=PortfolioRepository)
    portfolios.load.return_value = pl.DataFrame(
        {
            "model_name": [_MODEL],
            "model_version": [_VERSION],
            "open_time": [1],
        }
    )
    risks = MagicMock(spec=RiskRepository)
    risks.exists.return_value = False

    summary = _run(
        run_generation(
            pipeline=pipeline,
            portfolio_repository=portfolios,
            risk_repository=risks,
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


def test_format_summary_includes_policy_version_and_failed_tasks() -> None:
    """Summary rendering includes policy identity and failed-task labels."""
    text = format_summary(
        RiskGenerationSummary(
            policy=_POLICY,
            version=_VERSION,
            symbols_discovered=1,
            symbols_processed=1,
            timeframes_processed=1,
            successful_tasks=0,
            failed_tasks=1,
            skipped_tasks=0,
            rows_generated=0,
            duration_seconds=1.5,
            output_directory=Path("data/risks"),
            failed_task_labels=("BTCUSDT 1h 2024",),
        )
    )
    assert "CQROS Risk Generation Summary" in text
    assert f"Policy: {_POLICY}" in text
    assert f"Version: {_VERSION}" in text
    assert "Failed tasks: 1" in text
    assert "- BTCUSDT 1h 2024" in text
    assert "Output directory: data/risks" in text


def test_main_exit_code_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """main returns 0 when no tasks fail."""
    with (
        patch("cqros.cli.generate_risk.DEFAULT_STORAGE_ROOT", str(tmp_path)),
        patch("cqros.cli.generate_risk.build_risk_pipeline") as build_pipeline,
        patch("cqros.cli.generate_risk.discover_work", return_value=()),
    ):
        build_pipeline.return_value = MagicMock(spec=RiskPipeline)
        code = _run(main(["--policy", _POLICY, "--workers", "1"]))

    captured = capsys.readouterr()
    assert code == 0
    assert "CQROS Risk Generation Summary" in captured.out


def test_main_validation_error_exit_code() -> None:
    """Fatal CLI validation errors return exit code 1."""
    code = _run(main(["--policy", _POLICY, "--workers", "0"]))
    assert code == 1


def test_configure_logging_verbose(tmp_path: Path) -> None:
    """--verbose enables INFO logging for the cqros logger."""
    with (
        patch("cqros.cli.generate_risk.DEFAULT_STORAGE_ROOT", str(tmp_path)),
        patch("cqros.cli.generate_risk.build_risk_pipeline") as build_pipeline,
        patch("cqros.cli.generate_risk.discover_work", return_value=()),
    ):
        build_pipeline.return_value = MagicMock(spec=RiskPipeline)
        _run(main(["--policy", _POLICY, "--verbose"]))
    assert logging.getLogger("cqros").level == logging.INFO


def test_configure_logging_debug(tmp_path: Path) -> None:
    """--debug enables DEBUG logging for the cqros logger."""
    with (
        patch("cqros.cli.generate_risk.DEFAULT_STORAGE_ROOT", str(tmp_path)),
        patch("cqros.cli.generate_risk.build_risk_pipeline") as build_pipeline,
        patch("cqros.cli.generate_risk.discover_work", return_value=()),
    ):
        build_pipeline.return_value = MagicMock(spec=RiskPipeline)
        _run(main(["--policy", _POLICY, "--debug"]))
    assert logging.getLogger("cqros").level == logging.DEBUG


def test_risk_task_result_fields() -> None:
    """RiskTaskResult stores status metadata immutably."""
    result = RiskTaskResult(
        symbol="BTCUSDT",
        timeframe="1h",
        year=2024,
        status="failed",
        error_type="RuntimeError",
        error_message="boom",
    )
    assert result.status == "failed"
    assert result.error_type == "RuntimeError"


def test_build_default_policy_returns_fixed_risk() -> None:
    """Default composition-root policy is FixedRiskPolicy."""
    policy = build_default_policy()
    assert isinstance(policy, FixedRiskPolicy)


def test_build_policy_registry_registers_fixed_risk() -> None:
    """Default registry exposes FixedRiskPolicy under fixed_risk."""
    registry = build_policy_registry()
    assert registry.exists(_POLICY)
    assert isinstance(registry.get(_POLICY), FixedRiskPolicy)


def test_build_risk_pipeline_wires_registry(tmp_path: Path) -> None:
    """Pipeline composition injects the supplied policy registry."""
    registry = RiskPolicyRegistry()
    registry.register(_POLICY, FixedRiskPolicy())

    with patch("cqros.cli.generate_risk.RiskPipeline") as pipeline_cls:
        pipeline_cls.return_value = MagicMock(spec=RiskPipeline)
        build_risk_pipeline(
            _options(storage_root=tmp_path),
            policy_registry=registry,
        )

    pipeline_cls.assert_called_once_with(registry)


def test_default_pipeline_emits_optimizer_and_policy_lineage() -> None:
    """CLI-composed FixedRiskPolicy emits optimizer and policy on risk frames."""
    portfolios = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timeframe": ["1h"],
            "open_time": [datetime(2024, 1, 1, tzinfo=UTC)],
            "model_name": [_MODEL],
            "model_version": [_VERSION],
            "optimizer": ["equal_weight"],
            "signal": ["BUY"],
            "target_weight": [1.0],
        },
        schema={
            "symbol": pl.Utf8,
            "timeframe": pl.Utf8,
            "open_time": pl.Datetime("us", "UTC"),
            "model_name": pl.Utf8,
            "model_version": pl.Utf8,
            "optimizer": pl.Utf8,
            "signal": pl.Utf8,
            "target_weight": pl.Float64,
        },
    )
    pipeline = RiskPipeline(build_policy_registry())
    result = pipeline.run(_POLICY, portfolios)

    assert result.columns == list(CANONICAL_COLUMN_ORDER)
    assert result.schema == MERGED_RISK_SCHEMA
    assert result.get_column("optimizer").to_list() == ["equal_weight"]
    assert result.get_column("policy").to_list() == [_POLICY]
