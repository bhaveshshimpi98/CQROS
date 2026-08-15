"""Unit tests for CQROS OMS order-generation CLI."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

import cqros.cli.generate_orders as generate_orders_module
from cqros.cli.generate_orders import (
    DiscoveredWorkItem,
    OrderGenerationOptions,
    OrderGenerationSummary,
    OrderTaskResult,
    build_default_manager,
    build_manager_registry,
    build_options,
    build_order_pipeline,
    build_parser,
    discover_work,
    format_summary,
    main,
    run_generation,
)
from cqros.config.models import ResearchConfig
from cqros.core.constants import (
    DEFAULT_STORAGE_ROOT,
    STORAGE_DIR_ORDERS,
    STORAGE_DIR_RISKS,
)
from cqros.core.exceptions import ValidationError
from cqros.oms import (
    OrderManagerRegistry,
    OrderManagerType,
    OrderPipeline,
    SimpleOrderManager,
)
from cqros.storage import (
    OrderRepository,
    ParquetStore,
    RiskRepository,
    StorageLayout,
)

_MANAGER = OrderManagerType.SIMPLE.value
_POLICY = "fixed_risk"
_MODEL = "alpha-lgbm"
_VERSION = "1.0.0"


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def _options(
    *,
    storage_root: Path,
    manager: str = _MANAGER,
    policy: str | None = _POLICY,
    model: str | None = _MODEL,
    version: str | None = _VERSION,
    symbols: tuple[str, ...] | None = None,
    timeframes: tuple[str, ...] | None = None,
    years: tuple[int, ...] | None = None,
    overwrite: bool = False,
    workers: int = ResearchConfig().worker_count,
    verbose: bool = False,
    debug: bool = False,
) -> OrderGenerationOptions:
    """Build options for tests against a temporary storage root."""
    return OrderGenerationOptions(
        storage_root=storage_root,
        manager=manager,
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


def _touch_order(
    root: Path,
    *,
    manager: str,
    symbol: str,
    timeframe: str,
    year: int,
) -> Path:
    """Create an empty order year partition path on disk."""
    path = (
        root
        / STORAGE_DIR_ORDERS
        / manager
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
        "OrderGenerationOptions",
        "OrderGenerationSummary",
        "OrderTaskResult",
        "build_default_manager",
        "build_manager_registry",
        "build_options",
        "build_order_pipeline",
        "build_parser",
        "discover_work",
        "format_summary",
        "main",
        "run_generation",
    }
    assert expected.issubset(set(generate_orders_module.__all__))
    assert generate_orders_module.build_parser is build_parser
    assert generate_orders_module.main is main


def test_build_parser_requires_manager() -> None:
    """--manager is a required flag."""
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_build_parser_accepts_all_flags() -> None:
    """Parser accepts every documented order-generation flag."""
    args = build_parser().parse_args(
        [
            "--manager",
            _MANAGER,
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
    assert args.manager == _MANAGER
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
    """Explicit CLI flags map onto OrderGenerationOptions."""
    options = build_options(
        build_parser().parse_args(
            [
                "--manager",
                "twap",
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
    assert options.manager == "twap"
    assert options.policy == "kelly"
    assert options.model == "beta"
    assert options.version == "2.0.0"
    assert options.symbols == ("ETHUSDT",)
    assert options.timeframes == ("1d",)
    assert options.years == (2023,)
    assert options.overwrite is True
    assert options.workers == 8
    assert options.debug is True


def test_build_options_allows_optional_filters() -> None:
    """--policy, --model, and --version may be omitted."""
    options = build_options(build_parser().parse_args(["--manager", _MANAGER]))
    assert options.manager == _MANAGER
    assert options.policy is None
    assert options.model is None
    assert options.version is None


def test_build_options_rejects_non_positive_workers() -> None:
    """Non-positive --workers fails validation."""
    args = build_parser().parse_args(["--manager", _MANAGER, "--workers", "0"])
    with pytest.raises(ValidationError, match="workers must be greater than 0"):
        build_options(args)


def test_build_options_rejects_blank_manager() -> None:
    """Blank --manager fails validation."""
    args = build_parser().parse_args(["--manager", "   "])
    with pytest.raises(ValidationError, match="manager must be a non-empty string"):
        build_options(args)


def test_build_options_rejects_unsupported_timeframe() -> None:
    """Unsupported --timeframes values fail validation."""
    args = build_parser().parse_args(["--manager", _MANAGER, "--timeframes", "2x"])
    with pytest.raises(ValidationError, match="unsupported timeframe"):
        build_options(args)


def test_build_options_rejects_invalid_year() -> None:
    """Non-integer --years values fail validation."""
    args = build_parser().parse_args(["--manager", _MANAGER, "--years", "abc"])
    with pytest.raises(ValidationError, match="invalid year"):
        build_options(args)


def test_discover_work_finds_risk_partitions(tmp_path: Path) -> None:
    """Discovery walks risk partitions without hardcoding symbols."""
    _touch_risk(tmp_path, policy=_POLICY, symbol="BTCUSDT", timeframe="1h", year=2024)
    _touch_risk(tmp_path, policy=_POLICY, symbol="ETHUSDT", timeframe="1h", year=2023)
    _touch_risk(tmp_path, policy=_POLICY, symbol="ETHUSDT", timeframe="1h", year=2024)

    risks = RiskRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(risks, _options(storage_root=tmp_path))

    assert len(work) == 2
    assert work[0].policy == _POLICY
    assert work[0].symbol == "BTCUSDT"
    assert work[0].years == (2024,)
    assert work[1].symbol == "ETHUSDT"
    assert work[1].years == (2023, 2024)


def test_discover_work_filters_symbols_timeframes_years(tmp_path: Path) -> None:
    """Symbol, timeframe, and year filters apply to discovery."""
    _touch_risk(tmp_path, policy=_POLICY, symbol="BTCUSDT", timeframe="1h", year=2024)
    _touch_risk(tmp_path, policy=_POLICY, symbol="BTCUSDT", timeframe="4h", year=2024)
    _touch_risk(tmp_path, policy=_POLICY, symbol="ETHUSDT", timeframe="1h", year=2024)
    _touch_risk(tmp_path, policy=_POLICY, symbol="BTCUSDT", timeframe="1h", year=2025)

    risks = RiskRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(
        risks,
        _options(
            storage_root=tmp_path,
            symbols=("BTCUSDT",),
            timeframes=("1h",),
            years=(2024,),
        ),
    )

    assert work == (
        DiscoveredWorkItem(
            policy=_POLICY,
            symbol="BTCUSDT",
            timeframe="1h",
            years=(2024,),
        ),
    )


def test_discover_work_filters_policy(tmp_path: Path) -> None:
    """--policy limits discovery to one risk-policy tree."""
    _touch_risk(tmp_path, policy=_POLICY, symbol="BTCUSDT", timeframe="1h", year=2024)
    _touch_risk(tmp_path, policy="kelly", symbol="ETHUSDT", timeframe="1h", year=2024)

    risks = RiskRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(
        risks,
        _options(storage_root=tmp_path, policy=_POLICY),
    )

    assert len(work) == 1
    assert work[0].policy == _POLICY
    assert work[0].symbol == "BTCUSDT"


def test_discover_work_skips_missing_risk_partitions(tmp_path: Path) -> None:
    """Missing risk trees yield empty discovery instead of partial work."""
    risks = RiskRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(risks, _options(storage_root=tmp_path))
    assert work == ()


def test_run_generation_empty_work(tmp_path: Path) -> None:
    """Empty discovery produces a zeroed summary without invoking the pipeline."""
    pipeline = MagicMock(spec=OrderPipeline)
    risks = MagicMock(spec=RiskRepository)
    orders = MagicMock(spec=OrderRepository)
    summary = _run(
        run_generation(
            pipeline=pipeline,
            risk_repository=risks,
            order_repository=orders,
            options=_options(storage_root=tmp_path),
            work=(),
        )
    )
    assert summary.successful_tasks == 0
    assert summary.failed_tasks == 0
    assert summary.skipped_tasks == 0
    assert summary.manager == _MANAGER
    assert summary.version == _VERSION
    pipeline.run.assert_not_called()


def test_run_generation_skips_existing_without_overwrite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Existing order partitions are skipped unless --overwrite is set."""
    _touch_order(
        tmp_path,
        manager=_MANAGER,
        symbol="BTCUSDT",
        timeframe="1h",
        year=2024,
    )
    work = (
        DiscoveredWorkItem(
            policy=_POLICY,
            symbol="BTCUSDT",
            timeframe="1h",
            years=(2024,),
        ),
    )

    pipeline = MagicMock(spec=OrderPipeline)
    risks = MagicMock(spec=RiskRepository)
    orders = MagicMock(spec=OrderRepository)
    orders.exists.return_value = True

    summary = _run(
        run_generation(
            pipeline=pipeline,
            risk_repository=risks,
            order_repository=orders,
            options=_options(storage_root=tmp_path, workers=1),
            work=work,
        )
    )

    captured = capsys.readouterr().out
    assert summary.skipped_tasks == 1
    assert summary.successful_tasks == 0
    pipeline.run.assert_not_called()
    orders.save.assert_not_called()
    assert "SKIP BTCUSDT 1h 2024" in captured


def test_run_generation_overwrite_existing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--overwrite regenerates partitions that already exist."""
    work = (
        DiscoveredWorkItem(
            policy=_POLICY,
            symbol="BTCUSDT",
            timeframe="1h",
            years=(2024,),
        ),
    )
    pipeline = MagicMock(spec=OrderPipeline)
    pipeline.run.return_value = pl.DataFrame({"open_time": [1, 2]})
    risks = MagicMock(spec=RiskRepository)
    risks.load.return_value = pl.DataFrame(
        {
            "model_name": [_MODEL, _MODEL],
            "model_version": [_VERSION, _VERSION],
            "open_time": [1, 2],
        }
    )
    orders = MagicMock(spec=OrderRepository)
    orders.exists.return_value = True

    summary = _run(
        run_generation(
            pipeline=pipeline,
            risk_repository=risks,
            order_repository=orders,
            options=_options(storage_root=tmp_path, workers=1, overwrite=True),
            work=work,
        )
    )

    captured = capsys.readouterr().out
    assert summary.successful_tasks == 1
    assert summary.skipped_tasks == 0
    assert summary.rows_generated == 2
    pipeline.run.assert_called_once()
    orders.save.assert_called_once()
    assert "OK BTCUSDT 1h 2024 rows=2" in captured


def test_run_generation_success_and_progress(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Successful generation increments counters and prints progress."""
    work = (
        DiscoveredWorkItem(
            policy=_POLICY,
            symbol="BTCUSDT",
            timeframe="1h",
            years=(2024,),
        ),
    )
    pipeline = MagicMock(spec=OrderPipeline)
    pipeline.run.return_value = pl.DataFrame({"open_time": [1, 2, 3]})
    risks = MagicMock(spec=RiskRepository)
    risks.load.return_value = pl.DataFrame(
        {
            "model_name": [_MODEL, _MODEL, "other"],
            "model_version": [_VERSION, _VERSION, "9.9.9"],
            "open_time": [1, 2, 3],
        }
    )
    orders = MagicMock(spec=OrderRepository)
    orders.exists.return_value = False

    summary = _run(
        run_generation(
            pipeline=pipeline,
            risk_repository=risks,
            order_repository=orders,
            options=_options(storage_root=tmp_path, workers=1, overwrite=True),
            work=work,
        )
    )

    captured = capsys.readouterr().out
    assert summary.successful_tasks == 1
    assert summary.rows_generated == 3
    assert "OK BTCUSDT 1h 2024 rows=3" in captured
    pipeline.run.assert_called_once()
    assert pipeline.run.call_args.args[0] == _MANAGER
    filtered = pipeline.run.call_args.args[1]
    assert filtered.height == 2
    assert set(filtered.get_column("model_name").to_list()) == {_MODEL}
    risks.load.assert_called_once()
    assert risks.load.call_args.kwargs["policy"] == _POLICY


def test_run_generation_without_model_filter(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """When model filters are omitted, the full risk frame is evaluated."""
    work = (
        DiscoveredWorkItem(
            policy=_POLICY,
            symbol="BTCUSDT",
            timeframe="1h",
            years=(2024,),
        ),
    )
    pipeline = MagicMock(spec=OrderPipeline)
    pipeline.run.return_value = pl.DataFrame({"open_time": [1, 2, 3]})
    risks = MagicMock(spec=RiskRepository)
    risks.load.return_value = pl.DataFrame(
        {
            "model_name": [_MODEL, "other", "other"],
            "model_version": [_VERSION, "9.9.9", "9.9.9"],
            "open_time": [1, 2, 3],
        }
    )
    orders = MagicMock(spec=OrderRepository)
    orders.exists.return_value = False

    summary = _run(
        run_generation(
            pipeline=pipeline,
            risk_repository=risks,
            order_repository=orders,
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
    work = (
        DiscoveredWorkItem(
            policy=_POLICY,
            symbol="BTCUSDT",
            timeframe="1h",
            years=(2024, 2025),
        ),
    )
    pipeline = MagicMock(spec=OrderPipeline)
    pipeline.run.side_effect = [RuntimeError("boom"), pl.DataFrame({"open_time": [1]})]
    risks = MagicMock(spec=RiskRepository)
    risks.load.return_value = pl.DataFrame(
        {
            "model_name": [_MODEL],
            "model_version": [_VERSION],
            "open_time": [1],
        }
    )
    orders = MagicMock(spec=OrderRepository)
    orders.exists.return_value = False

    summary = _run(
        run_generation(
            pipeline=pipeline,
            risk_repository=risks,
            order_repository=orders,
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


def test_format_summary_includes_manager_version_and_failed_tasks() -> None:
    """Summary rendering includes manager identity and failed-task labels."""
    text = format_summary(
        OrderGenerationSummary(
            manager=_MANAGER,
            version=_VERSION,
            symbols_discovered=1,
            symbols_processed=1,
            timeframes_processed=1,
            successful_tasks=0,
            failed_tasks=1,
            skipped_tasks=0,
            rows_generated=0,
            duration_seconds=1.5,
            output_directory=Path("data/orders"),
            failed_task_labels=("BTCUSDT 1h 2024",),
        )
    )
    assert "CQROS Order Generation Summary" in text
    assert f"Manager: {_MANAGER}" in text
    assert f"Version: {_VERSION}" in text
    assert "Failed tasks: 1" in text
    assert "- BTCUSDT 1h 2024" in text
    assert "Output directory: data/orders" in text


def test_main_exit_code_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """main returns 0 when no tasks fail."""
    with (
        patch("cqros.cli.generate_orders.DEFAULT_STORAGE_ROOT", str(tmp_path)),
        patch("cqros.cli.generate_orders.build_order_pipeline") as build_pipeline,
        patch("cqros.cli.generate_orders.discover_work", return_value=()),
    ):
        build_pipeline.return_value = MagicMock(spec=OrderPipeline)
        code = _run(main(["--manager", _MANAGER, "--workers", "1"]))

    captured = capsys.readouterr()
    assert code == 0
    assert "CQROS Order Generation Summary" in captured.out


def test_main_validation_error_exit_code() -> None:
    """Fatal CLI validation errors return exit code 1."""
    code = _run(main(["--manager", _MANAGER, "--workers", "0"]))
    assert code == 1


def test_configure_logging_verbose(tmp_path: Path) -> None:
    """--verbose enables INFO logging for the cqros logger."""
    with (
        patch("cqros.cli.generate_orders.DEFAULT_STORAGE_ROOT", str(tmp_path)),
        patch("cqros.cli.generate_orders.build_order_pipeline") as build_pipeline,
        patch("cqros.cli.generate_orders.discover_work", return_value=()),
    ):
        build_pipeline.return_value = MagicMock(spec=OrderPipeline)
        _run(main(["--manager", _MANAGER, "--verbose"]))
    assert logging.getLogger("cqros").level == logging.INFO


def test_configure_logging_debug(tmp_path: Path) -> None:
    """--debug enables DEBUG logging for the cqros logger."""
    with (
        patch("cqros.cli.generate_orders.DEFAULT_STORAGE_ROOT", str(tmp_path)),
        patch("cqros.cli.generate_orders.build_order_pipeline") as build_pipeline,
        patch("cqros.cli.generate_orders.discover_work", return_value=()),
    ):
        build_pipeline.return_value = MagicMock(spec=OrderPipeline)
        _run(main(["--manager", _MANAGER, "--debug"]))
    assert logging.getLogger("cqros").level == logging.DEBUG


def test_order_task_result_fields() -> None:
    """OrderTaskResult stores status metadata immutably."""
    result = OrderTaskResult(
        symbol="BTCUSDT",
        timeframe="1h",
        year=2024,
        status="failed",
        error_type="RuntimeError",
        error_message="boom",
    )
    assert result.status == "failed"
    assert result.error_type == "RuntimeError"


def test_build_default_manager_returns_simple() -> None:
    """Default composition-root manager is SimpleOrderManager."""
    manager = build_default_manager()
    assert isinstance(manager, SimpleOrderManager)


def test_build_manager_registry_registers_simple() -> None:
    """Default registry exposes SimpleOrderManager under simple."""
    registry = build_manager_registry()
    assert registry.exists(_MANAGER)
    assert isinstance(registry.get(_MANAGER), SimpleOrderManager)


def test_build_order_pipeline_wires_registry(tmp_path: Path) -> None:
    """Pipeline composition injects the supplied manager registry."""
    registry = OrderManagerRegistry()
    registry.register(_MANAGER, SimpleOrderManager())

    with patch("cqros.cli.generate_orders.OrderPipeline") as pipeline_cls:
        pipeline_cls.return_value = MagicMock(spec=OrderPipeline)
        build_order_pipeline(
            _options(storage_root=tmp_path),
            manager_registry=registry,
        )

    pipeline_cls.assert_called_once_with(registry)
