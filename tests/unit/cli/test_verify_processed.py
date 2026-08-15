"""Unit tests for CQROS processed-data verification CLI."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from cqros.cli.verify_processed import (
    CLI_DATASETS,
    VerifyProcessedOptions,
    VerifyProcessedSummary,
    build_options,
    build_parser,
    build_verification_runner,
    discover_work,
    format_partition_failure,
    format_summary,
    main,
    run_verification,
)
from cqros.config.models import ResearchConfig
from cqros.core.constants import DEFAULT_STORAGE_ROOT, STORAGE_DIR_PROCESSED
from cqros.core.exceptions import ValidationError
from cqros.processing.exceptions import ProcessingValidationError
from cqros.processing.verification import (
    FundingVerifier,
    LongShortVerifier,
    OHLCVVerifier,
    OpenInterestVerifier,
    TakerVolumeVerifier,
    VerificationReport,
    VerificationRunner,
    VerificationSummary,
    VerificationTaskResult,
)
from cqros.processing.verification.exceptions import ERROR_REQUIRED_COLUMNS
from cqros.storage import (
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
    datasets: tuple[str, ...] | None = None,
    workers: int = ResearchConfig().worker_count,
    verbose: bool = False,
    debug: bool = False,
) -> VerifyProcessedOptions:
    """Build options for tests against a temporary storage root."""
    return VerifyProcessedOptions(
        storage_root=storage_root,
        symbols=symbols,
        timeframes=timeframes,
        datasets=datasets,
        workers=workers,
        verbose=verbose,
        debug=debug,
    )


def _touch_partition(
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


def _passed_report(*, rows: int = 10) -> VerificationReport:
    """Return a passing verification report."""
    return VerificationReport(
        rows_checked=rows,
        duplicate_timestamp_rows=0,
        null_rows=0,
        nan_rows=0,
        invalid_timestamp_rows=0,
        invalid_numeric_rows=0,
        warnings=(),
        passed=True,
    )


def _failed_report(*, null_rows: int = 1) -> VerificationReport:
    """Return a failing verification report with a positive counter."""
    return VerificationReport(
        rows_checked=10,
        duplicate_timestamp_rows=0,
        null_rows=null_rows,
        nan_rows=0,
        invalid_timestamp_rows=0,
        invalid_numeric_rows=0,
        warnings=(),
        passed=False,
    )


def test_build_parser_defaults() -> None:
    """Omitted optional flags keep discovery defaults."""
    args = build_parser().parse_args([])
    assert args.symbols is None
    assert args.timeframes is None
    assert args.datasets is None
    assert args.workers == ResearchConfig().worker_count
    assert args.verbose is False
    assert args.debug is False


def test_build_parser_accepts_all_flags() -> None:
    """Parser accepts every documented verification flag."""
    args = build_parser().parse_args(
        [
            "--dataset",
            "ohlcv",
            "--dataset",
            "funding",
            "--symbols",
            "BTCUSDT",
            "ETHUSDT",
            "--timeframes",
            "1h",
            "4h",
            "--workers",
            "2",
            "--verbose",
            "--debug",
        ]
    )
    assert args.datasets == ["ohlcv", "funding"]
    assert args.symbols == ["BTCUSDT", "ETHUSDT"]
    assert args.timeframes == ["1h", "4h"]
    assert args.workers == 2
    assert args.verbose is True
    assert args.debug is True


def test_build_parser_rejects_unknown_dataset() -> None:
    """Unsupported --dataset values are rejected by argparse."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--dataset", "liquidation"])


def test_build_options_defaults() -> None:
    """Omitted filters map to discovery-all options."""
    options = build_options(build_parser().parse_args([]))
    assert options.storage_root == Path(DEFAULT_STORAGE_ROOT)
    assert options.symbols is None
    assert options.timeframes is None
    assert options.datasets is None
    assert options.workers == ResearchConfig().worker_count
    assert options.verbose is False
    assert options.debug is False


def test_build_options_maps_filters() -> None:
    """Explicit CLI flags map onto VerifyProcessedOptions."""
    options = build_options(
        build_parser().parse_args(
            [
                "--dataset",
                "taker_volume",
                "--symbols",
                "ETHUSDT",
                "--timeframes",
                "1d",
                "--workers",
                "8",
                "--verbose",
            ]
        )
    )
    assert options.symbols == ("ETHUSDT",)
    assert options.timeframes == ("1d",)
    assert options.datasets == ("taker_volume",)
    assert options.workers == 8
    assert options.verbose is True


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


def test_discover_work_finds_symbols_and_years(tmp_path: Path) -> None:
    """Discovery uses the repository and does not hardcode symbols."""
    _touch_partition(tmp_path, dataset="ohlcv", symbol="BTCUSDT", timeframe="1h", year=2024)
    _touch_partition(tmp_path, dataset="ohlcv", symbol="ETHUSDT", timeframe="1h", year=2023)
    _touch_partition(tmp_path, dataset="ohlcv", symbol="ETHUSDT", timeframe="1h", year=2024)

    repository = ProcessedMarketDataRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(repository, _options(storage_root=tmp_path, datasets=("ohlcv",)))

    assert len(work) == 2
    assert work[0].symbol == "BTCUSDT"
    assert work[0].years == (2024,)
    assert work[1].symbol == "ETHUSDT"
    assert work[1].years == (2023, 2024)


def test_discover_work_filters_symbol(tmp_path: Path) -> None:
    """--symbols limits discovery to the allowlist."""
    _touch_partition(tmp_path, dataset="ohlcv", symbol="BTCUSDT", timeframe="1h", year=2024)
    _touch_partition(tmp_path, dataset="ohlcv", symbol="ETHUSDT", timeframe="1h", year=2024)

    repository = ProcessedMarketDataRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(
        repository,
        _options(storage_root=tmp_path, symbols=("BTCUSDT",), datasets=("ohlcv",)),
    )

    assert len(work) == 1
    assert work[0].symbol == "BTCUSDT"


def test_discover_work_filters_timeframe(tmp_path: Path) -> None:
    """--timeframes limits discovery to requested intervals."""
    _touch_partition(tmp_path, dataset="ohlcv", symbol="BTCUSDT", timeframe="1h", year=2024)
    _touch_partition(tmp_path, dataset="ohlcv", symbol="BTCUSDT", timeframe="4h", year=2024)
    _touch_partition(tmp_path, dataset="ohlcv", symbol="BTCUSDT", timeframe="1d", year=2024)

    repository = ProcessedMarketDataRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(
        repository,
        _options(
            storage_root=tmp_path,
            datasets=("ohlcv",),
            timeframes=("1h", "1d"),
        ),
    )

    assert {item.timeframe for item in work} == {"1h", "1d"}
    assert len(work) == 2


def test_discover_work_filters_dataset(tmp_path: Path) -> None:
    """Dataset filters apply and missing dataset trees are skipped."""
    _touch_partition(tmp_path, dataset="ohlcv", symbol="BTCUSDT", timeframe="1h", year=2024)
    _touch_partition(tmp_path, dataset="funding", symbol="BTCUSDT", timeframe="8h", year=2024)

    repository = ProcessedMarketDataRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(
        repository,
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

    repository = ProcessedMarketDataRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(
        repository,
        _options(storage_root=tmp_path, datasets=("long_short",)),
    )

    assert {item.storage_dataset for item in work} == {
        "global_long_short_account_ratio",
        "top_long_short_position_ratio",
    }
    assert all(item.cli_dataset == "long_short" for item in work)


def test_build_verification_runner_wires_dependencies(tmp_path: Path) -> None:
    """Dependency construction wires repository and verifiers once."""
    options = _options(storage_root=tmp_path)
    with (
        patch("cqros.cli.verify_processed.StorageLayout", wraps=StorageLayout) as layout_cls,
        patch("cqros.cli.verify_processed.ParquetStore", wraps=ParquetStore) as store_cls,
        patch(
            "cqros.cli.verify_processed.ProcessedMarketDataRepository",
            wraps=ProcessedMarketDataRepository,
        ) as processed_cls,
        patch("cqros.cli.verify_processed.OHLCVVerifier", wraps=OHLCVVerifier) as ohlcv_cls,
        patch(
            "cqros.cli.verify_processed.FundingVerifier",
            wraps=FundingVerifier,
        ) as funding_cls,
        patch(
            "cqros.cli.verify_processed.OpenInterestVerifier",
            wraps=OpenInterestVerifier,
        ) as oi_cls,
        patch(
            "cqros.cli.verify_processed.TakerVolumeVerifier",
            wraps=TakerVolumeVerifier,
        ) as taker_cls,
        patch(
            "cqros.cli.verify_processed.LongShortVerifier",
            wraps=LongShortVerifier,
        ) as ls_cls,
        patch(
            "cqros.cli.verify_processed.VerificationRunner",
            wraps=VerificationRunner,
        ) as runner_cls,
    ):
        runner = build_verification_runner(options)

    assert isinstance(runner, VerificationRunner)
    layout_cls.assert_called_once()
    store_cls.assert_called_once()
    processed_cls.assert_called_once()
    ohlcv_cls.assert_called_once()
    funding_cls.assert_called_once()
    oi_cls.assert_called_once()
    taker_cls.assert_called_once()
    ls_cls.assert_called_once()
    runner_cls.assert_called_once()


def test_run_verification_empty_work_passes(tmp_path: Path) -> None:
    """Empty discovery produces a PASS summary without invoking the runner."""
    runner = MagicMock(spec=VerificationRunner)
    summary = _run(
        run_verification(
            runner=runner,
            options=_options(storage_root=tmp_path),
            work=(),
        )
    )
    assert summary.repository_passed is True
    assert summary.successful_tasks == 0
    assert summary.failed_tasks == 0
    runner.verify_ohlcv.assert_not_called()


def test_run_verification_aggregates_pass(tmp_path: Path) -> None:
    """Successful reports with zero counters produce PASS status."""
    _touch_partition(tmp_path, dataset="ohlcv", symbol="BTCUSDT", timeframe="1h", year=2024)
    repository = ProcessedMarketDataRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(repository, _options(storage_root=tmp_path, datasets=("ohlcv",)))

    runner = MagicMock(spec=VerificationRunner)
    runner.verify_ohlcv.return_value = VerificationSummary(
        dataset="ohlcv",
        exchange="binance",
        market="usdt_perpetual",
        results=(
            VerificationTaskResult(
                symbol="BTCUSDT",
                timeframe="1h",
                year=2024,
                status="succeeded",
                report=_passed_report(rows=42),
            ),
        ),
    )

    summary = _run(
        run_verification(
            runner=runner,
            options=_options(storage_root=tmp_path, workers=1),
            work=work,
        )
    )

    assert summary.symbols_verified == 1
    assert summary.datasets_verified == 1
    assert summary.timeframes_verified == 1
    assert summary.successful_tasks == 1
    assert summary.failed_tasks == 0
    assert summary.rows_checked == 42
    assert summary.repository_passed is True


def test_run_verification_fail_on_counters(tmp_path: Path) -> None:
    """Positive verification counters produce FAIL repository status."""
    _touch_partition(tmp_path, dataset="ohlcv", symbol="BTCUSDT", timeframe="1h", year=2024)
    repository = ProcessedMarketDataRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(repository, _options(storage_root=tmp_path, datasets=("ohlcv",)))

    runner = MagicMock(spec=VerificationRunner)
    runner.verify_ohlcv.return_value = VerificationSummary(
        dataset="ohlcv",
        exchange="binance",
        market="usdt_perpetual",
        results=(
            VerificationTaskResult(
                symbol="BTCUSDT",
                timeframe="1h",
                year=2024,
                status="succeeded",
                report=_failed_report(null_rows=3),
            ),
        ),
    )

    summary = _run(
        run_verification(
            runner=runner,
            options=_options(storage_root=tmp_path, workers=1),
            work=work,
        )
    )

    assert summary.successful_tasks == 1
    assert summary.null_rows == 3
    assert summary.repository_passed is False


def test_run_verification_fail_on_task_failure(tmp_path: Path) -> None:
    """Failed tasks produce FAIL repository status."""
    _touch_partition(tmp_path, dataset="ohlcv", symbol="BTCUSDT", timeframe="1h", year=2024)
    repository = ProcessedMarketDataRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(repository, _options(storage_root=tmp_path, datasets=("ohlcv",)))

    runner = MagicMock(spec=VerificationRunner)
    runner.verify_ohlcv.return_value = VerificationSummary(
        dataset="ohlcv",
        exchange="binance",
        market="usdt_perpetual",
        results=(
            VerificationTaskResult(
                symbol="BTCUSDT",
                timeframe="1h",
                year=2024,
                status="failed",
                error_type="RuntimeError",
                error_message="boom",
            ),
        ),
    )

    summary = _run(
        run_verification(
            runner=runner,
            options=_options(storage_root=tmp_path, workers=1),
            work=work,
        )
    )

    assert summary.failed_tasks == 1
    assert summary.repository_passed is False


def test_run_verification_failure_isolation(tmp_path: Path) -> None:
    """A raised runner exception does not stop remaining symbols."""
    _touch_partition(tmp_path, dataset="ohlcv", symbol="BTCUSDT", timeframe="1h", year=2024)
    _touch_partition(tmp_path, dataset="ohlcv", symbol="ETHUSDT", timeframe="1h", year=2024)
    repository = ProcessedMarketDataRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(repository, _options(storage_root=tmp_path, datasets=("ohlcv",)))

    runner = MagicMock(spec=VerificationRunner)

    def _verify_ohlcv(**kwargs: object) -> VerificationSummary:
        symbols = cast(tuple[str, ...], kwargs["symbols"])
        symbol = symbols[0]
        if symbol == "BTCUSDT":
            raise RuntimeError("btc failed")
        return VerificationSummary(
            dataset="ohlcv",
            exchange="binance",
            market="usdt_perpetual",
            results=(
                VerificationTaskResult(
                    symbol=symbol,
                    timeframe="1h",
                    year=2024,
                    status="succeeded",
                    report=_passed_report(),
                ),
            ),
        )

    runner.verify_ohlcv.side_effect = _verify_ohlcv

    summary = _run(
        run_verification(
            runner=runner,
            options=_options(storage_root=tmp_path, workers=1),
            work=work,
        )
    )

    assert summary.successful_tasks == 1
    assert summary.failed_tasks == 1
    assert summary.repository_passed is False


def test_run_verification_respects_worker_count(tmp_path: Path) -> None:
    """Worker pool creates exactly worker_count concurrent workers."""
    _touch_partition(tmp_path, dataset="ohlcv", symbol="BTCUSDT", timeframe="1h", year=2024)
    _touch_partition(tmp_path, dataset="ohlcv", symbol="ETHUSDT", timeframe="1h", year=2024)
    repository = ProcessedMarketDataRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(repository, _options(storage_root=tmp_path, datasets=("ohlcv",)))

    runner = MagicMock(spec=VerificationRunner)
    runner.verify_ohlcv.return_value = VerificationSummary(
        dataset="ohlcv",
        exchange="binance",
        market="usdt_perpetual",
        results=(
            VerificationTaskResult(
                symbol="BTCUSDT",
                timeframe="1h",
                year=2024,
                status="succeeded",
                report=_passed_report(),
            ),
        ),
    )

    created: list[str] = []
    real_create_task = asyncio.create_task

    def _tracking_create_task(coro: object, *, name: str | None = None) -> asyncio.Task[object]:
        if name is not None:
            created.append(name)
        return real_create_task(cast(Coroutine[object, object, object], coro), name=name)

    with patch("cqros.cli.verify_processed.asyncio.create_task", side_effect=_tracking_create_task):
        _run(
            run_verification(
                runner=runner,
                options=_options(storage_root=tmp_path, workers=3),
                work=work,
            )
        )

    worker_names = [name for name in created if name.startswith("verify-processed-worker-")]
    assert len(worker_names) == 3


def test_format_summary_pass_contract() -> None:
    """PASS summary output matches the documented report shape."""
    text = format_summary(
        VerifyProcessedSummary(
            symbols_verified=529,
            datasets_verified=5,
            timeframes_verified=4,
            successful_tasks=10872,
            failed_tasks=0,
            rows_checked=6_606_604,
            duplicate_timestamps=0,
            null_rows=0,
            nan_rows=0,
            invalid_timestamps=0,
            invalid_numeric_rows=0,
            warnings=0,
            duration_seconds=1.234,
            repository_passed=True,
        )
    )
    assert "CQROS Verification Summary" in text
    assert "Symbols verified: 529" in text
    assert "Successful tasks: 10872" in text
    assert "Rows checked: 6606604" in text
    assert "Repository status:" in text
    assert "PASS" in text
    assert "FAIL" not in text.split("Repository status:")[-1]


def test_format_summary_fail_contract() -> None:
    """FAIL summary includes repository FAIL status."""
    text = format_summary(
        VerifyProcessedSummary(
            symbols_verified=1,
            datasets_verified=1,
            timeframes_verified=1,
            successful_tasks=1,
            failed_tasks=0,
            rows_checked=10,
            duplicate_timestamps=2,
            null_rows=0,
            nan_rows=0,
            invalid_timestamps=0,
            invalid_numeric_rows=0,
            warnings=0,
            duration_seconds=0.5,
            repository_passed=False,
        )
    )
    assert "Duplicate timestamps: 2" in text
    assert "FAIL" in text.split("Repository status:")[-1]


def test_main_exit_code_pass(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """main returns 0 when repository status is PASS."""
    _touch_partition(tmp_path, dataset="ohlcv", symbol="BTCUSDT", timeframe="1h", year=2024)

    runner = MagicMock(spec=VerificationRunner)
    runner.verify_ohlcv.return_value = VerificationSummary(
        dataset="ohlcv",
        exchange="binance",
        market="usdt_perpetual",
        results=(
            VerificationTaskResult(
                symbol="BTCUSDT",
                timeframe="1h",
                year=2024,
                status="succeeded",
                report=_passed_report(),
            ),
        ),
    )

    with (
        patch("cqros.cli.verify_processed.DEFAULT_STORAGE_ROOT", str(tmp_path)),
        patch(
            "cqros.cli.verify_processed.build_verification_runner",
            return_value=runner,
        ),
    ):
        code = _run(main(["--dataset", "ohlcv", "--workers", "1"]))

    captured = capsys.readouterr()
    assert code == 0
    assert "PASS" in captured.out


def test_main_exit_code_fail(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """main returns 1 when repository status is FAIL."""
    _touch_partition(tmp_path, dataset="ohlcv", symbol="BTCUSDT", timeframe="1h", year=2024)

    runner = MagicMock(spec=VerificationRunner)
    runner.verify_ohlcv.return_value = VerificationSummary(
        dataset="ohlcv",
        exchange="binance",
        market="usdt_perpetual",
        results=(
            VerificationTaskResult(
                symbol="BTCUSDT",
                timeframe="1h",
                year=2024,
                status="succeeded",
                report=_failed_report(),
            ),
        ),
    )

    with (
        patch("cqros.cli.verify_processed.DEFAULT_STORAGE_ROOT", str(tmp_path)),
        patch(
            "cqros.cli.verify_processed.build_verification_runner",
            return_value=runner,
        ),
    ):
        code = _run(main(["--dataset", "ohlcv", "--workers", "1"]))

    captured = capsys.readouterr()
    assert code == 1
    assert "FAIL" in captured.out


def test_main_validation_error_exit_code() -> None:
    """Fatal CLI validation errors return exit code 1."""
    code = _run(main(["--workers", "0"]))
    assert code == 1


def test_cli_datasets_match_process_universe_surface() -> None:
    """Verification CLI datasets mirror the process-universe surface."""
    assert CLI_DATASETS == (
        "ohlcv",
        "funding",
        "open_interest",
        "taker_volume",
        "long_short",
    )


def test_configure_logging_debug(tmp_path: Path) -> None:
    """--debug enables DEBUG logging for the cqros logger."""
    with (
        patch("cqros.cli.verify_processed.DEFAULT_STORAGE_ROOT", str(tmp_path)),
        patch("cqros.cli.verify_processed.build_verification_runner") as build_runner,
    ):
        build_runner.return_value = MagicMock(spec=VerificationRunner)
        _run(main(["--debug"]))
    assert logging.getLogger("cqros").level == logging.DEBUG


def test_format_partition_failure_processing_validation_error() -> None:
    """ProcessingValidationError failures include code and message fields."""
    text = format_partition_failure(
        dataset="Open Interest",
        symbol="BTCUSDT",
        timeframe="1h",
        partition="2025.parquet",
        verifier="OpenInterestVerifier",
        exception_type="ProcessingValidationError",
        message="Missing required column: timestamp",
        code=ERROR_REQUIRED_COLUMNS,
    )
    assert text.startswith("FAILED\n")
    assert "Dataset: Open Interest" in text
    assert "Symbol: BTCUSDT" in text
    assert "Timeframe: 1h" in text
    assert "Partition: 2025.parquet" in text
    assert "Verifier: OpenInterestVerifier" in text
    assert "Exception: ProcessingValidationError" in text
    assert f"Code: {ERROR_REQUIRED_COLUMNS}" in text
    assert "Message: Missing required column: timestamp" in text
    assert "Traceback" not in text


def test_format_partition_failure_generic_exception() -> None:
    """Generic exceptions omit Code and still surface type and message."""
    text = format_partition_failure(
        dataset="OHLCV",
        symbol="ETHUSDT",
        timeframe="4h",
        partition="2024.parquet",
        verifier="OHLCVVerifier",
        exception_type="TypeError",
        message="unsupported operand type(s)",
    )
    assert "Exception: TypeError" in text
    assert "Message: unsupported operand type(s)" in text
    assert "Code:" not in text
    assert "Traceback" not in text


def test_run_verification_prints_runner_task_failures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Failed runner task results print structured partition diagnostics."""
    _touch_partition(tmp_path, dataset="open_interest", symbol="BTCUSDT", timeframe="1h", year=2025)
    repository = ProcessedMarketDataRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(
        repository,
        _options(storage_root=tmp_path, datasets=("open_interest",)),
    )

    runner = MagicMock(spec=VerificationRunner)
    runner.verify_open_interest.return_value = VerificationSummary(
        dataset="open_interest",
        exchange="binance",
        market="usdt_perpetual",
        results=(
            VerificationTaskResult(
                symbol="BTCUSDT",
                timeframe="1h",
                year=2025,
                status="failed",
                error_type="ProcessingValidationError",
                error_message="Missing required column: timestamp",
                error_code=ERROR_REQUIRED_COLUMNS,
            ),
        ),
    )

    summary = _run(
        run_verification(
            runner=runner,
            options=_options(storage_root=tmp_path, workers=1),
            work=work,
        )
    )

    captured = capsys.readouterr().out
    assert summary.failed_tasks == 1
    assert "FAILED" in captured
    assert "Dataset: Open Interest" in captured
    assert "Symbol: BTCUSDT" in captured
    assert "Timeframe: 1h" in captured
    assert "Partition: 2025.parquet" in captured
    assert "Verifier: OpenInterestVerifier" in captured
    assert "Exception: ProcessingValidationError" in captured
    assert f"Code: {ERROR_REQUIRED_COLUMNS}" in captured
    assert "Message: Missing required column: timestamp" in captured
    assert "Traceback" not in captured


def test_run_verification_prints_raised_processing_validation_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Raised ProcessingValidationError is formatted with code and message."""
    _touch_partition(tmp_path, dataset="open_interest", symbol="BTCUSDT", timeframe="1h", year=2025)
    repository = ProcessedMarketDataRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(
        repository,
        _options(storage_root=tmp_path, datasets=("open_interest",)),
    )

    runner = MagicMock(spec=VerificationRunner)
    runner.verify_open_interest.side_effect = ProcessingValidationError(
        "Missing required column: timestamp",
        error_code=ERROR_REQUIRED_COLUMNS,
    )

    summary = _run(
        run_verification(
            runner=runner,
            options=_options(storage_root=tmp_path, workers=1),
            work=work,
        )
    )

    captured = capsys.readouterr().out
    assert summary.failed_tasks == 1
    assert "Exception: ProcessingValidationError" in captured
    assert f"Code: {ERROR_REQUIRED_COLUMNS}" in captured
    assert "Message: Missing required column: timestamp" in captured
    assert "Traceback" not in captured


def test_run_verification_prints_raised_generic_exception(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Raised generic exceptions print type and message without a code."""
    _touch_partition(tmp_path, dataset="ohlcv", symbol="BTCUSDT", timeframe="1h", year=2024)
    repository = ProcessedMarketDataRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(repository, _options(storage_root=tmp_path, datasets=("ohlcv",)))

    runner = MagicMock(spec=VerificationRunner)
    runner.verify_ohlcv.side_effect = TypeError("boom")

    summary = _run(
        run_verification(
            runner=runner,
            options=_options(storage_root=tmp_path, workers=1),
            work=work,
        )
    )

    captured = capsys.readouterr().out
    assert summary.failed_tasks == 1
    assert "Exception: TypeError" in captured
    assert "Message: boom" in captured
    assert "Code:" not in captured
    assert "Traceback" not in captured


def test_run_verification_debug_logs_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """--debug keeps the structured printout and logs the full traceback."""
    _touch_partition(tmp_path, dataset="ohlcv", symbol="BTCUSDT", timeframe="1h", year=2024)
    repository = ProcessedMarketDataRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(repository, _options(storage_root=tmp_path, datasets=("ohlcv",)))

    runner = MagicMock(spec=VerificationRunner)
    runner.verify_ohlcv.side_effect = RuntimeError("boom")

    with caplog.at_level(logging.ERROR, logger="cqros.cli.verify_processed"):
        summary = _run(
            run_verification(
                runner=runner,
                options=_options(storage_root=tmp_path, workers=1, debug=True),
                work=work,
            )
        )

    captured = capsys.readouterr().out
    assert summary.failed_tasks == 1
    assert "FAILED" in captured
    assert "Exception: RuntimeError" in captured
    assert "Message: boom" in captured
    assert "Traceback" not in captured
    assert "boom" in caplog.text
    assert "Traceback" in caplog.text
    assert "RuntimeError: boom" in caplog.text


def test_run_verification_normal_mode_does_not_log_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Normal mode prints structured failure details without a traceback."""
    _touch_partition(tmp_path, dataset="ohlcv", symbol="BTCUSDT", timeframe="1h", year=2024)
    repository = ProcessedMarketDataRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(repository, _options(storage_root=tmp_path, datasets=("ohlcv",)))

    runner = MagicMock(spec=VerificationRunner)
    runner.verify_ohlcv.side_effect = RuntimeError("boom")

    with caplog.at_level(logging.WARNING, logger="cqros.cli.verify_processed"):
        summary = _run(
            run_verification(
                runner=runner,
                options=_options(storage_root=tmp_path, workers=1, debug=False),
                work=work,
            )
        )

    captured = capsys.readouterr().out
    assert summary.failed_tasks == 1
    assert "FAILED" in captured
    assert "Traceback" not in captured
    assert "Traceback" not in caplog.text
    assert "Failed symbol dataset verification; continuing" in caplog.text


def test_run_verification_continues_after_printed_failures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Printed failures do not stop remaining symbols from verifying."""
    _touch_partition(tmp_path, dataset="ohlcv", symbol="BTCUSDT", timeframe="1h", year=2024)
    _touch_partition(tmp_path, dataset="ohlcv", symbol="ETHUSDT", timeframe="1h", year=2024)
    repository = ProcessedMarketDataRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(repository, _options(storage_root=tmp_path, datasets=("ohlcv",)))

    runner = MagicMock(spec=VerificationRunner)

    def _verify_ohlcv(**kwargs: object) -> VerificationSummary:
        symbols = cast(tuple[str, ...], kwargs["symbols"])
        symbol = symbols[0]
        if symbol == "BTCUSDT":
            raise ProcessingValidationError(
                "Missing required column: open_time",
                error_code=ERROR_REQUIRED_COLUMNS,
            )
        return VerificationSummary(
            dataset="ohlcv",
            exchange="binance",
            market="usdt_perpetual",
            results=(
                VerificationTaskResult(
                    symbol=symbol,
                    timeframe="1h",
                    year=2024,
                    status="succeeded",
                    report=_passed_report(),
                ),
            ),
        )

    runner.verify_ohlcv.side_effect = _verify_ohlcv

    summary = _run(
        run_verification(
            runner=runner,
            options=_options(storage_root=tmp_path, workers=1),
            work=work,
        )
    )

    captured = capsys.readouterr().out
    assert summary.successful_tasks == 1
    assert summary.failed_tasks == 1
    assert summary.repository_passed is False
    assert "FAILED" in captured
    assert "Symbol: BTCUSDT" in captured
    assert runner.verify_ohlcv.call_count == 2
