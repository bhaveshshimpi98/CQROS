"""Unit tests for CQROS factor-verification CLI.

All repository and verifier interactions are mocked. Tests perform no
filesystem writes.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from cqros.cli.verify_factors import (
    DiscoveredWorkItem,
    VerifyFactorsOptions,
    VerifyFactorsSummary,
    build_options,
    build_parser,
    discover_work,
    format_debug_diagnostics,
    format_global_failure_report,
    format_partition_failure,
    format_summary,
    main,
    run_verification,
)
from cqros.config.models import ResearchConfig
from cqros.core.constants import DEFAULT_STORAGE_ROOT
from cqros.core.exceptions import ValidationError
from cqros.factors import FactorPartitionRef, FactorsRepository, FactorVerifier
from cqros.factors.verification.diagnostics import (
    FactorInvalidNumericDiagnostic,
    FactorNullDiagnostic,
    FactorVerificationDiagnostics,
    FactorVerificationReport,
    FactorWarningDiagnostic,
    GlobalFailureFinding,
    InvalidNumericKind,
    NullClassification,
)
from cqros.factors.verification.exceptions import ERROR_SCHEMA_MISMATCH

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
    symbols: tuple[str, ...] | None = None,
    timeframes: tuple[str, ...] | None = None,
    years: tuple[int, ...] | None = None,
    workers: int = ResearchConfig().worker_count,
    verbose: bool = False,
    debug: bool = False,
) -> VerifyFactorsOptions:
    """Build options for tests against an in-memory storage root path."""
    return VerifyFactorsOptions(
        storage_root=storage_root,
        symbols=symbols,
        timeframes=timeframes,
        years=years,
        workers=workers,
        verbose=verbose,
        debug=debug,
    )


def _passed_report(*, rows: int = 10) -> FactorVerificationReport:
    """Return a passing verification report."""
    return FactorVerificationReport(
        rows_checked=rows,
        duplicate_timestamp_rows=0,
        null_rows=0,
        nan_rows=0,
        invalid_timestamp_rows=0,
        invalid_numeric_rows=0,
        warmup_null_rows=0,
        domain_null_rows=0,
        unexpected_null_rows=0,
        positive_inf_rows=0,
        negative_inf_rows=0,
        non_finite_rows=0,
        warnings=(),
        diagnostics=FactorVerificationDiagnostics(
            null_diagnostics=(),
            invalid_numeric_diagnostics=(),
            warning_diagnostics=(),
        ),
        passed=True,
    )


def _failed_report(*, null_rows: int = 1) -> FactorVerificationReport:
    """Return a failing verification report with unexpected NULLs."""
    return FactorVerificationReport(
        rows_checked=10,
        duplicate_timestamp_rows=0,
        null_rows=null_rows,
        nan_rows=0,
        invalid_timestamp_rows=0,
        invalid_numeric_rows=0,
        warmup_null_rows=0,
        domain_null_rows=0,
        unexpected_null_rows=null_rows,
        positive_inf_rows=0,
        negative_inf_rows=0,
        non_finite_rows=0,
        warnings=(f"UNEXPECTED_NULLS factor=rsi count={null_rows}",),
        diagnostics=FactorVerificationDiagnostics(
            null_diagnostics=(
                FactorNullDiagnostic(
                    factor_name="rsi",
                    count=null_rows,
                    first_open_time=1_704_067_200_000,
                    last_open_time=1_704_067_200_000,
                    only_at_beginning=False,
                    appears_after_valid=True,
                    classification=NullClassification.UNEXPECTED_NULLS,
                ),
            ),
            invalid_numeric_diagnostics=(),
            warning_diagnostics=(
                FactorWarningDiagnostic(
                    warning_type="UNEXPECTED_NULLS",
                    factor_name="rsi",
                    count=null_rows,
                ),
            ),
        ),
        passed=False,
    )


def _warmup_report(*, warmup_null_rows: int = 3) -> FactorVerificationReport:
    """Return a passing report that contains warmup-only NULLs."""
    return FactorVerificationReport(
        rows_checked=10,
        duplicate_timestamp_rows=0,
        null_rows=warmup_null_rows,
        nan_rows=0,
        invalid_timestamp_rows=0,
        invalid_numeric_rows=0,
        warmup_null_rows=warmup_null_rows,
        domain_null_rows=0,
        unexpected_null_rows=0,
        positive_inf_rows=0,
        negative_inf_rows=0,
        non_finite_rows=0,
        warnings=(f"WARMUP_NULLS factor=rsi count={warmup_null_rows}",),
        diagnostics=FactorVerificationDiagnostics(
            null_diagnostics=(
                FactorNullDiagnostic(
                    factor_name="rsi",
                    count=warmup_null_rows,
                    first_open_time=1_704_067_200_000,
                    last_open_time=1_704_240_000_000,
                    only_at_beginning=True,
                    appears_after_valid=False,
                    classification=NullClassification.WARMUP_NULLS,
                ),
            ),
            invalid_numeric_diagnostics=(),
            warning_diagnostics=(
                FactorWarningDiagnostic(
                    warning_type="WARMUP_NULLS",
                    factor_name="rsi",
                    count=warmup_null_rows,
                ),
            ),
        ),
        passed=True,
    )


def _domain_null_report(*, domain_null_rows: int = 2) -> FactorVerificationReport:
    """Return a passing report that contains domain NULLs only."""
    return FactorVerificationReport(
        rows_checked=10,
        duplicate_timestamp_rows=0,
        null_rows=domain_null_rows,
        nan_rows=0,
        invalid_timestamp_rows=0,
        invalid_numeric_rows=0,
        warmup_null_rows=0,
        domain_null_rows=domain_null_rows,
        unexpected_null_rows=0,
        positive_inf_rows=0,
        negative_inf_rows=0,
        non_finite_rows=0,
        warnings=(f"DOMAIN_NULLS factor=ease_of_movement count={domain_null_rows}",),
        diagnostics=FactorVerificationDiagnostics(
            null_diagnostics=(
                FactorNullDiagnostic(
                    factor_name="ease_of_movement",
                    count=domain_null_rows,
                    first_open_time=1_704_067_200_000,
                    last_open_time=1_704_240_000_000,
                    only_at_beginning=False,
                    appears_after_valid=True,
                    classification=NullClassification.DOMAIN_NULLS,
                ),
            ),
            invalid_numeric_diagnostics=(),
            warning_diagnostics=(
                FactorWarningDiagnostic(
                    warning_type="DOMAIN_NULLS",
                    factor_name="ease_of_movement",
                    count=domain_null_rows,
                ),
            ),
        ),
        passed=True,
    )


def _partition_ref(
    *,
    manager: str = _MANAGER,
    symbol: str = _SYMBOL,
    timeframe: str = _TIMEFRAME,
    year: int = _YEAR,
) -> FactorPartitionRef:
    """Build a factor partition reference for discovery mocks."""
    return FactorPartitionRef(
        manager=manager,
        exchange="binance",
        market="usdt_perpetual",
        symbol=symbol,
        timeframe=timeframe,
        year=year,
    )


def test_build_parser_defaults() -> None:
    """Omitted optional flags keep discovery defaults."""
    args = build_parser().parse_args([])
    assert args.symbols is None
    assert args.timeframes is None
    assert args.years is None
    assert args.workers == ResearchConfig().worker_count
    assert args.storage_root is None
    assert args.verbose is False
    assert args.debug is False


def test_build_parser_accepts_all_flags(tmp_path: Path) -> None:
    """Parser accepts every documented factor-verification flag."""
    args = build_parser().parse_args(
        [
            "--symbols",
            "BTCUSDT",
            "--timeframes",
            "1h",
            "--years",
            "2024",
            "--workers",
            "3",
            "--storage-root",
            str(tmp_path),
            "--verbose",
            "--debug",
        ]
    )
    assert args.symbols == ["BTCUSDT"]
    assert args.timeframes == ["1h"]
    assert args.years == ["2024"]
    assert args.workers == 3
    assert args.storage_root == tmp_path
    assert args.verbose is True
    assert args.debug is True


def test_build_options_defaults() -> None:
    """Omitted filters map to discovery-all options."""
    options = build_options(build_parser().parse_args([]))
    assert options.storage_root == Path(DEFAULT_STORAGE_ROOT)
    assert options.symbols is None
    assert options.timeframes is None
    assert options.years is None
    assert options.workers == ResearchConfig().worker_count


def test_build_options_maps_filters(tmp_path: Path) -> None:
    """Explicit CLI flags map onto VerifyFactorsOptions."""
    options = build_options(
        build_parser().parse_args(
            [
                "--symbols",
                "ETHUSDT",
                "--timeframes",
                "4h",
                "--years",
                "2025",
                "--workers",
                "2",
                "--storage-root",
                str(tmp_path),
                "--debug",
            ]
        )
    )
    assert options.symbols == ("ETHUSDT",)
    assert options.timeframes == ("4h",)
    assert options.years == (2025,)
    assert options.workers == 2
    assert options.storage_root == tmp_path
    assert options.debug is True


def test_build_options_rejects_non_positive_workers() -> None:
    """Non-positive --workers fails validation."""
    args = build_parser().parse_args(["--workers", "0"])
    with pytest.raises(ValidationError, match="workers must be greater than 0") as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-VERIFY-FACTORS-001"


def test_build_options_rejects_unsupported_timeframe() -> None:
    """Unsupported --timeframes values fail validation."""
    args = build_parser().parse_args(["--timeframes", "2x"])
    with pytest.raises(ValidationError, match="unsupported timeframe") as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-VERIFY-FACTORS-002"


def test_build_options_rejects_invalid_year() -> None:
    """Non-integer --years values fail validation."""
    args = build_parser().parse_args(["--years", "nope"])
    with pytest.raises(ValidationError, match="invalid year") as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-VERIFY-FACTORS-003"


def test_discover_work_finds_partitions() -> None:
    """Discovery groups mocked partitions without filesystem access."""
    repository = MagicMock(spec=FactorsRepository)
    repository.discover_partitions.return_value = (
        _partition_ref(symbol="BTCUSDT", year=2024),
        _partition_ref(symbol="ETHUSDT", year=2023),
        _partition_ref(symbol="ETHUSDT", year=2024),
    )
    work = discover_work(repository, _options(storage_root=Path("unused")))

    assert len(work) == 2
    assert work[0].symbol == "BTCUSDT"
    assert work[0].years == (2024,)
    assert work[1].symbol == "ETHUSDT"
    assert work[1].years == (2023, 2024)


def test_discover_work_filters() -> None:
    """Year filters apply in discover_work; repository receives CLI allowlists."""
    repository = MagicMock(spec=FactorsRepository)
    repository.discover_partitions.return_value = (
        _partition_ref(symbol="BTCUSDT", timeframe="1h", year=2024),
        _partition_ref(symbol="BTCUSDT", timeframe="1h", year=2025),
    )
    work = discover_work(
        repository,
        _options(
            storage_root=Path("unused"),
            symbols=("BTCUSDT",),
            timeframes=("1h",),
            years=(2024,),
        ),
    )

    assert work == (
        DiscoveredWorkItem(
            manager=_MANAGER,
            symbol="BTCUSDT",
            timeframe="1h",
            years=(2024,),
        ),
    )
    repository.discover_partitions.assert_called_once_with(
        symbols=("BTCUSDT",),
        timeframes=("1h",),
        exchange="binance",
        market="usdt_perpetual",
    )


def test_run_verification_empty_work_passes() -> None:
    """Empty discovery produces a PASS summary without invoking the verifier."""
    repository = MagicMock(spec=FactorsRepository)
    verifier = MagicMock(spec=FactorVerifier)
    summary = _run(
        run_verification(
            repository=repository,
            verifier=verifier,
            options=_options(storage_root=Path("unused")),
            work=(),
        )
    )
    assert summary.repository_passed is True
    assert summary.successful_tasks == 0
    assert summary.failed_tasks == 0
    verifier.verify.assert_not_called()


def test_run_verification_aggregates_pass() -> None:
    """Successful reports with zero counters produce PASS status."""
    work = (
        DiscoveredWorkItem(
            manager=_MANAGER,
            symbol="BTCUSDT",
            timeframe="1h",
            years=(2024,),
        ),
    )
    repository = MagicMock(spec=FactorsRepository)
    repository.load.return_value = pl.DataFrame({"open_time": [1]})
    verifier = MagicMock(spec=FactorVerifier)
    verifier.verify.return_value = _passed_report(rows=7)

    summary = _run(
        run_verification(
            repository=repository,
            verifier=verifier,
            options=_options(storage_root=Path("unused"), workers=1),
            work=work,
        )
    )

    assert summary.repository_passed is True
    assert summary.successful_tasks == 1
    assert summary.failed_tasks == 0
    assert summary.rows_checked == 7
    assert summary.datasets_verified == 1


def test_run_verification_repository_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Repository load failures produce FAIL and structured diagnostics."""
    work = (
        DiscoveredWorkItem(
            manager=_MANAGER,
            symbol="BTCUSDT",
            timeframe="1h",
            years=(2024,),
        ),
    )
    repository = MagicMock(spec=FactorsRepository)
    repository.load.side_effect = RuntimeError("partition missing")
    verifier = MagicMock(spec=FactorVerifier)

    summary = _run(
        run_verification(
            repository=repository,
            verifier=verifier,
            options=_options(storage_root=Path("unused"), workers=1),
            work=work,
        )
    )

    captured = capsys.readouterr().out
    assert summary.repository_passed is False
    assert summary.failed_tasks == 1
    assert summary.successful_tasks == 0
    assert "FAILED" in captured
    assert "Dataset: Factors" in captured
    assert "Verifier: FactorVerifier" in captured
    assert "partition missing" in captured
    verifier.verify.assert_not_called()


def test_run_verification_schema_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Raised schema exceptions produce FAIL and structured diagnostics."""
    work = (
        DiscoveredWorkItem(
            manager=_MANAGER,
            symbol="BTCUSDT",
            timeframe="1h",
            years=(2024,),
        ),
    )
    repository = MagicMock(spec=FactorsRepository)
    repository.load.return_value = pl.DataFrame({"open_time": [1]})
    verifier = MagicMock(spec=FactorVerifier)
    verifier.verify.side_effect = ValidationError(
        "factor schema dtype mismatch",
        error_code=ERROR_SCHEMA_MISMATCH,
        details={},
    )

    summary = _run(
        run_verification(
            repository=repository,
            verifier=verifier,
            options=_options(storage_root=Path("unused"), workers=1),
            work=work,
        )
    )

    captured = capsys.readouterr().out
    assert summary.repository_passed is False
    assert summary.failed_tasks == 1
    assert "FAILED" in captured
    assert "Dataset: Factors" in captured
    assert "Verifier: FactorVerifier" in captured
    assert f"Code: {ERROR_SCHEMA_MISMATCH}" in captured


def test_run_verification_fail_on_counters() -> None:
    """Positive verification counters produce FAIL repository status."""
    work = (
        DiscoveredWorkItem(
            manager=_MANAGER,
            symbol="BTCUSDT",
            timeframe="1h",
            years=(2024,),
        ),
    )
    repository = MagicMock(spec=FactorsRepository)
    repository.load.return_value = pl.DataFrame({"open_time": [1]})
    verifier = MagicMock(spec=FactorVerifier)
    verifier.verify.return_value = _failed_report(null_rows=2)

    summary = _run(
        run_verification(
            repository=repository,
            verifier=verifier,
            options=_options(storage_root=Path("unused"), workers=1),
            work=work,
        )
    )

    assert summary.repository_passed is False
    assert summary.successful_tasks == 1
    assert summary.null_rows == 2
    assert summary.unexpected_null_rows == 2
    assert summary.warnings == 1


def test_run_verification_warmup_nulls_do_not_fail() -> None:
    """Warmup-only NULLs keep repository status PASS."""
    work = (
        DiscoveredWorkItem(
            manager=_MANAGER,
            symbol="BTCUSDT",
            timeframe="1h",
            years=(2024,),
        ),
    )
    repository = MagicMock(spec=FactorsRepository)
    repository.load.return_value = pl.DataFrame({"open_time": [1]})
    verifier = MagicMock(spec=FactorVerifier)
    verifier.verify.return_value = _warmup_report(warmup_null_rows=5)

    summary = _run(
        run_verification(
            repository=repository,
            verifier=verifier,
            options=_options(storage_root=Path("unused"), workers=1),
            work=work,
        )
    )

    assert summary.repository_passed is True
    assert summary.warmup_null_rows == 5
    assert summary.domain_null_rows == 0
    assert summary.unexpected_null_rows == 0
    assert summary.warnings == 1


def test_run_verification_debug_prints_diagnostics(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--debug prints structured per-factor diagnostics."""
    work = (
        DiscoveredWorkItem(
            manager=_MANAGER,
            symbol="BTCUSDT",
            timeframe="1h",
            years=(2024,),
        ),
    )
    repository = MagicMock(spec=FactorsRepository)
    repository.load.return_value = pl.DataFrame({"open_time": [1]})
    verifier = MagicMock(spec=FactorVerifier)
    report = _failed_report(null_rows=2)
    report = FactorVerificationReport(
        rows_checked=report.rows_checked,
        duplicate_timestamp_rows=0,
        null_rows=2,
        nan_rows=0,
        invalid_timestamp_rows=0,
        invalid_numeric_rows=1,
        warmup_null_rows=0,
        domain_null_rows=0,
        unexpected_null_rows=2,
        positive_inf_rows=1,
        negative_inf_rows=0,
        non_finite_rows=0,
        warnings=report.warnings + ("POSITIVE_INFINITY factor=vwap_distance count=1",),
        diagnostics=FactorVerificationDiagnostics(
            null_diagnostics=report.diagnostics.null_diagnostics,
            invalid_numeric_diagnostics=(
                FactorInvalidNumericDiagnostic(
                    factor_name="vwap_distance",
                    open_time=1_704_067_200_000,
                    kind=InvalidNumericKind.POSITIVE_INFINITY,
                    count=1,
                ),
            ),
            warning_diagnostics=report.diagnostics.warning_diagnostics,
        ),
        passed=False,
    )
    verifier.verify.return_value = report

    summary = _run(
        run_verification(
            repository=repository,
            verifier=verifier,
            options=_options(storage_root=Path("unused"), workers=1, debug=True),
            work=work,
        )
    )

    captured = capsys.readouterr().out
    assert summary.repository_passed is False
    assert "Debug diagnostics:" in captured
    assert "Factor: rsi" in captured
    assert "Issue: Unexpected NULL" in captured
    assert "Factor: vwap_distance" in captured
    assert "Positive infinity" in captured


def test_run_verification_respects_worker_count() -> None:
    """Worker pool size is honored for concurrent symbols."""
    work = (
        DiscoveredWorkItem(
            manager=_MANAGER,
            symbol="BTCUSDT",
            timeframe="1h",
            years=(2024,),
        ),
        DiscoveredWorkItem(
            manager=_MANAGER,
            symbol="ETHUSDT",
            timeframe="1h",
            years=(2024,),
        ),
    )
    repository = MagicMock(spec=FactorsRepository)
    repository.load.return_value = pl.DataFrame({"open_time": [1]})
    verifier = MagicMock(spec=FactorVerifier)
    verifier.verify.return_value = _passed_report()

    with patch("cqros.cli.verify_factors.asyncio.create_task", wraps=asyncio.create_task) as create:
        summary = _run(
            run_verification(
                repository=repository,
                verifier=verifier,
                options=_options(storage_root=Path("unused"), workers=2),
                work=work,
            )
        )

    assert summary.successful_tasks == 2
    worker_names = [
        call.kwargs.get("name")
        for call in create.call_args_list
        if call.kwargs.get("name", "").startswith("verify-factors-worker-")
    ]
    assert len(worker_names) == 2


def test_format_summary_pass_and_fail() -> None:
    """Summary rendering mirrors the CQROS verification PASS/FAIL style."""
    passed = format_summary(
        VerifyFactorsSummary(
            symbols_verified=1,
            datasets_verified=1,
            timeframes_verified=1,
            successful_tasks=1,
            failed_tasks=0,
            rows_checked=10,
            duplicate_timestamps=0,
            null_rows=0,
            nan_rows=0,
            invalid_timestamps=0,
            invalid_numeric_rows=0,
            warmup_null_rows=0,
            domain_null_rows=0,
            unexpected_null_rows=0,
            positive_inf_rows=0,
            negative_inf_rows=0,
            non_finite_rows=0,
            warnings=0,
            duration_seconds=0.5,
            repository_passed=True,
        )
    )
    failed = format_summary(
        VerifyFactorsSummary(
            symbols_verified=1,
            datasets_verified=1,
            timeframes_verified=1,
            successful_tasks=0,
            failed_tasks=1,
            rows_checked=0,
            duplicate_timestamps=0,
            null_rows=1,
            nan_rows=0,
            invalid_timestamps=0,
            invalid_numeric_rows=0,
            warmup_null_rows=0,
            domain_null_rows=0,
            unexpected_null_rows=1,
            positive_inf_rows=0,
            negative_inf_rows=0,
            non_finite_rows=0,
            warnings=1,
            duration_seconds=0.5,
            repository_passed=False,
        )
    )
    assert "CQROS Verification Summary" in passed
    assert "PASS" in passed.split("Repository status:")[-1]
    assert "FAIL" in failed.split("Repository status:")[-1]
    assert "NULL rows: 1" in failed
    assert "Warmup NULLs: 0" in failed
    assert "Domain NULLs: 0" in failed
    assert "Unexpected NULLs: 1" in failed
    assert "Invalid +Inf: 0" in failed
    assert "Invalid -Inf: 0" in failed
    assert "Invalid NonFinite: 0" in failed
    assert "Invalid numeric rows: 0" in failed


def test_format_debug_diagnostics() -> None:
    """format_debug_diagnostics renders structured factor findings."""
    report = _warmup_report(warmup_null_rows=2)
    text = format_debug_diagnostics(
        manager=_MANAGER,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
        report=report,
    )
    assert "Debug diagnostics:" in text
    assert "Factor: rsi" in text
    assert "Issue: Warmup NULLs" in text


def test_run_verification_domain_nulls_do_not_fail() -> None:
    """Domain NULLs alone keep repository status PASS."""
    work = (
        DiscoveredWorkItem(
            manager=_MANAGER,
            symbol="BTCUSDT",
            timeframe="1h",
            years=(2024,),
        ),
    )
    repository = MagicMock(spec=FactorsRepository)
    repository.load.return_value = pl.DataFrame({"open_time": [1]})
    verifier = MagicMock(spec=FactorVerifier)
    verifier.verify.return_value = _domain_null_report(domain_null_rows=4)

    summary = _run(
        run_verification(
            repository=repository,
            verifier=verifier,
            options=_options(storage_root=Path("unused"), workers=1),
            work=work,
        )
    )

    assert summary.repository_passed is True
    assert summary.domain_null_rows == 4
    assert summary.unexpected_null_rows == 0
    assert summary.warmup_null_rows == 0


def test_format_partition_failure() -> None:
    """Structured failure reports include code and omit tracebacks."""
    text = format_partition_failure(
        dataset="Factors",
        symbol="BTCUSDT",
        timeframe="1h",
        partition="2025.parquet",
        verifier="FactorVerifier",
        exception_type="FactorValidationError",
        message="factor schema dtype mismatch",
        code=ERROR_SCHEMA_MISMATCH,
    )
    assert text.startswith("FAILED\n")
    assert "Dataset: Factors" in text
    assert "Verifier: FactorVerifier" in text
    assert f"Code: {ERROR_SCHEMA_MISMATCH}" in text
    assert "Traceback" not in text


def test_main_exit_code_pass(capsys: pytest.CaptureFixture[str]) -> None:
    """main returns 0 when repository status is PASS."""
    with (
        patch("cqros.cli.verify_factors.discover_work", return_value=()),
        patch("cqros.cli.verify_factors.FactorsRepository"),
        patch("cqros.cli.verify_factors.ParquetStore"),
        patch("cqros.cli.verify_factors.StorageLayout"),
    ):
        code = _run(main(["--workers", "1"]))

    captured = capsys.readouterr()
    assert code == 0
    assert "PASS" in captured.out


def test_main_exit_code_fail(capsys: pytest.CaptureFixture[str]) -> None:
    """main returns 1 when repository status is FAIL."""
    work = (
        DiscoveredWorkItem(
            manager=_MANAGER,
            symbol="BTCUSDT",
            timeframe="1h",
            years=(2024,),
        ),
    )

    with (
        patch("cqros.cli.verify_factors.discover_work", return_value=work),
        patch("cqros.cli.verify_factors.FactorsRepository") as repo_cls,
        patch("cqros.cli.verify_factors.FactorVerifier") as verifier_cls,
        patch("cqros.cli.verify_factors.ParquetStore"),
        patch("cqros.cli.verify_factors.StorageLayout"),
    ):
        repository = MagicMock(spec=FactorsRepository)
        repository.load.return_value = pl.DataFrame({"open_time": [1]})
        repo_cls.return_value = repository
        verifier = MagicMock(spec=FactorVerifier)
        verifier.verify.return_value = _failed_report()
        verifier_cls.return_value = verifier
        code = _run(main(["--workers", "1"]))

    captured = capsys.readouterr()
    assert code == 1
    assert "FAIL" in captured.out
    assert "CQROS Factor Failure Report" not in captured.out


def test_main_debug_prints_global_failure_report(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """FAIL with --debug emits the global Unexpected NULL / +Inf report."""
    work = (
        DiscoveredWorkItem(
            manager=_MANAGER,
            symbol="BTCUSDT",
            timeframe="1h",
            years=(2024,),
        ),
        DiscoveredWorkItem(
            manager=_MANAGER,
            symbol="ETHUSDT",
            timeframe="4h",
            years=(2026,),
        ),
    )
    null_report = _failed_report(null_rows=7)
    inf_report = FactorVerificationReport(
        rows_checked=10,
        duplicate_timestamp_rows=0,
        null_rows=0,
        nan_rows=0,
        invalid_timestamp_rows=0,
        invalid_numeric_rows=1,
        warmup_null_rows=0,
        domain_null_rows=0,
        unexpected_null_rows=0,
        positive_inf_rows=1,
        negative_inf_rows=0,
        non_finite_rows=0,
        warnings=("POSITIVE_INFINITY factor=price_zscore count=1",),
        diagnostics=FactorVerificationDiagnostics(
            null_diagnostics=(),
            invalid_numeric_diagnostics=(
                FactorInvalidNumericDiagnostic(
                    factor_name="price_zscore",
                    open_time=1_754_107_200_000,  # 2025-08-02 04:00 UTC approx
                    kind=InvalidNumericKind.POSITIVE_INFINITY,
                    count=1,
                ),
            ),
            warning_diagnostics=(
                FactorWarningDiagnostic(
                    warning_type="POSITIVE_INFINITY",
                    factor_name="price_zscore",
                    count=1,
                ),
            ),
        ),
        passed=False,
    )

    with (
        patch("cqros.cli.verify_factors.discover_work", return_value=work),
        patch("cqros.cli.verify_factors.FactorsRepository") as repo_cls,
        patch("cqros.cli.verify_factors.FactorVerifier") as verifier_cls,
        patch("cqros.cli.verify_factors.ParquetStore"),
        patch("cqros.cli.verify_factors.StorageLayout"),
    ):
        repository = MagicMock(spec=FactorsRepository)
        repository.load.return_value = pl.DataFrame({"open_time": [1]})
        repo_cls.return_value = repository
        verifier = MagicMock(spec=FactorVerifier)
        verifier.verify.side_effect = [null_report, inf_report]
        verifier_cls.return_value = verifier
        code = _run(main(["--workers", "1", "--debug"]))

    captured = capsys.readouterr().out
    assert code == 1
    assert "CQROS Factor Failure Report" in captured
    assert "Symbol: BTCUSDT" in captured
    assert "Issue: Unexpected NULL" in captured
    assert "Factor: rsi" in captured
    assert "Symbol: ETHUSDT" in captured
    assert "Timeframe: 4h" in captured
    assert "Year: 2026" in captured
    assert "Issue: +Inf" in captured
    assert "Factor: price_zscore" in captured
    assert "Timestamp:" in captured
    assert "Affected partitions: 2" in captured
    assert "Affected symbols: 2" in captured
    assert "Affected factors: 2" in captured


def test_format_global_failure_report_cli_export() -> None:
    """CLI re-exports the global failure report formatter."""
    text = format_global_failure_report(
        (
            GlobalFailureFinding(
                symbol="XYZUSDT",
                timeframe="4h",
                year=2026,
                factor_name="funding_rate_zscore",
                issue="Unexpected NULL",
                count=7,
                first_open_time=1_720_915_200_000,
                last_open_time=1_721_260_800_000,
            ),
        )
    )
    assert "Symbol: XYZUSDT" in text
    assert "Issue: Unexpected NULL" in text
    assert "Count: 7" in text
    assert "Affected partitions: 1" in text


def test_main_validation_error_exit_code() -> None:
    """Fatal CLI validation errors return exit code 1."""
    code = _run(main(["--workers", "0"]))
    assert code == 1


def test_configure_logging_debug() -> None:
    """--debug enables DEBUG logging for the cqros logger."""
    with (
        patch("cqros.cli.verify_factors.discover_work", return_value=()),
        patch("cqros.cli.verify_factors.FactorsRepository"),
        patch("cqros.cli.verify_factors.ParquetStore"),
        patch("cqros.cli.verify_factors.StorageLayout"),
    ):
        _run(main(["--debug"]))
    assert logging.getLogger("cqros").level == logging.DEBUG
