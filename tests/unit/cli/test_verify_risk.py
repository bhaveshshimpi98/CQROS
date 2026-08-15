"""Unit tests for CQROS risk-decision dataset verification CLI."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

import cqros.cli.verify_risk as verify_risk_module
from cqros.cli.verify_risk import (
    DiscoveredWorkItem,
    VerifyRiskOptions,
    VerifyRiskSummary,
    build_options,
    build_parser,
    discover_work,
    format_partition_failure,
    format_summary,
    main,
    run_verification,
)
from cqros.config.models import ResearchConfig
from cqros.core.constants import DEFAULT_STORAGE_ROOT, STORAGE_DIR_RISKS
from cqros.core.exceptions import ValidationError
from cqros.risk.verification import RiskVerifier, VerificationReport
from cqros.risk.verification.exceptions import ERROR_SCHEMA_MISMATCH
from cqros.storage import ParquetStore, RiskRepository, StorageLayout

_POLICY = "fixed_risk"
_MODEL = "alpha-lgbm"
_VERSION = "1.0.0"


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def _options(
    *,
    storage_root: Path,
    policy: str | None = None,
    model: str | None = None,
    version: str | None = None,
    symbols: tuple[str, ...] | None = None,
    timeframes: tuple[str, ...] | None = None,
    years: tuple[int, ...] | None = None,
    workers: int = ResearchConfig().worker_count,
    verbose: bool = False,
    debug: bool = False,
) -> VerifyRiskOptions:
    """Build options for tests against a temporary storage root."""
    return VerifyRiskOptions(
        storage_root=storage_root,
        policy=policy,
        model=model,
        version=version,
        symbols=symbols,
        timeframes=timeframes,
        years=years,
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


def _failed_report(
    *,
    invalid_numeric_rows: int = 1,
    warnings: tuple[str, ...] = ("Invalid decision values detected.",),
) -> VerificationReport:
    """Return a failing verification report with a positive counter."""
    return VerificationReport(
        rows_checked=10,
        duplicate_timestamp_rows=0,
        null_rows=0,
        nan_rows=0,
        invalid_timestamp_rows=0,
        invalid_numeric_rows=invalid_numeric_rows,
        warnings=warnings,
        passed=False,
    )


def test_package_exports() -> None:
    """Public CLI symbols are exported through module ``__all__``."""
    expected = {
        "DiscoveredWorkItem",
        "RiskTaskResult",
        "VerifyRiskOptions",
        "VerifyRiskSummary",
        "build_options",
        "build_parser",
        "discover_work",
        "format_partition_failure",
        "format_summary",
        "main",
        "run_verification",
    }
    assert expected.issubset(set(verify_risk_module.__all__))
    assert verify_risk_module.build_parser is build_parser
    assert verify_risk_module.main is main


def test_build_parser_defaults() -> None:
    """Omitted optional flags keep discovery defaults."""
    args = build_parser().parse_args([])
    assert args.policy is None
    assert args.model is None
    assert args.version is None
    assert args.symbols is None
    assert args.timeframes is None
    assert args.years is None
    assert args.workers == ResearchConfig().worker_count
    assert args.verbose is False
    assert args.debug is False


def test_build_parser_accepts_all_flags() -> None:
    """Parser accepts every documented risk-verification flag."""
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
            "--timeframes",
            "1h",
            "--years",
            "2024",
            "--workers",
            "3",
            "--verbose",
            "--debug",
        ]
    )
    assert args.policy == _POLICY
    assert args.model == _MODEL
    assert args.version == _VERSION
    assert args.symbols == ["BTCUSDT"]
    assert args.timeframes == ["1h"]
    assert args.years == ["2024"]
    assert args.workers == 3
    assert args.verbose is True
    assert args.debug is True


def test_build_options_defaults() -> None:
    """Omitted filters map to discovery-all options."""
    options = build_options(build_parser().parse_args([]))
    assert options.storage_root == Path(DEFAULT_STORAGE_ROOT)
    assert options.policy is None
    assert options.model is None
    assert options.version is None
    assert options.symbols is None
    assert options.timeframes is None
    assert options.years is None
    assert options.workers == ResearchConfig().worker_count


def test_build_options_maps_filters() -> None:
    """Explicit CLI flags map onto VerifyRiskOptions."""
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
                "4h",
                "--years",
                "2025",
                "--workers",
                "2",
                "--debug",
            ]
        )
    )
    assert options.policy == "kelly"
    assert options.model == "beta"
    assert options.version == "2.0.0"
    assert options.symbols == ("ETHUSDT",)
    assert options.timeframes == ("4h",)
    assert options.years == (2025,)
    assert options.workers == 2
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
    args = build_parser().parse_args(["--years", "nope"])
    with pytest.raises(ValidationError, match="invalid year"):
        build_options(args)


def test_build_options_rejects_empty_model() -> None:
    """Blank --model fails validation."""
    args = build_parser().parse_args(["--model", "  "])
    with pytest.raises(ValidationError, match="model must be a non-empty string"):
        build_options(args)


def test_build_options_rejects_empty_policy() -> None:
    """Blank --policy fails validation."""
    args = build_parser().parse_args(["--policy", "  "])
    with pytest.raises(ValidationError, match="policy must be a non-empty string"):
        build_options(args)


def test_discover_work_finds_partitions(tmp_path: Path) -> None:
    """Discovery walks risk partitions without hardcoding symbols."""
    _touch_risk(tmp_path, policy=_POLICY, symbol="BTCUSDT", timeframe="1h", year=2024)
    _touch_risk(tmp_path, policy=_POLICY, symbol="ETHUSDT", timeframe="1h", year=2023)
    _touch_risk(tmp_path, policy=_POLICY, symbol="ETHUSDT", timeframe="1h", year=2024)

    repository = RiskRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(repository, _options(storage_root=tmp_path))

    assert len(work) == 2
    assert work[0].policy == _POLICY
    assert work[0].symbol == "BTCUSDT"
    assert work[0].years == (2024,)
    assert work[1].symbol == "ETHUSDT"
    assert work[1].years == (2023, 2024)


def test_discover_work_filters(tmp_path: Path) -> None:
    """Policy, symbol, timeframe, and year filters apply to discovery."""
    _touch_risk(tmp_path, policy=_POLICY, symbol="BTCUSDT", timeframe="1h", year=2024)
    _touch_risk(tmp_path, policy=_POLICY, symbol="BTCUSDT", timeframe="4h", year=2024)
    _touch_risk(tmp_path, policy=_POLICY, symbol="ETHUSDT", timeframe="1h", year=2024)
    _touch_risk(tmp_path, policy="kelly", symbol="BTCUSDT", timeframe="1h", year=2024)
    _touch_risk(tmp_path, policy=_POLICY, symbol="BTCUSDT", timeframe="1h", year=2025)

    repository = RiskRepository(StorageLayout(tmp_path), ParquetStore())
    work = discover_work(
        repository,
        _options(
            storage_root=tmp_path,
            policy=_POLICY,
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


def test_run_verification_empty_work_passes(tmp_path: Path) -> None:
    """Empty discovery produces a PASS summary without invoking the verifier."""
    repository = MagicMock(spec=RiskRepository)
    verifier = MagicMock(spec=RiskVerifier)
    summary = _run(
        run_verification(
            repository=repository,
            verifier=verifier,
            options=_options(storage_root=tmp_path),
            work=(),
        )
    )
    assert summary.repository_passed is True
    assert summary.successful_tasks == 0
    assert summary.failed_tasks == 0
    verifier.verify.assert_not_called()


def test_run_verification_aggregates_pass(tmp_path: Path) -> None:
    """Successful reports with zero counters produce PASS status."""
    work = (
        DiscoveredWorkItem(
            policy=_POLICY,
            symbol="BTCUSDT",
            timeframe="1h",
            years=(2024,),
        ),
    )
    repository = MagicMock(spec=RiskRepository)
    repository.load.return_value = pl.DataFrame(
        {
            "model_name": [_MODEL],
            "model_version": [_VERSION],
            "open_time": [1],
        }
    )
    verifier = MagicMock(spec=RiskVerifier)
    verifier.verify.return_value = _passed_report(rows=7)

    summary = _run(
        run_verification(
            repository=repository,
            verifier=verifier,
            options=_options(storage_root=tmp_path, workers=1),
            work=work,
        )
    )

    assert summary.repository_passed is True
    assert summary.successful_tasks == 1
    assert summary.failed_tasks == 0
    assert summary.rows_checked == 7
    assert summary.datasets_verified == 1
    repository.load.assert_called_once()
    assert repository.load.call_args.kwargs["policy"] == _POLICY


def test_run_verification_fail_on_invalid_decision(tmp_path: Path) -> None:
    """Invalid decision warnings attribute to the decision counter."""
    work = (
        DiscoveredWorkItem(
            policy=_POLICY,
            symbol="BTCUSDT",
            timeframe="1h",
            years=(2024,),
        ),
    )
    repository = MagicMock(spec=RiskRepository)
    repository.load.return_value = pl.DataFrame({"open_time": [1]})
    verifier = MagicMock(spec=RiskVerifier)
    verifier.verify.return_value = _failed_report(
        invalid_numeric_rows=2,
        warnings=("Invalid decision values detected.",),
    )

    summary = _run(
        run_verification(
            repository=repository,
            verifier=verifier,
            options=_options(storage_root=tmp_path, workers=1),
            work=work,
        )
    )

    assert summary.repository_passed is False
    assert summary.successful_tasks == 1
    assert summary.invalid_decision_rows == 2
    assert summary.invalid_target_weight_rows == 0
    assert summary.invalid_approved_weight_rows == 0
    assert summary.warnings == 1


def test_run_verification_fail_on_invalid_weights(tmp_path: Path) -> None:
    """Invalid weight warnings attribute to the corresponding counters."""
    work = (
        DiscoveredWorkItem(
            policy=_POLICY,
            symbol="BTCUSDT",
            timeframe="1h",
            years=(2024,),
        ),
    )
    repository = MagicMock(spec=RiskRepository)
    repository.load.return_value = pl.DataFrame({"open_time": [1]})
    verifier = MagicMock(spec=RiskVerifier)
    verifier.verify.return_value = _failed_report(
        invalid_numeric_rows=3,
        warnings=(
            "Invalid target weight values detected.",
            "Invalid approved weight values detected.",
        ),
    )

    summary = _run(
        run_verification(
            repository=repository,
            verifier=verifier,
            options=_options(storage_root=tmp_path, workers=1),
            work=work,
        )
    )

    assert summary.repository_passed is False
    assert summary.invalid_decision_rows == 0
    assert summary.invalid_target_weight_rows == 3
    assert summary.invalid_approved_weight_rows == 3
    assert summary.warnings == 2


def test_run_verification_fail_on_task_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Raised verifier exceptions produce FAIL and structured diagnostics."""
    work = (
        DiscoveredWorkItem(
            policy=_POLICY,
            symbol="BTCUSDT",
            timeframe="1h",
            years=(2024,),
        ),
    )
    repository = MagicMock(spec=RiskRepository)
    repository.load.return_value = pl.DataFrame({"open_time": [1]})
    verifier = MagicMock(spec=RiskVerifier)
    verifier.verify.side_effect = ValidationError(
        "merged risk schema dtype mismatch",
        error_code=ERROR_SCHEMA_MISMATCH,
        details={},
    )

    summary = _run(
        run_verification(
            repository=repository,
            verifier=verifier,
            options=_options(storage_root=tmp_path, workers=1),
            work=work,
        )
    )

    captured = capsys.readouterr().out
    assert summary.repository_passed is False
    assert summary.failed_tasks == 1
    assert "FAILED" in captured
    assert "Dataset: Risks" in captured
    assert "Verifier: RiskVerifier" in captured
    assert f"Code: {ERROR_SCHEMA_MISMATCH}" in captured


def test_run_verification_failure_isolation(tmp_path: Path) -> None:
    """A failed year does not prevent later years from verifying."""
    work = (
        DiscoveredWorkItem(
            policy=_POLICY,
            symbol="BTCUSDT",
            timeframe="1h",
            years=(2024, 2025),
        ),
    )
    repository = MagicMock(spec=RiskRepository)
    repository.load.return_value = pl.DataFrame({"open_time": [1]})
    verifier = MagicMock(spec=RiskVerifier)
    verifier.verify.side_effect = [
        RuntimeError("boom"),
        _passed_report(rows=3),
    ]

    summary = _run(
        run_verification(
            repository=repository,
            verifier=verifier,
            options=_options(storage_root=tmp_path, workers=1),
            work=work,
        )
    )

    assert summary.failed_tasks == 1
    assert summary.successful_tasks == 1
    assert summary.rows_checked == 3


def test_run_verification_identity_filters(tmp_path: Path) -> None:
    """--model / --version restrict rows before verifier runs."""
    work = (
        DiscoveredWorkItem(
            policy=_POLICY,
            symbol="BTCUSDT",
            timeframe="1h",
            years=(2024,),
        ),
    )
    repository = MagicMock(spec=RiskRepository)
    repository.load.return_value = pl.DataFrame(
        {
            "model_name": [_MODEL, "other"],
            "model_version": [_VERSION, "9.9.9"],
            "open_time": [1, 2],
        }
    )
    verifier = MagicMock(spec=RiskVerifier)
    verifier.verify.return_value = _passed_report(rows=1)

    summary = _run(
        run_verification(
            repository=repository,
            verifier=verifier,
            options=_options(
                storage_root=tmp_path,
                workers=1,
                model=_MODEL,
                version=_VERSION,
            ),
            work=work,
        )
    )

    assert summary.successful_tasks == 1
    verified_frame = verifier.verify.call_args.args[0]
    assert verified_frame.height == 1
    assert verified_frame["model_name"].to_list() == [_MODEL]


def test_run_verification_model_filter_skips_non_matching(tmp_path: Path) -> None:
    """Partitions with no matching identity rows are excluded."""
    work = (
        DiscoveredWorkItem(
            policy=_POLICY,
            symbol="BTCUSDT",
            timeframe="1h",
            years=(2024,),
        ),
    )
    repository = MagicMock(spec=RiskRepository)
    repository.load.return_value = pl.DataFrame(
        {
            "model_name": ["other"],
            "model_version": ["9.9.9"],
            "open_time": [1],
        }
    )
    verifier = MagicMock(spec=RiskVerifier)

    summary = _run(
        run_verification(
            repository=repository,
            verifier=verifier,
            options=_options(
                storage_root=tmp_path,
                workers=1,
                model=_MODEL,
                version=_VERSION,
            ),
            work=work,
        )
    )

    assert summary.successful_tasks == 0
    assert summary.failed_tasks == 0
    assert summary.repository_passed is True
    verifier.verify.assert_not_called()


def test_run_verification_respects_worker_count(tmp_path: Path) -> None:
    """Worker pool size is honored for concurrent symbols."""
    work = (
        DiscoveredWorkItem(
            policy=_POLICY,
            symbol="BTCUSDT",
            timeframe="1h",
            years=(2024,),
        ),
        DiscoveredWorkItem(
            policy=_POLICY,
            symbol="ETHUSDT",
            timeframe="1h",
            years=(2024,),
        ),
    )
    repository = MagicMock(spec=RiskRepository)
    repository.load.return_value = pl.DataFrame({"open_time": [1]})
    verifier = MagicMock(spec=RiskVerifier)
    verifier.verify.return_value = _passed_report()

    with patch(
        "cqros.cli.verify_risk.asyncio.create_task",
        wraps=asyncio.create_task,
    ) as create:
        summary = _run(
            run_verification(
                repository=repository,
                verifier=verifier,
                options=_options(storage_root=tmp_path, workers=2),
                work=work,
            )
        )

    assert summary.successful_tasks == 2
    worker_names = [
        call.kwargs.get("name")
        for call in create.call_args_list
        if call.kwargs.get("name", "").startswith("verify-risk-worker-")
    ]
    assert len(worker_names) == 2


def test_format_summary_pass_and_fail() -> None:
    """Summary rendering mirrors the Portfolio CLI PASS/FAIL style."""
    passed = format_summary(
        VerifyRiskSummary(
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
            invalid_decision_rows=0,
            invalid_target_weight_rows=0,
            invalid_approved_weight_rows=0,
            warnings=0,
            duration_seconds=0.5,
            repository_passed=True,
        )
    )
    failed = format_summary(
        VerifyRiskSummary(
            symbols_verified=1,
            datasets_verified=1,
            timeframes_verified=1,
            successful_tasks=0,
            failed_tasks=1,
            rows_checked=0,
            duplicate_timestamps=0,
            null_rows=0,
            nan_rows=0,
            invalid_timestamps=0,
            invalid_decision_rows=1,
            invalid_target_weight_rows=2,
            invalid_approved_weight_rows=3,
            warnings=1,
            duration_seconds=0.5,
            repository_passed=False,
        )
    )
    assert "CQROS Verification Summary" in passed
    assert "PASS" in passed.split("Repository status:")[-1]
    assert "FAIL" in failed.split("Repository status:")[-1]
    assert "Invalid decision rows: 1" in failed
    assert "Invalid target weight rows: 2" in failed
    assert "Invalid approved weight rows: 3" in failed


def test_format_partition_failure() -> None:
    """Structured failure reports include code and omit tracebacks."""
    text = format_partition_failure(
        dataset="Risks",
        symbol="BTCUSDT",
        timeframe="1h",
        partition="2025.parquet",
        verifier="RiskVerifier",
        exception_type="RiskValidationError",
        message="merged risk schema dtype mismatch",
        code=ERROR_SCHEMA_MISMATCH,
    )
    assert text.startswith("FAILED\n")
    assert "Dataset: Risks" in text
    assert "Verifier: RiskVerifier" in text
    assert f"Code: {ERROR_SCHEMA_MISMATCH}" in text
    assert "Traceback" not in text


def test_main_exit_code_pass(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """main returns 0 when repository status is PASS."""
    with (
        patch("cqros.cli.verify_risk.DEFAULT_STORAGE_ROOT", str(tmp_path)),
        patch("cqros.cli.verify_risk.discover_work", return_value=()),
    ):
        code = _run(main(["--workers", "1"]))

    captured = capsys.readouterr()
    assert code == 0
    assert "PASS" in captured.out


def test_main_exit_code_fail(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """main returns 1 when repository status is FAIL."""
    work = (
        DiscoveredWorkItem(
            policy=_POLICY,
            symbol="BTCUSDT",
            timeframe="1h",
            years=(2024,),
        ),
    )

    with (
        patch("cqros.cli.verify_risk.DEFAULT_STORAGE_ROOT", str(tmp_path)),
        patch("cqros.cli.verify_risk.discover_work", return_value=work),
        patch("cqros.cli.verify_risk.RiskRepository") as repo_cls,
        patch("cqros.cli.verify_risk.RiskVerifier") as verifier_cls,
    ):
        repository = MagicMock(spec=RiskRepository)
        repository.load.return_value = pl.DataFrame({"open_time": [1]})
        repo_cls.return_value = repository
        verifier = MagicMock(spec=RiskVerifier)
        verifier.verify.return_value = _failed_report()
        verifier_cls.return_value = verifier
        code = _run(main(["--workers", "1"]))

    captured = capsys.readouterr()
    assert code == 1
    assert "FAIL" in captured.out


def test_main_validation_error_exit_code() -> None:
    """Fatal CLI validation errors return exit code 1."""
    code = _run(main(["--workers", "0"]))
    assert code == 1


def test_configure_logging_verbose(tmp_path: Path) -> None:
    """--verbose enables INFO logging for the cqros logger."""
    with (
        patch("cqros.cli.verify_risk.DEFAULT_STORAGE_ROOT", str(tmp_path)),
        patch("cqros.cli.verify_risk.discover_work", return_value=()),
    ):
        _run(main(["--verbose"]))
    assert logging.getLogger("cqros").level == logging.INFO


def test_configure_logging_debug(tmp_path: Path) -> None:
    """--debug enables DEBUG logging for the cqros logger."""
    with (
        patch("cqros.cli.verify_risk.DEFAULT_STORAGE_ROOT", str(tmp_path)),
        patch("cqros.cli.verify_risk.discover_work", return_value=()),
    ):
        _run(main(["--debug"]))
    assert logging.getLogger("cqros").level == logging.DEBUG
