"""Unit tests for CQROS factor validation report CLI."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from pathlib import Path

import polars as pl
import pytest

from cqros.cli.report_factor_validation import (
    ReportFactorValidationOptions,
    build_options,
    build_parser,
    format_summary,
    main,
    run_report,
)
from cqros.core.constants import EXCHANGE_BINANCE, MARKET_USDT_PERPETUAL
from cqros.core.exceptions import ValidationError
from cqros.factor_validation import FactorValidationRepository, FactorValidationStatus
from cqros.factor_validation.schema import CANONICAL_COLUMN_ORDER, FACTOR_VALIDATION_SCHEMA
from cqros.reporting.factor_validation_report import DEFAULT_OUTPUT_ROOT, REPORT_FORMATS
from cqros.storage import ParquetStore, StorageLayout

_MANAGER = "default"
_TIMEFRAME = "1h"
_YEAR = 2026
_VALIDATION_TIME = 1_718_452_800_000
_VALIDATION_START_TIME = 1_699_913_600_000
_VALIDATION_END_TIME = 1_718_452_800_000


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def _canonical_frame() -> pl.DataFrame:
    """Return a canonical passing factor validation frame."""
    return pl.DataFrame(
        {
            "factor_name": ["momentum", "mean_reversion"],
            "factor_version": ["1.0.0", "1.0.0"],
            "timeframe": [_TIMEFRAME, _TIMEFRAME],
            "validation_time": [_VALIDATION_TIME, _VALIDATION_TIME],
            "factor_category": ["price", "price"],
            "dataset_version": ["dataset-v1", "dataset-v1"],
            "label_version": ["label-v1", "label-v1"],
            "validation_start_time": [_VALIDATION_START_TIME, _VALIDATION_START_TIME],
            "validation_end_time": [_VALIDATION_END_TIME, _VALIDATION_END_TIME],
            "information_coefficient": [0.05, -0.02],
            "rank_information_coefficient": [0.12, -0.04],
            "ic_information_ratio": [1.1, -0.5],
            "ic_std": [0.1, 0.1],
            "ic_p_value": [0.01, 0.4],
            "ic_t_stat": [2.0, -0.5],
            "ic_decay": [0.0, 0.0],
            "turnover": [0.2, 0.3],
            "monotonicity_score": [0.7, -0.2],
            "quantile_spread": [0.01, -0.01],
            "observations": [100, 80],
            "ic_observations": [90, 70],
            "status": [
                FactorValidationStatus.PASS.value,
                FactorValidationStatus.FAIL.value,
            ],
        },
        schema=FACTOR_VALIDATION_SCHEMA,
    ).select(list(CANONICAL_COLUMN_ORDER))


def _seed(tmp_path: Path) -> None:
    """Persist a sample validation partition under ``tmp_path``."""
    repository = FactorValidationRepository(StorageLayout(tmp_path), ParquetStore())
    repository.save(
        _canonical_frame(),
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )


def test_build_parser_and_options(tmp_path: Path) -> None:
    """Parser requires manager and defaults formats to all three."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "--manager",
            "default",
            "--output",
            str(tmp_path / "reports"),
            "--storage-root",
            str(tmp_path),
        ]
    )
    options = build_options(args)
    assert options.manager == "default"
    assert options.output == tmp_path / "reports"
    assert options.formats == REPORT_FORMATS
    assert options.storage_root == tmp_path


def test_build_options_rejects_blank_manager() -> None:
    """build_options rejects a blank manager string."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "   "])
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-REPORT-FACTOR-VALIDATION-001"


def test_build_options_rejects_invalid_format(tmp_path: Path) -> None:
    """build_options rejects unsupported report formats."""
    parser = build_parser()
    args = parser.parse_args(
        ["--manager", "default", "--format", "pdf", "--storage-root", str(tmp_path)]
    )
    with pytest.raises(ValidationError) as exc_info:
        build_options(args)
    assert exc_info.value.error_code == "CLI-REPORT-FACTOR-VALIDATION-002"


def test_build_options_format_subset_and_dedupe(tmp_path: Path) -> None:
    """Selected formats are normalized, ordered, and de-duplicated."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "--manager",
            "default",
            "--format",
            "CSV",
            "html",
            "csv",
            "--storage-root",
            str(tmp_path),
        ]
    )
    options = build_options(args)
    assert options.formats == ("csv", "html")


def test_run_report_writes_artifacts(tmp_path: Path) -> None:
    """run_report writes the requested artifacts under output/manager."""
    _seed(tmp_path)
    output_root = tmp_path / "out"
    options = ReportFactorValidationOptions(
        storage_root=tmp_path,
        manager=_MANAGER,
        output=output_root,
        formats=("csv", "html", "xlsx"),
        verbose=False,
        debug=False,
    )
    repository = FactorValidationRepository(StorageLayout(tmp_path), ParquetStore())
    output_dir = run_report(repository=repository, options=options)
    assert output_dir == output_root / _MANAGER
    assert (output_dir / "validation_summary.csv").is_file()
    assert (output_dir / "top_factors.csv").is_file()
    assert (output_dir / "rejected_factors.csv").is_file()
    assert (output_dir / "validation_summary.xlsx").is_file()
    assert (output_dir / "validation_report.html").is_file()

    summary = pl.read_csv(output_dir / "validation_summary.csv")
    assert summary.height == 2
    assert summary.get_column("factor_name").to_list()[0] == "momentum"
    rejected = pl.read_csv(output_dir / "rejected_factors.csv")
    assert rejected.height == 1
    assert rejected.get_column("factor_name").to_list() == ["mean_reversion"]


def test_main_success(tmp_path: Path) -> None:
    """main returns success and prints completion summary."""
    _seed(tmp_path)
    output_root = tmp_path / "reports" / "factor_validation"
    code = _run(
        main(
            [
                "--manager",
                _MANAGER,
                "--output",
                str(output_root),
                "--storage-root",
                str(tmp_path),
                "--format",
                "csv",
            ]
        )
    )
    assert code == 0
    assert (output_root / _MANAGER / "validation_summary.csv").is_file()


def test_main_missing_manager_partitions(tmp_path: Path) -> None:
    """main fails when the selected manager has no ledgers."""
    code = _run(
        main(
            [
                "--manager",
                "missing",
                "--output",
                str(tmp_path / "reports"),
                "--storage-root",
                str(tmp_path),
            ]
        )
    )
    assert code == 1


def test_format_summary_text() -> None:
    """format_summary includes manager, output, and formats."""
    text = format_summary(
        manager="default",
        output_dir=Path("reports/factor_validation/default"),
        formats=REPORT_FORMATS,
    )
    assert "manager: default" in text
    assert "status: COMPLETE" in text
    assert "html, csv, xlsx" in text


def test_default_output_root() -> None:
    """Default output root matches the documented reports path."""
    assert DEFAULT_OUTPUT_ROOT.as_posix() == "reports/factor_validation"
    parser = build_parser()
    args = parser.parse_args(["--manager", "default"])
    options = build_options(args)
    assert options.output == DEFAULT_OUTPUT_ROOT
