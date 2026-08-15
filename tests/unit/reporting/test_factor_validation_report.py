"""Unit tests for CQROS ``FactorValidationReporter``."""

from __future__ import annotations

import zipfile
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.core.constants import EXCHANGE_BINANCE, MARKET_USDT_PERPETUAL
from cqros.factor_validation import (
    FactorValidationRepository,
    FactorValidationStatus,
)
from cqros.factor_validation.schema import CANONICAL_COLUMN_ORDER, FACTOR_VALIDATION_SCHEMA
from cqros.reporting import SUMMARY_COLUMNS, TOP_FACTOR_COUNT, FactorValidationReporter
from cqros.reporting.exceptions import ReportingValidationError
from cqros.storage import ParquetStore, StorageLayout

_MANAGER = "default"
_TIMEFRAME = "1h"
_YEAR = 2026
_VALIDATION_TIME = 1_718_452_800_000
_VALIDATION_START_TIME = 1_699_913_600_000
_VALIDATION_END_TIME = 1_718_452_800_000
_FACTOR_VERSION = "1.0.0"
_FACTOR_CATEGORY = "price"
_DATASET_VERSION = "dataset-v1"
_LABEL_VERSION = "label-v1"


def _canonical_rows(
    rows: list[dict[str, object]],
) -> pl.DataFrame:
    """Build a canonical factor validation frame from row dictionaries."""
    defaults: dict[str, object] = {
        "factor_version": _FACTOR_VERSION,
        "timeframe": _TIMEFRAME,
        "validation_time": _VALIDATION_TIME,
        "factor_category": _FACTOR_CATEGORY,
        "dataset_version": _DATASET_VERSION,
        "label_version": _LABEL_VERSION,
        "validation_start_time": _VALIDATION_START_TIME,
        "validation_end_time": _VALIDATION_END_TIME,
        "information_coefficient": 0.0,
        "rank_information_coefficient": 0.0,
        "ic_information_ratio": 0.0,
        "ic_std": 0.0,
        "ic_p_value": 1.0,
        "ic_t_stat": 0.0,
        "ic_decay": 0.0,
        "turnover": 0.0,
        "monotonicity_score": 0.0,
        "quantile_spread": 0.0,
        "observations": 10,
        "ic_observations": 10,
        "status": FactorValidationStatus.PASS.value,
    }
    materialized: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        payload = dict(defaults)
        payload.update(row)
        if "factor_name" not in payload:
            payload["factor_name"] = f"factor_{index}"
        materialized.append(payload)
    return pl.DataFrame(materialized, schema=FACTOR_VALIDATION_SCHEMA).select(
        list(CANONICAL_COLUMN_ORDER)
    )


def _save_frame(tmp_path: Path, frame: pl.DataFrame, *, manager: str = _MANAGER) -> None:
    """Persist a factor validation frame through the repository."""
    repository = FactorValidationRepository(StorageLayout(tmp_path), ParquetStore())
    repository.save(
        frame,
        manager=manager,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )


def _reporter(tmp_path: Path, *, manager: str = _MANAGER) -> FactorValidationReporter:
    """Construct a reporter bound to a temporary storage root."""
    repository = FactorValidationRepository(StorageLayout(tmp_path), ParquetStore())
    return FactorValidationReporter(
        repository,
        manager=manager,
        output_dir=tmp_path / "reports" / "factor_validation",
    )


def test_summarize_ranks_and_deduplicates(tmp_path: Path) -> None:
    """Summary keeps latest factor identity and sorts by Rank IC then IC."""
    frame = _canonical_rows(
        [
            {
                "factor_name": "alpha",
                "rank_information_coefficient": 0.10,
                "information_coefficient": 0.05,
                "ic_information_ratio": 1.0,
                "monotonicity_score": 0.2,
                "validation_time": 100,
            },
            {
                "factor_name": "beta",
                "rank_information_coefficient": 0.40,
                "information_coefficient": 0.20,
                "ic_information_ratio": 2.0,
                "monotonicity_score": 0.8,
                "validation_time": 100,
            },
            {
                "factor_name": "alpha",
                "rank_information_coefficient": 0.30,
                "information_coefficient": 0.15,
                "ic_information_ratio": 1.5,
                "monotonicity_score": 0.5,
                "validation_time": 200,
            },
            {
                "factor_name": "gamma",
                "rank_information_coefficient": 0.40,
                "information_coefficient": 0.25,
                "ic_information_ratio": 1.2,
                "monotonicity_score": 0.1,
                "validation_time": 100,
            },
        ]
    )
    _save_frame(tmp_path, frame)
    summary = _reporter(tmp_path).summarize()
    assert summary.columns == list(SUMMARY_COLUMNS)
    assert summary.get_column("factor_name").to_list() == ["gamma", "beta", "alpha"]
    assert summary.get_column("information_coefficient").to_list()[2] == 0.15


def test_top_and_rejected_exports(tmp_path: Path) -> None:
    """CSV exports include top factors and rejected FAIL rows."""
    rows: list[dict[str, object]] = []
    for index in range(25):
        rows.append(
            {
                "factor_name": f"pass_{index:02d}",
                "rank_information_coefficient": float(index),
                "information_coefficient": float(index) / 100.0,
                "ic_information_ratio": float(index) / 10.0,
                "monotonicity_score": float(index) / 50.0,
                "status": FactorValidationStatus.PASS.value,
            }
        )
    rows.append(
        {
            "factor_name": "failed_factor",
            "rank_information_coefficient": -0.1,
            "information_coefficient": -0.2,
            "ic_information_ratio": -1.0,
            "monotonicity_score": -0.5,
            "status": FactorValidationStatus.FAIL.value,
            "observations": 3,
        }
    )
    rows.append(
        {
            "factor_name": "skipped_factor",
            "status": FactorValidationStatus.SKIPPED.value,
            "rank_information_coefficient": None,
            "information_coefficient": None,
            "ic_information_ratio": None,
            "monotonicity_score": None,
        }
    )
    _save_frame(tmp_path, _canonical_rows(rows))
    reporter = _reporter(tmp_path)
    output_dir = reporter.write_csv()

    top = pl.read_csv(output_dir / "top_factors.csv")
    assert top.height == TOP_FACTOR_COUNT
    assert top.get_column("factor_name").to_list()[0] == "pass_24"

    rejected = pl.read_csv(output_dir / "rejected_factors.csv")
    assert rejected.height == 1
    assert rejected.columns == [
        "factor_name",
        "status",
        "observations",
        "IC",
        "Rank IC",
        "ICIR",
        "Monotonicity",
    ]
    assert rejected.get_column("factor_name").to_list() == ["failed_factor"]

    summary_csv = pl.read_csv(output_dir / "validation_summary.csv")
    assert summary_csv.height == 27
    assert summary_csv.columns == list(SUMMARY_COLUMNS)


def test_write_html_and_excel(tmp_path: Path) -> None:
    """HTML and XLSX writers emit expected files and workbook sheets."""
    frame = _canonical_rows(
        [
            {
                "factor_name": "winner",
                "rank_information_coefficient": 0.5,
                "information_coefficient": 0.2,
                "ic_information_ratio": 1.5,
                "monotonicity_score": 0.9,
                "status": FactorValidationStatus.PASS.value,
            },
            {
                "factor_name": "loser",
                "rank_information_coefficient": -0.2,
                "information_coefficient": -0.1,
                "ic_information_ratio": -0.5,
                "monotonicity_score": -0.1,
                "status": FactorValidationStatus.FAIL.value,
            },
        ]
    )
    _save_frame(tmp_path, frame)
    reporter = _reporter(tmp_path)
    html_path = reporter.write_html()
    xlsx_path = reporter.write_excel()

    html_text = html_path.read_text(encoding="utf-8")
    assert "Executive Summary" in html_text
    assert "Top Factors" in html_text
    assert "Rejected Factors" in html_text
    assert "Metric Distributions" in html_text
    assert "Status Breakdown" in html_text
    assert "Dataset Metadata" in html_text
    assert "winner" in html_text
    assert "loser" in html_text
    assert _MANAGER in html_text

    assert xlsx_path.is_file()
    with zipfile.ZipFile(xlsx_path) as archive:
        names = set(archive.namelist())
        assert "xl/workbook.xml" in names
        assert "xl/worksheets/sheet1.xml" in names
        assert "xl/worksheets/sheet2.xml" in names
        assert "xl/worksheets/sheet3.xml" in names
        workbook = archive.read("xl/workbook.xml").decode("utf-8")
        assert "Summary" in workbook
        assert "Top Factors" in workbook
        assert "Rejected Factors" in workbook


def test_write_all_formats(tmp_path: Path) -> None:
    """write() emits CSV, XLSX, and HTML under the manager output directory."""
    _save_frame(
        tmp_path,
        _canonical_rows(
            [
                {
                    "factor_name": "only",
                    "rank_information_coefficient": 0.1,
                    "information_coefficient": 0.05,
                    "ic_information_ratio": 0.8,
                    "monotonicity_score": 0.4,
                }
            ]
        ),
    )
    reporter = _reporter(tmp_path)
    output_dir = reporter.write()
    assert output_dir == tmp_path / "reports" / "factor_validation" / _MANAGER
    assert (output_dir / "validation_summary.csv").is_file()
    assert (output_dir / "top_factors.csv").is_file()
    assert (output_dir / "rejected_factors.csv").is_file()
    assert (output_dir / "validation_summary.xlsx").is_file()
    assert (output_dir / "validation_report.html").is_file()


def test_load_requires_partitions(tmp_path: Path) -> None:
    """load() fails when the manager has no validation ledgers."""
    reporter = _reporter(tmp_path, manager="missing")
    with pytest.raises(ReportingValidationError) as exc_info:
        reporter.load()
    assert exc_info.value.error_code == "REPORT-FACTOR-VALIDATION-002"


def test_blank_manager_rejected(tmp_path: Path) -> None:
    """Reporter construction rejects a blank manager identity."""
    repository = FactorValidationRepository(StorageLayout(tmp_path), ParquetStore())
    with pytest.raises(ReportingValidationError) as exc_info:
        FactorValidationReporter(repository, manager="   ")
    assert exc_info.value.error_code == "REPORT-FACTOR-VALIDATION-001"


def test_summary_tie_breakers(tmp_path: Path) -> None:
    """Equal Rank IC falls through to IC, ICIR, then monotonicity."""
    frame = _canonical_rows(
        [
            {
                "factor_name": "a",
                "rank_information_coefficient": 0.2,
                "information_coefficient": 0.1,
                "ic_information_ratio": 1.0,
                "monotonicity_score": 0.1,
            },
            {
                "factor_name": "b",
                "rank_information_coefficient": 0.2,
                "information_coefficient": 0.2,
                "ic_information_ratio": 0.5,
                "monotonicity_score": 0.9,
            },
            {
                "factor_name": "c",
                "rank_information_coefficient": 0.2,
                "information_coefficient": 0.2,
                "ic_information_ratio": 1.5,
                "monotonicity_score": 0.2,
            },
            {
                "factor_name": "d",
                "rank_information_coefficient": 0.2,
                "information_coefficient": 0.2,
                "ic_information_ratio": 1.5,
                "monotonicity_score": 0.8,
            },
        ]
    )
    _save_frame(tmp_path, frame)
    summary = _reporter(tmp_path).summarize()
    assert summary.get_column("factor_name").to_list() == ["d", "c", "b", "a"]
    expected = summary.select(list(SUMMARY_COLUMNS))
    assert_frame_equal(summary, expected)
