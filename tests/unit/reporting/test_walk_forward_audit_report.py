"""Unit tests for CQROS Walk-Forward consolidated audit reporting."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from cqros.reporting.walk_forward_audit_report import (
    DETAIL_COLUMNS,
    WalkForwardAuditReporter,
    aggregate_partition_frame,
    build_global_summary,
    build_timeframe_summary,
    forbidden_import_violations,
    format_discovery_table,
)
from cqros.storage import StorageLayout

_MANAGER = "default"
_ENGINE = "simple"
_TIMEFRAME = "1h"
_YEAR = 2026


def _write_partition(
    root: Path,
    frame: pl.DataFrame,
    *,
    manager: str = _MANAGER,
    timeframe: str = _TIMEFRAME,
    year: int = _YEAR,
) -> Path:
    """Persist a raw Walk-Forward parquet partition for audit tests."""
    path = (
        root
        / "walk_forward"
        / manager
        / "binance"
        / "usdt_perpetual"
        / timeframe
        / f"{year}.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)
    return path


def _canonical_fold_frame(*, rows: int = 4) -> pl.DataFrame:
    """Build a canonical Walk-Forward fold ledger without evaluation columns."""
    base_ts = 1_782_673_200_000  # 2026-06-28T19:00:00Z
    records: list[dict[str, object]] = []
    for index in range(rows):
        ts = base_ts + index * 3_600_000
        records.append(
            {
                "strategy_name": "default_strategy",
                "strategy_version": "v1",
                "timeframe": _TIMEFRAME,
                "fold_id": index + 1,
                "train_start": ts,
                "train_end": ts,
                "test_start": ts,
                "test_end": ts,
                "train_rows": 252,
                "test_rows": 63,
                "selected_factors": 2 + index,
                "model_version": "v1",
                "train_score": 0.01,
                "test_score": 0.02,
                "overfit_gap": 0.5,
                "status": "PASS" if index % 2 == 0 else "FAIL",
            }
        )
    return pl.DataFrame(records)


def _evaluation_style_frame() -> pl.DataFrame:
    """Build an evaluation-input-style frame with explicit audit columns."""
    return pl.DataFrame(
        [
            {
                "strategy_name": "default_strategy",
                "strategy_version": "v1",
                "symbol": "BTCUSDT",
                "timeframe": _TIMEFRAME,
                "fold_id": 1,
                "factor_name": "momentum",
                "factor_version": "1.0.0",
                "selected": True,
                "open_time": 1_700_000_000_000,
                "selection_time": 1_700_000_000_000,
                "future_return_1": 0.01,
                "is_oos": False,
                "status": "PASS",
            },
            {
                "strategy_name": "default_strategy",
                "strategy_version": "v1",
                "symbol": "BTCUSDT",
                "timeframe": _TIMEFRAME,
                "fold_id": 2,
                "factor_name": "momentum",
                "factor_version": "1.0.0",
                "selected": True,
                "open_time": 1_700_003_600_000,
                "selection_time": 1_700_003_600_000,
                "future_return_1": 0.03,
                "is_oos": True,
                "status": "PASS",
            },
            {
                "strategy_name": "default_strategy",
                "strategy_version": "v1",
                "symbol": "BTCUSDT",
                "timeframe": _TIMEFRAME,
                "fold_id": 3,
                "factor_name": "value",
                "factor_version": "2.0.0",
                "selected": False,
                "open_time": 1_700_007_200_000,
                "selection_time": 1_700_007_200_000,
                "future_return_1": None,
                "is_oos": True,
                "status": "FAIL",
            },
            {
                "strategy_name": "default_strategy",
                "strategy_version": "v1",
                "symbol": "ETHUSDT",
                "timeframe": _TIMEFRAME,
                "fold_id": 1,
                "factor_name": "momentum",
                "factor_version": "1.0.0",
                "selected": True,
                "open_time": 1_700_000_000_000,
                "selection_time": 1_700_000_000_000,
                "future_return_1": -0.02,
                "is_oos": True,
                "status": "PASS",
            },
        ]
    )


def test_aggregate_canonical_ledger_marks_missing_future_return() -> None:
    """Canonical ledgers without future_return_1 fail without fabrication."""
    rows = aggregate_partition_frame(
        _canonical_fold_frame(),
        manager=_MANAGER,
        engine=_ENGINE,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["rows"] == 4
    assert row["pass_rows"] == 2
    assert row["fail_rows"] == 2
    assert row["unique_folds"] == 4
    assert row["selected_rows"] is None
    assert row["unique_factors"] is None
    assert row["train_rows"] is None
    assert row["oos_rows"] is None
    assert row["selected_factor_memberships"] == 2 + 3 + 4 + 5
    assert row["status"] == "FAIL"
    assert "future_return_1 missing" in str(row["error"])


def test_aggregate_evaluation_style_metrics_and_symbol_isolation() -> None:
    """Evaluation-style artifacts populate return and selection metrics."""
    rows = aggregate_partition_frame(
        _evaluation_style_frame(),
        manager=_MANAGER,
        engine=_ENGINE,
        timeframe=_TIMEFRAME,
        year=2023,
    )
    by_symbol = {str(row["symbol"]): row for row in rows}
    assert set(by_symbol) == {"BTCUSDT", "ETHUSDT"}

    btc = by_symbol["BTCUSDT"]
    assert btc["rows"] == 3
    assert btc["selected_rows"] == 2
    assert btc["pass_rows"] == 2
    assert btc["fail_rows"] == 1
    assert btc["unique_factors"] == 2
    assert btc["unique_factor_versions"] == 2
    assert btc["unique_folds"] == 3
    assert btc["train_rows"] == 1
    assert btc["oos_rows"] == 2
    assert btc["future_return_1_non_null"] == 2
    assert btc["future_return_1_null"] == 1
    assert btc["future_return_1_mean"] == pytest.approx(0.02)
    assert btc["oos_future_return_mean"] == pytest.approx(0.03)
    assert btc["selected_factor_memberships"] == 2
    assert btc["status"] == "PASS"
    assert btc["error"] == ""

    eth = by_symbol["ETHUSDT"]
    assert eth["rows"] == 1
    assert eth["selected_rows"] == 1
    assert eth["future_return_1_mean"] == pytest.approx(-0.02)
    assert eth["status"] == "PASS"


def test_duplicate_primary_observations_fail() -> None:
    """Duplicate walk-forward primary keys are reported as FAIL."""
    frame = _canonical_fold_frame(rows=2).with_columns(pl.lit(1).alias("fold_id"))
    row = aggregate_partition_frame(
        frame,
        manager=_MANAGER,
        engine=_ENGINE,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )[0]
    assert row["status"] == "FAIL"
    assert "duplicate primary observations" in str(row["error"])


def test_timeframe_and_year_isolation_failures() -> None:
    """Mismatched timeframe or timestamp year mark the audit row FAIL."""
    frame = _canonical_fold_frame(rows=1).with_columns(pl.lit("4h").alias("timeframe"))
    row = aggregate_partition_frame(
        frame,
        manager=_MANAGER,
        engine=_ENGINE,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )[0]
    assert row["status"] == "FAIL"
    assert "timeframe isolation violated" in str(row["error"])

    row = aggregate_partition_frame(
        _evaluation_style_frame(),
        manager=_MANAGER,
        engine=_ENGINE,
        timeframe=_TIMEFRAME,
        year=2099,
    )[0]
    assert row["status"] == "FAIL"
    assert "year isolation violated" in str(row["error"])


def test_summaries_and_discovery_table_are_deterministic(tmp_path: Path) -> None:
    """CSV emission is deterministic and summaries reconcile detail totals."""
    path = _write_partition(tmp_path, _canonical_fold_frame())
    before = path.read_bytes()
    reporter = WalkForwardAuditReporter(
        StorageLayout(tmp_path),
        output_dir=tmp_path / "reports" / "walk_forward",
        engine=_ENGINE,
    )
    detail, parquet_paths, hashes_before = reporter.collect()
    discovery = format_discovery_table(detail)
    assert discovery.splitlines()[0] == (
        "TIMEFRAME | YEAR | SYMBOLS | ROWS | SELECTED | PASS | FAIL"
    )
    assert "1h | 2026 |" in discovery

    first = reporter.emit(detail, parquet_paths, hashes_before)
    second = reporter.emit(detail, parquet_paths, hashes_before)
    assert first.paths.detail.read_text(encoding="utf-8") == second.paths.detail.read_text(
        encoding="utf-8"
    )
    assert path.read_bytes() == before
    assert first.parquet_hashes_before == first.parquet_hashes_after

    timeframe_summary = build_timeframe_summary(detail)
    global_summary = build_global_summary(detail)
    assert detail.columns == list(DETAIL_COLUMNS)
    assert timeframe_summary.get_column("rows").to_list()[0] == 4
    totals = {
        str(metric): value
        for metric, value in global_summary.select(["metric", "value"]).iter_rows()
    }
    assert totals["total_timeframes"] == 1
    assert totals["total_years"] == 1
    assert totals["total_rows"] == 4
    assert totals["total_pass_rows"] == 2
    assert totals["total_fail_rows"] == 2
    assert totals["total_folds"] == 4


def test_forbidden_imports_are_absent_from_audit_module() -> None:
    """Audit module source must not import Alpha/Regime/Predictions/Signals/ML."""
    source = Path("src/cqros/reporting/walk_forward_audit_report.py").read_text(encoding="utf-8")
    assert forbidden_import_violations(source) == ()
    assert forbidden_import_violations("from cqros.alpha import X") == ("cqros.alpha",)
