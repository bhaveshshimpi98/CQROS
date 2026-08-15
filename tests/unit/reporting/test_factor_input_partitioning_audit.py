"""Unit tests for the factor-input partitioning implementation audit reporter."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import polars as pl

from cqros.reporting.factor_input_partitioning_audit import (
    AUDIT_CSV_NAME,
    HASHES_AFTER_NAME,
    HASHES_BEFORE_NAME,
    SUMMARY_TXT_NAME,
    FactorInputPartitioningAuditReporter,
)


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 6,
            "timeframe": ["1d"] * 6,
            "open_time": [1_000, 2_000, 3_000, 4_000, 5_000, 6_000],
            "open": [1.0] * 6,
            "high": [2.0] * 6,
            "low": [0.5] * 6,
            "close": [1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
            "volume": [10.0] * 6,
            "trade_count": [1, 2, 3, 4, 5, 6],
            "funding_rate": [None, None, None, 0.0001, 0.0001, 0.0001],
            "mark_price": [None, None, None, 1.5, 1.5, 1.5],
            "open_interest": [None, None, None, 100.0, 110.0, 120.0],
            "taker_buy_volume": [None, None, None, 4.0, 5.0, 6.0],
            "taker_sell_volume": [None, None, None, 6.0, 7.0, 8.0],
            "long_short_ratio": [None, None, None, 1.2, 1.1, 1.0],
        }
    )


def test_audit_reporter_writes_outputs_without_mutating_ledgers(tmp_path: Path) -> None:
    storage_root = tmp_path / "storage"
    output_root = tmp_path / "reports"
    storage_root.mkdir()
    frame = _frame()
    with (
        patch(
            "cqros.reporting.factor_input_partitioning_audit.load_factor_input_frame",
            return_value=frame,
        ),
        patch(
            "cqros.reporting.factor_input_partitioning_audit.hash_watched_production_artifacts",
            return_value={"walk_forward/x.parquet": "abc"},
        ),
    ):
        result = FactorInputPartitioningAuditReporter(
            storage_root=storage_root,
            output_root=output_root,
        ).run(year=2026, symbol="BTCUSDT", timeframe="1d")

    assert result.production_artifacts_unchanged is True
    assert result.old_aligned_bars == 3
    assert result.ohlcv_bars == 6
    assert result.bars_recovered_pvt == 3
    assert result.bars_recovered_obv == 3
    assert (output_root / SUMMARY_TXT_NAME).exists()
    assert (output_root / AUDIT_CSV_NAME).exists()
    assert (output_root / HASHES_BEFORE_NAME).exists()
    assert (output_root / HASHES_AFTER_NAME).exists()
    summary = (output_root / SUMMARY_TXT_NAME).read_text(encoding="utf-8")
    assert "POTENTIAL_BEHAVIOR_CHANGE" in summary or "recovered" in summary.lower()
    assert "join_asof(backward)" in summary
