"""Unit tests for the 1d factor-degeneration diagnostic reporter."""

from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl
import pytest

from cqros.core.constants import EXCHANGE_BINANCE, MARKET_USDT_PERPETUAL
from cqros.reporting.exceptions import ReportingValidationError
from cqros.reporting.factor_stability_1d_degeneration import (
    CAUSE_EVALUATION_BOUNDARY,
    CAUSE_INSUFFICIENT_HISTORY,
    CAUSE_ROLLING_WINDOW_TOO_LARGE,
    CAUSE_UNKNOWN,
    CROSS_TIMEFRAME_CSV_NAME,
    DATA_COVERAGE_CSV_NAME,
    FACTORS_CSV_NAME,
    FOLDS_CSV_NAME,
    GLOBAL_CSV_NAME,
    LOOKBACK_ANALYSIS_CSV_NAME,
    SUMMARY_TXT_NAME,
    TARGET_TIMEFRAME,
    VERDICT_MULTI_CAUSE,
    FactorStability1dDegenerationReporter,
    classify_factor_null_cause,
    classify_verdict,
    effective_warmup_bars,
    factor_lookback_catalog,
    forbidden_import_violations,
    hash_watched_production_artifacts,
)

_MANAGER = "default"
_YEAR = 2026
_EXCHANGE = EXCHANGE_BINANCE
_MARKET = MARKET_USDT_PERPETUAL


def _selection_frame() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for index, name in enumerate(
        ("atr_percent", "rsi", "buy_sell_imbalance", "atr_slope"),
        start=1,
    ):
        rows.append(
            {
                "factor_name": name,
                "factor_version": "1.0.0",
                "timeframe": TARGET_TIMEFRAME,
                "selection_time": 1_700_000_000_000,
                "factor_category": "price" if "atr" in name or name == "rsi" else "microstructure",
                "selected": True,
                "selection_score": float(5 - index),
                "selection_rank": index,
                "selection_reason": "test",
                "selection_ic": 0.05,
                "selected_direction": 1,
                "orientation_policy": "signed_ic_v1",
                "status": "SELECTED",
            }
        )
    return pl.DataFrame(rows)


def _oos_frame() -> pl.DataFrame:
    # 17 synthetic daily OOS timestamps starting 2026-06-29.
    base = 1_751_155_200_000  # 2026-06-29 UTC approx
    rows: list[dict[str, object]] = []
    for day in range(17):
        ts = base + day * 86_400_000
        for symbol_index in range(3):
            rows.append(
                {
                    "manager": _MANAGER,
                    "engine": "simple",
                    "symbol": f"S{symbol_index}",
                    "timeframe": TARGET_TIMEFRAME,
                    "year": _YEAR,
                    "fold_id": (day % 5) + 1,
                    "observation_time": ts,
                    "factor_name": "atr_percent",
                    "factor_version": "1.0.0",
                    "selected": True,
                    "partition": "OOS",
                    "future_return_1": 0.01,
                    "factor_value": None,
                    "selection_ic": 0.05,
                    "selected_direction": 1,
                    "orientation_policy": "signed_ic_v1",
                    "status": "PASS",
                }
            )
            rows.append(
                {
                    "manager": _MANAGER,
                    "engine": "simple",
                    "symbol": f"S{symbol_index}",
                    "timeframe": TARGET_TIMEFRAME,
                    "year": _YEAR,
                    "fold_id": (day % 5) + 1,
                    "observation_time": ts,
                    "factor_name": "rsi",
                    "factor_version": "1.0.0",
                    "selected": True,
                    "partition": "OOS",
                    "future_return_1": 0.01,
                    "factor_value": 50.0 if day >= 14 else None,
                    "selection_ic": 0.05,
                    "selected_direction": 1,
                    "orientation_policy": "signed_ic_v1",
                    "status": "PASS",
                }
            )
            rows.append(
                {
                    "manager": _MANAGER,
                    "engine": "simple",
                    "symbol": f"S{symbol_index}",
                    "timeframe": TARGET_TIMEFRAME,
                    "year": _YEAR,
                    "fold_id": (day % 5) + 1,
                    "observation_time": ts,
                    "factor_name": "buy_sell_imbalance",
                    "factor_version": "1.0.0",
                    "selected": True,
                    "partition": "OOS",
                    "future_return_1": 0.01,
                    "factor_value": 0.1,
                    "selection_ic": 0.05,
                    "selected_direction": 1,
                    "orientation_policy": "signed_ic_v1",
                    "status": "PASS",
                }
            )
            rows.append(
                {
                    "manager": _MANAGER,
                    "engine": "simple",
                    "symbol": f"S{symbol_index}",
                    "timeframe": TARGET_TIMEFRAME,
                    "year": _YEAR,
                    "fold_id": (day % 5) + 1,
                    "observation_time": ts,
                    "factor_name": "atr_slope",
                    "factor_version": "1.0.0",
                    "selected": True,
                    "partition": "OOS",
                    "future_return_1": 0.01,
                    "factor_value": None,
                    "selection_ic": 0.0,
                    "selected_direction": 1,
                    "orientation_policy": "signed_ic_v1",
                    "status": "PASS",
                }
            )
    return pl.DataFrame(rows)


def _store_frame() -> pl.DataFrame:
    base = 1_751_155_200_000
    rows: list[dict[str, object]] = []
    for day in range(37):
        ts = base + day * 86_400_000
        for name, lookback, first_valid in (
            ("atr_percent", 20, 19),
            ("rsi", 14, 13),
            ("buy_sell_imbalance", 0, 0),
            ("atr_slope", 20, 999),
        ):
            value: float | None
            if day >= first_valid:
                value = 1.0
            else:
                value = None
            rows.append(
                {
                    "symbol": "BTCUSDT",
                    "timeframe": TARGET_TIMEFRAME,
                    "open_time": ts,
                    "factor_name": name,
                    "factor_version": "1.0.0",
                    "factor_category": "price",
                    "factor_group": "price",
                    "factor_value": value,
                    "lookback": lookback,
                    "prediction_horizon": 1,
                    "enabled": True,
                    "status": "PASS",
                }
            )
    return pl.DataFrame(rows)


def _ohlcv_frame() -> pl.DataFrame:
    base = 1_735_689_600_000  # 2026-01-01 approx
    rows = []
    for day in range(216):
        rows.append(
            {
                "symbol": "BTCUSDT",
                "timeframe": TARGET_TIMEFRAME,
                "open_time": base + day * 86_400_000,
                "close_time": base + day * 86_400_000 + 86_399_999,
                "open": 1.0,
                "high": 1.1,
                "low": 0.9,
                "close": 1.05,
                "volume": 100.0,
                "quote_volume": 100.0,
                "trade_count": 10,
            }
        )
    return pl.DataFrame(rows)


def _write_lake(root: Path) -> None:
    selection = _selection_frame()
    oos = _oos_frame()
    store = _store_frame()
    ohlcv = _ohlcv_frame()
    path = root / "factor_selection" / _MANAGER / _EXCHANGE / _MARKET / TARGET_TIMEFRAME
    path.mkdir(parents=True, exist_ok=True)
    selection.write_parquet(path / f"{_YEAR}.parquet")
    path = root / "purged_cv_evaluation" / _MANAGER / _EXCHANGE / _MARKET / TARGET_TIMEFRAME
    path.mkdir(parents=True, exist_ok=True)
    oos.write_parquet(path / f"{_YEAR}.parquet")
    path = root / "factors" / _MANAGER / _EXCHANGE / _MARKET / "BTCUSDT" / TARGET_TIMEFRAME
    path.mkdir(parents=True, exist_ok=True)
    store.write_parquet(path / f"{_YEAR}.parquet")
    path = root / "processed" / "ohlcv" / _EXCHANGE / _MARKET / "BTCUSDT" / TARGET_TIMEFRAME
    path.mkdir(parents=True, exist_ok=True)
    ohlcv.write_parquet(path / f"{_YEAR}.parquet")
    # companions starting late
    for dataset, tcol in (
        ("taker_volume", "timestamp"),
        ("open_interest", "timestamp"),
        ("global_long_short_account_ratio", "timestamp"),
    ):
        companion_path = (
            root / "processed" / dataset / _EXCHANGE / _MARKET / "BTCUSDT" / TARGET_TIMEFRAME
        )
        companion_path.mkdir(parents=True, exist_ok=True)
        start = 1_751_155_200_000
        rows = []
        for day in range(30):
            row: dict[str, object] = {
                "symbol": "BTCUSDT",
                tcol: start + day * 86_400_000,
            }
            if dataset == "taker_volume":
                row.update({"buy_volume": 1.0, "sell_volume": 1.0, "buy_sell_ratio": 1.0})
            elif dataset == "open_interest":
                row.update({"open_interest": 1.0})
            else:
                row.update(
                    {
                        "long_account": 0.5,
                        "short_account": 0.5,
                        "long_short_ratio": 1.0,
                    }
                )
            rows.append(row)
        pl.DataFrame(rows).write_parquet(companion_path / f"{_YEAR}.parquet")
    funding_path = root / "processed" / "funding" / _EXCHANGE / _MARKET / "BTCUSDT" / "8h"
    funding_path.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "funding_time": [1_735_689_600_000],
            "funding_rate": [0.0001],
            "mark_price": [1.0],
        }
    ).write_parquet(funding_path / f"{_YEAR}.parquet")
    # watched ledgers
    for tier in (
        "walk_forward",
        "purged_cv",
        "walk_forward_evaluation",
    ):
        ledger_path = root / tier / _MANAGER / _EXCHANGE / _MARKET / TARGET_TIMEFRAME
        ledger_path.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"fold_id": [1], "status": ["PASS"]}).write_parquet(
            ledger_path / f"{_YEAR}.parquet"
        )


def test_effective_warmup_bars_matches_catalog() -> None:
    assert effective_warmup_bars("atr_percent") == 20
    assert effective_warmup_bars("atr_slope") == 39
    assert effective_warmup_bars("breakout_strength") == 21
    assert effective_warmup_bars("rsi") == 14
    assert "atr_percent" in factor_lookback_catalog()


def test_classify_factor_null_cause_evaluation_boundary() -> None:
    cause = classify_factor_null_cause(
        oos_null_rate=1.0,
        store_null_rate=0.5,
        required_lookback=20,
        effective_warmup=20,
        post_alignment_bars=37,
        store_first_non_null_ms=1_752_796_800_000,  # after oos
        oos_last_ms=1_752_537_600_000,
        input_null_rate=0.0,
        inputs_present=True,
    )
    assert cause == CAUSE_EVALUATION_BOUNDARY


def test_classify_factor_null_cause_rolling_window_too_large() -> None:
    cause = classify_factor_null_cause(
        oos_null_rate=1.0,
        store_null_rate=1.0,
        required_lookback=20,
        effective_warmup=39,
        post_alignment_bars=37,
        store_first_non_null_ms=None,
        oos_last_ms=1_752_537_600_000,
        input_null_rate=0.0,
        inputs_present=True,
    )
    assert cause == CAUSE_ROLLING_WINDOW_TOO_LARGE


def test_classify_factor_null_cause_insufficient_history() -> None:
    cause = classify_factor_null_cause(
        oos_null_rate=0.92,
        store_null_rate=0.38,
        required_lookback=14,
        effective_warmup=14,
        post_alignment_bars=37,
        store_first_non_null_ms=1_752_364_800_000,
        oos_last_ms=1_752_537_600_000,
        input_null_rate=0.0,
        inputs_present=True,
    )
    assert cause == CAUSE_INSUFFICIENT_HISTORY


def test_classify_factor_null_cause_unknown_for_dense_factor() -> None:
    cause = classify_factor_null_cause(
        oos_null_rate=0.0,
        store_null_rate=0.0,
        required_lookback=0,
        effective_warmup=0,
        post_alignment_bars=37,
        store_first_non_null_ms=1_751_155_200_000,
        oos_last_ms=1_752_537_600_000,
        input_null_rate=0.0,
        inputs_present=True,
    )
    assert cause == CAUSE_UNKNOWN


def test_classify_verdict_multi_cause() -> None:
    verdict, primary, fixability = classify_verdict(
        causes=[CAUSE_EVALUATION_BOUNDARY, CAUSE_ROLLING_WINDOW_TOO_LARGE, CAUSE_UNKNOWN],
        companion_alignment_truncates_history=True,
        fold_local_recompute=False,
        degenerate_100pct_count=9,
        high_missingness_count=14,
    )
    assert verdict == VERDICT_MULTI_CAUSE
    assert "companion alignment" in primary.lower()
    assert "DATA_PROBLEM" in fixability


def test_forbidden_imports() -> None:
    source = Path("src/cqros/reporting/factor_stability_1d_degeneration.py").read_text(
        encoding="utf-8"
    )
    assert forbidden_import_violations(source) == ()


def test_reporter_writes_bundle_and_preserves_hashes(tmp_path: Path) -> None:
    lake = tmp_path / "data"
    _write_lake(lake)
    output = tmp_path / "reports" / "1d_factor_degeneration"
    before = hash_watched_production_artifacts(lake)
    reporter = FactorStability1dDegenerationReporter(
        storage_root=lake,
        output_root=output,
        manager=_MANAGER,
    )
    result = reporter.run(year=_YEAR)
    after = hash_watched_production_artifacts(lake)
    assert before == after
    assert result.production_artifacts_unchanged is True
    assert result.verdict == VERDICT_MULTI_CAUSE
    assert (output / GLOBAL_CSV_NAME).exists()
    assert (output / FACTORS_CSV_NAME).exists()
    assert (output / FOLDS_CSV_NAME).exists()
    assert (output / DATA_COVERAGE_CSV_NAME).exists()
    assert (output / LOOKBACK_ANALYSIS_CSV_NAME).exists()
    assert (output / CROSS_TIMEFRAME_CSV_NAME).exists()
    assert (output / SUMMARY_TXT_NAME).exists()
    factors = pl.read_csv(output / FACTORS_CSV_NAME)
    atr = factors.filter(pl.col("factor") == "atr_percent").row(0, named=True)
    assert atr["likely_cause"] == CAUSE_EVALUATION_BOUNDARY
    slope = factors.filter(pl.col("factor") == "atr_slope").row(0, named=True)
    assert slope["likely_cause"] == CAUSE_ROLLING_WINDOW_TOO_LARGE
    # byte-identical watched parquet
    for rel, digest in before.items():
        assert hashlib.sha256((lake / rel).read_bytes()).hexdigest() == digest


def test_reporter_requires_manager(tmp_path: Path) -> None:
    with pytest.raises(ReportingValidationError):
        FactorStability1dDegenerationReporter(
            storage_root=tmp_path,
            output_root=tmp_path / "out",
            manager="  ",
        )
