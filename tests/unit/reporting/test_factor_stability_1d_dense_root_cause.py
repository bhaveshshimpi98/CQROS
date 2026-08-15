"""Unit tests for the 1d dense-factor root-cause diagnostic reporter."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from cqros.core.constants import EXCHANGE_BINANCE, MARKET_USDT_PERPETUAL
from cqros.reporting.factor_stability_1d_dense_root_cause import (
    ALIGNMENT_CSV_NAME,
    CROSS_TIMEFRAME_CSV_NAME,
    DATA_LINEAGE_CSV_NAME,
    DENSE_FACTORS,
    FACTORS_CSV_NAME,
    FOLDS_CSV_NAME,
    GLOBAL_CSV_NAME,
    IMPLEMENTATION_AUDIT_CSV_NAME,
    QUANTILE_CSV_NAME,
    SUMMARY_TXT_NAME,
    SYMBOL_CONTRIBUTORS_CSV_NAME,
    TARGET_TIMEFRAME,
    TIMESTAMPS_CSV_NAME,
    VERDICT_GENUINE_FACTOR_WEAKNESS,
    VERDICT_MULTI_CAUSE,
    FactorStability1dDenseRootCauseReporter,
    classify_verdict,
    companion_requirement,
    forbidden_import_violations,
    hash_watched_production_artifacts,
    reconstruct_obv,
    reconstruct_oi_level,
    reconstruct_pvt,
    verify_future_return_1_semantics,
)

_MANAGER = "default"
_YEAR = 2026
_EXCHANGE = EXCHANGE_BINANCE
_MARKET = MARKET_USDT_PERPETUAL
_BASE_TS = 1_751_155_200_000  # 2026-06-29 UTC


def test_reconstruct_pvt_matches_definition() -> None:
    frame = pl.DataFrame(
        {
            "close": [100.0, 110.0, 99.0],
            "volume": [10.0, 20.0, 30.0],
        }
    )
    series = reconstruct_pvt(frame)
    assert series[0] is None
    expected_1 = ((110.0 / 100.0) - 1.0) * 20.0
    expected_2 = expected_1 + (((99.0 / 110.0) - 1.0) * 30.0)
    assert abs(float(series[1]) - expected_1) < 1e-12
    assert abs(float(series[2]) - expected_2) < 1e-12


def test_reconstruct_obv_matches_definition() -> None:
    frame = pl.DataFrame(
        {
            "close": [10.0, 11.0, 11.0, 9.0],
            "volume": [1.0, 2.0, 3.0, 4.0],
        }
    )
    series = reconstruct_obv(frame)
    assert series[0] is None
    assert float(series[1]) == 2.0
    assert float(series[2]) == 2.0
    assert float(series[3]) == -2.0


def test_reconstruct_oi_level_casts_open_interest() -> None:
    frame = pl.DataFrame({"open_interest": [1, 2, None]})
    series = reconstruct_oi_level(frame)
    assert float(series[0]) == 1.0
    assert float(series[1]) == 2.0
    assert series[2] is None


def test_verify_future_return_1_semantics() -> None:
    closes = [100.0, 105.0, 84.0]
    labels = [(105.0 - 100.0) / 100.0, (84.0 - 105.0) / 105.0, None]
    assert verify_future_return_1_semantics(close=closes, future_return_1=labels)
    assert not verify_future_return_1_semantics(
        close=closes,
        future_return_1=[0.1, labels[1], None],
    )


def test_companion_requirement() -> None:
    assert companion_requirement("price_volume_trend") == (False, ())
    assert companion_requirement("on_balance_volume") == (False, ())
    assert companion_requirement("open_interest_level") == (True, ("open_interest",))


def test_classify_verdict_multi_cause_genuine_weakness() -> None:
    verdict, primary, secondary, confidence, next_step = classify_verdict(
        semantic_ok={name: True for name in DENSE_FACTORS},
        alignment_ok=True,
        ic_matches_canonical=True,
        companion_truncates_unnecessarily=True,
        low_statistical_power=True,
        negative_broad=True,
        negative_stable=True,
        symbol_concentrated=False,
        monotonicity={
            "price_volume_trend": "monotonic_negative",
            "on_balance_volume": "monotonic_negative",
            "open_interest_level": "noisy_no_relationship",
        },
        unique_oos_timestamps=17,
        mean_fold_ics={
            "price_volume_trend": -0.11,
            "on_balance_volume": -0.10,
            "open_interest_level": -0.03,
        },
    )
    assert verdict == VERDICT_MULTI_CAUSE
    assert primary == VERDICT_GENUINE_FACTOR_WEAKNESS
    assert "H. LOW_STATISTICAL_POWER" in secondary
    assert "G. COMPANION_ALIGNMENT_PROBLEM" in secondary
    assert confidence == "MEDIUM"
    assert "17" in next_step or "timestamp" in next_step.lower()


def test_forbidden_imports_clean_for_module() -> None:
    source = Path("src/cqros/reporting/factor_stability_1d_dense_root_cause.py").read_text(
        encoding="utf-8"
    )
    assert forbidden_import_violations(source) == ()


def _selection_frame() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for index, name in enumerate(DENSE_FACTORS, start=1):
        rows.append(
            {
                "factor_name": name,
                "factor_version": "1.0.0",
                "timeframe": TARGET_TIMEFRAME,
                "selection_time": 1_700_000_000_000,
                "factor_category": "volume",
                "selected": True,
                "selection_score": float(5 - index),
                "selection_rank": index,
                "selection_reason": "test",
                "selection_ic": 0.02,
                "selected_direction": 1,
                "orientation_policy": "signed_ic_v1",
                "status": "SELECTED",
            }
        )
    return pl.DataFrame(rows)


def _oos_frame() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    symbols = ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT", "FFFUSDT")
    for day in range(17):
        ts = _BASE_TS + day * 86_400_000
        fold_id = (day % 5) + 1
        for symbol_index, symbol in enumerate(symbols):
            # Broad negative rank relationship: higher factor -> lower return.
            factor_level = float(symbol_index + 1 + day)
            target = -0.01 * float(symbol_index) + 0.001 * day
            for name in DENSE_FACTORS:
                rows.append(
                    {
                        "manager": _MANAGER,
                        "engine": "simple",
                        "symbol": symbol,
                        "timeframe": TARGET_TIMEFRAME,
                        "year": _YEAR,
                        "fold_id": fold_id,
                        "observation_time": ts,
                        "factor_name": name,
                        "factor_version": "1.0.0",
                        "selected": True,
                        "partition": "OOS",
                        "future_return_1": target,
                        "factor_value": (
                            None if name != "open_interest_level" and day == 0 else factor_level
                        ),
                        "selection_ic": 0.02,
                        "selected_direction": 1,
                        "orientation_policy": "signed_ic_v1",
                        "status": "PASS",
                    }
                )
    return pl.DataFrame(rows)


def _ohlcv_frame() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    # Full year-like span before companion start.
    start = 1_735_689_600_000
    closes = [100.0]
    for day in range(250):
        ts = start + day * 86_400_000
        if day > 0:
            closes.append(closes[-1] * (1.0 + (0.001 if day % 2 == 0 else -0.0005)))
        rows.append(
            {
                "symbol": "BTCUSDT",
                "timeframe": TARGET_TIMEFRAME,
                "open_time": ts,
                "close_time": ts + 86_399_999,
                "open": closes[-1],
                "high": closes[-1] * 1.01,
                "low": closes[-1] * 0.99,
                "close": closes[-1],
                "volume": 1000.0 + day,
                "quote_volume": 1000.0 + day,
                "trade_count": 10,
            }
        )
    return pl.DataFrame(rows)


def _write_lake(root: Path) -> None:
    selection = _selection_frame()
    oos = _oos_frame()
    ohlcv = _ohlcv_frame()
    path = root / "factor_selection" / _MANAGER / _EXCHANGE / _MARKET / TARGET_TIMEFRAME
    path.mkdir(parents=True, exist_ok=True)
    selection.write_parquet(path / f"{_YEAR}.parquet")
    path = root / "purged_cv_evaluation" / _MANAGER / _EXCHANGE / _MARKET / TARGET_TIMEFRAME
    path.mkdir(parents=True, exist_ok=True)
    oos.write_parquet(path / f"{_YEAR}.parquet")

    # Companion-aligned factor store for BTCUSDT starting at OOS base.
    aligned = ohlcv.filter(pl.col("open_time") >= _BASE_TS).sort("open_time")
    oi_values = list(range(aligned.height))
    aligned = aligned.with_columns(pl.Series("open_interest", [float(v) for v in oi_values]))
    store_rows: list[dict[str, object]] = []
    pvt = reconstruct_pvt(aligned)
    obv = reconstruct_obv(aligned)
    oi = reconstruct_oi_level(aligned)
    for index, open_time in enumerate(aligned["open_time"].to_list()):
        store_rows.extend(
            [
                {
                    "symbol": "BTCUSDT",
                    "timeframe": TARGET_TIMEFRAME,
                    "open_time": open_time,
                    "factor_name": "price_volume_trend",
                    "factor_version": "1.0.0",
                    "factor_category": "volume",
                    "factor_group": "volume",
                    "factor_value": pvt[index],
                    "lookback": 0,
                    "prediction_horizon": 1,
                    "enabled": True,
                    "status": "PASS",
                },
                {
                    "symbol": "BTCUSDT",
                    "timeframe": TARGET_TIMEFRAME,
                    "open_time": open_time,
                    "factor_name": "on_balance_volume",
                    "factor_version": "1.0.0",
                    "factor_category": "volume",
                    "factor_group": "volume",
                    "factor_value": obv[index],
                    "lookback": 0,
                    "prediction_horizon": 1,
                    "enabled": True,
                    "status": "PASS",
                },
                {
                    "symbol": "BTCUSDT",
                    "timeframe": TARGET_TIMEFRAME,
                    "open_time": open_time,
                    "factor_name": "open_interest_level",
                    "factor_version": "1.0.0",
                    "factor_category": "open_interest",
                    "factor_group": "open_interest",
                    "factor_value": oi[index],
                    "lookback": 0,
                    "prediction_horizon": 1,
                    "enabled": True,
                    "status": "PASS",
                },
            ]
        )
    path = root / "factors" / _MANAGER / _EXCHANGE / _MARKET / "BTCUSDT" / TARGET_TIMEFRAME
    path.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(store_rows).write_parquet(path / f"{_YEAR}.parquet")

    path = root / "processed" / "ohlcv" / _EXCHANGE / _MARKET / "BTCUSDT" / TARGET_TIMEFRAME
    path.mkdir(parents=True, exist_ok=True)
    ohlcv.write_parquet(path / f"{_YEAR}.parquet")

    for dataset, tcol in (
        ("taker_volume", "timestamp"),
        ("open_interest", "timestamp"),
        ("global_long_short_account_ratio", "timestamp"),
    ):
        companion_path = (
            root / "processed" / dataset / _EXCHANGE / _MARKET / "BTCUSDT" / TARGET_TIMEFRAME
        )
        companion_path.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, object]] = []
        for day in range(40):
            row: dict[str, object] = {
                "symbol": "BTCUSDT",
                tcol: _BASE_TS + day * 86_400_000,
            }
            if dataset == "taker_volume":
                row.update({"buy_volume": 1.0, "sell_volume": 1.0, "buy_sell_ratio": 1.0})
            elif dataset == "open_interest":
                row.update({"open_interest": float(day)})
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

    for tier in ("walk_forward", "purged_cv", "walk_forward_evaluation"):
        ledger_path = root / tier / _MANAGER / _EXCHANGE / _MARKET / TARGET_TIMEFRAME
        ledger_path.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"fold_id": [1], "status": ["PASS"]}).write_parquet(
            ledger_path / f"{_YEAR}.parquet"
        )


def test_reporter_writes_bundle_and_preserves_hashes(tmp_path: Path) -> None:
    storage = tmp_path / "data"
    output = tmp_path / "reports" / "factor_stability" / "1d_dense_factor_root_cause"
    _write_lake(storage)
    before = hash_watched_production_artifacts(storage)
    reporter = FactorStability1dDenseRootCauseReporter(
        storage_root=storage,
        output_root=output,
        manager=_MANAGER,
    )
    result = reporter.run(year=_YEAR)
    after = hash_watched_production_artifacts(storage)
    assert before == after
    assert result.production_artifacts_unchanged is True
    assert result.deterministic is True
    assert (output / GLOBAL_CSV_NAME).exists()
    assert (output / FACTORS_CSV_NAME).exists()
    assert (output / FOLDS_CSV_NAME).exists()
    assert (output / TIMESTAMPS_CSV_NAME).exists()
    assert (output / ALIGNMENT_CSV_NAME).exists()
    assert (output / CROSS_TIMEFRAME_CSV_NAME).exists()
    assert (output / SYMBOL_CONTRIBUTORS_CSV_NAME).exists()
    assert (output / QUANTILE_CSV_NAME).exists()
    assert (output / IMPLEMENTATION_AUDIT_CSV_NAME).exists()
    assert (output / DATA_LINEAGE_CSV_NAME).exists()
    assert (output / SUMMARY_TXT_NAME).exists()
    assert (output / "hashes_before.txt").exists()
    assert (output / "hashes_after.txt").exists()
    factors = pl.read_csv(output / FACTORS_CSV_NAME)
    assert set(factors["factor"].to_list()) == set(DENSE_FACTORS)
    assert all(bool(v) for v in factors["semantic_correct"].to_list())


def test_reporter_deterministic(tmp_path: Path) -> None:
    storage = tmp_path / "data"
    output_a = tmp_path / "out_a"
    output_b = tmp_path / "out_b"
    _write_lake(storage)
    result_a = FactorStability1dDenseRootCauseReporter(
        storage_root=storage,
        output_root=output_a,
        manager=_MANAGER,
    ).run(year=_YEAR)
    result_b = FactorStability1dDenseRootCauseReporter(
        storage_root=storage,
        output_root=output_b,
        manager=_MANAGER,
    ).run(year=_YEAR)
    assert (output_a / GLOBAL_CSV_NAME).read_text(encoding="utf-8") == (
        output_b / GLOBAL_CSV_NAME
    ).read_text(encoding="utf-8")
    assert result_a.verdict == result_b.verdict
