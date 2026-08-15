"""Unit tests for the 1d dense-factor root-cause CLI."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from cqros.cli.report_factor_stability_1d_dense_root_cause import (
    build_options,
    build_parser,
    format_summary,
    main,
    run_report,
)
from cqros.core.constants import EXCHANGE_BINANCE, MARKET_USDT_PERPETUAL
from cqros.reporting.factor_stability_1d_dense_root_cause import (
    DENSE_FACTORS,
    TARGET_TIMEFRAME,
    reconstruct_obv,
    reconstruct_oi_level,
    reconstruct_pvt,
)

_MANAGER = "default"
_YEAR = 2026
_EXCHANGE = EXCHANGE_BINANCE
_MARKET = MARKET_USDT_PERPETUAL
_BASE_TS = 1_751_155_200_000


def _write_minimal_lake(root: Path) -> None:
    selection_rows: list[dict[str, object]] = []
    for index, name in enumerate(DENSE_FACTORS, start=1):
        selection_rows.append(
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
    path = root / "factor_selection" / _MANAGER / _EXCHANGE / _MARKET / TARGET_TIMEFRAME
    path.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(selection_rows).write_parquet(path / f"{_YEAR}.parquet")

    oos_rows: list[dict[str, object]] = []
    symbols = ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT", "FFFUSDT")
    for day in range(17):
        ts = _BASE_TS + day * 86_400_000
        for symbol_index, symbol in enumerate(symbols):
            for name in DENSE_FACTORS:
                oos_rows.append(
                    {
                        "manager": _MANAGER,
                        "engine": "simple",
                        "symbol": symbol,
                        "timeframe": TARGET_TIMEFRAME,
                        "year": _YEAR,
                        "fold_id": (day % 5) + 1,
                        "observation_time": ts,
                        "factor_name": name,
                        "factor_version": "1.0.0",
                        "selected": True,
                        "partition": "OOS",
                        "future_return_1": -0.01 * float(symbol_index),
                        "factor_value": float(symbol_index + 1),
                        "selection_ic": 0.02,
                        "selected_direction": 1,
                        "orientation_policy": "signed_ic_v1",
                        "status": "PASS",
                    }
                )
    path = root / "purged_cv_evaluation" / _MANAGER / _EXCHANGE / _MARKET / TARGET_TIMEFRAME
    path.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(oos_rows).write_parquet(path / f"{_YEAR}.parquet")

    ohlcv_rows: list[dict[str, object]] = []
    start = 1_735_689_600_000
    close = 100.0
    for day in range(250):
        ts = start + day * 86_400_000
        close *= 1.001
        ohlcv_rows.append(
            {
                "symbol": "BTCUSDT",
                "timeframe": TARGET_TIMEFRAME,
                "open_time": ts,
                "close_time": ts + 86_399_999,
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 100.0 + day,
                "quote_volume": 100.0 + day,
                "trade_count": 10,
            }
        )
    ohlcv = pl.DataFrame(ohlcv_rows)
    path = root / "processed" / "ohlcv" / _EXCHANGE / _MARKET / "BTCUSDT" / TARGET_TIMEFRAME
    path.mkdir(parents=True, exist_ok=True)
    ohlcv.write_parquet(path / f"{_YEAR}.parquet")

    aligned = ohlcv.filter(pl.col("open_time") >= _BASE_TS).sort("open_time")
    aligned = aligned.with_columns(
        pl.Series("open_interest", [float(i) for i in range(aligned.height)])
    )
    store_rows: list[dict[str, object]] = []
    pvt = reconstruct_pvt(aligned)
    obv = reconstruct_obv(aligned)
    oi = reconstruct_oi_level(aligned)
    for index, open_time in enumerate(aligned["open_time"].to_list()):
        for name, value in (
            ("price_volume_trend", pvt[index]),
            ("on_balance_volume", obv[index]),
            ("open_interest_level", oi[index]),
        ):
            store_rows.append(
                {
                    "symbol": "BTCUSDT",
                    "timeframe": TARGET_TIMEFRAME,
                    "open_time": open_time,
                    "factor_name": name,
                    "factor_version": "1.0.0",
                    "factor_category": "volume",
                    "factor_group": "volume",
                    "factor_value": value,
                    "lookback": 0,
                    "prediction_horizon": 1,
                    "enabled": True,
                    "status": "PASS",
                }
            )
    path = root / "factors" / _MANAGER / _EXCHANGE / _MARKET / "BTCUSDT" / TARGET_TIMEFRAME
    path.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(store_rows).write_parquet(path / f"{_YEAR}.parquet")

    oi_path = (
        root / "processed" / "open_interest" / _EXCHANGE / _MARKET / "BTCUSDT" / TARGET_TIMEFRAME
    )
    oi_path.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 40,
            "timestamp": [_BASE_TS + day * 86_400_000 for day in range(40)],
            "open_interest": [float(day) for day in range(40)],
        }
    ).write_parquet(oi_path / f"{_YEAR}.parquet")

    for tier in ("walk_forward", "purged_cv", "walk_forward_evaluation"):
        ledger = root / tier / _MANAGER / _EXCHANGE / _MARKET / TARGET_TIMEFRAME
        ledger.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"fold_id": [1], "status": ["PASS"]}).write_parquet(
            ledger / f"{_YEAR}.parquet"
        )


def test_build_parser_and_options(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--manager",
            _MANAGER,
            "--year",
            str(_YEAR),
            "--storage-root",
            str(tmp_path / "data"),
            "--output",
            str(tmp_path / "out"),
        ]
    )
    options = build_options(args)
    assert options.manager == _MANAGER
    assert options.year == _YEAR


def test_main_success(tmp_path: Path) -> None:
    storage = tmp_path / "data"
    output = tmp_path / "out"
    _write_minimal_lake(storage)
    code = main(
        [
            "--manager",
            _MANAGER,
            "--year",
            str(_YEAR),
            "--storage-root",
            str(storage),
            "--output",
            str(output),
        ]
    )
    assert code == 0
    assert (output / "summary.txt").exists()


def test_run_report_and_format_summary(tmp_path: Path) -> None:
    storage = tmp_path / "data"
    output = tmp_path / "out"
    _write_minimal_lake(storage)
    parser = build_parser()
    args = parser.parse_args(
        [
            "--manager",
            _MANAGER,
            "--year",
            str(_YEAR),
            "--storage-root",
            str(storage),
            "--output",
            str(output),
        ]
    )
    result = run_report(build_options(args))
    text = format_summary(result)
    assert "VERDICT:" in text
    assert "PRODUCTION_ARTIFACTS_UNCHANGED: True" in text
