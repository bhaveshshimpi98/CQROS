"""Unit tests for the 1d factor-degeneration diagnostic CLI."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from cqros.cli.report_factor_stability_1d_degeneration import (
    build_options,
    build_parser,
    format_summary,
    main,
)
from cqros.core.constants import EXCHANGE_BINANCE, MARKET_USDT_PERPETUAL
from cqros.reporting.factor_stability_1d_degeneration import TARGET_TIMEFRAME

_MANAGER = "default"
_YEAR = 2026
_EXCHANGE = EXCHANGE_BINANCE
_MARKET = MARKET_USDT_PERPETUAL


def _write_minimal_lake(root: Path) -> None:
    selection = pl.DataFrame(
        {
            "factor_name": ["buy_sell_imbalance"],
            "factor_version": ["1.0.0"],
            "timeframe": [TARGET_TIMEFRAME],
            "selection_time": [1],
            "factor_category": ["microstructure"],
            "selected": [True],
            "selection_score": [1.0],
            "selection_rank": [1],
            "selection_reason": ["test"],
            "selection_ic": [0.1],
            "selected_direction": [1],
            "orientation_policy": ["signed_ic_v1"],
            "status": ["SELECTED"],
        }
    )
    path = root / "factor_selection" / _MANAGER / _EXCHANGE / _MARKET / TARGET_TIMEFRAME
    path.mkdir(parents=True, exist_ok=True)
    selection.write_parquet(path / f"{_YEAR}.parquet")

    oos = pl.DataFrame(
        {
            "manager": [_MANAGER],
            "engine": ["simple"],
            "symbol": ["S0"],
            "timeframe": [TARGET_TIMEFRAME],
            "year": [_YEAR],
            "fold_id": [1],
            "observation_time": [1_751_155_200_000],
            "factor_name": ["buy_sell_imbalance"],
            "factor_version": ["1.0.0"],
            "selected": [True],
            "partition": ["OOS"],
            "future_return_1": [0.01],
            "factor_value": [0.2],
            "selection_ic": [0.1],
            "selected_direction": [1],
            "orientation_policy": ["signed_ic_v1"],
            "status": ["PASS"],
        }
    )
    path = root / "purged_cv_evaluation" / _MANAGER / _EXCHANGE / _MARKET / TARGET_TIMEFRAME
    path.mkdir(parents=True, exist_ok=True)
    oos.write_parquet(path / f"{_YEAR}.parquet")

    store = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timeframe": [TARGET_TIMEFRAME],
            "open_time": [1_751_155_200_000],
            "factor_name": ["buy_sell_imbalance"],
            "factor_version": ["1.0.0"],
            "factor_category": ["microstructure"],
            "factor_group": ["microstructure"],
            "factor_value": [0.2],
            "lookback": [0],
            "prediction_horizon": [1],
            "enabled": [True],
            "status": ["PASS"],
        }
    )
    path = root / "factors" / _MANAGER / _EXCHANGE / _MARKET / "BTCUSDT" / TARGET_TIMEFRAME
    path.mkdir(parents=True, exist_ok=True)
    store.write_parquet(path / f"{_YEAR}.parquet")

    ohlcv = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timeframe": [TARGET_TIMEFRAME],
            "open_time": [1_751_155_200_000],
            "close_time": [1_751_241_599_999],
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.05],
            "volume": [10.0],
            "quote_volume": [10.0],
            "trade_count": [1],
        }
    )
    path = root / "processed" / "ohlcv" / _EXCHANGE / _MARKET / "BTCUSDT" / TARGET_TIMEFRAME
    path.mkdir(parents=True, exist_ok=True)
    ohlcv.write_parquet(path / f"{_YEAR}.parquet")

    for tier in (
        "walk_forward",
        "purged_cv",
        "walk_forward_evaluation",
        "factor_selection",
    ):
        # factor_selection already written; ensure other watched tiers exist
        if tier == "factor_selection":
            continue
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
            "default",
            "--year",
            "2026",
            "--storage-root",
            str(tmp_path),
            "--output",
            str(tmp_path / "out"),
        ]
    )
    options = build_options(args)
    assert options.manager == "default"
    assert options.year == 2026


def test_main_success(tmp_path: Path, capsys) -> None:
    lake = tmp_path / "data"
    _write_minimal_lake(lake)
    output = tmp_path / "out"
    code = main(
        [
            "--manager",
            "default",
            "--year",
            "2026",
            "--storage-root",
            str(lake),
            "--output",
            str(output),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "VERDICT" in captured.out
    assert "production_artifacts_unchanged" in captured.out
    assert (output / "summary.txt").exists()
    assert format_summary.__name__ == "format_summary"
