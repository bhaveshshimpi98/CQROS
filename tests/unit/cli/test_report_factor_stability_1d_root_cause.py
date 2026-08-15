"""Unit tests for the 1d root-cause investigation CLI."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from cqros.cli.report_factor_stability_1d_root_cause import (
    build_options,
    build_parser,
    format_summary,
    main,
    run_report,
)
from cqros.core.constants import EXCHANGE_BINANCE, MARKET_USDT_PERPETUAL
from cqros.core.exceptions import ValidationError
from cqros.reporting.factor_stability_1d_root_cause import TARGET_TIMEFRAME

_MANAGER = "default"
_YEAR = 2026
_EXCHANGE = EXCHANGE_BINANCE
_MARKET = MARKET_USDT_PERPETUAL


def _selection_frame() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(4):
        direction = 1 if index % 2 == 0 else -1
        rows.append(
            {
                "factor_name": f"factor_{index}",
                "factor_version": "1.0.0",
                "timeframe": TARGET_TIMEFRAME,
                "selection_time": 1_700_000_000_000,
                "factor_category": "momentum",
                "selected": True,
                "selection_score": float(4 - index),
                "selection_rank": index + 1,
                "selection_reason": "test",
                "selection_ic": 0.05 if direction > 0 else -0.05,
                "selected_direction": direction,
                "orientation_policy": "signed_ic_v1",
                "status": "PASS",
            }
        )
    return pl.DataFrame(rows)


def _obs() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for fold_id in (1, 2, 3, 4, 5):
        for factor_index in range(4):
            direction = 1 if factor_index % 2 == 0 else -1
            for index in range(4):
                value: float | None
                if factor_index >= 2:
                    value = None
                elif direction > 0:
                    value = float(index + 1)
                else:
                    value = float(4 - index)
                rows.append(
                    {
                        "manager": _MANAGER,
                        "engine": "simple",
                        "symbol": f"S{index}",
                        "timeframe": TARGET_TIMEFRAME,
                        "year": _YEAR,
                        "fold_id": fold_id,
                        "observation_time": 1_700_000_000_000 + fold_id * 86_400_000 + index,
                        "factor_name": f"factor_{factor_index}",
                        "factor_version": "1.0.0",
                        "selected": True,
                        "partition": "OOS",
                        "future_return_1": -0.01 * float(index + 1),
                        "factor_value": value,
                        "selection_ic": 0.05 if direction > 0 else -0.05,
                        "selected_direction": direction,
                        "orientation_policy": "signed_ic_v1",
                        "prediction": None,
                        "residual": None,
                        "correct": None,
                        "status": "PASS",
                    }
                )
    return pl.DataFrame(rows)


def _ledger() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "strategy_name": ["s"] * 5,
            "strategy_version": ["v1"] * 5,
            "timeframe": [TARGET_TIMEFRAME] * 5,
            "fold_id": [1, 2, 3, 4, 5],
            "train_start": [1] * 5,
            "train_end": [2] * 5,
            "test_start": [3] * 5,
            "test_end": [4] * 5,
            "train_rows": [100] * 5,
            "test_rows": [20] * 5,
            "selected_factors": [4] * 5,
            "model_version": ["v1"] * 5,
            "train_score": [0.1] * 5,
            "test_score": [0.05] * 5,
            "overfit_gap": [0.05] * 5,
            "status": ["PASS"] * 5,
        }
    )


def _write(root: Path, tier: str, timeframe: str, frame: pl.DataFrame) -> None:
    path = root / tier / _MANAGER / _EXCHANGE / _MARKET / timeframe
    path.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path / f"{_YEAR}.parquet")


def _build_lake(root: Path) -> None:
    selection = _selection_frame()
    obs = _obs()
    ledger = _ledger()
    _write(root, "factor_selection", TARGET_TIMEFRAME, selection)
    _write(root, "purged_cv_evaluation", TARGET_TIMEFRAME, obs)
    _write(root, "walk_forward_evaluation", TARGET_TIMEFRAME, obs)
    _write(root, "purged_cv", TARGET_TIMEFRAME, ledger)
    _write(root, "walk_forward", TARGET_TIMEFRAME, ledger)
    peer = selection.with_columns(pl.lit("4h").alias("timeframe"))
    peer_obs = obs.with_columns(pl.lit("4h").alias("timeframe"))
    _write(root, "factor_selection", "4h", peer)
    _write(root, "walk_forward_evaluation", "4h", peer_obs)
    _write(root, "purged_cv_evaluation", "4h", peer_obs.clear())
    _write(root, "walk_forward", "4h", ledger.with_columns(pl.lit("4h").alias("timeframe")))
    _write(root, "purged_cv", "4h", ledger.with_columns(pl.lit("4h").alias("timeframe")))


def test_build_options_and_parser(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--manager",
            _MANAGER,
            "--storage-root",
            str(tmp_path),
            "--output",
            str(tmp_path / "out"),
            "--year",
            str(_YEAR),
        ]
    )
    options = build_options(args)
    assert options.manager == _MANAGER
    assert options.year == _YEAR
    assert options.storage_root == tmp_path


def test_build_options_rejects_blank_manager(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(["--manager", "  ", "--storage-root", str(tmp_path)])
    with pytest.raises(ValidationError):
        build_options(args)


def test_cli_main_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    lake = tmp_path / "data"
    output = tmp_path / "reports"
    _build_lake(lake)
    code = main(
        [
            "--manager",
            _MANAGER,
            "--storage-root",
            str(lake),
            "--output",
            str(output),
            "--year",
            str(_YEAR),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "CQROS 1d ROOT-CAUSE INVESTIGATION" in captured.out
    assert "Primary classification:" in captured.out
    assert (output / "1d_root_cause_global.csv").exists()
    assert (output / "1d_root_cause_summary.txt").exists()


def test_cli_main_missing_artifacts_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "--manager",
            _MANAGER,
            "--storage-root",
            str(tmp_path / "missing"),
            "--output",
            str(tmp_path / "reports"),
        ]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert "1d factor selection" in captured.err


def test_run_report_and_format_summary(tmp_path: Path) -> None:
    lake = tmp_path / "data"
    output = tmp_path / "reports"
    _build_lake(lake)
    parser = build_parser()
    args = parser.parse_args(
        [
            "--manager",
            _MANAGER,
            "--storage-root",
            str(lake),
            "--output",
            str(output),
        ]
    )
    options = build_options(args)
    result = run_report(options)
    text = format_summary(result)
    assert "production_artifacts_unchanged: True" in text
    assert "deterministic: True" in text
    assert "Report paths" in text
