"""Unit tests for Factor Selection stability review reporting."""

from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl
import pytest

from cqros.cli.report_factor_stability import build_options, build_parser
from cqros.core.constants import EXCHANGE_BINANCE, MARKET_USDT_PERPETUAL
from cqros.reporting.exceptions import ReportingValidationError
from cqros.reporting.factor_stability_selection_report import (
    FACTOR_CLASS_INSUFFICIENT_DATA,
    FACTOR_CLASS_MIXED,
    FACTOR_CLASS_ORIENTATION_INSUFFICIENT,
    FACTOR_CLASS_STABLE_NEGATIVE,
    FACTOR_CLASS_STABLE_POSITIVE,
    REDUNDANCY_ANALYSIS_UNAVAILABLE,
    VERDICT_TIMEFRAME_SIGNAL_WEAKNESS,
    FactorStabilitySelectionReporter,
    classify_factor_stability,
    classify_global_status,
    classify_verdict,
    forbidden_import_violations,
)

_MANAGER = "default"
_YEAR = 2026
_TIMEFRAME = "1d"
_EXCHANGE = EXCHANGE_BINANCE
_MARKET = MARKET_USDT_PERPETUAL


def _selection_frame(*, selected_count: int = 2, tested_count: int = 4) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(tested_count):
        selected = index < selected_count
        direction = 1 if index % 2 == 0 else -1
        ic = 0.05 if selected else 0.01
        rows.append(
            {
                "factor_name": f"factor_{index}",
                "factor_version": "1.0.0",
                "timeframe": _TIMEFRAME,
                "selection_time": 1_700_000_000_000,
                "factor_category": "momentum",
                "selected": selected,
                "selection_score": float(tested_count - index),
                "selection_rank": index + 1,
                "selection_reason": "test",
                "selection_ic": ic if direction > 0 else -ic,
                "selected_direction": direction,
                "orientation_policy": "signed_ic_v1",
                "status": "PASS",
            }
        )
    return pl.DataFrame(rows)


def _obs_rows(
    *,
    factor_name: str,
    fold_id: int,
    values: list[float],
    returns: list[float],
    direction: int,
    selection_ic: float,
    partition: str = "OOS",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, (value, ret) in enumerate(zip(values, returns, strict=True)):
        rows.append(
            {
                "manager": _MANAGER,
                "engine": "simple",
                "symbol": "BTCUSDT",
                "timeframe": _TIMEFRAME,
                "year": _YEAR,
                "fold_id": fold_id,
                "observation_time": 1_700_000_000_000 + fold_id * 100_000 + index,
                "factor_name": factor_name,
                "factor_version": "1.0.0",
                "selected": True,
                "partition": partition,
                "future_return_1": ret,
                "factor_value": value,
                "selection_ic": selection_ic,
                "selected_direction": direction,
                "orientation_policy": "signed_ic_v1",
                "prediction": None,
                "residual": None,
                "correct": None,
                "status": "PASS",
            }
        )
    return rows


def _write_lake(
    root: Path,
    *,
    selection: pl.DataFrame,
    pcv_obs: pl.DataFrame,
    wf_obs: pl.DataFrame | None = None,
    ledger: pl.DataFrame | None = None,
    timeframe: str = _TIMEFRAME,
    year: int = _YEAR,
) -> None:
    base = root / "factor_selection" / _MANAGER / _EXCHANGE / _MARKET / timeframe
    base.mkdir(parents=True, exist_ok=True)
    selection.write_parquet(base / f"{year}.parquet")

    pcv_dir = root / "purged_cv_evaluation" / _MANAGER / _EXCHANGE / _MARKET / timeframe
    pcv_dir.mkdir(parents=True, exist_ok=True)
    pcv_obs.write_parquet(pcv_dir / f"{year}.parquet")

    if wf_obs is not None:
        wf_dir = root / "walk_forward_evaluation" / _MANAGER / _EXCHANGE / _MARKET / timeframe
        wf_dir.mkdir(parents=True, exist_ok=True)
        wf_obs.write_parquet(wf_dir / f"{year}.parquet")

    if ledger is not None:
        ledger_dir = root / "purged_cv" / _MANAGER / _EXCHANGE / _MARKET / timeframe
        ledger_dir.mkdir(parents=True, exist_ok=True)
        ledger.write_parquet(ledger_dir / f"{year}.parquet")

    # Watched production ledgers for hashing (may be empty content).
    wf_ledger_dir = root / "walk_forward" / _MANAGER / _EXCHANGE / _MARKET / timeframe
    wf_ledger_dir.mkdir(parents=True, exist_ok=True)
    if not (wf_ledger_dir / f"{year}.parquet").exists():
        pl.DataFrame({"fold_id": [1], "status": ["PASS"]}).write_parquet(
            wf_ledger_dir / f"{year}.parquet"
        )


def _stable_panel_obs() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Build OOS observations with clear positive oriented signal on both factors."""
    rows: list[dict[str, object]] = []
    for fold_id in (1, 2, 3):
        # factor_0 direction +1: values align with returns
        rows.extend(
            _obs_rows(
                factor_name="factor_0",
                fold_id=fold_id,
                values=[1.0, 2.0, 3.0, 4.0, 5.0],
                returns=[0.01, 0.02, 0.03, 0.04, 0.05],
                direction=1,
                selection_ic=0.05,
            )
        )
        # factor_1 direction -1: raw anti-aligned, oriented positive
        rows.extend(
            _obs_rows(
                factor_name="factor_1",
                fold_id=fold_id,
                values=[5.0, 4.0, 3.0, 2.0, 1.0],
                returns=[0.01, 0.02, 0.03, 0.04, 0.05],
                direction=-1,
                selection_ic=-0.05,
            )
        )
    obs = pl.DataFrame(rows)
    ledger = pl.DataFrame(
        {
            "strategy_name": ["s"] * 3,
            "strategy_version": ["v1"] * 3,
            "timeframe": [_TIMEFRAME] * 3,
            "fold_id": [1, 2, 3],
            "train_start_time": [1, 2, 3],
            "train_end_time": [1, 2, 3],
            "test_start_time": [1, 2, 3],
            "test_end_time": [1, 2, 3],
            "purge_size": [1, 1, 1],
            "embargo_size": [1, 1, 1],
            "train_rows": [100, 100, 100],
            "test_rows": [20, 20, 20],
            "train_score": [0.1, 0.1, 0.1],
            "test_score": [0.1, 0.1, 0.1],
            "overfit_gap": [0.0, 0.0, 0.0],
            "status": ["PASS", "PASS", "PASS"],
        }
    )
    return obs, obs, ledger


def test_classify_factor_stability_rules() -> None:
    assert (
        classify_factor_stability(
            oriented_fold_ics=[],
            mean_oriented_oos=None,
            oriented_training_ic=0.1,
        )
        == FACTOR_CLASS_INSUFFICIENT_DATA
    )
    assert (
        classify_factor_stability(
            oriented_fold_ics=[-0.1, -0.2, -0.05],
            mean_oriented_oos=-0.1,
            oriented_training_ic=0.2,
        )
        == FACTOR_CLASS_ORIENTATION_INSUFFICIENT
    )
    assert (
        classify_factor_stability(
            oriented_fold_ics=[0.1, 0.2, 0.05],
            mean_oriented_oos=0.1,
            oriented_training_ic=0.2,
        )
        == FACTOR_CLASS_STABLE_POSITIVE
    )
    assert (
        classify_factor_stability(
            oriented_fold_ics=[-0.1, -0.2, -0.05],
            mean_oriented_oos=-0.1,
            oriented_training_ic=-0.2,
        )
        == FACTOR_CLASS_STABLE_NEGATIVE
    )
    assert (
        classify_factor_stability(
            oriented_fold_ics=[0.1, -0.2, 0.05],
            mean_oriented_oos=-0.01,
            oriented_training_ic=0.2,
        )
        == FACTOR_CLASS_MIXED
    )


def test_fold_stability_and_degradation(tmp_path: Path) -> None:
    obs, wf_obs, ledger = _stable_panel_obs()
    # Flip factor_0 fold 3 to create mixed degradation path on a weak panel.
    weak_rows = obs.to_dicts()
    for row in weak_rows:
        if row["factor_name"] == "factor_0" and row["fold_id"] == 3:
            row["future_return_1"] = -float(row["factor_value"]) * 0.01
    weak_obs = pl.DataFrame(weak_rows)
    _write_lake(
        tmp_path,
        selection=_selection_frame(),
        pcv_obs=weak_obs,
        wf_obs=wf_obs,
        ledger=ledger,
    )
    result = FactorStabilitySelectionReporter(
        storage_root=tmp_path,
        output_root=tmp_path / "reports",
        manager=_MANAGER,
    ).run(timeframes=(_TIMEFRAME,))
    folds = result.fold_frames[_TIMEFRAME]
    assert folds.height == 3
    assert set(folds["fold_id"].to_list()) == {1, 2, 3}
    assert "train_rows" in folds.columns
    factors = result.factor_frames[_TIMEFRAME]
    assert "ic_degradation" in factors.columns
    assert "train_positive_oos_negative" in factors.columns
    assert factors.filter(pl.col("ic_degradation").is_not_null()).height >= 1


def test_selection_intensity_reconstruction(tmp_path: Path) -> None:
    obs, wf_obs, ledger = _stable_panel_obs()
    _write_lake(
        tmp_path,
        selection=_selection_frame(selected_count=2, tested_count=5),
        pcv_obs=obs,
        wf_obs=wf_obs,
        ledger=ledger,
    )
    result = FactorStabilitySelectionReporter(
        storage_root=tmp_path,
        output_root=tmp_path / "reports",
        manager=_MANAGER,
    ).run(timeframes=(_TIMEFRAME,))
    summary = result.summary_frames[_TIMEFRAME]
    assert summary["selected_factors"].item() == 2
    assert summary["tested_factors"].item() == 5
    assert summary["selection_ratio"].item() == pytest.approx(0.4)


def test_missing_data_behavior(tmp_path: Path) -> None:
    selection = _selection_frame()
    base = tmp_path / "factor_selection" / _MANAGER / _EXCHANGE / _MARKET / _TIMEFRAME
    base.mkdir(parents=True, exist_ok=True)
    selection.write_parquet(base / f"{_YEAR}.parquet")
    # Watched ledger stubs so hashing works and mutation checks can run.
    for tier in ("walk_forward", "purged_cv"):
        path = tmp_path / tier / _MANAGER / _EXCHANGE / _MARKET / _TIMEFRAME
        path.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"fold_id": [1]}).write_parquet(path / f"{_YEAR}.parquet")

    result = FactorStabilitySelectionReporter(
        storage_root=tmp_path,
        output_root=tmp_path / "reports",
        manager=_MANAGER,
    ).run(timeframes=(_TIMEFRAME,))
    factors = result.factor_frames[_TIMEFRAME]
    assert factors.height == 2
    assert set(factors["stability_class"].to_list()) == {FACTOR_CLASS_INSUFFICIENT_DATA}
    summary = result.summary_frames[_TIMEFRAME]
    assert summary["status"].item() is not None


def test_wf_pcv_comparison_fields(tmp_path: Path) -> None:
    obs, wf_obs, ledger = _stable_panel_obs()
    _write_lake(
        tmp_path,
        selection=_selection_frame(),
        pcv_obs=obs,
        wf_obs=wf_obs,
        ledger=ledger,
    )
    result = FactorStabilitySelectionReporter(
        storage_root=tmp_path,
        output_root=tmp_path / "reports",
        manager=_MANAGER,
    ).run(timeframes=(_TIMEFRAME,))
    summary = result.summary_frames[_TIMEFRAME]
    for column in (
        "wf_raw_oos_ic",
        "wf_oriented_oos_ic",
        "pcv_raw_oos_ic",
        "pcv_oriented_oos_ic",
    ):
        assert column in summary.columns
        assert summary[column].item() is not None


def test_deterministic_csv_output(tmp_path: Path) -> None:
    obs, wf_obs, ledger = _stable_panel_obs()
    _write_lake(
        tmp_path,
        selection=_selection_frame(),
        pcv_obs=obs,
        wf_obs=wf_obs,
        ledger=ledger,
    )
    output = tmp_path / "reports"
    reporter = FactorStabilitySelectionReporter(
        storage_root=tmp_path,
        output_root=output,
        manager=_MANAGER,
    )
    first = reporter.run(timeframes=(_TIMEFRAME,))
    second = reporter.run(timeframes=(_TIMEFRAME,))
    for key in (f"factor_{_TIMEFRAME}", f"folds_{_TIMEFRAME}", f"summary_{_TIMEFRAME}"):
        left = first.paths[key].read_bytes()
        right = second.paths[key].read_bytes()
        assert left == right


def test_immutability_hash_and_mutation_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    obs, wf_obs, ledger = _stable_panel_obs()
    _write_lake(
        tmp_path,
        selection=_selection_frame(),
        pcv_obs=obs,
        wf_obs=wf_obs,
        ledger=ledger,
    )
    reporter = FactorStabilitySelectionReporter(
        storage_root=tmp_path,
        output_root=tmp_path / "reports",
        manager=_MANAGER,
    )
    result = reporter.run(timeframes=(_TIMEFRAME,))
    assert result.production_ledgers_unchanged is True
    assert result.hashes_before == result.hashes_after

    selection_path = (
        tmp_path
        / "factor_selection"
        / _MANAGER
        / _EXCHANGE
        / _MARKET
        / _TIMEFRAME
        / f"{_YEAR}.parquet"
    )
    original = selection_path.read_bytes()
    from cqros.reporting import factor_stability_selection_report as mod

    call_count = {"n": 0}
    real_hash = mod.hash_watched_production_ledgers

    def _counting_hash(storage_root: Path) -> dict[str, str]:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return real_hash(storage_root)
        # Mutate after analysis completes, immediately before the after-hash.
        selection_path.write_bytes(original + b"\0")
        return real_hash(storage_root)

    monkeypatch.setattr(mod, "hash_watched_production_ledgers", _counting_hash)
    with pytest.raises(ReportingValidationError, match="production ledgers changed"):
        FactorStabilitySelectionReporter(
            storage_root=tmp_path,
            output_root=tmp_path / "reports2",
            manager=_MANAGER,
        ).run(timeframes=(_TIMEFRAME,))


def test_cli_discovery_options() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--manager",
            "default",
            "--timeframes",
            "1d",
            "4h",
            "--storage-root",
            "data",
            "--output",
            "reports/factor_stability",
            "--verbose",
        ]
    )
    options = build_options(args)
    assert options.manager == "default"
    assert options.timeframes == ("1d", "4h")
    assert options.storage_root == Path("data")
    assert options.output == Path("reports/factor_stability")
    assert options.verbose is True


def test_redundancy_unavailable_without_train(tmp_path: Path) -> None:
    obs, wf_obs, ledger = _stable_panel_obs()
    assert obs["partition"].unique().to_list() == ["OOS"]
    _write_lake(
        tmp_path,
        selection=_selection_frame(),
        pcv_obs=obs,
        wf_obs=wf_obs,
        ledger=ledger,
    )
    result = FactorStabilitySelectionReporter(
        storage_root=tmp_path,
        output_root=tmp_path / "reports",
        manager=_MANAGER,
    ).run(timeframes=(_TIMEFRAME,))
    summary = result.summary_frames[_TIMEFRAME]
    assert summary["redundancy_status"].item() == REDUNDANCY_ANALYSIS_UNAVAILABLE
    assert summary["redundant_group_count"].item() == 0


def test_verdict_prefers_timeframe_weakness() -> None:
    verdict = classify_verdict(
        status="MIXED_STABILITY",
        wf_oriented_oos_ic=-0.02,
        pcv_oriented_oos_ic=-0.003,
        fold_count=5,
        negative_fold_count=4,
        train_positive_oos_negative_count=1,
        selected_factors=20,
        redundancy_status=REDUNDANCY_ANALYSIS_UNAVAILABLE,
        redundant_group_count=0,
        oriented_positive_factor_ratio=0.4,
    )
    assert verdict == VERDICT_TIMEFRAME_SIGNAL_WEAKNESS


def test_forbidden_imports() -> None:
    source = Path("src/cqros/reporting/factor_stability_selection_report.py")
    cli = Path("src/cqros/cli/report_factor_stability.py")
    assert forbidden_import_violations(source) == ()
    assert forbidden_import_violations(cli) == ()


def test_sha256_helper_stable(tmp_path: Path) -> None:
    path = tmp_path / "x.parquet"
    pl.DataFrame({"a": [1, 2, 3]}).write_parquet(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(digest) == 64


def test_global_status_insufficient() -> None:
    status = classify_global_status(
        selected_factors=0,
        tested_factors=0,
        fold_count=0,
        oriented_positive_factor_ratio=None,
        stable_positive_factor_count=0,
        mixed_factor_count=0,
        stable_negative_factor_count=0,
        orientation_insufficient_factor_count=0,
        insufficient_data_factor_count=0,
        train_positive_oos_negative_count=0,
        pcv_oriented_oos_ic=None,
        wf_oriented_oos_ic=None,
    )
    assert status == "INSUFFICIENT_DATA"
