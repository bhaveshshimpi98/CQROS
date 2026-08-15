"""Unit tests for the 1d Factor Stability root-cause reporter."""

from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl
import pytest

from cqros.core.constants import EXCHANGE_BINANCE, MARKET_USDT_PERPETUAL
from cqros.reporting.exceptions import ReportingValidationError
from cqros.reporting.factor_stability_1d_root_cause import (
    COMPARISON_TIMEFRAMES,
    FOLD_SCHEMA,
    PRIMARY_FACTOR_DEGENERATION,
    PRIMARY_GENUINE_TIMEFRAME_SIGNAL_WEAKNESS,
    PRIMARY_INSUFFICIENT_EVIDENCE,
    PRIMARY_STATISTICAL_POWER_LIMITATION,
    ROOT_CAUSE_COMPARISON_CSV_NAME,
    ROOT_CAUSE_DISTRIBUTION_CSV_NAME,
    ROOT_CAUSE_FACTORS_CSV_NAME,
    ROOT_CAUSE_FOLDS_CSV_NAME,
    ROOT_CAUSE_GLOBAL_CSV_NAME,
    ROOT_CAUSE_SUMMARY_TXT_NAME,
    TARGET_TIMEFRAME,
    FactorStability1dRootCauseReporter,
    _frame_from_dicts,
    _observations_aligned_to_pcv_folds,
    classify_primary_root_cause,
    forbidden_import_violations,
    hash_watched_production_artifacts,
)

_MANAGER = "default"
_YEAR = 2026
_EXCHANGE = EXCHANGE_BINANCE
_MARKET = MARKET_USDT_PERPETUAL


def _selection_frame(*, selected_count: int = 4, tested_count: int = 8) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(tested_count):
        selected = index < selected_count
        direction = 1 if index % 2 == 0 else -1
        ic = 0.08 if selected else 0.01
        rows.append(
            {
                "factor_name": f"factor_{index}",
                "factor_version": "1.0.0",
                "timeframe": TARGET_TIMEFRAME,
                "selection_time": 1_700_000_000_000,
                "factor_category": "momentum" if index < 2 else "volume",
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
    values: list[float | None],
    returns: list[float],
    direction: int,
    selection_ic: float,
    symbols: list[str] | None = None,
    base_time: int = 1_700_000_000_000,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    symbol_names = symbols or [f"S{index}" for index in range(len(values))]
    for index, (value, ret) in enumerate(zip(values, returns, strict=True)):
        rows.append(
            {
                "manager": _MANAGER,
                "engine": "simple",
                "symbol": symbol_names[index % len(symbol_names)],
                "timeframe": TARGET_TIMEFRAME,
                "year": _YEAR,
                "fold_id": fold_id,
                "observation_time": base_time + fold_id * 86_400_000 + index * 1_000,
                "factor_name": factor_name,
                "factor_version": "1.0.0",
                "selected": True,
                "partition": "OOS",
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


def _write_partition(
    root: Path,
    tier: str,
    timeframe: str,
    frame: pl.DataFrame,
    *,
    year: int = _YEAR,
) -> None:
    path = root / tier / _MANAGER / _EXCHANGE / _MARKET / timeframe
    path.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path / f"{year}.parquet")


def _ledger(fold_count: int = 5) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "strategy_name": ["s"] * fold_count,
            "strategy_version": ["v1"] * fold_count,
            "timeframe": [TARGET_TIMEFRAME] * fold_count,
            "fold_id": list(range(1, fold_count + 1)),
            "train_start": [1] * fold_count,
            "train_end": [2] * fold_count,
            "test_start": [3] * fold_count,
            "test_end": [4] * fold_count,
            "train_rows": [100] * fold_count,
            "test_rows": [20] * fold_count,
            "selected_factors": [4] * fold_count,
            "model_version": ["v1"] * fold_count,
            "train_score": [0.1] * fold_count,
            "test_score": [0.05] * fold_count,
            "overfit_gap": [0.05] * fold_count,
            "status": ["PASS"] * fold_count,
        }
    )


def _build_degenerate_1d_lake(root: Path) -> None:
    """Create a 1d lake with many null factors and negative fold ICs."""
    selection = _selection_frame(selected_count=4, tested_count=8)
    rows: list[dict[str, object]] = []
    # factor_0 / factor_1 usable; factor_2 / factor_3 fully null (degenerate)
    for fold_id in (1, 2, 3, 4, 5):
        rows.extend(
            _obs_rows(
                factor_name="factor_0",
                fold_id=fold_id,
                values=[1.0, 2.0, 3.0, 4.0],
                returns=[-0.01, -0.02, -0.03, -0.04],
                direction=1,
                selection_ic=0.08,
            )
        )
        rows.extend(
            _obs_rows(
                factor_name="factor_1",
                fold_id=fold_id,
                values=[4.0, 3.0, 2.0, 1.0],
                returns=[-0.01, -0.02, -0.03, -0.04],
                direction=-1,
                selection_ic=-0.08,
            )
        )
        rows.extend(
            _obs_rows(
                factor_name="factor_2",
                fold_id=fold_id,
                values=[None, None, None, None],
                returns=[-0.01, -0.02, -0.03, -0.04],
                direction=1,
                selection_ic=0.08,
            )
        )
        rows.extend(
            _obs_rows(
                factor_name="factor_3",
                fold_id=fold_id,
                values=[None, None, None, None],
                returns=[-0.01, -0.02, -0.03, -0.04],
                direction=1,
                selection_ic=0.08,
            )
        )
    obs = pl.DataFrame(rows)
    _write_partition(root, "factor_selection", TARGET_TIMEFRAME, selection)
    _write_partition(root, "purged_cv_evaluation", TARGET_TIMEFRAME, obs)
    _write_partition(root, "walk_forward_evaluation", TARGET_TIMEFRAME, obs)
    _write_partition(root, "purged_cv", TARGET_TIMEFRAME, _ledger())
    _write_partition(root, "walk_forward", TARGET_TIMEFRAME, _ledger())

    # Peer timeframe with richer timestamps / non-degenerate factors.
    peer_selection = selection.with_columns(pl.lit("4h").alias("timeframe"))
    peer_rows: list[dict[str, object]] = []
    for fold_id in (1, 2, 3, 4, 5):
        for factor_index in range(4):
            direction = 1 if factor_index % 2 == 0 else -1
            values = [float(i + 1) for i in range(8)]
            returns = [0.01 * float(i + 1) for i in range(8)]
            if direction < 0:
                values = list(reversed(values))
            peer_rows.extend(
                _obs_rows(
                    factor_name=f"factor_{factor_index}",
                    fold_id=fold_id,
                    values=values,
                    returns=returns,
                    direction=direction,
                    selection_ic=0.08 if direction > 0 else -0.08,
                    base_time=1_700_000_000_000 + 10_000_000 * fold_id,
                )
            )
    peer_obs = pl.DataFrame(peer_rows).with_columns(pl.lit("4h").alias("timeframe"))
    _write_partition(root, "factor_selection", "4h", peer_selection)
    _write_partition(root, "walk_forward_evaluation", "4h", peer_obs)
    _write_partition(root, "purged_cv_evaluation", "4h", peer_obs.clear())
    _write_partition(
        root, "walk_forward", "4h", _ledger().with_columns(pl.lit("4h").alias("timeframe"))
    )
    _write_partition(
        root, "purged_cv", "4h", _ledger().with_columns(pl.lit("4h").alias("timeframe"))
    )


def test_classify_primary_root_cause_degeneration() -> None:
    primary, secondary, confidence = classify_primary_root_cause(
        selected_factors=20,
        fold_count=5,
        degenerate_factor_count=9,
        high_missingness_factor_count=14,
        median_null_rate=0.85,
        unique_oos_timestamps=17,
        comparison_median_timestamps=800.0,
        median_cross_section=123.0,
        comparison_median_cross_section=100.0,
        target_std=0.11,
        comparison_median_target_std=0.02,
        redundant_group_count=0,
        redundancy_status="REDUNDANCY_ANALYSIS_UNAVAILABLE",
        selection_ratio=0.27,
        comparison_selection_ratio=0.27,
        train_positive_oos_negative_count=3,
        degradation_median=-0.01,
        negative_fold_count=4,
        oos_oriented_ic=-0.024,
        bootstrap_ci_low=-0.05,
        bootstrap_ci_high=0.01,
        alignment_issue=True,
    )
    assert primary == PRIMARY_FACTOR_DEGENERATION
    assert PRIMARY_STATISTICAL_POWER_LIMITATION in secondary
    assert confidence == "HIGH"


def test_classify_primary_insufficient_evidence() -> None:
    primary, secondary, confidence = classify_primary_root_cause(
        selected_factors=0,
        fold_count=0,
        degenerate_factor_count=0,
        high_missingness_factor_count=0,
        median_null_rate=None,
        unique_oos_timestamps=None,
        comparison_median_timestamps=None,
        median_cross_section=None,
        comparison_median_cross_section=None,
        target_std=None,
        comparison_median_target_std=None,
        redundant_group_count=0,
        redundancy_status="REDUNDANCY_ANALYSIS_UNAVAILABLE",
        selection_ratio=None,
        comparison_selection_ratio=None,
        train_positive_oos_negative_count=0,
        degradation_median=None,
        negative_fold_count=0,
        oos_oriented_ic=None,
        bootstrap_ci_low=None,
        bootstrap_ci_high=None,
        alignment_issue=False,
    )
    assert primary == PRIMARY_INSUFFICIENT_EVIDENCE
    assert secondary == ()
    assert confidence == "LOW"


def test_1d_panel_and_five_fold_discovery(tmp_path: Path) -> None:
    _build_degenerate_1d_lake(tmp_path)
    result = FactorStability1dRootCauseReporter(
        storage_root=tmp_path,
        output_root=tmp_path / "reports",
        manager=_MANAGER,
    ).run()
    assert result.global_frame["timeframe"][0] == TARGET_TIMEFRAME
    assert int(result.global_frame["selected_factors"][0]) == 4
    assert int(result.global_frame["fold_count"][0]) == 5
    assert result.fold_frame.height == 5
    assert sorted(result.fold_frame["fold_id"].to_list()) == [1, 2, 3, 4, 5]


def test_factor_fold_distribution_comparison_aggregations(tmp_path: Path) -> None:
    _build_degenerate_1d_lake(tmp_path)
    result = FactorStability1dRootCauseReporter(
        storage_root=tmp_path,
        output_root=tmp_path / "reports",
        manager=_MANAGER,
    ).run()
    assert result.factor_frame.height == 4
    assert int(result.factor_frame.filter(pl.col("degenerate_flag")).height) == 2
    assert "null_rate" in result.factor_frame.columns
    assert "oriented_oos_minus_is" in result.factor_frame.columns
    assert result.fold_frame["oriented_oos_ic"].null_count() < result.fold_frame.height
    assert result.distribution_frame.filter(pl.col("metric_group") == "target_overall").height > 0
    assert result.distribution_frame.filter(pl.col("metric_group") == "cross_section").height > 0
    assert TARGET_TIMEFRAME in result.comparison_frame["timeframe"].to_list()
    assert "4h" in result.comparison_frame["timeframe"].to_list()
    assert set(COMPARISON_TIMEFRAMES).issuperset(
        set(result.comparison_frame["timeframe"].to_list())
    )


def test_primary_classification_and_redundancy_train_boundary(tmp_path: Path) -> None:
    _build_degenerate_1d_lake(tmp_path)
    result = FactorStability1dRootCauseReporter(
        storage_root=tmp_path,
        output_root=tmp_path / "reports",
        manager=_MANAGER,
    ).run()
    assert result.primary_classification == PRIMARY_FACTOR_DEGENERATION
    assert result.global_frame["redundancy_status"][0] == "REDUNDANCY_ANALYSIS_UNAVAILABLE"
    assert "Primary classification:" in result.summary_text


def test_leakage_guard_ast() -> None:
    module = Path("src/cqros/reporting/factor_stability_1d_root_cause.py")
    cli = Path("src/cqros/cli/report_factor_stability_1d_root_cause.py")
    assert forbidden_import_violations(module) == ()
    assert forbidden_import_violations(cli) == ()


def test_production_artifact_immutability(tmp_path: Path) -> None:
    _build_degenerate_1d_lake(tmp_path)
    before = hash_watched_production_artifacts(tmp_path)
    result = FactorStability1dRootCauseReporter(
        storage_root=tmp_path,
        output_root=tmp_path / "reports",
        manager=_MANAGER,
    ).run()
    after = hash_watched_production_artifacts(tmp_path)
    assert before == after
    assert result.production_artifacts_unchanged is True
    assert (tmp_path / "reports" / "1d_root_cause_hashes_before.txt").exists()
    assert (tmp_path / "reports" / "1d_root_cause_hashes_after.txt").exists()


def test_deterministic_csv_generation(tmp_path: Path) -> None:
    _build_degenerate_1d_lake(tmp_path)
    output_a = tmp_path / "reports_a"
    output_b = tmp_path / "reports_b"
    first = FactorStability1dRootCauseReporter(
        storage_root=tmp_path,
        output_root=output_a,
        manager=_MANAGER,
    ).run()
    second = FactorStability1dRootCauseReporter(
        storage_root=tmp_path,
        output_root=output_b,
        manager=_MANAGER,
    ).run()
    assert first.deterministic is True
    assert second.deterministic is True
    for name in (
        ROOT_CAUSE_GLOBAL_CSV_NAME,
        ROOT_CAUSE_FOLDS_CSV_NAME,
        ROOT_CAUSE_FACTORS_CSV_NAME,
        ROOT_CAUSE_DISTRIBUTION_CSV_NAME,
        ROOT_CAUSE_COMPARISON_CSV_NAME,
        ROOT_CAUSE_SUMMARY_TXT_NAME,
    ):
        left = (output_a / name).read_bytes()
        right = (output_b / name).read_bytes()
        assert left == right
        assert hashlib.sha256(left).hexdigest() == hashlib.sha256(right).hexdigest()


def test_missing_1d_partition_fails(tmp_path: Path) -> None:
    with pytest.raises(ReportingValidationError, match="1d factor selection"):
        FactorStability1dRootCauseReporter(
            storage_root=tmp_path,
            output_root=tmp_path / "reports",
            manager=_MANAGER,
        ).run()


def test_train_oos_degradation_metric(tmp_path: Path) -> None:
    _build_degenerate_1d_lake(tmp_path)
    result = FactorStability1dRootCauseReporter(
        storage_root=tmp_path,
        output_root=tmp_path / "reports",
        manager=_MANAGER,
    ).run()
    usable = result.factor_frame.filter(pl.col("mean_oriented_oos_ic").is_not_null())
    assert usable.height >= 1
    assert "oriented_oos_minus_is" in usable.columns


def test_observations_aligned_to_pcv_folds_reassigns_fold_ids() -> None:
    observations = pl.DataFrame(
        {
            "observation_time": [100, 150, 200, 250],
            "fold_id": [10, 11, 12, 13],
            "factor_name": ["a", "a", "a", "a"],
        }
    )
    ledger = pl.DataFrame(
        {
            "fold_id": [1, 2],
            "test_start_time": [100, 200],
            "test_end_time": [200, 300],
        }
    )
    aligned = _observations_aligned_to_pcv_folds(observations, ledger)
    assert sorted(aligned["fold_id"].unique().to_list()) == [1, 2]
    assert aligned.filter(pl.col("fold_id") == 1).height == 2
    assert aligned.filter(pl.col("fold_id") == 2).height == 2

    frame = _frame_from_dicts(
        [
            {
                "timeframe": "1d",
                "year": 2026,
                "fold_id": 1,
                "train_rows": 10,
                "oos_rows": 5,
                "raw_oos_mean_return": None,
                "raw_oos_ic": None,
                "oriented_oos_ic": None,
                "positive_factor_count": 0,
                "negative_factor_count": 0,
                "median_factor_ic": None,
                "dispersion_factor_ic": None,
                "factors_positive_oos_ic": 0,
                "factors_negative_oos_ic": 0,
                "unique_timestamps": None,
                "median_cross_section": None,
            },
            {
                "timeframe": "1d",
                "year": 2026,
                "fold_id": 2,
                "train_rows": 10,
                "oos_rows": 5,
                "raw_oos_mean_return": 0.01,
                "raw_oos_ic": -0.02,
                "oriented_oos_ic": 0.353553,
                "positive_factor_count": 3,
                "negative_factor_count": 1,
                "median_factor_ic": 0.2,
                "dispersion_factor_ic": 0.1,
                "factors_positive_oos_ic": 3,
                "factors_negative_oos_ic": 1,
                "unique_timestamps": 4,
                "median_cross_section": 2.5,
            },
        ],
        FOLD_SCHEMA,
    )
    assert frame.height == 2
    assert frame.schema["oriented_oos_ic"] == pl.Float64
    assert frame["oriented_oos_ic"].to_list()[1] == pytest.approx(0.353553)


def test_genuine_weakness_classification_path() -> None:
    primary, _, confidence = classify_primary_root_cause(
        selected_factors=10,
        fold_count=5,
        degenerate_factor_count=0,
        high_missingness_factor_count=0,
        median_null_rate=0.0,
        unique_oos_timestamps=500,
        comparison_median_timestamps=500.0,
        median_cross_section=100.0,
        comparison_median_cross_section=100.0,
        target_std=0.02,
        comparison_median_target_std=0.02,
        redundant_group_count=0,
        redundancy_status="REDUNDANCY_ANALYSIS_UNAVAILABLE",
        selection_ratio=0.27,
        comparison_selection_ratio=0.27,
        train_positive_oos_negative_count=1,
        degradation_median=0.0,
        negative_fold_count=4,
        oos_oriented_ic=-0.03,
        bootstrap_ci_low=-0.04,
        bootstrap_ci_high=-0.01,
        alignment_issue=False,
    )
    assert primary == PRIMARY_GENUINE_TIMEFRAME_SIGNAL_WEAKNESS
    assert confidence == "HIGH"
