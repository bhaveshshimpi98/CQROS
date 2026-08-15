"""Unit tests for Factor Stability + OOS IC diagnostics."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import polars as pl
import pytest

from cqros.core.constants import EXCHANGE_BINANCE, MARKET_USDT_PERPETUAL
from cqros.purged_cv.engine import SimplePurgedCVEngine
from cqros.purged_cv.evaluation import PurgedCVEvaluator
from cqros.reporting.factor_stability_diagnostic import (
    ALIGNMENT_AUDIT_CSV_NAME,
    FAMILIES_CSV_NAME,
    FOLDS_CSV_NAME,
    GLOBAL_CSV_NAME,
    STABILITY_ALL_CSV_NAME,
    TIMEFRAMES_CSV_NAME,
    PanelDiagnosticBundle,
    aggregate_cross_timeframe,
    aggregate_families,
    build_global_summary,
    classify_cross_timeframe_stability,
    classify_is_oos,
    classify_orientation,
    compute_fold_factor_ic,
    compute_spearman_ic,
    detect_ic_methodology,
    forbidden_import_violations,
    verify_factor_value_integrity,
    verify_target_alignment,
    write_factor_stability_reports,
)
from cqros.walk_forward.evaluation_input import TARGET_COLUMN

_MANAGER = "default"
_ENGINE = "simple"
_YEAR = 2026
_TIMEFRAME = "1h"
_EXCHANGE = EXCHANGE_BINANCE
_MARKET = MARKET_USDT_PERPETUAL
_SYMBOL = "BTCUSDT"
_FACTOR = "momentum"
_FACTOR_VERSION = "1.0.0"


def _walk_forward(*, n_rows: int = 40) -> pl.DataFrame:
    base = 1_700_000_000_000
    rows: list[dict[str, object]] = []
    for index in range(n_rows):
        test_start = base + index * 3_600_000
        rows.append(
            {
                "strategy_name": "default_strategy",
                "strategy_version": "v1",
                "timeframe": _TIMEFRAME,
                "fold_id": index + 1,
                "train_start": test_start - 10 * 3_600_000,
                "train_end": test_start - 3_600_000,
                "test_start": test_start,
                "test_end": test_start + 2 * 3_600_000,
                "train_rows": 10,
                "test_rows": 3,
                "selected_factors": 1,
                "model_version": "v1",
                "train_score": 0.01 * float(index + 1),
                "test_score": 0.005 * float(index + 1),
                "overfit_gap": 0.005 * float(index + 1),
                "status": "PASS",
            }
        )
    return pl.DataFrame(rows)


def _evaluation_input(
    *,
    n_rows: int = 80,
    factor_values: list[float] | None = None,
    targets: list[float] | None = None,
    symbol: str = _SYMBOL,
    duplicate: bool = False,
    missing_target: bool = False,
) -> pl.DataFrame:
    open_times = [1_700_000_000_000 + index * 3_600_000 for index in range(n_rows)]
    values = factor_values or [float(index + 1) for index in range(n_rows)]
    returns = targets or [0.01 * ((-1) ** index) for index in range(n_rows)]
    rows: list[dict[str, object]] = []
    for index, open_time in enumerate(open_times):
        rows.append(
            {
                "symbol": symbol,
                "timeframe": _TIMEFRAME,
                "open_time": open_time,
                "factor_name": _FACTOR,
                "factor_version": _FACTOR_VERSION,
                "factor_value": values[index],
                "selected": True,
                "selection_time": open_time,
                "selection_ic": 0.08,
                "selected_direction": 1,
                "orientation_policy": "signed_ic_v1",
                TARGET_COLUMN: None if missing_target and index == 0 else returns[index],
            }
        )
    frame = pl.DataFrame(rows)
    if duplicate:
        frame = pl.concat([frame, frame.head(1)], how="vertical")
    return frame


def _observations_from_evaluator(
    evaluation_input: pl.DataFrame | None = None,
) -> pl.DataFrame:
    walk_forward = _walk_forward()
    purged_cv = SimplePurgedCVEngine(n_folds=5, purge_size=2, embargo_size=1).build(walk_forward)
    artifacts = PurgedCVEvaluator().evaluate(
        purged_cv,
        walk_forward,
        manager=_MANAGER,
        engine=_ENGINE,
        exchange=_EXCHANGE,
        market=_MARKET,
        year=_YEAR,
        evaluation_input=evaluation_input if evaluation_input is not None else _evaluation_input(),
    )
    return artifacts.observations


def test_target_alignment_pass() -> None:
    """Aligned open_time/selection_time passes diagnostic 1."""
    audit = verify_target_alignment(
        _evaluation_input(),
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert audit["status"][0] == "PASS"
    assert int(audit["alignment_fail"][0]) == 0


def test_target_alignment_duplicate_keys_fail() -> None:
    """Duplicate observation keys fail alignment."""
    audit = verify_target_alignment(
        _evaluation_input(duplicate=True),
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert audit["status"][0] == "FAIL"
    assert int(audit["duplicate_keys"][0]) > 0


def test_missing_target_reported() -> None:
    """Missing future_return_1 values are counted."""
    audit = verify_target_alignment(
        _evaluation_input(missing_target=True),
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert int(audit["missing_labels"][0]) >= 1


def test_factor_value_preservation() -> None:
    """Exact factor values are preserved across Factors and evaluation input."""
    evaluation_input = _evaluation_input()
    factors = evaluation_input.select(
        [
            "symbol",
            "timeframe",
            "open_time",
            "factor_name",
            "factor_version",
            "factor_value",
        ]
    )
    observations = _observations_from_evaluator(evaluation_input)
    audit = verify_factor_value_integrity(
        evaluation_input=evaluation_input,
        observations=observations,
        factors_frame=factors,
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert audit["status"][0] == "PASS"
    assert int(audit["value_mismatches"][0]) == 0


def test_factor_value_mismatch_detected() -> None:
    """Value integrity fails when canonical Factors differ."""
    evaluation_input = _evaluation_input()
    factors = evaluation_input.select(
        [
            "symbol",
            "timeframe",
            "open_time",
            "factor_name",
            "factor_version",
            "factor_value",
        ]
    ).with_columns(pl.col("factor_value") + 1.0)
    audit = verify_factor_value_integrity(
        evaluation_input=evaluation_input,
        observations=pl.DataFrame(),
        factors_frame=factors,
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert audit["status"][0] == "FAIL"
    assert int(audit["value_mismatches"][0]) > 0


def test_sign_inversion_identity() -> None:
    """IC_inverted equals -IC_original for Spearman."""
    values = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    targets = pl.Series([0.1, 0.2, 0.0, -0.1, 0.3])
    original = compute_spearman_ic(values, targets)
    inverted = compute_spearman_ic(-values, targets)
    assert original is not None
    assert inverted is not None
    assert inverted == pytest.approx(-original, rel=1e-9, abs=1e-12)


def test_fold_ic_from_factor_metrics_identity() -> None:
    """Factor-metric fold IC uses oos_ic and inverted identity."""
    metrics = pl.DataFrame(
        {
            "manager": [_MANAGER],
            "engine": [_ENGINE],
            "symbol": [""],
            "timeframe": [_TIMEFRAME],
            "year": [_YEAR],
            "fold_id": [1],
            "factor_name": [_FACTOR],
            "factor_version": [_FACTOR_VERSION],
            "train_rows": [10],
            "oos_rows": [20],
            "oos_return_mean": [0.01],
            "oos_return_std": [0.02],
            "oos_positive_rate": [0.5],
            "oos_ic": [-0.125],
            "status": ["PASS"],
        }
    )
    from cqros.reporting.factor_stability_diagnostic import fold_stability_from_factor_metrics

    frame = fold_stability_from_factor_metrics(metrics)
    assert frame.height == 1
    assert frame["ic"][0] == pytest.approx(-0.125)
    assert frame["ic_inverted"][0] == pytest.approx(0.125)


def test_fold_ic_and_aggregation() -> None:
    """Fold IC is computed per fold without pooling folds first."""
    observations = _observations_from_evaluator()
    fold_ic = compute_fold_factor_ic(observations)
    assert fold_ic.height > 0
    assert set(fold_ic["fold_id"].to_list())  # non-empty
    # Inverted identity holds row-wise where IC is defined.
    defined = fold_ic.filter(pl.col("ic").is_not_null())
    if defined.height > 0:
        for row in defined.iter_rows(named=True):
            assert row["ic_inverted"] == pytest.approx(-row["ic"], rel=1e-9, abs=1e-12)


def test_cross_timeframe_aggregation() -> None:
    """Cross-timeframe aggregation classifies stability."""
    frame = pl.DataFrame(
        {
            "manager": [_MANAGER, _MANAGER, _MANAGER, _MANAGER],
            "exchange": [_EXCHANGE] * 4,
            "market": [_MARKET] * 4,
            "timeframe": ["5m", "15m", "5m", "15m"],
            "year": [_YEAR] * 4,
            "factor_name": [_FACTOR] * 4,
            "factor_version": [_FACTOR_VERSION] * 4,
            "factor_family": ["price"] * 4,
            "fold_id": [1, 1, 2, 2],
            "ic": [-0.1, -0.2, -0.05, -0.15],
        }
    )
    aggregated = aggregate_cross_timeframe(frame)
    assert aggregated.height == 1
    assert aggregated["cross_timeframe_class"][0] == "CROSS_TIMEFRAME_INVERTED"
    assert int(aggregated["negative_timeframes"][0]) == 2


def test_family_aggregation() -> None:
    """Family aggregation uses explicit family metadata."""
    frame = pl.DataFrame(
        {
            "manager": [_MANAGER, _MANAGER],
            "exchange": [_EXCHANGE, _EXCHANGE],
            "market": [_MARKET, _MARKET],
            "timeframe": [_TIMEFRAME, _TIMEFRAME],
            "year": [_YEAR, _YEAR],
            "factor_name": ["a", "b"],
            "factor_version": ["1", "1"],
            "factor_family": ["price", "volume"],
            "fold_id": [1, 1],
            "ic": [-0.2, 0.1],
        }
    )
    families = aggregate_families(frame)
    assert set(families["family"].to_list()) == {"price", "volume"}


def test_is_oos_comparison_classes() -> None:
    """IS/OOS sign classes cover degradation and agreement."""
    assert classify_is_oos(selection_ic=0.2, oos_ic=-0.1) == "IS_POSITIVE_OOS_NEGATIVE"
    assert classify_is_oos(selection_ic=0.2, oos_ic=0.1) == "IS_POSITIVE_OOS_POSITIVE"
    assert classify_is_oos(selection_ic=-0.2, oos_ic=-0.1) == "IS_NEGATIVE_OOS_NEGATIVE"
    assert classify_is_oos(selection_ic=None, oos_ic=-0.1) == "IS_NEUTRAL_OOS"


def test_negative_ic_classification() -> None:
    """Orientation classes distinguish stable inversion from instability."""
    assert (
        classify_orientation(mean_ic=-0.1, positive_folds=0, negative_folds=5)
        == "ORIENTATION_REVERSAL_CANDIDATE"
    )
    assert (
        classify_orientation(mean_ic=0.1, positive_folds=5, negative_folds=0) == "STABLE_POSITIVE"
    )
    assert classify_orientation(mean_ic=-0.01, positive_folds=2, negative_folds=3) == "UNSTABLE"


def test_non_contiguous_training_supported() -> None:
    """Purged-CV evaluation with non-contiguous train still yields fold IC."""
    observations = _observations_from_evaluator()
    assert observations.filter(pl.col("partition") == "OOS").height > 0
    fold_ic = compute_fold_factor_ic(observations)
    assert fold_ic.height > 0


def test_missing_factor_yields_empty_fold_metrics() -> None:
    """No selected observations produce an empty fold metric frame."""
    empty = pl.DataFrame(
        {
            "manager": pl.Series([], dtype=pl.String),
            "timeframe": pl.Series([], dtype=pl.String),
            "year": pl.Series([], dtype=pl.Int32),
            "fold_id": pl.Series([], dtype=pl.Int32),
            "factor_name": pl.Series([], dtype=pl.String),
            "factor_version": pl.Series([], dtype=pl.String),
            "selected": pl.Series([], dtype=pl.Boolean),
            "partition": pl.Series([], dtype=pl.String),
            "factor_value": pl.Series([], dtype=pl.Float64),
            TARGET_COLUMN: pl.Series([], dtype=pl.Float64),
        }
    )
    fold_ic = compute_fold_factor_ic(empty)
    assert fold_ic.height == 0


def test_cross_symbol_isolation() -> None:
    """Multi-symbol observations remain separated by symbol in integrity joins."""
    btc = _evaluation_input(symbol="BTCUSDT", n_rows=20)
    eth = _evaluation_input(symbol="ETHUSDT", n_rows=20)
    panel = pl.concat([btc, eth], how="vertical")
    factors = panel.select(
        [
            "symbol",
            "timeframe",
            "open_time",
            "factor_name",
            "factor_version",
            "factor_value",
        ]
    )
    audit = verify_factor_value_integrity(
        evaluation_input=panel,
        observations=pl.DataFrame(),
        factors_frame=factors,
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert audit["status"][0] == "PASS"


def test_methodology_detection() -> None:
    """Methodology detector reports aligned pooled Spearman for OOS IC."""
    methodology = detect_ic_methodology()
    assert methodology["alignment_status"] == "METHODOLOGY_ALIGNED"
    assert "Spearman" in methodology["purged_cv_oos_ic"]
    assert "cross-sectional Pearson" in methodology["factor_validation_information_coefficient"]


def test_no_leakage_imports() -> None:
    """Diagnostic modules must not import Alpha/Regime/Predictions/Signals/ml."""
    root = Path(__file__).resolve().parents[3] / "src" / "cqros"
    paths = [
        root / "reporting" / "factor_stability_diagnostic.py",
        root / "cli" / "diagnose_factor_stability.py",
    ]
    forbidden = {
        "cqros.alpha",
        "cqros.regime",
        "cqros.predictions",
        "cqros.signals",
        "cqros.ml",
    }
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert forbidden_import_violations(source) == ()
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module)
        assert forbidden.isdisjoint(imported), f"{path.name} imports {imported & forbidden}"


def test_deterministic_csv(tmp_path: Path) -> None:
    """Repeated report writes are byte-identical."""
    observations = _observations_from_evaluator()
    fold_ic = compute_fold_factor_ic(observations).with_columns(
        [
            pl.lit("price").alias("factor_family"),
            pl.lit("ORIENTATION_REVERSAL_CANDIDATE").alias("orientation_class"),
            pl.lit(0.1).alias("selection_score"),
            pl.lit(1).alias("selection_rank"),
            pl.lit(0.2).alias("selection_metric_ic"),
            pl.lit(0.2).alias("selection_metric_rank_ic"),
            pl.lit("IS_POSITIVE_OOS_NEGATIVE").alias("is_oos_class"),
        ]
    )
    folds = fold_ic.select(
        [
            "manager",
            "exchange",
            "market",
            "timeframe",
            "year",
            "factor_name",
            "factor_version",
            "fold_id",
            "ic",
            "rows",
            "mean_return",
            "target_std",
            "positive_rate",
        ]
    )
    timeframes = aggregate_cross_timeframe(fold_ic)
    families = aggregate_families(fold_ic)
    alignment = verify_target_alignment(
        _evaluation_input(),
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    panel = PanelDiagnosticBundle(
        stability_all=fold_ic,
        folds=folds,
        alignment_audit=alignment,
        fold_count=5,
        alignment_ok=True,
        value_ok=True,
        selection_tested=10,
        selection_selected=1,
    )
    global_summary = build_global_summary(
        stability_all=fold_ic,
        folds=folds,
        timeframes=timeframes,
        families=families,
        alignment_audit=alignment,
        panel_results=[panel],
        methodology=detect_ic_methodology(),
    )
    first_dir = tmp_path / "a"
    second_dir = tmp_path / "b"
    first = write_factor_stability_reports(
        output_root=first_dir,
        stability_all=fold_ic,
        folds=folds,
        timeframes=timeframes,
        families=families,
        alignment_audit=alignment,
        global_summary=global_summary,
    )
    second = write_factor_stability_reports(
        output_root=second_dir,
        stability_all=fold_ic,
        folds=folds,
        timeframes=timeframes,
        families=families,
        alignment_audit=alignment,
        global_summary=global_summary,
    )
    for key in first:
        assert first[key].read_bytes() == second[key].read_bytes()
    assert first["all"].name == STABILITY_ALL_CSV_NAME
    assert first["folds"].name == FOLDS_CSV_NAME
    assert first["timeframes"].name == TIMEFRAMES_CSV_NAME
    assert first["families"].name == FAMILIES_CSV_NAME
    assert first["global"].name == GLOBAL_CSV_NAME
    assert first["alignment"].name == ALIGNMENT_AUDIT_CSV_NAME


def test_global_report_correctness() -> None:
    """Global summary answers critical questions explicitly."""
    frame = pl.DataFrame(
        {
            "manager": [_MANAGER] * 4,
            "exchange": [_EXCHANGE] * 4,
            "market": [_MARKET] * 4,
            "timeframe": [_TIMEFRAME] * 4,
            "year": [_YEAR] * 4,
            "factor_name": [_FACTOR] * 4,
            "factor_version": [_FACTOR_VERSION] * 4,
            "factor_family": ["price"] * 4,
            "fold_id": [1, 2, 3, 4],
            "ic": [-0.1, -0.2, -0.05, -0.15],
            "mean_return": [0.01, -0.01, 0.0, 0.02],
            "positive_rate": [0.5, 0.4, 0.45, 0.55],
            "orientation_class": ["ORIENTATION_REVERSAL_CANDIDATE"] * 4,
            "is_oos_class": ["IS_POSITIVE_OOS_NEGATIVE"] * 4,
            "selection_metric_rank_ic": [0.2] * 4,
            "selection_metric_ic": [0.2] * 4,
        }
    )
    timeframes = aggregate_cross_timeframe(frame)
    families = aggregate_families(frame)
    alignment = verify_target_alignment(
        _evaluation_input(),
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    panel = PanelDiagnosticBundle(
        stability_all=frame,
        folds=frame,
        alignment_audit=alignment,
        fold_count=5,
        alignment_ok=True,
        value_ok=True,
        selection_tested=20,
        selection_selected=2,
    )
    summary = build_global_summary(
        stability_all=frame,
        folds=frame,
        timeframes=timeframes,
        families=families,
        alignment_audit=alignment,
        panel_results=[panel],
        methodology=detect_ic_methodology(),
    )
    metrics = {row["metric"]: row["value"] for row in summary.iter_rows(named=True)}
    assert metrics["q1_future_return_1_aligned"] == "YES"
    assert metrics["q2_factor_values_preserved"] == "YES"
    assert metrics["q3_orientation_cause"] == "EVIDENCE FOR"
    assert metrics["q8_methodology_aligned"] == "YES"
    assert metrics["q9_selection_intensity_reconstructible"] == "YES"
    assert metrics["verdict"] == "PASS"
    assert metrics["q10_primary_conclusion"] in {
        "C. FACTOR_ORIENTATION_PROBLEM",
        "D. SELECTION_OVERFIT_SIGNAL",
        "H. NO_SINGLE_ROOT_CAUSE",
    }


def test_classify_cross_timeframe_helpers() -> None:
    """Cross-timeframe classifier covers single and mixed cases."""
    assert classify_cross_timeframe_stability(mean_ics=[-0.1])[0] == "SINGLE_TIMEFRAME"
    label, _ = classify_cross_timeframe_stability(mean_ics=[-0.1, 0.2])
    assert label == "CROSS_TIMEFRAME_UNSTABLE"


def test_production_immutability_hash_contract(tmp_path: Path) -> None:
    """Watched production parquet bytes remain unchanged after report writes."""
    ledger = (
        tmp_path / "purged_cv" / _MANAGER / _EXCHANGE / _MARKET / _TIMEFRAME / f"{_YEAR}.parquet"
    )
    ledger.parent.mkdir(parents=True, exist_ok=True)
    payload = b"immutable-ledger-bytes"
    ledger.write_bytes(payload)
    before = hashlib.sha256(ledger.read_bytes()).hexdigest()
    write_factor_stability_reports(
        output_root=tmp_path / "reports",
        stability_all=pl.DataFrame(),
        folds=pl.DataFrame(),
        timeframes=pl.DataFrame(),
        families=pl.DataFrame(),
        alignment_audit=pl.DataFrame(),
        global_summary=pl.DataFrame({"metric": ["verdict"], "value": ["PASS"]}),
    )
    after = hashlib.sha256(ledger.read_bytes()).hexdigest()
    assert before == after
    assert ledger.read_bytes() == payload
