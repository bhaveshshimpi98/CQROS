"""Unit tests for leakage-safe factor orientation policy."""

from __future__ import annotations

import ast
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.factor_selection import (
    FACTOR_ORIENTATION_POLICY,
    FACTOR_SELECTION_SCHEMA,
    ORIENTATION_SOURCE_METRIC,
    ORIENTATION_ZERO_IC_DIRECTION,
    SimpleFactorSelectionEngine,
    oriented_selection_ic,
    selected_direction_from_ic,
)
from cqros.factor_selection.schema import CANONICAL_COLUMN_ORDER
from cqros.purged_cv.engine import SimplePurgedCVEngine
from cqros.purged_cv.evaluation import PurgedCVEvaluator
from cqros.reporting.factor_orientation_diagnostic import (
    FactorOrientationReporter,
    build_orientation_summary,
)
from cqros.walk_forward.evaluation import WalkForwardEvaluator
from cqros.walk_forward.evaluation_input import (
    TARGET_COLUMN,
    assemble_walk_forward_input,
    require_orientation_metadata,
)
from cqros.walk_forward.exceptions import WalkForwardError

_TIMEFRAME = "1h"
_FACTOR_VERSION = "1.0.0"
_VALIDATION_TIME = 1_704_067_200_000
_FORBIDDEN_IMPORTS = (
    "cqros.alpha",
    "cqros.regime",
    "cqros.predictions",
    "cqros.signals",
    "cqros.ml",
)


def _validation_frame(
    *,
    names: list[str],
    ics: list[float],
    timeframe: str = _TIMEFRAME,
) -> pl.DataFrame:
    count = len(names)
    return pl.DataFrame(
        {
            "factor_name": names,
            "factor_version": [_FACTOR_VERSION] * count,
            "factor_category": ["price"] * count,
            "timeframe": [timeframe] * count,
            "validation_time": [_VALIDATION_TIME] * count,
            "information_coefficient": ics,
            "rank_information_coefficient": [abs(value) * 0.9 for value in ics],
            "ic_information_ratio": [abs(value) * 5.0 for value in ics],
            "ic_p_value": [0.01] * count,
            "ic_t_stat": [3.0] * count,
            "ic_decay": [0.5] * count,
            "turnover": [0.2] * count,
            "monotonicity_score": [0.5] * count,
            "quantile_spread": [0.05] * count,
            "observations": [200] * count,
            "status": ["PASS"] * count,
        }
    )


def test_discovered_selection_path_documents_abs_ic_without_direction() -> None:
    """Inspection evidence: ranking uses abs(IC); orientation is now explicit."""
    engine_path = Path("src/cqros/factor_selection/engine.py")
    text = engine_path.read_text(encoding="utf-8")
    assert "WEIGHT_ABS_IC" in text
    assert "information_coefficient" in text
    assert ".abs()" in text
    orientation_path = Path("src/cqros/factor_selection/orientation.py")
    assert orientation_path.exists()
    assert "FACTOR_ORIENTATION_POLICY" in orientation_path.read_text(encoding="utf-8")


def test_positive_ic_selects_positive_direction() -> None:
    """Positive selection IC maps to selected_direction = +1."""
    assert selected_direction_from_ic(0.08) == 1
    result = SimpleFactorSelectionEngine(top_n=1).build(
        _validation_frame(names=["pos"], ics=[0.08])
    )
    assert result["selected_direction"].to_list() == [1]
    assert result["selection_ic"].to_list() == [pytest.approx(0.08)]
    assert result["orientation_policy"].to_list() == [FACTOR_ORIENTATION_POLICY]


def test_negative_ic_selects_negative_direction() -> None:
    """Negative selection IC maps to selected_direction = -1."""
    assert selected_direction_from_ic(-0.08) == -1
    result = SimpleFactorSelectionEngine(top_n=1).build(
        _validation_frame(names=["neg"], ics=[-0.08])
    )
    assert result["selected_direction"].to_list() == [-1]
    assert oriented_selection_ic(-0.08, -1) == pytest.approx(0.08)


def test_zero_ic_deterministic_positive_direction() -> None:
    """Exact zero IC uses the locked +1 convention."""
    assert ORIENTATION_ZERO_IC_DIRECTION == 1
    assert selected_direction_from_ic(0.0) == 1
    result = SimpleFactorSelectionEngine(top_n=1).build(
        _validation_frame(names=["zero"], ics=[0.0])
    )
    assert result["selected_direction"].to_list() == [1]


def test_abs_ic_ranking_unchanged_with_mixed_signs() -> None:
    """Ranking remains abs(IC)-based; negative IC can outrank smaller positive IC."""
    result = SimpleFactorSelectionEngine(top_n=1).build(
        _validation_frame(names=["weak_pos", "strong_neg"], ics=[0.02, -0.20])
    )
    by_name = {row["factor_name"]: row for row in result.iter_rows(named=True)}
    assert by_name["strong_neg"]["selection_rank"] == 1
    assert by_name["strong_neg"]["selected"] is True
    assert by_name["strong_neg"]["selected_direction"] == -1
    assert by_name["weak_pos"]["selected"] is False
    assert by_name["weak_pos"]["selected_direction"] == 1


def test_raw_factor_values_unchanged_after_orientation_join() -> None:
    """Canonical factor_value remains raw after selection join."""
    from cqros.factors import FactorStatus
    from cqros.factors.schema import CANONICAL_COLUMN_ORDER as FACTOR_COLS
    from cqros.factors.schema import COLUMN_DTYPES as FACTOR_DTYPES
    from cqros.labels.schema import CANONICAL_COLUMN_ORDER as LABEL_COLS
    from cqros.labels.schema import COLUMN_DTYPES as LABEL_DTYPES

    open_times = [1_700_000_000_000, 1_700_003_600_000]
    selection = SimpleFactorSelectionEngine(top_n=1).build(
        _validation_frame(names=["momentum"], ics=[-0.12])
    )
    factors = pl.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "timeframe": [_TIMEFRAME, _TIMEFRAME],
            "open_time": open_times,
            "factor_name": ["momentum", "momentum"],
            "factor_version": [_FACTOR_VERSION, _FACTOR_VERSION],
            "factor_category": ["price", "price"],
            "factor_group": ["alpha", "alpha"],
            "factor_value": [1.5, -2.5],
            "lookback": [20, 20],
            "prediction_horizon": [1, 1],
            "enabled": [True, True],
            "status": [FactorStatus.ACTIVE.value, FactorStatus.ACTIVE.value],
        },
        schema=dict(FACTOR_DTYPES),
    ).select(list(FACTOR_COLS))
    labels = pl.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "timeframe": [_TIMEFRAME, _TIMEFRAME],
            "open_time": open_times,
            "future_return_1": [0.01, -0.02],
            "future_return_5": [0.0, 0.0],
            "future_return_10": [0.0, 0.0],
            "future_return_20": [0.0, 0.0],
            "direction_1": [1, -1],
            "direction_5": [0, 0],
            "direction_10": [0, 0],
            "direction_20": [0, 0],
        },
        schema=dict(LABEL_DTYPES),
    ).select(list(LABEL_COLS))
    joined = assemble_walk_forward_input(selection, factors, labels)
    assert joined["factor_value"].to_list() == [1.5, -2.5]
    assert joined["selected_direction"].to_list() == [-1, -1]
    oriented = (
        pl.Series(joined["factor_value"]) * pl.Series(joined["selected_direction"])
    ).to_list()
    assert oriented == [-1.5, 2.5]


def test_legacy_artifact_rejected_without_silent_default() -> None:
    """Pre-orientation Factor Selection artifacts require regeneration."""
    legacy = pl.DataFrame(
        {
            "factor_name": ["momentum"],
            "factor_version": [_FACTOR_VERSION],
            "timeframe": [_TIMEFRAME],
            "selected": [True],
        }
    )
    with pytest.raises(WalkForwardError, match="predates orientation policy"):
        require_orientation_metadata(legacy)


def test_walk_forward_inherits_direction_and_sign_inversion() -> None:
    """WF inherits selected_direction and reports raw vs oriented OOS IC."""
    open_times = [1_700_000_000_000 + index * 3_600_000 for index in range(12)]
    rows: list[dict[str, object]] = []
    for index, open_time in enumerate(open_times):
        rows.append(
            {
                "symbol": "BTCUSDT",
                "timeframe": _TIMEFRAME,
                "open_time": open_time,
                "factor_name": "momentum",
                "factor_version": _FACTOR_VERSION,
                "factor_value": float(index + 1),
                "selected": True,
                "selection_time": open_time,
                "selection_ic": -0.08,
                "selected_direction": -1,
                "orientation_policy": FACTOR_ORIENTATION_POLICY,
                TARGET_COLUMN: 0.01 * float(index + 1),
            }
        )
    evaluation_input = pl.DataFrame(rows)
    wf = WalkForwardEvaluator(train_window=4, test_window=2, step_size=2).evaluate(
        evaluation_input,
        manager="default",
        engine="simple",
        year=2026,
    )
    assert set(wf.factor_metrics["selected_direction"].to_list()) == {-1}
    assert set(wf.observations["selected_direction"].to_list()) == {-1}
    for row in wf.factor_metrics.iter_rows(named=True):
        if row["raw_oos_ic"] is None:
            continue
        assert row["oriented_oos_ic"] == pytest.approx(-float(row["raw_oos_ic"]), abs=1e-12)
        assert row["oos_ic"] == pytest.approx(row["oriented_oos_ic"], abs=1e-12)
        assert row["orientation_policy"] == FACTOR_ORIENTATION_POLICY


def test_purged_cv_inherits_direction_without_recomputing() -> None:
    """Purged-CV consumes inherited orientation and preserves raw OOS IC."""
    open_times = [1_700_000_000_000 + index * 3_600_000 for index in range(40)]
    walk_rows: list[dict[str, object]] = []
    for index in range(20):
        test_start = open_times[index]
        walk_rows.append(
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
                "train_score": 0.01,
                "test_score": 0.01,
                "overfit_gap": 0.0,
                "status": "PASS",
            }
        )
    walk_forward = pl.DataFrame(walk_rows)
    purged_cv = SimplePurgedCVEngine(n_folds=5, purge_size=2, embargo_size=1).build(walk_forward)
    eval_rows: list[dict[str, object]] = []
    for index, open_time in enumerate(open_times):
        eval_rows.append(
            {
                "symbol": "BTCUSDT",
                "timeframe": _TIMEFRAME,
                "open_time": open_time,
                "factor_name": "momentum",
                "factor_version": _FACTOR_VERSION,
                "factor_value": float(index + 1),
                "selected": True,
                "selection_time": open_time,
                "selection_ic": -0.08,
                "selected_direction": -1,
                "orientation_policy": FACTOR_ORIENTATION_POLICY,
                TARGET_COLUMN: 0.01 * float(index + 1),
            }
        )
    artifacts = PurgedCVEvaluator().evaluate(
        purged_cv,
        walk_forward,
        manager="default",
        engine="simple",
        exchange="binance",
        market="usdt_perpetual",
        year=2026,
        evaluation_input=pl.DataFrame(eval_rows),
    )
    assert "raw_oos_ic" in artifacts.factor_metrics.columns
    assert "oriented_oos_ic" in artifacts.factor_metrics.columns
    assert set(artifacts.factor_metrics["selected_direction"].drop_nulls().to_list()) == {-1}
    for row in artifacts.factor_metrics.iter_rows(named=True):
        if row["raw_oos_ic"] is None:
            continue
        assert row["oriented_oos_ic"] == pytest.approx(-float(row["raw_oos_ic"]), abs=1e-12)


def test_orientation_source_metric_and_policy_version() -> None:
    """Orientation policy metadata is versioned and selection-time based."""
    assert FACTOR_ORIENTATION_POLICY == "signed_ic_v1"
    assert ORIENTATION_SOURCE_METRIC == "information_coefficient"
    assert set(CANONICAL_COLUMN_ORDER).issuperset(
        {"selection_ic", "selected_direction", "orientation_policy"}
    )
    assert FACTOR_SELECTION_SCHEMA["selected_direction"] == pl.Int8


def test_cross_timeframe_isolation_for_orientation() -> None:
    """Directions are computed independently per timeframe row."""
    frame = pl.concat(
        [
            _validation_frame(names=["a"], ics=[0.10], timeframe="1h"),
            _validation_frame(names=["a"], ics=[-0.10], timeframe="4h"),
        ]
    )
    result = SimpleFactorSelectionEngine(top_n=1).build(frame)
    by_tf = {row["timeframe"]: row for row in result.iter_rows(named=True)}
    assert by_tf["1h"]["selected_direction"] == 1
    assert by_tf["4h"]["selected_direction"] == -1


def test_deterministic_orientation_output() -> None:
    """Repeated orientation-aware selection is byte-identical."""
    frame = _validation_frame(names=["a", "b"], ics=[0.05, -0.12])
    engine = SimpleFactorSelectionEngine(top_n=2)
    first = engine.build(frame)
    second = engine.build(frame)
    assert_frame_equal(first, second)


def test_orientation_modules_forbid_upward_imports() -> None:
    """Changed modules must not import Alpha/Regime/Predictions/Signals/ml."""
    roots = [
        Path("src/cqros/factor_selection"),
        Path("src/cqros/walk_forward"),
        Path("src/cqros/purged_cv"),
        Path("src/cqros/reporting/factor_orientation_diagnostic.py"),
    ]
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        else:
            files.extend(root.rglob("*.py"))
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                names = [node.module]
            else:
                continue
            for name in names:
                assert not any(
                    name == forbidden or name.startswith(f"{forbidden}.")
                    for forbidden in _FORBIDDEN_IMPORTS
                ), f"{path} imports {name}"


def test_orientation_summary_reporter_is_deterministic(tmp_path: Path) -> None:
    """Orientation summary CSV writing is deterministic."""
    metrics = pl.DataFrame(
        {
            "timeframe": ["1h", "1h", "4h"],
            "year": [2026, 2026, 2026],
            "factor_name": ["a", "b", "c"],
            "factor_version": ["v1", "v1", "v1"],
            "selected_direction": [1, -1, -1],
            "selection_ic": [0.1, -0.2, -0.05],
            "raw_oos_ic": [-0.1, 0.2, 0.03],
            "oriented_oos_ic": [-0.1, -0.2, -0.03],
        }
    )
    summary = build_orientation_summary(metrics)
    assert summary.filter(pl.col("timeframe") == "1h")["selected_factor_count"].item() == 2
    reporter = FactorOrientationReporter(tmp_path / "walk_forward")
    first = reporter.write_reports(summary=summary)
    second = reporter.write_reports(summary=build_orientation_summary(metrics))
    assert first["summary"].read_bytes() == second["summary"].read_bytes()
