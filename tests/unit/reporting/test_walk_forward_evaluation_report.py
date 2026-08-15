"""Unit tests for Walk-Forward evaluation CSV reporter."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from cqros.reporting.walk_forward_evaluation_report import (
    EVALUATION_ALL_CSV_NAME,
    EVALUATION_FACTORS_CSV_NAME,
    EVALUATION_FOLDS_CSV_NAME,
    EVALUATION_GLOBAL_CSV_NAME,
    WalkForwardEvaluationReporter,
    build_global_summary,
)
from cqros.walk_forward.evaluation import WalkForwardEvaluator
from cqros.walk_forward.evaluation_input import TARGET_COLUMN

_MANAGER = "default"
_ENGINE = "simple"
_YEAR = 2026
_TIMEFRAME = "1h"


def _evaluation_input() -> pl.DataFrame:
    open_times = [1_700_000_000_000 + index * 3_600_000 for index in range(8)]
    rows: list[dict[str, object]] = []
    for open_time in open_times:
        rows.append(
            {
                "symbol": "BTCUSDT",
                "timeframe": _TIMEFRAME,
                "open_time": open_time,
                "factor_name": "momentum",
                "factor_version": "1.0.0",
                "factor_value": 1.0,
                "selected": True,
                "selection_time": open_time,
                "selection_ic": 0.08,
                "selected_direction": 1,
                "orientation_policy": "signed_ic_v1",
                TARGET_COLUMN: 0.01,
            }
        )
    return pl.DataFrame(rows)


def test_reporter_writes_four_csv_files(tmp_path: Path) -> None:
    """Reporter emits all/folds/factors/global CSVs."""
    artifacts = WalkForwardEvaluator(
        train_window=4,
        test_window=2,
        step_size=2,
    ).evaluate(
        _evaluation_input(),
        manager=_MANAGER,
        engine=_ENGINE,
        year=_YEAR,
    )
    reporter = WalkForwardEvaluationReporter(tmp_path)
    paths = reporter.write_reports(
        summaries=artifacts.summary,
        fold_metrics=artifacts.fold_metrics,
        factor_metrics=artifacts.factor_metrics,
    )
    assert paths["all"].name == EVALUATION_ALL_CSV_NAME
    assert paths["folds"].name == EVALUATION_FOLDS_CSV_NAME
    assert paths["factors"].name == EVALUATION_FACTORS_CSV_NAME
    assert paths["global"].name == EVALUATION_GLOBAL_CSV_NAME
    for path in paths.values():
        assert path.is_file()
        assert path.stat().st_size > 0


def test_global_summary_documents_unavailable_metrics() -> None:
    """Global summary includes unavailable Sharpe/drawdown/prediction notes."""
    frame = build_global_summary(pl.DataFrame())
    metrics = set(frame["metric"].to_list())
    assert "unavailable_oos_sharpe" in metrics
    assert "unavailable_oos_max_drawdown" in metrics
    assert "unavailable_prediction" in metrics
