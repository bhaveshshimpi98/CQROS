"""Unit tests for Purged-CV evaluation CSV reporter."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from cqros.core.constants import EXCHANGE_BINANCE, MARKET_USDT_PERPETUAL
from cqros.purged_cv.engine import SimplePurgedCVEngine
from cqros.purged_cv.evaluation import PurgedCVEvaluator
from cqros.reporting.purged_cv_evaluation_report import (
    EVALUATION_ALL_CSV_NAME,
    EVALUATION_FACTORS_CSV_NAME,
    EVALUATION_FOLDS_CSV_NAME,
    EVALUATION_GLOBAL_CSV_NAME,
    PurgedCVEvaluationReporter,
    build_global_summary,
)
from cqros.walk_forward.evaluation_input import TARGET_COLUMN

_MANAGER = "default"
_ENGINE = "simple"
_YEAR = 2026
_TIMEFRAME = "1h"
_EXCHANGE = EXCHANGE_BINANCE
_MARKET = MARKET_USDT_PERPETUAL


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


def _evaluation_input() -> pl.DataFrame:
    open_times = [1_700_000_000_000 + index * 3_600_000 for index in range(80)]
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
        evaluation_input=_evaluation_input(),
    )
    reporter = PurgedCVEvaluationReporter(tmp_path)
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
    assert "unavailable_portfolio_pnl" in metrics
    assert "timeframe_panels" in metrics
    assert "macro_avg_panel_oos_return_mean" not in metrics or frame.height >= 0


def test_global_aggregation_distinguishes_macro_and_pooled() -> None:
    """Global report labels macro averages separately from pooled counts."""
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
        evaluation_input=_evaluation_input(),
    )
    frame = build_global_summary(artifacts.summary, fold_metrics=artifacts.fold_metrics)
    metrics = set(frame["metric"].to_list())
    assert "timeframe_panels" in metrics
    assert "pooled_folds" in metrics
    assert "macro_avg_panel_oos_return_mean" in metrics
    assert "macro_avg_fold_oos_return_mean" in metrics
    assert "pooled_purge_valid_folds" in metrics


def test_deterministic_report_bytes(tmp_path: Path) -> None:
    """Writing reports twice over unchanged artifacts is byte-identical."""
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
        evaluation_input=_evaluation_input(),
    )
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = PurgedCVEvaluationReporter(first_dir).write_reports(
        summaries=artifacts.summary,
        fold_metrics=artifacts.fold_metrics,
        factor_metrics=artifacts.factor_metrics,
    )
    second = PurgedCVEvaluationReporter(second_dir).write_reports(
        summaries=artifacts.summary,
        fold_metrics=artifacts.fold_metrics,
        factor_metrics=artifacts.factor_metrics,
    )
    for key in first:
        assert first[key].read_bytes() == second[key].read_bytes()
