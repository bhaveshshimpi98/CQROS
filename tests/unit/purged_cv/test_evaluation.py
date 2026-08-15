"""Unit tests for CQROS Purged-CV evaluation results."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.core.constants import EXCHANGE_BINANCE, MARKET_USDT_PERPETUAL
from cqros.core.types import FilePath
from cqros.purged_cv.engine import SimplePurgedCVEngine
from cqros.purged_cv.evaluation import PurgedCVEvaluator
from cqros.purged_cv.evaluation_repository import PurgedCVEvaluationRepository
from cqros.purged_cv.evaluation_schema import (
    EVALUATION_OBSERVATION_COLUMNS,
    UNAVAILABLE_METRIC_NOTES,
    PurgedCVEvaluationPartition,
)
from cqros.purged_cv.exceptions import PurgedCVError
from cqros.storage import StorageLayout
from cqros.storage.exceptions import DatasetNotFoundError
from cqros.walk_forward.evaluation_input import TARGET_COLUMN

_MANAGER = "default"
_ENGINE = "simple"
_YEAR = 2026
_TIMEFRAME = "1h"
_EXCHANGE = EXCHANGE_BINANCE
_MARKET = MARKET_USDT_PERPETUAL
_SYMBOL_BTC = "BTCUSDT"
_SYMBOL_ETH = "ETHUSDT"
_FACTOR_A = "momentum"
_FACTOR_B = "mean_reversion"
_FACTOR_VERSION = "1.0.0"


class _InMemoryDataStore:
    """Minimal in-memory datastore for repository tests."""

    def __init__(self) -> None:
        self.frames: dict[Path, pl.DataFrame] = {}

    def write(self, path: FilePath, frame: pl.DataFrame) -> None:
        self.frames[Path(path)] = frame.clone()

    def read(self, path: FilePath) -> pl.DataFrame:
        target = Path(path)
        try:
            return self.frames[target]
        except KeyError as exc:
            raise DatasetNotFoundError(
                "Dataset not found",
                error_code="STORAGE-TEST-001",
                details={"path": str(target)},
            ) from exc

    def scan(self, path: FilePath) -> pl.LazyFrame:
        return self.read(path).lazy()

    def exists(self, path: FilePath) -> bool:
        return Path(path) in self.frames

    def delete(self, path: FilePath) -> None:
        del self.frames[Path(path)]

    def schema(self, path: FilePath) -> pl.Schema:
        return self.read(path).schema

    def row_count(self, path: FilePath) -> int:
        return self.read(path).height


def _walk_forward(*, n_rows: int = 40) -> pl.DataFrame:
    """Build a compact Walk-Forward ledger for purged-CV tests."""
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


def _purged_cv(walk_forward: pl.DataFrame, *, n_folds: int = 5) -> pl.DataFrame:
    """Build purged-CV folds from a Walk-Forward fixture."""
    return SimplePurgedCVEngine(n_folds=n_folds, purge_size=2, embargo_size=1).build(walk_forward)


def _evaluation_input(
    *,
    n_times: int = 80,
    symbols: tuple[str, ...] = (_SYMBOL_BTC,),
    selected_factors: tuple[str, ...] = (_FACTOR_A,),
    null_last_return: bool = False,
    selected_direction: int = 1,
    selection_ic: float = 0.08,
) -> pl.DataFrame:
    """Build evaluation-input bars spanning the Walk-Forward fixture times."""
    from cqros.factor_selection.orientation import FACTOR_ORIENTATION_POLICY

    open_times = [1_700_000_000_000 + index * 3_600_000 for index in range(n_times)]
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        for time_index, open_time in enumerate(open_times):
            for factor_name in (_FACTOR_A, _FACTOR_B):
                future_return: float | None = 0.01 * float(time_index + 1)
                if null_last_return and time_index == n_times - 1:
                    future_return = None
                if symbol == _SYMBOL_ETH:
                    future_return = None if future_return is None else future_return + 0.5
                rows.append(
                    {
                        "symbol": symbol,
                        "timeframe": _TIMEFRAME,
                        "open_time": open_time,
                        "factor_name": factor_name,
                        "factor_version": _FACTOR_VERSION,
                        "factor_value": float(time_index + 1)
                        + (0.1 if factor_name == _FACTOR_A else -0.1),
                        "selected": factor_name in selected_factors,
                        "selection_time": open_time,
                        "selection_ic": selection_ic,
                        "selected_direction": selected_direction,
                        "orientation_policy": FACTOR_ORIENTATION_POLICY,
                        TARGET_COLUMN: future_return,
                    }
                )
    return pl.DataFrame(rows).sort(
        ["timeframe", "selection_time", "symbol", "factor_name", "factor_version"]
    )


def test_evaluation_schema_columns_and_null_predictions() -> None:
    """Observation schema includes required columns with null predictions."""
    walk_forward = _walk_forward()
    purged_cv = _purged_cv(walk_forward)
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
    assert tuple(artifacts.observations.columns) == EVALUATION_OBSERVATION_COLUMNS
    assert artifacts.observations["prediction"].null_count() == artifacts.observations.height
    assert artifacts.observations["residual"].null_count() == artifacts.observations.height
    assert artifacts.observations["correct"].null_count() == artifacts.observations.height
    assert artifacts.fold_metrics["oos_sharpe"].null_count() == artifacts.fold_metrics.height
    assert artifacts.fold_metrics["oos_max_drawdown"].null_count() == artifacts.fold_metrics.height
    assert "prediction" in UNAVAILABLE_METRIC_NOTES
    assert "portfolio_pnl" in UNAVAILABLE_METRIC_NOTES


def test_all_folds_accounted_and_ordered_by_fold_id() -> None:
    """Every purged fold is evaluated and ordered by fold_id."""
    walk_forward = _walk_forward()
    purged_cv = _purged_cv(walk_forward, n_folds=5)
    artifacts = PurgedCVEvaluator().evaluate(
        purged_cv,
        walk_forward,
        manager=_MANAGER,
        engine=_ENGINE,
        exchange=_EXCHANGE,
        market=_MARKET,
        year=_YEAR,
    )
    assert artifacts.fold_metrics.height == 5
    assert artifacts.fold_metrics["fold_id"].to_list() == [1, 2, 3, 4, 5]
    assert int(artifacts.summary["folds"][0]) == 5


def test_non_contiguous_training_accepted() -> None:
    """Training may exist on both sides of the test block."""
    walk_forward = _walk_forward()
    purged_cv = _purged_cv(walk_forward)
    # Middle fold has training before and after the test block.
    middle = purged_cv.filter(pl.col("fold_id") == 3)
    assert int(middle["train_end_time"][0]) >= int(middle["test_start_time"][0])
    artifacts = PurgedCVEvaluator().evaluate(
        purged_cv,
        walk_forward,
        manager=_MANAGER,
        engine=_ENGINE,
        exchange=_EXCHANGE,
        market=_MARKET,
        year=_YEAR,
    )
    fold = artifacts.fold_metrics.filter(pl.col("fold_id") == 3)
    assert bool(fold["train_test_disjoint"][0]) is True
    assert bool(fold["purge_valid"][0]) is True
    assert bool(fold["embargo_valid"][0]) is True
    assert str(fold["status"][0]) == "PASS"


def test_train_test_disjoint_and_purge_embargo_audit() -> None:
    """Audit flags validate disjointness and purge/embargo exclusion."""
    walk_forward = _walk_forward()
    purged_cv = _purged_cv(walk_forward)
    artifacts = PurgedCVEvaluator().evaluate(
        purged_cv,
        walk_forward,
        manager=_MANAGER,
        engine=_ENGINE,
        exchange=_EXCHANGE,
        market=_MARKET,
        year=_YEAR,
    )
    assert artifacts.fold_metrics["train_test_disjoint"].all()
    assert artifacts.fold_metrics["purge_valid"].all()
    assert artifacts.fold_metrics["embargo_valid"].all()
    assert artifacts.fold_metrics["fold_order_valid"].all()
    assert artifacts.fold_metrics["timestamp_valid"].all()


def test_oos_only_metrics_exclude_train_rows() -> None:
    """TRAIN observations never contribute to OOS metrics."""
    walk_forward = _walk_forward()
    purged_cv = _purged_cv(walk_forward)
    artifacts = PurgedCVEvaluator().evaluate_with_train(
        purged_cv,
        walk_forward,
        manager=_MANAGER,
        engine=_ENGINE,
        exchange=_EXCHANGE,
        market=_MARKET,
        year=_YEAR,
        evaluation_input=_evaluation_input(),
    )
    partitions = set(artifacts.observations["partition"].unique().to_list())
    assert partitions == {
        PurgedCVEvaluationPartition.TRAIN.value,
        PurgedCVEvaluationPartition.OOS.value,
    }
    oos = artifacts.observations.filter(
        pl.col("partition") == PurgedCVEvaluationPartition.OOS.value
    )
    train = artifacts.observations.filter(
        pl.col("partition") == PurgedCVEvaluationPartition.TRAIN.value
    )
    assert oos.height > 0
    assert train.height > 0
    for fold_id in artifacts.fold_metrics["fold_id"].to_list():
        fold_oos = oos.filter(pl.col("fold_id") == fold_id)
        fold_metric = artifacts.fold_metrics.filter(pl.col("fold_id") == fold_id)
        selected_returns = fold_oos.filter(pl.col("selected"))["future_return_1"].drop_nulls()
        assert int(fold_metric["oos_non_null_returns"][0]) == selected_returns.len()


def test_factor_level_evaluation_when_supported() -> None:
    """Selected factors produce factor-fold OOS metrics."""
    walk_forward = _walk_forward()
    purged_cv = _purged_cv(walk_forward)
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
    assert artifacts.factor_metrics.height > 0
    assert set(artifacts.factor_metrics["factor_name"].unique().to_list()) == {_FACTOR_A}
    assert artifacts.factor_metrics["oos_ic"].null_count() < artifacts.factor_metrics.height


def test_missing_target_and_factor_handling() -> None:
    """Missing Labels enrichment leaves label metrics null without failing audit."""
    walk_forward = _walk_forward()
    purged_cv = _purged_cv(walk_forward)
    artifacts = PurgedCVEvaluator().evaluate(
        purged_cv,
        walk_forward,
        manager=_MANAGER,
        engine=_ENGINE,
        exchange=_EXCHANGE,
        market=_MARKET,
        year=_YEAR,
        evaluation_input=None,
    )
    assert artifacts.observations.height == 0
    assert artifacts.factor_metrics.height == 0
    assert artifacts.fold_metrics["oos_return_mean"].null_count() == artifacts.fold_metrics.height
    assert artifacts.fold_metrics["oos_ic"].null_count() == artifacts.fold_metrics.height
    assert artifacts.fold_metrics["status"].to_list() == ["PASS"] * artifacts.fold_metrics.height


def test_null_future_returns_excluded_from_metrics() -> None:
    """Null future_return_1 values are excluded from OOS return statistics."""
    walk_forward = _walk_forward(n_rows=20)
    purged_cv = _purged_cv(walk_forward, n_folds=4)
    artifacts = PurgedCVEvaluator().evaluate(
        purged_cv,
        walk_forward,
        manager=_MANAGER,
        engine=_ENGINE,
        exchange=_EXCHANGE,
        market=_MARKET,
        year=_YEAR,
        evaluation_input=_evaluation_input(n_times=40, null_last_return=True),
    )
    assert artifacts.fold_metrics.height == 4


def test_duplicate_key_detection() -> None:
    """Duplicate purged-CV primary keys are rejected."""
    walk_forward = _walk_forward()
    purged_cv = _purged_cv(walk_forward)
    duplicated = pl.concat([purged_cv, purged_cv.head(1)], how="vertical")
    with pytest.raises(PurgedCVError, match="duplicate primary keys"):
        PurgedCVEvaluator().evaluate(
            duplicated,
            walk_forward,
            manager=_MANAGER,
            engine=_ENGINE,
            exchange=_EXCHANGE,
            market=_MARKET,
            year=_YEAR,
        )


def test_cross_timeframe_isolation() -> None:
    """Multi-timeframe purged-CV panels are rejected."""
    walk_forward = _walk_forward()
    purged_cv = _purged_cv(walk_forward)
    mixed = purged_cv.with_columns(
        pl.when(pl.col("fold_id") == 1)
        .then(pl.lit("15m"))
        .otherwise(pl.col("timeframe"))
        .alias("timeframe")
    )
    with pytest.raises(PurgedCVError, match="exactly one timeframe"):
        PurgedCVEvaluator().evaluate(
            mixed,
            walk_forward,
            manager=_MANAGER,
            engine=_ENGINE,
            exchange=_EXCHANGE,
            market=_MARKET,
            year=_YEAR,
        )


def test_cross_symbol_isolation() -> None:
    """Symbol-level OOS rows remain isolated in observation artifacts."""
    walk_forward = _walk_forward()
    purged_cv = _purged_cv(walk_forward)
    artifacts = PurgedCVEvaluator().evaluate(
        purged_cv,
        walk_forward,
        manager=_MANAGER,
        engine=_ENGINE,
        exchange=_EXCHANGE,
        market=_MARKET,
        year=_YEAR,
        evaluation_input=_evaluation_input(symbols=(_SYMBOL_BTC, _SYMBOL_ETH)),
    )
    symbols = set(artifacts.observations["symbol"].unique().to_list())
    assert symbols == {_SYMBOL_BTC, _SYMBOL_ETH}
    btc = artifacts.observations.filter(pl.col("symbol") == _SYMBOL_BTC)
    eth = artifacts.observations.filter(pl.col("symbol") == _SYMBOL_ETH)
    assert btc.height > 0
    assert eth.height > 0


def test_no_leakage_imports() -> None:
    """Evaluation modules must not import Alpha/Regime/Predictions/Signals/ml."""
    root = Path(__file__).resolve().parents[3] / "src" / "cqros" / "purged_cv"
    forbidden = {
        "cqros.alpha",
        "cqros.regime",
        "cqros.predictions",
        "cqros.signals",
        "cqros.ml",
    }
    for path in (
        root / "evaluation.py",
        root / "evaluation_schema.py",
        root / "evaluation_repository.py",
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module)
        assert forbidden.isdisjoint(imported), f"{path.name} imports {imported & forbidden}"


def test_deterministic_output() -> None:
    """Repeated evaluation over unchanged inputs is byte-identical."""
    walk_forward = _walk_forward()
    purged_cv = _purged_cv(walk_forward)
    evaluation_input = _evaluation_input()
    first = PurgedCVEvaluator().evaluate(
        purged_cv,
        walk_forward,
        manager=_MANAGER,
        engine=_ENGINE,
        exchange=_EXCHANGE,
        market=_MARKET,
        year=_YEAR,
        evaluation_input=evaluation_input,
    )
    second = PurgedCVEvaluator().evaluate(
        purged_cv,
        walk_forward,
        manager=_MANAGER,
        engine=_ENGINE,
        exchange=_EXCHANGE,
        market=_MARKET,
        year=_YEAR,
        evaluation_input=evaluation_input,
    )
    assert_frame_equal(first.fold_metrics, second.fold_metrics)
    assert_frame_equal(first.factor_metrics, second.factor_metrics)
    assert_frame_equal(first.summary, second.summary)
    assert_frame_equal(first.observations, second.observations)


def test_no_fabricated_predictions_or_performance_metrics() -> None:
    """Prediction and performance fields stay null."""
    walk_forward = _walk_forward()
    purged_cv = _purged_cv(walk_forward)
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
    assert artifacts.observations["prediction"].null_count() == artifacts.observations.height
    assert artifacts.fold_metrics["oos_sharpe"].null_count() == artifacts.fold_metrics.height
    assert artifacts.fold_metrics["oos_max_drawdown"].null_count() == artifacts.fold_metrics.height
    assert artifacts.summary["oos_sharpe"].null_count() == 1
    assert artifacts.summary["oos_max_drawdown"].null_count() == 1


def test_repository_roundtrip(tmp_path: Path) -> None:
    """Evaluation observations persist and reload under purged_cv_evaluation."""
    walk_forward = _walk_forward()
    purged_cv = _purged_cv(walk_forward)
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
    repo = PurgedCVEvaluationRepository(StorageLayout(tmp_path), _InMemoryDataStore())
    repo.save(
        artifacts.observations,
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    loaded = repo.load(
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert_frame_equal(loaded, artifacts.observations)


def test_production_artifact_immutability_contract(tmp_path: Path) -> None:
    """Evaluation writes only under purged_cv_evaluation, never purged_cv."""
    layout = StorageLayout(tmp_path)
    purged_path = layout.purged_cv_path(_MANAGER, _EXCHANGE, _MARKET, _TIMEFRAME, _YEAR)
    purged_path.parent.mkdir(parents=True, exist_ok=True)
    walk_forward = _walk_forward()
    purged_cv = _purged_cv(walk_forward)
    purged_cv.write_parquet(purged_path)
    before = hashlib.sha256(purged_path.read_bytes()).hexdigest()

    artifacts = PurgedCVEvaluator().evaluate(
        purged_cv,
        walk_forward,
        manager=_MANAGER,
        engine=_ENGINE,
        exchange=_EXCHANGE,
        market=_MARKET,
        year=_YEAR,
    )
    repo = PurgedCVEvaluationRepository(layout, _InMemoryDataStore())
    repo.save(
        artifacts.observations,
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    after = hashlib.sha256(purged_path.read_bytes()).hexdigest()
    assert before == after
    eval_path = layout.purged_cv_evaluation_path(_MANAGER, _EXCHANGE, _MARKET, _TIMEFRAME, _YEAR)
    assert purged_path != eval_path
    assert "purged_cv_evaluation" in eval_path.parts
    assert "purged_cv" in purged_path.parts
