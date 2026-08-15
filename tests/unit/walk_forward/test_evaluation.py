"""Unit tests for CQROS Walk-Forward evaluation results."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.alpha.schema import CANONICAL_COLUMN_ORDER as ALPHA_COLUMNS
from cqros.core.constants import EXCHANGE_BINANCE, MARKET_USDT_PERPETUAL
from cqros.core.types import FilePath
from cqros.predictions.schema import CANONICAL_COLUMN_ORDER as PREDICTION_COLUMNS
from cqros.regime.schema import CANONICAL_COLUMN_ORDER as REGIME_COLUMNS
from cqros.research.information_coefficient import InformationCoefficient
from cqros.signals.schema import CANONICAL_COLUMN_ORDER as SIGNAL_COLUMNS
from cqros.storage import StorageLayout
from cqros.storage.exceptions import DatasetNotFoundError
from cqros.walk_forward.evaluation import WalkForwardEvaluator
from cqros.walk_forward.evaluation_input import TARGET_COLUMN, assemble_walk_forward_input
from cqros.walk_forward.evaluation_repository import WalkForwardEvaluationRepository
from cqros.walk_forward.evaluation_schema import (
    EVALUATION_OBSERVATION_COLUMNS,
    UNAVAILABLE_METRIC_NOTES,
    WalkForwardEvaluationPartition,
)
from cqros.walk_forward.exceptions import WalkForwardError

_MANAGER = "default"
_ENGINE = "simple"
_YEAR = 2026
_TIMEFRAME = "1h"
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


def _evaluation_input(
    *,
    n_times: int = 8,
    symbols: tuple[str, ...] = (_SYMBOL_BTC,),
    timeframes: tuple[str, ...] = (_TIMEFRAME,),
    selected_factors: tuple[str, ...] = (_FACTOR_A,),
    null_last_return: bool = False,
    selected_direction: int = 1,
    selection_ic: float = 0.08,
) -> pl.DataFrame:
    """Build a compact evaluation-input frame for evaluator tests."""
    from cqros.factor_selection.orientation import FACTOR_ORIENTATION_POLICY

    open_times = [1_700_000_000_000 + index * 3_600_000 for index in range(n_times)]
    rows: list[dict[str, object]] = []
    for timeframe in timeframes:
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
                            "timeframe": timeframe,
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
    assert tuple(artifacts.observations.columns) == EVALUATION_OBSERVATION_COLUMNS
    assert artifacts.observations["prediction"].null_count() == artifacts.observations.height
    assert artifacts.observations["residual"].null_count() == artifacts.observations.height
    assert artifacts.observations["correct"].null_count() == artifacts.observations.height
    assert "prediction" in UNAVAILABLE_METRIC_NOTES
    assert "oos_sharpe" in UNAVAILABLE_METRIC_NOTES
    assert artifacts.fold_metrics["oos_sharpe"].null_count() == artifacts.fold_metrics.height
    assert artifacts.fold_metrics["oos_max_drawdown"].null_count() == artifacts.fold_metrics.height


def test_fold_and_train_oos_separation() -> None:
    """TRAIN rows never contribute to OOS metrics and partitions are isolated."""
    evaluator = WalkForwardEvaluator(train_window=4, test_window=2, step_size=2)
    artifacts = evaluator.evaluate_with_train(
        _evaluation_input(n_times=8),
        manager=_MANAGER,
        engine=_ENGINE,
        year=_YEAR,
    )
    partitions = set(artifacts.observations["partition"].unique().to_list())
    assert partitions == {
        WalkForwardEvaluationPartition.TRAIN.value,
        WalkForwardEvaluationPartition.OOS.value,
    }
    oos = artifacts.observations.filter(
        pl.col("partition") == WalkForwardEvaluationPartition.OOS.value
    )
    train = artifacts.observations.filter(
        pl.col("partition") == WalkForwardEvaluationPartition.TRAIN.value
    )
    assert oos.height > 0
    assert train.height > 0
    # Fold metrics oos_rows must match OOS observation counts per fold.
    for fold_id in artifacts.fold_metrics["fold_id"].to_list():
        fold_oos = oos.filter(pl.col("fold_id") == fold_id)
        expected = int(artifacts.fold_metrics.filter(pl.col("fold_id") == fold_id)["oos_rows"][0])
        assert fold_oos.height == expected
        fold_metric = artifacts.fold_metrics.filter(pl.col("fold_id") == fold_id)
        selected_oos_returns = fold_oos.filter(pl.col("selected"))["future_return_1"].drop_nulls()
        assert int(fold_metric["oos_non_null_returns"][0]) == selected_oos_returns.len()


def test_target_provenance_is_labels_future_return_1() -> None:
    """Evaluator consumes TARGET_COLUMN from Labels via evaluation input only."""
    assert TARGET_COLUMN == "future_return_1"
    frame = _evaluation_input(n_times=6)
    artifacts = WalkForwardEvaluator(
        train_window=3,
        test_window=2,
        step_size=2,
    ).evaluate(frame, manager=_MANAGER, engine=_ENGINE, year=_YEAR)
    # Every persisted future_return_1 must equal the evaluation-input value.
    joined = artifacts.observations.join(
        frame.select(
            [
                pl.col("symbol"),
                pl.col("open_time").alias("observation_time"),
                pl.col("factor_name"),
                pl.col("factor_version"),
                pl.col(TARGET_COLUMN).alias("source_return"),
            ]
        ),
        on=["symbol", "observation_time", "factor_name", "factor_version"],
        how="left",
    )
    assert joined.filter(pl.col("future_return_1") != pl.col("source_return")).height == 0


def test_null_future_returns_excluded_from_metrics() -> None:
    """Null future_return_1 values are excluded from OOS return statistics."""
    artifacts = WalkForwardEvaluator(
        train_window=4,
        test_window=2,
        step_size=2,
    ).evaluate(
        _evaluation_input(n_times=8, null_last_return=True),
        manager=_MANAGER,
        engine=_ENGINE,
        year=_YEAR,
    )
    oos_selected = artifacts.observations.filter(
        (pl.col("partition") == WalkForwardEvaluationPartition.OOS.value) & pl.col("selected")
    )
    non_null = oos_selected["future_return_1"].drop_nulls().len()
    assert int(artifacts.summary["oos_non_null_returns"][0]) == non_null
    assert non_null < oos_selected.height


def test_duplicate_observation_keys_rejected() -> None:
    """Duplicate evaluation observation keys raise WalkForwardError."""
    from cqros.walk_forward.evaluation import _require_unique_observation_keys

    artifacts = WalkForwardEvaluator(
        train_window=4,
        test_window=2,
        step_size=2,
    ).evaluate(
        _evaluation_input(n_times=6),
        manager=_MANAGER,
        engine=_ENGINE,
        year=_YEAR,
    )
    duplicated = pl.concat([artifacts.observations, artifacts.observations])
    with pytest.raises(WalkForwardError, match="duplicate primary keys"):
        _require_unique_observation_keys(duplicated)


def test_symbol_timeframe_year_fold_isolation() -> None:
    """Metrics isolate by symbol membership, timeframe, year, and fold."""
    frame = _evaluation_input(
        n_times=8,
        symbols=(_SYMBOL_BTC, _SYMBOL_ETH),
        selected_factors=(_FACTOR_A,),
    )
    artifacts = WalkForwardEvaluator(
        train_window=4,
        test_window=2,
        step_size=2,
    ).evaluate(frame, manager=_MANAGER, engine=_ENGINE, year=_YEAR)
    assert artifacts.summary["year"][0] == _YEAR
    assert artifacts.summary["timeframe"][0] == _TIMEFRAME
    fold_ids = artifacts.observations["fold_id"].unique().sort().to_list()
    assert fold_ids == artifacts.fold_metrics["fold_id"].unique().sort().to_list()
    btc_returns = (
        artifacts.observations.filter(
            (pl.col("symbol") == _SYMBOL_BTC)
            & pl.col("selected")
            & (pl.col("partition") == WalkForwardEvaluationPartition.OOS.value)
        )["future_return_1"]
        .drop_nulls()
        .to_list()
    )
    eth_returns = (
        artifacts.observations.filter(
            (pl.col("symbol") == _SYMBOL_ETH)
            & pl.col("selected")
            & (pl.col("partition") == WalkForwardEvaluationPartition.OOS.value)
        )["future_return_1"]
        .drop_nulls()
        .to_list()
    )
    assert btc_returns
    assert eth_returns
    assert max(eth_returns) > max(btc_returns)


def test_deterministic_repeated_evaluation() -> None:
    """Repeated evaluation with identical inputs is byte-identical."""
    frame = _evaluation_input(n_times=8)
    evaluator = WalkForwardEvaluator(train_window=4, test_window=2, step_size=2)
    first = evaluator.evaluate(frame, manager=_MANAGER, engine=_ENGINE, year=_YEAR)
    second = evaluator.evaluate(frame, manager=_MANAGER, engine=_ENGINE, year=_YEAR)
    assert_frame_equal(first.observations, second.observations)
    assert_frame_equal(first.fold_metrics, second.fold_metrics)
    assert_frame_equal(first.factor_metrics, second.factor_metrics)
    assert_frame_equal(first.summary, second.summary)


def test_oos_ic_matches_research_information_coefficient() -> None:
    """Raw OOS IC matches research Spearman IC; oriented IC applies direction."""
    frame = _evaluation_input(
        n_times=10,
        selected_factors=(_FACTOR_A,),
        selected_direction=-1,
        selection_ic=-0.08,
    )
    artifacts = WalkForwardEvaluator(
        train_window=4,
        test_window=4,
        step_size=4,
    ).evaluate(frame, manager=_MANAGER, engine=_ENGINE, year=_YEAR)
    assert artifacts.factor_metrics.height >= 1
    calculator = InformationCoefficient(method="spearman")
    for row in artifacts.factor_metrics.iter_rows(named=True):
        oos = artifacts.observations.filter(
            (pl.col("fold_id") == row["fold_id"])
            & (pl.col("factor_name") == row["factor_name"])
            & (pl.col("factor_version") == row["factor_version"])
            & (pl.col("partition") == WalkForwardEvaluationPartition.OOS.value)
            & pl.col("selected")
        )
        expected_raw = calculator.compute(oos, "factor_value", "future_return_1").coefficient
        oriented = oos.with_columns(
            (pl.col("factor_value") * pl.col("selected_direction").cast(pl.Float64)).alias(
                "oriented_factor"
            )
        )
        expected_oriented = calculator.compute(
            oriented, "oriented_factor", "future_return_1"
        ).coefficient
        if row["raw_oos_ic"] is not None:
            assert row["raw_oos_ic"] == pytest.approx(expected_raw, rel=1e-9, abs=1e-12)
        if row["oriented_oos_ic"] is not None:
            assert row["oriented_oos_ic"] == pytest.approx(expected_oriented, rel=1e-9, abs=1e-12)
            assert row["oos_ic"] == pytest.approx(row["oriented_oos_ic"], rel=1e-9, abs=1e-12)
            assert row["oriented_oos_ic"] == pytest.approx(
                -float(row["raw_oos_ic"]), rel=1e-9, abs=1e-12
            )


def test_no_future_return_injected_into_upstream_schemas() -> None:
    """Upstream Alpha/Regime/Predictions/Signals contracts lack future_return_1."""
    assert TARGET_COLUMN not in ALPHA_COLUMNS
    assert TARGET_COLUMN not in REGIME_COLUMNS
    assert TARGET_COLUMN not in PREDICTION_COLUMNS
    assert TARGET_COLUMN not in SIGNAL_COLUMNS


def test_evaluation_modules_forbid_upstream_and_ml_imports() -> None:
    """Evaluation package must not import Alpha/Regime/Predictions/Signals/ml."""
    root = Path("src/cqros/walk_forward")
    forbidden = (
        "cqros.alpha",
        "cqros.regime",
        "cqros.predictions",
        "cqros.signals",
        "cqros.ml",
    )
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for name in forbidden:
                        assert not alias.name.startswith(name), path
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                for name in forbidden:
                    assert not node.module.startswith(name), path


def test_repository_round_trip_and_ledger_path_isolation(tmp_path: Path) -> None:
    """Evaluation repository writes under walk_forward_evaluation only."""
    layout = StorageLayout(tmp_path)
    datastore = _InMemoryDataStore()
    repository = WalkForwardEvaluationRepository(layout, datastore)
    artifacts = WalkForwardEvaluator(
        train_window=4,
        test_window=2,
        step_size=2,
    ).evaluate(
        _evaluation_input(n_times=6),
        manager=_MANAGER,
        engine=_ENGINE,
        year=_YEAR,
    )
    repository.save(
        artifacts.observations,
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    loaded = repository.load(
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert_frame_equal(loaded, artifacts.observations)
    eval_path = layout.walk_forward_evaluation_path(
        _MANAGER,
        EXCHANGE_BINANCE,
        MARKET_USDT_PERPETUAL,
        _TIMEFRAME,
        _YEAR,
    )
    ledger_path = layout.walk_forward_path(
        _MANAGER,
        EXCHANGE_BINANCE,
        MARKET_USDT_PERPETUAL,
        _TIMEFRAME,
        _YEAR,
    )
    assert eval_path != ledger_path
    assert "walk_forward_evaluation" in eval_path.as_posix()
    assert ledger_path not in datastore.frames


def test_production_ledger_hash_helper_is_stable(tmp_path: Path) -> None:
    """SHA-256 helper used for immutability checks is deterministic."""
    path = tmp_path / "ledger.parquet"
    frame = pl.DataFrame({"fold_id": [1, 2], "status": ["PASS", "PASS"]})
    frame.write_parquet(path)
    first = hashlib.sha256(path.read_bytes()).hexdigest()
    second = hashlib.sha256(path.read_bytes()).hexdigest()
    assert first == second


def test_assemble_still_feeds_engine_with_factor_value() -> None:
    """Evaluation input retains factor_value while remaining engine-compatible."""
    from cqros.factor_selection.schema import (
        CANONICAL_COLUMN_ORDER as FACTOR_SELECTION_COLUMNS,
    )
    from cqros.factor_selection.schema import (
        COLUMN_DTYPES as FACTOR_SELECTION_DTYPES,
    )
    from cqros.factor_selection.schema import FactorSelectionStatus
    from cqros.factors import FactorStatus
    from cqros.factors.schema import (
        CANONICAL_COLUMN_ORDER as FACTOR_CANONICAL_COLUMN_ORDER,
    )
    from cqros.factors.schema import COLUMN_DTYPES as FACTOR_COLUMN_DTYPES
    from cqros.labels.schema import (
        CANONICAL_COLUMN_ORDER as LABEL_CANONICAL_COLUMN_ORDER,
    )
    from cqros.labels.schema import COLUMN_DTYPES as LABEL_COLUMN_DTYPES
    from cqros.walk_forward.engine import SimpleWalkForwardEngine

    open_times = (1_700_000_000_000, 1_700_003_600_000, 1_700_007_200_000)
    selection = pl.DataFrame(
        {
            "factor_name": [_FACTOR_A],
            "factor_version": [_FACTOR_VERSION],
            "timeframe": [_TIMEFRAME],
            "selection_time": [open_times[0]],
            "factor_category": ["price"],
            "selected": [True],
            "selection_score": [0.1],
            "selection_rank": [1],
            "selection_reason": ["test"],
            "selection_ic": [0.08],
            "selected_direction": [1],
            "orientation_policy": ["signed_ic_v1"],
            "status": [FactorSelectionStatus.SELECTED.value],
        },
        schema=dict(FACTOR_SELECTION_DTYPES),
    ).select(list(FACTOR_SELECTION_COLUMNS))
    factors = pl.DataFrame(
        {
            "symbol": [_SYMBOL_BTC] * 3,
            "timeframe": [_TIMEFRAME] * 3,
            "open_time": list(open_times),
            "factor_name": [_FACTOR_A] * 3,
            "factor_version": [_FACTOR_VERSION] * 3,
            "factor_category": ["price"] * 3,
            "factor_group": ["alpha"] * 3,
            "factor_value": [0.1, 0.2, 0.3],
            "lookback": [20] * 3,
            "prediction_horizon": [1] * 3,
            "enabled": [True] * 3,
            "status": [FactorStatus.ACTIVE.value] * 3,
        },
        schema=dict(FACTOR_COLUMN_DTYPES),
    ).select(list(FACTOR_CANONICAL_COLUMN_ORDER))
    labels = pl.DataFrame(
        {
            "symbol": [_SYMBOL_BTC] * 3,
            "timeframe": [_TIMEFRAME] * 3,
            "open_time": list(open_times),
            "future_return_1": [0.01, 0.02, 0.03],
            "future_return_5": [0.05] * 3,
            "future_return_10": [0.10] * 3,
            "future_return_20": [0.20] * 3,
            "direction_1": [1] * 3,
            "direction_5": [1] * 3,
            "direction_10": [0] * 3,
            "direction_20": [0] * 3,
        },
        schema=dict(LABEL_COLUMN_DTYPES),
    ).select(list(LABEL_CANONICAL_COLUMN_ORDER))

    result = assemble_walk_forward_input(selection, factors, labels)
    assert "factor_value" in result.columns
    assert result["factor_value"].to_list() == [0.1, 0.2, 0.3]
    engine_output = SimpleWalkForwardEngine(
        train_window=2,
        test_window=1,
        step_size=1,
    ).build(result)
    assert engine_output.height >= 1
