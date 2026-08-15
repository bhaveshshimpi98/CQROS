"""Unit tests for CQROS ``TrainingWorkflow``."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.core.constants import EXCHANGE_BINANCE, MARKET_USDT_PERPETUAL
from cqros.ml.dataset import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    DatasetLoader,
    DatasetSplitter,
    IdentityScaler,
)
from cqros.ml.evaluation import ModelEvaluator, TimeSeriesCrossValidator
from cqros.ml.experiments import ExperimentTracker
from cqros.ml.models import (
    LightGBMModel,
    ModelFramework,
    ModelMetadata,
    ModelRegistry,
    ModelTaskType,
    ModelValidationError,
)
from cqros.ml.optimization import HyperparameterOptimizer, OptimizationResult
from cqros.ml.training import ModelTrainer
from cqros.ml.workflow import TrainingWorkflow, WorkflowResult
from cqros.ml.workflow.workflow import TrainingWorkflow as TrainingWorkflowDirect
from cqros.storage import TrainingPartitionRef

_EXCHANGE = EXCHANGE_BINANCE
_MARKET = MARKET_USDT_PERPETUAL
_START = 1_700_000_000_000
_INTERVAL = 3_600_000
_MODEL_FEATURES = ("returns", "log_returns", "atr")
_LABEL_COLUMN = "future_return_1"


class _RecordingRepository:
    """Training repository stub that returns preloaded partitions."""

    def __init__(self, partitions: dict[tuple[str, str, int], pl.DataFrame]) -> None:
        self.partitions = partitions

    def discover_partitions(
        self,
        *,
        symbols: tuple[str, ...] | list[str] | None = None,
        timeframes: tuple[str, ...] | list[str] | None = None,
        exchange: str = _EXCHANGE,
        market: str = _MARKET,
    ) -> tuple[TrainingPartitionRef, ...]:
        del exchange, market
        symbol_filter = set(symbols) if symbols is not None else None
        timeframe_filter = set(timeframes) if timeframes is not None else None
        items: list[TrainingPartitionRef] = []
        for symbol, timeframe, year in self.partitions:
            if symbol_filter is not None and symbol not in symbol_filter:
                continue
            if timeframe_filter is not None and timeframe not in timeframe_filter:
                continue
            items.append(TrainingPartitionRef(symbol=symbol, timeframe=timeframe, year=year))
        return tuple(sorted(items, key=lambda item: (item.symbol, item.timeframe, item.year)))

    def load(
        self,
        *,
        exchange: str,
        market: str,
        symbol: str,
        timeframe: str,
        year: int,
    ) -> pl.DataFrame:
        del exchange, market
        return self.partitions[(symbol, timeframe, year)]


def _feature_values(row_count: int, *, value: float = 0.01) -> dict[str, list[float]]:
    """Build default float values for every feature column."""
    values: dict[str, list[float]] = {}
    for column in FEATURE_COLUMNS:
        values[column] = [value + float(index) for index in range(row_count)]
    return values


def _label_values(row_count: int) -> dict[str, list[float] | list[int]]:
    """Build default values for every label column."""
    values: dict[str, list[float] | list[int]] = {}
    for column in LABEL_COLUMNS:
        if column.startswith("direction_"):
            values[column] = [1 if index % 2 == 0 else 0 for index in range(row_count)]
        else:
            values[column] = [0.01 * float(index + 1) for index in range(row_count)]
    return values


def _training_frame(*, row_count: int = 100, symbol: str = "BTCUSDT") -> pl.DataFrame:
    """Build a chronologically ordered canonical ML dataset."""
    open_times = [_START + index * _INTERVAL for index in range(row_count)]
    data: dict[str, object] = {
        "symbol": [symbol] * row_count,
        "timeframe": ["1h"] * row_count,
        "open_time": open_times,
    }
    data.update(_feature_values(row_count))
    data.update(_label_values(row_count))
    frame = pl.DataFrame(data, schema=COLUMN_DTYPES)
    return frame.select(list(CANONICAL_COLUMN_ORDER))


def _metadata(*, name: str = "workflow-lgbm") -> ModelMetadata:
    """Build ModelMetadata aligned to the canonical training schema."""
    return ModelMetadata(
        name=name,
        version="1.0.0",
        framework=ModelFramework.LIGHTGBM,
        task_type=ModelTaskType.REGRESSION,
        feature_columns=_MODEL_FEATURES,
        label_column=_LABEL_COLUMN,
        description="LightGBM workflow test model",
    )


def _build_workflow(
    *,
    frame: pl.DataFrame | None = None,
    model_name: str = "workflow-lgbm",
    register_model: bool = True,
) -> tuple[TrainingWorkflow, ModelRegistry, ExperimentTracker]:
    """Wire a TrainingWorkflow with real CQROS components and a stub repository."""
    partitions: dict[tuple[str, str, int], pl.DataFrame] = {}
    if frame is not None:
        partitions[("BTCUSDT", "1h", 2024)] = frame

    registry = ModelRegistry()
    if register_model:
        registry.register(
            LightGBMModel(
                model_metadata=_metadata(name=model_name),
                num_boost_round=10,
            )
        )

    trainer = ModelTrainer(model_registry=registry)
    evaluator = ModelEvaluator()
    cross_validator = TimeSeriesCrossValidator(
        model_registry=registry,
        model_trainer=trainer,
        model_evaluator=evaluator,
    )
    optimizer = HyperparameterOptimizer(cross_validator=cross_validator)
    tracker = ExperimentTracker()
    workflow = TrainingWorkflow(
        dataset_loader=DatasetLoader(repository=_RecordingRepository(partitions)),  # type: ignore[arg-type]
        dataset_splitter=DatasetSplitter(),
        dataset_scaler=IdentityScaler(),
        model_registry=registry,
        model_trainer=trainer,
        model_evaluator=evaluator,
        cross_validator=cross_validator,
        hyperparameter_optimizer=optimizer,
        experiment_tracker=tracker,
    )
    return workflow, registry, tracker


def test_package_exports_training_workflow() -> None:
    """TrainingWorkflow and WorkflowResult are package exports."""
    import cqros.ml.workflow as workflow_package

    assert "TrainingWorkflow" in workflow_package.__all__
    assert "WorkflowResult" in workflow_package.__all__
    assert workflow_package.TrainingWorkflow is TrainingWorkflow
    assert TrainingWorkflow is TrainingWorkflowDirect


def test_successful_workflow() -> None:
    """TrainingWorkflow completes load, split, scale, train, evaluate, and record."""
    workflow, _registry, tracker = _build_workflow(frame=_training_frame(row_count=100))

    result = workflow.run(
        model_name="workflow-lgbm",
        experiment_id="exp-success",
        train_ratio=0.7,
        validation_ratio=0.15,
        test_ratio=0.15,
    )

    assert isinstance(result, WorkflowResult)
    assert result.experiment_id == "exp-success"
    assert result.model_metadata.name == "workflow-lgbm"
    assert result.train_result.train_rows == 70
    assert result.train_result.validation_rows == 15
    assert result.validation_result.dataset_rows == 15
    assert result.test_result.dataset_rows == 15
    assert result.cross_validation_result is None
    assert result.optimization_result is None
    assert result.duration >= 0.0
    assert set(result.validation_result.metrics) == {"mae", "mse", "rmse", "r2"}
    assert tracker.exists("exp-success") is True
    recorded = tracker.get("exp-success")
    assert recorded.model_name == "workflow-lgbm"
    assert recorded.train_rows == 70
    assert recorded.test_rows == 15
    assert recorded.cross_validation_metrics == {}


def test_workflow_with_cross_validation() -> None:
    """Optional walk-forward CV is executed when folds are provided."""
    workflow, _registry, tracker = _build_workflow(frame=_training_frame(row_count=120))

    result = workflow.run(
        model_name="workflow-lgbm",
        experiment_id="exp-cv",
        train_ratio=0.7,
        validation_ratio=0.15,
        test_ratio=0.15,
        cross_validation_folds=2,
    )

    assert result.cross_validation_result is not None
    assert result.cross_validation_result.fold_count == 2
    assert result.optimization_result is None
    recorded = tracker.get("exp-cv")
    assert "mae" in recorded.cross_validation_metrics
    assert recorded.best_metric == recorded.cross_validation_metrics["mae"]


def test_workflow_with_optimization() -> None:
    """Optional hyperparameter optimization is executed when configured."""
    workflow, _registry, tracker = _build_workflow(frame=_training_frame(row_count=120))

    result = workflow.run(
        model_name="workflow-lgbm",
        experiment_id="exp-hpo",
        train_ratio=0.7,
        validation_ratio=0.15,
        test_ratio=0.15,
        parameter_grid={"num_boost_round": [8, 12]},
        optimization_metric="mae",
        optimization_folds=2,
    )

    assert result.cross_validation_result is None
    assert isinstance(result.optimization_result, OptimizationResult)
    assert result.optimization_result.model_name == "workflow-lgbm"
    assert len(result.optimization_result.trials) == 2
    recorded = tracker.get("exp-hpo")
    assert recorded.parameters == result.optimization_result.best_parameters
    assert recorded.best_metric == result.optimization_result.best_score


def test_workflow_with_cross_validation_and_optimization() -> None:
    """Workflow supports both optional CV and hyperparameter optimization."""
    workflow, _registry, tracker = _build_workflow(frame=_training_frame(row_count=120))

    result = workflow.run(
        model_name="workflow-lgbm",
        experiment_id="exp-both",
        train_ratio=0.7,
        validation_ratio=0.15,
        test_ratio=0.15,
        cross_validation_folds=2,
        parameter_grid={"num_boost_round": [8, 12]},
        optimization_metric="mae",
        optimization_folds=2,
    )

    assert result.cross_validation_result is not None
    assert result.cross_validation_result.fold_count == 2
    assert result.optimization_result is not None
    assert len(result.optimization_result.trials) == 2
    recorded = tracker.get("exp-both")
    assert recorded.best_metric == result.optimization_result.best_score
    assert "mae" in recorded.cross_validation_metrics


def test_experiment_recording() -> None:
    """Successful workflows always record an experiment in the tracker."""
    workflow, _registry, tracker = _build_workflow(frame=_training_frame(row_count=80))

    result = workflow.run(
        model_name="workflow-lgbm",
        experiment_id="exp-record",
        train_ratio=0.6,
        validation_ratio=0.2,
        test_ratio=0.2,
        artifact_path="artifacts/exp-record",
        notes="workflow unit test",
    )

    assert tracker.exists("exp-record") is True
    recorded = tracker.get("exp-record")
    assert recorded.experiment_id == result.experiment_id
    assert recorded.artifact_path == "artifacts/exp-record"
    assert recorded.notes == "workflow unit test"
    assert recorded.feature_count == len(_MODEL_FEATURES)
    assert recorded.label_column == _LABEL_COLUMN
    assert recorded.parameters == {
        "train_ratio": 0.6,
        "validation_ratio": 0.2,
        "test_ratio": 0.2,
    }


def test_unknown_model_rejected() -> None:
    """Unknown model names raise ModelValidationError before loading data."""
    workflow, _registry, tracker = _build_workflow(
        frame=_training_frame(row_count=40),
        register_model=False,
    )

    with pytest.raises(ModelValidationError, match="not registered"):
        workflow.run(
            model_name="missing-model",
            experiment_id="exp-missing",
            train_ratio=0.7,
            validation_ratio=0.15,
            test_ratio=0.15,
        )

    assert tracker.list() == ()


def test_empty_dataset_rejected() -> None:
    """Empty loaded datasets raise ModelValidationError."""
    workflow, _registry, tracker = _build_workflow(frame=None)

    with pytest.raises(ModelValidationError, match="at least one row"):
        workflow.run(
            model_name="workflow-lgbm",
            experiment_id="exp-empty",
            train_ratio=0.7,
            validation_ratio=0.15,
            test_ratio=0.15,
        )

    assert tracker.list() == ()


def test_invalid_optimization_configuration_rejected() -> None:
    """Partial hyperparameter-optimization configuration is rejected."""
    workflow, _registry, _tracker = _build_workflow(frame=_training_frame(row_count=40))

    with pytest.raises(ModelValidationError, match="must all be provided together"):
        workflow.run(
            model_name="workflow-lgbm",
            experiment_id="exp-bad-hpo",
            train_ratio=0.7,
            validation_ratio=0.15,
            test_ratio=0.15,
            parameter_grid={"num_boost_round": [8]},
        )


def test_rejects_invalid_dependencies() -> None:
    """Constructor rejects dependencies with invalid types."""
    registry = ModelRegistry()
    trainer = ModelTrainer(model_registry=registry)
    evaluator = ModelEvaluator()
    cross_validator = TimeSeriesCrossValidator(
        model_registry=registry,
        model_trainer=trainer,
        model_evaluator=evaluator,
    )
    optimizer = HyperparameterOptimizer(cross_validator=cross_validator)
    with pytest.raises(ModelValidationError, match="DatasetLoader"):
        TrainingWorkflow(
            dataset_loader="not-a-loader",  # type: ignore[arg-type]
            dataset_splitter=DatasetSplitter(),
            dataset_scaler=IdentityScaler(),
            model_registry=registry,
            model_trainer=trainer,
            model_evaluator=evaluator,
            cross_validator=cross_validator,
            hyperparameter_optimizer=optimizer,
            experiment_tracker=ExperimentTracker(),
        )
