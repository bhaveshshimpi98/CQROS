"""Unit tests for CQROS ``ExperimentRecord`` schema."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cqros.ml.experiments.exceptions import ModelValidationError
from cqros.ml.experiments.schema import ExperimentRecord
from cqros.ml.models.metadata import ModelFramework, ModelTaskType


def _record(**overrides: object) -> ExperimentRecord:
    """Build ExperimentRecord with optional field overrides."""
    values: dict[str, object] = {
        "experiment_id": "exp-001",
        "timestamp": datetime(2026, 7, 29, 12, 0, tzinfo=UTC),
        "model_name": "alpha-lgbm",
        "framework": ModelFramework.LIGHTGBM,
        "task_type": ModelTaskType.REGRESSION,
        "label_column": "label",
        "feature_count": 2,
        "train_rows": 100,
        "validation_rows": 20,
        "test_rows": 10,
        "parameters": {"num_boost_round": 50},
        "cross_validation_metrics": {"mae": 0.1, "rmse": 0.2},
        "best_metric": 0.1,
        "artifact_path": "artifacts/exp-001",
        "notes": "baseline run",
    }
    values.update(overrides)
    return ExperimentRecord(**values)  # type: ignore[arg-type]


def test_valid_experiment_record() -> None:
    """Valid constructor arguments produce an immutable ExperimentRecord."""
    record = _record()
    assert record.experiment_id == "exp-001"
    assert record.framework is ModelFramework.LIGHTGBM
    assert record.task_type is ModelTaskType.REGRESSION
    assert record.parameters["num_boost_round"] == 50
    assert record.cross_validation_metrics["mae"] == 0.1
    assert record.best_metric == 0.1


def test_mappings_are_immutable_snapshots() -> None:
    """Parameter and metric mappings are frozen after construction."""
    mutable_params = {"num_boost_round": 10}
    mutable_metrics = {"mae": 0.5}
    record = _record(parameters=mutable_params, cross_validation_metrics=mutable_metrics)

    mutable_params["num_boost_round"] = 999
    mutable_metrics["mae"] = 999.0
    assert record.parameters["num_boost_round"] == 10
    assert record.cross_validation_metrics["mae"] == 0.5

    with pytest.raises(TypeError):
        record.parameters["num_boost_round"] = 20  # type: ignore[index]
    with pytest.raises(TypeError):
        record.cross_validation_metrics["mae"] = 0.0  # type: ignore[index]


def test_rejects_empty_experiment_id() -> None:
    """Empty experiment IDs raise ModelValidationError."""
    with pytest.raises(ModelValidationError, match="experiment_id"):
        _record(experiment_id="")
    with pytest.raises(ModelValidationError, match="experiment_id"):
        _record(experiment_id="   ")


def test_rejects_naive_timestamp() -> None:
    """Naive timestamps raise ModelValidationError."""
    with pytest.raises(ModelValidationError, match="timezone-aware"):
        _record(timestamp=datetime(2026, 7, 29, 12, 0))


def test_rejects_negative_row_counts() -> None:
    """Negative dataset sizes raise ModelValidationError."""
    with pytest.raises(ModelValidationError, match="train_rows"):
        _record(train_rows=-1)
    with pytest.raises(ModelValidationError, match="feature_count"):
        _record(feature_count=-5)


def test_rejects_invalid_framework_and_task_type() -> None:
    """Invalid enum fields raise ModelValidationError."""
    with pytest.raises(ModelValidationError, match="ModelFramework"):
        _record(framework="lightgbm")
    with pytest.raises(ModelValidationError, match="ModelTaskType"):
        _record(task_type="regression")


def test_rejects_non_numeric_cv_metric_values() -> None:
    """Non-numeric cross-validation metric values raise ModelValidationError."""
    with pytest.raises(ModelValidationError, match="numeric floats"):
        _record(cross_validation_metrics={"mae": "bad"})
