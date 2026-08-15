"""Unit tests for CQROS ``ExperimentTracker``."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cqros.ml.experiments import ExperimentRecord, ExperimentTracker
from cqros.ml.experiments.exceptions import ModelValidationError
from cqros.ml.experiments.tracker import ExperimentTracker as ExperimentTrackerDirect
from cqros.ml.models.metadata import ModelFramework, ModelTaskType


def _record(
    experiment_id: str,
    *,
    model_name: str = "alpha-lgbm",
    best_metric: float = 0.1,
) -> ExperimentRecord:
    """Build ExperimentRecord for tracker unit tests."""
    return ExperimentRecord(
        experiment_id=experiment_id,
        timestamp=datetime(2026, 7, 29, 12, 0, tzinfo=UTC),
        model_name=model_name,
        framework=ModelFramework.LIGHTGBM,
        task_type=ModelTaskType.REGRESSION,
        label_column="label",
        feature_count=2,
        train_rows=100,
        validation_rows=20,
        test_rows=10,
        parameters={"num_boost_round": 50},
        cross_validation_metrics={"mae": best_metric, "rmse": best_metric * 2},
        best_metric=best_metric,
        artifact_path=f"artifacts/{experiment_id}",
        notes="unit-test experiment",
    )


def test_package_exports_experiment_tracker() -> None:
    """ExperimentTracker and ExperimentRecord are package exports."""
    import cqros.ml.experiments as experiments_package

    assert "ExperimentTracker" in experiments_package.__all__
    assert "ExperimentRecord" in experiments_package.__all__
    assert experiments_package.ExperimentTracker is ExperimentTracker
    assert ExperimentTracker is ExperimentTrackerDirect


def test_record_and_retrieve() -> None:
    """record stores a record that get can retrieve by experiment ID."""
    tracker = ExperimentTracker()
    record = _record("exp-001")

    tracker.record(record)

    assert tracker.get("exp-001") is record
    assert tracker.exists("exp-001") is True


def test_exists() -> None:
    """exists reports presence without raising for unknown IDs."""
    tracker = ExperimentTracker()
    tracker.record(_record("exp-001"))

    assert tracker.exists("exp-001") is True
    assert tracker.exists("exp-missing") is False


def test_delete() -> None:
    """delete removes a recorded experiment."""
    tracker = ExperimentTracker()
    tracker.record(_record("exp-001"))

    tracker.delete("exp-001")

    assert tracker.exists("exp-001") is False
    with pytest.raises(ModelValidationError, match="not recorded"):
        tracker.get("exp-001")


def test_duplicate_ids_rejected() -> None:
    """Duplicate experiment IDs raise ModelValidationError."""
    tracker = ExperimentTracker()
    tracker.record(_record("exp-001", best_metric=0.1))

    with pytest.raises(ModelValidationError, match="already recorded"):
        tracker.record(_record("exp-001", best_metric=0.2))

    assert tracker.get("exp-001").best_metric == 0.1


def test_invalid_ids_rejected() -> None:
    """Empty experiment IDs raise ModelValidationError on tracker methods."""
    tracker = ExperimentTracker()
    tracker.record(_record("exp-001"))

    with pytest.raises(ModelValidationError, match="experiment_id"):
        tracker.get("")
    with pytest.raises(ModelValidationError, match="experiment_id"):
        tracker.exists("   ")
    with pytest.raises(ModelValidationError, match="experiment_id"):
        tracker.delete("")


def test_invalid_record_rejected() -> None:
    """Non-ExperimentRecord values raise ModelValidationError."""
    tracker = ExperimentTracker()
    with pytest.raises(ModelValidationError, match="ExperimentRecord"):
        tracker.record("not-a-record")  # type: ignore[arg-type]


def test_insertion_order_preserved() -> None:
    """list returns records in insertion order."""
    tracker = ExperimentTracker()
    first = _record("exp-a")
    second = _record("exp-b")
    third = _record("exp-c")

    tracker.record(first)
    tracker.record(second)
    tracker.record(third)

    assert tracker.list() == (first, second, third)


def test_list_returns_immutable_snapshot() -> None:
    """list returns a new tuple snapshot that is unaffected by later writes."""
    tracker = ExperimentTracker()
    first = _record("exp-001")
    tracker.record(first)

    snapshot = tracker.list()
    tracker.record(_record("exp-002"))
    tracker.delete("exp-001")

    assert snapshot == (first,)
    assert [record.experiment_id for record in tracker.list()] == ["exp-002"]


def test_get_unknown_id_rejected() -> None:
    """Unknown experiment IDs raise ModelValidationError on get/delete."""
    tracker = ExperimentTracker()
    with pytest.raises(ModelValidationError, match="not recorded"):
        tracker.get("missing")
    with pytest.raises(ModelValidationError, match="not recorded"):
        tracker.delete("missing")


def test_empty_tracker_list() -> None:
    """A new tracker lists no experiments."""
    tracker = ExperimentTracker()
    assert tracker.list() == ()
