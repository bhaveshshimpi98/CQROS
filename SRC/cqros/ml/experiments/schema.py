"""CQROS ML Experiment schema models.

Purpose:
    Provide immutable value objects that describe recorded ML experiments
    without coupling to training, evaluation, or persistence backends.

Responsibilities:
    - Define ``ExperimentRecord`` as the experiment metadata contract
    - Validate constructor invariants for experiment identity and metrics
    - Remain free of training, evaluation, I/O, and tracking orchestration

Dependencies:
    ``datetime``, ``cqros.ml.experiments.exceptions``, and
    ``cqros.ml.models.metadata``.

Public API:
    ``ExperimentRecord``
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Final, cast

from cqros.ml.experiments.exceptions import ModelValidationError
from cqros.ml.models.metadata import ModelFramework, ModelTaskType

__all__ = [
    "ExperimentRecord",
]

_ERROR_EXPERIMENT_ID_EMPTY: Final[str] = "ML-EXP-SCHEMA-001"
_ERROR_TIMESTAMP_TYPE: Final[str] = "ML-EXP-SCHEMA-002"
_ERROR_TIMESTAMP_TZ: Final[str] = "ML-EXP-SCHEMA-003"
_ERROR_MODEL_NAME_EMPTY: Final[str] = "ML-EXP-SCHEMA-004"
_ERROR_FRAMEWORK_TYPE: Final[str] = "ML-EXP-SCHEMA-005"
_ERROR_TASK_TYPE: Final[str] = "ML-EXP-SCHEMA-006"
_ERROR_LABEL_COLUMN_EMPTY: Final[str] = "ML-EXP-SCHEMA-007"
_ERROR_FEATURE_COUNT: Final[str] = "ML-EXP-SCHEMA-008"
_ERROR_TRAIN_ROWS: Final[str] = "ML-EXP-SCHEMA-009"
_ERROR_VALIDATION_ROWS: Final[str] = "ML-EXP-SCHEMA-010"
_ERROR_TEST_ROWS: Final[str] = "ML-EXP-SCHEMA-011"
_ERROR_PARAMETERS_TYPE: Final[str] = "ML-EXP-SCHEMA-012"
_ERROR_CV_METRICS_TYPE: Final[str] = "ML-EXP-SCHEMA-013"
_ERROR_CV_METRIC_VALUE: Final[str] = "ML-EXP-SCHEMA-014"
_ERROR_BEST_METRIC_TYPE: Final[str] = "ML-EXP-SCHEMA-015"
_ERROR_ARTIFACT_PATH_TYPE: Final[str] = "ML-EXP-SCHEMA-016"
_ERROR_NOTES_TYPE: Final[str] = "ML-EXP-SCHEMA-017"
_ERROR_PARAMETER_KEY: Final[str] = "ML-EXP-SCHEMA-018"
_ERROR_CV_METRIC_KEY: Final[str] = "ML-EXP-SCHEMA-019"


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    """Immutable metadata describing one CQROS ML experiment.

    Captures identity, model context, dataset sizes, parameters, and summary
    metrics for a completed experiment. This record does not train, evaluate,
    or persist model binaries.

    Attributes:
        experiment_id: Stable unique experiment identifier.
        timestamp: Timezone-aware UTC timestamp of the experiment.
        model_name: Registered model name used by the experiment.
        framework: Machine-learning framework used by the model.
        task_type: Supervised learning task type.
        label_column: Target label column name.
        feature_count: Number of feature columns consumed.
        train_rows: Number of training rows.
        validation_rows: Number of validation rows.
        test_rows: Number of test rows.
        parameters: Immutable mapping of experiment parameters.
        cross_validation_metrics: Immutable mapping of CV summary metrics.
        best_metric: Primary scalar metric retained for ranking/reporting.
        artifact_path: Optional artifact location reference string.
        notes: Free-form researcher notes.
    """

    experiment_id: str
    timestamp: datetime
    model_name: str
    framework: ModelFramework
    task_type: ModelTaskType
    label_column: str
    feature_count: int
    train_rows: int
    validation_rows: int
    test_rows: int
    parameters: Mapping[str, object]
    cross_validation_metrics: Mapping[str, float]
    best_metric: float
    artifact_path: str
    notes: str

    def __post_init__(self) -> None:
        """Normalize mapping fields and validate constructor invariants.

        Raises:
            ModelValidationError: If any experiment-record invariant is violated.
        """
        _require_non_empty_str(
            self.experiment_id,
            parameter="experiment_id",
            error_code=_ERROR_EXPERIMENT_ID_EMPTY,
        )
        _require_utc_timestamp(cast(object, self.timestamp))
        _require_non_empty_str(
            self.model_name,
            parameter="model_name",
            error_code=_ERROR_MODEL_NAME_EMPTY,
        )
        _require_framework(cast(object, self.framework))
        _require_task_type(cast(object, self.task_type))
        _require_non_empty_str(
            self.label_column,
            parameter="label_column",
            error_code=_ERROR_LABEL_COLUMN_EMPTY,
        )
        _require_non_negative_int(
            cast(object, self.feature_count),
            parameter="feature_count",
            error_code=_ERROR_FEATURE_COUNT,
        )
        _require_non_negative_int(
            cast(object, self.train_rows),
            parameter="train_rows",
            error_code=_ERROR_TRAIN_ROWS,
        )
        _require_non_negative_int(
            cast(object, self.validation_rows),
            parameter="validation_rows",
            error_code=_ERROR_VALIDATION_ROWS,
        )
        _require_non_negative_int(
            cast(object, self.test_rows),
            parameter="test_rows",
            error_code=_ERROR_TEST_ROWS,
        )
        object.__setattr__(
            self,
            "parameters",
            _freeze_object_mapping(
                cast(object, self.parameters),
                parameter="parameters",
                type_error_code=_ERROR_PARAMETERS_TYPE,
                key_error_code=_ERROR_PARAMETER_KEY,
            ),
        )
        object.__setattr__(
            self,
            "cross_validation_metrics",
            _freeze_float_mapping(
                cast(object, self.cross_validation_metrics),
                parameter="cross_validation_metrics",
                type_error_code=_ERROR_CV_METRICS_TYPE,
                key_error_code=_ERROR_CV_METRIC_KEY,
                value_error_code=_ERROR_CV_METRIC_VALUE,
            ),
        )
        if not isinstance(cast(object, self.best_metric), (int, float)) or isinstance(
            cast(object, self.best_metric),
            bool,
        ):
            raise ModelValidationError(
                "best_metric must be a numeric float",
                error_code=_ERROR_BEST_METRIC_TYPE,
                details={
                    "parameter": "best_metric",
                    "value_type": type(self.best_metric).__name__,
                },
            )
        object.__setattr__(self, "best_metric", float(self.best_metric))
        _require_str(
            cast(object, self.artifact_path),
            parameter="artifact_path",
            error_code=_ERROR_ARTIFACT_PATH_TYPE,
        )
        _require_str(
            cast(object, self.notes),
            parameter="notes",
            error_code=_ERROR_NOTES_TYPE,
        )


def _require_non_empty_str(value: object, *, parameter: str, error_code: str) -> None:
    """Raise when ``value`` is not a non-empty string."""
    if not isinstance(value, str) or value.strip() == "":
        raise ModelValidationError(
            f"{parameter} must be a non-empty string",
            error_code=error_code,
            details={"parameter": parameter, "value": value},
        )


def _require_str(value: object, *, parameter: str, error_code: str) -> None:
    """Raise when ``value`` is not a string."""
    if not isinstance(value, str):
        raise ModelValidationError(
            f"{parameter} must be a string",
            error_code=error_code,
            details={"parameter": parameter, "value_type": type(value).__name__},
        )


def _require_utc_timestamp(value: object) -> None:
    """Raise when ``value`` is not a timezone-aware UTC datetime."""
    if not isinstance(value, datetime):
        raise ModelValidationError(
            "timestamp must be a datetime instance",
            error_code=_ERROR_TIMESTAMP_TYPE,
            details={"parameter": "timestamp", "value_type": type(value).__name__},
        )
    if value.tzinfo is None or value.utcoffset() is None:
        raise ModelValidationError(
            "timestamp must be timezone-aware UTC",
            error_code=_ERROR_TIMESTAMP_TZ,
            details={"parameter": "timestamp", "tzinfo": value.tzinfo},
        )
    if value.utcoffset() != timedelta(0):
        raise ModelValidationError(
            "timestamp must be timezone-aware UTC",
            error_code=_ERROR_TIMESTAMP_TZ,
            details={
                "parameter": "timestamp",
                "utcoffset": str(value.utcoffset()),
                "expected": str(timedelta(0)),
                "reference": str(UTC),
            },
        )


def _require_framework(value: object) -> None:
    """Raise when ``value`` is not a ``ModelFramework`` member."""
    if not isinstance(value, ModelFramework):
        raise ModelValidationError(
            "framework must be a ModelFramework value",
            error_code=_ERROR_FRAMEWORK_TYPE,
            details={"parameter": "framework", "value_type": type(value).__name__},
        )


def _require_task_type(value: object) -> None:
    """Raise when ``value`` is not a ``ModelTaskType`` member."""
    if not isinstance(value, ModelTaskType):
        raise ModelValidationError(
            "task_type must be a ModelTaskType value",
            error_code=_ERROR_TASK_TYPE,
            details={"parameter": "task_type", "value_type": type(value).__name__},
        )


def _require_non_negative_int(value: object, *, parameter: str, error_code: str) -> None:
    """Raise when ``value`` is not a non-negative integer."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ModelValidationError(
            f"{parameter} must be a non-negative integer",
            error_code=error_code,
            details={"parameter": parameter, "value": value},
        )


def _freeze_object_mapping(
    value: object,
    *,
    parameter: str,
    type_error_code: str,
    key_error_code: str,
) -> Mapping[str, object]:
    """Validate a string-keyed mapping and return an immutable snapshot."""
    if not isinstance(value, Mapping):
        raise ModelValidationError(
            f"{parameter} must be a mapping",
            error_code=type_error_code,
            details={"parameter": parameter, "value_type": type(value).__name__},
        )

    frozen: dict[str, object] = {}
    for key, entry in cast(Mapping[object, object], value).items():
        if not isinstance(key, str) or key.strip() == "":
            raise ModelValidationError(
                f"{parameter} keys must be non-empty strings",
                error_code=key_error_code,
                details={"parameter": parameter, "key": key},
            )
        frozen[key] = entry
    return MappingProxyType(frozen)


def _freeze_float_mapping(
    value: object,
    *,
    parameter: str,
    type_error_code: str,
    key_error_code: str,
    value_error_code: str,
) -> Mapping[str, float]:
    """Validate a string-to-float mapping and return an immutable snapshot."""
    if not isinstance(value, Mapping):
        raise ModelValidationError(
            f"{parameter} must be a mapping",
            error_code=type_error_code,
            details={"parameter": parameter, "value_type": type(value).__name__},
        )

    frozen: dict[str, float] = {}
    for key, entry in cast(Mapping[object, object], value).items():
        if not isinstance(key, str) or key.strip() == "":
            raise ModelValidationError(
                f"{parameter} keys must be non-empty strings",
                error_code=key_error_code,
                details={"parameter": parameter, "key": key},
            )
        if not isinstance(entry, (int, float)) or isinstance(entry, bool):
            raise ModelValidationError(
                f"{parameter} values must be numeric floats",
                error_code=value_error_code,
                details={
                    "parameter": parameter,
                    "key": key,
                    "value_type": type(entry).__name__,
                },
            )
        frozen[key] = float(entry)
    return MappingProxyType(frozen)
