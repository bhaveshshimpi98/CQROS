"""CQROS ML Model metadata models.

Purpose:
    Provide immutable value objects and enumerations that describe machine
    learning models—not model parameters, predictions, or trained weights.

Responsibilities:
    - Define ``ModelFramework`` and ``ModelTaskType`` enumerations
    - Define ``ModelMetadata`` used by registries, pipelines, lineage,
      reporting, and documentation
    - Remain free of training, inference, serialization, and I/O logic

Dependencies:
    Python standard library only.

Public API:
    ``ModelFramework``, ``ModelTaskType``, ``ModelMetadata``
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, cast

from cqros.ml.models.exceptions import ModelValidationError

__all__ = [
    "ModelFramework",
    "ModelMetadata",
    "ModelTaskType",
]

_ERROR_NAME_EMPTY: Final[str] = "ML-MODEL-META-001"
_ERROR_VERSION_EMPTY: Final[str] = "ML-MODEL-META-002"
_ERROR_FRAMEWORK_TYPE: Final[str] = "ML-MODEL-META-003"
_ERROR_TASK_TYPE: Final[str] = "ML-MODEL-META-004"
_ERROR_FEATURE_COLUMNS_TYPE: Final[str] = "ML-MODEL-META-005"
_ERROR_FEATURE_COLUMNS_EMPTY: Final[str] = "ML-MODEL-META-006"
_ERROR_LABEL_COLUMN_EMPTY: Final[str] = "ML-MODEL-META-007"
_ERROR_DESCRIPTION_TYPE: Final[str] = "ML-MODEL-META-008"
_ERROR_FEATURE_COLUMN_ENTRY: Final[str] = "ML-MODEL-META-009"


class ModelFramework(StrEnum):
    """Supported machine-learning frameworks for CQROS models."""

    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    CATBOOST = "catboost"
    RANDOM_FOREST = "random_forest"


class ModelTaskType(StrEnum):
    """Supported supervised learning task types."""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Immutable metadata describing a single CQROS ML model.

    Captures identity, framework, task type, and column contracts for one
    model definition. This model does not train, predict, or serialize
    artifacts.

    Attributes:
        name: Stable model identifier.
        version: Semantic version of the model definition.
        framework: Machine-learning framework used by the model.
        task_type: Supervised learning task type.
        feature_columns: Feature column names consumed by the model.
        label_column: Target label column name.
        description: Human-readable summary of the model purpose.
    """

    name: str
    version: str
    framework: ModelFramework
    task_type: ModelTaskType
    feature_columns: tuple[str, ...]
    label_column: str
    description: str

    def __post_init__(self) -> None:
        """Normalize sequence fields and validate constructor invariants.

        Raises:
            ModelValidationError: If any metadata invariant is violated.
        """
        _require_non_empty_str(self.name, parameter="name", error_code=_ERROR_NAME_EMPTY)
        _require_non_empty_str(
            self.version,
            parameter="version",
            error_code=_ERROR_VERSION_EMPTY,
        )
        _require_framework(cast(object, self.framework))
        _require_task_type(cast(object, self.task_type))
        object.__setattr__(
            self,
            "feature_columns",
            _freeze_str_sequence(
                self.feature_columns,
                parameter="feature_columns",
                type_error_code=_ERROR_FEATURE_COLUMNS_TYPE,
                empty_error_code=_ERROR_FEATURE_COLUMNS_EMPTY,
            ),
        )
        _require_non_empty_str(
            self.label_column,
            parameter="label_column",
            error_code=_ERROR_LABEL_COLUMN_EMPTY,
        )
        _require_str(
            cast(object, self.description),
            parameter="description",
            error_code=_ERROR_DESCRIPTION_TYPE,
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


def _freeze_str_sequence(
    value: object,
    *,
    parameter: str,
    type_error_code: str,
    empty_error_code: str,
) -> tuple[str, ...]:
    """Validate a non-empty string sequence and return an immutable copy."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ModelValidationError(
            f"{parameter} must be a sequence of strings",
            error_code=type_error_code,
            details={"parameter": parameter, "value_type": type(value).__name__},
        )

    sequence = cast(Sequence[object], value)
    if len(sequence) == 0:
        raise ModelValidationError(
            f"{parameter} must contain at least one entry",
            error_code=empty_error_code,
            details={"parameter": parameter},
        )

    frozen: list[str] = []
    for index, entry in enumerate(sequence):
        if not isinstance(entry, str) or entry.strip() == "":
            raise ModelValidationError(
                f"{parameter} entries must be non-empty strings",
                error_code=_ERROR_FEATURE_COLUMN_ENTRY,
                details={"parameter": parameter, "index": index, "value": entry},
            )
        frozen.append(entry)
    return tuple(frozen)
