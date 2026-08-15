"""CQROS Factor Research Engine abstract base factor.

Purpose:
    Provide a calculation-agnostic abstract base class that every concrete
    alpha factor inherits from, eliminating metadata boilerplate while
    remaining free of registry, pipeline, storage, and research logic.

Responsibilities:
    - Hold immutable factor metadata including every ``FACTOR_SCHEMA``
      metadata field
    - Validate obvious constructor invariants
    - Expose a ``metadata`` snapshot as ``FactorMetadata``
    - Define the single abstract ``compute`` contract
    - Behave as an immutable, hashable value object

Dependencies:
    ``polars``, the Python standard library, ``cqros.core.exceptions``,
    ``cqros.factors.metadata``, ``cqros.factors.schema.FactorStatus``, and
    structural compatibility with ``cqros.factors.interfaces.Factor``.

Public API:
    ``BaseFactor``
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, cast

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.metadata import FactorMetadata
from cqros.factors.schema import FactorStatus

__all__ = ["BaseFactor"]

_ERROR_NAME_EMPTY: Final[str] = "FACTOR-BASE-001"
_ERROR_VERSION_EMPTY: Final[str] = "FACTOR-BASE-002"
_ERROR_DESCRIPTION_EMPTY: Final[str] = "FACTOR-BASE-003"
_ERROR_CATEGORY_EMPTY: Final[str] = "FACTOR-BASE-004"
_ERROR_REQUIRED_FEATURES_TYPE: Final[str] = "FACTOR-BASE-005"
_ERROR_PRODUCED_COLUMNS_EMPTY: Final[str] = "FACTOR-BASE-006"
_ERROR_PRODUCED_COLUMNS_TYPE: Final[str] = "FACTOR-BASE-007"
_ERROR_LOOKBACK: Final[str] = "FACTOR-BASE-008"
_ERROR_ENTRY: Final[str] = "FACTOR-BASE-009"
_ERROR_FACTOR_GROUP_EMPTY: Final[str] = "FACTOR-BASE-010"
_ERROR_PREDICTION_HORIZON: Final[str] = "FACTOR-BASE-011"
_ERROR_ENABLED_TYPE: Final[str] = "FACTOR-BASE-012"
_ERROR_STATUS_TYPE: Final[str] = "FACTOR-BASE-013"

_DEFAULT_FACTOR_GROUP: Final[str] = "alpha"
_DEFAULT_PREDICTION_HORIZON: Final[int] = 1
_DEFAULT_ENABLED: Final[bool] = True
_DEFAULT_STATUS: Final[FactorStatus] = FactorStatus.ACTIVE


@dataclass(frozen=True, slots=True)
class BaseFactor(ABC):
    """Abstract immutable base for every CQROS alpha factor implementation.

    Concrete factors supply only ``compute``. Metadata is provided at
    construction time and remains fixed for the lifetime of the instance.
    This class intentionally does not perform dataframe validation, schema
    inspection, registry registration, persistence, logging, IC calculation,
    or factor signal computation.

    Attributes:
        name: Stable factor identifier used by registries and pipelines.
        version: Semantic version of the factor formula and parameters.
        description: Human-readable summary of what the factor computes.
        category: Factor category classification (for example ``momentum`` or
            ``value``).
        required_features: Feature names that must be present before
            ``compute`` runs.
        produced_columns: Output column names added by ``compute``.
        lookback: Minimum historical row count required for a fully defined
            warm-up window.
        factor_group: Research group classification for ``FACTOR_SCHEMA``.
        prediction_horizon: Forward horizon associated with the factor.
        enabled: Whether the factor is enabled for research use.
        status: Lifecycle status stored as ``FactorStatus``.
    """

    name: str
    version: str
    description: str
    category: str
    required_features: tuple[str, ...]
    produced_columns: tuple[str, ...]
    lookback: int
    factor_group: str = _DEFAULT_FACTOR_GROUP
    prediction_horizon: int = _DEFAULT_PREDICTION_HORIZON
    enabled: bool = _DEFAULT_ENABLED
    status: FactorStatus = _DEFAULT_STATUS

    def __post_init__(self) -> None:
        """Normalize sequence fields and validate constructor invariants.

        Raises:
            ValidationError: If any metadata invariant is violated.
        """
        _require_non_empty_str(self.name, parameter="name", error_code=_ERROR_NAME_EMPTY)
        _require_non_empty_str(self.version, parameter="version", error_code=_ERROR_VERSION_EMPTY)
        _require_non_empty_str(
            self.description,
            parameter="description",
            error_code=_ERROR_DESCRIPTION_EMPTY,
        )
        _require_non_empty_str(
            self.category,
            parameter="category",
            error_code=_ERROR_CATEGORY_EMPTY,
        )
        _require_non_empty_str(
            self.factor_group,
            parameter="factor_group",
            error_code=_ERROR_FACTOR_GROUP_EMPTY,
        )

        object.__setattr__(
            self,
            "required_features",
            _freeze_str_sequence(
                self.required_features,
                parameter="required_features",
                type_error_code=_ERROR_REQUIRED_FEATURES_TYPE,
                allow_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "produced_columns",
            _freeze_str_sequence(
                self.produced_columns,
                parameter="produced_columns",
                type_error_code=_ERROR_PRODUCED_COLUMNS_TYPE,
                allow_empty=False,
                empty_error_code=_ERROR_PRODUCED_COLUMNS_EMPTY,
            ),
        )
        _require_non_negative_int(
            cast(object, self.lookback),
            parameter="lookback",
            error_code=_ERROR_LOOKBACK,
        )
        _require_non_negative_int(
            cast(object, self.prediction_horizon),
            parameter="prediction_horizon",
            error_code=_ERROR_PREDICTION_HORIZON,
        )
        enabled_value = cast(object, self.enabled)
        if not isinstance(enabled_value, bool):
            raise ValidationError(
                "enabled must be a boolean",
                error_code=_ERROR_ENABLED_TYPE,
                details={"parameter": "enabled", "value": enabled_value},
            )
        status_value = cast(object, self.status)
        if not isinstance(status_value, FactorStatus):
            raise ValidationError(
                "status must be a FactorStatus member",
                error_code=_ERROR_STATUS_TYPE,
                details={
                    "parameter": "status",
                    "value": status_value,
                    "value_type": type(status_value).__name__,
                },
            )

    @property
    def metadata(self) -> FactorMetadata:
        """Return an immutable metadata snapshot of this factor.

        Returns:
            A ``FactorMetadata`` value object mirroring constructor fields.
        """
        return FactorMetadata(
            name=self.name,
            version=self.version,
            description=self.description,
            category=self.category,
            required_features=self.required_features,
            produced_columns=self.produced_columns,
            lookback=self.lookback,
            factor_group=self.factor_group,
            prediction_horizon=self.prediction_horizon,
            enabled=self.enabled,
            status=self.status,
        )

    @abstractmethod
    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Compute factor signal columns from ``frame`` without mutating it.

        Args:
            frame: Input research DataFrame containing required features.
                Must not be mutated.

        Returns:
            A new DataFrame containing the original columns plus
            ``produced_columns``.
        """

    def __str__(self) -> str:
        """Return a compact human-readable factor identity."""
        return f"{self.name}@{self.version}"

    def __repr__(self) -> str:
        """Return an unambiguous representation including all metadata."""
        return (
            f"{type(self).__name__}("
            f"name={self.name!r}, "
            f"version={self.version!r}, "
            f"description={self.description!r}, "
            f"category={self.category!r}, "
            f"required_features={self.required_features!r}, "
            f"produced_columns={self.produced_columns!r}, "
            f"lookback={self.lookback!r}, "
            f"factor_group={self.factor_group!r}, "
            f"prediction_horizon={self.prediction_horizon!r}, "
            f"enabled={self.enabled!r}, "
            f"status={self.status!r})"
        )


def _require_non_empty_str(value: object, *, parameter: str, error_code: str) -> None:
    """Raise ``ValidationError`` when ``value`` is not a non-empty string."""
    if not isinstance(value, str) or value.strip() == "":
        raise ValidationError(
            f"{parameter} must be a non-empty string",
            error_code=error_code,
            details={"parameter": parameter, "value": value},
        )


def _require_non_negative_int(value: object, *, parameter: str, error_code: str) -> None:
    """Raise ``ValidationError`` when ``value`` is not a non-negative int."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValidationError(
            f"{parameter} must be an integer greater than or equal to 0",
            error_code=error_code,
            details={"parameter": parameter, "value": value},
        )


def _freeze_str_sequence(
    value: object,
    *,
    parameter: str,
    type_error_code: str,
    allow_empty: bool,
    empty_error_code: str | None = None,
) -> tuple[str, ...]:
    """Validate a string sequence and return an immutable tuple copy.

    Args:
        value: Candidate sequence of feature or column names.
        parameter: Field name used in error messages.
        type_error_code: Error code when ``value`` is not a string sequence.
        allow_empty: Whether an empty sequence is permitted.
        empty_error_code: Error code when emptiness is rejected.

    Returns:
        A new tuple of validated strings.

    Raises:
        ValidationError: If ``value`` fails structural or emptiness checks.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValidationError(
            f"{parameter} must be a sequence of strings",
            error_code=type_error_code,
            details={"parameter": parameter, "value_type": type(value).__name__},
        )

    sequence = cast(Sequence[object], value)
    if not allow_empty and len(sequence) == 0:
        raise ValidationError(
            f"{parameter} must contain at least one entry",
            error_code=empty_error_code or type_error_code,
            details={"parameter": parameter},
        )

    frozen: list[str] = []
    for index, entry in enumerate(sequence):
        if not isinstance(entry, str) or entry.strip() == "":
            raise ValidationError(
                f"{parameter} entries must be non-empty strings",
                error_code=_ERROR_ENTRY,
                details={"parameter": parameter, "index": index, "value": entry},
            )
        frozen.append(entry)
    return tuple(frozen)
