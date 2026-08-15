"""CQROS Feature Engine abstract base feature.

Purpose:
    Provide a calculation-agnostic abstract base class that every concrete
    feature inherits from, eliminating metadata boilerplate while remaining
    free of registry, pipeline, storage, and validation logic.

Responsibilities:
    - Hold immutable feature metadata
    - Validate obvious constructor invariants
    - Define the single abstract ``transform`` contract
    - Behave as an immutable, hashable value object

Dependencies:
    ``polars``, the Python standard library, ``cqros.core.exceptions``, and
    structural compatibility with ``cqros.features.interfaces.Feature``.

Public API:
    ``BaseFeature``
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, cast

import polars as pl

from cqros.core.exceptions import ValidationError

__all__ = ["BaseFeature"]

_ERROR_NAME_EMPTY: Final[str] = "FEATURE-BASE-001"
_ERROR_VERSION_EMPTY: Final[str] = "FEATURE-BASE-002"
_ERROR_CATEGORY_EMPTY: Final[str] = "FEATURE-BASE-003"
_ERROR_DESCRIPTION_TYPE: Final[str] = "FEATURE-BASE-004"
_ERROR_REQUIRED_COLUMNS_TYPE: Final[str] = "FEATURE-BASE-005"
_ERROR_PRODUCED_COLUMNS_EMPTY: Final[str] = "FEATURE-BASE-006"
_ERROR_PRODUCED_COLUMNS_TYPE: Final[str] = "FEATURE-BASE-007"
_ERROR_LOOKBACK: Final[str] = "FEATURE-BASE-008"
_ERROR_DEPENDENCIES_TYPE: Final[str] = "FEATURE-BASE-009"
_ERROR_COLUMN_ENTRY: Final[str] = "FEATURE-BASE-010"


@dataclass(frozen=True, slots=True)
class BaseFeature(ABC):
    """Abstract immutable base for every CQROS feature implementation.

    Concrete features supply only ``transform``. Metadata is provided at
    construction time and remains fixed for the lifetime of the instance.
    This class intentionally does not perform dataframe validation, schema
    inspection, dependency resolution, registry registration, persistence,
    logging, or feature calculation.

    Attributes:
        name: Stable feature identifier used by registries and pipelines.
        version: Semantic version of the feature formula and parameters.
        category: Feature group classification (for example ``trend`` or
            ``momentum``).
        description: Human-readable summary of what the feature computes.
        required_columns: Input column names that must be present before
            ``transform`` runs.
        produced_columns: Output column names added by ``transform``.
        lookback: Minimum historical row count required for a fully defined
            warm-up window.
        dependencies: Names of other features that must be computed first.

    Notes:
        ``warmup_rows`` defaults to ``max(0, lookback - 1)`` (rolling-window
        semantics). Features whose leading undefined region differs—such as
        ``shift(lookback)`` momentum—must override ``warmup_rows``.
    """

    name: str
    version: str
    category: str
    description: str
    required_columns: tuple[str, ...]
    produced_columns: tuple[str, ...]
    lookback: int
    dependencies: tuple[str, ...] = ()

    @property
    def warmup_rows(self) -> int:
        """Leading undefined rows before the first fully defined value.

        Defaults to rolling-window semantics ``max(0, lookback - 1)``.
        Override on features whose initialization leaves a different leading
        null count (for example ``shift(lookback)`` momentum).
        """
        return max(0, self.lookback - 1)

    def __post_init__(self) -> None:
        """Normalize sequence fields and validate constructor invariants.

        Raises:
            ValidationError: If any metadata invariant is violated.
        """
        _require_non_empty_str(self.name, parameter="name", error_code=_ERROR_NAME_EMPTY)
        _require_non_empty_str(self.version, parameter="version", error_code=_ERROR_VERSION_EMPTY)
        _require_non_empty_str(
            self.category,
            parameter="category",
            error_code=_ERROR_CATEGORY_EMPTY,
        )
        _require_str(
            cast(object, self.description),
            parameter="description",
            error_code=_ERROR_DESCRIPTION_TYPE,
        )

        object.__setattr__(
            self,
            "required_columns",
            _freeze_str_sequence(
                self.required_columns,
                parameter="required_columns",
                type_error_code=_ERROR_REQUIRED_COLUMNS_TYPE,
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
        object.__setattr__(
            self,
            "dependencies",
            _freeze_str_sequence(
                self.dependencies,
                parameter="dependencies",
                type_error_code=_ERROR_DEPENDENCIES_TYPE,
                allow_empty=True,
            ),
        )
        _require_non_negative_int(
            cast(object, self.lookback),
            parameter="lookback",
            error_code=_ERROR_LOOKBACK,
        )

    @abstractmethod
    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Compute feature columns from ``frame`` without mutating it.

        Args:
            frame: Input market or research DataFrame. Must not be mutated.

        Returns:
            A new DataFrame containing the original columns plus
            ``produced_columns``.
        """

    def __str__(self) -> str:
        """Return a compact human-readable feature identity."""
        return f"{self.name}@{self.version}"

    def __repr__(self) -> str:
        """Return an unambiguous representation including all metadata."""
        return (
            f"{type(self).__name__}("
            f"name={self.name!r}, "
            f"version={self.version!r}, "
            f"category={self.category!r}, "
            f"description={self.description!r}, "
            f"required_columns={self.required_columns!r}, "
            f"produced_columns={self.produced_columns!r}, "
            f"lookback={self.lookback!r}, "
            f"warmup_rows={self.warmup_rows!r}, "
            f"dependencies={self.dependencies!r})"
        )


def _require_non_empty_str(value: object, *, parameter: str, error_code: str) -> None:
    """Raise ``ValidationError`` when ``value`` is not a non-empty string."""
    if not isinstance(value, str) or value.strip() == "":
        raise ValidationError(
            f"{parameter} must be a non-empty string",
            error_code=error_code,
            details={"parameter": parameter, "value": value},
        )


def _require_str(value: object, *, parameter: str, error_code: str) -> None:
    """Raise ``ValidationError`` when ``value`` is not a string."""
    if not isinstance(value, str):
        raise ValidationError(
            f"{parameter} must be a string",
            error_code=error_code,
            details={"parameter": parameter, "value_type": type(value).__name__},
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
        value: Candidate sequence of column or feature names.
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
                error_code=_ERROR_COLUMN_ENTRY,
                details={"parameter": parameter, "index": index, "value": entry},
            )
        frozen.append(entry)
    return tuple(frozen)
