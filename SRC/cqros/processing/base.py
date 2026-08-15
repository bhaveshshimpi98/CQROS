"""CQROS Data Processing Framework abstract base processing step.

Purpose:
    Provide a calculation-agnostic abstract base class that every concrete
    processing step inherits from, eliminating metadata boilerplate while
    remaining free of registry, pipeline, storage, and dataset-specific
    logic.

Responsibilities:
    - Hold immutable processing-step metadata
    - Validate obvious constructor invariants
    - Expose a ``metadata`` snapshot as ``ProcessingMetadata``
    - Define the single abstract ``process`` contract
    - Behave as an immutable, hashable value object

Dependencies:
    ``polars``, the Python standard library, ``cqros.core.exceptions``,
    ``cqros.processing.metadata``, and structural compatibility with
    ``cqros.processing.interfaces.ProcessingStep``.

Public API:
    ``BaseProcessingStep``
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.processing.metadata import ProcessingMetadata

__all__ = ["BaseProcessingStep"]

_ERROR_NAME_EMPTY: Final[str] = "PROCESSING-BASE-001"
_ERROR_VERSION_EMPTY: Final[str] = "PROCESSING-BASE-002"
_ERROR_DESCRIPTION_EMPTY: Final[str] = "PROCESSING-BASE-003"


@dataclass(frozen=True, slots=True)
class BaseProcessingStep(ABC):
    """Abstract immutable base for every CQROS processing-step implementation.

    Concrete steps supply only ``process``. Metadata is provided at
    construction time and remains fixed for the lifetime of the instance.
    This class intentionally does not perform dataframe validation, schema
    inspection, registry registration, persistence, logging, or
    dataset-specific cleaning logic.

    Attributes:
        name: Stable processing-step identifier used by registries and
            pipelines.
        version: Semantic version of the step formula and parameters.
        description: Human-readable summary of what the step does.
    """

    name: str
    version: str
    description: str

    def __post_init__(self) -> None:
        """Validate constructor invariants.

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

    @property
    def metadata(self) -> ProcessingMetadata:
        """Return an immutable metadata snapshot of this processing step.

        Returns:
            A ``ProcessingMetadata`` value object mirroring constructor fields.
        """
        return ProcessingMetadata(
            name=self.name,
            version=self.version,
            description=self.description,
        )

    @abstractmethod
    def process(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Transform ``frame`` without mutating it.

        Args:
            frame: Input market DataFrame. Must not be mutated.

        Returns:
            A new DataFrame produced by this processing step.
        """

    def __str__(self) -> str:
        """Return a compact human-readable processing-step identity."""
        return f"{self.name}@{self.version}"

    def __repr__(self) -> str:
        """Return an unambiguous representation including all metadata."""
        return (
            f"{type(self).__name__}("
            f"name={self.name!r}, "
            f"version={self.version!r}, "
            f"description={self.description!r})"
        )


def _require_non_empty_str(value: object, *, parameter: str, error_code: str) -> None:
    """Raise ``ValidationError`` when ``value`` is not a non-empty string."""
    if not isinstance(value, str) or value.strip() == "":
        raise ValidationError(
            f"{parameter} must be a non-empty string",
            error_code=error_code,
            details={"parameter": parameter, "value": value},
        )
