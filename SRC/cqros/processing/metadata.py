"""CQROS Data Processing Framework metadata models.

Purpose:
    Provide immutable value objects that describe processing steps—not
    processed dataset values.

Responsibilities:
    - Define ``ProcessingMetadata`` used by base steps, registries,
      lineage, and reporting
    - Remain free of execution, validation pipelines, serialization, and I/O

Dependencies:
    Python standard library only.

Public API:
    ``ProcessingMetadata``
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ProcessingMetadata"]


@dataclass(frozen=True, slots=True)
class ProcessingMetadata:
    """Immutable metadata describing a single CQROS processing step.

    Captures identity and description for one processing-step definition.
    This model does not execute processing, validate dataframes, or
    perform I/O.

    Attributes:
        name: Stable processing-step identifier.
        version: Semantic version of the step formula and parameters.
        description: Human-readable summary of what the step does.
    """

    name: str
    version: str
    description: str
