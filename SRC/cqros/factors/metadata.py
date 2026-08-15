"""CQROS Factor Research Engine metadata models.

Purpose:
    Provide immutable value objects that describe factors—not factor values
    or computed alpha signals.

Responsibilities:
    - Define ``FactorMetadata`` used by base factors, lineage, reporting,
      wide-to-long enrichment, and experiment tracking
    - Carry every canonical ``FACTOR_SCHEMA`` metadata field so transformers
      never fabricate values
    - Remain free of execution, validation pipelines, serialization, and I/O

Dependencies:
    Python standard library and ``cqros.factors.schema.FactorStatus``.

Public API:
    ``FactorMetadata``

Notes:
    ``FACTOR_SCHEMA`` metadata columns map as follows:

    - ``factor_name`` ← ``name``
    - ``factor_version`` ← ``version``
    - ``factor_category`` ← ``category``
    - ``factor_group`` ← ``factor_group``
    - ``lookback`` ← ``lookback``
    - ``prediction_horizon`` ← ``prediction_horizon``
    - ``enabled`` ← ``enabled``
    - ``status`` ← ``status`` (``FactorStatus`` string value)

    Fields present on this model but absent from ``FACTOR_SCHEMA``
    (``description``, ``required_features``, ``produced_columns``) remain
    research-identity metadata only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from cqros.factors.schema import FactorStatus

__all__ = ["FactorMetadata"]


def _freeze_str_tuple(value: Sequence[str]) -> tuple[str, ...]:
    """Return an immutable tuple copy of a string sequence."""
    return tuple(value)


@dataclass(frozen=True, slots=True)
class FactorMetadata:
    """Immutable metadata describing a single CQROS alpha factor.

    Captures identity, classification, feature dependencies, output column
    contracts, and every canonical ``FACTOR_SCHEMA`` metadata field for one
    factor definition. This model does not compute factors, validate
    dataframes, or perform statistical analysis.

    Attributes:
        name: Stable factor identifier (``factor_name`` in ``FACTOR_SCHEMA``).
        version: Semantic version of the factor formula and parameters
            (``factor_version`` in ``FACTOR_SCHEMA``).
        description: Human-readable summary of what the factor computes.
        category: Factor category classification (``factor_category``).
        required_features: Feature names required before ``compute`` runs.
        produced_columns: Output column names produced by ``compute``.
        lookback: Minimum historical row count for a fully defined window.
        factor_group: Research group classification (``factor_group``).
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
    factor_group: str
    prediction_horizon: int
    enabled: bool
    status: FactorStatus

    def __post_init__(self) -> None:
        """Freeze collection fields into immutable tuples."""
        object.__setattr__(
            self,
            "required_features",
            _freeze_str_tuple(self.required_features),
        )
        object.__setattr__(
            self,
            "produced_columns",
            _freeze_str_tuple(self.produced_columns),
        )
