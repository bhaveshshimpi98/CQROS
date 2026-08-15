"""CQROS Factor Orthogonalization Engine registry.

Purpose:
    Own the lifecycle of exactly one ``FactorOrthogonalizationEngine``
    instance and expose pure build delegation for Factor Orthogonalization
    pipelines.

Responsibilities:
    - Accept an injected ``FactorOrthogonalizationEngine``
    - Expose the registered engine through the ``engine`` property
    - Delegate ``build`` calls directly to the registered engine
    - Remain free of orthogonalization math, persistence, and trading

Dependencies:
    ``polars`` and ``cqros.factor_orthogonalization.engine``.

Public API:
    ``FactorOrthogonalizationRegistry``

Notes:
    This registry stores a single engine reference. It does not cache build
    results and does not mutate caller-supplied DataFrames. Unlike earlier
    scaffolds, a default engine is not constructed because combination-unit
    orthogonalization requires an injected observation source.
"""

from __future__ import annotations

import polars as pl

from cqros.factor_orthogonalization.engine import (
    FactorOrthogonalizationEngine,
    LineageContext,
)
from cqros.factor_orthogonalization.exceptions import FactorOrthogonalizationError

__all__ = ["FactorOrthogonalizationRegistry"]

_ERROR_ENGINE_REQUIRED = "FORTH_REGISTRY_ENGINE_REQUIRED"


class FactorOrthogonalizationRegistry:
    """Lifecycle owner for a single CQROS factor orthogonalization engine.

    Args:
        engine: Required factor orthogonalization engine. Must be injected
            because observation-source wiring belongs to the composition root.
    """

    __slots__ = ("_engine",)

    _engine: FactorOrthogonalizationEngine

    def __init__(self, engine: FactorOrthogonalizationEngine | None) -> None:
        """Initialize the registry with an injected engine.

        Args:
            engine: Factor orthogonalization engine to own.

        Raises:
            FactorOrthogonalizationError: If ``engine`` is ``None``.
        """
        if engine is None:
            raise FactorOrthogonalizationError(
                "engine must be injected; SimpleFactorOrthogonalizationEngine "
                "cannot be defaulted without FactorObservationSource",
                error_code=_ERROR_ENGINE_REQUIRED,
                details={"engine": None},
            )
        self._engine = engine

    @property
    def engine(self) -> FactorOrthogonalizationEngine:
        """Return the registered factor orthogonalization engine."""
        return self._engine

    def build(
        self,
        factor_combination: pl.DataFrame,
        *,
        lineage: LineageContext,
    ) -> pl.DataFrame:
        """Delegate orthogonalization generation to the registered engine.

        Args:
            factor_combination: Canonical Factor Combination dataset.
                Must not be mutated.
            lineage: Validation window and source version provenance.

        Returns:
            Engine output DataFrame from ``FactorOrthogonalizationEngine.build``.
        """
        return self._engine.build(factor_combination, lineage=lineage)
