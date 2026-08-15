"""CQROS Factor Combination Engine registry.

Purpose:
    Own the lifecycle of exactly one ``FactorCombinationEngine`` instance and
    expose pure build delegation for Factor Combination pipelines.

Responsibilities:
    - Accept an injected ``FactorCombinationEngine`` or default to
      ``SimpleFactorCombinationEngine``
    - Expose the registered engine through the ``engine`` property
    - Delegate ``build`` calls directly to the registered engine
    - Remain free of combination math, persistence, and trading

Dependencies:
    ``polars`` and ``cqros.factor_combination.engine``.

Public API:
    ``FactorCombinationRegistry``

Notes:
    This registry stores a single engine reference. It does not cache build
    results and does not mutate caller-supplied DataFrames.
"""

from __future__ import annotations

import polars as pl

from cqros.factor_combination.engine import (
    FactorCombinationEngine,
    SimpleFactorCombinationEngine,
)

__all__ = ["FactorCombinationRegistry"]


class FactorCombinationRegistry:
    """Lifecycle owner for a single CQROS factor combination engine.

    The registry holds exactly one ``FactorCombinationEngine`` instance.
    When no engine is injected, ``SimpleFactorCombinationEngine`` is used.
    ``build`` delegates exclusively to the registered engine with no
    additional transformation.

    Args:
        engine: Optional factor combination engine. Defaults to
            ``SimpleFactorCombinationEngine`` when ``None``.

    Notes:
        Future engines (genetic search, optimization, PCA, ML, and similar)
        may replace the default through constructor injection without
        changing pipeline call sites.
    """

    __slots__ = ("_engine",)

    _engine: FactorCombinationEngine

    def __init__(self, engine: FactorCombinationEngine | None = None) -> None:
        """Initialize the registry with an injected or default engine.

        Args:
            engine: Factor combination engine to own. Instantiates
                ``SimpleFactorCombinationEngine`` when ``None``.
        """
        self._engine = SimpleFactorCombinationEngine() if engine is None else engine

    @property
    def engine(self) -> FactorCombinationEngine:
        """Return the registered factor combination engine.

        Returns:
            The owned ``FactorCombinationEngine`` instance.
        """
        return self._engine

    def build(self, factor_timeframe_analysis: pl.DataFrame) -> pl.DataFrame:
        """Delegate combination generation to the registered engine.

        Args:
            factor_timeframe_analysis: Canonical Factor Timeframe Analysis
                dataset. Must not be mutated.

        Returns:
            Engine output DataFrame from ``FactorCombinationEngine.build``.
        """
        return self._engine.build(factor_timeframe_analysis)
