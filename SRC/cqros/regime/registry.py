"""CQROS Regime Engine registry.

Purpose:
    Own the lifecycle of exactly one ``RegimeEngine`` instance and expose
    pure build delegation for Regime pipelines.

Responsibilities:
    - Accept an injected ``RegimeEngine`` or default to ``SimpleRegimeEngine``
    - Expose the registered engine through the ``engine`` property
    - Delegate ``build`` calls directly to the registered engine
    - Remain free of regime math, persistence, and trading

Dependencies:
    ``polars`` and ``cqros.regime.engine``.

Public API:
    ``RegimeRegistry``

Notes:
    This registry stores a single engine reference. It does not cache build
    results and does not mutate caller-supplied DataFrames.
"""

from __future__ import annotations

import polars as pl

from cqros.regime.engine import RegimeEngine, SimpleRegimeEngine

__all__ = ["RegimeRegistry"]


class RegimeRegistry:
    """Lifecycle owner for a single CQROS regime engine.

    The registry holds exactly one ``RegimeEngine`` instance. When no engine
    is injected, ``SimpleRegimeEngine`` is used. ``build`` delegates
    exclusively to the registered engine with no additional transformation.

    Args:
        engine: Optional regime engine. Defaults to ``SimpleRegimeEngine``
            when ``None``.

    Notes:
        Future engines (HMM, clustering, statistical detection, and similar)
        may replace the default through constructor injection without changing
        pipeline call sites.
    """

    __slots__ = ("_engine",)

    _engine: RegimeEngine

    def __init__(self, engine: RegimeEngine | None = None) -> None:
        """Initialize the registry with an injected or default engine.

        Args:
            engine: Regime engine to own. Instantiates ``SimpleRegimeEngine``
                when ``None``.
        """
        self._engine = SimpleRegimeEngine() if engine is None else engine

    @property
    def engine(self) -> RegimeEngine:
        """Return the registered regime engine.

        Returns:
            The owned ``RegimeEngine`` instance.
        """
        return self._engine

    def build(self, alpha: pl.DataFrame) -> pl.DataFrame:
        """Delegate regime generation to the registered engine.

        Args:
            alpha: Canonical Alpha dataset. Must not be mutated.

        Returns:
            Engine output DataFrame from ``RegimeEngine.build``.
        """
        return self._engine.build(alpha)
