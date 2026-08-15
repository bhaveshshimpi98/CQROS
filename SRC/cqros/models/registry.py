"""CQROS Models Engine registry.

Purpose:
    Own the lifecycle of exactly one ``ModelEngine`` instance and expose
    pure build delegation for Models pipelines.

Responsibilities:
    - Accept an injected ``ModelEngine`` or default to ``SimpleModelEngine``
    - Expose the registered engine through the ``engine`` property
    - Delegate ``build`` calls directly to the registered engine
    - Remain free of model math, persistence, and trading

Dependencies:
    ``polars`` and ``cqros.models.engine``.

Public API:
    ``ModelRegistry``

Notes:
    This registry stores a single engine reference. It does not cache build
    results and does not mutate caller-supplied DataFrames.
"""

from __future__ import annotations

import polars as pl

from cqros.models.engine import ModelEngine, SimpleModelEngine

__all__ = ["ModelRegistry"]


class ModelRegistry:
    """Lifecycle owner for a single CQROS model engine.

    The registry holds exactly one ``ModelEngine`` instance. When no engine
    is injected, ``SimpleModelEngine`` is used. ``build`` delegates
    exclusively to the registered engine with no additional transformation.

    Args:
        engine: Optional model engine. Defaults to ``SimpleModelEngine``
            when ``None``.

    Notes:
        Future engines (linear, tree, ensemble, neural, and similar) may
        replace the default through constructor injection without changing
        pipeline call sites.
    """

    __slots__ = ("_engine",)

    _engine: ModelEngine

    def __init__(self, engine: ModelEngine | None = None) -> None:
        """Initialize the registry with an injected or default engine.

        Args:
            engine: Model engine to own. Instantiates ``SimpleModelEngine``
                when ``None``.
        """
        self._engine = SimpleModelEngine() if engine is None else engine

    @property
    def engine(self) -> ModelEngine:
        """Return the registered model engine.

        Returns:
            The owned ``ModelEngine`` instance.
        """
        return self._engine

    def build(self, regime: pl.DataFrame) -> pl.DataFrame:
        """Delegate model generation to the registered engine.

        Args:
            regime: Canonical Regime dataset. Must not be mutated.

        Returns:
            Engine output DataFrame from ``ModelEngine.build``.
        """
        return self._engine.build(regime)
