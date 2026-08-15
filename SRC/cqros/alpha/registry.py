"""CQROS Alpha Engine registry.

Purpose:
    Own the lifecycle of exactly one ``AlphaEngine`` instance and expose
    pure build delegation for Alpha pipelines.

Responsibilities:
    - Accept an injected ``AlphaEngine``
    - Expose the registered engine through the ``engine`` property
    - Delegate ``build`` calls directly to the registered engine
    - Remain free of alpha math, persistence, and trading

Dependencies:
    ``polars`` and ``cqros.alpha.engine``.

Public API:
    ``AlphaRegistry``

Notes:
    This registry stores a single engine reference. It does not cache build
    results and does not mutate caller-supplied DataFrames. A default engine
    is not constructed because combination-unit alpha requires an injected
    observation source.
"""

from __future__ import annotations

from typing import Final

import polars as pl

from cqros.alpha.engine import AlphaEngine
from cqros.alpha.exceptions import AlphaError

__all__ = ["AlphaRegistry"]

_ERROR_ENGINE_REQUIRED: Final[str] = "ALPHA_REGISTRY_ENGINE_REQUIRED"


class AlphaRegistry:
    """Lifecycle owner for a single CQROS alpha engine.

    The registry holds exactly one ``AlphaEngine`` instance. The engine must
    be injected because observation-source wiring belongs to the composition
    root. ``build`` delegates exclusively to the registered engine with no
    additional transformation.

    Args:
        engine: Required alpha engine.
    """

    __slots__ = ("_engine",)

    _engine: AlphaEngine

    def __init__(self, engine: AlphaEngine | None = None) -> None:
        """Initialize the registry with an injected engine.

        Args:
            engine: Alpha engine to own.

        Raises:
            AlphaError: If ``engine`` is ``None``.
        """
        if engine is None:
            raise AlphaError(
                "engine must be injected; SimpleAlphaEngine "
                "cannot be defaulted without FactorObservationSource",
                error_code=_ERROR_ENGINE_REQUIRED,
                details={"engine": None},
            )
        self._engine = engine

    @property
    def engine(self) -> AlphaEngine:
        """Return the registered alpha engine.

        Returns:
            The owned ``AlphaEngine`` instance.
        """
        return self._engine

    def build(
        self,
        factor_orthogonalization: pl.DataFrame,
        *,
        symbol: str,
    ) -> pl.DataFrame:
        """Delegate alpha generation to the registered engine.

        Args:
            factor_orthogonalization: Canonical Factor Orthogonalization
                dataset. Must not be mutated.
            symbol: Tradeable symbol for which alpha rows are generated.

        Returns:
            Engine output DataFrame from ``AlphaEngine.build``.
        """
        return self._engine.build(factor_orthogonalization, symbol=symbol)
