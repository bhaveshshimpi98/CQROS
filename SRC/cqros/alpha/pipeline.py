"""CQROS Alpha package pipeline.

Purpose:
    Orchestrate conversion of canonical Factor Orthogonalization datasets
    into canonical Alpha datasets through ``AlphaRegistry``, then persist
    partitions through ``AlphaRepository``.

Responsibilities:
    - Require an injected ``AlphaRegistry`` owning an observation-backed engine
    - Require an injected ``AlphaRepository`` for persistence
    - Delegate alpha generation exclusively to the registry with the requested
      symbol participating in generation
    - Persist the generated partition through the repository
    - Return the generated Alpha DataFrame
    - Preserve Factor-Orthogonalization-frame immutability
    - Remain free of alpha algorithms, ranking, statistics, verification,
      exchange APIs, and CLI logic

Dependencies:
    ``polars``, ``cqros.core.types``, ``cqros.alpha.exceptions``,
    ``cqros.alpha.registry``, and ``cqros.alpha.repository``.

Public API:
    ``AlphaPipeline``
"""

from __future__ import annotations

from typing import Final

import polars as pl

from cqros.alpha.exceptions import AlphaError
from cqros.alpha.registry import AlphaRegistry
from cqros.alpha.repository import AlphaRepository
from cqros.core.types import Exchange, Market, Symbol, Timeframe

__all__ = ["AlphaPipeline"]

_ERROR_REGISTRY_REQUIRED: Final[str] = "ALPHA_PIPE_REGISTRY_REQUIRED"
_ERROR_REPOSITORY_REQUIRED: Final[str] = "ALPHA_PIPE_REPOSITORY_REQUIRED"


class AlphaPipeline:
    """Orchestrator for Alpha generation and persistence.

    The pipeline delegates alpha generation to ``AlphaRegistry.build`` and
    persists the result through ``AlphaRepository.save``. Alpha semantics
    remain exclusively in the engine owned by the registry. The
    caller-supplied Factor Orthogonalization frame is never mutated.

    Args:
        registry: Registry used to generate alpha rows. Required because
            combination-unit alpha needs an observation-backed engine.
        repository: Persistence facade for alpha partitions. Must be
            injected; cannot be defaulted without storage dependencies.
    """

    __slots__ = ("_registry", "_repository")

    _registry: AlphaRegistry
    _repository: AlphaRepository

    def __init__(
        self,
        registry: AlphaRegistry | None = None,
        repository: AlphaRepository | None = None,
    ) -> None:
        """Initialize the pipeline with required registry and repository.

        Args:
            registry: Alpha registry. Required.
            repository: Alpha repository. Required for persistence.

        Raises:
            AlphaError: If ``registry`` or ``repository`` is ``None``.
        """
        if registry is None:
            raise AlphaError(
                "registry must be injected; AlphaRegistry "
                "cannot be defaulted without an observation-backed engine",
                error_code=_ERROR_REGISTRY_REQUIRED,
                details={"registry": None},
            )
        if repository is None:
            raise AlphaError(
                "repository must be injected; AlphaRepository "
                "cannot be defaulted without StorageLayout and IDataStore",
                error_code=_ERROR_REPOSITORY_REQUIRED,
                details={"repository": None},
            )
        self._registry = registry
        self._repository = repository

    def build(
        self,
        factor_orthogonalization: pl.DataFrame,
        *,
        manager: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> pl.DataFrame:
        """Generate alpha rows, persist the partition, and return the frame.

        Generation is delegated to ``AlphaRegistry.build`` with ``symbol``
        participating in alpha construction. Persistence is delegated to
        ``AlphaRepository.save``. The original Factor Orthogonalization
        frame is never mutated.

        Args:
            factor_orthogonalization: Canonical Factor Orthogonalization
                dataset.
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            The Alpha DataFrame produced by the registry.
        """
        created = self._registry.build(factor_orthogonalization, symbol=symbol)
        self._repository.save(
            created,
            manager=manager,
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        )
        return created
