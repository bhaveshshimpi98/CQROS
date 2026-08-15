"""CQROS Regime package pipeline.

Purpose:
    Orchestrate conversion of canonical Alpha datasets into canonical
    Regime datasets through ``RegimeRegistry``, then persist partitions
    through ``RegimeRepository``.

Responsibilities:
    - Accept an injected ``RegimeRegistry`` or default to a new registry
      owning ``SimpleRegimeEngine``
    - Require an injected ``RegimeRepository`` for persistence
    - Delegate regime generation exclusively to the registry
    - Persist the generated partition through the repository
    - Return the generated Regime DataFrame
    - Preserve Alpha-frame immutability
    - Remain free of regime algorithms, ranking, statistics, verification,
      exchange APIs, and CLI logic

Dependencies:
    ``polars``, ``cqros.core.types``, ``cqros.regime.exceptions``,
    ``cqros.regime.registry``, and ``cqros.regime.repository``.

Public API:
    ``RegimePipeline``
"""

from __future__ import annotations

from typing import Final

import polars as pl

from cqros.core.types import Exchange, Market, Symbol, Timeframe
from cqros.regime.exceptions import RegimeError
from cqros.regime.registry import RegimeRegistry
from cqros.regime.repository import RegimeRepository

__all__ = ["RegimePipeline"]

_ERROR_REPOSITORY_REQUIRED: Final[str] = "REGIME_PIPE_REPOSITORY_REQUIRED"


class RegimePipeline:
    """Orchestrator for Regime generation and persistence.

    The pipeline delegates regime generation to ``RegimeRegistry.build`` and
    persists the result through ``RegimeRepository.save``. Regime semantics
    remain exclusively in the engine owned by the registry. The
    caller-supplied Alpha frame is never mutated.

    Args:
        registry: Registry used to generate regime rows. Defaults to a new
            ``RegimeRegistry`` when ``None``.
        repository: Persistence facade for regime partitions. Must be
            injected; cannot be defaulted without storage dependencies.
    """

    __slots__ = ("_registry", "_repository")

    _registry: RegimeRegistry
    _repository: RegimeRepository

    def __init__(
        self,
        registry: RegimeRegistry | None = None,
        repository: RegimeRepository | None = None,
    ) -> None:
        """Initialize the pipeline with optional registry and repository.

        Args:
            registry: Regime registry. Instantiates ``RegimeRegistry`` when
                ``None``.
            repository: Regime repository. Required for persistence; must
                not be ``None``.

        Raises:
            RegimeError: If ``repository`` is ``None``.
        """
        self._registry = RegimeRegistry() if registry is None else registry
        if repository is None:
            raise RegimeError(
                "repository must be injected; RegimeRepository "
                "cannot be defaulted without StorageLayout and IDataStore",
                error_code=_ERROR_REPOSITORY_REQUIRED,
                details={"repository": None},
            )
        self._repository = repository

    def build(
        self,
        alpha: pl.DataFrame,
        *,
        manager: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> pl.DataFrame:
        """Generate regime rows, persist the partition, and return the frame.

        Generation is delegated to ``RegimeRegistry.build``. Persistence is
        delegated to ``RegimeRepository.save``. The original Alpha frame is
        never mutated.

        Args:
            alpha: Canonical Alpha dataset.
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            The Regime DataFrame produced by the registry.
        """
        created = self._registry.build(alpha)
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
