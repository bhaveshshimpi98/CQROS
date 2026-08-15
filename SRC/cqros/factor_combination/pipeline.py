"""CQROS Factor Combination package pipeline.

Purpose:
    Orchestrate conversion of canonical Factor Timeframe Analysis datasets
    into canonical Factor Combination datasets through
    ``FactorCombinationRegistry``, then persist partitions through
    ``FactorCombinationRepository``.

Responsibilities:
    - Accept an injected ``FactorCombinationRegistry`` or default to a new
      registry owning ``SimpleFactorCombinationEngine``
    - Require an injected ``FactorCombinationRepository`` for persistence
    - Delegate combination generation exclusively to the registry
    - Persist the generated partition through the repository
    - Return the generated Factor Combination DataFrame
    - Preserve Factor-Timeframe-Analysis-frame immutability
    - Remain free of combination algorithms, ranking, statistics,
      verification, exchange APIs, and CLI logic

Dependencies:
    ``polars``, ``cqros.core.types``, ``cqros.factor_combination.exceptions``,
    ``cqros.factor_combination.registry``, and
    ``cqros.factor_combination.repository``.

Public API:
    ``FactorCombinationPipeline``
"""

from __future__ import annotations

from typing import Final

import polars as pl

from cqros.core.types import Exchange, Market, Timeframe
from cqros.factor_combination.exceptions import FactorCombinationError
from cqros.factor_combination.registry import FactorCombinationRegistry
from cqros.factor_combination.repository import FactorCombinationRepository

__all__ = ["FactorCombinationPipeline"]

_ERROR_REPOSITORY_REQUIRED: Final[str] = "FCOMB_PIPE_REPOSITORY_REQUIRED"


class FactorCombinationPipeline:
    """Orchestrator for Factor Combination generation and persistence.

    The pipeline delegates combination generation to
    ``FactorCombinationRegistry.build`` and persists the result through
    ``FactorCombinationRepository.save``. Combination semantics remain
    exclusively in the engine owned by the registry. The caller-supplied
    Factor Timeframe Analysis frame is never mutated.

    Args:
        registry: Registry used to generate combinations. Defaults to a new
            ``FactorCombinationRegistry`` when ``None``.
        repository: Persistence facade for factor combination partitions.
            Must be injected; cannot be defaulted without storage
            dependencies.
    """

    __slots__ = ("_registry", "_repository")

    _registry: FactorCombinationRegistry
    _repository: FactorCombinationRepository

    def __init__(
        self,
        registry: FactorCombinationRegistry | None = None,
        repository: FactorCombinationRepository | None = None,
    ) -> None:
        """Initialize the pipeline with optional registry and repository.

        Args:
            registry: Factor combination registry. Instantiates
                ``FactorCombinationRegistry`` when ``None``.
            repository: Factor combination repository. Required for
                persistence; must not be ``None``.

        Raises:
            FactorCombinationError: If ``repository`` is ``None``.
        """
        self._registry = FactorCombinationRegistry() if registry is None else registry
        if repository is None:
            raise FactorCombinationError(
                "repository must be injected; FactorCombinationRepository "
                "cannot be defaulted without StorageLayout and IDataStore",
                error_code=_ERROR_REPOSITORY_REQUIRED,
                details={"repository": None},
            )
        self._repository = repository

    def build(
        self,
        factor_timeframe_analysis: pl.DataFrame,
        *,
        manager: str,
        exchange: Exchange,
        market: Market,
        timeframe: Timeframe,
        year: int,
    ) -> pl.DataFrame:
        """Generate combinations, persist the partition, and return the frame.

        Generation is delegated to ``FactorCombinationRegistry.build``.
        Persistence is delegated to ``FactorCombinationRepository.save``.
        The original Factor Timeframe Analysis frame is never mutated.

        Args:
            factor_timeframe_analysis: Canonical Factor Timeframe Analysis
                dataset.
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            The Factor Combination DataFrame produced by the registry.
        """
        created = self._registry.build(factor_timeframe_analysis)
        self._repository.save(
            created,
            manager=manager,
            exchange=exchange,
            market=market,
            timeframe=timeframe,
            year=year,
        )
        return created
