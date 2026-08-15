"""CQROS Models package pipeline.

Purpose:
    Orchestrate conversion of canonical Regime datasets into canonical
    Models datasets through ``ModelRegistry``, then persist partitions
    through ``ModelRepository``.

Responsibilities:
    - Accept an injected ``ModelRegistry`` or default to a new registry
      owning ``SimpleModelEngine``
    - Require an injected ``ModelRepository`` for persistence
    - Delegate model generation exclusively to the registry
    - Persist the generated partition through the repository
    - Return the generated Models DataFrame
    - Preserve Regime-frame immutability
    - Remain free of model algorithms, ranking, statistics, verification,
      exchange APIs, and CLI logic

Dependencies:
    ``polars``, ``cqros.core.types``, ``cqros.models.exceptions``,
    ``cqros.models.registry``, and ``cqros.models.repository``.

Public API:
    ``ModelPipeline``
"""

from __future__ import annotations

from typing import Final

import polars as pl

from cqros.core.types import Exchange, Market, Symbol, Timeframe
from cqros.models.exceptions import ModelError
from cqros.models.registry import ModelRegistry
from cqros.models.repository import ModelRepository

__all__ = ["ModelPipeline"]

_ERROR_REPOSITORY_REQUIRED: Final[str] = "MODEL_PIPE_REPOSITORY_REQUIRED"


class ModelPipeline:
    """Orchestrator for Models generation and persistence.

    The pipeline delegates model generation to ``ModelRegistry.build`` and
    persists the result through ``ModelRepository.save``. Model semantics
    remain exclusively in the engine owned by the registry. The
    caller-supplied Regime frame is never mutated.

    Args:
        registry: Registry used to generate model rows. Defaults to a new
            ``ModelRegistry`` when ``None``.
        repository: Persistence facade for models partitions. Must be
            injected; cannot be defaulted without storage dependencies.
    """

    __slots__ = ("_registry", "_repository")

    _registry: ModelRegistry
    _repository: ModelRepository

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        repository: ModelRepository | None = None,
    ) -> None:
        """Initialize the pipeline with optional registry and repository.

        Args:
            registry: Model registry. Instantiates ``ModelRegistry`` when
                ``None``.
            repository: Models repository. Required for persistence; must
                not be ``None``.

        Raises:
            ModelError: If ``repository`` is ``None``.
        """
        self._registry = ModelRegistry() if registry is None else registry
        if repository is None:
            raise ModelError(
                "repository must be injected; ModelRepository "
                "cannot be defaulted without StorageLayout and IDataStore",
                error_code=_ERROR_REPOSITORY_REQUIRED,
                details={"repository": None},
            )
        self._repository = repository

    def build(
        self,
        regime: pl.DataFrame,
        *,
        manager: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> pl.DataFrame:
        """Generate model rows, persist the partition, and return the frame.

        Generation is delegated to ``ModelRegistry.build``. Persistence is
        delegated to ``ModelRepository.save``. The original Regime frame is
        never mutated.

        Args:
            regime: Canonical Regime dataset.
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            The Models DataFrame produced by the registry.
        """
        created = self._registry.build(regime)
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
