"""CQROS Factor Orthogonalization package pipeline.

Purpose:
    Orchestrate conversion of canonical Factor Combination datasets into
    canonical Factor Orthogonalization datasets through
    ``FactorOrthogonalizationRegistry``, then persist partitions through
    ``FactorOrthogonalizationRepository``.

Responsibilities:
    - Require injected registry and repository
    - Delegate orthogonalization generation exclusively to the registry
    - Persist the generated partition through the repository
    - Return the generated Factor Orthogonalization DataFrame
    - Preserve Factor-Combination-frame immutability
    - Remain free of orthogonalization algorithms, ranking, statistics,
      verification, exchange APIs, and CLI logic

Dependencies:
    ``polars``, ``cqros.core.types``,
    ``cqros.factor_orthogonalization.engine``,
    ``cqros.factor_orthogonalization.exceptions``,
    ``cqros.factor_orthogonalization.registry``, and
    ``cqros.factor_orthogonalization.repository``.

Public API:
    ``FactorOrthogonalizationPipeline``
"""

from __future__ import annotations

from typing import Final

import polars as pl

from cqros.core.types import Exchange, Market, Timeframe
from cqros.factor_orthogonalization.engine import LineageContext
from cqros.factor_orthogonalization.exceptions import FactorOrthogonalizationError
from cqros.factor_orthogonalization.registry import FactorOrthogonalizationRegistry
from cqros.factor_orthogonalization.repository import FactorOrthogonalizationRepository

__all__ = ["FactorOrthogonalizationPipeline"]

_ERROR_REGISTRY_REQUIRED: Final[str] = "FORTH_PIPE_REGISTRY_REQUIRED"
_ERROR_REPOSITORY_REQUIRED: Final[str] = "FORTH_PIPE_REPOSITORY_REQUIRED"


class FactorOrthogonalizationPipeline:
    """Orchestrator for Factor Orthogonalization generation and persistence.

    Args:
        registry: Registry used to generate orthogonalization rows. Required.
        repository: Persistence facade for factor orthogonalization
            partitions. Required.
    """

    __slots__ = ("_registry", "_repository")

    _registry: FactorOrthogonalizationRegistry
    _repository: FactorOrthogonalizationRepository

    def __init__(
        self,
        registry: FactorOrthogonalizationRegistry | None = None,
        repository: FactorOrthogonalizationRepository | None = None,
    ) -> None:
        """Initialize the pipeline with required registry and repository.

        Raises:
            FactorOrthogonalizationError: If ``registry`` or ``repository``
                is ``None``.
        """
        if registry is None:
            raise FactorOrthogonalizationError(
                "registry must be injected; FactorOrthogonalizationRegistry "
                "cannot be defaulted without an observation-backed engine",
                error_code=_ERROR_REGISTRY_REQUIRED,
                details={"registry": None},
            )
        if repository is None:
            raise FactorOrthogonalizationError(
                "repository must be injected; FactorOrthogonalizationRepository "
                "cannot be defaulted without StorageLayout and IDataStore",
                error_code=_ERROR_REPOSITORY_REQUIRED,
                details={"repository": None},
            )
        self._registry = registry
        self._repository = repository

    def build(
        self,
        factor_combination: pl.DataFrame,
        *,
        lineage: LineageContext,
        manager: str,
        exchange: Exchange,
        market: Market,
        timeframe: Timeframe,
        year: int,
    ) -> pl.DataFrame:
        """Generate orthogonalization rows, persist, and return the frame.

        Args:
            factor_combination: Canonical Factor Combination dataset.
            lineage: Validation window and source version provenance.
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            The Factor Orthogonalization DataFrame produced by the registry.
        """
        created = self._registry.build(factor_combination, lineage=lineage)
        self._repository.save(
            created,
            manager=manager,
            exchange=exchange,
            market=market,
            timeframe=timeframe,
            year=year,
        )
        return created
