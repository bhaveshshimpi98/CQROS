"""CQROS production executable-factor catalog.

Purpose:
    Select the subset of registered factors that can execute against a
    concrete set of available input columns.

Responsibilities:
    - Inspect each registered factor's ``required_features``
    - Return factors whose requirements are fully satisfied by the caller-
      supplied available columns
    - Preserve deterministic registry listing order
    - Remain free of factor compute logic, pipeline execution, persistence,
      CLI, allowlists, and hard-coded factor names

Dependencies:
    ``cqros.factors.interfaces.Factor`` and ``cqros.factors.registry``.

Public API:
    ``ExecutableFactorCatalog``
"""

from __future__ import annotations

from collections.abc import Sequence

from cqros.factors.interfaces import Factor
from cqros.factors.registry import FactorRegistry

__all__ = ["ExecutableFactorCatalog"]


class ExecutableFactorCatalog:
    """Filter a factor registry to factors executable on available columns.

    A factor is executable when every name in ``factor.required_features`` is
    present in the caller-supplied available column set. Selection is derived
    solely from factor metadata and available columns; no allowlist is kept.

    Args:
        registry: Authoritative production (or test) factor catalog.
    """

    __slots__ = ("_registry",)

    _registry: FactorRegistry

    def __init__(self, registry: FactorRegistry) -> None:
        """Initialize the catalog over ``registry``.

        Args:
            registry: Factor catalog providing ``list()`` for candidates.
        """
        self._registry = registry

    @property
    def registry(self) -> FactorRegistry:
        """Return the underlying factor registry."""
        return self._registry

    def get_executable_factors(
        self,
        available_columns: Sequence[str],
    ) -> tuple[Factor, ...]:
        """Return registered factors satisfiable by ``available_columns``.

        Args:
            available_columns: Column names present on the merged input frame
                (for example the processed-market factor-input join).

        Returns:
            Deterministic tuple of executable factors in ``registry.list()``
            order. Factors whose ``required_features`` are not a subset of
            ``available_columns`` are omitted.
        """
        available = frozenset(available_columns)
        return tuple(
            factor for factor in self._registry.list() if _requirements_satisfied(factor, available)
        )

    def get_skipped_factors(
        self,
        available_columns: Sequence[str],
    ) -> tuple[Factor, ...]:
        """Return registered factors that cannot execute on ``available_columns``.

        Args:
            available_columns: Column names present on the merged input frame.

        Returns:
            Deterministic tuple of skipped factors in ``registry.list()`` order.
        """
        available = frozenset(available_columns)
        return tuple(
            factor
            for factor in self._registry.list()
            if not _requirements_satisfied(factor, available)
        )


def _requirements_satisfied(factor: Factor, available: frozenset[str]) -> bool:
    """Return whether every required feature of ``factor`` is in ``available``."""
    return all(feature in available for feature in factor.required_features)
