"""CQROS Factor Research Engine registry.

Purpose:
    Provide the authoritative in-memory catalog of available research
    factors for registration and lookup.

Responsibilities:
    - Register immutable ``Factor`` instances by unique name
    - Enforce unique produced output columns across the catalog
    - Provide deterministic lookup, listing, category filtering, and
      metadata projection
    - Reject duplicate and blank factor names
    - Remain free of execution, IC calculation, dependency resolution,
      storage, pipeline, and dataframe logic

Dependencies:
    ``cqros.factors.exceptions``, ``cqros.factors.interfaces.Factor``, and
    ``cqros.factors.metadata.FactorMetadata``.

Public API:
    ``FactorRegistry``

Notes:
    This registry is not thread-safe. Callers that share one instance across
    threads must provide their own synchronization.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Final

from cqros.factors.exceptions import FactorRegistrationError, UnknownFactorError
from cqros.factors.interfaces import Factor
from cqros.factors.metadata import FactorMetadata

__all__ = ["FactorRegistry"]

_ERROR_NAME_BLANK: Final[str] = "FACTOR-REG-001"
_ERROR_DUPLICATE: Final[str] = "FACTOR-REG-002"
_ERROR_UNKNOWN: Final[str] = "FACTOR-REG-003"
_ERROR_DUPLICATE_COLUMN: Final[str] = "FACTOR-REG-004"


class FactorRegistry:
    """Authoritative catalog of registered CQROS research factors.

    Factors are indexed by name. Produced output columns are also unique
    across the catalog. The registry stores references to the supplied
    ``Factor`` instances and never mutates them. Returned collections are
    new tuples and do not expose the internal mapping.

    Notes:
        This registry is not thread-safe. Concurrent mutation from multiple
        threads requires external synchronization.
    """

    __slots__ = ("_factors", "_column_owners")

    def __init__(self) -> None:
        """Initialize an empty factor registry."""
        self._factors: dict[str, Factor] = {}
        self._column_owners: dict[str, str] = {}

    def register(self, factor: Factor) -> None:
        """Register one factor by name.

        Args:
            factor: Factor instance to register. Must not be mutated by the
                registry after registration.

        Raises:
            FactorRegistrationError: If ``factor.name`` is blank, a factor
                with the same name already exists, or any produced column is
                already claimed by another registered factor.
        """
        name = _require_factor_name(factor.name)
        if name in self._factors:
            raise FactorRegistrationError(
                f"factor already registered: {name}",
                error_code=_ERROR_DUPLICATE,
                details={"name": name},
            )
        columns = _require_unique_produced_columns(
            factor.produced_columns,
            factor_name=name,
            existing_owners=self._column_owners,
            pending_owners=None,
        )
        self._factors[name] = factor
        self._column_owners.update(columns)

    def register_many(self, factors: Iterable[Factor]) -> None:
        """Register multiple factors atomically.

        Either every factor in ``factors`` is registered, or the registry
        remains unchanged when any registration would fail.

        Args:
            factors: Factors to register.

        Raises:
            FactorRegistrationError: If any factor name is blank, already
                registered, duplicated within ``factors``, or any produced
                column conflicts with an existing or batch-local column.
        """
        pending: dict[str, Factor] = {}
        pending_columns: dict[str, str] = {}
        for factor in factors:
            name = _require_factor_name(factor.name)
            if name in self._factors or name in pending:
                raise FactorRegistrationError(
                    f"factor already registered: {name}",
                    error_code=_ERROR_DUPLICATE,
                    details={"name": name},
                )
            columns = _require_unique_produced_columns(
                factor.produced_columns,
                factor_name=name,
                existing_owners=self._column_owners,
                pending_owners=pending_columns,
            )
            pending[name] = factor
            pending_columns.update(columns)
        self._factors.update(pending)
        self._column_owners.update(pending_columns)

    def get(self, name: str) -> Factor:
        """Return the registered factor for ``name``.

        Args:
            name: Factor name to look up.

        Returns:
            The registered factor instance.

        Raises:
            UnknownFactorError: If no factor is registered under ``name``.
        """
        factor = self._factors.get(name)
        if factor is None:
            raise UnknownFactorError(
                f"factor not registered: {name}",
                error_code=_ERROR_UNKNOWN,
                details={"name": name},
            )
        return factor

    def exists(self, name: str) -> bool:
        """Return whether a factor is registered under ``name``.

        Args:
            name: Factor name to check.

        Returns:
            ``True`` when the name is registered, otherwise ``False``.
        """
        return name in self._factors

    def remove(self, name: str) -> None:
        """Remove a registered factor by name.

        Args:
            name: Factor name to remove.

        Raises:
            UnknownFactorError: If no factor is registered under ``name``.
        """
        factor = self._factors.get(name)
        if factor is None:
            raise UnknownFactorError(
                f"factor not registered: {name}",
                error_code=_ERROR_UNKNOWN,
                details={"name": name},
            )
        del self._factors[name]
        for column in factor.produced_columns:
            owner = self._column_owners.get(column)
            if owner == name:
                del self._column_owners[column]

    def clear(self) -> None:
        """Remove all registered factors."""
        self._factors.clear()
        self._column_owners.clear()

    def names(self) -> tuple[str, ...]:
        """Return registered factor names in alphabetical order.

        Returns:
            A new tuple of factor names.
        """
        return tuple(sorted(self._factors))

    def list(self) -> tuple[Factor, ...]:
        """Return registered factors sorted alphabetically by name.

        Returns:
            A new tuple of registered factor instances.
        """
        return tuple(self._factors[name] for name in sorted(self._factors))

    def by_category(self, category: str) -> tuple[Factor, ...]:
        """Return registered factors belonging to ``category``.

        Args:
            category: Factor category to filter on. Matching is exact and
                case-sensitive against each factor's ``category`` attribute.

        Returns:
            A new tuple of matching factors sorted alphabetically by name.
            Returns an empty tuple when no factors match.
        """
        return tuple(
            self._factors[name]
            for name in sorted(self._factors)
            if self._factors[name].category == category
        )

    def categories(self) -> tuple[str, ...]:
        """Return every registered category in alphabetical order.

        Returns:
            A new tuple of unique category names.
        """
        return tuple(sorted({factor.category for factor in self._factors.values()}))

    def metadata(self) -> tuple[FactorMetadata, ...]:
        """Return metadata snapshots for all registered factors.

        Metadata is projected from each factor's public attributes.

        Returns:
            A new tuple of ``FactorMetadata`` objects sorted alphabetically
            by factor name.
        """
        return tuple(_to_factor_metadata(factor) for factor in self.list())

    def metadata_for(self, name: str) -> FactorMetadata:
        """Return the metadata snapshot for one registered factor.

        Args:
            name: Factor name to look up.

        Returns:
            Immutable ``FactorMetadata`` for the registered factor.

        Raises:
            UnknownFactorError: If no factor is registered under ``name``.
        """
        return _to_factor_metadata(self.get(name))


def _require_factor_name(name: object) -> str:
    """Validate and return a non-blank factor name.

    Args:
        name: Candidate factor name.

    Returns:
        The validated factor name.

    Raises:
        FactorRegistrationError: If ``name`` is not a non-blank string.
    """
    if not isinstance(name, str) or name.strip() == "":
        raise FactorRegistrationError(
            "factor name must be a non-blank string",
            error_code=_ERROR_NAME_BLANK,
            details={"name": name},
        )
    return name


def _require_unique_produced_columns(
    produced_columns: Sequence[str],
    *,
    factor_name: str,
    existing_owners: dict[str, str],
    pending_owners: dict[str, str] | None,
) -> dict[str, str]:
    """Validate produced columns and return column-to-owner mappings.

    Args:
        produced_columns: Output columns claimed by the candidate factor.
        factor_name: Factor that owns ``produced_columns``.
        existing_owners: Column ownership already committed in the registry.
        pending_owners: Column ownership claimed earlier in the same batch,
            or ``None`` for single-factor registration.

    Returns:
        Mapping of each produced column to ``factor_name``.

    Raises:
        FactorRegistrationError: If any column is duplicated within the
            factor, within the batch, or against an already registered factor.
    """
    claimed: dict[str, str] = {}
    for column in produced_columns:
        if column in claimed:
            raise FactorRegistrationError(
                f"produced column already registered: {column}",
                error_code=_ERROR_DUPLICATE_COLUMN,
                details={
                    "column": column,
                    "name": factor_name,
                    "owner": factor_name,
                },
            )
        owner = existing_owners.get(column)
        if owner is None and pending_owners is not None:
            owner = pending_owners.get(column)
        if owner is not None:
            raise FactorRegistrationError(
                f"produced column already registered: {column}",
                error_code=_ERROR_DUPLICATE_COLUMN,
                details={
                    "column": column,
                    "name": factor_name,
                    "owner": owner,
                },
            )
        claimed[column] = factor_name
    return claimed


def _to_factor_metadata(factor: Factor) -> FactorMetadata:
    """Project a registered factor into immutable metadata."""
    return FactorMetadata(
        name=factor.name,
        version=factor.version,
        description=factor.description,
        category=factor.category,
        required_features=tuple(factor.required_features),
        produced_columns=tuple(factor.produced_columns),
        lookback=factor.lookback,
        factor_group=factor.factor_group,
        prediction_horizon=factor.prediction_horizon,
        enabled=factor.enabled,
        status=factor.status,
    )
