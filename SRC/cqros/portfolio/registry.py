"""CQROS Portfolio optimizer registry.

Purpose:
    Provide the authoritative in-memory catalog of available portfolio
    optimizer implementations for registration and lookup.

Responsibilities:
    - Register ``PortfolioOptimizer`` instances by unique name
    - Provide deterministic lookup, existence checks, and ordered listing
    - Reject duplicates, blank names, and objects that do not implement
      ``PortfolioOptimizer``
    - Remain free of optimization, allocation, persistence, and trading
      logic

Dependencies:
    ``cqros.portfolio.exceptions`` and ``cqros.portfolio.interfaces``.

Public API:
    ``PortfolioOptimizerRegistry``

Notes:
    This registry is not thread-safe. Callers that share one instance across
    threads must provide their own synchronization.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from cqros.portfolio.exceptions import PortfolioValidationError
from cqros.portfolio.interfaces import PortfolioOptimizer

__all__ = ["PortfolioOptimizerRegistry"]

_ERROR_NOT_OPTIMIZER: Final[str] = "PORTFOLIO_REG_NOT_OPTIMIZER"
_ERROR_NAME_BLANK: Final[str] = "PORTFOLIO_REG_NAME_BLANK"
_ERROR_DUPLICATE: Final[str] = "PORTFOLIO_REG_DUPLICATE"
_ERROR_UNKNOWN: Final[str] = "PORTFOLIO_REG_UNKNOWN"


class PortfolioOptimizerRegistry:
    """Authoritative catalog of registered CQROS portfolio optimizers.

    Optimizers are indexed by caller-supplied unique names. The registry
    stores references to the supplied ``PortfolioOptimizer`` instances and
    never mutates, instantiates, or invokes them. Returned name collections
    are new tuples and do not expose the internal mapping. Insertion order
    is preserved.

    Notes:
        This registry is not thread-safe. Concurrent mutation from multiple
        threads requires external synchronization.
    """

    __slots__ = ("_optimizers",)

    def __init__(self) -> None:
        """Initialize an empty portfolio optimizer registry."""
        self._optimizers: dict[str, PortfolioOptimizer] = {}

    def register(self, name: str, optimizer: PortfolioOptimizer) -> None:
        """Register one optimizer under ``name``.

        Args:
            name: Unique registry key for the optimizer.
            optimizer: Optimizer instance to register. Must not be mutated by
                the registry after registration.

        Raises:
            PortfolioValidationError: If ``name`` is blank, ``optimizer`` does
                not implement ``PortfolioOptimizer``, or a name is already
                registered.
        """
        validated_name = _require_name(name)
        validated_optimizer = _require_optimizer(optimizer)
        if validated_name in self._optimizers:
            raise PortfolioValidationError(
                f"optimizer already registered: {validated_name}",
                error_code=_ERROR_DUPLICATE,
                details={"name": validated_name},
            )
        self._optimizers[validated_name] = validated_optimizer

    def register_many(self, mapping: Mapping[str, PortfolioOptimizer]) -> None:
        """Register multiple optimizers atomically.

        Either every entry in ``mapping`` is registered, or the registry
        remains unchanged when any registration would fail.

        Args:
            mapping: Mapping of unique names to optimizer instances.

        Raises:
            PortfolioValidationError: If any name is blank, already
                registered, duplicated within ``mapping``, or any value does
                not implement ``PortfolioOptimizer``.
        """
        pending: dict[str, PortfolioOptimizer] = {}
        for name, optimizer in mapping.items():
            validated_name = _require_name(name)
            validated_optimizer = _require_optimizer(optimizer)
            if validated_name in self._optimizers or validated_name in pending:
                raise PortfolioValidationError(
                    f"optimizer already registered: {validated_name}",
                    error_code=_ERROR_DUPLICATE,
                    details={"name": validated_name},
                )
            pending[validated_name] = validated_optimizer
        self._optimizers.update(pending)

    def get(self, name: str) -> PortfolioOptimizer:
        """Return the registered optimizer for ``name``.

        Args:
            name: Optimizer name to look up.

        Returns:
            The registered optimizer instance.

        Raises:
            PortfolioValidationError: If no optimizer is registered under
                ``name``.
        """
        optimizer = self._optimizers.get(name)
        if optimizer is None:
            raise PortfolioValidationError(
                f"optimizer not registered: {name}",
                error_code=_ERROR_UNKNOWN,
                details={"name": name},
            )
        return optimizer

    def exists(self, name: str) -> bool:
        """Return whether an optimizer is registered under ``name``.

        Args:
            name: Optimizer name to check.

        Returns:
            ``True`` when the name is registered, otherwise ``False``.
        """
        return name in self._optimizers

    def list(self) -> tuple[str, ...]:
        """Return registered optimizer names in insertion order.

        Returns:
            A new tuple of registered names.
        """
        return tuple(self._optimizers)

    def clear(self) -> None:
        """Remove all registered optimizers."""
        self._optimizers.clear()


def _require_name(name: object) -> str:
    """Validate and return a non-blank optimizer registry name.

    Args:
        name: Candidate registry name.

    Returns:
        The validated name string.

    Raises:
        PortfolioValidationError: If ``name`` is not a non-blank string.
    """
    if not isinstance(name, str) or name.strip() == "":
        raise PortfolioValidationError(
            "optimizer name must be a non-blank string",
            error_code=_ERROR_NAME_BLANK,
            details={"name": name},
        )
    return name


def _require_optimizer(optimizer: object) -> PortfolioOptimizer:
    """Validate that ``optimizer`` implements ``PortfolioOptimizer``.

    Args:
        optimizer: Candidate optimizer instance.

    Returns:
        The validated optimizer.

    Raises:
        PortfolioValidationError: If ``optimizer`` does not implement
            ``PortfolioOptimizer``.
    """
    if not isinstance(optimizer, PortfolioOptimizer):
        raise PortfolioValidationError(
            "optimizer must implement the PortfolioOptimizer protocol",
            error_code=_ERROR_NOT_OPTIMIZER,
            details={"value_type": type(optimizer).__name__},
        )
    return optimizer
