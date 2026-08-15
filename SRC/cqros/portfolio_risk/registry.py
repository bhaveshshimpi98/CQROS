"""CQROS Portfolio Risk Manager registry.

Purpose:
    Provide the authoritative in-memory catalog of available portfolio risk
    manager implementations for registration and lookup.

Responsibilities:
    - Register ``PortfolioRiskManager`` instances by unique name
    - Provide deterministic lookup, existence checks, and ordered listing
    - Reject duplicates, blank names, and objects that do not implement
      ``PortfolioRiskManager``
    - Remain free of portfolio-risk math, persistence, and trading

Dependencies:
    ``cqros.portfolio_risk.exceptions`` and ``cqros.portfolio_risk.manager``.

Public API:
    ``PortfolioRiskManagerRegistry``

Notes:
    This registry is not thread-safe. Callers that share one instance across
    threads must provide their own synchronization.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from cqros.portfolio_risk.exceptions import PortfolioRiskValidationError
from cqros.portfolio_risk.manager import PortfolioRiskManager

__all__ = ["PortfolioRiskManagerRegistry"]

_ERROR_NOT_MANAGER: Final[str] = "PRISK_REG_NOT_MANAGER"
_ERROR_NAME_BLANK: Final[str] = "PRISK_REG_NAME_BLANK"
_ERROR_DUPLICATE: Final[str] = "PRISK_REG_DUPLICATE"
_ERROR_UNKNOWN: Final[str] = "PRISK_REG_UNKNOWN"


class PortfolioRiskManagerRegistry:
    """Authoritative catalog of registered CQROS portfolio risk managers.

    Managers are indexed by caller-supplied unique names. The registry stores
    references to the supplied ``PortfolioRiskManager`` instances and never
    mutates, instantiates, or invokes them. Returned name collections are new
    tuples and do not expose the internal mapping. Insertion order is
    preserved.

    Notes:
        This registry is not thread-safe. Concurrent mutation from multiple
        threads requires external synchronization.
    """

    __slots__ = ("_managers",)

    def __init__(self) -> None:
        """Initialize an empty portfolio risk manager registry."""
        self._managers: dict[str, PortfolioRiskManager] = {}

    def register(self, name: str, manager: PortfolioRiskManager) -> None:
        """Register one portfolio risk manager under ``name``.

        Args:
            name: Unique registry key for the manager.
            manager: Portfolio risk manager instance to register. Must not be
                mutated by the registry after registration.

        Raises:
            PortfolioRiskValidationError: If ``name`` is blank, ``manager`` does
                not implement ``PortfolioRiskManager``, or a name is already
                registered.
        """
        validated_name = _require_name(name)
        validated_manager = _require_manager(manager)
        if validated_name in self._managers:
            raise PortfolioRiskValidationError(
                f"manager already registered: {validated_name}",
                error_code=_ERROR_DUPLICATE,
                details={"name": validated_name},
            )
        self._managers[validated_name] = validated_manager

    def register_many(self, mapping: Mapping[str, PortfolioRiskManager]) -> None:
        """Register multiple portfolio risk managers atomically.

        Either every entry in ``mapping`` is registered, or the registry
        remains unchanged when any registration would fail.

        Args:
            mapping: Mapping of unique names to portfolio risk manager instances.

        Raises:
            PortfolioRiskValidationError: If any name is blank, already
                registered, duplicated within ``mapping``, or any value does
                not implement ``PortfolioRiskManager``.
        """
        pending: dict[str, PortfolioRiskManager] = {}
        for name, manager in mapping.items():
            validated_name = _require_name(name)
            validated_manager = _require_manager(manager)
            if validated_name in self._managers or validated_name in pending:
                raise PortfolioRiskValidationError(
                    f"manager already registered: {validated_name}",
                    error_code=_ERROR_DUPLICATE,
                    details={"name": validated_name},
                )
            pending[validated_name] = validated_manager
        self._managers.update(pending)

    def get(self, name: str) -> PortfolioRiskManager:
        """Return the registered portfolio risk manager for ``name``.

        Args:
            name: Manager name to look up.

        Returns:
            The registered portfolio risk manager instance.

        Raises:
            PortfolioRiskValidationError: If no manager is registered under
                ``name``.
        """
        manager = self._managers.get(name)
        if manager is None:
            raise PortfolioRiskValidationError(
                f"manager not registered: {name}",
                error_code=_ERROR_UNKNOWN,
                details={"name": name},
            )
        return manager

    def exists(self, name: str) -> bool:
        """Return whether a manager is registered under ``name``.

        Args:
            name: Manager name to check.

        Returns:
            ``True`` when the name is registered, otherwise ``False``.
        """
        return name in self._managers

    def list(self) -> tuple[str, ...]:
        """Return registered manager names in insertion order.

        Returns:
            A new tuple of registered names.
        """
        return tuple(self._managers)

    def clear(self) -> None:
        """Remove all registered managers."""
        self._managers.clear()


def _require_name(name: object) -> str:
    """Validate and return a non-blank manager registry name."""
    if not isinstance(name, str) or name.strip() == "":
        raise PortfolioRiskValidationError(
            "manager name must be a non-blank string",
            error_code=_ERROR_NAME_BLANK,
            details={"name": name},
        )
    return name


def _require_manager(manager: object) -> PortfolioRiskManager:
    """Validate that ``manager`` implements ``PortfolioRiskManager``."""
    if not isinstance(manager, PortfolioRiskManager):
        raise PortfolioRiskValidationError(
            "manager must implement the PortfolioRiskManager protocol",
            error_code=_ERROR_NOT_MANAGER,
            details={"value_type": type(manager).__name__},
        )
    return manager
