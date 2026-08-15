"""CQROS Order Manager registry.

Purpose:
    Provide the authoritative in-memory catalog of available order-manager
    implementations for registration and lookup.

Responsibilities:
    - Register ``OrderManager`` instances by unique name
    - Provide deterministic lookup, existence checks, and ordered listing
    - Reject duplicates, blank names, and objects that do not implement
      ``OrderManager``
    - Remain free of order generation, execution, persistence, and trading

Dependencies:
    ``cqros.oms.exceptions`` and ``cqros.oms.interfaces``.

Public API:
    ``OrderManagerRegistry``

Notes:
    This registry is not thread-safe. Callers that share one instance across
    threads must provide their own synchronization.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from cqros.oms.exceptions import OMSValidationError
from cqros.oms.interfaces import OrderManager

__all__ = ["OrderManagerRegistry"]

_ERROR_NOT_MANAGER: Final[str] = "OMS_REG_NOT_MANAGER"
_ERROR_NAME_BLANK: Final[str] = "OMS_REG_NAME_BLANK"
_ERROR_DUPLICATE: Final[str] = "OMS_REG_DUPLICATE"
_ERROR_UNKNOWN: Final[str] = "OMS_REG_UNKNOWN"


class OrderManagerRegistry:
    """Authoritative catalog of registered CQROS order managers.

    Managers are indexed by caller-supplied unique names. The registry stores
    references to the supplied ``OrderManager`` instances and never mutates,
    instantiates, or invokes them. Returned name collections are new tuples
    and do not expose the internal mapping. Insertion order is preserved.

    Notes:
        This registry is not thread-safe. Concurrent mutation from multiple
        threads requires external synchronization.
    """

    __slots__ = ("_managers",)

    def __init__(self) -> None:
        """Initialize an empty order manager registry."""
        self._managers: dict[str, OrderManager] = {}

    def register(self, name: str, manager: OrderManager) -> None:
        """Register one order manager under ``name``.

        Args:
            name: Unique registry key for the manager.
            manager: Order manager instance to register. Must not be mutated
                by the registry after registration.

        Raises:
            OMSValidationError: If ``name`` is blank, ``manager`` does not
                implement ``OrderManager``, or a name is already registered.
        """
        validated_name = _require_name(name)
        validated_manager = _require_manager(manager)
        if validated_name in self._managers:
            raise OMSValidationError(
                f"manager already registered: {validated_name}",
                error_code=_ERROR_DUPLICATE,
                details={"name": validated_name},
            )
        self._managers[validated_name] = validated_manager

    def register_many(self, mapping: Mapping[str, OrderManager]) -> None:
        """Register multiple order managers atomically.

        Either every entry in ``mapping`` is registered, or the registry
        remains unchanged when any registration would fail.

        Args:
            mapping: Mapping of unique names to order manager instances.

        Raises:
            OMSValidationError: If any name is blank, already registered,
                duplicated within ``mapping``, or any value does not implement
                ``OrderManager``.
        """
        pending: dict[str, OrderManager] = {}
        for name, manager in mapping.items():
            validated_name = _require_name(name)
            validated_manager = _require_manager(manager)
            if validated_name in self._managers or validated_name in pending:
                raise OMSValidationError(
                    f"manager already registered: {validated_name}",
                    error_code=_ERROR_DUPLICATE,
                    details={"name": validated_name},
                )
            pending[validated_name] = validated_manager
        self._managers.update(pending)

    def get(self, name: str) -> OrderManager:
        """Return the registered order manager for ``name``.

        Args:
            name: Manager name to look up.

        Returns:
            The registered order manager instance.

        Raises:
            OMSValidationError: If no manager is registered under ``name``.
        """
        manager = self._managers.get(name)
        if manager is None:
            raise OMSValidationError(
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
    """Validate and return a non-blank manager registry name.

    Args:
        name: Candidate registry name.

    Returns:
        The validated name string.

    Raises:
        OMSValidationError: If ``name`` is not a non-blank string.
    """
    if not isinstance(name, str) or name.strip() == "":
        raise OMSValidationError(
            "manager name must be a non-blank string",
            error_code=_ERROR_NAME_BLANK,
            details={"name": name},
        )
    return name


def _require_manager(manager: object) -> OrderManager:
    """Validate that ``manager`` implements ``OrderManager``.

    Args:
        manager: Candidate order manager instance.

    Returns:
        The validated order manager.

    Raises:
        OMSValidationError: If ``manager`` does not implement ``OrderManager``.
    """
    if not isinstance(manager, OrderManager):
        raise OMSValidationError(
            "manager must implement the OrderManager protocol",
            error_code=_ERROR_NOT_MANAGER,
            details={"value_type": type(manager).__name__},
        )
    return manager
