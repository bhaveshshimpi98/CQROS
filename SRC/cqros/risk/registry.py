"""CQROS Risk policy registry.

Purpose:
    Provide the authoritative in-memory catalog of available risk-manager
    implementations for registration and lookup.

Responsibilities:
    - Register ``RiskManager`` instances by unique name
    - Provide deterministic lookup, existence checks, and ordered listing
    - Reject duplicates, blank names, and objects that do not implement
      ``RiskManager``
    - Remain free of policy logic, calculations, persistence, and trading

Dependencies:
    ``cqros.risk.exceptions`` and ``cqros.risk.interfaces``.

Public API:
    ``RiskPolicyRegistry``

Notes:
    This registry is not thread-safe. Callers that share one instance across
    threads must provide their own synchronization.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from cqros.risk.exceptions import RiskValidationError
from cqros.risk.interfaces import RiskManager

__all__ = ["RiskPolicyRegistry"]

_ERROR_NOT_MANAGER: Final[str] = "RISK_REG_NOT_MANAGER"
_ERROR_NAME_BLANK: Final[str] = "RISK_REG_NAME_BLANK"
_ERROR_DUPLICATE: Final[str] = "RISK_REG_DUPLICATE"
_ERROR_UNKNOWN: Final[str] = "RISK_REG_UNKNOWN"


class RiskPolicyRegistry:
    """Authoritative catalog of registered CQROS risk managers.

    Policies are indexed by caller-supplied unique names. The registry stores
    references to the supplied ``RiskManager`` instances and never mutates,
    instantiates, or invokes them. Returned name collections are new tuples
    and do not expose the internal mapping. Insertion order is preserved.

    Notes:
        This registry is not thread-safe. Concurrent mutation from multiple
        threads requires external synchronization.
    """

    __slots__ = ("_policies",)

    def __init__(self) -> None:
        """Initialize an empty risk policy registry."""
        self._policies: dict[str, RiskManager] = {}

    def register(self, name: str, policy: RiskManager) -> None:
        """Register one risk manager under ``name``.

        Args:
            name: Unique registry key for the policy.
            policy: Risk manager instance to register. Must not be mutated by
                the registry after registration.

        Raises:
            RiskValidationError: If ``name`` is blank, ``policy`` does not
                implement ``RiskManager``, or a name is already registered.
        """
        validated_name = _require_name(name)
        validated_policy = _require_policy(policy)
        if validated_name in self._policies:
            raise RiskValidationError(
                f"policy already registered: {validated_name}",
                error_code=_ERROR_DUPLICATE,
                details={"name": validated_name},
            )
        self._policies[validated_name] = validated_policy

    def register_many(self, mapping: Mapping[str, RiskManager]) -> None:
        """Register multiple risk managers atomically.

        Either every entry in ``mapping`` is registered, or the registry
        remains unchanged when any registration would fail.

        Args:
            mapping: Mapping of unique names to risk manager instances.

        Raises:
            RiskValidationError: If any name is blank, already registered,
                duplicated within ``mapping``, or any value does not implement
                ``RiskManager``.
        """
        pending: dict[str, RiskManager] = {}
        for name, policy in mapping.items():
            validated_name = _require_name(name)
            validated_policy = _require_policy(policy)
            if validated_name in self._policies or validated_name in pending:
                raise RiskValidationError(
                    f"policy already registered: {validated_name}",
                    error_code=_ERROR_DUPLICATE,
                    details={"name": validated_name},
                )
            pending[validated_name] = validated_policy
        self._policies.update(pending)

    def get(self, name: str) -> RiskManager:
        """Return the registered risk manager for ``name``.

        Args:
            name: Policy name to look up.

        Returns:
            The registered risk manager instance.

        Raises:
            RiskValidationError: If no policy is registered under ``name``.
        """
        policy = self._policies.get(name)
        if policy is None:
            raise RiskValidationError(
                f"policy not registered: {name}",
                error_code=_ERROR_UNKNOWN,
                details={"name": name},
            )
        return policy

    def exists(self, name: str) -> bool:
        """Return whether a policy is registered under ``name``.

        Args:
            name: Policy name to check.

        Returns:
            ``True`` when the name is registered, otherwise ``False``.
        """
        return name in self._policies

    def list(self) -> tuple[str, ...]:
        """Return registered policy names in insertion order.

        Returns:
            A new tuple of registered names.
        """
        return tuple(self._policies)

    def clear(self) -> None:
        """Remove all registered policies."""
        self._policies.clear()


def _require_name(name: object) -> str:
    """Validate and return a non-blank policy registry name.

    Args:
        name: Candidate registry name.

    Returns:
        The validated name string.

    Raises:
        RiskValidationError: If ``name`` is not a non-blank string.
    """
    if not isinstance(name, str) or name.strip() == "":
        raise RiskValidationError(
            "policy name must be a non-blank string",
            error_code=_ERROR_NAME_BLANK,
            details={"name": name},
        )
    return name


def _require_policy(policy: object) -> RiskManager:
    """Validate that ``policy`` implements ``RiskManager``.

    Args:
        policy: Candidate risk manager instance.

    Returns:
        The validated risk manager.

    Raises:
        RiskValidationError: If ``policy`` does not implement ``RiskManager``.
    """
    if not isinstance(policy, RiskManager):
        raise RiskValidationError(
            "policy must implement the RiskManager protocol",
            error_code=_ERROR_NOT_MANAGER,
            details={"value_type": type(policy).__name__},
        )
    return policy
