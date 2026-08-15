"""CQROS Signal policy registry.

Purpose:
    Provide the authoritative in-memory catalog of available signal-policy
    implementations for registration and lookup.

Responsibilities:
    - Register ``SignalPolicy`` instances by unique name
    - Provide deterministic lookup, existence checks, and ordered listing
    - Reject duplicates, blank names, and objects that do not implement
      ``SignalPolicy``
    - Remain free of policy logic, signal generation, persistence, and trading

Dependencies:
    ``cqros.signals.exceptions`` and ``cqros.signals.interfaces``.

Public API:
    ``SignalPolicyRegistry``

Notes:
    This registry is not thread-safe. Callers that share one instance across
    threads must provide their own synchronization.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from cqros.signals.exceptions import SignalValidationError
from cqros.signals.interfaces import SignalPolicy

__all__ = ["SignalPolicyRegistry"]

_ERROR_NOT_POLICY: Final[str] = "SIGNAL_REG_NOT_POLICY"
_ERROR_NAME_BLANK: Final[str] = "SIGNAL_REG_NAME_BLANK"
_ERROR_DUPLICATE: Final[str] = "SIGNAL_REG_DUPLICATE"
_ERROR_UNKNOWN: Final[str] = "SIGNAL_REG_UNKNOWN"


class SignalPolicyRegistry:
    """Authoritative catalog of registered CQROS signal policies.

    Policies are indexed by caller-supplied unique names. The registry stores
    references to the supplied ``SignalPolicy`` instances and never mutates,
    instantiates, or invokes them. Returned name collections are new tuples
    and do not expose the internal mapping. Insertion order is preserved.

    Notes:
        This registry is not thread-safe. Concurrent mutation from multiple
        threads requires external synchronization.
    """

    __slots__ = ("_policies",)

    def __init__(self) -> None:
        """Initialize an empty signal policy registry."""
        self._policies: dict[str, SignalPolicy] = {}

    def register(self, name: str, policy: SignalPolicy) -> None:
        """Register one signal policy under ``name``.

        Args:
            name: Unique registry key for the policy.
            policy: Signal policy instance to register. Must not be mutated by
                the registry after registration.

        Raises:
            SignalValidationError: If ``name`` is blank, ``policy`` does not
                implement ``SignalPolicy``, or a name is already registered.
        """
        validated_name = _require_name(name)
        validated_policy = _require_policy(policy)
        if validated_name in self._policies:
            raise SignalValidationError(
                f"policy already registered: {validated_name}",
                error_code=_ERROR_DUPLICATE,
                details={"name": validated_name},
            )
        self._policies[validated_name] = validated_policy

    def register_many(self, mapping: Mapping[str, SignalPolicy]) -> None:
        """Register multiple signal policies atomically.

        Either every entry in ``mapping`` is registered, or the registry
        remains unchanged when any registration would fail.

        Args:
            mapping: Mapping of unique names to signal policy instances.

        Raises:
            SignalValidationError: If any name is blank, already registered,
                duplicated within ``mapping``, or any value does not implement
                ``SignalPolicy``.
        """
        pending: dict[str, SignalPolicy] = {}
        for name, policy in mapping.items():
            validated_name = _require_name(name)
            validated_policy = _require_policy(policy)
            if validated_name in self._policies or validated_name in pending:
                raise SignalValidationError(
                    f"policy already registered: {validated_name}",
                    error_code=_ERROR_DUPLICATE,
                    details={"name": validated_name},
                )
            pending[validated_name] = validated_policy
        self._policies.update(pending)

    def get(self, name: str) -> SignalPolicy:
        """Return the registered signal policy for ``name``.

        Args:
            name: Policy name to look up.

        Returns:
            The registered signal policy instance.

        Raises:
            SignalValidationError: If no policy is registered under ``name``.
        """
        policy = self._policies.get(name)
        if policy is None:
            raise SignalValidationError(
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
        SignalValidationError: If ``name`` is not a non-blank string.
    """
    if not isinstance(name, str) or name.strip() == "":
        raise SignalValidationError(
            "policy name must be a non-blank string",
            error_code=_ERROR_NAME_BLANK,
            details={"name": name},
        )
    return name


def _require_policy(policy: object) -> SignalPolicy:
    """Validate that ``policy`` implements ``SignalPolicy``.

    Args:
        policy: Candidate signal policy instance.

    Returns:
        The validated signal policy.

    Raises:
        SignalValidationError: If ``policy`` does not implement ``SignalPolicy``.
    """
    if not isinstance(policy, SignalPolicy):
        raise SignalValidationError(
            "policy must implement the SignalPolicy protocol",
            error_code=_ERROR_NOT_POLICY,
            details={"value_type": type(policy).__name__},
        )
    return policy
