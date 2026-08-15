"""CQROS Execution Simulator registry.

Purpose:
    Provide the authoritative in-memory catalog of available execution
    simulator implementations for registration and lookup.

Responsibilities:
    - Register ``ExecutionSimulator`` instances by unique name
    - Provide deterministic lookup, existence checks, and ordered listing
    - Reject duplicates, blank names, and objects that do not implement
      ``ExecutionSimulator``
    - Remain free of simulation, persistence, and trading

Dependencies:
    ``cqros.execution.exceptions`` and ``cqros.execution.simulator``.

Public API:
    ``ExecutionSimulatorRegistry``

Notes:
    This registry is not thread-safe. Callers that share one instance across
    threads must provide their own synchronization.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from cqros.execution.exceptions import ExecutionValidationError
from cqros.execution.simulator import ExecutionSimulator

__all__ = ["ExecutionSimulatorRegistry"]

_ERROR_NOT_SIMULATOR: Final[str] = "EXEC_REG_NOT_SIMULATOR"
_ERROR_NAME_BLANK: Final[str] = "EXEC_REG_NAME_BLANK"
_ERROR_DUPLICATE: Final[str] = "EXEC_REG_DUPLICATE"
_ERROR_UNKNOWN: Final[str] = "EXEC_REG_UNKNOWN"


class ExecutionSimulatorRegistry:
    """Authoritative catalog of registered CQROS execution simulators.

    Simulators are indexed by caller-supplied unique names. The registry
    stores references to the supplied ``ExecutionSimulator`` instances and
    never mutates, instantiates, or invokes them. Returned name collections
    are new tuples and do not expose the internal mapping. Insertion order
    is preserved.

    Notes:
        This registry is not thread-safe. Concurrent mutation from multiple
        threads requires external synchronization.
    """

    __slots__ = ("_simulators",)

    def __init__(self) -> None:
        """Initialize an empty execution simulator registry."""
        self._simulators: dict[str, ExecutionSimulator] = {}

    def register(self, name: str, simulator: ExecutionSimulator) -> None:
        """Register one execution simulator under ``name``.

        Args:
            name: Unique registry key for the simulator.
            simulator: Execution simulator instance to register. Must not be
                mutated by the registry after registration.

        Raises:
            ExecutionValidationError: If ``name`` is blank, ``simulator`` does
                not implement ``ExecutionSimulator``, or a name is already
                registered.
        """
        validated_name = _require_name(name)
        validated_simulator = _require_simulator(simulator)
        if validated_name in self._simulators:
            raise ExecutionValidationError(
                f"simulator already registered: {validated_name}",
                error_code=_ERROR_DUPLICATE,
                details={"name": validated_name},
            )
        self._simulators[validated_name] = validated_simulator

    def register_many(self, mapping: Mapping[str, ExecutionSimulator]) -> None:
        """Register multiple execution simulators atomically.

        Either every entry in ``mapping`` is registered, or the registry
        remains unchanged when any registration would fail.

        Args:
            mapping: Mapping of unique names to execution simulator instances.

        Raises:
            ExecutionValidationError: If any name is blank, already
                registered, duplicated within ``mapping``, or any value does
                not implement ``ExecutionSimulator``.
        """
        pending: dict[str, ExecutionSimulator] = {}
        for name, simulator in mapping.items():
            validated_name = _require_name(name)
            validated_simulator = _require_simulator(simulator)
            if validated_name in self._simulators or validated_name in pending:
                raise ExecutionValidationError(
                    f"simulator already registered: {validated_name}",
                    error_code=_ERROR_DUPLICATE,
                    details={"name": validated_name},
                )
            pending[validated_name] = validated_simulator
        self._simulators.update(pending)

    def get(self, name: str) -> ExecutionSimulator:
        """Return the registered execution simulator for ``name``.

        Args:
            name: Simulator name to look up.

        Returns:
            The registered execution simulator instance.

        Raises:
            ExecutionValidationError: If no simulator is registered under
                ``name``.
        """
        simulator = self._simulators.get(name)
        if simulator is None:
            raise ExecutionValidationError(
                f"simulator not registered: {name}",
                error_code=_ERROR_UNKNOWN,
                details={"name": name},
            )
        return simulator

    def exists(self, name: str) -> bool:
        """Return whether a simulator is registered under ``name``.

        Args:
            name: Simulator name to check.

        Returns:
            ``True`` when the name is registered, otherwise ``False``.
        """
        return name in self._simulators

    def list(self) -> tuple[str, ...]:
        """Return registered simulator names in insertion order.

        Returns:
            A new tuple of registered names.
        """
        return tuple(self._simulators)

    def clear(self) -> None:
        """Remove all registered simulators."""
        self._simulators.clear()


def _require_name(name: object) -> str:
    """Validate and return a non-blank simulator registry name.

    Args:
        name: Candidate registry name.

    Returns:
        The validated name string.

    Raises:
        ExecutionValidationError: If ``name`` is not a non-blank string.
    """
    if not isinstance(name, str) or name.strip() == "":
        raise ExecutionValidationError(
            "simulator name must be a non-blank string",
            error_code=_ERROR_NAME_BLANK,
            details={"name": name},
        )
    return name


def _require_simulator(simulator: object) -> ExecutionSimulator:
    """Validate that ``simulator`` implements ``ExecutionSimulator``.

    Args:
        simulator: Candidate execution simulator instance.

    Returns:
        The validated execution simulator.

    Raises:
        ExecutionValidationError: If ``simulator`` does not implement
            ``ExecutionSimulator``.
    """
    if not isinstance(simulator, ExecutionSimulator):
        raise ExecutionValidationError(
            "simulator must implement the ExecutionSimulator protocol",
            error_code=_ERROR_NOT_SIMULATOR,
            details={"value_type": type(simulator).__name__},
        )
    return simulator
