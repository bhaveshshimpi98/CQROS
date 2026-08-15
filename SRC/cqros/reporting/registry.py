"""CQROS Reporting Engine registry.

Purpose:
    Provide the authoritative in-memory catalog of available reporting
    engine implementations for registration and lookup.

Responsibilities:
    - Register ``ReportingEngine`` instances by unique name
    - Provide deterministic lookup, existence checks, and ordered listing
    - Reject duplicates, blank names, and objects that do not implement
      ``ReportingEngine``
    - Remain free of reporting generation, persistence, and trading

Dependencies:
    ``cqros.reporting.engine`` and ``cqros.reporting.exceptions``.

Public API:
    ``ReportingEngineRegistry``

Notes:
    This registry is not thread-safe. Callers that share one instance across
    threads must provide their own synchronization.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from cqros.reporting.engine import ReportingEngine
from cqros.reporting.exceptions import ReportingValidationError

__all__ = ["ReportingEngineRegistry"]

_ERROR_NOT_ENGINE: Final[str] = "REP_REG_NOT_ENGINE"
_ERROR_NAME_BLANK: Final[str] = "REP_REG_NAME_BLANK"
_ERROR_DUPLICATE: Final[str] = "REP_REG_DUPLICATE"
_ERROR_UNKNOWN: Final[str] = "REP_REG_UNKNOWN"


class ReportingEngineRegistry:
    """Authoritative catalog of registered CQROS reporting engines.

    Engines are indexed by caller-supplied unique names. The registry stores
    references to the supplied ``ReportingEngine`` instances and never
    mutates, instantiates, or invokes them. Returned name collections are new
    tuples and do not expose the internal mapping. Insertion order is
    preserved.

    Notes:
        This registry is not thread-safe. Concurrent mutation from multiple
        threads requires external synchronization.
    """

    __slots__ = ("_engines",)

    def __init__(self) -> None:
        """Initialize an empty reporting engine registry."""
        self._engines: dict[str, ReportingEngine] = {}

    def register(self, name: str, engine: ReportingEngine) -> None:
        """Register one reporting engine under ``name``.

        Args:
            name: Unique registry key for the engine.
            engine: Reporting engine instance to register. Must not be
                mutated by the registry after registration.

        Raises:
            ReportingValidationError: If ``name`` is blank, ``engine`` does
                not implement ``ReportingEngine``, or a name is already
                registered.
        """
        validated_name = _require_name(name)
        validated_engine = _require_engine(engine)
        if validated_name in self._engines:
            raise ReportingValidationError(
                f"engine already registered: {validated_name}",
                error_code=_ERROR_DUPLICATE,
                details={"name": validated_name},
            )
        self._engines[validated_name] = validated_engine

    def register_many(self, mapping: Mapping[str, ReportingEngine]) -> None:
        """Register multiple reporting engines atomically.

        Either every entry in ``mapping`` is registered, or the registry
        remains unchanged when any registration would fail.

        Args:
            mapping: Mapping of unique names to reporting engine instances.

        Raises:
            ReportingValidationError: If any name is blank, already
                registered, duplicated within ``mapping``, or any value does
                not implement ``ReportingEngine``.
        """
        pending: dict[str, ReportingEngine] = {}
        for name, engine in mapping.items():
            validated_name = _require_name(name)
            validated_engine = _require_engine(engine)
            if validated_name in self._engines or validated_name in pending:
                raise ReportingValidationError(
                    f"engine already registered: {validated_name}",
                    error_code=_ERROR_DUPLICATE,
                    details={"name": validated_name},
                )
            pending[validated_name] = validated_engine
        self._engines.update(pending)

    def get(self, name: str) -> ReportingEngine:
        """Return the registered reporting engine for ``name``.

        Args:
            name: Engine name to look up.

        Returns:
            The registered reporting engine instance.

        Raises:
            ReportingValidationError: If no engine is registered under
                ``name``.
        """
        engine = self._engines.get(name)
        if engine is None:
            raise ReportingValidationError(
                f"engine not registered: {name}",
                error_code=_ERROR_UNKNOWN,
                details={"name": name},
            )
        return engine

    def exists(self, name: str) -> bool:
        """Return whether an engine is registered under ``name``.

        Args:
            name: Engine name to check.

        Returns:
            ``True`` when the name is registered, otherwise ``False``.
        """
        return name in self._engines

    def names(self) -> tuple[str, ...]:
        """Return registered engine names in insertion order.

        Returns:
            A new tuple of registered names.
        """
        return tuple(self._engines)

    def clear(self) -> None:
        """Remove all registered engines."""
        self._engines.clear()


def _require_name(name: object) -> str:
    """Validate and return a non-blank engine registry name."""
    if not isinstance(name, str) or name.strip() == "":
        raise ReportingValidationError(
            "engine name must be a non-blank string",
            error_code=_ERROR_NAME_BLANK,
            details={"name": name},
        )
    return name


def _require_engine(engine: object) -> ReportingEngine:
    """Validate that ``engine`` implements ``ReportingEngine``."""
    if not isinstance(engine, ReportingEngine):
        raise ReportingValidationError(
            "engine must implement the ReportingEngine protocol",
            error_code=_ERROR_NOT_ENGINE,
            details={"value_type": type(engine).__name__},
        )
    return engine
