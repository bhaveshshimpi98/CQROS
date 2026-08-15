"""CQROS Exit Engine registry.

Purpose:
    Provide the authoritative in-memory catalog of available exit-engine
    implementations for registration and lookup.

Responsibilities:
    - Register ``ExitEngine`` instances by unique name
    - Provide deterministic lookup, existence checks, and ordered listing
    - Reject duplicates, blank names, and objects that do not implement
      ``ExitEngine``
    - Remain free of exit math, persistence, and trading

Dependencies:
    ``cqros.exit_engine.exceptions`` and ``cqros.exit_engine.engine``.

Public API:
    ``ExitEngineRegistry``

Notes:
    This registry is not thread-safe. Callers that share one instance across
    threads must provide their own synchronization.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from cqros.exit_engine.engine import ExitEngine
from cqros.exit_engine.exceptions import ExitEngineValidationError

__all__ = ["ExitEngineRegistry"]

_ERROR_NOT_ENGINE: Final[str] = "EXIT_REG_NOT_ENGINE"
_ERROR_NAME_BLANK: Final[str] = "EXIT_REG_NAME_BLANK"
_ERROR_DUPLICATE: Final[str] = "EXIT_REG_DUPLICATE"
_ERROR_UNKNOWN: Final[str] = "EXIT_REG_UNKNOWN"


class ExitEngineRegistry:
    """Authoritative catalog of registered CQROS exit engines.

    Engines are indexed by caller-supplied unique names. The registry stores
    references to the supplied ``ExitEngine`` instances and never mutates,
    instantiates, or invokes them. Returned name collections are new tuples
    and do not expose the internal mapping. Insertion order is preserved.

    Notes:
        This registry is not thread-safe. Concurrent mutation from multiple
        threads requires external synchronization.
    """

    __slots__ = ("_engines",)

    def __init__(self) -> None:
        """Initialize an empty exit-engine registry."""
        self._engines: dict[str, ExitEngine] = {}

    def register(self, name: str, engine: ExitEngine) -> None:
        """Register one exit engine under ``name``.

        Args:
            name: Unique registry key for the engine.
            engine: Exit engine instance to register. Must not be mutated by
                the registry after registration.

        Raises:
            ExitEngineValidationError: If ``name`` is blank, ``engine`` does
                not implement ``ExitEngine``, or a name is already registered.
        """
        validated_name = _require_name(name)
        validated_engine = _require_engine(engine)
        if validated_name in self._engines:
            raise ExitEngineValidationError(
                f"engine already registered: {validated_name}",
                error_code=_ERROR_DUPLICATE,
                details={"name": validated_name},
            )
        self._engines[validated_name] = validated_engine

    def register_many(self, mapping: Mapping[str, ExitEngine]) -> None:
        """Register multiple exit engines atomically.

        Either every entry in ``mapping`` is registered, or the registry
        remains unchanged when any registration would fail.

        Args:
            mapping: Mapping of unique names to exit engine instances.

        Raises:
            ExitEngineValidationError: If any name is blank, already
                registered, duplicated within ``mapping``, or any value does
                not implement ``ExitEngine``.
        """
        pending: dict[str, ExitEngine] = {}
        for name, engine in mapping.items():
            validated_name = _require_name(name)
            validated_engine = _require_engine(engine)
            if validated_name in self._engines or validated_name in pending:
                raise ExitEngineValidationError(
                    f"engine already registered: {validated_name}",
                    error_code=_ERROR_DUPLICATE,
                    details={"name": validated_name},
                )
            pending[validated_name] = validated_engine
        self._engines.update(pending)

    def get(self, name: str) -> ExitEngine:
        """Return the registered exit engine for ``name``.

        Args:
            name: Engine name to look up.

        Returns:
            The registered exit engine instance.

        Raises:
            ExitEngineValidationError: If no engine is registered under
                ``name``.
        """
        engine = self._engines.get(name)
        if engine is None:
            raise ExitEngineValidationError(
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

    def list(self) -> tuple[str, ...]:
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
        raise ExitEngineValidationError(
            "engine name must be a non-blank string",
            error_code=_ERROR_NAME_BLANK,
            details={"name": name},
        )
    return name


def _require_engine(engine: object) -> ExitEngine:
    """Validate that ``engine`` implements ``ExitEngine``."""
    if not isinstance(engine, ExitEngine):
        raise ExitEngineValidationError(
            "engine must implement the ExitEngine protocol",
            error_code=_ERROR_NOT_ENGINE,
            details={"value_type": type(engine).__name__},
        )
    return engine
