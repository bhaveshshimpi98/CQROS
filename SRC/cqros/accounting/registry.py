"""CQROS Portfolio Accounting Engine registry.

Purpose:
    Provide the authoritative in-memory catalog of available accounting engine
    implementations for registration and lookup.

Responsibilities:
    - Register ``AccountingEngine`` instances by unique name
    - Provide deterministic lookup, existence checks, and ordered listing
    - Reject duplicates, blank names, and objects that do not implement
      ``AccountingEngine``
    - Remain free of accounting math, persistence, and trading

Dependencies:
    ``cqros.accounting.exceptions`` and ``cqros.accounting.engine``.

Public API:
    ``AccountingEngineRegistry``

Notes:
    This registry is not thread-safe. Callers that share one instance across
    threads must provide their own synchronization.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from cqros.accounting.engine import AccountingEngine
from cqros.accounting.exceptions import AccountingValidationError

__all__ = ["AccountingEngineRegistry"]

_ERROR_NOT_ENGINE: Final[str] = "ACC_REG_NOT_ENGINE"
_ERROR_NAME_BLANK: Final[str] = "ACC_REG_NAME_BLANK"
_ERROR_DUPLICATE: Final[str] = "ACC_REG_DUPLICATE"
_ERROR_UNKNOWN: Final[str] = "ACC_REG_UNKNOWN"


class AccountingEngineRegistry:
    """Authoritative catalog of registered CQROS accounting engines.

    Engines are indexed by caller-supplied unique names. The registry stores
    references to the supplied ``AccountingEngine`` instances and never mutates,
    instantiates, or invokes them. Returned name collections are new tuples
    and do not expose the internal mapping. Insertion order is preserved.

    Notes:
        This registry is not thread-safe. Concurrent mutation from multiple
        threads requires external synchronization.
    """

    __slots__ = ("_engines",)

    def __init__(self) -> None:
        """Initialize an empty accounting engine registry."""
        self._engines: dict[str, AccountingEngine] = {}

    def register(self, name: str, engine: AccountingEngine) -> None:
        """Register one accounting engine under ``name``.

        Args:
            name: Unique registry key for the engine.
            engine: Accounting engine instance to register. Must not be mutated
                by the registry after registration.

        Raises:
            AccountingValidationError: If ``name`` is blank, ``engine`` does not
                implement ``AccountingEngine``, or a name is already registered.
        """
        validated_name = _require_name(name)
        validated_engine = _require_engine(engine)
        if validated_name in self._engines:
            raise AccountingValidationError(
                f"engine already registered: {validated_name}",
                error_code=_ERROR_DUPLICATE,
                details={"name": validated_name},
            )
        self._engines[validated_name] = validated_engine

    def register_many(self, mapping: Mapping[str, AccountingEngine]) -> None:
        """Register multiple accounting engines atomically.

        Either every entry in ``mapping`` is registered, or the registry
        remains unchanged when any registration would fail.

        Args:
            mapping: Mapping of unique names to accounting engine instances.

        Raises:
            AccountingValidationError: If any name is blank, already registered,
                duplicated within ``mapping``, or any value does not implement
                ``AccountingEngine``.
        """
        pending: dict[str, AccountingEngine] = {}
        for name, engine in mapping.items():
            validated_name = _require_name(name)
            validated_engine = _require_engine(engine)
            if validated_name in self._engines or validated_name in pending:
                raise AccountingValidationError(
                    f"engine already registered: {validated_name}",
                    error_code=_ERROR_DUPLICATE,
                    details={"name": validated_name},
                )
            pending[validated_name] = validated_engine
        self._engines.update(pending)

    def get(self, name: str) -> AccountingEngine:
        """Return the registered accounting engine for ``name``.

        Args:
            name: Engine name to look up.

        Returns:
            The registered accounting engine instance.

        Raises:
            AccountingValidationError: If no engine is registered under ``name``.
        """
        engine = self._engines.get(name)
        if engine is None:
            raise AccountingValidationError(
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
        raise AccountingValidationError(
            "engine name must be a non-blank string",
            error_code=_ERROR_NAME_BLANK,
            details={"name": name},
        )
    return name


def _require_engine(engine: object) -> AccountingEngine:
    """Validate that ``engine`` implements ``AccountingEngine``."""
    if not isinstance(engine, AccountingEngine):
        raise AccountingValidationError(
            "engine must implement the AccountingEngine protocol",
            error_code=_ERROR_NOT_ENGINE,
            details={"value_type": type(engine).__name__},
        )
    return engine
