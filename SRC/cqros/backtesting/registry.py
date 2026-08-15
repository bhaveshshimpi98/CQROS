"""CQROS Backtesting Engine registry.

Purpose:
    Provide the authoritative in-memory catalog of available backtesting
    engine implementations for registration and lookup.

Responsibilities:
    - Register ``BacktestingEngine`` instances by unique name
    - Provide deterministic lookup, existence checks, and ordered listing
    - Reject duplicates, blank names, and objects that do not implement
      ``BacktestingEngine``
    - Remain free of performance math, persistence, and trading

Dependencies:
    ``cqros.backtesting.exceptions`` and ``cqros.backtesting.engine``.

Public API:
    ``BacktestingRegistry``

Notes:
    This registry is not thread-safe. Callers that share one instance across
    threads must provide their own synchronization.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from cqros.backtesting.engine import BacktestingEngine
from cqros.backtesting.exceptions import BacktestingValidationError

__all__ = ["BacktestingRegistry"]

_ERROR_NOT_ENGINE: Final[str] = "BT_REG_NOT_ENGINE"
_ERROR_NAME_BLANK: Final[str] = "BT_REG_NAME_BLANK"
_ERROR_DUPLICATE: Final[str] = "BT_REG_DUPLICATE"
_ERROR_UNKNOWN: Final[str] = "BT_REG_UNKNOWN"


class BacktestingRegistry:
    """Authoritative catalog of registered CQROS backtesting engines.

    Engines are indexed by caller-supplied unique names. The registry stores
    references to the supplied ``BacktestingEngine`` instances and never
    mutates, instantiates, or invokes them. Returned name collections are new
    tuples and do not expose the internal mapping. Insertion order is
    preserved.

    Notes:
        This registry is not thread-safe. Concurrent mutation from multiple
        threads requires external synchronization.
    """

    __slots__ = ("_engines",)

    def __init__(self) -> None:
        """Initialize an empty backtesting engine registry."""
        self._engines: dict[str, BacktestingEngine] = {}

    def register(self, name: str, engine: BacktestingEngine) -> None:
        """Register one backtesting engine under ``name``.

        Args:
            name: Unique registry key for the engine.
            engine: Backtesting engine instance to register. Must not be
                mutated by the registry after registration.

        Raises:
            BacktestingValidationError: If ``name`` is blank, ``engine`` does
                not implement ``BacktestingEngine``, or a name is already
                registered.
        """
        validated_name = _require_name(name)
        validated_engine = _require_engine(engine)
        if validated_name in self._engines:
            raise BacktestingValidationError(
                f"engine already registered: {validated_name}",
                error_code=_ERROR_DUPLICATE,
                details={"name": validated_name},
            )
        self._engines[validated_name] = validated_engine

    def register_many(self, mapping: Mapping[str, BacktestingEngine]) -> None:
        """Register multiple backtesting engines atomically.

        Either every entry in ``mapping`` is registered, or the registry
        remains unchanged when any registration would fail.

        Args:
            mapping: Mapping of unique names to backtesting engine instances.

        Raises:
            BacktestingValidationError: If any name is blank, already
                registered, duplicated within ``mapping``, or any value does
                not implement ``BacktestingEngine``.
        """
        pending: dict[str, BacktestingEngine] = {}
        for name, engine in mapping.items():
            validated_name = _require_name(name)
            validated_engine = _require_engine(engine)
            if validated_name in self._engines or validated_name in pending:
                raise BacktestingValidationError(
                    f"engine already registered: {validated_name}",
                    error_code=_ERROR_DUPLICATE,
                    details={"name": validated_name},
                )
            pending[validated_name] = validated_engine
        self._engines.update(pending)

    def get(self, name: str) -> BacktestingEngine:
        """Return the registered backtesting engine for ``name``.

        Args:
            name: Engine name to look up.

        Returns:
            The registered backtesting engine instance.

        Raises:
            BacktestingValidationError: If no engine is registered under
                ``name``.
        """
        engine = self._engines.get(name)
        if engine is None:
            raise BacktestingValidationError(
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
        raise BacktestingValidationError(
            "engine name must be a non-blank string",
            error_code=_ERROR_NAME_BLANK,
            details={"name": name},
        )
    return name


def _require_engine(engine: object) -> BacktestingEngine:
    """Validate that ``engine`` implements ``BacktestingEngine``."""
    if not isinstance(engine, BacktestingEngine):
        raise BacktestingValidationError(
            "engine must implement the BacktestingEngine protocol",
            error_code=_ERROR_NOT_ENGINE,
            details={"value_type": type(engine).__name__},
        )
    return engine
