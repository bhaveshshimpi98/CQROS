"""CQROS Factor Timeframe Analysis Engine registry.

Purpose:
    Provide the authoritative in-memory catalog of available factor
    timeframe analysis engine implementations for registration and lookup.

Responsibilities:
    - Register ``FactorTimeframeAnalysisEngine`` instances by unique name
    - Provide deterministic lookup, existence checks, and ordered listing
    - Reject duplicates, blank names, and objects that do not implement
      ``FactorTimeframeAnalysisEngine``
    - Remain free of analysis math, persistence, and trading

Dependencies:
    ``cqros.factor_timeframe_analysis.engine`` and
    ``cqros.factor_timeframe_analysis.exceptions``.

Public API:
    ``FactorTimeframeAnalysisEngineRegistry``

Notes:
    This registry is not thread-safe. Callers that share one instance across
    threads must provide their own synchronization.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from cqros.factor_timeframe_analysis.engine import FactorTimeframeAnalysisEngine
from cqros.factor_timeframe_analysis.exceptions import FactorTimeframeAnalysisError

__all__ = ["FactorTimeframeAnalysisEngineRegistry"]

_ERROR_NOT_ENGINE: Final[str] = "FTA_REG_NOT_ENGINE"
_ERROR_NAME_BLANK: Final[str] = "FTA_REG_NAME_BLANK"
_ERROR_DUPLICATE: Final[str] = "FTA_REG_DUPLICATE"
_ERROR_UNKNOWN: Final[str] = "FTA_REG_UNKNOWN"


class FactorTimeframeAnalysisEngineRegistry:
    """Authoritative catalog of registered CQROS timeframe analysis engines.

    Engines are indexed by caller-supplied unique names. The registry stores
    references to the supplied ``FactorTimeframeAnalysisEngine`` instances
    and never mutates, instantiates, or invokes them. Returned name
    collections are new tuples and do not expose the internal mapping.
    Insertion order is preserved.

    Notes:
        This registry is not thread-safe. Concurrent mutation from multiple
        threads requires external synchronization.
    """

    __slots__ = ("_engines",)

    def __init__(self) -> None:
        """Initialize an empty factor timeframe analysis engine registry."""
        self._engines: dict[str, FactorTimeframeAnalysisEngine] = {}

    def register(self, name: str, engine: FactorTimeframeAnalysisEngine) -> None:
        """Register one timeframe analysis engine under ``name``.

        Args:
            name: Unique registry key for the engine.
            engine: Timeframe analysis engine instance to register. Must not
                be mutated by the registry after registration.

        Raises:
            FactorTimeframeAnalysisError: If ``name`` is blank, ``engine``
                does not implement ``FactorTimeframeAnalysisEngine``, or a
                name is already registered.
        """
        validated_name = _require_name(name)
        validated_engine = _require_engine(engine)
        if validated_name in self._engines:
            raise FactorTimeframeAnalysisError(
                f"engine already registered: {validated_name}",
                error_code=_ERROR_DUPLICATE,
                details={"name": validated_name},
            )
        self._engines[validated_name] = validated_engine

    def register_many(self, mapping: Mapping[str, FactorTimeframeAnalysisEngine]) -> None:
        """Register multiple timeframe analysis engines atomically.

        Either every entry in ``mapping`` is registered, or the registry
        remains unchanged when any registration would fail.

        Args:
            mapping: Mapping of unique names to timeframe analysis engine
                instances.

        Raises:
            FactorTimeframeAnalysisError: If any name is blank, already
                registered, duplicated within ``mapping``, or any value does
                not implement ``FactorTimeframeAnalysisEngine``.
        """
        pending: dict[str, FactorTimeframeAnalysisEngine] = {}
        for name, engine in mapping.items():
            validated_name = _require_name(name)
            validated_engine = _require_engine(engine)
            if validated_name in self._engines or validated_name in pending:
                raise FactorTimeframeAnalysisError(
                    f"engine already registered: {validated_name}",
                    error_code=_ERROR_DUPLICATE,
                    details={"name": validated_name},
                )
            pending[validated_name] = validated_engine
        self._engines.update(pending)

    def get(self, name: str) -> FactorTimeframeAnalysisEngine:
        """Return the registered timeframe analysis engine for ``name``.

        Args:
            name: Engine name to look up.

        Returns:
            The registered timeframe analysis engine instance.

        Raises:
            FactorTimeframeAnalysisError: If no engine is registered under
                ``name``.
        """
        engine = self._engines.get(name)
        if engine is None:
            raise FactorTimeframeAnalysisError(
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
        raise FactorTimeframeAnalysisError(
            "engine name must be a non-blank string",
            error_code=_ERROR_NAME_BLANK,
            details={"name": name},
        )
    return name


def _require_engine(engine: object) -> FactorTimeframeAnalysisEngine:
    """Validate that ``engine`` implements ``FactorTimeframeAnalysisEngine``."""
    if not isinstance(engine, FactorTimeframeAnalysisEngine):
        raise FactorTimeframeAnalysisError(
            "engine must implement the FactorTimeframeAnalysisEngine protocol",
            error_code=_ERROR_NOT_ENGINE,
            details={"value_type": type(engine).__name__},
        )
    return engine
