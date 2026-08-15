"""CQROS Performance Engine package pipeline.

Purpose:
    Orchestrate deterministic conversion of canonical backtesting ledgers into
    canonical performance-metrics datasets through registered
    ``PerformanceEngine`` implementations.

Responsibilities:
    - Validate engine names and resolve engines from
      ``PerformanceEngineRegistry``
    - Validate canonical backtesting DataFrame structure
    - Delegate metric computation exclusively to an injected engine
    - Validate required performance schema columns on the engine output
    - Reject duplicate primary keys
    - Finalize outputs against ``PERFORMANCE_SCHEMA``
    - Preserve backtesting-frame immutability
    - Remain free of performance algorithms, persistence, verification,
      exchange APIs, and CLI logic

Dependencies:
    ``polars``, ``cqros.performance.engine``, ``cqros.performance.exceptions``,
    ``cqros.performance.registry``, and ``cqros.performance.schema``.

Public API:
    ``PerformancePipeline``
"""

from __future__ import annotations

from typing import Final

import polars as pl

from cqros.performance.engine import validate_backtesting_frame
from cqros.performance.exceptions import PerformanceValidationError
from cqros.performance.registry import PerformanceEngineRegistry
from cqros.performance.schema import (
    PERFORMANCE_COLUMNS,
    PERFORMANCE_SCHEMA,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
)

__all__ = ["PerformancePipeline"]

_ERROR_NAME_BLANK: Final[str] = "PERF_PIPE_NAME_BLANK"
_ERROR_INVALID_OUTPUT: Final[str] = "PERF_PIPE_INVALID_OUTPUT"
_ERROR_OUTPUT_EMPTY: Final[str] = "PERF_PIPE_OUTPUT_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "PERF_PIPE_MISSING_COLUMNS"
_ERROR_DUPLICATE_KEYS: Final[str] = "PERF_PIPE_DUPLICATE_KEYS"

_PRIMARY_KEY_LIST: Final[list[str]] = list(PRIMARY_KEY_COLUMNS)


class PerformancePipeline:
    """Deterministic orchestrator for canonical performance-metrics assembly.

    The pipeline resolves a registered ``PerformanceEngine``, validates a
    canonical backtesting DataFrame, delegates metric computation, and
    finalizes the result to ``PERFORMANCE_SCHEMA``. Performance semantics
    remain exclusively in the engine. The caller-supplied backtesting frame
    is never mutated.

    Args:
        registry: Registry used to resolve performance-engine
            implementations.
    """

    __slots__ = ("_registry",)

    _registry: PerformanceEngineRegistry

    def __init__(self, registry: PerformanceEngineRegistry) -> None:
        """Initialize the pipeline with a performance engine registry.

        Args:
            registry: Registry containing ``PerformanceEngine``
                implementations.
        """
        self._registry = registry

    def run(
        self,
        engine_name: str,
        backtesting_frame: pl.DataFrame,
    ) -> pl.DataFrame:
        """Resolve an engine and produce a finalized performance frame.

        ``engine_name`` is validated and resolved first. ``backtesting_frame``
        is validated through ``validate_backtesting_frame``. Metric
        computation is then delegated to ``PerformanceEngine.build``. The
        engine output is checked against ``REQUIRED_COLUMNS`` /
        ``PERFORMANCE_COLUMNS``, rejected when primary keys are duplicated,
        reordered to ``PERFORMANCE_COLUMNS``, and cast to
        ``PERFORMANCE_SCHEMA``. The original backtesting frame is never
        mutated.

        Args:
            engine_name: Registry key of the performance engine to execute.
            backtesting_frame: Canonical backtesting dataset.

        Returns:
            A new DataFrame containing the finalized performance metrics.

        Raises:
            PerformanceValidationError: If ``engine_name`` is invalid, the
                engine is unknown, ``backtesting_frame`` is invalid, or the
                engine output fails performance-schema finalization.
        """
        validated_name = _require_engine_name(engine_name)
        engine = self._registry.get(validated_name)
        frame = validate_backtesting_frame(backtesting_frame)
        created = engine.build(frame)
        return _finalize(created)


def _finalize(frame: object) -> pl.DataFrame:
    """Apply schema checks, uniqueness checks, ordering, and casting."""
    if not isinstance(frame, pl.DataFrame):
        raise PerformanceValidationError(
            "engine output must be a polars DataFrame",
            error_code=_ERROR_INVALID_OUTPUT,
            details={"actual_type": type(frame).__name__},
        )
    if frame.height == 0:
        raise PerformanceValidationError(
            "engine output must contain at least one row",
            error_code=_ERROR_OUTPUT_EMPTY,
            details={"rows": frame.height},
        )
    _require_performance_schema_columns(frame)
    _require_unique_primary_keys(frame)
    ordered = frame.select(list(PERFORMANCE_COLUMNS))
    return ordered.cast(PERFORMANCE_SCHEMA)


def _require_engine_name(name: object) -> str:
    """Validate and return a non-blank engine name."""
    if not isinstance(name, str) or name.strip() == "":
        raise PerformanceValidationError(
            "engine_name must be a non-blank string",
            error_code=_ERROR_NAME_BLANK,
            details={"engine_name": name},
        )
    return name


def _require_performance_schema_columns(frame: pl.DataFrame) -> None:
    """Raise when any required performance-schema column is missing."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise PerformanceValidationError(
            "performance schema is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "missing_columns": tuple(missing),
                "required_columns": REQUIRED_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )


def _require_unique_primary_keys(frame: pl.DataFrame) -> None:
    """Raise when primary-key combinations are duplicated in ``frame``."""
    unique_keys = frame.select(_PRIMARY_KEY_LIST).n_unique()
    if unique_keys != frame.height:
        raise PerformanceValidationError(
            "performance frame contains duplicate primary keys",
            error_code=_ERROR_DUPLICATE_KEYS,
            details={
                "primary_key_columns": PRIMARY_KEY_COLUMNS,
                "row_count": frame.height,
                "unique_key_count": unique_keys,
            },
        )
