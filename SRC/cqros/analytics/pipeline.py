"""CQROS Analytics Engine package pipeline.

Purpose:
    Orchestrate deterministic conversion of canonical performance ledgers into
    canonical analytics datasets through registered ``AnalyticsEngine``
    implementations.

Responsibilities:
    - Validate engine names and resolve engines from
      ``AnalyticsEngineRegistry``
    - Validate canonical performance DataFrame structure
    - Delegate analytics computation exclusively to an injected engine
    - Validate required analytics schema columns on the engine output
    - Reject duplicate primary keys
    - Finalize outputs against ``ANALYTICS_SCHEMA``
    - Preserve performance-frame immutability
    - Remain free of analytics algorithms, persistence, verification,
      exchange APIs, and CLI logic

Dependencies:
    ``polars``, ``cqros.analytics.engine``, ``cqros.analytics.exceptions``,
    ``cqros.analytics.registry``, and ``cqros.analytics.schema``.

Public API:
    ``AnalyticsPipeline``
"""

from __future__ import annotations

from typing import Final

import polars as pl

from cqros.analytics.engine import validate_performance_frame
from cqros.analytics.exceptions import AnalyticsValidationError
from cqros.analytics.registry import AnalyticsEngineRegistry
from cqros.analytics.schema import (
    ANALYTICS_COLUMNS,
    ANALYTICS_SCHEMA,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
)

__all__ = ["AnalyticsPipeline"]

_ERROR_NAME_BLANK: Final[str] = "ANA_PIPE_NAME_BLANK"
_ERROR_INVALID_OUTPUT: Final[str] = "ANA_PIPE_INVALID_OUTPUT"
_ERROR_OUTPUT_EMPTY: Final[str] = "ANA_PIPE_OUTPUT_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "ANA_PIPE_MISSING_COLUMNS"
_ERROR_MISSING_PRIMARY_KEYS: Final[str] = "ANA_PIPE_MISSING_PRIMARY_KEYS"
_ERROR_DUPLICATE_KEYS: Final[str] = "ANA_PIPE_DUPLICATE_KEYS"
_ERROR_SCHEMA_CAST: Final[str] = "ANA_PIPE_SCHEMA_CAST"

_PRIMARY_KEY_LIST: Final[list[str]] = list(PRIMARY_KEY_COLUMNS)


class AnalyticsPipeline:
    """Deterministic orchestrator for canonical analytics assembly.

    The pipeline resolves a registered ``AnalyticsEngine``, validates a
    canonical performance DataFrame, delegates analytics computation, and
    finalizes the result to ``ANALYTICS_SCHEMA``. Analytics semantics remain
    exclusively in the engine. The caller-supplied performance frame is never
    mutated.

    Args:
        registry: Registry used to resolve analytics-engine implementations.
    """

    __slots__ = ("_registry",)

    _registry: AnalyticsEngineRegistry

    def __init__(self, registry: AnalyticsEngineRegistry) -> None:
        """Initialize the pipeline with an analytics engine registry.

        Args:
            registry: Registry containing ``AnalyticsEngine`` implementations.
        """
        self._registry = registry

    def run(
        self,
        engine_name: str,
        performance_frame: pl.DataFrame,
    ) -> pl.DataFrame:
        """Resolve an engine and produce a finalized analytics frame.

        ``engine_name`` is validated and resolved first. ``performance_frame``
        is validated through ``validate_performance_frame``. Analytics
        computation is then delegated to ``AnalyticsEngine.build``. The
        engine output is checked against ``REQUIRED_COLUMNS`` /
        ``ANALYTICS_COLUMNS``, rejected when primary keys are missing or
        duplicated, reordered to ``ANALYTICS_COLUMNS``, and cast to
        ``ANALYTICS_SCHEMA``. The original performance frame is never
        mutated.

        Args:
            engine_name: Registry key of the analytics engine to execute.
            performance_frame: Canonical performance dataset.

        Returns:
            A new DataFrame containing the finalized analytics metrics.

        Raises:
            AnalyticsValidationError: If ``engine_name`` is invalid, the
                engine is unknown, ``performance_frame`` is invalid, or the
                engine output fails analytics-schema finalization.
        """
        validated_name = _require_engine_name(engine_name)
        engine = self._registry.get(validated_name)
        frame = validate_performance_frame(performance_frame)
        created = engine.build(frame)
        return _finalize(created)


def _finalize(frame: object) -> pl.DataFrame:
    """Apply schema checks, uniqueness checks, ordering, and casting."""
    if not isinstance(frame, pl.DataFrame):
        raise AnalyticsValidationError(
            "engine output must be a polars DataFrame",
            error_code=_ERROR_INVALID_OUTPUT,
            details={"actual_type": type(frame).__name__},
        )
    if frame.height == 0:
        raise AnalyticsValidationError(
            "engine output must contain at least one row",
            error_code=_ERROR_OUTPUT_EMPTY,
            details={"rows": frame.height},
        )
    _require_primary_key_columns(frame)
    _require_analytics_schema_columns(frame)
    _require_unique_primary_keys(frame)
    ordered = frame.select(list(ANALYTICS_COLUMNS))
    try:
        return ordered.cast(ANALYTICS_SCHEMA)
    except (pl.exceptions.PolarsError, TypeError, ValueError) as exc:
        raise AnalyticsValidationError(
            "engine output cannot be cast to ANALYTICS_SCHEMA",
            error_code=_ERROR_SCHEMA_CAST,
            details={"reason": str(exc)},
        ) from exc


def _require_engine_name(name: object) -> str:
    """Validate and return a non-blank engine name."""
    if not isinstance(name, str) or name.strip() == "":
        raise AnalyticsValidationError(
            "engine_name must be a non-blank string",
            error_code=_ERROR_NAME_BLANK,
            details={"engine_name": name},
        )
    return name


def _require_analytics_schema_columns(frame: pl.DataFrame) -> None:
    """Raise when any required analytics-schema column is missing."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise AnalyticsValidationError(
            "analytics schema is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "missing_columns": tuple(missing),
                "required_columns": REQUIRED_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )


def _require_primary_key_columns(frame: pl.DataFrame) -> None:
    """Raise when any primary-key column is missing from ``frame``."""
    missing = [column for column in PRIMARY_KEY_COLUMNS if column not in frame.columns]
    if missing:
        raise AnalyticsValidationError(
            "analytics frame is missing primary key columns",
            error_code=_ERROR_MISSING_PRIMARY_KEYS,
            details={
                "missing_columns": tuple(missing),
                "primary_key_columns": PRIMARY_KEY_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )


def _require_unique_primary_keys(frame: pl.DataFrame) -> None:
    """Raise when primary-key combinations are duplicated in ``frame``."""
    unique_keys = frame.select(_PRIMARY_KEY_LIST).n_unique()
    if unique_keys != frame.height:
        raise AnalyticsValidationError(
            "analytics frame contains duplicate primary keys",
            error_code=_ERROR_DUPLICATE_KEYS,
            details={
                "primary_key_columns": PRIMARY_KEY_COLUMNS,
                "row_count": frame.height,
                "unique_key_count": unique_keys,
            },
        )
