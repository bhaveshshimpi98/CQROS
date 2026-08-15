"""CQROS Reporting Engine package pipeline.

Purpose:
    Orchestrate deterministic conversion of canonical analytics ledgers into
    canonical reporting datasets through registered ``ReportingEngine``
    implementations.

Responsibilities:
    - Validate engine names and resolve engines from
      ``ReportingEngineRegistry``
    - Validate canonical analytics DataFrame structure
    - Delegate reporting metadata assembly exclusively to an injected engine
    - Validate required reporting schema columns on the engine output
    - Reject duplicate primary keys
    - Finalize outputs against ``REPORTING_SCHEMA``
    - Preserve analytics-frame immutability
    - Remain free of reporting algorithms, persistence, verification,
      exchange APIs, and CLI logic

Dependencies:
    ``polars``, ``cqros.reporting.engine``, ``cqros.reporting.exceptions``,
    ``cqros.reporting.registry``, and ``cqros.reporting.schema``.

Public API:
    ``ReportingPipeline``
"""

from __future__ import annotations

from typing import Final

import polars as pl

from cqros.reporting.engine import validate_analytics_frame
from cqros.reporting.exceptions import ReportingValidationError
from cqros.reporting.registry import ReportingEngineRegistry
from cqros.reporting.schema import (
    PRIMARY_KEY_COLUMNS,
    REPORTING_COLUMNS,
    REPORTING_SCHEMA,
    REQUIRED_COLUMNS,
)

__all__ = ["ReportingPipeline"]

_ERROR_NAME_BLANK: Final[str] = "REP_PIPE_NAME_BLANK"
_ERROR_INVALID_OUTPUT: Final[str] = "REP_PIPE_INVALID_OUTPUT"
_ERROR_OUTPUT_EMPTY: Final[str] = "REP_PIPE_OUTPUT_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "REP_PIPE_MISSING_COLUMNS"
_ERROR_MISSING_PRIMARY_KEYS: Final[str] = "REP_PIPE_MISSING_PRIMARY_KEYS"
_ERROR_DUPLICATE_KEYS: Final[str] = "REP_PIPE_DUPLICATE_KEYS"
_ERROR_SCHEMA_CAST: Final[str] = "REP_PIPE_SCHEMA_CAST"

_PRIMARY_KEY_LIST: Final[list[str]] = list(PRIMARY_KEY_COLUMNS)


class ReportingPipeline:
    """Deterministic orchestrator for canonical reporting assembly.

    The pipeline resolves a registered ``ReportingEngine``, validates a
    canonical analytics DataFrame, delegates reporting metadata generation, and
    finalizes the result to ``REPORTING_SCHEMA``. Reporting semantics remain
    exclusively in the engine. The caller-supplied analytics frame is never
    mutated.

    Args:
        registry: Registry used to resolve reporting-engine implementations.
    """

    __slots__ = ("_registry",)

    _registry: ReportingEngineRegistry

    def __init__(self, registry: ReportingEngineRegistry) -> None:
        """Initialize the pipeline with a reporting engine registry.

        Args:
            registry: Registry containing ``ReportingEngine`` implementations.
        """
        self._registry = registry

    def run(
        self,
        engine_name: str,
        analytics_frame: pl.DataFrame,
    ) -> pl.DataFrame:
        """Resolve an engine and produce a finalized reporting frame.

        ``engine_name`` is validated and resolved first. ``analytics_frame``
        is validated through ``validate_analytics_frame``. Reporting
        generation is then delegated to ``ReportingEngine.build``. The
        engine output is checked against ``REQUIRED_COLUMNS`` /
        ``REPORTING_COLUMNS``, rejected when primary keys are missing or
        duplicated, reordered to ``REPORTING_COLUMNS``, and cast to
        ``REPORTING_SCHEMA``. The original analytics frame is never
        mutated.

        Args:
            engine_name: Registry key of the reporting engine to execute.
            analytics_frame: Canonical analytics dataset.

        Returns:
            A new DataFrame containing the finalized reporting metadata.

        Raises:
            ReportingValidationError: If ``engine_name`` is invalid, the
                engine is unknown, ``analytics_frame`` is invalid, or the
                engine output fails reporting-schema finalization.
        """
        validated_name = _require_engine_name(engine_name)
        engine = self._registry.get(validated_name)
        frame = validate_analytics_frame(analytics_frame)
        created = engine.build(frame)
        return _finalize(created)


def _finalize(frame: object) -> pl.DataFrame:
    """Apply schema checks, uniqueness checks, ordering, and casting."""
    if not isinstance(frame, pl.DataFrame):
        raise ReportingValidationError(
            "engine output must be a polars DataFrame",
            error_code=_ERROR_INVALID_OUTPUT,
            details={"actual_type": type(frame).__name__},
        )
    if frame.height == 0:
        raise ReportingValidationError(
            "engine output must contain at least one row",
            error_code=_ERROR_OUTPUT_EMPTY,
            details={"rows": frame.height},
        )
    _require_primary_key_columns(frame)
    _require_reporting_schema_columns(frame)
    _require_unique_primary_keys(frame)
    ordered = frame.select(list(REPORTING_COLUMNS))
    try:
        return ordered.cast(REPORTING_SCHEMA)
    except (pl.exceptions.PolarsError, TypeError, ValueError) as exc:
        raise ReportingValidationError(
            "engine output cannot be cast to REPORTING_SCHEMA",
            error_code=_ERROR_SCHEMA_CAST,
            details={"reason": str(exc)},
        ) from exc


def _require_engine_name(name: object) -> str:
    """Validate and return a non-blank engine name."""
    if not isinstance(name, str) or name.strip() == "":
        raise ReportingValidationError(
            "engine_name must be a non-blank string",
            error_code=_ERROR_NAME_BLANK,
            details={"engine_name": name},
        )
    return name


def _require_reporting_schema_columns(frame: pl.DataFrame) -> None:
    """Raise when any required reporting-schema column is missing."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ReportingValidationError(
            "reporting schema is missing required columns",
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
        raise ReportingValidationError(
            "reporting frame is missing primary key columns",
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
        raise ReportingValidationError(
            "reporting frame contains duplicate primary keys",
            error_code=_ERROR_DUPLICATE_KEYS,
            details={
                "primary_key_columns": PRIMARY_KEY_COLUMNS,
                "row_count": frame.height,
                "unique_key_count": unique_keys,
            },
        )
