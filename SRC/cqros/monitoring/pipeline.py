"""CQROS Monitoring Engine package pipeline.

Purpose:
    Orchestrate deterministic conversion of canonical reporting ledgers into
    canonical monitoring datasets through registered ``MonitoringEngine``
    implementations.

Responsibilities:
    - Validate engine names and resolve engines from
      ``MonitoringEngineRegistry``
    - Validate canonical reporting DataFrame structure
    - Delegate monitoring-row assembly exclusively to an injected engine
    - Validate required monitoring schema columns on the engine output
    - Reject duplicate primary keys
    - Finalize outputs against ``MONITORING_SCHEMA``
    - Preserve reporting-frame immutability
    - Remain free of monitoring algorithms, persistence, verification,
      exchange APIs, and CLI logic

Dependencies:
    ``polars``, ``cqros.monitoring.engine``, ``cqros.monitoring.exceptions``,
    ``cqros.monitoring.registry``, and ``cqros.monitoring.schema``.

Public API:
    ``MonitoringPipeline``
"""

from __future__ import annotations

from typing import Final

import polars as pl

from cqros.monitoring.engine import validate_reporting_frame
from cqros.monitoring.exceptions import MonitoringValidationError
from cqros.monitoring.registry import MonitoringEngineRegistry
from cqros.monitoring.schema import (
    MONITORING_COLUMNS,
    MONITORING_SCHEMA,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
)

__all__ = ["MonitoringPipeline"]

_ERROR_NAME_BLANK: Final[str] = "MON_PIPE_NAME_BLANK"
_ERROR_INVALID_OUTPUT: Final[str] = "MON_PIPE_INVALID_OUTPUT"
_ERROR_OUTPUT_EMPTY: Final[str] = "MON_PIPE_OUTPUT_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "MON_PIPE_MISSING_COLUMNS"
_ERROR_MISSING_PRIMARY_KEYS: Final[str] = "MON_PIPE_MISSING_PRIMARY_KEYS"
_ERROR_DUPLICATE_KEYS: Final[str] = "MON_PIPE_DUPLICATE_KEYS"
_ERROR_SCHEMA_CAST: Final[str] = "MON_PIPE_SCHEMA_CAST"

_PRIMARY_KEY_LIST: Final[list[str]] = list(PRIMARY_KEY_COLUMNS)


class MonitoringPipeline:
    """Deterministic orchestrator for canonical monitoring assembly.

    The pipeline resolves a registered ``MonitoringEngine``, validates a
    canonical reporting DataFrame, delegates monitoring-row generation, and
    finalizes the result to ``MONITORING_SCHEMA``. Monitoring semantics remain
    exclusively in the engine. The caller-supplied reporting frame is never
    mutated.

    Args:
        registry: Registry used to resolve monitoring-engine implementations.
    """

    __slots__ = ("_registry",)

    _registry: MonitoringEngineRegistry

    def __init__(self, registry: MonitoringEngineRegistry) -> None:
        """Initialize the pipeline with a monitoring engine registry.

        Args:
            registry: Registry containing ``MonitoringEngine`` implementations.
        """
        self._registry = registry

    def run(
        self,
        engine_name: str,
        reporting_frame: pl.DataFrame,
    ) -> pl.DataFrame:
        """Resolve an engine and produce a finalized monitoring frame.

        ``engine_name`` is validated and resolved first. ``reporting_frame``
        is validated through ``validate_reporting_frame``. Monitoring
        generation is then delegated to ``MonitoringEngine.build``. The
        engine output is checked against ``REQUIRED_COLUMNS`` /
        ``MONITORING_COLUMNS``, rejected when primary keys are missing or
        duplicated, reordered to ``MONITORING_COLUMNS``, and cast to
        ``MONITORING_SCHEMA``. The original reporting frame is never
        mutated.

        Args:
            engine_name: Registry key of the monitoring engine to execute.
            reporting_frame: Canonical reporting dataset.

        Returns:
            A new DataFrame containing the finalized monitoring rows.

        Raises:
            MonitoringValidationError: If ``engine_name`` is invalid, the
                engine is unknown, ``reporting_frame`` is invalid, or the
                engine output fails monitoring-schema finalization.
        """
        validated_name = _require_engine_name(engine_name)
        engine = self._registry.get(validated_name)
        frame = validate_reporting_frame(reporting_frame)
        created = engine.build(frame)
        return _finalize(created)


def _finalize(frame: object) -> pl.DataFrame:
    """Apply schema checks, uniqueness checks, ordering, and casting."""
    if not isinstance(frame, pl.DataFrame):
        raise MonitoringValidationError(
            "engine output must be a polars DataFrame",
            error_code=_ERROR_INVALID_OUTPUT,
            details={"actual_type": type(frame).__name__},
        )
    if frame.height == 0:
        raise MonitoringValidationError(
            "engine output must contain at least one row",
            error_code=_ERROR_OUTPUT_EMPTY,
            details={"rows": frame.height},
        )
    _require_primary_key_columns(frame)
    _require_monitoring_schema_columns(frame)
    _require_unique_primary_keys(frame)
    ordered = frame.select(list(MONITORING_COLUMNS))
    try:
        return ordered.cast(MONITORING_SCHEMA)
    except (pl.exceptions.PolarsError, TypeError, ValueError) as exc:
        raise MonitoringValidationError(
            "engine output cannot be cast to MONITORING_SCHEMA",
            error_code=_ERROR_SCHEMA_CAST,
            details={"reason": str(exc)},
        ) from exc


def _require_engine_name(name: object) -> str:
    """Validate and return a non-blank engine name."""
    if not isinstance(name, str) or name.strip() == "":
        raise MonitoringValidationError(
            "engine_name must be a non-blank string",
            error_code=_ERROR_NAME_BLANK,
            details={"engine_name": name},
        )
    return name


def _require_monitoring_schema_columns(frame: pl.DataFrame) -> None:
    """Raise when any required monitoring-schema column is missing."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise MonitoringValidationError(
            "monitoring schema is missing required columns",
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
        raise MonitoringValidationError(
            "monitoring frame is missing primary key columns",
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
        raise MonitoringValidationError(
            "monitoring frame contains duplicate primary keys",
            error_code=_ERROR_DUPLICATE_KEYS,
            details={
                "primary_key_columns": PRIMARY_KEY_COLUMNS,
                "row_count": frame.height,
                "unique_key_count": unique_keys,
            },
        )
