"""CQROS Factor Timeframe Analysis Engine package pipeline.

Purpose:
    Orchestrate deterministic conversion of canonical Factor Selection
    datasets into canonical factor-timeframe-analysis datasets through
    registered ``FactorTimeframeAnalysisEngine`` implementations.

Responsibilities:
    - Validate engine names and resolve engines from
      ``FactorTimeframeAnalysisEngineRegistry``
    - Validate canonical Factor Selection DataFrame structure
    - Delegate analysis-row assembly exclusively to an injected engine
    - Validate required timeframe-analysis schema columns on the engine
      output
    - Reject missing or duplicate primary keys
    - Finalize outputs against ``TIMEFRAME_ANALYSIS_SCHEMA``
    - Preserve Factor-Selection-frame immutability
    - Remain free of analysis algorithms, persistence, verification,
      exchange APIs, and CLI logic

Dependencies:
    ``polars``, ``cqros.factor_timeframe_analysis.engine``,
    ``cqros.factor_timeframe_analysis.exceptions``,
    ``cqros.factor_timeframe_analysis.registry``, and
    ``cqros.factor_timeframe_analysis.schema``.

Public API:
    ``FactorTimeframeAnalysisPipeline``
"""

from __future__ import annotations

from typing import Final

import polars as pl

from cqros.factor_timeframe_analysis.engine import validate_factor_selection_frame
from cqros.factor_timeframe_analysis.exceptions import FactorTimeframeAnalysisError
from cqros.factor_timeframe_analysis.registry import FactorTimeframeAnalysisEngineRegistry
from cqros.factor_timeframe_analysis.schema import (
    FACTOR_TIMEFRAME_ANALYSIS_COLUMNS,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    TIMEFRAME_ANALYSIS_SCHEMA,
)

__all__ = ["FactorTimeframeAnalysisPipeline"]

_ERROR_NAME_BLANK: Final[str] = "FTA_PIPE_NAME_BLANK"
_ERROR_INVALID_OUTPUT: Final[str] = "FTA_PIPE_INVALID_OUTPUT"
_ERROR_OUTPUT_EMPTY: Final[str] = "FTA_PIPE_OUTPUT_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "FTA_PIPE_MISSING_COLUMNS"
_ERROR_MISSING_PRIMARY_KEYS: Final[str] = "FTA_PIPE_MISSING_PRIMARY_KEYS"
_ERROR_DUPLICATE_KEYS: Final[str] = "FTA_PIPE_DUPLICATE_KEYS"
_ERROR_SCHEMA_CAST: Final[str] = "FTA_PIPE_SCHEMA_CAST"

_PRIMARY_KEY_LIST: Final[list[str]] = list(PRIMARY_KEY_COLUMNS)


class FactorTimeframeAnalysisPipeline:
    """Deterministic orchestrator for canonical timeframe-analysis assembly.

    The pipeline resolves a registered ``FactorTimeframeAnalysisEngine``,
    validates a canonical Factor Selection DataFrame, delegates
    analysis-row generation, and finalizes the result to
    ``TIMEFRAME_ANALYSIS_SCHEMA``. Analysis semantics remain exclusively in
    the engine. The caller-supplied Factor Selection frame is never mutated.

    Args:
        registry: Registry used to resolve timeframe-analysis-engine
            implementations.
    """

    __slots__ = ("_registry",)

    _registry: FactorTimeframeAnalysisEngineRegistry

    def __init__(self, registry: FactorTimeframeAnalysisEngineRegistry) -> None:
        """Initialize the pipeline with a timeframe analysis engine registry.

        Args:
            registry: Registry containing ``FactorTimeframeAnalysisEngine``
                implementations.
        """
        self._registry = registry

    def run(
        self,
        engine_name: str,
        factor_selection_frame: pl.DataFrame,
    ) -> pl.DataFrame:
        """Resolve an engine and produce a finalized timeframe-analysis frame.

        ``engine_name`` is validated and resolved first.
        ``factor_selection_frame`` is validated through
        ``validate_factor_selection_frame``. Analysis generation is then
        delegated to ``FactorTimeframeAnalysisEngine.build``. The engine
        output is checked against ``REQUIRED_COLUMNS`` /
        ``FACTOR_TIMEFRAME_ANALYSIS_COLUMNS``, rejected when primary keys
        are missing or duplicated, reordered to
        ``FACTOR_TIMEFRAME_ANALYSIS_COLUMNS``, and cast to
        ``TIMEFRAME_ANALYSIS_SCHEMA``. The original Factor Selection frame
        is never mutated.

        Args:
            engine_name: Registry key of the timeframe analysis engine to
                execute.
            factor_selection_frame: Canonical Factor Selection dataset.

        Returns:
            A new DataFrame containing the finalized timeframe analysis rows.

        Raises:
            FactorTimeframeAnalysisError: If ``engine_name`` is invalid, the
                engine is unknown, ``factor_selection_frame`` is invalid, or
                the engine output fails timeframe-analysis-schema
                finalization.
        """
        validated_name = _require_engine_name(engine_name)
        engine = self._registry.get(validated_name)
        frame = validate_factor_selection_frame(factor_selection_frame)
        created = engine.build(frame)
        return _finalize(created)


def _finalize(frame: object) -> pl.DataFrame:
    """Apply schema checks, uniqueness checks, ordering, and casting."""
    if not isinstance(frame, pl.DataFrame):
        raise FactorTimeframeAnalysisError(
            "engine output must be a polars DataFrame",
            error_code=_ERROR_INVALID_OUTPUT,
            details={"actual_type": type(frame).__name__},
        )
    if frame.height == 0:
        raise FactorTimeframeAnalysisError(
            "engine output must contain at least one row",
            error_code=_ERROR_OUTPUT_EMPTY,
            details={"rows": frame.height},
        )
    _require_primary_key_columns(frame)
    _require_timeframe_analysis_schema_columns(frame)
    _require_unique_primary_keys(frame)
    ordered = frame.select(list(FACTOR_TIMEFRAME_ANALYSIS_COLUMNS))
    try:
        return ordered.cast(TIMEFRAME_ANALYSIS_SCHEMA)
    except (pl.exceptions.PolarsError, TypeError, ValueError) as exc:
        raise FactorTimeframeAnalysisError(
            "engine output cannot be cast to TIMEFRAME_ANALYSIS_SCHEMA",
            error_code=_ERROR_SCHEMA_CAST,
            details={"reason": str(exc)},
        ) from exc


def _require_engine_name(name: object) -> str:
    """Validate and return a non-blank engine name."""
    if not isinstance(name, str) or name.strip() == "":
        raise FactorTimeframeAnalysisError(
            "engine_name must be a non-blank string",
            error_code=_ERROR_NAME_BLANK,
            details={"engine_name": name},
        )
    return name


def _require_timeframe_analysis_schema_columns(frame: pl.DataFrame) -> None:
    """Raise when any required timeframe-analysis-schema column is missing."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise FactorTimeframeAnalysisError(
            "factor timeframe analysis schema is missing required columns",
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
        raise FactorTimeframeAnalysisError(
            "factor timeframe analysis frame is missing primary key columns",
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
        raise FactorTimeframeAnalysisError(
            "factor timeframe analysis frame contains duplicate primary keys",
            error_code=_ERROR_DUPLICATE_KEYS,
            details={
                "primary_key_columns": PRIMARY_KEY_COLUMNS,
                "row_count": frame.height,
                "unique_key_count": unique_keys,
            },
        )
