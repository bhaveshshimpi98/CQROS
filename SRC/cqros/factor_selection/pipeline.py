"""CQROS Factor Selection Engine package pipeline.

Purpose:
    Orchestrate deterministic conversion of canonical Factor Validation
    datasets into canonical factor-selection datasets through registered
    ``FactorSelectionEngine`` implementations.

Responsibilities:
    - Validate engine names and resolve engines from
      ``FactorSelectionEngineRegistry``
    - Validate canonical Factor Validation DataFrame structure
    - Delegate selection-row assembly exclusively to an injected engine
    - Validate required factor-selection schema columns on the engine output
    - Reject missing or duplicate primary keys
    - Finalize outputs against ``FACTOR_SELECTION_SCHEMA``
    - Preserve Factor-Validation-frame immutability
    - Remain free of selection algorithms, persistence, verification,
      exchange APIs, and CLI logic

Dependencies:
    ``polars``, ``cqros.factor_selection.engine``,
    ``cqros.factor_selection.exceptions``,
    ``cqros.factor_selection.registry``, and
    ``cqros.factor_selection.schema``.

Public API:
    ``FactorSelectionPipeline``
"""

from __future__ import annotations

from typing import Final

import polars as pl

from cqros.factor_selection.engine import validate_factor_validation_frame
from cqros.factor_selection.exceptions import FactorSelectionError
from cqros.factor_selection.registry import FactorSelectionEngineRegistry
from cqros.factor_selection.schema import (
    ELIGIBILITY_COLUMN_DTYPES,
    ELIGIBILITY_COLUMNS,
    FACTOR_SELECTION_COLUMNS,
    FACTOR_SELECTION_SCHEMA,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
)

__all__ = ["FactorSelectionPipeline"]

_ERROR_NAME_BLANK: Final[str] = "FSEL_PIPE_NAME_BLANK"
_ERROR_INVALID_OUTPUT: Final[str] = "FSEL_PIPE_INVALID_OUTPUT"
_ERROR_OUTPUT_EMPTY: Final[str] = "FSEL_PIPE_OUTPUT_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "FSEL_PIPE_MISSING_COLUMNS"
_ERROR_MISSING_PRIMARY_KEYS: Final[str] = "FSEL_PIPE_MISSING_PRIMARY_KEYS"
_ERROR_DUPLICATE_KEYS: Final[str] = "FSEL_PIPE_DUPLICATE_KEYS"
_ERROR_SCHEMA_CAST: Final[str] = "FSEL_PIPE_SCHEMA_CAST"

_PRIMARY_KEY_LIST: Final[list[str]] = list(PRIMARY_KEY_COLUMNS)


class FactorSelectionPipeline:
    """Deterministic orchestrator for canonical factor-selection assembly.

    The pipeline resolves a registered ``FactorSelectionEngine``, validates a
    canonical Factor Validation DataFrame, delegates selection-row generation,
    and finalizes the result to ``FACTOR_SELECTION_SCHEMA``. Selection
    semantics remain exclusively in the engine. The caller-supplied Factor
    Validation frame is never mutated.

    Args:
        registry: Registry used to resolve factor-selection-engine
            implementations.
    """

    __slots__ = ("_registry",)

    _registry: FactorSelectionEngineRegistry

    def __init__(self, registry: FactorSelectionEngineRegistry) -> None:
        """Initialize the pipeline with a factor selection engine registry.

        Args:
            registry: Registry containing ``FactorSelectionEngine``
                implementations.
        """
        self._registry = registry

    def run(
        self,
        engine_name: str,
        factor_validation_frame: pl.DataFrame,
    ) -> pl.DataFrame:
        """Resolve an engine and produce a finalized factor-selection frame.

        ``engine_name`` is validated and resolved first.
        ``factor_validation_frame`` is validated through
        ``validate_factor_validation_frame``. Selection generation is then
        delegated to ``FactorSelectionEngine.build``. The engine output is
        checked against ``REQUIRED_COLUMNS`` / ``FACTOR_SELECTION_COLUMNS``,
        rejected when primary keys are missing or duplicated, reordered to
        ``FACTOR_SELECTION_COLUMNS``, and cast to ``FACTOR_SELECTION_SCHEMA``.
        The original Factor Validation frame is never mutated.

        Args:
            engine_name: Registry key of the factor selection engine to
                execute.
            factor_validation_frame: Canonical Factor Validation dataset.

        Returns:
            A new DataFrame containing the finalized factor selection rows.

        Raises:
            FactorSelectionError: If ``engine_name`` is invalid, the engine
                is unknown, ``factor_validation_frame`` is invalid, or the
                engine output fails factor-selection-schema finalization.
        """
        validated_name = _require_engine_name(engine_name)
        engine = self._registry.get(validated_name)
        frame = validate_factor_validation_frame(factor_validation_frame)
        created = engine.build(frame)
        return _finalize(created)


def _finalize(frame: object) -> pl.DataFrame:
    """Apply schema checks, uniqueness checks, ordering, and casting.

    Eligibility metadata columns are preserved when present; they extend the
    canonical schema without replacing it.
    """
    if not isinstance(frame, pl.DataFrame):
        raise FactorSelectionError(
            "engine output must be a polars DataFrame",
            error_code=_ERROR_INVALID_OUTPUT,
            details={"actual_type": type(frame).__name__},
        )
    if frame.height == 0:
        raise FactorSelectionError(
            "engine output must contain at least one row",
            error_code=_ERROR_OUTPUT_EMPTY,
            details={"rows": frame.height},
        )
    _require_primary_key_columns(frame)
    _require_factor_selection_schema_columns(frame)
    _require_unique_primary_keys(frame)
    # Determine which columns to keep: canonical + any eligibility extensions.
    extra_cols = [c for c in ELIGIBILITY_COLUMNS if c in frame.columns]
    ordered = frame.select(list(FACTOR_SELECTION_COLUMNS) + extra_cols)
    combined_schema = pl.Schema(
        [(col, FACTOR_SELECTION_SCHEMA[col]) for col in FACTOR_SELECTION_COLUMNS]
        + [(col, ELIGIBILITY_COLUMN_DTYPES[col]) for col in extra_cols]
    )
    try:
        return ordered.cast(combined_schema)
    except (pl.exceptions.PolarsError, TypeError, ValueError) as exc:
        raise FactorSelectionError(
            "engine output cannot be cast to FACTOR_SELECTION_SCHEMA",
            error_code=_ERROR_SCHEMA_CAST,
            details={"reason": str(exc)},
        ) from exc


def _require_engine_name(name: object) -> str:
    """Validate and return a non-blank engine name."""
    if not isinstance(name, str) or name.strip() == "":
        raise FactorSelectionError(
            "engine_name must be a non-blank string",
            error_code=_ERROR_NAME_BLANK,
            details={"engine_name": name},
        )
    return name


def _require_factor_selection_schema_columns(frame: pl.DataFrame) -> None:
    """Raise when any required factor-selection-schema column is missing."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise FactorSelectionError(
            "factor selection schema is missing required columns",
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
        raise FactorSelectionError(
            "factor selection frame is missing primary key columns",
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
        raise FactorSelectionError(
            "factor selection frame contains duplicate primary keys",
            error_code=_ERROR_DUPLICATE_KEYS,
            details={
                "primary_key_columns": PRIMARY_KEY_COLUMNS,
                "row_count": frame.height,
                "unique_key_count": unique_keys,
            },
        )
