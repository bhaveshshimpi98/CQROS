"""CQROS Walk-Forward Engine package pipeline.

Purpose:
    Orchestrate deterministic conversion of canonical Factor Selection
    datasets into canonical walk-forward datasets through registered
    ``WalkForwardEngine`` implementations.

Responsibilities:
    - Validate engine names and resolve engines from
      ``WalkForwardEngineRegistry``
    - Validate canonical Factor Selection DataFrame structure
    - Delegate walk-forward-row assembly exclusively to an injected engine
    - Validate required walk-forward schema columns on the engine output
    - Reject missing or duplicate primary keys
    - Finalize outputs against ``WALK_FORWARD_SCHEMA``
    - Preserve Factor-Selection-frame immutability
    - Remain free of walk-forward algorithms, persistence, verification,
      exchange APIs, and CLI logic

Dependencies:
    ``polars``, ``cqros.walk_forward.engine``,
    ``cqros.walk_forward.exceptions``,
    ``cqros.walk_forward.registry``, and
    ``cqros.walk_forward.schema``.

Public API:
    ``WalkForwardPipeline``
"""

from __future__ import annotations

from typing import Final

import polars as pl

from cqros.walk_forward.engine import validate_factor_selection_frame
from cqros.walk_forward.exceptions import WalkForwardError
from cqros.walk_forward.registry import WalkForwardEngineRegistry
from cqros.walk_forward.schema import (
    CANONICAL_COLUMN_ORDER,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    WALK_FORWARD_SCHEMA,
)

__all__ = ["WalkForwardPipeline"]

_ERROR_NAME_BLANK: Final[str] = "WF_PIPE_NAME_BLANK"
_ERROR_INVALID_OUTPUT: Final[str] = "WF_PIPE_INVALID_OUTPUT"
_ERROR_OUTPUT_EMPTY: Final[str] = "WF_PIPE_OUTPUT_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "WF_PIPE_MISSING_COLUMNS"
_ERROR_MISSING_PRIMARY_KEYS: Final[str] = "WF_PIPE_MISSING_PRIMARY_KEYS"
_ERROR_DUPLICATE_KEYS: Final[str] = "WF_PIPE_DUPLICATE_KEYS"
_ERROR_SCHEMA_CAST: Final[str] = "WF_PIPE_SCHEMA_CAST"

_PRIMARY_KEY_LIST: Final[list[str]] = list(PRIMARY_KEY_COLUMNS)


class WalkForwardPipeline:
    """Deterministic orchestrator for canonical walk-forward assembly.

    The pipeline resolves a registered ``WalkForwardEngine``, validates a
    canonical Factor Selection DataFrame, delegates walk-forward-row
    generation, and finalizes the result to ``WALK_FORWARD_SCHEMA``.
    Walk-forward semantics remain exclusively in the engine. The
    caller-supplied Factor Selection frame is never mutated.

    Args:
        registry: Registry used to resolve walk-forward-engine
            implementations.
    """

    __slots__ = ("_registry",)

    _registry: WalkForwardEngineRegistry

    def __init__(self, registry: WalkForwardEngineRegistry) -> None:
        """Initialize the pipeline with a walk-forward engine registry.

        Args:
            registry: Registry containing ``WalkForwardEngine``
                implementations.
        """
        self._registry = registry

    def run(
        self,
        engine_name: str,
        factor_selection_frame: pl.DataFrame,
    ) -> pl.DataFrame:
        """Resolve an engine and produce a finalized walk-forward frame.

        ``engine_name`` is validated and resolved first.
        ``factor_selection_frame`` is validated through
        ``validate_factor_selection_frame``. Walk-forward generation is then
        delegated to ``WalkForwardEngine.build``. The engine output is
        checked against ``REQUIRED_COLUMNS`` / ``CANONICAL_COLUMN_ORDER``,
        rejected when primary keys are missing or duplicated, reordered to
        ``CANONICAL_COLUMN_ORDER``, and cast to ``WALK_FORWARD_SCHEMA``.
        The original Factor Selection frame is never mutated.

        Args:
            engine_name: Registry key of the walk-forward engine to execute.
            factor_selection_frame: Canonical Factor Selection dataset.

        Returns:
            A new DataFrame containing the finalized walk-forward rows.

        Raises:
            WalkForwardError: If ``engine_name`` is invalid, the engine is
                unknown, ``factor_selection_frame`` is invalid, or the engine
                output fails walk-forward-schema finalization.
        """
        validated_name = _require_engine_name(engine_name)
        engine = self._registry.get(validated_name)
        frame = validate_factor_selection_frame(factor_selection_frame)
        created = engine.build(frame)
        return _finalize(created)


def _finalize(frame: object) -> pl.DataFrame:
    """Apply schema checks, uniqueness checks, ordering, and casting."""
    if not isinstance(frame, pl.DataFrame):
        raise WalkForwardError(
            "engine output must be a polars DataFrame",
            error_code=_ERROR_INVALID_OUTPUT,
            details={"actual_type": type(frame).__name__},
        )
    if frame.height == 0:
        raise WalkForwardError(
            "engine output must contain at least one row",
            error_code=_ERROR_OUTPUT_EMPTY,
            details={"rows": frame.height},
        )
    _require_primary_key_columns(frame)
    _require_walk_forward_schema_columns(frame)
    _require_unique_primary_keys(frame)
    ordered = frame.select(list(CANONICAL_COLUMN_ORDER))
    try:
        return ordered.cast(WALK_FORWARD_SCHEMA)
    except (pl.exceptions.PolarsError, TypeError, ValueError) as exc:
        raise WalkForwardError(
            "engine output cannot be cast to WALK_FORWARD_SCHEMA",
            error_code=_ERROR_SCHEMA_CAST,
            details={"reason": str(exc)},
        ) from exc


def _require_engine_name(name: object) -> str:
    """Validate and return a non-blank engine name."""
    if not isinstance(name, str) or name.strip() == "":
        raise WalkForwardError(
            "engine_name must be a non-blank string",
            error_code=_ERROR_NAME_BLANK,
            details={"engine_name": name},
        )
    return name


def _require_walk_forward_schema_columns(frame: pl.DataFrame) -> None:
    """Raise when any required walk-forward-schema column is missing."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise WalkForwardError(
            "walk forward schema is missing required columns",
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
        raise WalkForwardError(
            "walk forward frame is missing primary key columns",
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
        raise WalkForwardError(
            "walk forward frame contains duplicate primary keys",
            error_code=_ERROR_DUPLICATE_KEYS,
            details={
                "primary_key_columns": PRIMARY_KEY_COLUMNS,
                "row_count": frame.height,
                "unique_key_count": unique_keys,
            },
        )
