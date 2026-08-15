"""CQROS Position Engine package pipeline.

Purpose:
    Orchestrate deterministic conversion of canonical executed-trade datasets
    into canonical position datasets through registered ``PositionEngine``
    implementations.

Responsibilities:
    - Validate engine names and resolve engines from
      ``PositionEngineRegistry``
    - Validate canonical trade DataFrame structure
    - Delegate position accounting exclusively to an injected engine
    - Validate required position schema columns on the engine output
    - Reject duplicate position identifiers
    - Finalize outputs against the canonical merged position schema
    - Preserve trade-frame immutability
    - Remain free of accounting algorithms, persistence, verification,
      exchange APIs, and CLI logic

Dependencies:
    ``polars``, ``cqros.positions.engine``, ``cqros.positions.exceptions``,
    ``cqros.positions.registry``, and ``cqros.positions.schema``.

Public API:
    ``PositionPipeline``
"""

from __future__ import annotations

from typing import Final

import polars as pl

from cqros.positions.engine import validate_trade_frame
from cqros.positions.exceptions import PositionValidationError
from cqros.positions.registry import PositionEngineRegistry
from cqros.positions.schema import (
    CANONICAL_COLUMN_ORDER,
    MERGED_POSITION_SCHEMA,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
)

__all__ = ["PositionPipeline"]

_ERROR_NAME_BLANK: Final[str] = "POS_PIPE_NAME_BLANK"
_ERROR_MANAGER_BLANK: Final[str] = "POS_PIPE_MANAGER_BLANK"
_ERROR_INVALID_OUTPUT: Final[str] = "POS_PIPE_INVALID_OUTPUT"
_ERROR_OUTPUT_EMPTY: Final[str] = "POS_PIPE_OUTPUT_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "POS_PIPE_MISSING_COLUMNS"
_ERROR_DUPLICATE_KEYS: Final[str] = "POS_PIPE_DUPLICATE_KEYS"

_PRIMARY_KEY_LIST: Final[list[str]] = list(PRIMARY_KEY_COLUMNS)

_DEFAULT_ENGINE: Final[str] = "average_cost"


class PositionPipeline:
    """Deterministic orchestrator for canonical position dataset assembly.

    The pipeline resolves a registered ``PositionEngine``, validates a
    canonical executed-trade DataFrame, delegates accounting, and finalizes
    the result to the canonical merged position schema. Accounting semantics
    remain exclusively in the engine. The caller-supplied trade frame is
    never mutated.

    Args:
        engine_registry: Registry used to resolve position-engine
            implementations.
    """

    __slots__ = ("_engine_registry",)

    _engine_registry: PositionEngineRegistry

    def __init__(self, engine_registry: PositionEngineRegistry) -> None:
        """Initialize the pipeline with a position engine registry.

        Args:
            engine_registry: Registry containing ``PositionEngine``
                implementations.
        """
        self._engine_registry = engine_registry

    def run(
        self,
        trades: pl.DataFrame,
        *,
        manager: str,
        engine_name: str = _DEFAULT_ENGINE,
    ) -> pl.DataFrame:
        """Resolve an engine and produce a finalized position frame.

        ``engine_name`` is validated and resolved first. ``manager`` is
        validated as a non-blank lineage identity. ``trades`` is validated
        through ``validate_trade_frame``. Accounting is then delegated to
        ``PositionEngine.build``. The engine output is checked against
        ``REQUIRED_COLUMNS``, rejected when primary keys are duplicated,
        reordered to ``CANONICAL_COLUMN_ORDER``, and cast to
        ``MERGED_POSITION_SCHEMA``. The original ``trades`` frame is never
        mutated.

        Args:
            trades: Canonical executed-trade dataset.
            manager: Order manager identity stamped onto every position row
                and used for partition lineage.
            engine_name: Registry key of the position engine to execute.
                Defaults to ``average_cost``.

        Returns:
            A new DataFrame containing the finalized merged position matrix.

        Raises:
            PositionValidationError: If ``engine_name`` is invalid, the engine
                is unknown, ``manager`` is blank, ``trades`` is invalid, or
                the engine output fails position-schema finalization.
        """
        validated_name = _require_engine_name(engine_name)
        validated_manager = _require_manager(manager)
        engine = self._engine_registry.get(validated_name)
        frame = validate_trade_frame(trades)
        created = engine.build(frame, manager=validated_manager)
        return _finalize(created)


def _finalize(frame: object) -> pl.DataFrame:
    """Apply schema checks, uniqueness checks, ordering, and casting."""
    if not isinstance(frame, pl.DataFrame):
        raise PositionValidationError(
            "engine output must be a polars DataFrame",
            error_code=_ERROR_INVALID_OUTPUT,
            details={"actual_type": type(frame).__name__},
        )
    if frame.height == 0:
        raise PositionValidationError(
            "engine output must contain at least one row",
            error_code=_ERROR_OUTPUT_EMPTY,
            details={"rows": frame.height},
        )
    _require_position_schema_columns(frame)
    _require_unique_primary_keys(frame)
    ordered = frame.select(list(CANONICAL_COLUMN_ORDER))
    return ordered.cast(MERGED_POSITION_SCHEMA)


def _require_engine_name(name: object) -> str:
    """Validate and return a non-blank engine name."""
    if not isinstance(name, str) or name.strip() == "":
        raise PositionValidationError(
            "engine_name must be a non-blank string",
            error_code=_ERROR_NAME_BLANK,
            details={"engine_name": name},
        )
    return name


def _require_manager(manager: object) -> str:
    """Validate and return a non-blank manager identity."""
    if not isinstance(manager, str) or manager.strip() == "":
        raise PositionValidationError(
            "manager must be a non-blank string",
            error_code=_ERROR_MANAGER_BLANK,
            details={"manager": manager},
        )
    return manager


def _require_position_schema_columns(frame: pl.DataFrame) -> None:
    """Raise when any required merged-schema column is missing."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise PositionValidationError(
            "merged position schema is missing required columns",
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
        raise PositionValidationError(
            "position frame contains duplicate primary keys",
            error_code=_ERROR_DUPLICATE_KEYS,
            details={
                "primary_key_columns": PRIMARY_KEY_COLUMNS,
                "row_count": frame.height,
                "unique_key_count": unique_keys,
            },
        )
