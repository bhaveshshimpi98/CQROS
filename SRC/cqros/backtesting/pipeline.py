"""CQROS Backtesting Engine package pipeline.

Purpose:
    Orchestrate deterministic reconstruction of canonical historical
    performance ledgers through registered ``BacktestingEngine``
    implementations.

Responsibilities:
    - Validate engine names and resolve engines from ``BacktestingRegistry``
    - Validate canonical accounting, position, and exit-engine DataFrames
    - Delegate performance reconstruction exclusively to an injected engine
    - Validate required backtesting schema columns on the engine output
    - Reject duplicate primary keys
    - Finalize outputs against the canonical merged backtesting schema
    - Preserve input-frame immutability
    - Remain free of performance algorithms, persistence, verification,
      exchange APIs, and CLI logic

Dependencies:
    ``polars``, ``cqros.backtesting.engine``, ``cqros.backtesting.exceptions``,
    ``cqros.backtesting.registry``, and ``cqros.backtesting.schema``.

Public API:
    ``BacktestingPipeline``
"""

from __future__ import annotations

from typing import Final

import polars as pl

from cqros.backtesting.engine import (
    validate_accounting_frame,
    validate_exit_engine_frame,
    validate_position_frame,
)
from cqros.backtesting.exceptions import BacktestingValidationError
from cqros.backtesting.registry import BacktestingRegistry
from cqros.backtesting.schema import (
    CANONICAL_COLUMN_ORDER,
    MERGED_BACKTESTING_SCHEMA,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
)

__all__ = ["BacktestingPipeline"]

_ERROR_NAME_BLANK: Final[str] = "BT_PIPE_NAME_BLANK"
_ERROR_MANAGER_BLANK: Final[str] = "BT_PIPE_MANAGER_BLANK"
_ERROR_INVALID_OUTPUT: Final[str] = "BT_PIPE_INVALID_OUTPUT"
_ERROR_OUTPUT_EMPTY: Final[str] = "BT_PIPE_OUTPUT_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "BT_PIPE_MISSING_COLUMNS"
_ERROR_DUPLICATE_KEYS: Final[str] = "BT_PIPE_DUPLICATE_KEYS"

_PRIMARY_KEY_LIST: Final[list[str]] = list(PRIMARY_KEY_COLUMNS)

_DEFAULT_ENGINE: Final[str] = "simple"


class BacktestingPipeline:
    """Deterministic orchestrator for canonical performance-ledger assembly.

    The pipeline resolves a registered ``BacktestingEngine``, validates
    canonical input DataFrames, delegates reconstruction, and finalizes the
    result to the canonical merged backtesting schema. Performance semantics
    remain exclusively in the engine. The caller-supplied frames are never
    mutated.

    Args:
        registry: Registry used to resolve backtesting-engine implementations.
    """

    __slots__ = ("_registry",)

    _registry: BacktestingRegistry

    def __init__(self, registry: BacktestingRegistry) -> None:
        """Initialize the pipeline with a backtesting engine registry.

        Args:
            registry: Registry containing ``BacktestingEngine``
                implementations.
        """
        self._registry = registry

    def run(
        self,
        accounting: pl.DataFrame,
        positions: pl.DataFrame,
        exit_engine: pl.DataFrame,
        *,
        manager: str,
        engine_name: str = _DEFAULT_ENGINE,
    ) -> pl.DataFrame:
        """Resolve an engine and produce a finalized performance ledger.

        ``engine_name`` is validated and resolved first. ``manager`` is
        validated as a non-blank lineage identity. Input frames are validated
        through their respective validators. Reconstruction is then delegated
        to ``BacktestingEngine.build``. The engine output is checked against
        ``REQUIRED_COLUMNS``, rejected when primary keys are duplicated,
        reordered to ``CANONICAL_COLUMN_ORDER``, and cast to
        ``MERGED_BACKTESTING_SCHEMA``. The original input frames are never
        mutated.

        Args:
            accounting: Canonical accounting dataset.
            positions: Canonical position dataset.
            exit_engine: Canonical exit-engine dataset.
            manager: Order manager identity stamped onto every ledger row
                and used for partition lineage.
            engine_name: Registry key of the backtesting engine to execute.
                Defaults to ``simple``.

        Returns:
            A new DataFrame containing the finalized merged performance
            ledger.

        Raises:
            BacktestingValidationError: If ``engine_name`` is invalid, the
                engine is unknown, ``manager`` is blank, any input is
                invalid, or the engine output fails schema finalization.
        """
        validated_name = _require_engine_name(engine_name)
        validated_manager = _require_manager(manager)
        engine = self._registry.get(validated_name)
        accounting_frame = validate_accounting_frame(accounting)
        position_frame = validate_position_frame(positions)
        exit_frame = validate_exit_engine_frame(exit_engine)
        created = engine.build(
            accounting_frame,
            position_frame,
            exit_frame,
            manager=validated_manager,
        )
        return _finalize(created)


def _finalize(frame: object) -> pl.DataFrame:
    """Apply schema checks, uniqueness checks, ordering, and casting."""
    if not isinstance(frame, pl.DataFrame):
        raise BacktestingValidationError(
            "engine output must be a polars DataFrame",
            error_code=_ERROR_INVALID_OUTPUT,
            details={"actual_type": type(frame).__name__},
        )
    if frame.height == 0:
        raise BacktestingValidationError(
            "engine output must contain at least one row",
            error_code=_ERROR_OUTPUT_EMPTY,
            details={"rows": frame.height},
        )
    _require_backtesting_schema_columns(frame)
    _require_unique_primary_keys(frame)
    ordered = frame.select(list(CANONICAL_COLUMN_ORDER))
    return ordered.cast(MERGED_BACKTESTING_SCHEMA)


def _require_engine_name(name: object) -> str:
    """Validate and return a non-blank engine name."""
    if not isinstance(name, str) or name.strip() == "":
        raise BacktestingValidationError(
            "engine_name must be a non-blank string",
            error_code=_ERROR_NAME_BLANK,
            details={"engine_name": name},
        )
    return name


def _require_manager(manager: object) -> str:
    """Validate and return a non-blank manager identity."""
    if not isinstance(manager, str) or manager.strip() == "":
        raise BacktestingValidationError(
            "manager must be a non-blank string",
            error_code=_ERROR_MANAGER_BLANK,
            details={"manager": manager},
        )
    return manager


def _require_backtesting_schema_columns(frame: pl.DataFrame) -> None:
    """Raise when any required merged-schema column is missing."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise BacktestingValidationError(
            "merged backtesting schema is missing required columns",
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
        raise BacktestingValidationError(
            "backtesting frame contains duplicate primary keys",
            error_code=_ERROR_DUPLICATE_KEYS,
            details={
                "primary_key_columns": PRIMARY_KEY_COLUMNS,
                "row_count": frame.height,
                "unique_key_count": unique_keys,
            },
        )
