"""CQROS Exit Engine package pipeline.

Purpose:
    Orchestrate deterministic conversion of canonical open-position inputs
    into canonical exit recommendation datasets through registered
    ``ExitEngine`` implementations.

Responsibilities:
    - Validate engine names and resolve engines from ``ExitEngineRegistry``
    - Validate canonical input DataFrame structure
    - Delegate exit evaluation exclusively to an injected engine
    - Validate required exit-engine schema columns on the engine output
    - Reject duplicate primary keys
    - Finalize outputs against the canonical merged exit-engine schema
    - Preserve input-frame immutability
    - Remain free of exit algorithms, persistence, verification, exchange
      APIs, and CLI logic

Dependencies:
    ``polars``, ``cqros.exit_engine.engine``, ``cqros.exit_engine.exceptions``,
    ``cqros.exit_engine.registry``, and ``cqros.exit_engine.schema``.

Public API:
    ``ExitEnginePipeline``
"""

from __future__ import annotations

from typing import Final

import polars as pl

from cqros.exit_engine.engine import (
    validate_accounting_frame,
    validate_portfolio_risk_frame,
    validate_position_frame,
    validate_pyramiding_frame,
    validate_trade_management_frame,
)
from cqros.exit_engine.exceptions import ExitEngineValidationError
from cqros.exit_engine.registry import ExitEngineRegistry
from cqros.exit_engine.schema import (
    CANONICAL_COLUMN_ORDER,
    MERGED_EXIT_ENGINE_SCHEMA,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
)

__all__ = ["ExitEnginePipeline"]

_ERROR_NAME_BLANK: Final[str] = "EXIT_PIPE_NAME_BLANK"
_ERROR_MANAGER_BLANK: Final[str] = "EXIT_PIPE_MANAGER_BLANK"
_ERROR_INVALID_OUTPUT: Final[str] = "EXIT_PIPE_INVALID_OUTPUT"
_ERROR_MISSING_COLUMNS: Final[str] = "EXIT_PIPE_MISSING_COLUMNS"
_ERROR_DUPLICATE_KEYS: Final[str] = "EXIT_PIPE_DUPLICATE_KEYS"

_PRIMARY_KEY_LIST: Final[list[str]] = list(PRIMARY_KEY_COLUMNS)

_DEFAULT_ENGINE: Final[str] = "simple"


class ExitEnginePipeline:
    """Deterministic orchestrator for canonical exit-recommendation assembly.

    The pipeline resolves a registered ``ExitEngine``, validates canonical
    input DataFrames, delegates evaluation, and finalizes the result to the
    canonical merged exit-engine schema. Exit semantics remain exclusively
    in the engine. The caller-supplied frames are never mutated. Empty
    outputs are permitted when no OPEN positions exist.

    Args:
        registry: Registry used to resolve exit-engine implementations.
    """

    __slots__ = ("_registry",)

    _registry: ExitEngineRegistry

    def __init__(self, registry: ExitEngineRegistry) -> None:
        """Initialize the pipeline with an exit-engine registry.

        Args:
            registry: Registry containing ``ExitEngine`` implementations.
        """
        self._registry = registry

    def run(
        self,
        positions: pl.DataFrame,
        accounting: pl.DataFrame,
        portfolio_risk: pl.DataFrame,
        trade_management: pl.DataFrame,
        pyramiding: pl.DataFrame,
        *,
        manager: str,
        engine_name: str = _DEFAULT_ENGINE,
    ) -> pl.DataFrame:
        """Resolve an engine and produce a finalized exit frame.

        ``engine_name`` is validated and resolved first. ``manager`` is
        validated as a non-blank lineage identity. Input frames are validated
        through their respective frame validators. Evaluation is then
        delegated to ``ExitEngine.evaluate``. The engine output is checked
        against ``REQUIRED_COLUMNS``, rejected when primary keys are
        duplicated, reordered to ``CANONICAL_COLUMN_ORDER``, and cast to
        ``MERGED_EXIT_ENGINE_SCHEMA``. The original frames are never mutated.

        Args:
            positions: Canonical position dataset.
            accounting: Canonical accounting dataset.
            portfolio_risk: Canonical portfolio-risk dataset.
            trade_management: Canonical trade-management dataset.
            pyramiding: Canonical pyramiding dataset.
            manager: Order manager identity stamped onto every decision row
                and used for partition lineage.
            engine_name: Registry key of the exit engine to execute.
                Defaults to ``simple``.

        Returns:
            A new DataFrame containing the finalized merged exit matrix.
            May be empty when no OPEN positions exist.

        Raises:
            ExitEngineValidationError: If ``engine_name`` is invalid, the
                engine is unknown, ``manager`` is blank, inputs are invalid,
                or the engine output fails schema finalization.
        """
        validated_name = _require_engine_name(engine_name)
        validated_manager = _require_manager(manager)
        engine = self._registry.get(validated_name)
        positions_frame = validate_position_frame(positions)
        accounting_frame = validate_accounting_frame(accounting)
        risk_frame = validate_portfolio_risk_frame(portfolio_risk)
        tm_frame = validate_trade_management_frame(trade_management)
        pyramid_frame = validate_pyramiding_frame(pyramiding)
        created = engine.evaluate(
            positions_frame,
            accounting_frame,
            risk_frame,
            tm_frame,
            pyramid_frame,
            manager=validated_manager,
        )
        return _finalize(created)


def _finalize(frame: object) -> pl.DataFrame:
    """Apply schema checks, uniqueness checks, ordering, and casting."""
    if not isinstance(frame, pl.DataFrame):
        raise ExitEngineValidationError(
            "engine output must be a polars DataFrame",
            error_code=_ERROR_INVALID_OUTPUT,
            details={"actual_type": type(frame).__name__},
        )
    if frame.height == 0:
        return pl.DataFrame(schema=MERGED_EXIT_ENGINE_SCHEMA)
    _require_exit_schema_columns(frame)
    _require_unique_primary_keys(frame)
    ordered = frame.select(list(CANONICAL_COLUMN_ORDER))
    return ordered.cast(MERGED_EXIT_ENGINE_SCHEMA)


def _require_engine_name(name: object) -> str:
    """Validate and return a non-blank engine name."""
    if not isinstance(name, str) or name.strip() == "":
        raise ExitEngineValidationError(
            "engine_name must be a non-blank string",
            error_code=_ERROR_NAME_BLANK,
            details={"engine_name": name},
        )
    return name


def _require_manager(manager: object) -> str:
    """Validate and return a non-blank manager identity."""
    if not isinstance(manager, str) or manager.strip() == "":
        raise ExitEngineValidationError(
            "manager must be a non-blank string",
            error_code=_ERROR_MANAGER_BLANK,
            details={"manager": manager},
        )
    return manager


def _require_exit_schema_columns(frame: pl.DataFrame) -> None:
    """Raise when any required merged-schema column is missing."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ExitEngineValidationError(
            "merged exit-engine schema is missing required columns",
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
        raise ExitEngineValidationError(
            "exit-engine frame contains duplicate primary keys",
            error_code=_ERROR_DUPLICATE_KEYS,
            details={
                "primary_key_columns": PRIMARY_KEY_COLUMNS,
                "row_count": frame.height,
                "unique_key_count": unique_keys,
            },
        )
