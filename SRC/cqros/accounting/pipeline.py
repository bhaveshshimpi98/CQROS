"""CQROS Portfolio Accounting Engine package pipeline.

Purpose:
    Orchestrate deterministic conversion of canonical position datasets into
    canonical portfolio accounting datasets through registered
    ``AccountingEngine`` implementations.

Responsibilities:
    - Validate engine names and resolve engines from
      ``AccountingEngineRegistry``
    - Validate canonical position DataFrame structure
    - Delegate accounting exclusively to an injected engine
    - Validate required accounting schema columns on the engine output
    - Reject duplicate primary keys
    - Finalize outputs against the canonical merged accounting schema
    - Preserve position-frame immutability
    - Remain free of accounting algorithms, persistence, verification,
      exchange APIs, and CLI logic

Dependencies:
    ``polars``, ``cqros.accounting.engine``, ``cqros.accounting.exceptions``,
    ``cqros.accounting.registry``, and ``cqros.accounting.schema``.

Public API:
    ``AccountingPipeline``
"""

from __future__ import annotations

from typing import Final

import polars as pl

from cqros.accounting.engine import validate_position_frame
from cqros.accounting.exceptions import AccountingValidationError
from cqros.accounting.registry import AccountingEngineRegistry
from cqros.accounting.schema import (
    CANONICAL_COLUMN_ORDER,
    MERGED_ACCOUNTING_SCHEMA,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
)

__all__ = ["AccountingPipeline"]

_ERROR_NAME_BLANK: Final[str] = "ACC_PIPE_NAME_BLANK"
_ERROR_MANAGER_BLANK: Final[str] = "ACC_PIPE_MANAGER_BLANK"
_ERROR_INVALID_OUTPUT: Final[str] = "ACC_PIPE_INVALID_OUTPUT"
_ERROR_OUTPUT_EMPTY: Final[str] = "ACC_PIPE_OUTPUT_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "ACC_PIPE_MISSING_COLUMNS"
_ERROR_DUPLICATE_KEYS: Final[str] = "ACC_PIPE_DUPLICATE_KEYS"

_PRIMARY_KEY_LIST: Final[list[str]] = list(PRIMARY_KEY_COLUMNS)

_DEFAULT_ENGINE: Final[str] = "simple"


class AccountingPipeline:
    """Deterministic orchestrator for canonical accounting dataset assembly.

    The pipeline resolves a registered ``AccountingEngine``, validates a
    canonical position DataFrame, delegates accounting, and finalizes the
    result to the canonical merged accounting schema. Accounting semantics
    remain exclusively in the engine. The caller-supplied position frame is
    never mutated.

    Args:
        engine_registry: Registry used to resolve accounting-engine
            implementations.
    """

    __slots__ = ("_engine_registry",)

    _engine_registry: AccountingEngineRegistry

    def __init__(self, engine_registry: AccountingEngineRegistry) -> None:
        """Initialize the pipeline with an accounting engine registry.

        Args:
            engine_registry: Registry containing ``AccountingEngine``
                implementations.
        """
        self._engine_registry = engine_registry

    def run(
        self,
        positions: pl.DataFrame,
        *,
        manager: str,
        engine_name: str = _DEFAULT_ENGINE,
    ) -> pl.DataFrame:
        """Resolve an engine and produce a finalized accounting frame.

        ``engine_name`` is validated and resolved first. ``manager`` is
        validated as a non-blank lineage identity. ``positions`` is validated
        through ``validate_position_frame``. Accounting is then delegated to
        ``AccountingEngine.build``. The engine output is checked against
        ``REQUIRED_COLUMNS``, rejected when primary keys are duplicated,
        reordered to ``CANONICAL_COLUMN_ORDER``, and cast to
        ``MERGED_ACCOUNTING_SCHEMA``. The original ``positions`` frame is never
        mutated.

        Args:
            positions: Canonical position dataset.
            manager: Order manager identity stamped onto every accounting row
                and used for partition lineage.
            engine_name: Registry key of the accounting engine to execute.
                Defaults to ``simple``.

        Returns:
            A new DataFrame containing the finalized merged accounting matrix.

        Raises:
            AccountingValidationError: If ``engine_name`` is invalid, the engine
                is unknown, ``manager`` is blank, ``positions`` is invalid, or
                the engine output fails accounting-schema finalization.
        """
        validated_name = _require_engine_name(engine_name)
        validated_manager = _require_manager(manager)
        engine = self._engine_registry.get(validated_name)
        frame = validate_position_frame(positions)
        created = engine.build(frame, manager=validated_manager)
        return _finalize(created)


def _finalize(frame: object) -> pl.DataFrame:
    """Apply schema checks, uniqueness checks, ordering, and casting."""
    if not isinstance(frame, pl.DataFrame):
        raise AccountingValidationError(
            "engine output must be a polars DataFrame",
            error_code=_ERROR_INVALID_OUTPUT,
            details={"actual_type": type(frame).__name__},
        )
    if frame.height == 0:
        raise AccountingValidationError(
            "engine output must contain at least one row",
            error_code=_ERROR_OUTPUT_EMPTY,
            details={"rows": frame.height},
        )
    _require_accounting_schema_columns(frame)
    _require_unique_primary_keys(frame)
    ordered = frame.select(list(CANONICAL_COLUMN_ORDER))
    return ordered.cast(MERGED_ACCOUNTING_SCHEMA)


def _require_engine_name(name: object) -> str:
    """Validate and return a non-blank engine name."""
    if not isinstance(name, str) or name.strip() == "":
        raise AccountingValidationError(
            "engine_name must be a non-blank string",
            error_code=_ERROR_NAME_BLANK,
            details={"engine_name": name},
        )
    return name


def _require_manager(manager: object) -> str:
    """Validate and return a non-blank manager identity."""
    if not isinstance(manager, str) or manager.strip() == "":
        raise AccountingValidationError(
            "manager must be a non-blank string",
            error_code=_ERROR_MANAGER_BLANK,
            details={"manager": manager},
        )
    return manager


def _require_accounting_schema_columns(frame: pl.DataFrame) -> None:
    """Raise when any required merged-schema column is missing."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise AccountingValidationError(
            "merged accounting schema is missing required columns",
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
        raise AccountingValidationError(
            "accounting frame contains duplicate primary keys",
            error_code=_ERROR_DUPLICATE_KEYS,
            details={
                "primary_key_columns": PRIMARY_KEY_COLUMNS,
                "row_count": frame.height,
                "unique_key_count": unique_keys,
            },
        )
