"""CQROS Execution Engine package pipeline.

Purpose:
    Orchestrate deterministic conversion of canonical OMS order datasets into
    canonical executed-trade datasets through registered
    ``ExecutionSimulator`` implementations.

Responsibilities:
    - Validate simulator names and resolve simulators from
      ``ExecutionSimulatorRegistry``
    - Validate canonical order DataFrame structure
    - Delegate fill simulation exclusively to an injected simulator
    - Validate required trade schema columns on the simulator output
    - Reject duplicate primary keys
    - Finalize outputs against the canonical merged trade schema
    - Preserve order-frame immutability
    - Remain free of fill algorithms, persistence, verification, exchange
      APIs, and CLI logic

Dependencies:
    ``polars``, ``cqros.execution.exceptions``, ``cqros.execution.registry``,
    ``cqros.execution.schema``, and ``cqros.execution.simulator``.

Public API:
    ``ExecutionPipeline``
"""

from __future__ import annotations

from typing import Final

import polars as pl

from cqros.execution.exceptions import ExecutionValidationError
from cqros.execution.registry import ExecutionSimulatorRegistry
from cqros.execution.schema import (
    CANONICAL_COLUMN_ORDER,
    MERGED_TRADE_SCHEMA,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
)
from cqros.execution.simulator import validate_order_frame

__all__ = ["ExecutionPipeline"]

_ERROR_NAME_BLANK: Final[str] = "EXEC_PIPE_NAME_BLANK"
_ERROR_MANAGER_BLANK: Final[str] = "EXEC_PIPE_MANAGER_BLANK"
_ERROR_INVALID_OUTPUT: Final[str] = "EXEC_PIPE_INVALID_OUTPUT"
_ERROR_OUTPUT_EMPTY: Final[str] = "EXEC_PIPE_OUTPUT_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "EXEC_PIPE_MISSING_COLUMNS"
_ERROR_DUPLICATE_KEYS: Final[str] = "EXEC_PIPE_DUPLICATE_KEYS"

_PRIMARY_KEY_LIST: Final[list[str]] = list(PRIMARY_KEY_COLUMNS)

_DEFAULT_SIMULATOR: Final[str] = "simple"


class ExecutionPipeline:
    """Deterministic orchestrator for canonical executed-trade dataset assembly.

    The pipeline resolves a registered ``ExecutionSimulator``, validates a
    canonical OMS order DataFrame, delegates fill simulation, and finalizes
    the result to the canonical merged trade schema. Fill semantics remain
    exclusively in the simulator. The caller-supplied order frame is never
    mutated.

    Args:
        simulator_registry: Registry used to resolve execution-simulator
            implementations.
    """

    __slots__ = ("_simulator_registry",)

    _simulator_registry: ExecutionSimulatorRegistry

    def __init__(self, simulator_registry: ExecutionSimulatorRegistry) -> None:
        """Initialize the pipeline with an execution simulator registry.

        Args:
            simulator_registry: Registry containing ``ExecutionSimulator``
                implementations.
        """
        self._simulator_registry = simulator_registry

    def run(
        self,
        orders: pl.DataFrame,
        *,
        manager: str,
        simulator_name: str = _DEFAULT_SIMULATOR,
    ) -> pl.DataFrame:
        """Resolve a simulator and produce a finalized trade frame.

        ``simulator_name`` is validated and resolved first. ``manager`` is
        validated as a non-blank lineage identity. ``orders`` is validated
        through ``validate_order_frame``. Fill simulation is then delegated to
        ``ExecutionSimulator.execute``. The simulator output is checked against
        ``REQUIRED_COLUMNS``, rejected when primary keys are duplicated,
        reordered to ``CANONICAL_COLUMN_ORDER``, and cast to
        ``MERGED_TRADE_SCHEMA``. The original ``orders`` frame is never
        mutated.

        Args:
            orders: Canonical OMS order dataset.
            manager: Order manager identity stamped onto every trade row and
                used for partition lineage.
            simulator_name: Registry key of the execution simulator to
                execute. Defaults to ``simple``.

        Returns:
            A new DataFrame containing the finalized merged trade matrix.

        Raises:
            ExecutionValidationError: If ``simulator_name`` is invalid, the
                simulator is unknown, ``manager`` is blank, ``orders`` is
                invalid, or the simulator output fails trade-schema
                finalization.
        """
        validated_name = _require_simulator_name(simulator_name)
        validated_manager = _require_manager(manager)
        simulator = self._simulator_registry.get(validated_name)
        frame = validate_order_frame(orders)
        created = simulator.execute(frame, manager=validated_manager)
        return _finalize(created)


def _finalize(frame: object) -> pl.DataFrame:
    """Apply schema checks, uniqueness checks, ordering, and casting.

    Args:
        frame: Frame produced by ``ExecutionSimulator.execute``.

    Returns:
        Finalized merged trade DataFrame.

    Raises:
        ExecutionValidationError: If ``frame`` is not a non-empty DataFrame,
            required columns are missing, or primary keys are duplicated.
    """
    if not isinstance(frame, pl.DataFrame):
        raise ExecutionValidationError(
            "simulator output must be a polars DataFrame",
            error_code=_ERROR_INVALID_OUTPUT,
            details={"actual_type": type(frame).__name__},
        )
    if frame.height == 0:
        raise ExecutionValidationError(
            "simulator output must contain at least one row",
            error_code=_ERROR_OUTPUT_EMPTY,
            details={"rows": frame.height},
        )
    _require_trade_schema_columns(frame)
    _require_unique_primary_keys(frame)
    ordered = frame.select(list(CANONICAL_COLUMN_ORDER))
    return ordered.cast(MERGED_TRADE_SCHEMA)


def _require_simulator_name(name: object) -> str:
    """Validate and return a non-blank simulator name.

    Args:
        name: Candidate simulator registry key.

    Returns:
        The validated name string.

    Raises:
        ExecutionValidationError: If ``name`` is not a non-blank string.
    """
    if not isinstance(name, str) or name.strip() == "":
        raise ExecutionValidationError(
            "simulator_name must be a non-blank string",
            error_code=_ERROR_NAME_BLANK,
            details={"simulator_name": name},
        )
    return name


def _require_manager(manager: object) -> str:
    """Validate and return a non-blank manager identity.

    Args:
        manager: Candidate manager string.

    Returns:
        The validated manager string.

    Raises:
        ExecutionValidationError: If ``manager`` is not a non-blank string.
    """
    if not isinstance(manager, str) or manager.strip() == "":
        raise ExecutionValidationError(
            "manager must be a non-blank string",
            error_code=_ERROR_MANAGER_BLANK,
            details={"manager": manager},
        )
    return manager


def _require_trade_schema_columns(frame: pl.DataFrame) -> None:
    """Raise when any required merged-schema column is missing.

    Args:
        frame: Candidate trade DataFrame.

    Raises:
        ExecutionValidationError: If one or more ``REQUIRED_COLUMNS`` are
            absent.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ExecutionValidationError(
            "merged trade schema is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "missing_columns": tuple(missing),
                "required_columns": REQUIRED_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )


def _require_unique_primary_keys(frame: pl.DataFrame) -> None:
    """Raise when primary-key combinations are duplicated in ``frame``.

    Args:
        frame: DataFrame containing primary-key columns.

    Raises:
        ExecutionValidationError: If any primary-key combination appears more
            than once.
    """
    unique_keys = frame.select(_PRIMARY_KEY_LIST).n_unique()
    if unique_keys != frame.height:
        raise ExecutionValidationError(
            "trade frame contains duplicate primary keys",
            error_code=_ERROR_DUPLICATE_KEYS,
            details={
                "primary_key_columns": PRIMARY_KEY_COLUMNS,
                "row_count": frame.height,
                "unique_key_count": unique_keys,
            },
        )
