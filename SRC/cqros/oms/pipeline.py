"""CQROS Order Management System package pipeline.

Purpose:
    Orchestrate deterministic conversion of canonical risk-decision datasets
    into canonical OMS order datasets through registered ``OrderManager``
    implementations.

Responsibilities:
    - Validate manager names and resolve managers from
      ``OrderManagerRegistry``
    - Validate canonical risk DataFrame structure
    - Delegate order creation exclusively to an injected order manager
    - Validate required OMS Order schema columns on the manager output
    - Reject duplicate primary keys
    - Finalize outputs against the canonical merged order schema
    - Preserve risk-frame immutability
    - Remain free of order-generation algorithms, persistence, verification,
      exchange APIs, execution, and CLI logic

Dependencies:
    ``polars``, ``cqros.oms.exceptions``, ``cqros.oms.interfaces``,
    ``cqros.oms.registry``, and ``cqros.oms.schema``.

Public API:
    ``OrderPipeline``
"""

from __future__ import annotations

from typing import Final

import polars as pl

from cqros.oms.exceptions import OMSValidationError
from cqros.oms.interfaces import validate_risk_frame
from cqros.oms.registry import OrderManagerRegistry
from cqros.oms.schema import (
    CANONICAL_COLUMN_ORDER,
    MERGED_ORDER_SCHEMA,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
)

__all__ = ["OrderPipeline"]

_ERROR_NAME_BLANK: Final[str] = "OMS_PIPE_NAME_BLANK"
_ERROR_INVALID_OUTPUT: Final[str] = "OMS_PIPE_INVALID_OUTPUT"
_ERROR_OUTPUT_EMPTY: Final[str] = "OMS_PIPE_OUTPUT_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "OMS_PIPE_MISSING_COLUMNS"
_ERROR_DUPLICATE_KEYS: Final[str] = "OMS_PIPE_DUPLICATE_KEYS"

_PRIMARY_KEY_LIST: Final[list[str]] = list(PRIMARY_KEY_COLUMNS)


class OrderPipeline:
    """Deterministic orchestrator for canonical OMS order dataset assembly.

    The pipeline resolves a registered ``OrderManager``, validates a canonical
    risk-decision DataFrame, delegates order creation, and finalizes the
    result to the canonical merged order schema. Order-creation semantics
    remain exclusively in the manager. The caller-supplied risk frame is
    never mutated.

    Args:
        manager_registry: Registry used to resolve order-manager
            implementations.
    """

    __slots__ = ("_manager_registry",)

    _manager_registry: OrderManagerRegistry

    def __init__(self, manager_registry: OrderManagerRegistry) -> None:
        """Initialize the pipeline with an order manager registry.

        Args:
            manager_registry: Registry containing ``OrderManager``
                implementations.
        """
        self._manager_registry = manager_registry

    def run(self, manager_name: str, risk_decisions: pl.DataFrame) -> pl.DataFrame:
        """Resolve an order manager and produce a finalized order frame.

        ``manager_name`` is validated and resolved first. ``risk_decisions`` is
        validated through ``validate_risk_frame``. Order creation is then
        delegated to ``OrderManager.create_orders``. The manager output is
        checked against ``REQUIRED_COLUMNS``, rejected when primary keys are
        duplicated, reordered to ``CANONICAL_COLUMN_ORDER``, and cast to
        ``MERGED_ORDER_SCHEMA``. The original ``risk_decisions`` frame is never
        mutated.

        Args:
            manager_name: Registry key of the order manager to execute.
            risk_decisions: Canonical risk-decision dataset.

        Returns:
            A new DataFrame containing the finalized merged order matrix.

        Raises:
            OMSValidationError: If ``manager_name`` is invalid, the manager is
                unknown, ``risk_decisions`` is invalid, or the manager output
                fails order-schema finalization.
        """
        validated_name = _require_manager_name(manager_name)
        manager = self._manager_registry.get(validated_name)
        frame = validate_risk_frame(risk_decisions)
        created = manager.create_orders(frame)
        return _finalize(created)


def _finalize(frame: object) -> pl.DataFrame:
    """Apply schema checks, uniqueness checks, ordering, and casting.

    Args:
        frame: Frame produced by ``OrderManager.create_orders``.

    Returns:
        Finalized merged order DataFrame.

    Raises:
        OMSValidationError: If ``frame`` is not a non-empty DataFrame,
            required columns are missing, or primary keys are duplicated.
    """
    if not isinstance(frame, pl.DataFrame):
        raise OMSValidationError(
            "manager output must be a polars DataFrame",
            error_code=_ERROR_INVALID_OUTPUT,
            details={"actual_type": type(frame).__name__},
        )
    if frame.height == 0:
        raise OMSValidationError(
            "manager output must contain at least one row",
            error_code=_ERROR_OUTPUT_EMPTY,
            details={"rows": frame.height},
        )
    _require_order_schema_columns(frame)
    _require_unique_primary_keys(frame)
    ordered = frame.select(list(CANONICAL_COLUMN_ORDER))
    return ordered.cast(MERGED_ORDER_SCHEMA)


def _require_manager_name(name: object) -> str:
    """Validate and return a non-blank manager name.

    Args:
        name: Candidate manager registry key.

    Returns:
        The validated name string.

    Raises:
        OMSValidationError: If ``name`` is not a non-blank string.
    """
    if not isinstance(name, str) or name.strip() == "":
        raise OMSValidationError(
            "manager_name must be a non-blank string",
            error_code=_ERROR_NAME_BLANK,
            details={"manager_name": name},
        )
    return name


def _require_order_schema_columns(frame: pl.DataFrame) -> None:
    """Raise when any required merged-schema column is missing.

    Args:
        frame: Candidate order DataFrame.

    Raises:
        OMSValidationError: If one or more ``REQUIRED_COLUMNS`` are absent.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise OMSValidationError(
            "merged order schema is missing required columns",
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
        OMSValidationError: If any primary-key combination appears more than
            once.
    """
    unique_keys = frame.select(_PRIMARY_KEY_LIST).n_unique()
    if unique_keys != frame.height:
        raise OMSValidationError(
            "order frame contains duplicate primary keys",
            error_code=_ERROR_DUPLICATE_KEYS,
            details={
                "primary_key_columns": PRIMARY_KEY_COLUMNS,
                "row_count": frame.height,
                "unique_key_count": unique_keys,
            },
        )
