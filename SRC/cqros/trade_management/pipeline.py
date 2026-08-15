"""CQROS Trade Management Engine package pipeline.

Purpose:
    Orchestrate deterministic conversion of canonical open-position inputs
    into canonical trade management decision datasets through registered
    ``TradeManagementManager`` implementations.

Responsibilities:
    - Validate manager names and resolve managers from
      ``TradeManagementManagerRegistry``
    - Validate canonical input DataFrame structure
    - Delegate trade-management evaluation exclusively to an injected manager
    - Validate required trade-management schema columns on the manager output
    - Reject duplicate primary keys
    - Finalize outputs against the canonical merged trade-management schema
    - Preserve input-frame immutability
    - Remain free of trade-management algorithms, persistence, verification,
      exchange APIs, and CLI logic

Dependencies:
    ``polars``, ``cqros.trade_management.manager``,
    ``cqros.trade_management.exceptions``, ``cqros.trade_management.registry``,
    and ``cqros.trade_management.schema``.

Public API:
    ``TradeManagementPipeline``
"""

from __future__ import annotations

from typing import Final

import polars as pl

from cqros.trade_management.exceptions import TradeManagementValidationError
from cqros.trade_management.manager import (
    validate_accounting_frame,
    validate_market_price_frame,
    validate_portfolio_risk_frame,
    validate_position_frame,
)
from cqros.trade_management.registry import TradeManagementManagerRegistry
from cqros.trade_management.schema import (
    CANONICAL_COLUMN_ORDER,
    MERGED_TRADE_MANAGEMENT_SCHEMA,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
)

__all__ = ["TradeManagementPipeline"]

_ERROR_NAME_BLANK: Final[str] = "TME_PIPE_NAME_BLANK"
_ERROR_MANAGER_BLANK: Final[str] = "TME_PIPE_MANAGER_BLANK"
_ERROR_INVALID_OUTPUT: Final[str] = "TME_PIPE_INVALID_OUTPUT"
_ERROR_OUTPUT_EMPTY: Final[str] = "TME_PIPE_OUTPUT_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "TME_PIPE_MISSING_COLUMNS"
_ERROR_DUPLICATE_KEYS: Final[str] = "TME_PIPE_DUPLICATE_KEYS"

_PRIMARY_KEY_LIST: Final[list[str]] = list(PRIMARY_KEY_COLUMNS)

_DEFAULT_TRADE_MANAGER: Final[str] = "simple"


class TradeManagementPipeline:
    """Deterministic orchestrator for canonical trade-management assembly.

    The pipeline resolves a registered ``TradeManagementManager``, validates
    canonical input DataFrames, delegates evaluation, and finalizes the
    result to the canonical merged trade-management schema. Trade-management
    semantics remain exclusively in the manager. The caller-supplied frames
    are never mutated.

    Args:
        manager_registry: Registry used to resolve trade-management-manager
            implementations.
    """

    __slots__ = ("_manager_registry",)

    _manager_registry: TradeManagementManagerRegistry

    def __init__(self, manager_registry: TradeManagementManagerRegistry) -> None:
        """Initialize the pipeline with a trade management manager registry.

        Args:
            manager_registry: Registry containing ``TradeManagementManager``
                implementations.
        """
        self._manager_registry = manager_registry

    def run(
        self,
        positions: pl.DataFrame,
        accounting: pl.DataFrame,
        portfolio_risk: pl.DataFrame,
        market_prices: pl.DataFrame,
        *,
        manager: str,
        trade_manager_name: str = _DEFAULT_TRADE_MANAGER,
    ) -> pl.DataFrame:
        """Resolve a manager and produce a finalized trade-management frame.

        ``trade_manager_name`` is validated and resolved first. ``manager`` is
        validated as a non-blank lineage identity. Input frames are validated
        through their respective frame validators. Evaluation is then
        delegated to ``TradeManagementManager.evaluate``. The manager output
        is checked against ``REQUIRED_COLUMNS``, rejected when primary keys
        are duplicated, reordered to ``CANONICAL_COLUMN_ORDER``, and cast to
        ``MERGED_TRADE_MANAGEMENT_SCHEMA``. The original frames are never
        mutated.

        Args:
            positions: Canonical position dataset.
            accounting: Canonical accounting dataset.
            portfolio_risk: Canonical portfolio-risk dataset.
            market_prices: Market price dataset keyed by open_time.
            manager: Order manager identity stamped onto every decision row
                and used for partition lineage.
            trade_manager_name: Registry key of the trade management manager
                to execute. Defaults to ``simple``.

        Returns:
            A new DataFrame containing the finalized merged trade-management
            matrix.

        Raises:
            TradeManagementValidationError: If ``trade_manager_name`` is
                invalid, the manager is unknown, ``manager`` is blank, inputs
                are invalid, or the manager output fails schema finalization.
        """
        validated_name = _require_trade_manager_name(trade_manager_name)
        validated_manager = _require_manager(manager)
        trade_manager = self._manager_registry.get(validated_name)
        positions_frame = validate_position_frame(positions)
        accounting_frame = validate_accounting_frame(accounting)
        risk_frame = validate_portfolio_risk_frame(portfolio_risk)
        prices_frame = validate_market_price_frame(market_prices)
        created = trade_manager.evaluate(
            positions_frame,
            accounting_frame,
            risk_frame,
            prices_frame,
            manager=validated_manager,
        )
        return _finalize(created)


def _finalize(frame: object) -> pl.DataFrame:
    """Apply schema checks, uniqueness checks, ordering, and casting."""
    if not isinstance(frame, pl.DataFrame):
        raise TradeManagementValidationError(
            "manager output must be a polars DataFrame",
            error_code=_ERROR_INVALID_OUTPUT,
            details={"actual_type": type(frame).__name__},
        )
    if frame.height == 0:
        raise TradeManagementValidationError(
            "manager output must contain at least one row",
            error_code=_ERROR_OUTPUT_EMPTY,
            details={"rows": frame.height},
        )
    _require_trade_management_schema_columns(frame)
    _require_unique_primary_keys(frame)
    ordered = frame.select(list(CANONICAL_COLUMN_ORDER))
    return ordered.cast(MERGED_TRADE_MANAGEMENT_SCHEMA)


def _require_trade_manager_name(name: object) -> str:
    """Validate and return a non-blank trade-manager name."""
    if not isinstance(name, str) or name.strip() == "":
        raise TradeManagementValidationError(
            "trade_manager_name must be a non-blank string",
            error_code=_ERROR_NAME_BLANK,
            details={"trade_manager_name": name},
        )
    return name


def _require_manager(manager: object) -> str:
    """Validate and return a non-blank manager identity."""
    if not isinstance(manager, str) or manager.strip() == "":
        raise TradeManagementValidationError(
            "manager must be a non-blank string",
            error_code=_ERROR_MANAGER_BLANK,
            details={"manager": manager},
        )
    return manager


def _require_trade_management_schema_columns(frame: pl.DataFrame) -> None:
    """Raise when any required merged-schema column is missing."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise TradeManagementValidationError(
            "merged trade-management schema is missing required columns",
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
        raise TradeManagementValidationError(
            "trade-management frame contains duplicate primary keys",
            error_code=_ERROR_DUPLICATE_KEYS,
            details={
                "primary_key_columns": PRIMARY_KEY_COLUMNS,
                "row_count": frame.height,
                "unique_key_count": unique_keys,
            },
        )
