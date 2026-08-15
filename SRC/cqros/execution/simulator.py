"""CQROS execution simulator contracts and market fill implementation.

Purpose:
    Convert canonical OMS order datasets into canonical executed-trade
    datasets using deterministic, zero-cost market fill rules.

Responsibilities:
    - Define ``ExecutionSimulator`` as the shared simulation contract
    - Provide ``SimpleExecutionSimulator`` for immediate market fills
    - Validate order DataFrame structure and required OMS columns
    - Preserve upstream metadata lineage onto every trade row
    - Remain free of persistence, verification, CLI, and exchange APIs

Dependencies:
    ``polars``, ``cqros.execution.exceptions``, and ``cqros.execution.schema``.

Public API:
    ``ExecutionSimulator``, ``SimpleExecutionSimulator``,
    ``ORDER_INPUT_COLUMNS``, ``validate_order_frame``
"""

from __future__ import annotations

from typing import Final, Protocol, runtime_checkable

import polars as pl

from cqros.execution.exceptions import ExecutionValidationError
from cqros.execution.schema import (
    CANONICAL_COLUMN_ORDER,
    MERGED_TRADE_SCHEMA,
    ExecutionStatus,
)

__all__ = [
    "ORDER_INPUT_COLUMNS",
    "ExecutionSimulator",
    "SimpleExecutionSimulator",
    "validate_order_frame",
]

_ERROR_FRAME_TYPE: Final[str] = "EXEC_FRAME_TYPE"
_ERROR_FRAME_EMPTY: Final[str] = "EXEC_FRAME_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "EXEC_MISSING_COLUMNS"
_ERROR_DUPLICATE_KEYS: Final[str] = "EXEC_DUPLICATE_KEYS"
_ERROR_MANAGER_BLANK: Final[str] = "EXEC_MANAGER_BLANK"
_ERROR_NO_MARKET_ORDERS: Final[str] = "EXEC_NO_MARKET_ORDERS"

_ORDER_TYPE_MARKET: Final[str] = "MARKET"

# OMS order columns required to assemble a trade row.
ORDER_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
    "model_name",
    "model_version",
    "policy",
    "optimizer",
    "side",
    "order_type",
    "quantity",
    "limit_price",
)

_ORDER_PRIMARY_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
)

_ORDER_PRIMARY_KEY_LIST: Final[list[str]] = list(_ORDER_PRIMARY_KEY_COLUMNS)


@runtime_checkable
class ExecutionSimulator(Protocol):
    """Structural contract for converting order frames into trade frames.

    Implementations own fill semantics. Pipeline orchestration delegates
    execution exclusively through this contract. Implementations must return
    a new DataFrame and must not mutate the input order frame.
    """

    def execute(self, orders: pl.DataFrame, *, manager: str) -> pl.DataFrame:
        """Convert a canonical OMS order DataFrame into a trade DataFrame.

        Args:
            orders: Canonical OMS order dataset. Must not be mutated.
            manager: Order manager identity preserved onto every trade row.

        Returns:
            A new DataFrame containing the columns required by the merged
            trade schema contract.
        """
        ...


class SimpleExecutionSimulator:
    """Fill market orders immediately with zero fees and zero slippage.

    Rules:
        - Only ``MARKET`` order rows are executed
        - ``executed_price = requested_price`` (``limit_price``, null filled
          as ``0.0`` when absent — v1 has no market-data mid price)
        - ``executed_quantity = requested_quantity`` (``quantity``)
        - ``fees = 0`` and ``slippage = 0``
        - ``status = FILLED``
        - ``execution_time = open_time`` (no latency)
        - ``signal`` is preserved when present; otherwise derived from ``side``
        - No randomness, spread, partial fills, or funding

    Notes:
        Implementations must not mutate the caller-supplied DataFrame.
    """

    __slots__ = ()

    def execute(self, orders: pl.DataFrame, *, manager: str) -> pl.DataFrame:
        """Convert OMS market orders into filled trade rows.

        Args:
            orders: Canonical OMS order dataset. Must not be mutated.
            manager: Order manager identity stamped onto every trade row.

        Returns:
            A new DataFrame matching ``MERGED_TRADE_SCHEMA``.

        Raises:
            ExecutionValidationError: If ``orders`` fails structural
                validation, ``manager`` is blank, required columns are
                missing, primary keys are duplicated, or no market orders
                remain after filtering.
        """
        frame = validate_order_frame(orders)
        validated_manager = _require_manager(manager)
        _require_order_columns(frame)
        _require_unique_primary_keys(frame)
        return _build_trade_frame(frame, manager=validated_manager)


def validate_order_frame(frame: object) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty Polars DataFrame.

    Args:
        frame: Candidate OMS order dataset passed to a simulator.

    Returns:
        ``frame`` as a DataFrame after structural checks.

    Raises:
        ExecutionValidationError: If ``frame`` is not a Polars DataFrame or
            contains no rows.
    """
    if not isinstance(frame, pl.DataFrame):
        raise ExecutionValidationError(
            "frame must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={"actual_type": type(frame).__name__},
        )
    if frame.height == 0:
        raise ExecutionValidationError(
            "frame must contain at least one row",
            error_code=_ERROR_FRAME_EMPTY,
            details={"rows": frame.height},
        )
    return frame


def _build_trade_frame(frame: pl.DataFrame, *, manager: str) -> pl.DataFrame:
    """Assemble a canonical trade DataFrame from eligible market orders.

    Args:
        frame: Validated OMS order DataFrame.
        manager: Order manager identity for lineage.

    Returns:
        A new DataFrame ordered and cast to ``MERGED_TRADE_SCHEMA``.

    Raises:
        ExecutionValidationError: If no ``MARKET`` rows remain after
            filtering.
    """
    eligible = frame.filter(pl.col("order_type") == _ORDER_TYPE_MARKET)
    if eligible.height == 0:
        raise ExecutionValidationError(
            "orders must contain at least one MARKET row",
            error_code=_ERROR_NO_MARKET_ORDERS,
            details={"rows": frame.height, "market_rows": 0},
        )

    signal_expr = pl.col("signal") if "signal" in eligible.columns else pl.col("side")
    assembled = eligible.select(
        pl.col("symbol"),
        pl.col("timeframe"),
        pl.col("open_time"),
        pl.col("model_name"),
        pl.col("model_version"),
        pl.col("optimizer"),
        pl.col("policy"),
        pl.lit(manager).alias("manager"),
        signal_expr.alias("signal"),
        pl.col("side"),
        pl.col("order_type"),
        pl.col("quantity").alias("requested_quantity"),
        pl.col("quantity").alias("executed_quantity"),
        pl.col("limit_price").fill_null(0.0).alias("requested_price"),
        pl.col("limit_price").fill_null(0.0).alias("executed_price"),
        pl.lit(0.0, dtype=pl.Float64).alias("fees"),
        pl.lit(0.0, dtype=pl.Float64).alias("slippage"),
        pl.lit(ExecutionStatus.FILLED.value).alias("status"),
        pl.col("open_time").alias("execution_time"),
    )
    return assembled.select(list(CANONICAL_COLUMN_ORDER)).cast(MERGED_TRADE_SCHEMA)


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


def _require_order_columns(frame: pl.DataFrame) -> None:
    """Raise when any required OMS order column is missing.

    Args:
        frame: Candidate order DataFrame.

    Raises:
        ExecutionValidationError: If one or more required columns are absent.
    """
    missing = [column for column in ORDER_INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise ExecutionValidationError(
            "order frame is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "missing_columns": tuple(missing),
                "required_columns": ORDER_INPUT_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )


def _require_unique_primary_keys(frame: pl.DataFrame) -> None:
    """Raise when order primary-key combinations are duplicated.

    Args:
        frame: DataFrame containing order primary-key columns.

    Raises:
        ExecutionValidationError: If any primary-key combination appears more
            than once.
    """
    unique_keys = frame.select(_ORDER_PRIMARY_KEY_LIST).n_unique()
    if unique_keys != frame.height:
        raise ExecutionValidationError(
            "order frame contains duplicate primary keys",
            error_code=_ERROR_DUPLICATE_KEYS,
            details={
                "primary_key_columns": _ORDER_PRIMARY_KEY_COLUMNS,
                "row_count": frame.height,
                "unique_key_count": unique_keys,
            },
        )
