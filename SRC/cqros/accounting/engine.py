"""CQROS Portfolio Accounting Engine contracts and simple implementation.

Purpose:
    Convert canonical position datasets into canonical portfolio accounting
    snapshots using deterministic mark-to-market accounting.

Responsibilities:
    - Define ``AccountingEngine`` as the shared accounting contract
    - Provide ``SimplePortfolioAccountingEngine`` for long-only cash
      mark-to-market accounting
    - Validate position DataFrame structure and required position columns
    - Preserve upstream metadata lineage onto every accounting row
    - Remain free of persistence, verification, CLI, leverage, shorts,
      funding, margin, liquidation, commissions, slippage, and interest

Dependencies:
    ``polars``, ``cqros.accounting.exceptions``, and ``cqros.accounting.schema``.

Public API:
    ``AccountingEngine``, ``SimplePortfolioAccountingEngine``,
    ``POSITION_INPUT_COLUMNS``, ``validate_position_frame``
"""

from __future__ import annotations

import math
from typing import Final, Protocol, runtime_checkable

import polars as pl

from cqros.accounting.exceptions import AccountingValidationError
from cqros.accounting.schema import (
    CANONICAL_COLUMN_ORDER,
    MERGED_ACCOUNTING_SCHEMA,
)

__all__ = [
    "POSITION_INPUT_COLUMNS",
    "AccountingEngine",
    "SimplePortfolioAccountingEngine",
    "validate_position_frame",
]

_ERROR_FRAME_TYPE: Final[str] = "ACC_FRAME_TYPE"
_ERROR_FRAME_EMPTY: Final[str] = "ACC_FRAME_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "ACC_MISSING_COLUMNS"
_ERROR_MANAGER_BLANK: Final[str] = "ACC_MANAGER_BLANK"
_ERROR_CASH_NON_FINITE: Final[str] = "ACC_CASH_NON_FINITE"
_ERROR_INVALID_SIDE: Final[str] = "ACC_INVALID_SIDE"

_SIDE_LONG: Final[str] = "LONG"

# Position columns required to assemble an accounting row.
POSITION_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "position_id",
    "side",
    "status",
    "quantity",
    "average_entry_price",
    "market_price",
    "realized_pnl",
    "unrealized_pnl",
    "opened_at",
    "model_name",
    "model_version",
    "optimizer",
    "policy",
    "manager",
)


@runtime_checkable
class AccountingEngine(Protocol):
    """Structural contract for converting position frames into accounting frames.

    Implementations own portfolio accounting semantics. Pipeline orchestration
    delegates exclusively through this contract. Implementations must return a
    new DataFrame and must not mutate the input position frame.
    """

    def build(self, positions: pl.DataFrame, *, manager: str) -> pl.DataFrame:
        """Convert a canonical position DataFrame into an accounting DataFrame.

        Args:
            positions: Canonical position dataset. Must not be mutated.
            manager: Order manager identity preserved onto every accounting row.

        Returns:
            A new DataFrame containing the columns required by the merged
            accounting schema contract.
        """
        ...


class SimplePortfolioAccountingEngine:
    """Build mark-to-market accounting snapshots from position rows.

    Rules:
        - ``mark_price`` is taken from position ``market_price``
        - ``market_value = quantity * mark_price``
        - ``position_value = market_value``
        - ``unrealized_pnl = (mark_price - average_entry_price) * quantity``
        - ``total_pnl = realized_pnl + unrealized_pnl``
        - ``equity = cash + market_value`` (per row)
        - ``gross_exposure = Σ |position_value|`` across the input frame
        - ``net_exposure = Σ signed position_value`` across the input frame
        - ``return_pct = total_pnl / equity`` when equity is non-zero, else ``0``
        - No leverage, shorts, funding, borrowing, liquidation, commissions,
          slippage, or interest

    Args:
        cash: Portfolio cash balance broadcast onto every output row. Defaults
            to ``0.0`` when no cash ledger is available from positions.

    Notes:
        Implementations must not mutate the caller-supplied DataFrame.
    """

    __slots__ = ("_cash",)

    def __init__(self, *, cash: float = 0.0) -> None:
        """Initialize the engine with an optional cash balance.

        Args:
            cash: Portfolio cash balance applied to every accounting row.

        Raises:
            AccountingValidationError: If ``cash`` is not a finite float.
        """
        # ``bool`` is a subclass of ``int``; reject it before float coercion.
        if type(cash) is bool:
            raise AccountingValidationError(
                "cash must be a finite number",
                error_code=_ERROR_CASH_NON_FINITE,
                details={"cash": cash, "cash_type": type(cash).__name__},
            )
        cash_value = float(cash)
        if not math.isfinite(cash_value):
            raise AccountingValidationError(
                "cash must be a finite number",
                error_code=_ERROR_CASH_NON_FINITE,
                details={"cash": cash},
            )
        self._cash = cash_value

    def build(self, positions: pl.DataFrame, *, manager: str) -> pl.DataFrame:
        """Convert positions into finalized accounting rows.

        Args:
            positions: Canonical position dataset. Must not be mutated.
            manager: Order manager identity stamped onto every accounting row.

        Returns:
            A new DataFrame matching ``MERGED_ACCOUNTING_SCHEMA``.

        Raises:
            AccountingValidationError: If ``positions`` fails structural
                validation, ``manager`` is blank, required columns are missing,
                or an unsupported position side is encountered.
        """
        frame = validate_position_frame(positions)
        validated_manager = _require_manager(manager)
        _require_position_columns(frame)
        return _build_accounting_frame(
            frame,
            manager=validated_manager,
            cash=self._cash,
        )


def validate_position_frame(frame: object) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty Polars DataFrame.

    Args:
        frame: Candidate position dataset passed to an engine.

    Returns:
        ``frame`` as a DataFrame after structural checks.

    Raises:
        AccountingValidationError: If ``frame`` is not a Polars DataFrame or
            contains no rows.
    """
    if not isinstance(frame, pl.DataFrame):
        raise AccountingValidationError(
            "frame must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={"actual_type": type(frame).__name__},
        )
    if frame.height == 0:
        raise AccountingValidationError(
            "frame must contain at least one row",
            error_code=_ERROR_FRAME_EMPTY,
            details={"rows": frame.height},
        )
    return frame


def _build_accounting_frame(
    frame: pl.DataFrame,
    *,
    manager: str,
    cash: float,
) -> pl.DataFrame:
    """Assemble canonical accounting rows from a validated position frame.

    Args:
        frame: Validated position DataFrame.
        manager: Order manager identity for lineage.
        cash: Portfolio cash balance broadcast onto every row.

    Returns:
        A new DataFrame ordered and cast to ``MERGED_ACCOUNTING_SCHEMA``.
    """
    sides = frame["side"].to_list()
    for side in sides:
        if str(side) != _SIDE_LONG:
            raise AccountingValidationError(
                f"unsupported position side: {side}",
                error_code=_ERROR_INVALID_SIDE,
                details={"side": side, "allowed_sides": (_SIDE_LONG,)},
            )

    quantity = pl.col("quantity")
    mark_price = pl.col("market_price")
    average_entry = pl.col("average_entry_price")
    realized = pl.col("realized_pnl")

    market_value = quantity * mark_price
    position_value = market_value
    unrealized = (mark_price - average_entry) * quantity
    total_pnl = realized + unrealized
    equity = pl.lit(cash) + market_value
    signed_value = position_value  # LONG-only: signed exposure equals value
    absolute_value = position_value.abs()

    # Portfolio-level exposures computed once and broadcast to every row.
    gross_exposure = absolute_value.sum()
    net_exposure = signed_value.sum()

    return_pct = pl.when(equity != 0.0).then(total_pnl / equity).otherwise(pl.lit(0.0))

    assembled = frame.select(
        pl.col("symbol"),
        pl.col("timeframe"),
        pl.col("opened_at").alias("open_time"),
        pl.lit(manager).alias("manager"),
        pl.col("position_id"),
        pl.col("status").alias("position_status"),
        quantity.alias("quantity"),
        average_entry.alias("average_entry_price"),
        mark_price.alias("mark_price"),
        position_value.alias("position_value"),
        market_value.alias("market_value"),
        pl.lit(cash).alias("cash"),
        realized.alias("realized_pnl"),
        unrealized.alias("unrealized_pnl"),
        total_pnl.alias("total_pnl"),
        gross_exposure.alias("gross_exposure"),
        net_exposure.alias("net_exposure"),
        equity.alias("equity"),
        return_pct.alias("return_pct"),
        pl.col("model_name"),
        pl.col("model_version"),
        pl.col("optimizer"),
        pl.col("policy"),
    )
    ordered_rows = assembled.sort("open_time", maintain_order=True)
    return ordered_rows.select(list(CANONICAL_COLUMN_ORDER)).cast(MERGED_ACCOUNTING_SCHEMA)


def _require_manager(manager: object) -> str:
    """Validate and return a non-blank manager identity."""
    if not isinstance(manager, str) or manager.strip() == "":
        raise AccountingValidationError(
            "manager must be a non-blank string",
            error_code=_ERROR_MANAGER_BLANK,
            details={"manager": manager},
        )
    return manager


def _require_position_columns(frame: pl.DataFrame) -> None:
    """Raise when any required position column is missing."""
    missing = [column for column in POSITION_INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise AccountingValidationError(
            "position frame is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "missing_columns": tuple(missing),
                "required_columns": POSITION_INPUT_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )
