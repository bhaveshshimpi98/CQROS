"""CQROS Position Engine contracts and average-cost implementation.

Purpose:
    Convert canonical executed-trade datasets into canonical portfolio
    position datasets using deterministic average-cost accounting.

Responsibilities:
    - Define ``PositionEngine`` as the shared accounting contract
    - Provide ``AverageCostPositionEngine`` for long-only cash positions
    - Validate trade DataFrame structure and required trade columns
    - Consume trades ordered by ``execution_time``
    - Preserve upstream metadata lineage onto every position row
    - Remain free of persistence, verification, CLI, leverage, shorts,
      funding, margin, and liquidation logic

Dependencies:
    ``polars``, ``cqros.positions.exceptions``, and ``cqros.positions.schema``.

Public API:
    ``PositionEngine``, ``AverageCostPositionEngine``,
    ``TRADE_INPUT_COLUMNS``, ``validate_trade_frame``
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Final, Protocol, runtime_checkable

import polars as pl

from cqros.positions.exceptions import PositionValidationError
from cqros.positions.schema import (
    CANONICAL_COLUMN_ORDER,
    MERGED_POSITION_SCHEMA,
    PositionSide,
    PositionStatus,
)

__all__ = [
    "TRADE_INPUT_COLUMNS",
    "AverageCostPositionEngine",
    "PositionEngine",
    "validate_trade_frame",
]

_ERROR_FRAME_TYPE: Final[str] = "POS_FRAME_TYPE"
_ERROR_FRAME_EMPTY: Final[str] = "POS_FRAME_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "POS_MISSING_COLUMNS"
_ERROR_MANAGER_BLANK: Final[str] = "POS_MANAGER_BLANK"
_ERROR_INVALID_SIDE: Final[str] = "POS_INVALID_SIDE"
_ERROR_NO_SHORTS: Final[str] = "POS_NO_SHORTS"
_ERROR_OVERSELL: Final[str] = "POS_OVERSELL"
_ERROR_NON_POSITIVE_QTY: Final[str] = "POS_NON_POSITIVE_QTY"

_SIDE_BUY: Final[str] = "BUY"
_SIDE_SELL: Final[str] = "SELL"

# Executed-trade columns required to assemble a position row.
TRADE_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "model_name",
    "model_version",
    "optimizer",
    "policy",
    "side",
    "executed_quantity",
    "executed_price",
    "fees",
    "execution_time",
)


@runtime_checkable
class PositionEngine(Protocol):
    """Structural contract for converting trade frames into position frames.

    Implementations own position accounting semantics. Pipeline orchestration
    delegates exclusively through this contract. Implementations must return a
    new DataFrame and must not mutate the input trade frame.
    """

    def build(self, trades: pl.DataFrame, *, manager: str) -> pl.DataFrame:
        """Convert a canonical executed-trade DataFrame into a position DataFrame.

        Args:
            trades: Canonical executed-trade dataset. Must not be mutated.
            manager: Order manager identity preserved onto every position row.

        Returns:
            A new DataFrame containing the columns required by the merged
            position schema contract.
        """
        ...


@dataclass(slots=True)
class _OpenPosition:
    """Mutable in-memory state for one open average-cost position."""

    position_id: str
    symbol: str
    timeframe: str
    quantity: float
    average_entry_price: float
    market_price: float
    realized_pnl: float
    fees_paid: float
    opened_at: datetime
    updated_at: datetime
    model_name: str
    model_version: str
    optimizer: str
    policy: str
    manager: str


class AverageCostPositionEngine:
    """Build long-only positions with average-cost realized PnL accounting.

    Rules:
        - Trades are consumed in ascending ``execution_time`` order
        - ``BUY`` opens or increases a long position and recalculates the
          weighted average entry price
        - ``SELL`` reduces quantity and realizes PnL against average cost
        - Positions close when quantity reaches zero
        - No leverage, shorts, funding, margin, borrowing, or liquidation
        - ``market_price`` tracks the latest executed trade price
        - ``unrealized_pnl = (market_price - average_entry_price) * quantity``
        - Opening-trade metadata is preserved for the position lifecycle

    Notes:
        Implementations must not mutate the caller-supplied DataFrame.
    """

    __slots__ = ()

    def build(self, trades: pl.DataFrame, *, manager: str) -> pl.DataFrame:
        """Convert executed trades into finalized position rows.

        Args:
            trades: Canonical executed-trade dataset. Must not be mutated.
            manager: Order manager identity stamped onto every position row.

        Returns:
            A new DataFrame matching ``MERGED_POSITION_SCHEMA``.

        Raises:
            PositionValidationError: If ``trades`` fails structural validation,
                ``manager`` is blank, required columns are missing, a trade
                side is unsupported, quantity is non-positive, a sell would
                create a short, or a sell exceeds open quantity.
        """
        frame = validate_trade_frame(trades)
        validated_manager = _require_manager(manager)
        _require_trade_columns(frame)
        return _build_position_frame(frame, manager=validated_manager)


def validate_trade_frame(frame: object) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty Polars DataFrame.

    Args:
        frame: Candidate executed-trade dataset passed to an engine.

    Returns:
        ``frame`` as a DataFrame after structural checks.

    Raises:
        PositionValidationError: If ``frame`` is not a Polars DataFrame or
            contains no rows.
    """
    if not isinstance(frame, pl.DataFrame):
        raise PositionValidationError(
            "frame must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={"actual_type": type(frame).__name__},
        )
    if frame.height == 0:
        raise PositionValidationError(
            "frame must contain at least one row",
            error_code=_ERROR_FRAME_EMPTY,
            details={"rows": frame.height},
        )
    return frame


def _build_position_frame(frame: pl.DataFrame, *, manager: str) -> pl.DataFrame:
    """Assemble canonical position rows from trades ordered by execution time.

    Args:
        frame: Validated executed-trade DataFrame.
        manager: Order manager identity for lineage.

    Returns:
        A new DataFrame ordered and cast to ``MERGED_POSITION_SCHEMA``.
    """
    ordered = frame.sort("execution_time", maintain_order=True)
    open_positions: dict[tuple[str, str], _OpenPosition] = {}
    closed_rows: list[dict[str, object]] = []
    next_position_index = 1

    for row in ordered.iter_rows(named=True):
        symbol = str(row["symbol"])
        timeframe = str(row["timeframe"])
        key = (symbol, timeframe)
        side = str(row["side"])
        quantity = float(row["executed_quantity"])
        price = float(row["executed_price"])
        fees = float(row["fees"])
        execution_time = row["execution_time"]
        if not isinstance(execution_time, datetime):
            raise PositionValidationError(
                "execution_time must be a timezone-aware datetime",
                error_code=_ERROR_MISSING_COLUMNS,
                details={"execution_time_type": type(execution_time).__name__},
            )
        if quantity <= 0.0:
            raise PositionValidationError(
                "executed_quantity must be positive",
                error_code=_ERROR_NON_POSITIVE_QTY,
                details={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "executed_quantity": quantity,
                },
            )

        if side == _SIDE_BUY:
            next_position_index = _apply_buy(
                open_positions=open_positions,
                key=key,
                quantity=quantity,
                price=price,
                fees=fees,
                execution_time=execution_time,
                row=row,
                manager=manager,
                next_position_index=next_position_index,
            )
        elif side == _SIDE_SELL:
            _apply_sell(
                open_positions=open_positions,
                closed_rows=closed_rows,
                key=key,
                quantity=quantity,
                price=price,
                fees=fees,
                execution_time=execution_time,
            )
        else:
            raise PositionValidationError(
                f"unsupported trade side: {side}",
                error_code=_ERROR_INVALID_SIDE,
                details={"side": side, "allowed_sides": (_SIDE_BUY, _SIDE_SELL)},
            )

    rows = list(closed_rows)
    for position in open_positions.values():
        rows.append(_open_position_to_row(position))

    if not rows:
        raise PositionValidationError(
            "frame must contain at least one row",
            error_code=_ERROR_FRAME_EMPTY,
            details={"rows": 0},
        )

    assembled = pl.DataFrame(rows)
    ordered_rows = assembled.sort("opened_at", maintain_order=True)
    return ordered_rows.select(list(CANONICAL_COLUMN_ORDER)).cast(MERGED_POSITION_SCHEMA)


def _apply_buy(
    *,
    open_positions: dict[tuple[str, str], _OpenPosition],
    key: tuple[str, str],
    quantity: float,
    price: float,
    fees: float,
    execution_time: datetime,
    row: dict[str, object],
    manager: str,
    next_position_index: int,
) -> int:
    """Apply a BUY trade to open-position state.

    Returns:
        The next unused position index.
    """
    existing = open_positions.get(key)
    if existing is None:
        position_id = _format_position_id(next_position_index)
        open_positions[key] = _OpenPosition(
            position_id=position_id,
            symbol=key[0],
            timeframe=key[1],
            quantity=quantity,
            average_entry_price=price,
            market_price=price,
            realized_pnl=0.0,
            fees_paid=fees,
            opened_at=execution_time,
            updated_at=execution_time,
            model_name=str(row["model_name"]),
            model_version=str(row["model_version"]),
            optimizer=str(row["optimizer"]),
            policy=str(row["policy"]),
            manager=manager,
        )
        return next_position_index + 1

    total_quantity = existing.quantity + quantity
    existing.average_entry_price = (
        (existing.average_entry_price * existing.quantity) + (price * quantity)
    ) / total_quantity
    existing.quantity = total_quantity
    existing.market_price = price
    existing.fees_paid += fees
    existing.updated_at = execution_time
    return next_position_index


def _apply_sell(
    *,
    open_positions: dict[tuple[str, str], _OpenPosition],
    closed_rows: list[dict[str, object]],
    key: tuple[str, str],
    quantity: float,
    price: float,
    fees: float,
    execution_time: datetime,
) -> None:
    """Apply a SELL trade to open-position state using average-cost accounting."""
    existing = open_positions.get(key)
    if existing is None:
        raise PositionValidationError(
            "SELL cannot open a short position",
            error_code=_ERROR_NO_SHORTS,
            details={"symbol": key[0], "timeframe": key[1], "executed_quantity": quantity},
        )
    if quantity > existing.quantity:
        raise PositionValidationError(
            "SELL quantity exceeds open position quantity",
            error_code=_ERROR_OVERSELL,
            details={
                "symbol": key[0],
                "timeframe": key[1],
                "executed_quantity": quantity,
                "open_quantity": existing.quantity,
            },
        )

    realized = (price - existing.average_entry_price) * quantity
    existing.realized_pnl += realized
    existing.quantity -= quantity
    existing.market_price = price
    existing.fees_paid += fees
    existing.updated_at = execution_time

    if existing.quantity == 0.0:
        closed_rows.append(_closed_position_to_row(existing, closed_at=execution_time))
        del open_positions[key]


def _open_position_to_row(position: _OpenPosition) -> dict[str, object]:
    """Serialize an open position into a canonical row mapping."""
    unrealized = (position.market_price - position.average_entry_price) * position.quantity
    return {
        "symbol": position.symbol,
        "timeframe": position.timeframe,
        "position_id": position.position_id,
        "side": PositionSide.LONG.value,
        "status": PositionStatus.OPEN.value,
        "quantity": position.quantity,
        "average_entry_price": position.average_entry_price,
        "market_price": position.market_price,
        "realized_pnl": position.realized_pnl,
        "unrealized_pnl": unrealized,
        "fees_paid": position.fees_paid,
        "opened_at": position.opened_at,
        "updated_at": position.updated_at,
        "closed_at": None,
        "model_name": position.model_name,
        "model_version": position.model_version,
        "optimizer": position.optimizer,
        "policy": position.policy,
        "manager": position.manager,
    }


def _closed_position_to_row(
    position: _OpenPosition,
    *,
    closed_at: datetime,
) -> dict[str, object]:
    """Serialize a fully closed position into a canonical row mapping."""
    return {
        "symbol": position.symbol,
        "timeframe": position.timeframe,
        "position_id": position.position_id,
        "side": PositionSide.LONG.value,
        "status": PositionStatus.CLOSED.value,
        "quantity": 0.0,
        "average_entry_price": position.average_entry_price,
        "market_price": position.market_price,
        "realized_pnl": position.realized_pnl,
        "unrealized_pnl": 0.0,
        "fees_paid": position.fees_paid,
        "opened_at": position.opened_at,
        "updated_at": position.updated_at,
        "closed_at": closed_at,
        "model_name": position.model_name,
        "model_version": position.model_version,
        "optimizer": position.optimizer,
        "policy": position.policy,
        "manager": position.manager,
    }


def _format_position_id(index: int) -> str:
    """Return a deterministic zero-padded position identifier."""
    return f"pos-{index:08d}"


def _require_manager(manager: object) -> str:
    """Validate and return a non-blank manager identity."""
    if not isinstance(manager, str) or manager.strip() == "":
        raise PositionValidationError(
            "manager must be a non-blank string",
            error_code=_ERROR_MANAGER_BLANK,
            details={"manager": manager},
        )
    return manager


def _require_trade_columns(frame: pl.DataFrame) -> None:
    """Raise when any required executed-trade column is missing."""
    missing = [column for column in TRADE_INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise PositionValidationError(
            "trade frame is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "missing_columns": tuple(missing),
                "required_columns": TRADE_INPUT_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )
