"""CQROS Backtesting Engine contracts and simple implementation.

Purpose:
    Reconstruct an immutable historical performance ledger from canonical
    accounting, position, and exit-engine datasets. The engine never trades,
    executes orders, mutates positions, or updates accounting.

Responsibilities:
    - Define ``BacktestingEngine`` as the shared performance-ledger contract
    - Provide ``SimpleBacktestingEngine`` for deterministic equity-curve
      reconstruction
    - Validate accounting, position, and exit-engine DataFrame structure
    - Remain free of persistence, verification, CLI, broker APIs, order
      submission, and portfolio construction

Dependencies:
    ``polars``, ``cqros.backtesting.exceptions``, and
    ``cqros.backtesting.schema``.

Public API:
    ``BacktestingEngine``, ``SimpleBacktestingEngine``,
    ``ACCOUNTING_INPUT_COLUMNS``, ``POSITION_INPUT_COLUMNS``,
    ``EXIT_ENGINE_INPUT_COLUMNS``, ``validate_accounting_frame``,
    ``validate_position_frame``, ``validate_exit_engine_frame``
"""

from __future__ import annotations

from datetime import datetime
from typing import Final, Protocol, runtime_checkable

import polars as pl

from cqros.backtesting.exceptions import BacktestingValidationError
from cqros.backtesting.schema import (
    CANONICAL_COLUMN_ORDER,
    MERGED_BACKTESTING_SCHEMA,
    BacktestingStatus,
)

__all__ = [
    "ACCOUNTING_INPUT_COLUMNS",
    "EXIT_ENGINE_INPUT_COLUMNS",
    "POSITION_INPUT_COLUMNS",
    "BacktestingEngine",
    "SimpleBacktestingEngine",
    "validate_accounting_frame",
    "validate_exit_engine_frame",
    "validate_position_frame",
]

_ERROR_FRAME_TYPE: Final[str] = "BT_FRAME_TYPE"
_ERROR_FRAME_EMPTY: Final[str] = "BT_FRAME_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "BT_MISSING_COLUMNS"
_ERROR_MANAGER_BLANK: Final[str] = "BT_MANAGER_BLANK"
_ERROR_NO_TIMESTAMPS: Final[str] = "BT_NO_TIMESTAMPS"
_ERROR_SYMBOL_BLANK: Final[str] = "BT_SYMBOL_BLANK"
_ERROR_TIMEFRAME_BLANK: Final[str] = "BT_TIMEFRAME_BLANK"

_POSITION_STATUS_CLOSED: Final[str] = "CLOSED"
_EXIT_ACTION_FULL: Final[str] = "FULL_EXIT"

# Accounting columns required to assemble a performance ledger row.
ACCOUNTING_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
    "cash",
    "position_value",
    "realized_pnl",
    "unrealized_pnl",
    "position_id",
    "position_status",
)

# Position columns required for completed-trade statistics.
POSITION_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "position_id",
    "status",
    "realized_pnl",
    "opened_at",
    "updated_at",
    "closed_at",
)

# Exit-engine columns required for evaluation-timestamp alignment.
EXIT_ENGINE_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
    "position_id",
    "exit_action",
)


@runtime_checkable
class BacktestingEngine(Protocol):
    """Structural contract for converting inputs into a performance ledger.

    Implementations own performance-reconstruction semantics. Pipeline
    orchestration delegates exclusively through this contract.
    Implementations must return a new DataFrame and must not mutate the
    input frames.
    """

    def build(
        self,
        accounting: pl.DataFrame,
        positions: pl.DataFrame,
        exit_engine: pl.DataFrame,
        *,
        manager: str,
    ) -> pl.DataFrame:
        """Convert canonical inputs into a backtesting performance DataFrame.

        Args:
            accounting: Canonical accounting dataset. Must not be mutated.
            positions: Canonical position dataset. Must not be mutated.
            exit_engine: Canonical exit-engine dataset. Must not be mutated.
            manager: Order manager identity preserved onto every ledger row.

        Returns:
            A new DataFrame containing the columns required by the merged
            backtesting schema contract.
        """
        ...


class SimpleBacktestingEngine:
    """Reconstruct a deterministic historical performance ledger.

    Rules:
        - One output row per completed evaluation timestamp
          (``symbol``, ``timeframe``, ``open_time``)
        - Evaluation timestamps are the intersection of accounting and
          exit-engine ``open_time`` values
        - ``equity = cash + unrealized_pnl``
        - ``peak_equity`` is the running maximum equity
        - ``drawdown = (peak_equity - equity) / peak_equity`` when peak is
          positive, otherwise ``0``
        - ``realized_pnl`` is the portfolio sum of accounting realized PnL
          at the evaluation timestamp (already cumulative per position)
        - ``unrealized_pnl`` is the latest portfolio unrealized PnL
        - ``total_pnl = realized_pnl + unrealized_pnl``
        - ``daily_return = equity_t / equity_(t-1) - 1`` (``0`` on first row)
        - ``cumulative_return = equity / initial_equity - 1``
        - Completed trades are CLOSED positions whose completion time is at
          or before the evaluation timestamp, plus FULL_EXIT recommendations
          at or before the timestamp whose position is not already counted
        - Winning trades have realized PnL ``> 0``; losing trades ``< 0``
        - ``win_rate = winning_trades / trade_count`` when trades exist,
          otherwise ``0``
        - ``profit_factor = gross_profit / gross_loss``; ``NULL`` when there
          are no losses
        - ``sharpe_stub`` and ``sortino_stub`` are always ``NULL``
        - ``max_drawdown`` is the running maximum drawdown
        - ``status`` is ``ACTIVE`` until the final row, then ``FINISHED``

    Notes:
        Implementations must not mutate the caller-supplied DataFrames.
    """

    __slots__ = ()

    def build(
        self,
        accounting: pl.DataFrame,
        positions: pl.DataFrame,
        exit_engine: pl.DataFrame,
        *,
        manager: str,
    ) -> pl.DataFrame:
        """Convert inputs into a finalized performance ledger.

        Args:
            accounting: Canonical accounting dataset. Must not be mutated.
            positions: Canonical position dataset. Must not be mutated.
            exit_engine: Canonical exit-engine dataset. Must not be mutated.
            manager: Order manager identity stamped onto every ledger row.

        Returns:
            A new DataFrame matching ``MERGED_BACKTESTING_SCHEMA``.

        Raises:
            BacktestingValidationError: If any input fails structural
                validation, ``manager`` is blank, required columns are
                missing, or no evaluation timestamps can be formed.
        """
        accounting_frame = validate_accounting_frame(accounting)
        position_frame = validate_position_frame(positions)
        exit_frame = validate_exit_engine_frame(exit_engine)
        validated_manager = _require_manager(manager)
        _require_columns(accounting_frame, ACCOUNTING_INPUT_COLUMNS, "accounting")
        _require_columns(position_frame, POSITION_INPUT_COLUMNS, "positions")
        _require_columns(exit_frame, EXIT_ENGINE_INPUT_COLUMNS, "exit_engine")
        return _build_performance_ledger(
            accounting_frame,
            position_frame,
            exit_frame,
            manager=validated_manager,
        )


def validate_accounting_frame(frame: object) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty Polars DataFrame.

    Args:
        frame: Candidate accounting dataset passed to an engine.

    Returns:
        ``frame`` as a DataFrame after structural checks.

    Raises:
        BacktestingValidationError: If ``frame`` is not a Polars DataFrame or
            contains no rows.
    """
    return _validate_non_empty_frame(frame, dataset="accounting")


def validate_position_frame(frame: object) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty Polars DataFrame.

    Args:
        frame: Candidate position dataset passed to an engine.

    Returns:
        ``frame`` as a DataFrame after structural checks.

    Raises:
        BacktestingValidationError: If ``frame`` is not a Polars DataFrame or
            contains no rows.
    """
    return _validate_non_empty_frame(frame, dataset="positions")


def validate_exit_engine_frame(frame: object) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty Polars DataFrame.

    Args:
        frame: Candidate exit-engine dataset passed to an engine.

    Returns:
        ``frame`` as a DataFrame after structural checks.

    Raises:
        BacktestingValidationError: If ``frame`` is not a Polars DataFrame or
            contains no rows.
    """
    return _validate_non_empty_frame(frame, dataset="exit_engine")


def _validate_non_empty_frame(frame: object, *, dataset: str) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty Polars DataFrame."""
    if not isinstance(frame, pl.DataFrame):
        raise BacktestingValidationError(
            f"{dataset} frame must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={"dataset": dataset, "actual_type": type(frame).__name__},
        )
    if frame.height == 0:
        raise BacktestingValidationError(
            f"{dataset} frame must contain at least one row",
            error_code=_ERROR_FRAME_EMPTY,
            details={"dataset": dataset, "rows": frame.height},
        )
    return frame


def _build_performance_ledger(
    accounting: pl.DataFrame,
    positions: pl.DataFrame,
    exit_engine: pl.DataFrame,
    *,
    manager: str,
) -> pl.DataFrame:
    """Assemble canonical performance rows from validated inputs."""
    timestamps = _evaluation_timestamps(accounting, exit_engine)
    if len(timestamps) == 0:
        raise BacktestingValidationError(
            "no completed evaluation timestamps found",
            error_code=_ERROR_NO_TIMESTAMPS,
            details={
                "accounting_rows": accounting.height,
                "exit_engine_rows": exit_engine.height,
            },
        )

    symbol = _require_identity_value(accounting, "symbol")
    timeframe = _require_identity_value(accounting, "timeframe")
    completed_trades = _completed_trades(positions, exit_engine, accounting)

    equity_values: list[float] = []
    cash_values: list[float] = []
    position_values: list[float] = []
    realized_values: list[float] = []
    unrealized_values: list[float] = []
    total_pnl_values: list[float] = []
    drawdown_values: list[float] = []
    peak_values: list[float] = []
    daily_return_values: list[float] = []
    cumulative_return_values: list[float] = []
    trade_counts: list[int] = []
    winning_counts: list[int] = []
    losing_counts: list[int] = []
    win_rates: list[float] = []
    profit_factors: list[float | None] = []
    max_drawdowns: list[float] = []
    statuses: list[str] = []

    peak_equity = 0.0
    max_drawdown = 0.0
    previous_equity: float | None = None
    initial_equity: float | None = None

    for index, open_time in enumerate(timestamps):
        snapshot = accounting.filter(pl.col("open_time") == open_time)
        # Cash is portfolio-level and identical across accounting rows.
        cash = float(snapshot["cash"][0])
        position_value = float(snapshot["position_value"].sum())
        realized = float(snapshot["realized_pnl"].sum())
        unrealized = float(snapshot["unrealized_pnl"].sum())
        equity = cash + unrealized
        total_pnl = realized + unrealized

        if initial_equity is None:
            initial_equity = equity
        peak_equity = max(peak_equity, equity)
        if peak_equity > 0.0:
            drawdown = (peak_equity - equity) / peak_equity
        else:
            drawdown = 0.0
        max_drawdown = max(max_drawdown, drawdown)

        if previous_equity is None or previous_equity == 0.0:
            daily_return = 0.0
        else:
            daily_return = equity / previous_equity - 1.0

        if initial_equity == 0.0:
            cumulative_return = 0.0
        else:
            cumulative_return = equity / initial_equity - 1.0

        trade_stats = _trade_stats_at(completed_trades, open_time)
        is_final = index == len(timestamps) - 1
        status = BacktestingStatus.FINISHED.value if is_final else BacktestingStatus.ACTIVE.value

        equity_values.append(equity)
        cash_values.append(cash)
        position_values.append(position_value)
        realized_values.append(realized)
        unrealized_values.append(unrealized)
        total_pnl_values.append(total_pnl)
        drawdown_values.append(drawdown)
        peak_values.append(peak_equity)
        daily_return_values.append(daily_return)
        cumulative_return_values.append(cumulative_return)
        trade_counts.append(trade_stats.trade_count)
        winning_counts.append(trade_stats.winning_trades)
        losing_counts.append(trade_stats.losing_trades)
        win_rates.append(trade_stats.win_rate)
        profit_factors.append(trade_stats.profit_factor)
        max_drawdowns.append(max_drawdown)
        statuses.append(status)
        previous_equity = equity

    assembled = pl.DataFrame(
        {
            "symbol": [symbol] * len(timestamps),
            "timeframe": [timeframe] * len(timestamps),
            "open_time": timestamps,
            "manager": [manager] * len(timestamps),
            "equity": equity_values,
            "cash": cash_values,
            "position_value": position_values,
            "realized_pnl": realized_values,
            "unrealized_pnl": unrealized_values,
            "total_pnl": total_pnl_values,
            "drawdown": drawdown_values,
            "peak_equity": peak_values,
            "daily_return": daily_return_values,
            "cumulative_return": cumulative_return_values,
            "trade_count": trade_counts,
            "winning_trades": winning_counts,
            "losing_trades": losing_counts,
            "win_rate": win_rates,
            "profit_factor": profit_factors,
            "sharpe_stub": [None] * len(timestamps),
            "sortino_stub": [None] * len(timestamps),
            "max_drawdown": max_drawdowns,
            "status": statuses,
        }
    )
    ordered = assembled.sort("open_time", maintain_order=True)
    return ordered.select(list(CANONICAL_COLUMN_ORDER)).cast(MERGED_BACKTESTING_SCHEMA)


class _TradeRecord:
    """Immutable completed-trade record used for running trade statistics."""

    __slots__ = ("completed_at", "position_id", "realized_pnl")

    def __init__(
        self,
        *,
        position_id: str,
        completed_at: datetime,
        realized_pnl: float,
    ) -> None:
        self.position_id = position_id
        self.completed_at = completed_at
        self.realized_pnl = realized_pnl


class _TradeStats:
    """Immutable trade-statistic snapshot at one evaluation timestamp."""

    __slots__ = (
        "losing_trades",
        "profit_factor",
        "trade_count",
        "win_rate",
        "winning_trades",
    )

    def __init__(
        self,
        *,
        trade_count: int,
        winning_trades: int,
        losing_trades: int,
        win_rate: float,
        profit_factor: float | None,
    ) -> None:
        self.trade_count = trade_count
        self.winning_trades = winning_trades
        self.losing_trades = losing_trades
        self.win_rate = win_rate
        self.profit_factor = profit_factor


def _evaluation_timestamps(
    accounting: pl.DataFrame,
    exit_engine: pl.DataFrame,
) -> list[datetime]:
    """Return sorted open_time values present in both accounting and exit data."""
    accounting_times = set(accounting["open_time"].to_list())
    exit_times = set(exit_engine["open_time"].to_list())
    shared = accounting_times & exit_times
    return sorted(shared)


def _completed_trades(
    positions: pl.DataFrame,
    exit_engine: pl.DataFrame,
    accounting: pl.DataFrame,
) -> tuple[_TradeRecord, ...]:
    """Build completed-trade records from CLOSED positions and FULL_EXIT rows."""
    records: dict[str, _TradeRecord] = {}

    closed = positions.filter(pl.col("status") == _POSITION_STATUS_CLOSED)
    for row in closed.iter_rows(named=True):
        position_id = str(row["position_id"])
        completed_at = row["closed_at"]
        if completed_at is None:
            completed_at = row["updated_at"]
        if completed_at is None:
            completed_at = row["opened_at"]
        records[position_id] = _TradeRecord(
            position_id=position_id,
            completed_at=completed_at,
            realized_pnl=float(row["realized_pnl"]),
        )

    full_exits = exit_engine.filter(pl.col("exit_action") == _EXIT_ACTION_FULL)
    for row in full_exits.iter_rows(named=True):
        position_id = str(row["position_id"])
        if position_id in records:
            continue
        realized_rows = accounting.filter(pl.col("position_id") == position_id)
        if realized_rows.height == 0:
            realized_pnl = 0.0
        else:
            latest = realized_rows.sort("open_time", descending=True)
            realized_pnl = float(latest["realized_pnl"][0])
        records[position_id] = _TradeRecord(
            position_id=position_id,
            completed_at=row["open_time"],
            realized_pnl=realized_pnl,
        )

    return tuple(sorted(records.values(), key=lambda item: (item.completed_at, item.position_id)))


def _trade_stats_at(
    trades: tuple[_TradeRecord, ...],
    open_time: datetime,
) -> _TradeStats:
    """Compute running trade statistics for trades completed by ``open_time``."""
    realized: list[float] = [
        trade.realized_pnl for trade in trades if trade.completed_at <= open_time
    ]
    trade_count = len(realized)
    winning = sum(1 for value in realized if value > 0.0)
    losing = sum(1 for value in realized if value < 0.0)
    win_rate = (winning / trade_count) if trade_count > 0 else 0.0
    gross_profit = sum(value for value in realized if value > 0.0)
    gross_loss = abs(sum(value for value in realized if value < 0.0))
    if gross_loss == 0.0:
        profit_factor: float | None = None
    else:
        profit_factor = gross_profit / gross_loss
    return _TradeStats(
        trade_count=trade_count,
        winning_trades=winning,
        losing_trades=losing,
        win_rate=win_rate,
        profit_factor=profit_factor,
    )


def _require_manager(manager: object) -> str:
    """Validate and return a non-blank manager identity."""
    if not isinstance(manager, str) or manager.strip() == "":
        raise BacktestingValidationError(
            "manager must be a non-blank string",
            error_code=_ERROR_MANAGER_BLANK,
            details={"manager": manager},
        )
    return manager


def _require_columns(
    frame: pl.DataFrame,
    required: tuple[str, ...],
    dataset: str,
) -> None:
    """Raise when any required column is missing from ``frame``."""
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise BacktestingValidationError(
            f"{dataset} frame is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "dataset": dataset,
                "missing_columns": tuple(missing),
                "required_columns": required,
                "available_columns": tuple(frame.columns),
            },
        )


def _require_identity_value(frame: pl.DataFrame, column: str) -> str:
    """Return the first non-blank identity value from ``column``."""
    values = frame[column].to_list()
    if len(values) == 0:
        error_code = _ERROR_SYMBOL_BLANK if column == "symbol" else _ERROR_TIMEFRAME_BLANK
        raise BacktestingValidationError(
            f"{column} must be a non-blank string",
            error_code=error_code,
            details={"column": column},
        )
    value = str(values[0])
    if value.strip() == "":
        error_code = _ERROR_SYMBOL_BLANK if column == "symbol" else _ERROR_TIMEFRAME_BLANK
        raise BacktestingValidationError(
            f"{column} must be a non-blank string",
            error_code=error_code,
            details={"column": column, "value": value},
        )
    return value
