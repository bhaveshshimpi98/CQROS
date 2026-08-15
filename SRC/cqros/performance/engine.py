"""CQROS Performance Engine contracts and simple implementation.

Purpose:
    Convert a canonical backtesting performance ledger into a deterministic
    performance-metrics DataFrame conforming to ``PERFORMANCE_SCHEMA``.

Responsibilities:
    - Define ``PerformanceEngine`` as the shared metrics contract
    - Provide ``SimplePerformanceEngine`` for equity-curve and trade metrics
    - Validate backtesting DataFrame structure and finite numeric outputs
    - Remain free of persistence, verification, CLI, storage, and file I/O

Dependencies:
    ``math``, ``polars``, ``cqros.core.constants``,
    ``cqros.performance.exceptions``, and ``cqros.performance.schema``.

Public API:
    ``PerformanceEngine``, ``SimplePerformanceEngine``,
    ``BACKTESTING_INPUT_COLUMNS``, ``validate_backtesting_frame``
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Final, Protocol, runtime_checkable

import polars as pl

from cqros.core.constants import DAYS_PER_YEAR, SECONDS_PER_DAY
from cqros.performance.exceptions import PerformanceValidationError
from cqros.performance.schema import (
    CANONICAL_COLUMN_ORDER,
    PERFORMANCE_SCHEMA,
    PerformanceStatus,
)

__all__ = [
    "BACKTESTING_INPUT_COLUMNS",
    "PerformanceEngine",
    "SimplePerformanceEngine",
    "validate_backtesting_frame",
]

_ERROR_FRAME_TYPE: Final[str] = "PERF_FRAME_TYPE"
_ERROR_FRAME_EMPTY: Final[str] = "PERF_FRAME_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "PERF_MISSING_COLUMNS"
_ERROR_NON_FINITE: Final[str] = "PERF_NON_FINITE"
_ERROR_OPEN_TIME_ORDER: Final[str] = "PERF_OPEN_TIME_ORDER"
_ERROR_TRADE_COUNT: Final[str] = "PERF_TRADE_COUNT"

_RISK_FREE_RATE: Final[float] = 0.0
_SECONDS_PER_YEAR: Final[float] = float(SECONDS_PER_DAY * DAYS_PER_YEAR)

# Backtesting columns required to assemble a performance-metrics row.
BACKTESTING_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
    "manager",
    "equity",
    "daily_return",
    "drawdown",
    "realized_pnl",
    "trade_count",
)


@runtime_checkable
class PerformanceEngine(Protocol):
    """Structural contract for converting backtesting ledgers into metrics.

    Implementations own performance-metric semantics. Pipeline orchestration
    delegates exclusively through this contract. Implementations must return
    a new DataFrame and must not mutate the input frame.
    """

    def build(self, backtesting: pl.DataFrame) -> pl.DataFrame:
        """Convert a backtesting ledger into a performance-metrics DataFrame.

        Args:
            backtesting: Canonical backtesting dataset. Must not be mutated.

        Returns:
            A new DataFrame containing the columns required by
            ``PERFORMANCE_SCHEMA``.
        """
        ...


class SimplePerformanceEngine:
    """Compute deterministic performance metrics from a backtesting ledger.

    Rules:
        - One output row per input evaluation timestamp
        - Identity columns (``symbol``, ``timeframe``, ``open_time``,
          ``manager``) are preserved from the input row
        - Metrics at row ``i`` use only information from rows ``0..i``
        - ``starting_equity`` is equity at the first timestamp
        - ``ending_equity`` is equity at the current timestamp
        - ``net_profit = ending_equity - starting_equity``
        - ``total_return = ending / starting - 1`` when starting ``> 0``,
          otherwise ``0``
        - ``cagr`` uses calendar years between first and current
          ``open_time`` with ``DAYS_PER_YEAR`` day-count
        - ``volatility`` is the sample standard deviation of
          ``daily_return`` values annualized by ``sqrt(periods_per_year)``
        - ``downside_volatility`` annualizes
          ``sqrt(mean(min(r, 0)^2))`` (MAR = 0)
        - ``max_drawdown`` is the running maximum of ``drawdown``
        - ``drawdown_duration`` is the longest consecutive streak of bars
          with ``drawdown > 0``
        - ``sharpe_ratio`` and ``sortino_ratio`` use a zero risk-free rate
        - ``calmar_ratio = cagr / max_drawdown`` when drawdown ``> 0``
        - Completed trades are inferred when ``trade_count`` increases;
          realized PnL deltas since the prior count change are attributed
          evenly across newly counted trades
        - ``profit_factor`` is ``NULL`` when ``gross_loss == 0``
        - ``status`` is ``ACTIVE`` until the final row, then ``FINISHED``

    Notes:
        Implementations must not mutate the caller-supplied DataFrame.
    """

    __slots__ = ()

    def build(self, backtesting: pl.DataFrame) -> pl.DataFrame:
        """Convert a backtesting ledger into finalized performance metrics.

        Args:
            backtesting: Canonical backtesting dataset. Must not be mutated.

        Returns:
            A new DataFrame matching ``PERFORMANCE_SCHEMA``.

        Raises:
            PerformanceValidationError: If the input fails structural
                validation, required columns are missing, timestamps are
                unsorted, or a numeric output is non-finite.
        """
        frame = validate_backtesting_frame(backtesting)
        _require_columns(frame, BACKTESTING_INPUT_COLUMNS, "backtesting")
        ordered = frame.sort("open_time", maintain_order=True)
        _require_sorted_open_times(ordered)
        return _build_performance_metrics(ordered)


def validate_backtesting_frame(frame: object) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty Polars DataFrame.

    Args:
        frame: Candidate backtesting dataset passed to an engine.

    Returns:
        ``frame`` as a DataFrame after structural checks.

    Raises:
        PerformanceValidationError: If ``frame`` is not a Polars DataFrame or
            contains no rows.
    """
    if not isinstance(frame, pl.DataFrame):
        raise PerformanceValidationError(
            "backtesting frame must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={"dataset": "backtesting", "actual_type": type(frame).__name__},
        )
    if frame.height == 0:
        raise PerformanceValidationError(
            "backtesting frame must contain at least one row",
            error_code=_ERROR_FRAME_EMPTY,
            details={"dataset": "backtesting", "rows": frame.height},
        )
    return frame


class _TradeRecord:
    """Immutable completed-trade record inferred from the backtesting ledger."""

    __slots__ = ("completed_at", "realized_pnl")

    def __init__(self, *, completed_at: datetime, realized_pnl: float) -> None:
        self.completed_at = completed_at
        self.realized_pnl = realized_pnl


class _MetricSnapshot:
    """Immutable metric values for one evaluation timestamp."""

    __slots__ = (
        "average_loss",
        "average_win",
        "calmar_ratio",
        "cagr",
        "downside_volatility",
        "drawdown_duration",
        "ending_equity",
        "expectancy",
        "first_trade_time",
        "gross_loss",
        "gross_profit",
        "last_trade_time",
        "losing_trades",
        "max_drawdown",
        "net_profit",
        "profit_factor",
        "sharpe_ratio",
        "sortino_ratio",
        "starting_equity",
        "total_return",
        "total_trades",
        "volatility",
        "win_rate",
        "winning_trades",
    )

    def __init__(
        self,
        *,
        total_return: float,
        cagr: float,
        volatility: float,
        downside_volatility: float,
        max_drawdown: float,
        drawdown_duration: int,
        sharpe_ratio: float | None,
        sortino_ratio: float | None,
        calmar_ratio: float | None,
        total_trades: int,
        winning_trades: int,
        losing_trades: int,
        win_rate: float,
        average_win: float | None,
        average_loss: float | None,
        profit_factor: float | None,
        expectancy: float,
        starting_equity: float,
        ending_equity: float,
        net_profit: float,
        gross_profit: float,
        gross_loss: float,
        first_trade_time: datetime | None,
        last_trade_time: datetime | None,
    ) -> None:
        self.total_return = total_return
        self.cagr = cagr
        self.volatility = volatility
        self.downside_volatility = downside_volatility
        self.max_drawdown = max_drawdown
        self.drawdown_duration = drawdown_duration
        self.sharpe_ratio = sharpe_ratio
        self.sortino_ratio = sortino_ratio
        self.calmar_ratio = calmar_ratio
        self.total_trades = total_trades
        self.winning_trades = winning_trades
        self.losing_trades = losing_trades
        self.win_rate = win_rate
        self.average_win = average_win
        self.average_loss = average_loss
        self.profit_factor = profit_factor
        self.expectancy = expectancy
        self.starting_equity = starting_equity
        self.ending_equity = ending_equity
        self.net_profit = net_profit
        self.gross_profit = gross_profit
        self.gross_loss = gross_loss
        self.first_trade_time = first_trade_time
        self.last_trade_time = last_trade_time


def _build_performance_metrics(backtesting: pl.DataFrame) -> pl.DataFrame:
    """Assemble canonical performance-metric rows from a sorted ledger."""
    open_times = backtesting["open_time"].to_list()
    equities = [float(value) for value in backtesting["equity"].to_list()]
    daily_returns = [float(value) for value in backtesting["daily_return"].to_list()]
    drawdowns = [float(value) for value in backtesting["drawdown"].to_list()]
    realized_values = [float(value) for value in backtesting["realized_pnl"].to_list()]
    trade_counts = [int(value) for value in backtesting["trade_count"].to_list()]

    trades = _infer_completed_trades(
        open_times=open_times,
        realized_values=realized_values,
        trade_counts=trade_counts,
    )
    starting_equity = equities[0]
    row_count = len(open_times)

    total_returns: list[float] = []
    cagrs: list[float] = []
    volatilities: list[float] = []
    downside_volatilities: list[float] = []
    max_drawdowns: list[float] = []
    drawdown_durations: list[int] = []
    sharpe_ratios: list[float | None] = []
    sortino_ratios: list[float | None] = []
    calmar_ratios: list[float | None] = []
    total_trades_values: list[int] = []
    winning_trades_values: list[int] = []
    losing_trades_values: list[int] = []
    win_rates: list[float] = []
    average_wins: list[float | None] = []
    average_losses: list[float | None] = []
    profit_factors: list[float | None] = []
    expectancies: list[float] = []
    starting_equities: list[float] = []
    ending_equities: list[float] = []
    net_profits: list[float] = []
    gross_profits: list[float] = []
    gross_losses: list[float] = []
    first_trade_times: list[datetime | None] = []
    last_trade_times: list[datetime | None] = []
    statuses: list[str] = []

    for index in range(row_count):
        snapshot = _metrics_at(
            open_times=open_times[: index + 1],
            equities=equities[: index + 1],
            daily_returns=daily_returns[: index + 1],
            drawdowns=drawdowns[: index + 1],
            trades=trades,
            starting_equity=starting_equity,
            evaluation_time=open_times[index],
        )
        _validate_finite_snapshot(snapshot, row_index=index)

        is_final = index == row_count - 1
        status = PerformanceStatus.FINISHED.value if is_final else PerformanceStatus.ACTIVE.value

        total_returns.append(snapshot.total_return)
        cagrs.append(snapshot.cagr)
        volatilities.append(snapshot.volatility)
        downside_volatilities.append(snapshot.downside_volatility)
        max_drawdowns.append(snapshot.max_drawdown)
        drawdown_durations.append(snapshot.drawdown_duration)
        sharpe_ratios.append(snapshot.sharpe_ratio)
        sortino_ratios.append(snapshot.sortino_ratio)
        calmar_ratios.append(snapshot.calmar_ratio)
        total_trades_values.append(snapshot.total_trades)
        winning_trades_values.append(snapshot.winning_trades)
        losing_trades_values.append(snapshot.losing_trades)
        win_rates.append(snapshot.win_rate)
        average_wins.append(snapshot.average_win)
        average_losses.append(snapshot.average_loss)
        profit_factors.append(snapshot.profit_factor)
        expectancies.append(snapshot.expectancy)
        starting_equities.append(snapshot.starting_equity)
        ending_equities.append(snapshot.ending_equity)
        net_profits.append(snapshot.net_profit)
        gross_profits.append(snapshot.gross_profit)
        gross_losses.append(snapshot.gross_loss)
        first_trade_times.append(snapshot.first_trade_time)
        last_trade_times.append(snapshot.last_trade_time)
        statuses.append(status)

    assembled = pl.DataFrame(
        {
            "symbol": backtesting["symbol"].to_list(),
            "timeframe": backtesting["timeframe"].to_list(),
            "open_time": open_times,
            "manager": backtesting["manager"].to_list(),
            "total_return": total_returns,
            "cagr": cagrs,
            "volatility": volatilities,
            "downside_volatility": downside_volatilities,
            "max_drawdown": max_drawdowns,
            "drawdown_duration": drawdown_durations,
            "sharpe_ratio": sharpe_ratios,
            "sortino_ratio": sortino_ratios,
            "calmar_ratio": calmar_ratios,
            "total_trades": total_trades_values,
            "winning_trades": winning_trades_values,
            "losing_trades": losing_trades_values,
            "win_rate": win_rates,
            "average_win": average_wins,
            "average_loss": average_losses,
            "profit_factor": profit_factors,
            "expectancy": expectancies,
            "starting_equity": starting_equities,
            "ending_equity": ending_equities,
            "net_profit": net_profits,
            "gross_profit": gross_profits,
            "gross_loss": gross_losses,
            "first_trade_time": first_trade_times,
            "last_trade_time": last_trade_times,
            "status": statuses,
        }
    )
    return assembled.select(list(CANONICAL_COLUMN_ORDER)).cast(PERFORMANCE_SCHEMA)


def _metrics_at(
    *,
    open_times: list[datetime],
    equities: list[float],
    daily_returns: list[float],
    drawdowns: list[float],
    trades: tuple[_TradeRecord, ...],
    starting_equity: float,
    evaluation_time: datetime,
) -> _MetricSnapshot:
    """Compute performance metrics for one evaluation timestamp."""
    ending_equity = equities[-1]
    net_profit = ending_equity - starting_equity
    if starting_equity > 0.0:
        total_return = ending_equity / starting_equity - 1.0
    else:
        total_return = 0.0

    cagr = _cagr(
        starting_equity=starting_equity,
        ending_equity=ending_equity,
        start_time=open_times[0],
        end_time=evaluation_time,
    )
    periods_per_year = _periods_per_year(open_times)
    return_sample = daily_returns[1:] if len(daily_returns) > 1 else []
    sample_std = _sample_std(return_sample)
    downside_dev = _downside_deviation(return_sample)

    if sample_std is None:
        volatility = 0.0
        sharpe_ratio: float | None = None
    else:
        volatility = sample_std * math.sqrt(periods_per_year)
        if sample_std > 0.0:
            mean_return = sum(return_sample) / len(return_sample)
            sharpe_ratio = ((mean_return - _RISK_FREE_RATE) / sample_std) * math.sqrt(
                periods_per_year
            )
        else:
            sharpe_ratio = None

    if downside_dev is None:
        downside_volatility = 0.0
        sortino_ratio: float | None = None
    else:
        downside_volatility = downside_dev * math.sqrt(periods_per_year)
        if downside_dev > 0.0 and len(return_sample) > 0:
            mean_return = sum(return_sample) / len(return_sample)
            sortino_ratio = ((mean_return - _RISK_FREE_RATE) / downside_dev) * math.sqrt(
                periods_per_year
            )
        else:
            sortino_ratio = None

    max_drawdown = max(drawdowns) if drawdowns else 0.0
    drawdown_duration = _max_drawdown_duration(drawdowns)
    if max_drawdown > 0.0:
        calmar_ratio: float | None = cagr / max_drawdown
    else:
        calmar_ratio = None

    completed = tuple(trade for trade in trades if trade.completed_at <= evaluation_time)
    trade_stats = _trade_metrics(completed)

    return _MetricSnapshot(
        total_return=total_return,
        cagr=cagr,
        volatility=volatility,
        downside_volatility=downside_volatility,
        max_drawdown=max_drawdown,
        drawdown_duration=drawdown_duration,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        total_trades=trade_stats.total_trades,
        winning_trades=trade_stats.winning_trades,
        losing_trades=trade_stats.losing_trades,
        win_rate=trade_stats.win_rate,
        average_win=trade_stats.average_win,
        average_loss=trade_stats.average_loss,
        profit_factor=trade_stats.profit_factor,
        expectancy=trade_stats.expectancy,
        starting_equity=starting_equity,
        ending_equity=ending_equity,
        net_profit=net_profit,
        gross_profit=trade_stats.gross_profit,
        gross_loss=trade_stats.gross_loss,
        first_trade_time=trade_stats.first_trade_time,
        last_trade_time=trade_stats.last_trade_time,
    )


class _TradeMetrics:
    """Immutable trade-statistic snapshot at one evaluation timestamp."""

    __slots__ = (
        "average_loss",
        "average_win",
        "expectancy",
        "first_trade_time",
        "gross_loss",
        "gross_profit",
        "last_trade_time",
        "losing_trades",
        "profit_factor",
        "total_trades",
        "win_rate",
        "winning_trades",
    )

    def __init__(
        self,
        *,
        total_trades: int,
        winning_trades: int,
        losing_trades: int,
        win_rate: float,
        average_win: float | None,
        average_loss: float | None,
        profit_factor: float | None,
        expectancy: float,
        gross_profit: float,
        gross_loss: float,
        first_trade_time: datetime | None,
        last_trade_time: datetime | None,
    ) -> None:
        self.total_trades = total_trades
        self.winning_trades = winning_trades
        self.losing_trades = losing_trades
        self.win_rate = win_rate
        self.average_win = average_win
        self.average_loss = average_loss
        self.profit_factor = profit_factor
        self.expectancy = expectancy
        self.gross_profit = gross_profit
        self.gross_loss = gross_loss
        self.first_trade_time = first_trade_time
        self.last_trade_time = last_trade_time


def _trade_metrics(trades: tuple[_TradeRecord, ...]) -> _TradeMetrics:
    """Compute trade statistics from completed-trade records."""
    total_trades = len(trades)
    if total_trades == 0:
        return _TradeMetrics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            average_win=None,
            average_loss=None,
            profit_factor=None,
            expectancy=0.0,
            gross_profit=0.0,
            gross_loss=0.0,
            first_trade_time=None,
            last_trade_time=None,
        )

    pnls = [trade.realized_pnl for trade in trades]
    wins = [value for value in pnls if value > 0.0]
    losses = [value for value in pnls if value < 0.0]
    winning_trades = len(wins)
    losing_trades = len(losses)
    win_rate = winning_trades / total_trades
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    average_win = (gross_profit / winning_trades) if winning_trades > 0 else None
    average_loss = (sum(losses) / losing_trades) if losing_trades > 0 else None
    if gross_loss == 0.0:
        profit_factor: float | None = None
    else:
        profit_factor = gross_profit / gross_loss
    expectancy = sum(pnls) / total_trades
    return _TradeMetrics(
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        average_win=average_win,
        average_loss=average_loss,
        profit_factor=profit_factor,
        expectancy=expectancy,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        first_trade_time=trades[0].completed_at,
        last_trade_time=trades[-1].completed_at,
    )


def _infer_completed_trades(
    *,
    open_times: list[datetime],
    realized_values: list[float],
    trade_counts: list[int],
) -> tuple[_TradeRecord, ...]:
    """Infer completed trades from running trade-count and realized PnL."""
    records: list[_TradeRecord] = []
    previous_count = 0
    realized_at_previous_count = 0.0

    for open_time, realized, trade_count in zip(
        open_times,
        realized_values,
        trade_counts,
        strict=True,
    ):
        if trade_count < previous_count:
            raise PerformanceValidationError(
                "trade_count must be non-decreasing across open_time",
                error_code=_ERROR_TRADE_COUNT,
                details={
                    "open_time": open_time,
                    "trade_count": trade_count,
                    "previous_count": previous_count,
                },
            )
        if trade_count > previous_count:
            new_trades = trade_count - previous_count
            delta = realized - realized_at_previous_count
            per_trade = delta / float(new_trades)
            for _ in range(new_trades):
                records.append(_TradeRecord(completed_at=open_time, realized_pnl=per_trade))
            previous_count = trade_count
            realized_at_previous_count = realized

    return tuple(records)


def _cagr(
    *,
    starting_equity: float,
    ending_equity: float,
    start_time: datetime,
    end_time: datetime,
) -> float:
    """Return CAGR between two equity points, or ``0`` when undefined."""
    if starting_equity <= 0.0 or ending_equity <= 0.0:
        return 0.0
    elapsed_seconds = (end_time - start_time).total_seconds()
    if elapsed_seconds <= 0.0:
        return 0.0
    years = elapsed_seconds / _SECONDS_PER_YEAR
    return (ending_equity / starting_equity) ** (1.0 / years) - 1.0


def _periods_per_year(open_times: list[datetime]) -> float:
    """Estimate periods-per-year from average bar spacing."""
    if len(open_times) < 2:
        return float(DAYS_PER_YEAR)
    elapsed_seconds = (open_times[-1] - open_times[0]).total_seconds()
    if elapsed_seconds <= 0.0:
        return float(DAYS_PER_YEAR)
    bar_seconds = elapsed_seconds / float(len(open_times) - 1)
    if bar_seconds <= 0.0:
        return float(DAYS_PER_YEAR)
    return _SECONDS_PER_YEAR / bar_seconds


def _sample_std(values: list[float]) -> float | None:
    """Return the sample standard deviation, or ``None`` when undefined."""
    count = len(values)
    if count < 2:
        return None
    mean = sum(values) / float(count)
    variance = sum((value - mean) ** 2 for value in values) / float(count - 1)
    return math.sqrt(variance)


def _downside_deviation(values: list[float]) -> float | None:
    """Return downside deviation for MAR = 0, or ``None`` when empty."""
    count = len(values)
    if count == 0:
        return None
    mean_square = sum(min(value, 0.0) ** 2 for value in values) / float(count)
    return math.sqrt(mean_square)


def _max_drawdown_duration(drawdowns: list[float]) -> int:
    """Return the longest consecutive streak with ``drawdown > 0``."""
    longest = 0
    current = 0
    for drawdown in drawdowns:
        if drawdown > 0.0:
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    return longest


def _validate_finite_snapshot(snapshot: _MetricSnapshot, *, row_index: int) -> None:
    """Raise when any required numeric metric is non-finite."""
    candidates: tuple[tuple[str, float | None], ...] = (
        ("total_return", snapshot.total_return),
        ("cagr", snapshot.cagr),
        ("volatility", snapshot.volatility),
        ("downside_volatility", snapshot.downside_volatility),
        ("max_drawdown", snapshot.max_drawdown),
        ("sharpe_ratio", snapshot.sharpe_ratio),
        ("sortino_ratio", snapshot.sortino_ratio),
        ("calmar_ratio", snapshot.calmar_ratio),
        ("win_rate", snapshot.win_rate),
        ("average_win", snapshot.average_win),
        ("average_loss", snapshot.average_loss),
        ("profit_factor", snapshot.profit_factor),
        ("expectancy", snapshot.expectancy),
        ("starting_equity", snapshot.starting_equity),
        ("ending_equity", snapshot.ending_equity),
        ("net_profit", snapshot.net_profit),
        ("gross_profit", snapshot.gross_profit),
        ("gross_loss", snapshot.gross_loss),
    )
    for name, value in candidates:
        if value is None:
            continue
        if not math.isfinite(value):
            raise PerformanceValidationError(
                f"performance metric '{name}' must be finite",
                error_code=_ERROR_NON_FINITE,
                details={"column": name, "value": value, "row_index": row_index},
            )


def _require_columns(
    frame: pl.DataFrame,
    required: tuple[str, ...],
    dataset: str,
) -> None:
    """Raise when any required column is missing from ``frame``."""
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise PerformanceValidationError(
            f"{dataset} frame is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "dataset": dataset,
                "missing_columns": tuple(missing),
                "required_columns": required,
                "available_columns": tuple(frame.columns),
            },
        )


def _require_sorted_open_times(frame: pl.DataFrame) -> None:
    """Raise when ``open_time`` is not non-decreasing after sorting."""
    open_times = frame["open_time"].to_list()
    for index in range(1, len(open_times)):
        if open_times[index] < open_times[index - 1]:
            raise PerformanceValidationError(
                "open_time must be sorted in non-decreasing order",
                error_code=_ERROR_OPEN_TIME_ORDER,
                details={
                    "index": index,
                    "open_time": open_times[index],
                    "previous_open_time": open_times[index - 1],
                },
            )
