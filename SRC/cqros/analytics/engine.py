"""CQROS Analytics Engine contracts and simple implementation.

Purpose:
    Convert a canonical performance-metrics ledger into a deterministic
    analytics DataFrame conforming to ``ANALYTICS_SCHEMA``.

Responsibilities:
    - Define ``AnalyticsEngine`` as the shared analytics contract
    - Provide ``SimpleAnalyticsEngine`` for expanding-window analytics
    - Validate performance DataFrame structure and finite numeric outputs
    - Remain free of persistence, verification, CLI, storage, and file I/O

Dependencies:
    ``math``, ``polars``, ``cqros.analytics.exceptions``, and
    ``cqros.analytics.schema``.

Public API:
    ``AnalyticsEngine``, ``SimpleAnalyticsEngine``,
    ``PERFORMANCE_INPUT_COLUMNS``, ``validate_performance_frame``
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Final, Protocol, runtime_checkable

import polars as pl

from cqros.analytics.exceptions import AnalyticsValidationError
from cqros.analytics.schema import (
    ANALYTICS_SCHEMA,
    CANONICAL_COLUMN_ORDER,
)

__all__ = [
    "PERFORMANCE_INPUT_COLUMNS",
    "AnalyticsEngine",
    "SimpleAnalyticsEngine",
    "validate_performance_frame",
]

_ERROR_FRAME_TYPE: Final[str] = "ANA_FRAME_TYPE"
_ERROR_FRAME_EMPTY: Final[str] = "ANA_FRAME_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "ANA_MISSING_COLUMNS"
_ERROR_NON_FINITE: Final[str] = "ANA_NON_FINITE"
_ERROR_OPEN_TIME_ORDER: Final[str] = "ANA_OPEN_TIME_ORDER"

_BENCHMARK_DEFAULT: Final[float] = 0.0

# Performance columns required to assemble an analytics row.
PERFORMANCE_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
    "manager",
    "total_return",
    "volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "win_rate",
    "profit_factor",
    "expectancy",
    "cagr",
    "calmar_ratio",
    "net_profit",
    "status",
)


@runtime_checkable
class AnalyticsEngine(Protocol):
    """Structural contract for converting performance ledgers into analytics.

    Implementations own analytics-metric semantics. Pipeline orchestration
    delegates exclusively through this contract. Implementations must return
    a new DataFrame and must not mutate the input frame.
    """

    def build(self, performance: pl.DataFrame) -> pl.DataFrame:
        """Convert a performance ledger into an analytics DataFrame.

        Args:
            performance: Canonical performance dataset. Must not be mutated.

        Returns:
            A new DataFrame containing the columns required by
            ``ANALYTICS_SCHEMA``.
        """
        ...


class SimpleAnalyticsEngine:
    """Compute deterministic expanding-window analytics from performance rows.

    Rules:
        - One output row per input evaluation timestamp
        - Identity columns (``symbol``, ``timeframe``, ``open_time``,
          ``manager``) are preserved from the input row
        - ``open_time`` is emitted as epoch milliseconds (``Int64``)
        - Metrics at row ``i`` use only information from rows ``0..i``
        - Rolling return/risk/trade metrics map from the performance metrics
          already evaluated through the current timestamp
        - ``rolling_recovery_factor = net_profit / max_drawdown`` when
          drawdown ``> 0``, otherwise ``NULL``
        - Benchmark fields are deterministic ``0.0`` stubs in v1
        - ``status`` is copied from the corresponding performance row

    Notes:
        Implementations must not mutate the caller-supplied DataFrame.
    """

    __slots__ = ()

    def build(self, performance: pl.DataFrame) -> pl.DataFrame:
        """Convert a performance ledger into finalized analytics metrics.

        Args:
            performance: Canonical performance dataset. Must not be mutated.

        Returns:
            A new DataFrame matching ``ANALYTICS_SCHEMA``.

        Raises:
            AnalyticsValidationError: If the input fails structural
                validation, required columns are missing, timestamps are
                unsorted, or a numeric output is non-finite.
        """
        frame = validate_performance_frame(performance)
        _require_columns(frame, PERFORMANCE_INPUT_COLUMNS, "performance")
        ordered = frame.sort("open_time", maintain_order=True)
        _require_sorted_open_times(ordered)
        return _build_analytics_metrics(ordered)


def validate_performance_frame(frame: object) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty Polars DataFrame.

    Args:
        frame: Candidate performance dataset passed to an engine.

    Returns:
        ``frame`` as a DataFrame after structural checks.

    Raises:
        AnalyticsValidationError: If ``frame`` is not a Polars DataFrame or
            contains no rows.
    """
    if not isinstance(frame, pl.DataFrame):
        raise AnalyticsValidationError(
            "performance frame must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={"dataset": "performance", "actual_type": type(frame).__name__},
        )
    if frame.height == 0:
        raise AnalyticsValidationError(
            "performance frame must contain at least one row",
            error_code=_ERROR_FRAME_EMPTY,
            details={"dataset": "performance", "rows": frame.height},
        )
    return frame


class _AnalyticsSnapshot:
    """Immutable analytics values for one evaluation timestamp."""

    __slots__ = (
        "benchmark_alpha",
        "benchmark_beta",
        "benchmark_correlation",
        "benchmark_information_ratio",
        "benchmark_return",
        "benchmark_tracking_error",
        "rolling_calmar",
        "rolling_cagr",
        "rolling_expectancy",
        "rolling_max_drawdown",
        "rolling_profit_factor",
        "rolling_recovery_factor",
        "rolling_return",
        "rolling_sharpe",
        "rolling_sortino",
        "rolling_volatility",
        "rolling_win_rate",
        "status",
    )

    def __init__(
        self,
        *,
        rolling_return: float,
        rolling_volatility: float,
        rolling_sharpe: float | None,
        rolling_sortino: float | None,
        rolling_max_drawdown: float,
        rolling_win_rate: float,
        rolling_profit_factor: float | None,
        rolling_expectancy: float,
        rolling_cagr: float,
        rolling_calmar: float | None,
        rolling_recovery_factor: float | None,
        benchmark_return: float,
        benchmark_alpha: float,
        benchmark_beta: float,
        benchmark_correlation: float,
        benchmark_tracking_error: float,
        benchmark_information_ratio: float,
        status: str,
    ) -> None:
        self.rolling_return = rolling_return
        self.rolling_volatility = rolling_volatility
        self.rolling_sharpe = rolling_sharpe
        self.rolling_sortino = rolling_sortino
        self.rolling_max_drawdown = rolling_max_drawdown
        self.rolling_win_rate = rolling_win_rate
        self.rolling_profit_factor = rolling_profit_factor
        self.rolling_expectancy = rolling_expectancy
        self.rolling_cagr = rolling_cagr
        self.rolling_calmar = rolling_calmar
        self.rolling_recovery_factor = rolling_recovery_factor
        self.benchmark_return = benchmark_return
        self.benchmark_alpha = benchmark_alpha
        self.benchmark_beta = benchmark_beta
        self.benchmark_correlation = benchmark_correlation
        self.benchmark_tracking_error = benchmark_tracking_error
        self.benchmark_information_ratio = benchmark_information_ratio
        self.status = status


def _build_analytics_metrics(performance: pl.DataFrame) -> pl.DataFrame:
    """Assemble canonical analytics rows from a sorted performance ledger."""
    open_times = [_to_epoch_ms(value) for value in performance["open_time"].to_list()]
    total_returns = [_as_float(value) for value in performance["total_return"].to_list()]
    volatilities = [_as_float(value) for value in performance["volatility"].to_list()]
    sharpe_ratios = [_as_optional_float(value) for value in performance["sharpe_ratio"].to_list()]
    sortino_ratios = [_as_optional_float(value) for value in performance["sortino_ratio"].to_list()]
    max_drawdowns = [_as_float(value) for value in performance["max_drawdown"].to_list()]
    win_rates = [_as_float(value) for value in performance["win_rate"].to_list()]
    profit_factors = [_as_optional_float(value) for value in performance["profit_factor"].to_list()]
    expectancies = [_as_float(value) for value in performance["expectancy"].to_list()]
    cagrs = [_as_float(value) for value in performance["cagr"].to_list()]
    calmar_ratios = [_as_optional_float(value) for value in performance["calmar_ratio"].to_list()]
    net_profits = [_as_float(value) for value in performance["net_profit"].to_list()]
    statuses = [str(value) for value in performance["status"].to_list()]

    row_count = len(open_times)

    rolling_returns: list[float] = []
    rolling_volatilities: list[float] = []
    rolling_sharpes: list[float | None] = []
    rolling_sortinos: list[float | None] = []
    rolling_max_drawdowns: list[float] = []
    rolling_win_rates: list[float] = []
    rolling_profit_factors: list[float | None] = []
    rolling_expectancies: list[float] = []
    rolling_cagrs: list[float] = []
    rolling_calmars: list[float | None] = []
    rolling_recovery_factors: list[float | None] = []
    benchmark_returns: list[float] = []
    benchmark_alphas: list[float] = []
    benchmark_betas: list[float] = []
    benchmark_correlations: list[float] = []
    benchmark_tracking_errors: list[float] = []
    benchmark_information_ratios: list[float] = []
    output_statuses: list[str] = []

    for index in range(row_count):
        # Expanding window through the current timestamp (rows 0..index).
        snapshot = _metrics_at(
            total_return=total_returns[index],
            volatility=volatilities[index],
            sharpe_ratio=sharpe_ratios[index],
            sortino_ratio=sortino_ratios[index],
            max_drawdown=max_drawdowns[index],
            win_rate=win_rates[index],
            profit_factor=profit_factors[index],
            expectancy=expectancies[index],
            cagr=cagrs[index],
            calmar_ratio=calmar_ratios[index],
            net_profit=net_profits[index],
            status=statuses[index],
        )
        _validate_finite_snapshot(snapshot, row_index=index)

        rolling_returns.append(snapshot.rolling_return)
        rolling_volatilities.append(snapshot.rolling_volatility)
        rolling_sharpes.append(snapshot.rolling_sharpe)
        rolling_sortinos.append(snapshot.rolling_sortino)
        rolling_max_drawdowns.append(snapshot.rolling_max_drawdown)
        rolling_win_rates.append(snapshot.rolling_win_rate)
        rolling_profit_factors.append(snapshot.rolling_profit_factor)
        rolling_expectancies.append(snapshot.rolling_expectancy)
        rolling_cagrs.append(snapshot.rolling_cagr)
        rolling_calmars.append(snapshot.rolling_calmar)
        rolling_recovery_factors.append(snapshot.rolling_recovery_factor)
        benchmark_returns.append(snapshot.benchmark_return)
        benchmark_alphas.append(snapshot.benchmark_alpha)
        benchmark_betas.append(snapshot.benchmark_beta)
        benchmark_correlations.append(snapshot.benchmark_correlation)
        benchmark_tracking_errors.append(snapshot.benchmark_tracking_error)
        benchmark_information_ratios.append(snapshot.benchmark_information_ratio)
        output_statuses.append(snapshot.status)

    assembled = pl.DataFrame(
        {
            "symbol": performance["symbol"].to_list(),
            "timeframe": performance["timeframe"].to_list(),
            "open_time": open_times,
            "manager": performance["manager"].to_list(),
            "rolling_return": rolling_returns,
            "rolling_volatility": rolling_volatilities,
            "rolling_sharpe": rolling_sharpes,
            "rolling_sortino": rolling_sortinos,
            "rolling_max_drawdown": rolling_max_drawdowns,
            "rolling_win_rate": rolling_win_rates,
            "rolling_profit_factor": rolling_profit_factors,
            "rolling_expectancy": rolling_expectancies,
            "rolling_cagr": rolling_cagrs,
            "rolling_calmar": rolling_calmars,
            "rolling_recovery_factor": rolling_recovery_factors,
            "benchmark_return": benchmark_returns,
            "benchmark_alpha": benchmark_alphas,
            "benchmark_beta": benchmark_betas,
            "benchmark_correlation": benchmark_correlations,
            "benchmark_tracking_error": benchmark_tracking_errors,
            "benchmark_information_ratio": benchmark_information_ratios,
            "status": output_statuses,
        }
    )
    return assembled.select(list(CANONICAL_COLUMN_ORDER)).cast(ANALYTICS_SCHEMA)


def _metrics_at(
    *,
    total_return: float,
    volatility: float,
    sharpe_ratio: float | None,
    sortino_ratio: float | None,
    max_drawdown: float,
    win_rate: float,
    profit_factor: float | None,
    expectancy: float,
    cagr: float,
    calmar_ratio: float | None,
    net_profit: float,
    status: str,
) -> _AnalyticsSnapshot:
    """Compute analytics metrics for one expanding-window evaluation."""
    if max_drawdown > 0.0:
        recovery_factor: float | None = net_profit / max_drawdown
    else:
        recovery_factor = None

    return _AnalyticsSnapshot(
        rolling_return=total_return,
        rolling_volatility=volatility,
        rolling_sharpe=sharpe_ratio,
        rolling_sortino=sortino_ratio,
        rolling_max_drawdown=max_drawdown,
        rolling_win_rate=win_rate,
        rolling_profit_factor=profit_factor,
        rolling_expectancy=expectancy,
        rolling_cagr=cagr,
        rolling_calmar=calmar_ratio,
        rolling_recovery_factor=recovery_factor,
        benchmark_return=_BENCHMARK_DEFAULT,
        benchmark_alpha=_BENCHMARK_DEFAULT,
        benchmark_beta=_BENCHMARK_DEFAULT,
        benchmark_correlation=_BENCHMARK_DEFAULT,
        benchmark_tracking_error=_BENCHMARK_DEFAULT,
        benchmark_information_ratio=_BENCHMARK_DEFAULT,
        status=status,
    )


def _to_epoch_ms(value: object) -> int:
    """Convert a performance ``open_time`` value to epoch milliseconds."""
    if isinstance(value, datetime):
        return int(value.timestamp() * 1000.0)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    raise AnalyticsValidationError(
        "open_time must be datetime or integer epoch milliseconds",
        error_code=_ERROR_FRAME_TYPE,
        details={"actual_type": type(value).__name__, "value": repr(value)},
    )


def _as_float(value: object) -> float:
    """Coerce a required numeric cell to ``float``."""
    if value is None:
        return float("nan")
    return float(value)  # type: ignore[arg-type]


def _as_optional_float(value: object) -> float | None:
    """Coerce an optional numeric cell to ``float`` or ``None``."""
    if value is None:
        return None
    return float(value)  # type: ignore[arg-type]


def _validate_finite_snapshot(snapshot: _AnalyticsSnapshot, *, row_index: int) -> None:
    """Raise when any required numeric metric is non-finite."""
    candidates: tuple[tuple[str, float | None], ...] = (
        ("rolling_return", snapshot.rolling_return),
        ("rolling_volatility", snapshot.rolling_volatility),
        ("rolling_sharpe", snapshot.rolling_sharpe),
        ("rolling_sortino", snapshot.rolling_sortino),
        ("rolling_max_drawdown", snapshot.rolling_max_drawdown),
        ("rolling_win_rate", snapshot.rolling_win_rate),
        ("rolling_profit_factor", snapshot.rolling_profit_factor),
        ("rolling_expectancy", snapshot.rolling_expectancy),
        ("rolling_cagr", snapshot.rolling_cagr),
        ("rolling_calmar", snapshot.rolling_calmar),
        ("rolling_recovery_factor", snapshot.rolling_recovery_factor),
        ("benchmark_return", snapshot.benchmark_return),
        ("benchmark_alpha", snapshot.benchmark_alpha),
        ("benchmark_beta", snapshot.benchmark_beta),
        ("benchmark_correlation", snapshot.benchmark_correlation),
        ("benchmark_tracking_error", snapshot.benchmark_tracking_error),
        ("benchmark_information_ratio", snapshot.benchmark_information_ratio),
    )
    for name, value in candidates:
        if value is None:
            continue
        if not math.isfinite(value):
            raise AnalyticsValidationError(
                f"analytics metric '{name}' must be finite",
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
        raise AnalyticsValidationError(
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
            raise AnalyticsValidationError(
                "open_time must be sorted in non-decreasing order",
                error_code=_ERROR_OPEN_TIME_ORDER,
                details={
                    "index": index,
                    "open_time": open_times[index],
                    "previous_open_time": open_times[index - 1],
                },
            )
