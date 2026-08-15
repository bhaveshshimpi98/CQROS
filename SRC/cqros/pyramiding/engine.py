"""CQROS Pyramiding Engine contracts and simple implementation.

Purpose:
    Convert open position snapshots, accounting, portfolio risk, trade
    management, and market prices into deterministic pyramiding
    recommendations. The engine never executes orders, mutates positions, or
    changes accounting.

Responsibilities:
    - Define ``PyramidingEngine`` as the shared pyramiding contract
    - Provide ``SimplePyramidingEngine`` for long-only add recommendations
    - Validate positions, accounting, portfolio-risk, trade-management, and
      market-price frames
    - Remain free of persistence, verification, CLI, broker APIs, position
      mutation, order submission, and portfolio construction

Dependencies:
    ``polars``, ``math``, ``cqros.pyramiding.exceptions``, and
    ``cqros.pyramiding.schema``.

Public API:
    ``PyramidingEngine``, ``SimplePyramidingEngine``,
    ``ACCOUNTING_INPUT_COLUMNS``, ``POSITION_INPUT_COLUMNS``,
    ``PORTFOLIO_RISK_INPUT_COLUMNS``, ``TRADE_MANAGEMENT_INPUT_COLUMNS``,
    ``MARKET_PRICE_INPUT_COLUMNS``, ``validate_accounting_frame``,
    ``validate_position_frame``, ``validate_portfolio_risk_frame``,
    ``validate_trade_management_frame``, ``validate_market_price_frame``
"""

from __future__ import annotations

import math
from typing import Final, Protocol, runtime_checkable

import polars as pl

from cqros.pyramiding.exceptions import PyramidingValidationError
from cqros.pyramiding.schema import (
    CANONICAL_COLUMN_ORDER,
    DEFAULT_ADD_FRACTION,
    DEFAULT_MAX_ADDS,
    DEFAULT_MIN_PROFIT_PERCENT,
    MERGED_PYRAMIDING_SCHEMA,
    PyramidingReason,
)

__all__ = [
    "ACCOUNTING_INPUT_COLUMNS",
    "MARKET_PRICE_INPUT_COLUMNS",
    "PORTFOLIO_RISK_INPUT_COLUMNS",
    "POSITION_INPUT_COLUMNS",
    "TRADE_MANAGEMENT_INPUT_COLUMNS",
    "PyramidingEngine",
    "SimplePyramidingEngine",
    "validate_accounting_frame",
    "validate_market_price_frame",
    "validate_portfolio_risk_frame",
    "validate_position_frame",
    "validate_trade_management_frame",
]

_ERROR_FRAME_TYPE: Final[str] = "PYR_FRAME_TYPE"
_ERROR_FRAME_EMPTY: Final[str] = "PYR_FRAME_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "PYR_MISSING_COLUMNS"
_ERROR_MANAGER_BLANK: Final[str] = "PYR_MANAGER_BLANK"
_ERROR_POSITION_IDS: Final[str] = "PYR_POSITION_IDS"
_ERROR_MARKET_COVERAGE: Final[str] = "PYR_MARKET_COVERAGE"
_ERROR_RISK_COVERAGE: Final[str] = "PYR_RISK_COVERAGE"
_ERROR_TM_COVERAGE: Final[str] = "PYR_TM_COVERAGE"
_ERROR_LIMIT_NON_FINITE: Final[str] = "PYR_LIMIT_NON_FINITE"
_ERROR_LIMIT_RANGE: Final[str] = "PYR_LIMIT_RANGE"
_ERROR_MAX_ADDS: Final[str] = "PYR_MAX_ADDS_INVALID"

_POSITION_STATUS_OPEN: Final[str] = "OPEN"
_POSITION_SIDE_LONG: Final[str] = "LONG"
_RISK_NORMAL: Final[str] = "NORMAL"
_RISK_WARNING: Final[str] = "WARNING"
_RISK_SHUTDOWN: Final[str] = "SHUTDOWN"
_SHUTDOWN_COOLDOWN: Final[str] = "COOLDOWN"
_TM_ACTION_HOLD: Final[frozenset[str]] = frozenset({"HOLD", "NONE"})
_TM_REASON_TRAILING: Final[str] = "TRAILING_STOP"
_TM_REASON_BREAKEVEN: Final[str] = "BREAKEVEN"

# Accounting columns required to assemble a pyramiding recommendation row.
ACCOUNTING_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
    "position_id",
    "position_status",
    "quantity",
    "average_entry_price",
)

# Position columns required for side eligibility and identity integrity.
POSITION_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "position_id",
    "side",
)

# Portfolio-risk columns required for state, cooldown, and shutdown gating.
PORTFOLIO_RISK_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
    "position_id",
    "portfolio_risk_state",
    "shutdown_reason",
    "cooldown_until",
)

# Trade-management columns required for HOLD / trailing / break-even gating.
TRADE_MANAGEMENT_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
    "position_id",
    "management_action",
    "action_reason",
)

# Market-price columns required for current_price and highest_price.
MARKET_PRICE_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
    "price",
    "high",
)


@runtime_checkable
class PyramidingEngine(Protocol):
    """Structural contract for converting inputs into pyramiding recommendations.

    Implementations own pyramiding semantics. Pipeline orchestration delegates
    exclusively through this contract. Implementations must return a new
    DataFrame and must not mutate the input frames.
    """

    def evaluate(
        self,
        positions: pl.DataFrame,
        accounting: pl.DataFrame,
        portfolio_risk: pl.DataFrame,
        trade_management: pl.DataFrame,
        market_prices: pl.DataFrame,
        *,
        manager: str,
    ) -> pl.DataFrame:
        """Convert input frames into a pyramiding recommendation frame.

        Args:
            positions: Canonical position dataset. Must not be mutated.
            accounting: Canonical accounting dataset. Must not be mutated.
            portfolio_risk: Canonical portfolio-risk dataset. Must not be
                mutated.
            trade_management: Canonical trade-management dataset. Must not be
                mutated.
            market_prices: Market prices keyed by symbol, timeframe, and
                open_time. Must not be mutated.
            manager: Order manager identity preserved onto every row.

        Returns:
            A new DataFrame containing the columns required by the merged
            pyramiding schema contract.
        """
        ...


class SimplePyramidingEngine:
    """Evaluate deterministic long-only pyramiding recommendations.

    Rules (v1):
        - Only LONG, OPEN positions may pyramid.
        - Portfolio risk must be ``NORMAL``.
        - Trade management action must be HOLD (``HOLD`` or ``NONE``).
        - Never pyramid after trailing-stop or break-even actions.
        - Never pyramid during cooldown or after portfolio shutdown.
        - Maximum ``max_adds`` adds; each add is ``add_fraction`` of the
          current theoretical position size.
        - First add requires ``min_profit_percent`` unrealized profit versus
          entry; each subsequent add requires another ``min_profit_percent``.
        - Theoretical size grows after each ``READY_TO_ADD`` within the
          evaluation pass so successive recommendations match the CQROS
          pyramiding example without mutating accounting.

    Args:
        max_adds: Maximum number of adds allowed per position.
        add_fraction: Fraction of current size added on each pyramid step.
        min_profit_percent: Minimum unrealized profit fraction before the
            first add (and between subsequent adds).

    Notes:
        Implementations must not mutate the caller-supplied DataFrames.
        ``trade_id`` mirrors ``position_id`` for CQROS v1 lineage.
    """

    __slots__ = (
        "_add_fraction",
        "_max_adds",
        "_min_profit_percent",
    )

    def __init__(
        self,
        *,
        max_adds: int = DEFAULT_MAX_ADDS,
        add_fraction: float = DEFAULT_ADD_FRACTION,
        min_profit_percent: float = DEFAULT_MIN_PROFIT_PERCENT,
    ) -> None:
        """Initialize the engine with configurable pyramiding rule parameters.

        Args:
            max_adds: Maximum number of adds allowed per position.
            add_fraction: Fraction of current size added on each step.
            min_profit_percent: Minimum unrealized profit fraction per add.

        Raises:
            PyramidingValidationError: If any parameter is invalid.
        """
        self._max_adds = _require_positive_int(max_adds, parameter="max_adds")
        self._add_fraction = _require_fraction(
            add_fraction,
            parameter="add_fraction",
            allow_one=True,
        )
        self._min_profit_percent = _require_fraction(
            min_profit_percent,
            parameter="min_profit_percent",
            allow_one=False,
        )

    def evaluate(
        self,
        positions: pl.DataFrame,
        accounting: pl.DataFrame,
        portfolio_risk: pl.DataFrame,
        trade_management: pl.DataFrame,
        market_prices: pl.DataFrame,
        *,
        manager: str,
    ) -> pl.DataFrame:
        """Evaluate accounting snapshots into finalized pyramiding rows.

        Args:
            positions: Canonical position dataset. Must not be mutated.
            accounting: Canonical accounting dataset. Must not be mutated.
            portfolio_risk: Canonical portfolio-risk dataset. Must not be
                mutated.
            trade_management: Canonical trade-management dataset. Must not be
                mutated.
            market_prices: Market price dataset. Must not be mutated.
            manager: Order manager identity stamped onto every decision row.

        Returns:
            A new DataFrame matching ``MERGED_PYRAMIDING_SCHEMA``.

        Raises:
            PyramidingValidationError: If inputs fail structural validation,
                ``manager`` is blank, required columns are missing, or
                coverage fails.
        """
        positions_frame = validate_position_frame(positions)
        accounting_frame = validate_accounting_frame(accounting)
        risk_frame = validate_portfolio_risk_frame(portfolio_risk)
        tm_frame = validate_trade_management_frame(trade_management)
        prices_frame = validate_market_price_frame(market_prices)
        validated_manager = _require_manager(manager)
        _require_position_columns(positions_frame)
        _require_accounting_columns(accounting_frame)
        _require_portfolio_risk_columns(risk_frame)
        _require_trade_management_columns(tm_frame)
        _require_market_price_columns(prices_frame)
        _require_position_identity_coverage(accounting_frame, positions_frame)
        return _build_pyramiding_frame(
            accounting_frame,
            positions_frame,
            risk_frame,
            tm_frame,
            prices_frame,
            manager=validated_manager,
            max_adds=self._max_adds,
            add_fraction=self._add_fraction,
            min_profit_percent=self._min_profit_percent,
        )


def validate_accounting_frame(frame: object) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty Polars DataFrame.

    Args:
        frame: Candidate accounting dataset passed to an engine.

    Returns:
        ``frame`` as a DataFrame after structural checks.

    Raises:
        PyramidingValidationError: If ``frame`` is not a Polars DataFrame or
            contains no rows.
    """
    return _validate_non_empty_frame(frame, frame_name="accounting")


def validate_position_frame(frame: object) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty Polars DataFrame.

    Args:
        frame: Candidate position dataset passed to an engine.

    Returns:
        ``frame`` as a DataFrame after structural checks.

    Raises:
        PyramidingValidationError: If ``frame`` is not a Polars DataFrame or
            contains no rows.
    """
    return _validate_non_empty_frame(frame, frame_name="positions")


def validate_portfolio_risk_frame(frame: object) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty Polars DataFrame.

    Args:
        frame: Candidate portfolio-risk dataset passed to an engine.

    Returns:
        ``frame`` as a DataFrame after structural checks.

    Raises:
        PyramidingValidationError: If ``frame`` is not a Polars DataFrame or
            contains no rows.
    """
    return _validate_non_empty_frame(frame, frame_name="portfolio_risk")


def validate_trade_management_frame(frame: object) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty Polars DataFrame.

    Args:
        frame: Candidate trade-management dataset passed to an engine.

    Returns:
        ``frame`` as a DataFrame after structural checks.

    Raises:
        PyramidingValidationError: If ``frame`` is not a Polars DataFrame or
            contains no rows.
    """
    return _validate_non_empty_frame(frame, frame_name="trade_management")


def validate_market_price_frame(frame: object) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty Polars DataFrame.

    Args:
        frame: Candidate market-price dataset passed to an engine.

    Returns:
        ``frame`` as a DataFrame after structural checks.

    Raises:
        PyramidingValidationError: If ``frame`` is not a Polars DataFrame or
            contains no rows.
    """
    return _validate_non_empty_frame(frame, frame_name="market_prices")


def _validate_non_empty_frame(frame: object, *, frame_name: str) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty Polars DataFrame."""
    if not isinstance(frame, pl.DataFrame):
        raise PyramidingValidationError(
            f"{frame_name} frame must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={"actual_type": type(frame).__name__, "frame": frame_name},
        )
    if frame.height == 0:
        raise PyramidingValidationError(
            f"{frame_name} frame must contain at least one row",
            error_code=_ERROR_FRAME_EMPTY,
            details={"rows": frame.height, "frame": frame_name},
        )
    return frame


def _build_pyramiding_frame(
    accounting: pl.DataFrame,
    positions: pl.DataFrame,
    portfolio_risk: pl.DataFrame,
    trade_management: pl.DataFrame,
    market_prices: pl.DataFrame,
    *,
    manager: str,
    max_adds: int,
    add_fraction: float,
    min_profit_percent: float,
) -> pl.DataFrame:
    """Assemble canonical pyramiding rows from validated input frames."""
    sides = positions.select(
        [
            pl.col("symbol"),
            pl.col("timeframe"),
            pl.col("position_id"),
            pl.col("side"),
        ]
    )
    prices = market_prices.select(
        [
            pl.col("symbol"),
            pl.col("timeframe"),
            pl.col("open_time"),
            pl.col("price").alias("current_price"),
            pl.col("high").alias("bar_high"),
        ]
    )
    risk = portfolio_risk.select(
        [
            pl.col("symbol"),
            pl.col("timeframe"),
            pl.col("open_time"),
            pl.col("position_id"),
            pl.col("portfolio_risk_state"),
            pl.col("shutdown_reason"),
            pl.col("cooldown_until"),
        ]
    )
    tm = trade_management.select(
        [
            pl.col("symbol"),
            pl.col("timeframe"),
            pl.col("open_time"),
            pl.col("position_id"),
            pl.col("management_action"),
            pl.col("action_reason"),
        ]
    )

    joined = (
        accounting.join(
            sides,
            on=["symbol", "timeframe", "position_id"],
            how="left",
        )
        .join(
            prices,
            on=["symbol", "timeframe", "open_time"],
            how="left",
        )
        .join(
            risk,
            on=["symbol", "timeframe", "open_time", "position_id"],
            how="left",
        )
        .join(
            tm,
            on=["symbol", "timeframe", "open_time", "position_id"],
            how="left",
        )
    )

    missing_prices = joined.filter(pl.col("current_price").is_null() | pl.col("bar_high").is_null())
    if missing_prices.height > 0:
        raise PyramidingValidationError(
            "market prices are missing for one or more accounting snapshots",
            error_code=_ERROR_MARKET_COVERAGE,
            details={
                "missing_rows": missing_prices.height,
                "sample_open_times": tuple(missing_prices["open_time"].head(5).to_list()),
            },
        )

    missing_risk = joined.filter(pl.col("portfolio_risk_state").is_null())
    if missing_risk.height > 0:
        raise PyramidingValidationError(
            "portfolio-risk rows are missing for one or more accounting snapshots",
            error_code=_ERROR_RISK_COVERAGE,
            details={
                "missing_rows": missing_risk.height,
                "sample_position_ids": tuple(missing_risk["position_id"].head(5).to_list()),
            },
        )

    missing_tm = joined.filter(pl.col("management_action").is_null())
    if missing_tm.height > 0:
        raise PyramidingValidationError(
            "trade-management rows are missing for one or more accounting snapshots",
            error_code=_ERROR_TM_COVERAGE,
            details={
                "missing_rows": missing_tm.height,
                "sample_position_ids": tuple(missing_tm["position_id"].head(5).to_list()),
            },
        )

    ordered = joined.sort(["position_id", "open_time"], maintain_order=True)

    symbols = ordered["symbol"].to_list()
    timeframes = ordered["timeframe"].to_list()
    open_times = ordered["open_time"].to_list()
    position_ids = ordered["position_id"].to_list()
    position_statuses = ordered["position_status"].to_list()
    sides_list = ordered["side"].to_list()
    quantities = ordered["quantity"].to_list()
    entry_prices = ordered["average_entry_price"].to_list()
    current_prices = ordered["current_price"].to_list()
    bar_highs = ordered["bar_high"].to_list()
    risk_states = ordered["portfolio_risk_state"].to_list()
    shutdown_reasons = ordered["shutdown_reason"].to_list()
    cooldown_untils = ordered["cooldown_until"].to_list()
    management_actions = ordered["management_action"].to_list()
    action_reasons = ordered["action_reason"].to_list()

    theoretical_size: dict[str, float] = {}
    adds_completed: dict[str, int] = {}
    highest_by_position: dict[str, float] = {}

    out_managers: list[str] = []
    out_symbols: list[str] = []
    out_timeframes: list[str] = []
    out_open_times: list[object] = []
    out_position_ids: list[str] = []
    out_trade_ids: list[str] = []
    out_entry: list[float] = []
    out_current: list[float] = []
    out_highest: list[float] = []
    out_size: list[float] = []
    out_add_number: list[int] = []
    out_max_adds: list[int] = []
    out_additional: list[float] = []
    out_recommended: list[float] = []
    out_profit: list[float] = []
    out_allow: list[bool] = []
    out_reason: list[str] = []

    for index in range(len(open_times)):
        position_id = str(position_ids[index])
        entry_price = float(entry_prices[index])
        current_price = float(current_prices[index])
        bar_high = float(bar_highs[index])
        quantity = float(quantities[index])
        status = str(position_statuses[index])
        side = str(sides_list[index]) if sides_list[index] is not None else ""
        risk_state = str(risk_states[index])
        shutdown_reason = (
            str(shutdown_reasons[index]) if shutdown_reasons[index] is not None else ""
        )
        cooldown_until = cooldown_untils[index]
        management_action = str(management_actions[index])
        action_reason = str(action_reasons[index]) if action_reasons[index] is not None else ""
        open_time = open_times[index]

        if position_id not in theoretical_size:
            theoretical_size[position_id] = quantity
            adds_completed[position_id] = 0

        prior_high = highest_by_position.get(position_id)
        if prior_high is None or bar_high > prior_high:
            highest = bar_high
        else:
            highest = prior_high
        if current_price > highest:
            highest = current_price
        highest_by_position[position_id] = highest

        size = theoretical_size[position_id]
        completed = adds_completed[position_id]
        profit_pct = (current_price - entry_price) / entry_price if entry_price != 0.0 else 0.0

        reason, allow, add_number, additional, recommended = _decide_pyramiding(
            status=status,
            side=side,
            risk_state=risk_state,
            shutdown_reason=shutdown_reason,
            cooldown_until=cooldown_until,
            open_time=open_time,
            management_action=management_action,
            action_reason=action_reason,
            completed=completed,
            max_adds=max_adds,
            size=size,
            profit_pct=profit_pct,
            add_fraction=add_fraction,
            min_profit_percent=min_profit_percent,
        )

        if allow:
            theoretical_size[position_id] = recommended
            adds_completed[position_id] = completed + 1

        out_managers.append(manager)
        out_symbols.append(str(symbols[index]))
        out_timeframes.append(str(timeframes[index]))
        out_open_times.append(open_time)
        out_position_ids.append(position_id)
        out_trade_ids.append(position_id)
        out_entry.append(entry_price)
        out_current.append(current_price)
        out_highest.append(highest)
        out_size.append(size)
        out_add_number.append(add_number)
        out_max_adds.append(max_adds)
        out_additional.append(additional)
        out_recommended.append(recommended)
        out_profit.append(profit_pct)
        out_allow.append(allow)
        out_reason.append(reason)

    assembled = pl.DataFrame(
        {
            "manager": out_managers,
            "symbol": out_symbols,
            "timeframe": out_timeframes,
            "open_time": out_open_times,
            "position_id": out_position_ids,
            "trade_id": out_trade_ids,
            "entry_price": out_entry,
            "current_price": out_current,
            "highest_price": out_highest,
            "position_size": out_size,
            "add_number": out_add_number,
            "max_adds": out_max_adds,
            "additional_size": out_additional,
            "recommended_size": out_recommended,
            "profit_pct": out_profit,
            "allow_pyramid": out_allow,
            "reason": out_reason,
        }
    )
    ordered_out = assembled.sort(
        ["open_time", "position_id"],
        maintain_order=True,
    )
    return ordered_out.select(list(CANONICAL_COLUMN_ORDER)).cast(MERGED_PYRAMIDING_SCHEMA)


def _decide_pyramiding(
    *,
    status: str,
    side: str,
    risk_state: str,
    shutdown_reason: str,
    cooldown_until: object,
    open_time: object,
    management_action: str,
    action_reason: str,
    completed: int,
    max_adds: int,
    size: float,
    profit_pct: float,
    add_fraction: float,
    min_profit_percent: float,
) -> tuple[str, bool, int, float, float]:
    """Return reason, allow flag, add number, additional size, recommended size."""
    if status != _POSITION_STATUS_OPEN or side != _POSITION_SIDE_LONG:
        return (
            PyramidingReason.NOT_ELIGIBLE.value,
            False,
            completed,
            0.0,
            size,
        )

    cooldown_active = _is_cooldown_active(
        risk_state=risk_state,
        shutdown_reason=shutdown_reason,
        cooldown_until=cooldown_until,
        open_time=open_time,
    )
    if risk_state == _RISK_SHUTDOWN and not cooldown_active:
        return (
            PyramidingReason.PORTFOLIO_SHUTDOWN.value,
            False,
            completed,
            0.0,
            size,
        )
    if cooldown_active:
        return (
            PyramidingReason.COOLDOWN_ACTIVE.value,
            False,
            completed,
            0.0,
            size,
        )
    if risk_state == _RISK_WARNING:
        return (
            PyramidingReason.PORTFOLIO_WARNING.value,
            False,
            completed,
            0.0,
            size,
        )
    if risk_state != _RISK_NORMAL:
        return (
            PyramidingReason.NOT_ELIGIBLE.value,
            False,
            completed,
            0.0,
            size,
        )

    if action_reason == _TM_REASON_TRAILING:
        return (
            PyramidingReason.TRAILING_STOP_ACTIVE.value,
            False,
            completed,
            0.0,
            size,
        )
    if action_reason == _TM_REASON_BREAKEVEN:
        return (
            PyramidingReason.BREAKEVEN_ACTIVE.value,
            False,
            completed,
            0.0,
            size,
        )
    if management_action not in _TM_ACTION_HOLD:
        return (
            PyramidingReason.NOT_ELIGIBLE.value,
            False,
            completed,
            0.0,
            size,
        )

    if completed >= max_adds:
        return (
            PyramidingReason.MAX_ADDS_REACHED.value,
            False,
            completed,
            0.0,
            size,
        )

    required_profit = (completed + 1) * min_profit_percent
    if profit_pct + 1e-12 < required_profit:
        return (
            PyramidingReason.INSUFFICIENT_PROFIT.value,
            False,
            completed,
            0.0,
            size,
        )

    additional = size * add_fraction
    recommended = size + additional
    return (
        PyramidingReason.READY_TO_ADD.value,
        True,
        completed + 1,
        additional,
        recommended,
    )


def _is_cooldown_active(
    *,
    risk_state: str,
    shutdown_reason: str,
    cooldown_until: object,
    open_time: object,
) -> bool:
    """Return whether portfolio-risk cooldown blocks pyramiding.

    Cooldown is identified by ``shutdown_reason=COOLDOWN``. Daily-loss and
    exposure shutdowns remain ``PORTFOLIO_SHUTDOWN`` even when
    ``cooldown_until`` is populated on the triggering bar.
    """
    _ = (risk_state, cooldown_until, open_time)
    return shutdown_reason == _SHUTDOWN_COOLDOWN


def _require_manager(manager: object) -> str:
    """Validate and return a non-blank manager identity."""
    if not isinstance(manager, str) or manager.strip() == "":
        raise PyramidingValidationError(
            "manager must be a non-blank string",
            error_code=_ERROR_MANAGER_BLANK,
            details={"manager": manager},
        )
    return manager


def _require_accounting_columns(frame: pl.DataFrame) -> None:
    """Raise when any required accounting column is missing."""
    _require_columns(frame, ACCOUNTING_INPUT_COLUMNS, frame_name="accounting")


def _require_position_columns(frame: pl.DataFrame) -> None:
    """Raise when any required position column is missing."""
    _require_columns(frame, POSITION_INPUT_COLUMNS, frame_name="positions")


def _require_portfolio_risk_columns(frame: pl.DataFrame) -> None:
    """Raise when any required portfolio-risk column is missing."""
    _require_columns(frame, PORTFOLIO_RISK_INPUT_COLUMNS, frame_name="portfolio_risk")


def _require_trade_management_columns(frame: pl.DataFrame) -> None:
    """Raise when any required trade-management column is missing."""
    _require_columns(frame, TRADE_MANAGEMENT_INPUT_COLUMNS, frame_name="trade_management")


def _require_market_price_columns(frame: pl.DataFrame) -> None:
    """Raise when any required market-price column is missing."""
    _require_columns(frame, MARKET_PRICE_INPUT_COLUMNS, frame_name="market_prices")


def _require_columns(
    frame: pl.DataFrame,
    required: tuple[str, ...],
    *,
    frame_name: str,
) -> None:
    """Raise when any required column is missing from ``frame``."""
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise PyramidingValidationError(
            f"{frame_name} frame is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "missing_columns": tuple(missing),
                "required_columns": required,
                "available_columns": tuple(frame.columns),
                "frame": frame_name,
            },
        )


def _require_position_identity_coverage(
    accounting: pl.DataFrame,
    positions: pl.DataFrame,
) -> None:
    """Raise when accounting position_ids are absent from the position frame."""
    accounting_ids = set(accounting["position_id"].to_list())
    position_ids = set(positions["position_id"].to_list())
    missing = tuple(sorted(accounting_ids - position_ids))
    if missing:
        raise PyramidingValidationError(
            "accounting position_id values are missing from positions",
            error_code=_ERROR_POSITION_IDS,
            details={"missing_position_ids": missing},
        )


def _require_fraction(
    value: object,
    *,
    parameter: str,
    allow_one: bool,
) -> float:
    """Validate and return a finite fraction in ``[0, 1)`` or ``[0, 1]``."""
    if type(value) is bool:
        raise PyramidingValidationError(
            f"{parameter} must be a finite number",
            error_code=_ERROR_LIMIT_NON_FINITE,
            details={"parameter": parameter, "value": value},
        )
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise PyramidingValidationError(
            f"{parameter} must be a finite number",
            error_code=_ERROR_LIMIT_NON_FINITE,
            details={"parameter": parameter, "value": value},
        ) from exc
    if not math.isfinite(number):
        raise PyramidingValidationError(
            f"{parameter} must be a finite number",
            error_code=_ERROR_LIMIT_NON_FINITE,
            details={"parameter": parameter, "value": value},
        )
    upper_ok = number <= 1.0 if allow_one else number < 1.0
    if number < 0.0 or not upper_ok:
        raise PyramidingValidationError(
            f"{parameter} must be a finite number in the configured range",
            error_code=_ERROR_LIMIT_RANGE,
            details={"parameter": parameter, "value": value, "allow_one": allow_one},
        )
    return number


def _require_positive_int(value: object, *, parameter: str) -> int:
    """Validate and return a positive integer."""
    if type(value) is bool or not isinstance(value, int):
        raise PyramidingValidationError(
            f"{parameter} must be a positive integer",
            error_code=_ERROR_MAX_ADDS,
            details={"parameter": parameter, "value": value},
        )
    if value < 1:
        raise PyramidingValidationError(
            f"{parameter} must be a positive integer",
            error_code=_ERROR_MAX_ADDS,
            details={"parameter": parameter, "value": value},
        )
    return value
