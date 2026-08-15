"""CQROS Exit Engine contracts and simple implementation.

Purpose:
    Convert open position snapshots, accounting, portfolio risk, trade
    management, and pyramiding into deterministic exit recommendations. The
    engine never executes orders, mutates positions, updates accounting, or
    touches OMS.

Responsibilities:
    - Define ``ExitEngine`` as the shared exit-recommendation contract
    - Provide ``SimpleExitEngine`` for priority-ordered exit rules
    - Validate positions, accounting, portfolio-risk, trade-management, and
      pyramiding frames
    - Remain free of persistence, verification, CLI, broker APIs, position
      mutation, order submission, and portfolio construction

Dependencies:
    ``polars``, ``math``, ``cqros.exit_engine.exceptions``, and
    ``cqros.exit_engine.schema``.

Public API:
    ``ExitEngine``, ``SimpleExitEngine``, ``ACCOUNTING_INPUT_COLUMNS``,
    ``POSITION_INPUT_COLUMNS``, ``PORTFOLIO_RISK_INPUT_COLUMNS``,
    ``TRADE_MANAGEMENT_INPUT_COLUMNS``, ``PYRAMIDING_INPUT_COLUMNS``,
    ``validate_accounting_frame``, ``validate_position_frame``,
    ``validate_portfolio_risk_frame``, ``validate_trade_management_frame``,
    ``validate_pyramiding_frame``
"""

from __future__ import annotations

import math
from typing import Final, Protocol, runtime_checkable

import polars as pl

from cqros.exit_engine.exceptions import ExitEngineValidationError
from cqros.exit_engine.schema import (
    CANONICAL_COLUMN_ORDER,
    DEFAULT_INITIAL_RISK_PERCENT,
    DEFAULT_PARTIAL_EXIT_PERCENT,
    DEFAULT_TAKE_PROFIT_MULTIPLE,
    MERGED_EXIT_ENGINE_SCHEMA,
    PRIORITY_ALPHA_DECAY,
    PRIORITY_BREAK_EVEN,
    PRIORITY_COOLDOWN,
    PRIORITY_HOLD,
    PRIORITY_PORTFOLIO_SHUTDOWN,
    PRIORITY_TAKE_PROFIT,
    PRIORITY_TRAILING_STOP,
    ExitAction,
    ExitReason,
)

__all__ = [
    "ACCOUNTING_INPUT_COLUMNS",
    "PORTFOLIO_RISK_INPUT_COLUMNS",
    "POSITION_INPUT_COLUMNS",
    "PYRAMIDING_INPUT_COLUMNS",
    "TRADE_MANAGEMENT_INPUT_COLUMNS",
    "ExitEngine",
    "SimpleExitEngine",
    "validate_accounting_frame",
    "validate_portfolio_risk_frame",
    "validate_position_frame",
    "validate_pyramiding_frame",
    "validate_trade_management_frame",
]

_ERROR_FRAME_TYPE: Final[str] = "EXIT_FRAME_TYPE"
_ERROR_FRAME_EMPTY: Final[str] = "EXIT_FRAME_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "EXIT_MISSING_COLUMNS"
_ERROR_MANAGER_BLANK: Final[str] = "EXIT_MANAGER_BLANK"
_ERROR_POSITION_IDS: Final[str] = "EXIT_POSITION_IDS"
_ERROR_RISK_COVERAGE: Final[str] = "EXIT_RISK_COVERAGE"
_ERROR_TM_COVERAGE: Final[str] = "EXIT_TM_COVERAGE"
_ERROR_PYRAMID_COVERAGE: Final[str] = "EXIT_PYRAMID_COVERAGE"
_ERROR_LIMIT_NON_FINITE: Final[str] = "EXIT_LIMIT_NON_FINITE"
_ERROR_LIMIT_RANGE: Final[str] = "EXIT_LIMIT_RANGE"

_POSITION_STATUS_OPEN: Final[str] = "OPEN"
_POSITION_SIDE_LONG: Final[str] = "LONG"
_RISK_SHUTDOWN: Final[str] = "SHUTDOWN"
_SHUTDOWN_COOLDOWN: Final[str] = "COOLDOWN"
_TM_REASON_TRAILING: Final[str] = "TRAILING_STOP"
_TM_REASON_BREAKEVEN: Final[str] = "BREAKEVEN"
_TM_REASON_ALPHA_DECAY: Final[str] = "ALPHA_DECAY"

# Accounting columns required to assemble an exit recommendation row.
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

# Trade-management columns required for price, action, and exit triggers.
TRADE_MANAGEMENT_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
    "position_id",
    "current_price",
    "management_action",
    "action_reason",
)

# Pyramiding columns required for pyramid-state lineage.
PYRAMIDING_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
    "position_id",
    "reason",
)


@runtime_checkable
class ExitEngine(Protocol):
    """Structural contract for converting inputs into exit recommendations.

    Implementations own exit semantics. Pipeline orchestration delegates
    exclusively through this contract. Implementations must return a new
    DataFrame and must not mutate the input frames.
    """

    def evaluate(
        self,
        positions: pl.DataFrame,
        accounting: pl.DataFrame,
        portfolio_risk: pl.DataFrame,
        trade_management: pl.DataFrame,
        pyramiding: pl.DataFrame,
        *,
        manager: str,
    ) -> pl.DataFrame:
        """Convert input frames into an exit recommendation frame.

        Args:
            positions: Canonical position dataset. Must not be mutated.
            accounting: Canonical accounting dataset. Must not be mutated.
            portfolio_risk: Canonical portfolio-risk dataset. Must not be
                mutated.
            trade_management: Canonical trade-management dataset. Must not be
                mutated.
            pyramiding: Canonical pyramiding dataset. Must not be mutated.
            manager: Order manager identity preserved onto every row.

        Returns:
            A new DataFrame containing the columns required by the merged
            exit-engine schema contract. One row per OPEN position snapshot.
        """
        ...


class SimpleExitEngine:
    """Evaluate deterministic exit recommendations for open positions.

    Rules (v1 priority order):
        1. Portfolio shutdown → ``FULL_EXIT`` / ``PORTFOLIO_SHUTDOWN``.
        2. Cooldown → ``HOLD`` / ``COOLDOWN`` (no exit).
        3. Trailing stop (trade management) → ``FULL_EXIT`` /
           ``TRAILING_STOP``.
        4. Break-even stop (trade management) → ``FULL_EXIT`` /
           ``BREAK_EVEN``.
        5. Risk/reward ≥ ``take_profit_multiple`` × initial risk →
           ``PARTIAL_EXIT`` / ``TAKE_PROFIT`` at ``partial_exit_percent``.
        6. Alpha decay (trade management) → ``FULL_EXIT`` / ``ALPHA_DECAY``.
        7. Time stop (stub) → always falls through.
        8. Regime exit (stub) → always falls through.
        Otherwise → ``HOLD`` / ``NONE``.

    Args:
        initial_risk_percent: Initial risk as a fraction of entry price.
        take_profit_multiple: Reward multiple of initial risk that triggers
            a partial take-profit exit.
        partial_exit_percent: Fraction of quantity recommended on take-profit.

    Notes:
        Implementations must not mutate the caller-supplied DataFrames.
        Closed positions are excluded from the output. ``created_at`` equals
        ``open_time`` so research outputs remain deterministic.
    """

    __slots__ = (
        "_initial_risk_percent",
        "_partial_exit_percent",
        "_take_profit_multiple",
    )

    def __init__(
        self,
        *,
        initial_risk_percent: float = DEFAULT_INITIAL_RISK_PERCENT,
        take_profit_multiple: float = DEFAULT_TAKE_PROFIT_MULTIPLE,
        partial_exit_percent: float = DEFAULT_PARTIAL_EXIT_PERCENT,
    ) -> None:
        """Initialize the engine with configurable exit-rule parameters.

        Args:
            initial_risk_percent: Initial risk fraction of entry price.
            take_profit_multiple: Reward multiple that triggers take-profit.
            partial_exit_percent: Fraction of quantity exited on take-profit.

        Raises:
            ExitEngineValidationError: If any parameter is invalid.
        """
        self._initial_risk_percent = _require_fraction(
            initial_risk_percent,
            parameter="initial_risk_percent",
            allow_one=False,
        )
        self._take_profit_multiple = _require_positive_finite(
            take_profit_multiple,
            parameter="take_profit_multiple",
        )
        self._partial_exit_percent = _require_fraction(
            partial_exit_percent,
            parameter="partial_exit_percent",
            allow_one=True,
        )

    def evaluate(
        self,
        positions: pl.DataFrame,
        accounting: pl.DataFrame,
        portfolio_risk: pl.DataFrame,
        trade_management: pl.DataFrame,
        pyramiding: pl.DataFrame,
        *,
        manager: str,
    ) -> pl.DataFrame:
        """Evaluate open-position snapshots into finalized exit rows.

        Args:
            positions: Canonical position dataset. Must not be mutated.
            accounting: Canonical accounting dataset. Must not be mutated.
            portfolio_risk: Canonical portfolio-risk dataset. Must not be
                mutated.
            trade_management: Canonical trade-management dataset. Must not be
                mutated.
            pyramiding: Canonical pyramiding dataset. Must not be mutated.
            manager: Order manager identity stamped onto every decision row.

        Returns:
            A new DataFrame matching ``MERGED_EXIT_ENGINE_SCHEMA``. May be
            empty when no OPEN positions are present.

        Raises:
            ExitEngineValidationError: If inputs fail structural validation,
                ``manager`` is blank, required columns are missing, or
                coverage fails.
        """
        positions_frame = validate_position_frame(positions)
        accounting_frame = validate_accounting_frame(accounting)
        risk_frame = validate_portfolio_risk_frame(portfolio_risk)
        tm_frame = validate_trade_management_frame(trade_management)
        pyramid_frame = validate_pyramiding_frame(pyramiding)
        validated_manager = _require_manager(manager)
        _require_position_columns(positions_frame)
        _require_accounting_columns(accounting_frame)
        _require_portfolio_risk_columns(risk_frame)
        _require_trade_management_columns(tm_frame)
        _require_pyramiding_columns(pyramid_frame)
        _require_position_identity_coverage(accounting_frame, positions_frame)
        return _build_exit_frame(
            accounting_frame,
            positions_frame,
            risk_frame,
            tm_frame,
            pyramid_frame,
            manager=validated_manager,
            initial_risk_percent=self._initial_risk_percent,
            take_profit_multiple=self._take_profit_multiple,
            partial_exit_percent=self._partial_exit_percent,
        )


def validate_accounting_frame(frame: object) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty Polars DataFrame.

    Args:
        frame: Candidate accounting dataset passed to an engine.

    Returns:
        ``frame`` as a DataFrame after structural checks.

    Raises:
        ExitEngineValidationError: If ``frame`` is not a Polars DataFrame or
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
        ExitEngineValidationError: If ``frame`` is not a Polars DataFrame or
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
        ExitEngineValidationError: If ``frame`` is not a Polars DataFrame or
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
        ExitEngineValidationError: If ``frame`` is not a Polars DataFrame or
            contains no rows.
    """
    return _validate_non_empty_frame(frame, frame_name="trade_management")


def validate_pyramiding_frame(frame: object) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty Polars DataFrame.

    Args:
        frame: Candidate pyramiding dataset passed to an engine.

    Returns:
        ``frame`` as a DataFrame after structural checks.

    Raises:
        ExitEngineValidationError: If ``frame`` is not a Polars DataFrame or
            contains no rows.
    """
    return _validate_non_empty_frame(frame, frame_name="pyramiding")


def _validate_non_empty_frame(frame: object, *, frame_name: str) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty Polars DataFrame."""
    if not isinstance(frame, pl.DataFrame):
        raise ExitEngineValidationError(
            f"{frame_name} frame must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={"actual_type": type(frame).__name__, "frame": frame_name},
        )
    if frame.height == 0:
        raise ExitEngineValidationError(
            f"{frame_name} frame must contain at least one row",
            error_code=_ERROR_FRAME_EMPTY,
            details={"rows": frame.height, "frame": frame_name},
        )
    return frame


def _build_exit_frame(
    accounting: pl.DataFrame,
    positions: pl.DataFrame,
    portfolio_risk: pl.DataFrame,
    trade_management: pl.DataFrame,
    pyramiding: pl.DataFrame,
    *,
    manager: str,
    initial_risk_percent: float,
    take_profit_multiple: float,
    partial_exit_percent: float,
) -> pl.DataFrame:
    """Assemble canonical exit rows from validated input frames."""
    open_accounting = accounting.filter(pl.col("position_status") == _POSITION_STATUS_OPEN)
    if open_accounting.height == 0:
        return pl.DataFrame(schema=MERGED_EXIT_ENGINE_SCHEMA)

    sides = positions.select(
        [
            pl.col("symbol"),
            pl.col("timeframe"),
            pl.col("position_id"),
            pl.col("side"),
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
            pl.col("current_price"),
            pl.col("management_action"),
            pl.col("action_reason"),
        ]
    )
    pyramid = pyramiding.select(
        [
            pl.col("symbol"),
            pl.col("timeframe"),
            pl.col("open_time"),
            pl.col("position_id"),
            pl.col("reason").alias("pyramid_state"),
        ]
    )

    joined = (
        open_accounting.join(
            sides,
            on=["symbol", "timeframe", "position_id"],
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
        .join(
            pyramid,
            on=["symbol", "timeframe", "open_time", "position_id"],
            how="left",
        )
    )

    missing_risk = joined.filter(pl.col("portfolio_risk_state").is_null())
    if missing_risk.height > 0:
        raise ExitEngineValidationError(
            "portfolio-risk rows are missing for one or more open accounting snapshots",
            error_code=_ERROR_RISK_COVERAGE,
            details={
                "missing_rows": missing_risk.height,
                "sample_position_ids": tuple(missing_risk["position_id"].head(5).to_list()),
            },
        )

    missing_tm = joined.filter(pl.col("management_action").is_null())
    if missing_tm.height > 0:
        raise ExitEngineValidationError(
            "trade-management rows are missing for one or more open accounting snapshots",
            error_code=_ERROR_TM_COVERAGE,
            details={
                "missing_rows": missing_tm.height,
                "sample_position_ids": tuple(missing_tm["position_id"].head(5).to_list()),
            },
        )

    missing_pyramid = joined.filter(pl.col("pyramid_state").is_null())
    if missing_pyramid.height > 0:
        raise ExitEngineValidationError(
            "pyramiding rows are missing for one or more open accounting snapshots",
            error_code=_ERROR_PYRAMID_COVERAGE,
            details={
                "missing_rows": missing_pyramid.height,
                "sample_position_ids": tuple(missing_pyramid["position_id"].head(5).to_list()),
            },
        )

    ordered = joined.sort(["open_time", "position_id"], maintain_order=True)

    symbols = ordered["symbol"].to_list()
    timeframes = ordered["timeframe"].to_list()
    open_times = ordered["open_time"].to_list()
    position_ids = ordered["position_id"].to_list()
    sides_list = ordered["side"].to_list()
    quantities = ordered["quantity"].to_list()
    entry_prices = ordered["average_entry_price"].to_list()
    current_prices = ordered["current_price"].to_list()
    risk_states = ordered["portfolio_risk_state"].to_list()
    shutdown_reasons = ordered["shutdown_reason"].to_list()
    management_actions = ordered["management_action"].to_list()
    action_reasons = ordered["action_reason"].to_list()
    pyramid_states = ordered["pyramid_state"].to_list()

    out_symbols: list[str] = []
    out_timeframes: list[str] = []
    out_open_times: list[object] = []
    out_position_ids: list[str] = []
    out_managers: list[str] = []
    out_entry: list[float] = []
    out_current: list[float] = []
    out_quantity: list[float] = []
    out_rr: list[float] = []
    out_risk_state: list[str] = []
    out_trade_state: list[str] = []
    out_pyramid_state: list[str] = []
    out_action: list[str] = []
    out_reason: list[str] = []
    out_rec_qty: list[float] = []
    out_rec_pct: list[float] = []
    out_priority: list[int] = []
    out_created: list[object] = []

    for index in range(len(open_times)):
        entry_price = float(entry_prices[index])
        current_price = float(current_prices[index])
        quantity = float(quantities[index])
        side = str(sides_list[index]) if sides_list[index] is not None else ""
        risk_state = str(risk_states[index])
        shutdown_reason = (
            str(shutdown_reasons[index]) if shutdown_reasons[index] is not None else ""
        )
        management_action = str(management_actions[index])
        action_reason = str(action_reasons[index]) if action_reasons[index] is not None else ""
        pyramid_state = str(pyramid_states[index])
        open_time = open_times[index]

        risk_reward_ratio = _compute_risk_reward_ratio(
            entry_price=entry_price,
            current_price=current_price,
            side=side,
            initial_risk_percent=initial_risk_percent,
        )
        exit_action, exit_reason, recommended_percent, priority = _decide_exit(
            risk_state=risk_state,
            shutdown_reason=shutdown_reason,
            action_reason=action_reason,
            risk_reward_ratio=risk_reward_ratio,
            take_profit_multiple=take_profit_multiple,
            partial_exit_percent=partial_exit_percent,
        )
        recommended_quantity = quantity * recommended_percent

        out_symbols.append(str(symbols[index]))
        out_timeframes.append(str(timeframes[index]))
        out_open_times.append(open_time)
        out_position_ids.append(str(position_ids[index]))
        out_managers.append(manager)
        out_entry.append(entry_price)
        out_current.append(current_price)
        out_quantity.append(quantity)
        out_rr.append(risk_reward_ratio)
        out_risk_state.append(risk_state)
        out_trade_state.append(management_action)
        out_pyramid_state.append(pyramid_state)
        out_action.append(exit_action)
        out_reason.append(exit_reason)
        out_rec_qty.append(recommended_quantity)
        out_rec_pct.append(recommended_percent)
        out_priority.append(priority)
        out_created.append(open_time)

    assembled = pl.DataFrame(
        {
            "symbol": out_symbols,
            "timeframe": out_timeframes,
            "open_time": out_open_times,
            "position_id": out_position_ids,
            "manager": out_managers,
            "entry_price": out_entry,
            "current_price": out_current,
            "quantity": out_quantity,
            "risk_reward_ratio": out_rr,
            "risk_state": out_risk_state,
            "trade_state": out_trade_state,
            "pyramid_state": out_pyramid_state,
            "exit_action": out_action,
            "exit_reason": out_reason,
            "recommended_quantity": out_rec_qty,
            "recommended_percent": out_rec_pct,
            "priority": out_priority,
            "created_at": out_created,
        }
    )
    ordered_out = assembled.sort(
        ["open_time", "position_id"],
        maintain_order=True,
    )
    return ordered_out.select(list(CANONICAL_COLUMN_ORDER)).cast(MERGED_EXIT_ENGINE_SCHEMA)


def _compute_risk_reward_ratio(
    *,
    entry_price: float,
    current_price: float,
    side: str,
    initial_risk_percent: float,
) -> float:
    """Return current reward divided by initial risk distance."""
    initial_risk = entry_price * initial_risk_percent
    if initial_risk == 0.0:
        return 0.0
    if side == _POSITION_SIDE_LONG:
        current_reward = current_price - entry_price
    else:
        current_reward = entry_price - current_price
    return current_reward / initial_risk


def _decide_exit(
    *,
    risk_state: str,
    shutdown_reason: str,
    action_reason: str,
    risk_reward_ratio: float,
    take_profit_multiple: float,
    partial_exit_percent: float,
) -> tuple[str, str, float, int]:
    """Return exit_action, exit_reason, recommended_percent, priority."""
    cooldown_active = shutdown_reason == _SHUTDOWN_COOLDOWN

    # 1. Portfolio shutdown — immediate FULL_EXIT.
    if risk_state == _RISK_SHUTDOWN and not cooldown_active:
        return (
            ExitAction.FULL_EXIT.value,
            ExitReason.PORTFOLIO_SHUTDOWN.value,
            1.0,
            PRIORITY_PORTFOLIO_SHUTDOWN,
        )

    # 2. Cooldown — no exit; HOLD only.
    if cooldown_active:
        return (
            ExitAction.HOLD.value,
            ExitReason.COOLDOWN.value,
            0.0,
            PRIORITY_COOLDOWN,
        )

    # 3. Trailing stop from trade management.
    if action_reason == _TM_REASON_TRAILING:
        return (
            ExitAction.FULL_EXIT.value,
            ExitReason.TRAILING_STOP.value,
            1.0,
            PRIORITY_TRAILING_STOP,
        )

    # 4. Break-even stop from trade management.
    if action_reason == _TM_REASON_BREAKEVEN:
        return (
            ExitAction.FULL_EXIT.value,
            ExitReason.BREAK_EVEN.value,
            1.0,
            PRIORITY_BREAK_EVEN,
        )

    # 5. Risk/reward take-profit partial exit.
    if risk_reward_ratio >= take_profit_multiple:
        return (
            ExitAction.PARTIAL_EXIT.value,
            ExitReason.TAKE_PROFIT.value,
            partial_exit_percent,
            PRIORITY_TAKE_PROFIT,
        )

    # 6. Alpha decay from trade management.
    if action_reason == _TM_REASON_ALPHA_DECAY:
        return (
            ExitAction.FULL_EXIT.value,
            ExitReason.ALPHA_DECAY.value,
            1.0,
            PRIORITY_ALPHA_DECAY,
        )

    # 7. Time stop stub — always HOLD / NONE.
    # 8. Regime exit stub — always HOLD / NONE.
    return (
        ExitAction.HOLD.value,
        ExitReason.NONE.value,
        0.0,
        PRIORITY_HOLD,
    )


def _require_manager(manager: object) -> str:
    """Validate and return a non-blank manager identity."""
    if not isinstance(manager, str) or manager.strip() == "":
        raise ExitEngineValidationError(
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


def _require_pyramiding_columns(frame: pl.DataFrame) -> None:
    """Raise when any required pyramiding column is missing."""
    _require_columns(frame, PYRAMIDING_INPUT_COLUMNS, frame_name="pyramiding")


def _require_columns(
    frame: pl.DataFrame,
    required: tuple[str, ...],
    *,
    frame_name: str,
) -> None:
    """Raise when any required column is missing from ``frame``."""
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ExitEngineValidationError(
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
        raise ExitEngineValidationError(
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
        raise ExitEngineValidationError(
            f"{parameter} must be a finite number",
            error_code=_ERROR_LIMIT_NON_FINITE,
            details={"parameter": parameter, "value": value},
        )
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ExitEngineValidationError(
            f"{parameter} must be a finite number",
            error_code=_ERROR_LIMIT_NON_FINITE,
            details={"parameter": parameter, "value": value},
        ) from exc
    if not math.isfinite(number):
        raise ExitEngineValidationError(
            f"{parameter} must be a finite number",
            error_code=_ERROR_LIMIT_NON_FINITE,
            details={"parameter": parameter, "value": value},
        )
    upper_ok = number <= 1.0 if allow_one else number < 1.0
    if number < 0.0 or not upper_ok:
        raise ExitEngineValidationError(
            f"{parameter} must be a finite number in the configured range",
            error_code=_ERROR_LIMIT_RANGE,
            details={"parameter": parameter, "value": value, "allow_one": allow_one},
        )
    return number


def _require_positive_finite(value: object, *, parameter: str) -> float:
    """Validate and return a finite number strictly greater than zero."""
    if type(value) is bool:
        raise ExitEngineValidationError(
            f"{parameter} must be a finite number greater than zero",
            error_code=_ERROR_LIMIT_NON_FINITE,
            details={"parameter": parameter, "value": value},
        )
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ExitEngineValidationError(
            f"{parameter} must be a finite number greater than zero",
            error_code=_ERROR_LIMIT_NON_FINITE,
            details={"parameter": parameter, "value": value},
        ) from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ExitEngineValidationError(
            f"{parameter} must be a finite number greater than zero",
            error_code=_ERROR_LIMIT_RANGE,
            details={"parameter": parameter, "value": value},
        )
    return number
