"""CQROS Trade Management Engine contracts and simple implementation.

Purpose:
    Convert open position snapshots, accounting, portfolio risk, and market
    prices into canonical trade management decisions using deterministic
    trailing-stop and break-even rules.

Responsibilities:
    - Define ``TradeManagementManager`` as the shared trade-management contract
    - Provide ``SimpleTradeManagementManager`` for trailing-stop and
      break-even evaluation (v1 only)
    - Validate positions, accounting, portfolio-risk, and market-price frames
    - Preserve upstream metadata lineage onto every trade-management row
    - Remain free of persistence, verification, CLI, broker APIs, position
      mutation, order submission, pyramiding, partial exits, alpha-decay,
      time exits, ATR/volatility/regime/correlation exits, and sizing

Dependencies:
    ``polars``, ``math``, ``cqros.trade_management.exceptions``, and
    ``cqros.trade_management.schema``.

Public API:
    ``TradeManagementManager``, ``SimpleTradeManagementManager``,
    ``ACCOUNTING_INPUT_COLUMNS``, ``POSITION_INPUT_COLUMNS``,
    ``PORTFOLIO_RISK_INPUT_COLUMNS``, ``MARKET_PRICE_INPUT_COLUMNS``,
    ``validate_accounting_frame``, ``validate_position_frame``,
    ``validate_portfolio_risk_frame``, ``validate_market_price_frame``
"""

from __future__ import annotations

import math
from typing import Final, Protocol, runtime_checkable

import polars as pl

from cqros.trade_management.exceptions import TradeManagementValidationError
from cqros.trade_management.schema import (
    CANONICAL_COLUMN_ORDER,
    DEFAULT_INITIAL_RISK_PERCENT,
    DEFAULT_TRAIL_PERCENT,
    MERGED_TRADE_MANAGEMENT_SCHEMA,
    ManagementAction,
    ShutdownReason,
)

__all__ = [
    "ACCOUNTING_INPUT_COLUMNS",
    "MARKET_PRICE_INPUT_COLUMNS",
    "PORTFOLIO_RISK_INPUT_COLUMNS",
    "POSITION_INPUT_COLUMNS",
    "SimpleTradeManagementManager",
    "TradeManagementManager",
    "validate_accounting_frame",
    "validate_market_price_frame",
    "validate_portfolio_risk_frame",
    "validate_position_frame",
]

_ERROR_FRAME_TYPE: Final[str] = "TME_FRAME_TYPE"
_ERROR_FRAME_EMPTY: Final[str] = "TME_FRAME_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "TME_MISSING_COLUMNS"
_ERROR_MANAGER_BLANK: Final[str] = "TME_MANAGER_BLANK"
_ERROR_POSITION_IDS: Final[str] = "TME_POSITION_IDS"
_ERROR_NO_OPEN: Final[str] = "TME_NO_OPEN_POSITIONS"
_ERROR_MARKET_COVERAGE: Final[str] = "TME_MARKET_COVERAGE"
_ERROR_RISK_COVERAGE: Final[str] = "TME_RISK_COVERAGE"
_ERROR_LIMIT_NON_FINITE: Final[str] = "TME_LIMIT_NON_FINITE"
_ERROR_LIMIT_RANGE: Final[str] = "TME_LIMIT_RANGE"

_POSITION_STATUS_OPEN: Final[str] = "OPEN"

# Accounting columns required to assemble a trade-management decision row.
ACCOUNTING_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
    "position_id",
    "position_status",
    "quantity",
    "average_entry_price",
    "unrealized_pnl",
    "model_name",
    "model_version",
    "optimizer",
    "policy",
)

# Position columns required to confirm accounting position identity integrity.
POSITION_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "position_id",
)

# Portfolio-risk columns required for risk_state and pyramid gating.
PORTFOLIO_RISK_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
    "position_id",
    "portfolio_risk_state",
    "allow_new_entries",
)

# Market-price columns required for current_price at each open_time.
MARKET_PRICE_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
    "price",
)


@runtime_checkable
class TradeManagementManager(Protocol):
    """Structural contract for converting open-position inputs into decisions.

    Implementations own trade-management semantics. Pipeline orchestration
    delegates exclusively through this contract. Implementations must return a
    new DataFrame and must not mutate the input frames.
    """

    def evaluate(
        self,
        positions: pl.DataFrame,
        accounting: pl.DataFrame,
        portfolio_risk: pl.DataFrame,
        market_prices: pl.DataFrame,
        *,
        manager: str,
    ) -> pl.DataFrame:
        """Convert input frames into a trade-management decision frame.

        Args:
            positions: Canonical position dataset. Must not be mutated.
            accounting: Canonical accounting dataset. Must not be mutated.
            portfolio_risk: Canonical portfolio-risk dataset. Must not be
                mutated.
            market_prices: Latest market prices keyed by symbol, timeframe,
                and open_time. Must not be mutated.
            manager: Order manager identity preserved onto every decision row.

        Returns:
            A new DataFrame containing the columns required by the merged
            trade-management schema contract.
        """
        ...


class SimpleTradeManagementManager:
    """Evaluate deterministic trailing-stop and break-even decisions.

    Rules (v1):
        - Trailing Stop-Loss (LONG): maintain ``highest_price`` as the running
          maximum of ``current_price`` per ``position_id``. Compute
          ``trail_price = highest_price * (1 - trail_percent)``. When
          ``current_price <= trail_price``, emit
          ``management_action=UPDATE_STOP`` with
          ``action_reason=TRAILING_STOP``.
        - Break-even Stop (LONG): when
          ``unrealized_reward >= initial_risk`` where
          ``unrealized_reward = current_price - entry_price`` and
          ``initial_risk = entry_price * initial_risk_percent``, emit
          ``management_action=UPDATE_STOP`` with
          ``action_reason=BREAKEVEN`` and ``stop_price = entry_price``.
        - Priority (highest first): trailing-stop hit, then break-even,
          otherwise ``NONE``.
        - Portfolio Risk: record ``risk_state`` from portfolio risk.
          ``allow_pyramid`` is always ``False`` in v1 (stub). When
          ``allow_new_entries=False``, pyramiding remains blocked.
        - Not implemented: pyramiding, partial exits, alpha decay, time
          exits, ATR/volatility/regime/correlation exits, scaling.

    Args:
        trail_percent: Trailing-stop distance as a fraction of
            ``highest_price``.
        initial_risk_percent: Break-even trigger as a fraction of
            ``entry_price`` (1R price distance).

    Notes:
        Implementations must not mutate the caller-supplied DataFrames.
        Only accounting rows with ``position_status=OPEN`` produce decisions.
    """

    __slots__ = (
        "_initial_risk_percent",
        "_trail_percent",
    )

    def __init__(
        self,
        *,
        trail_percent: float = DEFAULT_TRAIL_PERCENT,
        initial_risk_percent: float = DEFAULT_INITIAL_RISK_PERCENT,
    ) -> None:
        """Initialize the manager with configurable stop-rule fractions.

        Args:
            trail_percent: Trailing-stop fraction of highest price.
            initial_risk_percent: Break-even reward fraction of entry price.

        Raises:
            TradeManagementValidationError: If any fraction is invalid.
        """
        self._trail_percent = _require_fraction(
            trail_percent,
            parameter="trail_percent",
        )
        self._initial_risk_percent = _require_fraction(
            initial_risk_percent,
            parameter="initial_risk_percent",
        )

    def evaluate(
        self,
        positions: pl.DataFrame,
        accounting: pl.DataFrame,
        portfolio_risk: pl.DataFrame,
        market_prices: pl.DataFrame,
        *,
        manager: str,
    ) -> pl.DataFrame:
        """Evaluate open-position snapshots into finalized management rows.

        Args:
            positions: Canonical position dataset. Must not be mutated.
            accounting: Canonical accounting dataset. Must not be mutated.
            portfolio_risk: Canonical portfolio-risk dataset. Must not be
                mutated.
            market_prices: Market price dataset. Must not be mutated.
            manager: Order manager identity stamped onto every decision row.

        Returns:
            A new DataFrame matching ``MERGED_TRADE_MANAGEMENT_SCHEMA``.

        Raises:
            TradeManagementValidationError: If inputs fail structural
                validation, ``manager`` is blank, required columns are
                missing, coverage fails, or no open positions remain.
        """
        positions_frame = validate_position_frame(positions)
        accounting_frame = validate_accounting_frame(accounting)
        risk_frame = validate_portfolio_risk_frame(portfolio_risk)
        prices_frame = validate_market_price_frame(market_prices)
        validated_manager = _require_manager(manager)
        _require_position_columns(positions_frame)
        _require_accounting_columns(accounting_frame)
        _require_portfolio_risk_columns(risk_frame)
        _require_market_price_columns(prices_frame)
        _require_position_identity_coverage(accounting_frame, positions_frame)
        return _build_trade_management_frame(
            accounting_frame,
            risk_frame,
            prices_frame,
            manager=validated_manager,
            trail_percent=self._trail_percent,
            initial_risk_percent=self._initial_risk_percent,
        )


def validate_accounting_frame(frame: object) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty Polars DataFrame.

    Args:
        frame: Candidate accounting dataset passed to a manager.

    Returns:
        ``frame`` as a DataFrame after structural checks.

    Raises:
        TradeManagementValidationError: If ``frame`` is not a Polars DataFrame
            or contains no rows.
    """
    return _validate_non_empty_frame(frame, frame_name="accounting")


def validate_position_frame(frame: object) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty Polars DataFrame.

    Args:
        frame: Candidate position dataset passed to a manager.

    Returns:
        ``frame`` as a DataFrame after structural checks.

    Raises:
        TradeManagementValidationError: If ``frame`` is not a Polars DataFrame
            or contains no rows.
    """
    return _validate_non_empty_frame(frame, frame_name="positions")


def validate_portfolio_risk_frame(frame: object) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty Polars DataFrame.

    Args:
        frame: Candidate portfolio-risk dataset passed to a manager.

    Returns:
        ``frame`` as a DataFrame after structural checks.

    Raises:
        TradeManagementValidationError: If ``frame`` is not a Polars DataFrame
            or contains no rows.
    """
    return _validate_non_empty_frame(frame, frame_name="portfolio_risk")


def validate_market_price_frame(frame: object) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty Polars DataFrame.

    Args:
        frame: Candidate market-price dataset passed to a manager.

    Returns:
        ``frame`` as a DataFrame after structural checks.

    Raises:
        TradeManagementValidationError: If ``frame`` is not a Polars DataFrame
            or contains no rows.
    """
    return _validate_non_empty_frame(frame, frame_name="market_prices")


def _validate_non_empty_frame(frame: object, *, frame_name: str) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty Polars DataFrame."""
    if not isinstance(frame, pl.DataFrame):
        raise TradeManagementValidationError(
            f"{frame_name} frame must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={"actual_type": type(frame).__name__, "frame": frame_name},
        )
    if frame.height == 0:
        raise TradeManagementValidationError(
            f"{frame_name} frame must contain at least one row",
            error_code=_ERROR_FRAME_EMPTY,
            details={"rows": frame.height, "frame": frame_name},
        )
    return frame


def _build_trade_management_frame(
    accounting: pl.DataFrame,
    portfolio_risk: pl.DataFrame,
    market_prices: pl.DataFrame,
    *,
    manager: str,
    trail_percent: float,
    initial_risk_percent: float,
) -> pl.DataFrame:
    """Assemble canonical trade-management rows from validated input frames.

    Args:
        accounting: Validated accounting DataFrame.
        portfolio_risk: Validated portfolio-risk DataFrame.
        market_prices: Validated market-price DataFrame.
        manager: Order manager identity for lineage.
        trail_percent: Trailing-stop fraction.
        initial_risk_percent: Break-even reward fraction of entry.

    Returns:
        A new DataFrame ordered and cast to ``MERGED_TRADE_MANAGEMENT_SCHEMA``.
    """
    open_rows = accounting.filter(pl.col("position_status") == _POSITION_STATUS_OPEN)
    if open_rows.height == 0:
        raise TradeManagementValidationError(
            "accounting frame contains no open position snapshots",
            error_code=_ERROR_NO_OPEN,
            details={"rows": accounting.height, "open_rows": 0},
        )

    prices = market_prices.select(
        [
            pl.col("symbol"),
            pl.col("timeframe"),
            pl.col("open_time"),
            pl.col("price").alias("current_price"),
        ]
    )
    risk = portfolio_risk.select(
        [
            pl.col("symbol"),
            pl.col("timeframe"),
            pl.col("open_time"),
            pl.col("position_id"),
            pl.col("portfolio_risk_state").alias("risk_state"),
            pl.col("allow_new_entries"),
        ]
    )

    joined = open_rows.join(
        prices,
        on=["symbol", "timeframe", "open_time"],
        how="left",
    ).join(
        risk,
        on=["symbol", "timeframe", "open_time", "position_id"],
        how="left",
    )

    missing_prices = joined.filter(pl.col("current_price").is_null())
    if missing_prices.height > 0:
        raise TradeManagementValidationError(
            "market prices are missing for one or more open-position snapshots",
            error_code=_ERROR_MARKET_COVERAGE,
            details={
                "missing_rows": missing_prices.height,
                "sample_open_times": tuple(missing_prices["open_time"].head(5).to_list()),
            },
        )

    missing_risk = joined.filter(pl.col("risk_state").is_null())
    if missing_risk.height > 0:
        raise TradeManagementValidationError(
            "portfolio-risk rows are missing for one or more open-position snapshots",
            error_code=_ERROR_RISK_COVERAGE,
            details={
                "missing_rows": missing_risk.height,
                "sample_position_ids": tuple(missing_risk["position_id"].head(5).to_list()),
            },
        )

    ordered = joined.sort(["position_id", "open_time"], maintain_order=True)

    symbols = ordered["symbol"].to_list()
    timeframes = ordered["timeframe"].to_list()
    open_times = ordered["open_time"].to_list()
    position_ids = ordered["position_id"].to_list()
    position_statuses = ordered["position_status"].to_list()
    quantities = ordered["quantity"].to_list()
    entry_prices = ordered["average_entry_price"].to_list()
    current_prices = ordered["current_price"].to_list()
    unrealized_pnls = ordered["unrealized_pnl"].to_list()
    risk_states = ordered["risk_state"].to_list()
    allow_new_entries = ordered["allow_new_entries"].to_list()
    model_names = ordered["model_name"].to_list()
    model_versions = ordered["model_version"].to_list()
    optimizers = ordered["optimizer"].to_list()
    policies = ordered["policy"].to_list()

    highest_by_position: dict[str, float] = {}
    lowest_by_position: dict[str, float] = {}

    out_highest: list[float] = []
    out_lowest: list[float] = []
    out_actions: list[str] = []
    out_reasons: list[str] = []
    out_stop: list[float | None] = []
    out_take_profit: list[float | None] = []
    out_trail: list[float] = []
    out_breakeven: list[float | None] = []
    out_allow_pyramid: list[bool] = []
    out_exit_quantity: list[float] = []

    trail_factor = 1.0 - trail_percent

    for index in range(len(open_times)):
        position_id = str(position_ids[index])
        entry_price = float(entry_prices[index])
        current_price = float(current_prices[index])
        allow_entries = bool(allow_new_entries[index])

        prior_high = highest_by_position.get(position_id)
        prior_low = lowest_by_position.get(position_id)
        if prior_high is None or current_price > prior_high:
            highest = current_price
        else:
            highest = prior_high
        if prior_low is None or current_price < prior_low:
            lowest = current_price
        else:
            lowest = prior_low
        highest_by_position[position_id] = highest
        lowest_by_position[position_id] = lowest

        trail_price = highest * trail_factor
        initial_risk = entry_price * initial_risk_percent
        unrealized_reward = current_price - entry_price
        breakeven_active = unrealized_reward >= initial_risk

        if current_price <= trail_price:
            action = ManagementAction.UPDATE_STOP.value
            reason = ShutdownReason.TRAILING_STOP.value
            stop_price: float | None = trail_price
            breakeven_price: float | None = entry_price if breakeven_active else None
        elif breakeven_active:
            action = ManagementAction.UPDATE_STOP.value
            reason = ShutdownReason.BREAKEVEN.value
            stop_price = entry_price
            breakeven_price = entry_price
        else:
            action = ManagementAction.NONE.value
            reason = ShutdownReason.NONE.value
            stop_price = None
            breakeven_price = None

        # v1 stub: pyramiding is never allowed. When portfolio risk reports
        # allow_new_entries=False the pyramid path remains blocked as well.
        allow_pyramid = allow_entries and False

        out_highest.append(highest)
        out_lowest.append(lowest)
        out_actions.append(action)
        out_reasons.append(reason)
        out_stop.append(stop_price)
        out_take_profit.append(None)
        out_trail.append(trail_price)
        out_breakeven.append(breakeven_price)
        out_allow_pyramid.append(allow_pyramid)
        out_exit_quantity.append(0.0)

    assembled = pl.DataFrame(
        {
            "symbol": symbols,
            "timeframe": timeframes,
            "open_time": open_times,
            "manager": [manager] * len(symbols),
            "position_id": position_ids,
            "position_status": position_statuses,
            "quantity": quantities,
            "entry_price": entry_prices,
            "current_price": current_prices,
            "highest_price": out_highest,
            "lowest_price": out_lowest,
            "unrealized_pnl": unrealized_pnls,
            "risk_state": risk_states,
            "management_action": out_actions,
            "action_reason": out_reasons,
            "stop_price": out_stop,
            "take_profit_price": out_take_profit,
            "trail_price": out_trail,
            "breakeven_price": out_breakeven,
            "allow_pyramid": out_allow_pyramid,
            "exit_quantity": out_exit_quantity,
            "model_name": model_names,
            "model_version": model_versions,
            "optimizer": optimizers,
            "policy": policies,
        }
    )
    ordered_out = assembled.sort(
        ["open_time", "position_id"],
        maintain_order=True,
    )
    return ordered_out.select(list(CANONICAL_COLUMN_ORDER)).cast(MERGED_TRADE_MANAGEMENT_SCHEMA)


def _require_manager(manager: object) -> str:
    """Validate and return a non-blank manager identity."""
    if not isinstance(manager, str) or manager.strip() == "":
        raise TradeManagementValidationError(
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
        raise TradeManagementValidationError(
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
        raise TradeManagementValidationError(
            "accounting position_id values are missing from positions",
            error_code=_ERROR_POSITION_IDS,
            details={"missing_position_ids": missing},
        )


def _require_fraction(value: object, *, parameter: str) -> float:
    """Validate and return a finite fraction in ``[0, 1)``."""
    if type(value) is bool:
        raise TradeManagementValidationError(
            f"{parameter} must be a finite number in [0, 1)",
            error_code=_ERROR_LIMIT_NON_FINITE,
            details={"parameter": parameter, "value": value},
        )
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TradeManagementValidationError(
            f"{parameter} must be a finite number in [0, 1)",
            error_code=_ERROR_LIMIT_NON_FINITE,
            details={"parameter": parameter, "value": value},
        ) from exc
    if not math.isfinite(number):
        raise TradeManagementValidationError(
            f"{parameter} must be a finite number in [0, 1)",
            error_code=_ERROR_LIMIT_NON_FINITE,
            details={"parameter": parameter, "value": value},
        )
    if number < 0.0 or number >= 1.0:
        raise TradeManagementValidationError(
            f"{parameter} must be a finite number in [0, 1)",
            error_code=_ERROR_LIMIT_RANGE,
            details={"parameter": parameter, "value": value},
        )
    return number
