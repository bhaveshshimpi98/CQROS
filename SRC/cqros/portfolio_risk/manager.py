"""CQROS Portfolio Risk Manager contracts and simple implementation.

Purpose:
    Convert canonical accounting snapshots into canonical portfolio risk
    decisions using deterministic portfolio-level risk rules.

Responsibilities:
    - Define ``PortfolioRiskManager`` as the shared portfolio-risk contract
    - Provide ``SimplePortfolioRiskManager`` for daily-loss, cooldown, and
      exposure-limit evaluation
    - Validate accounting and position DataFrame structure
    - Preserve upstream metadata lineage onto every portfolio-risk row
    - Remain free of persistence, verification, CLI, broker APIs, position
      mutation, order submission, trailing stops, pyramiding, and sizing

Dependencies:
    ``polars``, ``datetime``, ``cqros.portfolio_risk.exceptions``, and
    ``cqros.portfolio_risk.schema``.

Public API:
    ``PortfolioRiskManager``, ``SimplePortfolioRiskManager``,
    ``ACCOUNTING_INPUT_COLUMNS``, ``POSITION_INPUT_COLUMNS``,
    ``validate_accounting_frame``, ``validate_position_frame``
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Final, Protocol, runtime_checkable

import polars as pl

from cqros.portfolio_risk.exceptions import PortfolioRiskValidationError
from cqros.portfolio_risk.schema import (
    CANONICAL_COLUMN_ORDER,
    DEFAULT_COOLDOWN_HOURS,
    DEFAULT_DAILY_LOSS_LIMIT,
    DEFAULT_GROSS_EXPOSURE_LIMIT,
    MERGED_PORTFOLIO_RISK_SCHEMA,
    PortfolioRiskState,
    ShutdownReason,
)

__all__ = [
    "ACCOUNTING_INPUT_COLUMNS",
    "POSITION_INPUT_COLUMNS",
    "PortfolioRiskManager",
    "SimplePortfolioRiskManager",
    "validate_accounting_frame",
    "validate_position_frame",
]

_ERROR_FRAME_TYPE: Final[str] = "PRISK_FRAME_TYPE"
_ERROR_FRAME_EMPTY: Final[str] = "PRISK_FRAME_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "PRISK_MISSING_COLUMNS"
_ERROR_MANAGER_BLANK: Final[str] = "PRISK_MANAGER_BLANK"
_ERROR_POSITION_IDS: Final[str] = "PRISK_POSITION_IDS"
_ERROR_LIMIT_NON_FINITE: Final[str] = "PRISK_LIMIT_NON_FINITE"
_ERROR_COOLDOWN_INVALID: Final[str] = "PRISK_COOLDOWN_INVALID"

# Accounting columns required to assemble a portfolio-risk decision row.
ACCOUNTING_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
    "position_id",
    "equity",
    "gross_exposure",
    "net_exposure",
    "realized_pnl",
    "unrealized_pnl",
    "total_pnl",
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


@runtime_checkable
class PortfolioRiskManager(Protocol):
    """Structural contract for converting accounting frames into risk decisions.

    Implementations own portfolio-risk semantics. Pipeline orchestration
    delegates exclusively through this contract. Implementations must return a
    new DataFrame and must not mutate the input accounting or position frames.
    """

    def evaluate(
        self,
        accounting: pl.DataFrame,
        positions: pl.DataFrame,
        *,
        manager: str,
    ) -> pl.DataFrame:
        """Convert accounting and position frames into a portfolio-risk frame.

        Args:
            accounting: Canonical accounting dataset. Must not be mutated.
            positions: Canonical position dataset. Must not be mutated.
            manager: Order manager identity preserved onto every risk row.

        Returns:
            A new DataFrame containing the columns required by the merged
            portfolio-risk schema contract.
        """
        ...


class SimplePortfolioRiskManager:
    """Evaluate deterministic portfolio-level risk decisions.

    Rules (v1):
        - Daily Loss Shutdown: when
          ``daily_total_pnl <= -daily_loss_limit * equity``, set
          ``portfolio_risk_state=SHUTDOWN``, ``allow_new_entries=False``,
          ``shutdown_reason=DAILY_LOSS_LIMIT``, and
          ``cooldown_until=open_time + cooldown_hours``.
        - 24-hour Cooldown: while ``open_time < cooldown_until`` from a prior
          daily-loss shutdown, keep entries blocked with
          ``shutdown_reason=COOLDOWN``.
        - Portfolio Exposure Limit: when
          ``gross_exposure / equity > gross_exposure_limit`` (equity > 0), or
          ``gross_exposure > 0`` when equity is zero, set
          ``portfolio_risk_state=WARNING``, ``allow_new_entries=False``,
          ``shutdown_reason=EXPOSURE_LIMIT``.
        - Net Exposure: recorded from accounting; not enforced in v1.

    Priority (highest first): cooldown, daily loss, exposure, otherwise normal.

    Args:
        daily_loss_limit: Maximum allowed daily loss as a fraction of equity.
        gross_exposure_limit: Maximum allowed gross exposure as a fraction of
            equity.
        cooldown_hours: Hours after a daily-loss shutdown before entries may
            resume.

    Notes:
        Implementations must not mutate the caller-supplied DataFrames.
        Daily PnL fields are taken from the accounting snapshot PnL columns
        (realized / unrealized / total). Drawdown is the running peak-equity
        drawdown computed chronologically across the accounting frame.
    """

    __slots__ = (
        "_cooldown_hours",
        "_daily_loss_limit",
        "_gross_exposure_limit",
    )

    def __init__(
        self,
        *,
        daily_loss_limit: float = DEFAULT_DAILY_LOSS_LIMIT,
        gross_exposure_limit: float = DEFAULT_GROSS_EXPOSURE_LIMIT,
        cooldown_hours: int = DEFAULT_COOLDOWN_HOURS,
    ) -> None:
        """Initialize the manager with configurable portfolio-risk limits.

        Args:
            daily_loss_limit: Daily loss fraction of equity that triggers
                shutdown.
            gross_exposure_limit: Gross exposure fraction of equity that
                triggers a warning.
            cooldown_hours: Cooldown duration after a daily-loss shutdown.

        Raises:
            PortfolioRiskValidationError: If any limit is invalid.
        """
        self._daily_loss_limit = _require_finite_non_negative(
            daily_loss_limit,
            parameter="daily_loss_limit",
        )
        self._gross_exposure_limit = _require_finite_non_negative(
            gross_exposure_limit,
            parameter="gross_exposure_limit",
        )
        self._cooldown_hours = _require_positive_hours(cooldown_hours)

    def evaluate(
        self,
        accounting: pl.DataFrame,
        positions: pl.DataFrame,
        *,
        manager: str,
    ) -> pl.DataFrame:
        """Evaluate accounting snapshots into finalized portfolio-risk rows.

        Args:
            accounting: Canonical accounting dataset. Must not be mutated.
            positions: Canonical position dataset. Must not be mutated.
            manager: Order manager identity stamped onto every risk row.

        Returns:
            A new DataFrame matching ``MERGED_PORTFOLIO_RISK_SCHEMA``.

        Raises:
            PortfolioRiskValidationError: If inputs fail structural validation,
                ``manager`` is blank, required columns are missing, or
                accounting position identities are absent from ``positions``.
        """
        accounting_frame = validate_accounting_frame(accounting)
        positions_frame = validate_position_frame(positions)
        validated_manager = _require_manager(manager)
        _require_accounting_columns(accounting_frame)
        _require_position_columns(positions_frame)
        _require_position_identity_coverage(accounting_frame, positions_frame)
        return _build_portfolio_risk_frame(
            accounting_frame,
            manager=validated_manager,
            daily_loss_limit=self._daily_loss_limit,
            gross_exposure_limit=self._gross_exposure_limit,
            cooldown_hours=self._cooldown_hours,
        )


def validate_accounting_frame(frame: object) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty Polars DataFrame.

    Args:
        frame: Candidate accounting dataset passed to a manager.

    Returns:
        ``frame`` as a DataFrame after structural checks.

    Raises:
        PortfolioRiskValidationError: If ``frame`` is not a Polars DataFrame or
            contains no rows.
    """
    if not isinstance(frame, pl.DataFrame):
        raise PortfolioRiskValidationError(
            "accounting frame must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={"actual_type": type(frame).__name__, "frame": "accounting"},
        )
    if frame.height == 0:
        raise PortfolioRiskValidationError(
            "accounting frame must contain at least one row",
            error_code=_ERROR_FRAME_EMPTY,
            details={"rows": frame.height, "frame": "accounting"},
        )
    return frame


def validate_position_frame(frame: object) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty Polars DataFrame.

    Args:
        frame: Candidate position dataset passed to a manager.

    Returns:
        ``frame`` as a DataFrame after structural checks.

    Raises:
        PortfolioRiskValidationError: If ``frame`` is not a Polars DataFrame or
            contains no rows.
    """
    if not isinstance(frame, pl.DataFrame):
        raise PortfolioRiskValidationError(
            "position frame must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={"actual_type": type(frame).__name__, "frame": "positions"},
        )
    if frame.height == 0:
        raise PortfolioRiskValidationError(
            "position frame must contain at least one row",
            error_code=_ERROR_FRAME_EMPTY,
            details={"rows": frame.height, "frame": "positions"},
        )
    return frame


def _build_portfolio_risk_frame(
    frame: pl.DataFrame,
    *,
    manager: str,
    daily_loss_limit: float,
    gross_exposure_limit: float,
    cooldown_hours: int,
) -> pl.DataFrame:
    """Assemble canonical portfolio-risk rows from a validated accounting frame.

    Args:
        frame: Validated accounting DataFrame.
        manager: Order manager identity for lineage.
        daily_loss_limit: Daily loss fraction of equity.
        gross_exposure_limit: Gross exposure fraction of equity.
        cooldown_hours: Cooldown duration after daily-loss shutdown.

    Returns:
        A new DataFrame ordered and cast to ``MERGED_PORTFOLIO_RISK_SCHEMA``.
    """
    ordered = frame.sort(["open_time", "position_id"], maintain_order=True)

    symbols = ordered["symbol"].to_list()
    timeframes = ordered["timeframe"].to_list()
    open_times = ordered["open_time"].to_list()
    position_ids = ordered["position_id"].to_list()
    equities = ordered["equity"].to_list()
    gross_exposures = ordered["gross_exposure"].to_list()
    net_exposures = ordered["net_exposure"].to_list()
    realized = ordered["realized_pnl"].to_list()
    unrealized = ordered["unrealized_pnl"].to_list()
    totals = ordered["total_pnl"].to_list()
    model_names = ordered["model_name"].to_list()
    model_versions = ordered["model_version"].to_list()
    optimizers = ordered["optimizer"].to_list()
    policies = ordered["policy"].to_list()

    cooldown_delta = timedelta(hours=cooldown_hours)
    active_cooldown_until: datetime | None = None
    peak_equity = 0.0

    out_equity: list[float] = []
    out_gross: list[float] = []
    out_net: list[float] = []
    out_daily_realized: list[float] = []
    out_daily_unrealized: list[float] = []
    out_daily_total: list[float] = []
    out_daily_return: list[float] = []
    out_daily_drawdown: list[float] = []
    out_states: list[str] = []
    out_allow: list[bool] = []
    out_reasons: list[str] = []
    out_cooldown: list[datetime | None] = []

    for index in range(len(open_times)):
        open_time = _as_utc_datetime(open_times[index])
        equity = float(equities[index])
        gross = float(gross_exposures[index])
        net = float(net_exposures[index])
        daily_realized = float(realized[index])
        daily_unrealized = float(unrealized[index])
        daily_total = float(totals[index])

        if equity != 0.0:
            daily_return = daily_total / equity
        else:
            daily_return = 0.0

        if equity > peak_equity:
            peak_equity = equity
        if peak_equity > 0.0:
            daily_drawdown = (peak_equity - equity) / peak_equity
        else:
            daily_drawdown = 0.0
        if daily_drawdown < 0.0:
            daily_drawdown = 0.0

        if active_cooldown_until is not None and open_time >= active_cooldown_until:
            active_cooldown_until = None

        daily_loss_breached = _is_daily_loss_breached(
            daily_total_pnl=daily_total,
            equity=equity,
            daily_loss_limit=daily_loss_limit,
        )
        exposure_breached = _is_exposure_breached(
            gross_exposure=gross,
            equity=equity,
            gross_exposure_limit=gross_exposure_limit,
        )

        if active_cooldown_until is not None and open_time < active_cooldown_until:
            state = PortfolioRiskState.SHUTDOWN.value
            reason = ShutdownReason.COOLDOWN.value
            allow = False
            cooldown_until: datetime | None = active_cooldown_until
        elif daily_loss_breached:
            state = PortfolioRiskState.SHUTDOWN.value
            reason = ShutdownReason.DAILY_LOSS_LIMIT.value
            allow = False
            cooldown_until = open_time + cooldown_delta
            active_cooldown_until = cooldown_until
        elif exposure_breached:
            state = PortfolioRiskState.WARNING.value
            reason = ShutdownReason.EXPOSURE_LIMIT.value
            allow = False
            cooldown_until = None
        else:
            state = PortfolioRiskState.NORMAL.value
            reason = ShutdownReason.NONE.value
            allow = True
            cooldown_until = None

        out_equity.append(equity)
        out_gross.append(gross)
        out_net.append(net)
        out_daily_realized.append(daily_realized)
        out_daily_unrealized.append(daily_unrealized)
        out_daily_total.append(daily_total)
        out_daily_return.append(daily_return)
        out_daily_drawdown.append(daily_drawdown)
        out_states.append(state)
        out_allow.append(allow)
        out_reasons.append(reason)
        out_cooldown.append(cooldown_until)

    assembled = pl.DataFrame(
        {
            "symbol": symbols,
            "timeframe": timeframes,
            "open_time": open_times,
            "manager": [manager] * len(symbols),
            "position_id": position_ids,
            "equity": out_equity,
            "gross_exposure": out_gross,
            "net_exposure": out_net,
            "daily_realized_pnl": out_daily_realized,
            "daily_unrealized_pnl": out_daily_unrealized,
            "daily_total_pnl": out_daily_total,
            "daily_return_pct": out_daily_return,
            "daily_drawdown_pct": out_daily_drawdown,
            "portfolio_risk_state": out_states,
            "allow_new_entries": out_allow,
            "shutdown_reason": out_reasons,
            "cooldown_until": out_cooldown,
            "model_name": model_names,
            "model_version": model_versions,
            "optimizer": optimizers,
            "policy": policies,
        }
    )
    return assembled.select(list(CANONICAL_COLUMN_ORDER)).cast(MERGED_PORTFOLIO_RISK_SCHEMA)


def _is_daily_loss_breached(
    *,
    daily_total_pnl: float,
    equity: float,
    daily_loss_limit: float,
) -> bool:
    """Return whether daily total PnL breaches the daily loss limit."""
    if equity > 0.0:
        return daily_total_pnl <= -daily_loss_limit * equity
    return daily_total_pnl < 0.0


def _is_exposure_breached(
    *,
    gross_exposure: float,
    equity: float,
    gross_exposure_limit: float,
) -> bool:
    """Return whether gross exposure breaches the configured equity fraction."""
    if equity > 0.0:
        return (gross_exposure / equity) > gross_exposure_limit
    return gross_exposure > 0.0


def _as_utc_datetime(value: object) -> datetime:
    """Normalize a timestamp value to a timezone-aware UTC datetime."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    raise PortfolioRiskValidationError(
        "open_time must be a datetime",
        error_code=_ERROR_FRAME_TYPE,
        details={"actual_type": type(value).__name__},
    )


def _require_manager(manager: object) -> str:
    """Validate and return a non-blank manager identity."""
    if not isinstance(manager, str) or manager.strip() == "":
        raise PortfolioRiskValidationError(
            "manager must be a non-blank string",
            error_code=_ERROR_MANAGER_BLANK,
            details={"manager": manager},
        )
    return manager


def _require_accounting_columns(frame: pl.DataFrame) -> None:
    """Raise when any required accounting column is missing."""
    missing = [column for column in ACCOUNTING_INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise PortfolioRiskValidationError(
            "accounting frame is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "missing_columns": tuple(missing),
                "required_columns": ACCOUNTING_INPUT_COLUMNS,
                "available_columns": tuple(frame.columns),
                "frame": "accounting",
            },
        )


def _require_position_columns(frame: pl.DataFrame) -> None:
    """Raise when any required position column is missing."""
    missing = [column for column in POSITION_INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise PortfolioRiskValidationError(
            "position frame is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "missing_columns": tuple(missing),
                "required_columns": POSITION_INPUT_COLUMNS,
                "available_columns": tuple(frame.columns),
                "frame": "positions",
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
        raise PortfolioRiskValidationError(
            "accounting position_id values are missing from positions",
            error_code=_ERROR_POSITION_IDS,
            details={"missing_position_ids": missing},
        )


def _require_finite_non_negative(value: object, *, parameter: str) -> float:
    """Validate and return a finite non-negative float limit."""
    if type(value) is bool:
        raise PortfolioRiskValidationError(
            f"{parameter} must be a finite non-negative number",
            error_code=_ERROR_LIMIT_NON_FINITE,
            details={"parameter": parameter, "value": value},
        )
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise PortfolioRiskValidationError(
            f"{parameter} must be a finite non-negative number",
            error_code=_ERROR_LIMIT_NON_FINITE,
            details={"parameter": parameter, "value": value},
        ) from exc
    if not math.isfinite(number) or number < 0.0:
        raise PortfolioRiskValidationError(
            f"{parameter} must be a finite non-negative number",
            error_code=_ERROR_LIMIT_NON_FINITE,
            details={"parameter": parameter, "value": value},
        )
    return number


def _require_positive_hours(value: object) -> int:
    """Validate and return a positive integer cooldown hour count."""
    if type(value) is bool:
        raise PortfolioRiskValidationError(
            "cooldown_hours must be a positive integer",
            error_code=_ERROR_COOLDOWN_INVALID,
            details={"cooldown_hours": value},
        )
    if not isinstance(value, int) or value <= 0:
        raise PortfolioRiskValidationError(
            "cooldown_hours must be a positive integer",
            error_code=_ERROR_COOLDOWN_INVALID,
            details={"cooldown_hours": value},
        )
    return value
