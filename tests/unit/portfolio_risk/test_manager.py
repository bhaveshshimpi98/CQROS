"""Unit tests for CQROS ``SimplePortfolioRiskManager``."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.portfolio_risk import (
    ACCOUNTING_INPUT_COLUMNS,
    CANONICAL_COLUMN_ORDER,
    DEFAULT_COOLDOWN_HOURS,
    DEFAULT_DAILY_LOSS_LIMIT,
    MERGED_PORTFOLIO_RISK_SCHEMA,
    POSITION_INPUT_COLUMNS,
    PortfolioRiskState,
    PortfolioRiskValidationError,
    ShutdownReason,
    SimplePortfolioRiskManager,
    validate_accounting_frame,
    validate_position_frame,
)

_TIMEFRAME = "1h"
_MODEL_NAME = "alpha-lgbm"
_MODEL_VERSION = "1.0.0"
_POLICY = "fixed_risk"
_OPTIMIZER = "equal_weight"
_MANAGER = "simple"
_EQUITY = 1000.0


def _open_time(index: int) -> datetime:
    """Build a deterministic UTC open_time timestamp for row ``index``."""
    return datetime(2024, 1, 1, hour=index, tzinfo=UTC)


def _accounting_frame(
    *,
    symbols: list[str] | None = None,
    position_ids: list[str] | None = None,
    open_times: list[datetime] | None = None,
    equities: list[float] | None = None,
    gross_exposures: list[float] | None = None,
    net_exposures: list[float] | None = None,
    realized: list[float] | None = None,
    unrealized: list[float] | None = None,
    totals: list[float] | None = None,
) -> pl.DataFrame:
    """Build an accounting-shaped frame for portfolio-risk manager tests."""
    symbols = symbols if symbols is not None else ["BTCUSDT"]
    row_count = len(symbols)
    position_ids = (
        position_ids
        if position_ids is not None
        else [f"pos-{index + 1:08d}" for index in range(row_count)]
    )
    open_times = (
        open_times if open_times is not None else [_open_time(index) for index in range(row_count)]
    )
    equities = equities if equities is not None else [_EQUITY] * row_count
    gross_exposures = gross_exposures if gross_exposures is not None else [500.0] * row_count
    net_exposures = net_exposures if net_exposures is not None else [500.0] * row_count
    realized = realized if realized is not None else [0.0] * row_count
    unrealized = unrealized if unrealized is not None else [0.0] * row_count
    totals = totals if totals is not None else [0.0] * row_count
    return pl.DataFrame(
        {
            "symbol": symbols,
            "timeframe": [_TIMEFRAME] * row_count,
            "open_time": open_times,
            "position_id": position_ids,
            "equity": equities,
            "gross_exposure": gross_exposures,
            "net_exposure": net_exposures,
            "realized_pnl": realized,
            "unrealized_pnl": unrealized,
            "total_pnl": totals,
            "model_name": [_MODEL_NAME] * row_count,
            "model_version": [_MODEL_VERSION] * row_count,
            "optimizer": [_OPTIMIZER] * row_count,
            "policy": [_POLICY] * row_count,
        }
    )


def _position_frame(
    *,
    symbols: list[str] | None = None,
    position_ids: list[str] | None = None,
) -> pl.DataFrame:
    """Build a minimal position identity frame matching accounting position_ids."""
    symbols = symbols if symbols is not None else ["BTCUSDT"]
    row_count = len(symbols)
    position_ids = (
        position_ids
        if position_ids is not None
        else [f"pos-{index + 1:08d}" for index in range(row_count)]
    )
    return pl.DataFrame(
        {
            "symbol": symbols,
            "timeframe": [_TIMEFRAME] * row_count,
            "position_id": position_ids,
        }
    )


def test_accounting_and_position_input_columns_contract() -> None:
    """Input column contracts enumerate the columns the manager consumes."""
    for column in (
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
    ):
        assert column in ACCOUNTING_INPUT_COLUMNS
    for column in ("symbol", "timeframe", "position_id"):
        assert column in POSITION_INPUT_COLUMNS


def test_validate_frames_reject_invalid_inputs() -> None:
    """Frame validators reject non-DataFrame and empty frames."""
    with pytest.raises(PortfolioRiskValidationError) as exc_info:
        validate_accounting_frame("not-a-frame")
    assert exc_info.value.error_code == "PRISK_FRAME_TYPE"
    with pytest.raises(PortfolioRiskValidationError) as exc_info:
        validate_accounting_frame(pl.DataFrame({"symbol": []}))
    assert exc_info.value.error_code == "PRISK_FRAME_EMPTY"

    with pytest.raises(PortfolioRiskValidationError) as exc_info:
        validate_position_frame("not-a-frame")
    assert exc_info.value.error_code == "PRISK_FRAME_TYPE"
    with pytest.raises(PortfolioRiskValidationError) as exc_info:
        validate_position_frame(pl.DataFrame({"symbol": []}))
    assert exc_info.value.error_code == "PRISK_FRAME_EMPTY"


def test_evaluate_rejects_empty_and_non_dataframe() -> None:
    """The manager rejects empty datasets and non-DataFrame inputs."""
    manager = SimplePortfolioRiskManager()
    positions = _position_frame()
    with pytest.raises(PortfolioRiskValidationError) as exc_info:
        manager.evaluate(pl.DataFrame({"symbol": []}), positions, manager=_MANAGER)
    assert exc_info.value.error_code == "PRISK_FRAME_EMPTY"
    with pytest.raises(PortfolioRiskValidationError) as exc_info:
        manager.evaluate("nope", positions, manager=_MANAGER)  # type: ignore[arg-type]
    assert exc_info.value.error_code == "PRISK_FRAME_TYPE"
    with pytest.raises(PortfolioRiskValidationError) as exc_info:
        manager.evaluate(_accounting_frame(), pl.DataFrame({"symbol": []}), manager=_MANAGER)
    assert exc_info.value.error_code == "PRISK_FRAME_EMPTY"
    with pytest.raises(PortfolioRiskValidationError) as exc_info:
        manager.evaluate(
            _accounting_frame(),
            "nope",  # type: ignore[arg-type]
            manager=_MANAGER,
        )
    assert exc_info.value.error_code == "PRISK_FRAME_TYPE"


def test_single_row_normal_no_loss_exposure_ok() -> None:
    """A single healthy row evaluates to NORMAL with entries allowed."""
    accounting = _accounting_frame(
        totals=[0.0],
        gross_exposures=[500.0],
        equities=[_EQUITY],
    )
    positions = _position_frame()
    original_accounting = accounting.clone()
    original_positions = positions.clone()
    result = SimplePortfolioRiskManager().evaluate(
        accounting,
        positions,
        manager=_MANAGER,
    )
    assert_frame_equal(accounting, original_accounting)
    assert_frame_equal(positions, original_positions)
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result.schema == MERGED_PORTFOLIO_RISK_SCHEMA
    assert result.height == 1
    assert result["portfolio_risk_state"].to_list() == [PortfolioRiskState.NORMAL.value]
    assert result["allow_new_entries"].to_list() == [True]
    assert result["shutdown_reason"].to_list() == [ShutdownReason.NONE.value]
    assert result["cooldown_until"].to_list() == [None]
    assert result["daily_total_pnl"].to_list() == [0.0]
    assert result["net_exposure"].to_list() == [500.0]


def test_daily_shutdown_boundary_exactly_two_percent() -> None:
    """Exactly -2% of equity triggers DAILY_LOSS_LIMIT shutdown."""
    loss = -DEFAULT_DAILY_LOSS_LIMIT * _EQUITY
    accounting = _accounting_frame(totals=[loss], equities=[_EQUITY])
    result = SimplePortfolioRiskManager().evaluate(
        accounting,
        _position_frame(),
        manager=_MANAGER,
    )
    assert result["portfolio_risk_state"].to_list() == [PortfolioRiskState.SHUTDOWN.value]
    assert result["allow_new_entries"].to_list() == [False]
    assert result["shutdown_reason"].to_list() == [ShutdownReason.DAILY_LOSS_LIMIT.value]
    expected_cooldown = _open_time(0) + timedelta(hours=DEFAULT_COOLDOWN_HOURS)
    assert result["cooldown_until"].to_list() == [expected_cooldown]


def test_no_shutdown_when_above_daily_loss_threshold() -> None:
    """Losses strictly above the -2% threshold do not trigger shutdown."""
    # -19.99 is greater than -20.0, so the daily-loss rule must not fire.
    accounting = _accounting_frame(totals=[-19.99], equities=[_EQUITY])
    result = SimplePortfolioRiskManager().evaluate(
        accounting,
        _position_frame(),
        manager=_MANAGER,
    )
    assert result["portfolio_risk_state"].to_list() == [PortfolioRiskState.NORMAL.value]
    assert result["allow_new_entries"].to_list() == [True]
    assert result["shutdown_reason"].to_list() == [ShutdownReason.NONE.value]


def test_cooldown_active_within_twenty_four_hours() -> None:
    """Rows inside the cooldown window remain SHUTDOWN with COOLDOWN reason."""
    loss = -DEFAULT_DAILY_LOSS_LIMIT * _EQUITY
    open_times = [
        datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
        datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
    ]
    accounting = _accounting_frame(
        symbols=["BTCUSDT", "BTCUSDT"],
        position_ids=["pos-00000001", "pos-00000002"],
        open_times=open_times,
        totals=[loss, 0.0],
    )
    positions = _position_frame(
        symbols=["BTCUSDT", "BTCUSDT"],
        position_ids=["pos-00000001", "pos-00000002"],
    )
    result = SimplePortfolioRiskManager().evaluate(
        accounting,
        positions,
        manager=_MANAGER,
    )
    assert result["portfolio_risk_state"].to_list() == [
        PortfolioRiskState.SHUTDOWN.value,
        PortfolioRiskState.SHUTDOWN.value,
    ]
    assert result["shutdown_reason"].to_list() == [
        ShutdownReason.DAILY_LOSS_LIMIT.value,
        ShutdownReason.COOLDOWN.value,
    ]
    cooldown_until = open_times[0] + timedelta(hours=DEFAULT_COOLDOWN_HOURS)
    assert result["cooldown_until"].to_list() == [cooldown_until, cooldown_until]
    assert result["allow_new_entries"].to_list() == [False, False]


def test_cooldown_expired_after_cooldown_until() -> None:
    """Rows at or after cooldown_until resume normal evaluation."""
    loss = -DEFAULT_DAILY_LOSS_LIMIT * _EQUITY
    open_times = [
        datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
        datetime(2024, 1, 2, 0, 0, tzinfo=UTC),
    ]
    accounting = _accounting_frame(
        symbols=["BTCUSDT", "BTCUSDT"],
        position_ids=["pos-00000001", "pos-00000002"],
        open_times=open_times,
        totals=[loss, 0.0],
        gross_exposures=[500.0, 500.0],
    )
    positions = _position_frame(
        symbols=["BTCUSDT", "BTCUSDT"],
        position_ids=["pos-00000001", "pos-00000002"],
    )
    result = SimplePortfolioRiskManager().evaluate(
        accounting,
        positions,
        manager=_MANAGER,
    )
    assert result["portfolio_risk_state"].to_list() == [
        PortfolioRiskState.SHUTDOWN.value,
        PortfolioRiskState.NORMAL.value,
    ]
    assert result["shutdown_reason"].to_list() == [
        ShutdownReason.DAILY_LOSS_LIMIT.value,
        ShutdownReason.NONE.value,
    ]
    assert result["allow_new_entries"].to_list() == [False, True]
    assert result["cooldown_until"].to_list()[1] is None


def test_exposure_exceeded_produces_warning() -> None:
    """Gross exposure strictly above equity triggers WARNING/EXPOSURE_LIMIT."""
    accounting = _accounting_frame(
        totals=[0.0],
        equities=[_EQUITY],
        gross_exposures=[_EQUITY + 1.0],
    )
    result = SimplePortfolioRiskManager().evaluate(
        accounting,
        _position_frame(),
        manager=_MANAGER,
    )
    assert result["portfolio_risk_state"].to_list() == [PortfolioRiskState.WARNING.value]
    assert result["allow_new_entries"].to_list() == [False]
    assert result["shutdown_reason"].to_list() == [ShutdownReason.EXPOSURE_LIMIT.value]
    assert result["cooldown_until"].to_list() == [None]


def test_normal_exposure_at_limit() -> None:
    """Gross exposure exactly equal to equity remains NORMAL."""
    accounting = _accounting_frame(
        totals=[0.0],
        equities=[_EQUITY],
        gross_exposures=[_EQUITY],
    )
    result = SimplePortfolioRiskManager().evaluate(
        accounting,
        _position_frame(),
        manager=_MANAGER,
    )
    assert result["portfolio_risk_state"].to_list() == [PortfolioRiskState.NORMAL.value]
    assert result["allow_new_entries"].to_list() == [True]
    assert result["shutdown_reason"].to_list() == [ShutdownReason.NONE.value]


def test_multiple_rows_evaluated_chronologically() -> None:
    """Multiple rows are sorted chronologically and evaluated in order."""
    open_times = [_open_time(2), _open_time(0), _open_time(1)]
    accounting = _accounting_frame(
        symbols=["BTCUSDT", "BTCUSDT", "BTCUSDT"],
        position_ids=["pos-00000003", "pos-00000001", "pos-00000002"],
        open_times=open_times,
        totals=[0.0, 0.0, 0.0],
        gross_exposures=[500.0, 500.0, 500.0],
    )
    positions = _position_frame(
        symbols=["BTCUSDT", "BTCUSDT", "BTCUSDT"],
        position_ids=["pos-00000003", "pos-00000001", "pos-00000002"],
    )
    result = SimplePortfolioRiskManager().evaluate(
        accounting,
        positions,
        manager=_MANAGER,
    )
    assert result["open_time"].to_list() == [_open_time(0), _open_time(1), _open_time(2)]
    assert result["position_id"].to_list() == [
        "pos-00000001",
        "pos-00000002",
        "pos-00000003",
    ]
    assert result["portfolio_risk_state"].to_list() == [
        PortfolioRiskState.NORMAL.value,
        PortfolioRiskState.NORMAL.value,
        PortfolioRiskState.NORMAL.value,
    ]


def test_lineage_is_preserved_and_manager_is_stamped() -> None:
    """Lineage metadata is preserved while manager is stamped from the argument."""
    accounting = _accounting_frame()
    result = SimplePortfolioRiskManager().evaluate(
        accounting,
        _position_frame(),
        manager="ledger",
    )
    assert result["manager"].to_list() == ["ledger"]
    assert result["model_name"].to_list() == [_MODEL_NAME]
    assert result["model_version"].to_list() == [_MODEL_VERSION]
    assert result["optimizer"].to_list() == [_OPTIMIZER]
    assert result["policy"].to_list() == [_POLICY]


def test_output_canonical_ordering_and_dtype_schema() -> None:
    """Manager output enforces canonical order and merged-schema dtypes."""
    accounting = _accounting_frame()
    result = SimplePortfolioRiskManager().evaluate(
        accounting,
        _position_frame(),
        manager=_MANAGER,
    )
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result.schema == MERGED_PORTFOLIO_RISK_SCHEMA
    assert result.schema["open_time"] == pl.Datetime("us", "UTC")
    assert result.schema["equity"] == pl.Float64
    assert result.schema["allow_new_entries"] == pl.Boolean


def test_inputs_are_immutable() -> None:
    """evaluate must not mutate caller-supplied accounting or position frames."""
    accounting = _accounting_frame(totals=[-5.0])
    positions = _position_frame()
    accounting_before = accounting.clone()
    positions_before = positions.clone()
    SimplePortfolioRiskManager().evaluate(accounting, positions, manager=_MANAGER)
    assert_frame_equal(accounting, accounting_before)
    assert_frame_equal(positions, positions_before)


def test_missing_position_id_coverage_fails() -> None:
    """Accounting position_ids absent from positions raise validation errors."""
    accounting = _accounting_frame(position_ids=["pos-missing"])
    positions = _position_frame(position_ids=["pos-00000001"])
    with pytest.raises(PortfolioRiskValidationError) as exc_info:
        SimplePortfolioRiskManager().evaluate(accounting, positions, manager=_MANAGER)
    assert exc_info.value.error_code == "PRISK_POSITION_IDS"


def test_empty_datasets_rejected() -> None:
    """Empty accounting or position datasets are rejected."""
    manager = SimplePortfolioRiskManager()
    with pytest.raises(PortfolioRiskValidationError) as exc_info:
        manager.evaluate(
            pl.DataFrame({column: [] for column in ACCOUNTING_INPUT_COLUMNS}),
            _position_frame(),
            manager=_MANAGER,
        )
    assert exc_info.value.error_code == "PRISK_FRAME_EMPTY"
    with pytest.raises(PortfolioRiskValidationError) as exc_info:
        manager.evaluate(
            _accounting_frame(),
            pl.DataFrame({column: [] for column in POSITION_INPUT_COLUMNS}),
            manager=_MANAGER,
        )
    assert exc_info.value.error_code == "PRISK_FRAME_EMPTY"


def test_manager_rejects_blank_manager() -> None:
    """Blank managers raise validation errors."""
    with pytest.raises(PortfolioRiskValidationError) as exc_info:
        SimplePortfolioRiskManager().evaluate(
            _accounting_frame(),
            _position_frame(),
            manager="   ",
        )
    assert exc_info.value.error_code == "PRISK_MANAGER_BLANK"


def test_manager_rejects_missing_accounting_columns() -> None:
    """Missing required accounting columns raise validation errors."""
    accounting = _accounting_frame().drop("total_pnl")
    with pytest.raises(PortfolioRiskValidationError) as exc_info:
        SimplePortfolioRiskManager().evaluate(
            accounting,
            _position_frame(),
            manager=_MANAGER,
        )
    assert exc_info.value.error_code == "PRISK_MISSING_COLUMNS"


def test_net_exposure_is_recorded_only() -> None:
    """Net exposure is copied from accounting and is not used for decisions."""
    accounting = _accounting_frame(
        totals=[0.0],
        equities=[_EQUITY],
        gross_exposures=[500.0],
        net_exposures=[250.0],
    )
    result = SimplePortfolioRiskManager().evaluate(
        accounting,
        _position_frame(),
        manager=_MANAGER,
    )
    assert result["net_exposure"].to_list() == [250.0]
    assert result["portfolio_risk_state"].to_list() == [PortfolioRiskState.NORMAL.value]
