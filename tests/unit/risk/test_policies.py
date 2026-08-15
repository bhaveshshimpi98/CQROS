"""Unit tests for the CQROS Fixed Risk Management policy."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.risk import (
    AlphaDecayPolicy,
    DailyLossPolicy,
    ExposurePolicy,
    FixedRiskPolicy,
    PortfolioRiskPolicy,
    PositionSizingPolicy,
    PyramidingPolicy,
    RiskManager,
    RiskRewardPolicy,
    RiskValidationError,
    TrailingStopPolicy,
)
from cqros.risk.enums import RiskDecision
from cqros.risk.policies import AlphaDecayPolicy as AlphaDecayPolicyDirect
from cqros.risk.policies import DailyLossPolicy as DailyLossPolicyDirect
from cqros.risk.policies import ExposurePolicy as ExposurePolicyDirect
from cqros.risk.policies import FixedRiskPolicy as FixedRiskPolicyDirect
from cqros.risk.policies import (
    PortfolioRiskPolicy as PortfolioRiskPolicyDirect,
)
from cqros.risk.policies import (
    PositionSizingPolicy as PositionSizingPolicyDirect,
)
from cqros.risk.policies import PyramidingPolicy as PyramidingPolicyDirect
from cqros.risk.policies import RiskRewardPolicy as RiskRewardPolicyDirect
from cqros.risk.policies import TrailingStopPolicy as TrailingStopPolicyDirect
from cqros.risk.schema import CANONICAL_COLUMN_ORDER, MERGED_RISK_SCHEMA

_TIMEFRAME = "1h"
_MODEL_NAME = "alpha-lgbm"
_MODEL_VERSION = "1.0.0"
_OPTIMIZER = "equal_weight"
_PLACEHOLDER_REASON = "placeholder_policy"
_FIXED_RISK_POLICY = "fixed_risk"


def _open_time(index: int) -> datetime:
    """Build a deterministic UTC open_time for row ``index``."""
    return datetime(2024, 1, 1, tzinfo=UTC).replace(microsecond=index)


def _portfolio_frame(
    *,
    signals: list[str],
    symbols: list[str] | None = None,
    target_weights: list[float] | None = None,
    optimizer: str = _OPTIMIZER,
) -> pl.DataFrame:
    """Build a canonical portfolio DataFrame for policy tests."""
    row_count = len(signals)
    return pl.DataFrame(
        {
            "symbol": (
                symbols if symbols is not None else [f"SYM{index}" for index in range(row_count)]
            ),
            "timeframe": [_TIMEFRAME] * row_count,
            "open_time": [_open_time(0)] * row_count,
            "model_name": [_MODEL_NAME] * row_count,
            "model_version": [_MODEL_VERSION] * row_count,
            "optimizer": [optimizer] * row_count,
            "signal": signals,
            "target_weight": (target_weights if target_weights is not None else [0.0] * row_count),
        },
        schema={
            "symbol": pl.Utf8,
            "timeframe": pl.Utf8,
            "open_time": pl.Datetime("us", "UTC"),
            "model_name": pl.Utf8,
            "model_version": pl.Utf8,
            "optimizer": pl.Utf8,
            "signal": pl.Utf8,
            "target_weight": pl.Float64,
        },
    )


def _expected_risk_frame(portfolios: pl.DataFrame) -> pl.DataFrame:
    """Build the placeholder Risk Decision frame for ``portfolios``."""
    return (
        portfolios.select(
            "symbol",
            "timeframe",
            "open_time",
            "model_name",
            "model_version",
            "optimizer",
            pl.lit(_FIXED_RISK_POLICY).alias("policy"),
            "signal",
            "target_weight",
            pl.col("target_weight").alias("approved_weight"),
            pl.lit(RiskDecision.APPROVE.value).alias("decision"),
            pl.lit(_PLACEHOLDER_REASON).alias("reason"),
        )
        .select(list(CANONICAL_COLUMN_ORDER))
        .cast(MERGED_RISK_SCHEMA)
    )


class _TrackingPositionSizing(PositionSizingPolicy):
    """Record evaluation order for position sizing."""

    def __init__(self, order: list[str]) -> None:
        super().__init__()
        self._order = order

    def evaluate(self, portfolios: pl.DataFrame) -> pl.DataFrame:
        self._order.append("position_sizing")
        return super().evaluate(portfolios)


class _TrackingRiskReward(RiskRewardPolicy):
    """Record evaluation order for risk/reward."""

    def __init__(self, order: list[str]) -> None:
        super().__init__()
        self._order = order

    def evaluate(self, portfolios: pl.DataFrame) -> pl.DataFrame:
        self._order.append("risk_reward")
        return super().evaluate(portfolios)


class _TrackingDailyLoss(DailyLossPolicy):
    """Record evaluation order for daily loss."""

    def __init__(self, order: list[str]) -> None:
        super().__init__()
        self._order = order

    def evaluate(self, portfolios: pl.DataFrame) -> pl.DataFrame:
        self._order.append("daily_loss")
        return super().evaluate(portfolios)


class _TrackingAlphaDecay(AlphaDecayPolicy):
    """Record evaluation order for alpha decay."""

    def __init__(self, order: list[str]) -> None:
        super().__init__()
        self._order = order

    def evaluate(self, portfolios: pl.DataFrame) -> pl.DataFrame:
        self._order.append("alpha_decay")
        return super().evaluate(portfolios)


class _TrackingTrailingStop(TrailingStopPolicy):
    """Record evaluation order for trailing stop."""

    def __init__(self, order: list[str]) -> None:
        super().__init__()
        self._order = order

    def evaluate(self, portfolios: pl.DataFrame) -> pl.DataFrame:
        self._order.append("trailing_stop")
        return super().evaluate(portfolios)


class _TrackingPyramiding(PyramidingPolicy):
    """Record evaluation order for pyramiding."""

    def __init__(self, order: list[str]) -> None:
        super().__init__()
        self._order = order

    def evaluate(self, portfolios: pl.DataFrame) -> pl.DataFrame:
        self._order.append("pyramiding")
        return super().evaluate(portfolios)


class _TrackingExposure(ExposurePolicy):
    """Record evaluation order for exposure."""

    def __init__(self, order: list[str]) -> None:
        super().__init__()
        self._order = order

    def evaluate(self, portfolios: pl.DataFrame) -> pl.DataFrame:
        self._order.append("exposure")
        return super().evaluate(portfolios)


class _TrackingPortfolioRisk(PortfolioRiskPolicy):
    """Record evaluation order for portfolio risk."""

    def __init__(self, order: list[str]) -> None:
        super().__init__()
        self._order = order

    def evaluate(self, portfolios: pl.DataFrame) -> pl.DataFrame:
        self._order.append("portfolio_risk")
        return super().evaluate(portfolios)


def test_policies_are_exported_from_package() -> None:
    """Package exports match the policies module by identity."""
    assert FixedRiskPolicy is FixedRiskPolicyDirect
    assert PositionSizingPolicy is PositionSizingPolicyDirect
    assert RiskRewardPolicy is RiskRewardPolicyDirect
    assert DailyLossPolicy is DailyLossPolicyDirect
    assert AlphaDecayPolicy is AlphaDecayPolicyDirect
    assert TrailingStopPolicy is TrailingStopPolicyDirect
    assert PyramidingPolicy is PyramidingPolicyDirect
    assert ExposurePolicy is ExposurePolicyDirect
    assert PortfolioRiskPolicy is PortfolioRiskPolicyDirect


def test_fixed_risk_policy_satisfies_protocol() -> None:
    """FixedRiskPolicy structurally satisfies RiskManager."""
    assert isinstance(FixedRiskPolicy(), RiskManager)


def test_fixed_risk_policy_constructs_with_defaults() -> None:
    """Default construction succeeds and evaluates a non-empty frame."""
    frame = _portfolio_frame(signals=["BUY"])
    result = FixedRiskPolicy().evaluate(frame)
    assert_frame_equal(result, _expected_risk_frame(frame))


def test_fixed_risk_policy_dependency_injection() -> None:
    """Injected stage instances are retained and used during evaluation."""
    order: list[str] = []
    position_sizing = _TrackingPositionSizing(order)
    risk_reward = _TrackingRiskReward(order)
    daily_loss = _TrackingDailyLoss(order)
    alpha_decay = _TrackingAlphaDecay(order)
    trailing_stop = _TrackingTrailingStop(order)
    pyramiding = _TrackingPyramiding(order)
    exposure = _TrackingExposure(order)
    portfolio_risk = _TrackingPortfolioRisk(order)

    policy = FixedRiskPolicy(
        position_sizing=position_sizing,
        risk_reward=risk_reward,
        daily_loss=daily_loss,
        alpha_decay=alpha_decay,
        trailing_stop=trailing_stop,
        pyramiding=pyramiding,
        exposure=exposure,
        portfolio_risk=portfolio_risk,
    )
    frame = _portfolio_frame(signals=["BUY", "SELL"])
    result = policy.evaluate(frame)

    assert_frame_equal(result, _expected_risk_frame(frame))
    assert order == [
        "position_sizing",
        "risk_reward",
        "daily_loss",
        "alpha_decay",
        "trailing_stop",
        "pyramiding",
        "exposure",
        "portfolio_risk",
    ]


def test_policy_execution_order_with_partial_injection() -> None:
    """Partial injection still preserves full stage order with defaults."""
    order: list[str] = []
    policy = FixedRiskPolicy(
        position_sizing=_TrackingPositionSizing(order),
        portfolio_risk=_TrackingPortfolioRisk(order),
    )
    policy.evaluate(_portfolio_frame(signals=["HOLD"]))
    assert order[0] == "position_sizing"
    assert order[-1] == "portfolio_risk"
    assert len(order) == 2


def test_invalid_injected_policy_type_raises() -> None:
    """Non-matching injected dependencies raise RiskValidationError."""
    with pytest.raises(RiskValidationError) as exc_info:
        FixedRiskPolicy(position_sizing=object())  # type: ignore[arg-type]

    error = exc_info.value
    assert error.error_code == "RISK_POLICY_TYPE"
    assert error.details["parameter"] == "position_sizing"
    assert error.details["expected_type"] == "PositionSizingPolicy"
    assert error.details["actual_type"] == "object"


def test_placeholder_policies_return_input_unchanged() -> None:
    """Each placeholder stage returns the input DataFrame identity."""
    frame = _portfolio_frame(signals=["BUY"], target_weights=[1.0])
    stages = (
        PositionSizingPolicy(),
        RiskRewardPolicy(),
        DailyLossPolicy(),
        AlphaDecayPolicy(),
        TrailingStopPolicy(),
        PyramidingPolicy(),
        ExposurePolicy(),
        PortfolioRiskPolicy(),
    )
    for stage in stages:
        assert stage.evaluate(frame) is frame


def test_empty_dataframe_raise() -> None:
    """Empty portfolio frames are rejected by shared validation."""
    empty = _portfolio_frame(signals=["BUY"]).clear()
    with pytest.raises(RiskValidationError) as exc_info:
        FixedRiskPolicy().evaluate(empty)
    assert exc_info.value.error_code == "RISK_FRAME_EMPTY"


def test_non_dataframe_input_raise() -> None:
    """Non-DataFrame inputs are rejected by shared validation."""
    with pytest.raises(RiskValidationError) as exc_info:
        FixedRiskPolicy().evaluate([{"signal": "BUY"}])  # type: ignore[arg-type]
    assert exc_info.value.error_code == "RISK_FRAME_TYPE"


def test_input_dataframe_is_not_mutated() -> None:
    """evaluate leaves the caller-supplied DataFrame unchanged."""
    frame = _portfolio_frame(signals=["BUY", "HOLD"], target_weights=[0.5, 0.0])
    original = frame.clone()
    FixedRiskPolicy().evaluate(frame)
    assert_frame_equal(frame, original)


def test_evaluate_produces_placeholder_risk_decision_frame() -> None:
    """Placeholder evaluation assembles the canonical Risk Decision schema."""
    frame = _portfolio_frame(
        signals=["HOLD", "SELL", "BUY"],
        symbols=["AAA", "BBB", "CCC"],
        target_weights=[0.0, -0.5, 0.5],
        optimizer="equal_weight",
    )
    result = FixedRiskPolicy().evaluate(frame)

    assert result.columns == list(CANONICAL_COLUMN_ORDER)
    assert result.schema == MERGED_RISK_SCHEMA
    assert_frame_equal(result, _expected_risk_frame(frame))
    assert result.get_column("optimizer").to_list() == [
        "equal_weight",
        "equal_weight",
        "equal_weight",
    ]
    assert result.get_column("policy").to_list() == [
        _FIXED_RISK_POLICY,
        _FIXED_RISK_POLICY,
        _FIXED_RISK_POLICY,
    ]
    assert result.get_column("approved_weight").to_list() == pytest.approx(
        [0.0, -0.5, 0.5],
    )
    assert result.get_column("decision").to_list() == [
        RiskDecision.APPROVE.value,
        RiskDecision.APPROVE.value,
        RiskDecision.APPROVE.value,
    ]
    assert result.get_column("reason").to_list() == [
        _PLACEHOLDER_REASON,
        _PLACEHOLDER_REASON,
        _PLACEHOLDER_REASON,
    ]


def test_evaluate_preserves_optimizer_lineage() -> None:
    """FixedRiskPolicy preserves optimizer values from the portfolio frame."""
    frame = _portfolio_frame(signals=["BUY", "SELL"], optimizer="kelly")
    result = FixedRiskPolicy().evaluate(frame)
    assert result.get_column("optimizer").to_list() == ["kelly", "kelly"]
    assert result.get_column("policy").to_list() == [
        _FIXED_RISK_POLICY,
        _FIXED_RISK_POLICY,
    ]
