"""Unit tests for CQROS taker volume features."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.features.exceptions import FeatureExecutionError
from cqros.features.taker import (
    BuyPressureFeature,
    BuySellRatioFeature,
    DeltaVolumeFeature,
    FlowImbalanceFeature,
    SellPressureFeature,
)
from tests.unit.features._helpers import assert_protocol_and_immutability


def _taker() -> pl.DataFrame:
    """Build a deterministic taker-volume fixture."""
    return pl.DataFrame(
        {
            "buy_volume": [60.0, 0.0, 40.0],
            "sell_volume": [40.0, 0.0, 0.0],
        }
    )


def test_buy_pressure() -> None:
    """Buy pressure is buy / (buy + sell); zero total yields null."""
    result = BuyPressureFeature().transform(_taker())
    values = result.get_column("buy_pressure").to_list()
    assert values[0] == pytest.approx(0.6)
    assert values[1] is None
    assert values[2] == pytest.approx(1.0)
    assert BuyPressureFeature().category == "taker"
    assert BuyPressureFeature().lookback == 0


def test_sell_pressure() -> None:
    """Sell pressure is sell / (buy + sell); zero total yields null."""
    result = SellPressureFeature().transform(_taker())
    values = result.get_column("sell_pressure").to_list()
    assert values[0] == pytest.approx(0.4)
    assert values[1] is None
    assert values[2] == pytest.approx(0.0)


def test_buy_sell_ratio() -> None:
    """Buy/sell ratio is buy / sell; zero sell yields null."""
    result = BuySellRatioFeature().transform(_taker())
    values = result.get_column("buy_sell_ratio").to_list()
    assert values[0] == pytest.approx(1.5)
    assert values[1] is None
    assert values[2] is None


def test_flow_imbalance() -> None:
    """Flow imbalance is (buy - sell) / (buy + sell)."""
    result = FlowImbalanceFeature().transform(_taker())
    values = result.get_column("flow_imbalance").to_list()
    assert values[0] == pytest.approx(0.2)
    assert values[1] is None
    assert values[2] == pytest.approx(1.0)


def test_delta_volume() -> None:
    """Delta volume is buy - sell."""
    result = DeltaVolumeFeature().transform(_taker())
    assert result.get_column("delta_volume").to_list() == pytest.approx([20.0, 0.0, 40.0])


@pytest.mark.parametrize(
    ("factory", "error_code"),
    [
        (BuyPressureFeature, "FEATURE-BUY-PRESSURE-001"),
        (SellPressureFeature, "FEATURE-SELL-PRESSURE-001"),
        (BuySellRatioFeature, "FEATURE-BUY-SELL-RATIO-001"),
        (FlowImbalanceFeature, "FEATURE-FLOW-IMBALANCE-001"),
        (DeltaVolumeFeature, "FEATURE-DELTA-VOLUME-001"),
    ],
)
def test_taker_missing_column(factory: type, error_code: str) -> None:
    """Missing buy_volume raises FeatureExecutionError."""
    with pytest.raises(
        FeatureExecutionError,
        match="required column missing: buy_volume",
    ) as exc_info:
        factory().transform(pl.DataFrame({"sell_volume": [1.0, 2.0]}))
    assert exc_info.value.error_code == error_code


def test_taker_immutability_and_exports() -> None:
    """Taker transforms are immutable and package-exported."""
    assert_protocol_and_immutability(
        BuyPressureFeature(),
        output_column="buy_pressure",
        frame=_taker(),
    )
    import cqros.features as features_package
    import cqros.features.taker as taker_package

    for name in (
        "BuyPressureFeature",
        "SellPressureFeature",
        "BuySellRatioFeature",
        "FlowImbalanceFeature",
        "DeltaVolumeFeature",
    ):
        assert name in taker_package.__all__
        assert name in features_package.__all__
