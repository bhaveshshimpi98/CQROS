"""Unit tests for CQROS Batch-5 market microstructure factors."""

from __future__ import annotations

import math

import polars as pl
import pytest

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError
from cqros.factors.interfaces import Factor
from cqros.factors.microstructure import (
    AggressiveBuyRatioFactor,
    AggressiveSellRatioFactor,
    BuySellImbalanceFactor,
    MicroPricePressureFactor,
    OrderFlowMomentumFactor,
    SignedVolumeFactor,
    TradeImbalanceFactor,
    TradeIntensityFactor,
    VWAPDistanceFactor,
    VWAPZScoreFactor,
)
from cqros.factors.microstructure.aggressive_buy_ratio import (
    AggressiveBuyRatioFactor as AggressiveBuyRatioFactorDirect,
)
from cqros.factors.microstructure.aggressive_sell_ratio import (
    AggressiveSellRatioFactor as AggressiveSellRatioFactorDirect,
)
from cqros.factors.microstructure.buy_sell_imbalance import (
    BuySellImbalanceFactor as BuySellImbalanceFactorDirect,
)
from cqros.factors.microstructure.micro_price_pressure import (
    MicroPricePressureFactor as MicroPricePressureFactorDirect,
)
from cqros.factors.microstructure.order_flow_momentum import (
    OrderFlowMomentumFactor as OrderFlowMomentumFactorDirect,
)
from cqros.factors.microstructure.signed_volume import (
    SignedVolumeFactor as SignedVolumeFactorDirect,
)
from cqros.factors.microstructure.trade_imbalance import (
    TradeImbalanceFactor as TradeImbalanceFactorDirect,
)
from cqros.factors.microstructure.trade_intensity import (
    TradeIntensityFactor as TradeIntensityFactorDirect,
)
from cqros.factors.microstructure.vwap_distance import (
    VWAPDistanceFactor as VWAPDistanceFactorDirect,
)
from cqros.factors.microstructure.vwap_zscore import (
    VWAPZScoreFactor as VWAPZScoreFactorDirect,
)


def _taker_frame() -> pl.DataFrame:
    """Return a deterministic taker-volume fixture."""
    return pl.DataFrame(
        {
            "taker_buy_volume": [60.0, 80.0, 40.0, 90.0, 70.0, 100.0],
            "taker_sell_volume": [40.0, 20.0, 60.0, 30.0, 50.0, 40.0],
            "volume": [100.0, 100.0, 100.0, 120.0, 120.0, 140.0],
        }
    )


def _vwap_frame() -> pl.DataFrame:
    """Return a deterministic close/vwap fixture."""
    return pl.DataFrame(
        {
            "close": [100.0, 102.0, 101.0, 105.0, 104.0, 108.0],
            "vwap": [99.0, 100.0, 101.0, 103.0, 104.0, 106.0],
        }
    )


def _assert_protocol_and_immutability(
    factor: BaseFactor,
    *,
    output_column: str,
    frame: pl.DataFrame,
) -> None:
    """Assert Factor protocol conformance and compute immutability."""
    assert isinstance(factor, BaseFactor)
    assert isinstance(factor, Factor)
    original_columns = list(frame.columns)
    probe_column = original_columns[0]
    original_values = frame.get_column(probe_column).to_list()
    result = factor.compute(frame)
    assert list(frame.columns) == original_columns
    assert frame.get_column(probe_column).to_list() == original_values
    assert output_column not in frame.columns
    assert output_column in result.columns
    assert result.height == frame.height
    assert result is not frame
    assert result.columns == [*original_columns, output_column]
    assert result.schema[output_column] == pl.Float64


def _assert_missing_column(
    factor: BaseFactor,
    *,
    missing: str,
    error_code: str,
    frame: pl.DataFrame,
) -> None:
    """Assert missing required columns raise FactorError."""
    with pytest.raises(FactorError, match=f"required column missing: {missing}") as exc_info:
        factor.compute(frame)
    error = exc_info.value
    assert error.error_code == error_code
    assert error.details["factor"] == factor.name
    assert error.details["required_column"] == missing


def _assert_empty_and_single_row(
    factor: BaseFactor,
    *,
    output_column: str,
    columns: tuple[str, ...],
    single_expected: float | None = None,
) -> None:
    """Assert empty and single-row frames produce null-safe outputs."""
    empty = pl.DataFrame({name: pl.Series(name, [], dtype=pl.Float64) for name in columns})
    empty_result = factor.compute(empty)
    assert empty_result.height == 0
    assert output_column in empty_result.columns
    assert empty_result.schema[output_column] == pl.Float64

    single = pl.DataFrame({name: [10.0] for name in columns})
    single_result = factor.compute(single)
    assert single_result.height == 1
    assert single_result.get_column(output_column).to_list() == [single_expected]


def _assert_determinism(
    factory: object,
    *,
    output_column: str,
    frame: pl.DataFrame,
) -> None:
    """Assert identical inputs produce identical outputs."""
    assert callable(factory)
    first = factory().compute(frame).get_column(output_column).to_list()
    second = factory().compute(frame).get_column(output_column).to_list()
    assert first == second


def _population_zscore(values: list[float]) -> float | None:
    """Return population z-score for a fully observed window."""
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    if std == 0.0:
        return 0.0
    return (values[-1] - mean) / std


# ---------------------------------------------------------------------------
# BuySellImbalanceFactor
# ---------------------------------------------------------------------------


def test_buy_sell_imbalance_metadata_and_math() -> None:
    """Buy/sell imbalance metadata and point formula are correct."""
    factor = BuySellImbalanceFactor()
    assert factor.name == "buy_sell_imbalance"
    assert factor.version == "1.0.0"
    assert factor.category == "microstructure"
    assert factor.lookback == 0
    assert factor.required_features == ("taker_buy_volume", "taker_sell_volume")
    assert factor.produced_columns == ("buy_sell_imbalance",)
    assert BuySellImbalanceFactor is BuySellImbalanceFactorDirect

    frame = pl.DataFrame(
        {
            "taker_buy_volume": [60.0, 0.0, 50.0],
            "taker_sell_volume": [40.0, 0.0, 50.0],
        }
    )
    values = factor.compute(frame).get_column("buy_sell_imbalance").to_list()
    assert values[0] == pytest.approx(0.2)
    assert values[1] is None
    assert values[2] == pytest.approx(0.0)


def test_buy_sell_imbalance_validation_and_edges() -> None:
    """Buy/sell imbalance covers validation, nulls, and immutability."""
    with pytest.raises(ValidationError, match="lookback must be 0"):
        BuySellImbalanceFactor(lookback=1)

    null_frame = pl.DataFrame(
        {
            "taker_buy_volume": [60.0, None, 40.0],
            "taker_sell_volume": [40.0, 20.0, 60.0],
        }
    )
    null_values = (
        BuySellImbalanceFactor().compute(null_frame).get_column("buy_sell_imbalance").to_list()
    )
    assert null_values[1] is None

    rising = BuySellImbalanceFactor().compute(
        pl.DataFrame(
            {
                "taker_buy_volume": [10.0, 30.0, 50.0],
                "taker_sell_volume": [50.0, 30.0, 10.0],
            }
        )
    )
    assert rising.get_column("buy_sell_imbalance").to_list()[0] < 0.0
    assert rising.get_column("buy_sell_imbalance").to_list()[2] > 0.0

    factor = BuySellImbalanceFactor()
    _assert_missing_column(
        factor,
        missing="taker_buy_volume",
        error_code="FACTOR-BUY-SELL-IMBALANCE-002",
        frame=pl.DataFrame({"taker_sell_volume": [1.0]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="buy_sell_imbalance",
        frame=_taker_frame().select(["taker_buy_volume", "taker_sell_volume"]),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="buy_sell_imbalance",
        columns=("taker_buy_volume", "taker_sell_volume"),
        single_expected=0.0,
    )
    _assert_determinism(
        BuySellImbalanceFactor,
        output_column="buy_sell_imbalance",
        frame=_taker_frame().select(["taker_buy_volume", "taker_sell_volume"]),
    )


# ---------------------------------------------------------------------------
# AggressiveBuyRatioFactor / AggressiveSellRatioFactor
# ---------------------------------------------------------------------------


def test_aggressive_buy_ratio_metadata_and_math() -> None:
    """Aggressive buy ratio matches rolling buy over rolling volume."""
    factor = AggressiveBuyRatioFactor()
    assert factor.name == "aggressive_buy_ratio"
    assert factor.lookback == 20
    assert factor.category == "microstructure"
    assert AggressiveBuyRatioFactor is AggressiveBuyRatioFactorDirect

    frame = pl.DataFrame(
        {
            "taker_buy_volume": [10.0, 20.0, 30.0],
            "volume": [100.0, 100.0, 100.0],
        }
    )
    values = (
        AggressiveBuyRatioFactor(lookback=2)
        .compute(frame)
        .get_column("aggressive_buy_ratio")
        .to_list()
    )
    assert values[0] is None
    assert values[1] == pytest.approx(30.0 / 200.0)
    assert values[2] == pytest.approx(50.0 / 200.0)


def test_aggressive_sell_ratio_and_shared_edges() -> None:
    """Aggressive sell ratio math plus shared validation/edge contracts."""
    factor = AggressiveSellRatioFactor()
    assert factor.name == "aggressive_sell_ratio"
    assert factor.lookback == 20
    assert AggressiveSellRatioFactor is AggressiveSellRatioFactorDirect

    frame = pl.DataFrame(
        {
            "taker_sell_volume": [10.0, 20.0, 30.0],
            "volume": [100.0, 100.0, 100.0],
        }
    )
    values = (
        AggressiveSellRatioFactor(lookback=2)
        .compute(frame)
        .get_column("aggressive_sell_ratio")
        .to_list()
    )
    assert values[0] is None
    assert values[1] == pytest.approx(30.0 / 200.0)

    with pytest.raises(ValidationError, match="lookback must be an integer greater than 0"):
        AggressiveBuyRatioFactor(lookback=0)
    with pytest.raises(ValidationError, match="lookback must be an integer greater than 0"):
        AggressiveSellRatioFactor(lookback=0)

    zero_volume = (
        AggressiveBuyRatioFactor(lookback=2)
        .compute(
            pl.DataFrame(
                {
                    "taker_buy_volume": [0.0, 0.0],
                    "volume": [0.0, 0.0],
                }
            )
        )
        .get_column("aggressive_buy_ratio")
        .to_list()
    )
    assert zero_volume[1] is None

    buy_factor = AggressiveBuyRatioFactor(lookback=2)
    _assert_protocol_and_immutability(
        buy_factor,
        output_column="aggressive_buy_ratio",
        frame=_taker_frame().select(["taker_buy_volume", "volume"]),
    )
    _assert_empty_and_single_row(
        buy_factor,
        output_column="aggressive_buy_ratio",
        columns=("taker_buy_volume", "volume"),
        single_expected=None,
    )
    sell_factor = AggressiveSellRatioFactor(lookback=2)
    _assert_missing_column(
        sell_factor,
        missing="volume",
        error_code="FACTOR-AGGRESSIVE-SELL-RATIO-002",
        frame=pl.DataFrame({"taker_sell_volume": [1.0]}),
    )
    _assert_determinism(
        lambda: AggressiveSellRatioFactor(lookback=2),
        output_column="aggressive_sell_ratio",
        frame=_taker_frame().select(["taker_sell_volume", "volume"]),
    )


# ---------------------------------------------------------------------------
# TradeImbalanceFactor
# ---------------------------------------------------------------------------


def test_trade_imbalance_metadata_and_math() -> None:
    """Trade imbalance matches rolling count imbalance ratio."""
    factor = TradeImbalanceFactor()
    assert factor.name == "trade_imbalance"
    assert factor.lookback == 20
    assert factor.required_features == ("buy_trade_count", "sell_trade_count")
    assert TradeImbalanceFactor is TradeImbalanceFactorDirect

    frame = pl.DataFrame(
        {
            "buy_trade_count": [5.0, 7.0, 3.0],
            "sell_trade_count": [5.0, 3.0, 7.0],
        }
    )
    values = TradeImbalanceFactor(lookback=2).compute(frame).get_column("trade_imbalance").to_list()
    assert values[0] is None
    assert values[1] == pytest.approx(((5.0 + 7.0) - (5.0 + 3.0)) / (5.0 + 7.0 + 5.0 + 3.0))
    assert values[2] == pytest.approx(((7.0 + 3.0) - (3.0 + 7.0)) / (7.0 + 3.0 + 3.0 + 7.0))


def test_trade_imbalance_validation_and_edges() -> None:
    """Trade imbalance covers zero totals, validation, and schema contracts."""
    with pytest.raises(ValidationError, match="lookback must be an integer greater than 0"):
        TradeImbalanceFactor(lookback=0)

    zero = (
        TradeImbalanceFactor(lookback=2)
        .compute(
            pl.DataFrame(
                {
                    "buy_trade_count": [0.0, 0.0],
                    "sell_trade_count": [0.0, 0.0],
                }
            )
        )
        .get_column("trade_imbalance")
        .to_list()
    )
    assert zero[1] is None

    rising = (
        TradeImbalanceFactor(lookback=2)
        .compute(
            pl.DataFrame(
                {
                    "buy_trade_count": [1.0, 2.0, 8.0],
                    "sell_trade_count": [8.0, 2.0, 1.0],
                }
            )
        )
        .get_column("trade_imbalance")
        .to_list()
    )
    assert rising[2] > 0.0

    factor = TradeImbalanceFactor(lookback=2)
    _assert_protocol_and_immutability(
        factor,
        output_column="trade_imbalance",
        frame=pl.DataFrame(
            {
                "buy_trade_count": [1.0, 2.0, 3.0],
                "sell_trade_count": [3.0, 2.0, 1.0],
            }
        ),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="trade_imbalance",
        columns=("buy_trade_count", "sell_trade_count"),
        single_expected=None,
    )


# ---------------------------------------------------------------------------
# SignedVolumeFactor / OrderFlowMomentumFactor
# ---------------------------------------------------------------------------


def test_signed_volume_metadata_and_math() -> None:
    """Signed volume matches rolling sum of buy minus sell."""
    factor = SignedVolumeFactor()
    assert factor.name == "signed_volume"
    assert factor.lookback == 20
    assert SignedVolumeFactor is SignedVolumeFactorDirect

    frame = pl.DataFrame(
        {
            "taker_buy_volume": [10.0, 20.0, 15.0],
            "taker_sell_volume": [5.0, 8.0, 20.0],
        }
    )
    values = SignedVolumeFactor(lookback=2).compute(frame).get_column("signed_volume").to_list()
    assert values[0] is None
    assert values[1] == pytest.approx((10.0 - 5.0) + (20.0 - 8.0))
    assert values[2] == pytest.approx((20.0 - 8.0) + (15.0 - 20.0))


def test_order_flow_momentum_metadata_and_math() -> None:
    """Order-flow momentum matches absolute change in signed volume."""
    factor = OrderFlowMomentumFactor()
    assert factor.name == "order_flow_momentum"
    assert factor.lookback == 20
    assert OrderFlowMomentumFactor is OrderFlowMomentumFactorDirect

    frame = pl.DataFrame(
        {
            "taker_buy_volume": [10.0, 20.0, 15.0, 25.0],
            "taker_sell_volume": [5.0, 8.0, 20.0, 10.0],
        }
    )
    signed = [5.0, 12.0, -5.0, 15.0]
    values = (
        OrderFlowMomentumFactor(lookback=2)
        .compute(frame)
        .get_column("order_flow_momentum")
        .to_list()
    )
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx(signed[2] - signed[0])
    assert values[3] == pytest.approx(signed[3] - signed[1])


def test_signed_volume_and_order_flow_edges() -> None:
    """Signed volume and order-flow momentum validation and edge contracts."""
    with pytest.raises(ValidationError, match="lookback must be an integer greater than 0"):
        SignedVolumeFactor(lookback=0)
    with pytest.raises(ValidationError, match="lookback must be an integer greater than 0"):
        OrderFlowMomentumFactor(lookback=0)

    constant = (
        SignedVolumeFactor(lookback=2)
        .compute(
            pl.DataFrame(
                {
                    "taker_buy_volume": [10.0, 10.0, 10.0],
                    "taker_sell_volume": [10.0, 10.0, 10.0],
                }
            )
        )
        .get_column("signed_volume")
        .to_list()
    )
    assert constant[2] == pytest.approx(0.0)

    signed_factor = SignedVolumeFactor(lookback=2)
    _assert_protocol_and_immutability(
        signed_factor,
        output_column="signed_volume",
        frame=_taker_frame().select(["taker_buy_volume", "taker_sell_volume"]),
    )
    _assert_empty_and_single_row(
        signed_factor,
        output_column="signed_volume",
        columns=("taker_buy_volume", "taker_sell_volume"),
        single_expected=None,
    )
    momentum_factor = OrderFlowMomentumFactor(lookback=2)
    _assert_empty_and_single_row(
        momentum_factor,
        output_column="order_flow_momentum",
        columns=("taker_buy_volume", "taker_sell_volume"),
        single_expected=None,
    )
    _assert_determinism(
        lambda: OrderFlowMomentumFactor(lookback=2),
        output_column="order_flow_momentum",
        frame=_taker_frame().select(["taker_buy_volume", "taker_sell_volume"]),
    )


# ---------------------------------------------------------------------------
# MicroPricePressure / VWAPDistance / VWAPZScore
# ---------------------------------------------------------------------------


def test_micro_price_pressure_metadata_and_math() -> None:
    """Micro-price pressure matches rolling mean of relative close-VWAP gap."""
    factor = MicroPricePressureFactor()
    assert factor.name == "micro_price_pressure"
    assert factor.lookback == 20
    assert MicroPricePressureFactor is MicroPricePressureFactorDirect

    frame = pl.DataFrame(
        {
            "close": [110.0, 100.0, 120.0],
            "vwap": [100.0, 100.0, 100.0],
        }
    )
    relatives = [0.10, 0.0, 0.20]
    values = (
        MicroPricePressureFactor(lookback=2)
        .compute(frame)
        .get_column("micro_price_pressure")
        .to_list()
    )
    assert values[0] is None
    assert values[1] == pytest.approx((relatives[0] + relatives[1]) / 2.0)
    assert values[2] == pytest.approx((relatives[1] + relatives[2]) / 2.0)


def test_vwap_distance_metadata_and_math() -> None:
    """VWAP distance matches close versus rolling-mean VWAP."""
    factor = VWAPDistanceFactor()
    assert factor.name == "vwap_distance"
    assert factor.lookback == 20
    assert VWAPDistanceFactor is VWAPDistanceFactorDirect

    frame = pl.DataFrame(
        {
            "close": [100.0, 110.0, 120.0],
            "vwap": [100.0, 100.0, 100.0],
        }
    )
    values = VWAPDistanceFactor(lookback=2).compute(frame).get_column("vwap_distance").to_list()
    assert values[0] is None
    assert values[1] == pytest.approx((110.0 - 100.0) / 100.0)
    assert values[2] == pytest.approx((120.0 - 100.0) / 100.0)


def test_vwap_zscore_metadata_and_math() -> None:
    """VWAP z-score matches population z-score of close-VWAP basis."""
    factor = VWAPZScoreFactor()
    assert factor.name == "vwap_zscore"
    assert factor.lookback == 20
    assert VWAPZScoreFactor is VWAPZScoreFactorDirect

    frame = pl.DataFrame(
        {
            "close": [101.0, 103.0, 102.0, 106.0],
            "vwap": [100.0, 100.0, 100.0, 100.0],
        }
    )
    basis = [1.0, 3.0, 2.0, 6.0]
    values = VWAPZScoreFactor(lookback=3).compute(frame).get_column("vwap_zscore").to_list()
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx(_population_zscore(basis[0:3]))
    assert values[3] == pytest.approx(_population_zscore(basis[1:4]))


def test_vwap_family_validation_and_edges() -> None:
    """VWAP-family factors cover validation, constants, and schema contracts."""
    with pytest.raises(ValidationError, match="lookback must be an integer greater than 0"):
        MicroPricePressureFactor(lookback=0)
    with pytest.raises(
        ValidationError,
        match="lookback must be an integer greater than or equal to 2",
    ):
        VWAPDistanceFactor(lookback=1)
    with pytest.raises(
        ValidationError,
        match="lookback must be an integer greater than or equal to 2",
    ):
        VWAPZScoreFactor(lookback=1)

    zero_vwap = (
        MicroPricePressureFactor(lookback=2)
        .compute(pl.DataFrame({"close": [10.0, 11.0], "vwap": [0.0, 0.0]}))
        .get_column("micro_price_pressure")
        .to_list()
    )
    assert zero_vwap[1] is None

    constant_z = (
        VWAPZScoreFactor(lookback=3)
        .compute(
            pl.DataFrame(
                {
                    "close": [100.0, 100.0, 100.0, 100.0],
                    "vwap": [100.0, 100.0, 100.0, 100.0],
                }
            )
        )
        .get_column("vwap_zscore")
        .to_list()
    )
    assert constant_z[0] is None
    assert constant_z[1] is None
    assert constant_z[2] == pytest.approx(0.0)
    assert constant_z[3] == pytest.approx(0.0)

    rising = (
        VWAPDistanceFactor(lookback=2)
        .compute(
            pl.DataFrame(
                {
                    "close": [100.0, 110.0, 120.0],
                    "vwap": [100.0, 100.0, 100.0],
                }
            )
        )
        .get_column("vwap_distance")
        .to_list()
    )
    falling = (
        VWAPDistanceFactor(lookback=2)
        .compute(
            pl.DataFrame(
                {
                    "close": [100.0, 90.0, 80.0],
                    "vwap": [100.0, 100.0, 100.0],
                }
            )
        )
        .get_column("vwap_distance")
        .to_list()
    )
    assert rising[2] > 0.0
    assert falling[2] < 0.0

    pressure = MicroPricePressureFactor(lookback=2)
    _assert_protocol_and_immutability(
        pressure,
        output_column="micro_price_pressure",
        frame=_vwap_frame(),
    )
    _assert_empty_and_single_row(
        pressure,
        output_column="micro_price_pressure",
        columns=("close", "vwap"),
        single_expected=None,
    )
    distance = VWAPDistanceFactor(lookback=2)
    _assert_empty_and_single_row(
        distance,
        output_column="vwap_distance",
        columns=("close", "vwap"),
        single_expected=None,
    )
    zscore = VWAPZScoreFactor(lookback=2)
    _assert_missing_column(
        zscore,
        missing="vwap",
        error_code="FACTOR-VWAP-ZSCORE-002",
        frame=pl.DataFrame({"close": [1.0, 2.0]}),
    )
    _assert_determinism(
        lambda: VWAPZScoreFactor(lookback=3),
        output_column="vwap_zscore",
        frame=_vwap_frame(),
    )


# ---------------------------------------------------------------------------
# TradeIntensityFactor
# ---------------------------------------------------------------------------


def test_trade_intensity_metadata_and_math() -> None:
    """Trade intensity matches trade_count over rolling mean."""
    factor = TradeIntensityFactor()
    assert factor.name == "trade_intensity"
    assert factor.lookback == 20
    assert factor.required_features == ("trade_count",)
    assert TradeIntensityFactor is TradeIntensityFactorDirect

    frame = pl.DataFrame({"trade_count": [10.0, 20.0, 30.0, 40.0]})
    values = TradeIntensityFactor(lookback=2).compute(frame).get_column("trade_intensity").to_list()
    assert values[0] is None
    assert values[1] == pytest.approx(20.0 / 15.0)
    assert values[2] == pytest.approx(30.0 / 25.0)
    assert values[3] == pytest.approx(40.0 / 35.0)


def test_trade_intensity_validation_and_edges() -> None:
    """Trade intensity covers trends, zero mean, validation, and schema."""
    with pytest.raises(ValidationError, match="lookback must be an integer greater than 0"):
        TradeIntensityFactor(lookback=0)

    zero_mean = (
        TradeIntensityFactor(lookback=2)
        .compute(pl.DataFrame({"trade_count": [0.0, 0.0, 5.0]}))
        .get_column("trade_intensity")
        .to_list()
    )
    assert zero_mean[1] is None

    rising = (
        TradeIntensityFactor(lookback=3)
        .compute(pl.DataFrame({"trade_count": [1.0, 2.0, 3.0, 4.0]}))
        .get_column("trade_intensity")
        .to_list()
    )
    falling = (
        TradeIntensityFactor(lookback=3)
        .compute(pl.DataFrame({"trade_count": [4.0, 3.0, 2.0, 1.0]}))
        .get_column("trade_intensity")
        .to_list()
    )
    constant = (
        TradeIntensityFactor(lookback=2)
        .compute(pl.DataFrame({"trade_count": [5.0, 5.0, 5.0]}))
        .get_column("trade_intensity")
        .to_list()
    )
    assert rising[3] > 1.0
    assert falling[3] < 1.0
    assert constant[2] == pytest.approx(1.0)

    null_values = (
        TradeIntensityFactor(lookback=2)
        .compute(pl.DataFrame({"trade_count": [10.0, None, 12.0]}))
        .get_column("trade_intensity")
        .to_list()
    )
    assert null_values[1] is None
    assert null_values[2] is None

    factor = TradeIntensityFactor(lookback=2)
    _assert_missing_column(
        factor,
        missing="trade_count",
        error_code="FACTOR-TRADE-INTENSITY-002",
        frame=pl.DataFrame({"volume": [1.0, 2.0]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="trade_intensity",
        frame=pl.DataFrame({"trade_count": [1.0, 2.0, 3.0]}),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="trade_intensity",
        columns=("trade_count",),
        single_expected=None,
    )
    _assert_determinism(
        lambda: TradeIntensityFactor(lookback=2),
        output_column="trade_intensity",
        frame=pl.DataFrame({"trade_count": [10.0, 20.0, 30.0, 40.0]}),
    )
    large = TradeIntensityFactor(lookback=50).compute(
        pl.DataFrame({"trade_count": [float(i) for i in range(1, 11)]})
    )
    assert all(value is None for value in large.get_column("trade_intensity").to_list())


def test_batch5_package_exports() -> None:
    """Microstructure package exports all ten Batch-5 factors."""
    exported = {
        AggressiveBuyRatioFactor,
        AggressiveSellRatioFactor,
        BuySellImbalanceFactor,
        MicroPricePressureFactor,
        OrderFlowMomentumFactor,
        SignedVolumeFactor,
        TradeImbalanceFactor,
        TradeIntensityFactor,
        VWAPDistanceFactor,
        VWAPZScoreFactor,
    }
    assert len(exported) == 10
