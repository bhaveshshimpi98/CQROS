"""Unit tests for CQROS Batch-4 volume and liquidity factors."""

from __future__ import annotations

import math

import polars as pl
import pytest

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError
from cqros.factors.interfaces import Factor
from cqros.factors.volume import (
    AccumulationDistributionFactor,
    ChaikinMoneyFlowFactor,
    EaseOfMovementFactor,
    MoneyFlowIndexFactor,
    OnBalanceVolumeFactor,
    PriceVolumeTrendFactor,
    RelativeVolumeFactor,
    VolumeRateOfChangeFactor,
    VolumeTrendFactor,
    VolumeZScoreFactor,
)
from cqros.factors.volume.accumulation_distribution import (
    AccumulationDistributionFactor as AccumulationDistributionFactorDirect,
)
from cqros.factors.volume.chaikin_money_flow import (
    ChaikinMoneyFlowFactor as ChaikinMoneyFlowFactorDirect,
)
from cqros.factors.volume.ease_of_movement import (
    EaseOfMovementFactor as EaseOfMovementFactorDirect,
)
from cqros.factors.volume.money_flow_index import (
    MoneyFlowIndexFactor as MoneyFlowIndexFactorDirect,
)
from cqros.factors.volume.on_balance_volume import (
    OnBalanceVolumeFactor as OnBalanceVolumeFactorDirect,
)
from cqros.factors.volume.price_volume_trend import (
    PriceVolumeTrendFactor as PriceVolumeTrendFactorDirect,
)
from cqros.factors.volume.relative_volume import (
    RelativeVolumeFactor as RelativeVolumeFactorDirect,
)
from cqros.factors.volume.volume_rate_of_change import (
    VolumeRateOfChangeFactor as VolumeRateOfChangeFactorDirect,
)
from cqros.factors.volume.volume_trend import VolumeTrendFactor as VolumeTrendFactorDirect
from cqros.factors.volume.volume_zscore import VolumeZScoreFactor as VolumeZScoreFactorDirect


def _ohlcv_frame() -> pl.DataFrame:
    """Return a deterministic OHLCV fixture used across Batch-4 tests."""
    return pl.DataFrame(
        {
            "high": [11.0, 12.0, 13.0, 12.5, 14.0, 15.0, 14.5, 16.0],
            "low": [9.0, 10.0, 11.0, 10.5, 12.0, 13.0, 12.5, 14.0],
            "close": [10.0, 11.5, 12.0, 11.0, 13.5, 14.0, 13.0, 15.5],
            "volume": [100.0, 150.0, 120.0, 180.0, 200.0, 160.0, 140.0, 220.0],
        }
    )


def _volume_frame(volumes: list[float | None]) -> pl.DataFrame:
    """Return a volume-only DataFrame."""
    return pl.DataFrame({"volume": volumes})


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


def _money_flow_multiplier(high: float, low: float, close: float) -> float:
    """Return classic money-flow multiplier for one bar."""
    if high == low:
        return 0.0
    return ((close - low) - (high - close)) / (high - low)


def _population_zscore(values: list[float]) -> float | None:
    """Return population z-score for a fully observed window."""
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    if std == 0.0:
        return 0.0
    return (values[-1] - mean) / std


def _ols_slope(values: list[float]) -> float:
    """Return OLS slope of values against 0..n-1."""
    n = len(values)
    xs = list(range(n))
    sum_x = sum(xs)
    sum_y = sum(values)
    sum_xy = sum(x * y for x, y in zip(xs, values, strict=True))
    sum_x2 = sum(x * x for x in xs)
    return (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)


# ---------------------------------------------------------------------------
# OnBalanceVolumeFactor
# ---------------------------------------------------------------------------


def test_on_balance_volume_metadata() -> None:
    """OnBalanceVolumeFactor exposes the fixed production metadata contract."""
    factor = OnBalanceVolumeFactor()
    assert factor.name == "on_balance_volume"
    assert factor.version == "1.0.0"
    assert factor.category == "volume"
    assert factor.required_features == ("close", "volume")
    assert factor.produced_columns == ("on_balance_volume",)
    assert factor.lookback == 0
    assert factor.metadata.name == "on_balance_volume"
    assert OnBalanceVolumeFactor is OnBalanceVolumeFactorDirect


def test_on_balance_volume_correctness_and_warmup() -> None:
    """OBV matches cumulative signed volume and nulls the first row."""
    frame = pl.DataFrame(
        {
            "close": [10.0, 11.0, 10.5, 10.5, 12.0],
            "volume": [100.0, 200.0, 150.0, 120.0, 300.0],
        }
    )
    values = OnBalanceVolumeFactor().compute(frame).get_column("on_balance_volume").to_list()
    assert values == [None, 200.0, 50.0, 50.0, 350.0]


def test_on_balance_volume_null_and_validation() -> None:
    """OBV handles nulls, invalid lookback, missing columns, and edge frames."""
    frame = pl.DataFrame(
        {
            "close": [10.0, None, 12.0, 13.0],
            "volume": [100.0, 200.0, 150.0, 120.0],
        }
    )
    values = OnBalanceVolumeFactor().compute(frame).get_column("on_balance_volume").to_list()
    assert values[0] is None
    assert values[1] is None
    assert values[2] is None
    assert values[3] == pytest.approx(120.0)

    with pytest.raises(ValidationError, match="lookback must be 0") as exc_info:
        OnBalanceVolumeFactor(lookback=1)
    assert exc_info.value.error_code == "FACTOR-ON-BALANCE-VOLUME-001"

    _assert_missing_column(
        OnBalanceVolumeFactor(),
        missing="close",
        error_code="FACTOR-ON-BALANCE-VOLUME-002",
        frame=pl.DataFrame({"volume": [1.0, 2.0]}),
    )
    _assert_missing_column(
        OnBalanceVolumeFactor(),
        missing="volume",
        error_code="FACTOR-ON-BALANCE-VOLUME-002",
        frame=pl.DataFrame({"close": [1.0, 2.0]}),
    )
    factor = OnBalanceVolumeFactor()
    _assert_protocol_and_immutability(
        factor,
        output_column="on_balance_volume",
        frame=pl.DataFrame({"close": [1.0, 2.0, 3.0], "volume": [10.0, 20.0, 30.0]}),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="on_balance_volume",
        columns=("close", "volume"),
    )
    _assert_determinism(
        OnBalanceVolumeFactor,
        output_column="on_balance_volume",
        frame=pl.DataFrame({"close": [1.0, 2.0, 1.5], "volume": [10.0, 20.0, 15.0]}),
    )


# ---------------------------------------------------------------------------
# AccumulationDistributionFactor
# ---------------------------------------------------------------------------


def test_accumulation_distribution_metadata_and_math() -> None:
    """ADL metadata and cumulative money-flow volume are correct."""
    factor = AccumulationDistributionFactor()
    assert factor.name == "accumulation_distribution"
    assert factor.category == "volume"
    assert factor.lookback == 0
    assert factor.required_features == ("high", "low", "close", "volume")
    assert AccumulationDistributionFactor is AccumulationDistributionFactorDirect

    frame = pl.DataFrame(
        {
            "high": [12.0, 14.0, 13.0],
            "low": [10.0, 11.0, 11.0],
            "close": [11.0, 13.0, 12.0],
            "volume": [100.0, 200.0, 150.0],
        }
    )
    expected0 = _money_flow_multiplier(12.0, 10.0, 11.0) * 100.0
    expected1 = expected0 + _money_flow_multiplier(14.0, 11.0, 13.0) * 200.0
    expected2 = expected1 + _money_flow_multiplier(13.0, 11.0, 12.0) * 150.0
    values = factor.compute(frame).get_column("accumulation_distribution").to_list()
    assert values[0] == pytest.approx(expected0)
    assert values[1] == pytest.approx(expected1)
    assert values[2] == pytest.approx(expected2)


def test_accumulation_distribution_edge_cases() -> None:
    """ADL handles zero range, validation, missing columns, and immutability."""
    frame = pl.DataFrame(
        {
            "high": [10.0, 10.0],
            "low": [10.0, 10.0],
            "close": [10.0, 10.0],
            "volume": [100.0, 200.0],
        }
    )
    values = (
        AccumulationDistributionFactor()
        .compute(frame)
        .get_column("accumulation_distribution")
        .to_list()
    )
    assert values == [0.0, 0.0]

    with pytest.raises(ValidationError, match="lookback must be 0"):
        AccumulationDistributionFactor(lookback=5)

    factor = AccumulationDistributionFactor()
    _assert_missing_column(
        factor,
        missing="high",
        error_code="FACTOR-ACCUMULATION-DISTRIBUTION-002",
        frame=pl.DataFrame({"low": [1.0], "close": [1.0], "volume": [1.0]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="accumulation_distribution",
        frame=_ohlcv_frame(),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="accumulation_distribution",
        columns=("high", "low", "close", "volume"),
        single_expected=0.0,
    )
    _assert_determinism(
        AccumulationDistributionFactor,
        output_column="accumulation_distribution",
        frame=_ohlcv_frame(),
    )


# ---------------------------------------------------------------------------
# ChaikinMoneyFlowFactor
# ---------------------------------------------------------------------------


def test_chaikin_money_flow_metadata_and_math() -> None:
    """CMF metadata and rolling money-flow ratio are correct."""
    factor = ChaikinMoneyFlowFactor()
    assert factor.name == "chaikin_money_flow"
    assert factor.lookback == 20
    assert factor.category == "volume"
    assert ChaikinMoneyFlowFactor is ChaikinMoneyFlowFactorDirect

    frame = pl.DataFrame(
        {
            "high": [12.0, 14.0, 13.0],
            "low": [10.0, 11.0, 11.0],
            "close": [11.0, 13.0, 12.0],
            "volume": [100.0, 200.0, 150.0],
        }
    )
    mfvs = [
        _money_flow_multiplier(12.0, 10.0, 11.0) * 100.0,
        _money_flow_multiplier(14.0, 11.0, 13.0) * 200.0,
        _money_flow_multiplier(13.0, 11.0, 12.0) * 150.0,
    ]
    values = (
        ChaikinMoneyFlowFactor(lookback=2).compute(frame).get_column("chaikin_money_flow").to_list()
    )
    assert values[0] is None
    assert values[1] == pytest.approx((mfvs[0] + mfvs[1]) / 300.0)
    assert values[2] == pytest.approx((mfvs[1] + mfvs[2]) / 350.0)


def test_chaikin_money_flow_validation_and_edges() -> None:
    """CMF rejects invalid lookback and handles zero volume / edge frames."""
    with pytest.raises(ValidationError, match="lookback must be an integer greater than 0"):
        ChaikinMoneyFlowFactor(lookback=0)
    with pytest.raises(ValidationError, match="lookback must be an integer greater than or equal"):
        ChaikinMoneyFlowFactor(lookback=-1)

    zero_volume = pl.DataFrame(
        {
            "high": [12.0, 13.0],
            "low": [10.0, 11.0],
            "close": [11.0, 12.0],
            "volume": [0.0, 0.0],
        }
    )
    values = (
        ChaikinMoneyFlowFactor(lookback=2)
        .compute(zero_volume)
        .get_column("chaikin_money_flow")
        .to_list()
    )
    assert values[1] is None

    factor = ChaikinMoneyFlowFactor(lookback=2)
    _assert_missing_column(
        factor,
        missing="volume",
        error_code="FACTOR-CHAIKIN-MONEY-FLOW-002",
        frame=pl.DataFrame({"high": [1.0], "low": [1.0], "close": [1.0]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="chaikin_money_flow",
        frame=_ohlcv_frame(),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="chaikin_money_flow",
        columns=("high", "low", "close", "volume"),
    )
    large = ChaikinMoneyFlowFactor(lookback=50).compute(_ohlcv_frame())
    assert all(value is None for value in large.get_column("chaikin_money_flow").to_list())


# ---------------------------------------------------------------------------
# MoneyFlowIndexFactor
# ---------------------------------------------------------------------------


def test_money_flow_index_metadata_and_math() -> None:
    """MFI metadata and positive/negative money-flow ratio are correct."""
    factor = MoneyFlowIndexFactor()
    assert factor.name == "money_flow_index"
    assert factor.lookback == 14
    assert factor.category == "volume"
    assert MoneyFlowIndexFactor is MoneyFlowIndexFactorDirect

    frame = pl.DataFrame(
        {
            "high": [11.0, 12.0, 13.0, 12.0],
            "low": [9.0, 10.0, 11.0, 10.0],
            "close": [10.0, 11.0, 12.0, 11.0],
            "volume": [100.0, 150.0, 120.0, 180.0],
        }
    )
    typical = [
        (11.0 + 9.0 + 10.0) / 3.0,
        (12.0 + 10.0 + 11.0) / 3.0,
        (13.0 + 11.0 + 12.0) / 3.0,
        (12.0 + 10.0 + 11.0) / 3.0,
    ]
    raw = [tp * vol for tp, vol in zip(typical, [100.0, 150.0, 120.0, 180.0], strict=True)]
    # lookback=2 at index 2 uses rows [1, 2] (both positive typical-price moves).
    expected_2 = 100.0
    # lookback=2 at index 3 uses rows [2, 3] (positive then negative).
    expected_3 = 100.0 - (100.0 / (1.0 + (raw[2] / raw[3])))

    values = (
        MoneyFlowIndexFactor(lookback=2).compute(frame).get_column("money_flow_index").to_list()
    )
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx(expected_2)
    assert values[3] == pytest.approx(expected_3)


def test_money_flow_index_validation_and_edges() -> None:
    """MFI rejects invalid lookback and covers schema/immutability contracts."""
    with pytest.raises(
        ValidationError,
        match="lookback must be an integer greater than or equal to 2",
    ):
        MoneyFlowIndexFactor(lookback=1)

    factor = MoneyFlowIndexFactor(lookback=2)
    _assert_missing_column(
        factor,
        missing="low",
        error_code="FACTOR-MONEY-FLOW-INDEX-002",
        frame=pl.DataFrame({"high": [1.0], "close": [1.0], "volume": [1.0]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="money_flow_index",
        frame=_ohlcv_frame(),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="money_flow_index",
        columns=("high", "low", "close", "volume"),
    )
    _assert_determinism(
        lambda: MoneyFlowIndexFactor(lookback=2),
        output_column="money_flow_index",
        frame=_ohlcv_frame(),
    )


# ---------------------------------------------------------------------------
# VolumeRateOfChangeFactor
# ---------------------------------------------------------------------------


def test_volume_rate_of_change_metadata_and_math() -> None:
    """VROC matches fractional volume change over lookback."""
    factor = VolumeRateOfChangeFactor()
    assert factor.name == "volume_rate_of_change"
    assert factor.lookback == 20
    assert factor.required_features == ("volume",)
    assert VolumeRateOfChangeFactor is VolumeRateOfChangeFactorDirect

    frame = _volume_frame([100.0, 110.0, 121.0, 133.1])
    values = (
        VolumeRateOfChangeFactor(lookback=2)
        .compute(frame)
        .get_column("volume_rate_of_change")
        .to_list()
    )
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx((121.0 / 100.0) - 1.0)
    assert values[3] == pytest.approx((133.1 / 110.0) - 1.0)


def test_volume_rate_of_change_trends_and_validation() -> None:
    """VROC covers increasing/decreasing/constant volume and validation."""
    rising = (
        VolumeRateOfChangeFactor(lookback=2)
        .compute(_volume_frame([1.0, 2.0, 3.0, 4.0]))
        .get_column("volume_rate_of_change")
        .to_list()
    )
    falling = (
        VolumeRateOfChangeFactor(lookback=2)
        .compute(_volume_frame([4.0, 3.0, 2.0, 1.0]))
        .get_column("volume_rate_of_change")
        .to_list()
    )
    constant = (
        VolumeRateOfChangeFactor(lookback=2)
        .compute(_volume_frame([10.0, 10.0, 10.0, 10.0]))
        .get_column("volume_rate_of_change")
        .to_list()
    )
    assert rising[3] == pytest.approx(1.0)
    assert falling[3] == pytest.approx((1.0 / 3.0) - 1.0)
    assert constant[3] == pytest.approx(0.0)

    with pytest.raises(ValidationError, match="lookback must be an integer greater than 0"):
        VolumeRateOfChangeFactor(lookback=0)

    null_frame = _volume_frame([10.0, None, 12.0, 13.0])
    null_values = (
        VolumeRateOfChangeFactor(lookback=2)
        .compute(null_frame)
        .get_column("volume_rate_of_change")
        .to_list()
    )
    assert null_values[0] is None
    assert null_values[1] is None
    assert null_values[2] == pytest.approx((12.0 / 10.0) - 1.0)
    assert null_values[3] is None

    factor = VolumeRateOfChangeFactor(lookback=2)
    _assert_missing_column(
        factor,
        missing="volume",
        error_code="FACTOR-VOLUME-RATE-OF-CHANGE-002",
        frame=pl.DataFrame({"close": [1.0, 2.0]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="volume_rate_of_change",
        frame=_volume_frame([1.0, 2.0, 3.0, 4.0]),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="volume_rate_of_change",
        columns=("volume",),
    )


def test_volume_rate_of_change_previous_volume_zero_returns_null() -> None:
    """VROC is null when the lagged volume is zero."""
    values = (
        VolumeRateOfChangeFactor(lookback=1)
        .compute(_volume_frame([0.0, 100.0, 50.0]))
        .get_column("volume_rate_of_change")
        .to_list()
    )
    assert values[0] is None  # warmup
    assert values[1] is None  # previous volume == 0
    assert values[2] == pytest.approx((50.0 / 100.0) - 1.0)


def test_volume_rate_of_change_consecutive_zeros_return_null() -> None:
    """VROC is null across consecutive zero lagged volumes."""
    values = (
        VolumeRateOfChangeFactor(lookback=1)
        .compute(_volume_frame([0.0, 0.0, 0.0, 10.0]))
        .get_column("volume_rate_of_change")
        .to_list()
    )
    assert values[0] is None
    assert values[1] is None  # prev == 0
    assert values[2] is None  # prev == 0
    assert values[3] is None  # prev == 0


def test_volume_rate_of_change_current_volume_zero() -> None:
    """Current volume of zero with non-zero lag yields -1.0, never Inf."""
    values = (
        VolumeRateOfChangeFactor(lookback=1)
        .compute(_volume_frame([100.0, 0.0, 50.0]))
        .get_column("volume_rate_of_change")
        .to_list()
    )
    assert values[0] is None
    assert values[1] == pytest.approx(-1.0)
    assert values[2] is None  # previous volume == 0


def test_volume_rate_of_change_never_emits_inf_or_nan() -> None:
    """VROC never emits +Inf, -Inf, or NaN when lagged volume is zero."""
    values = (
        VolumeRateOfChangeFactor(lookback=2)
        .compute(_volume_frame([0.0, 0.0, 0.0, 5.0, 0.0, 10.0]))
        .get_column("volume_rate_of_change")
        .to_list()
    )
    for value in values:
        if value is None:
            continue
        assert math.isfinite(value)
        assert not math.isnan(value)
        assert not math.isinf(value)
        assert value != float("inf")
        assert value != float("-inf")


# ---------------------------------------------------------------------------
# RelativeVolumeFactor
# ---------------------------------------------------------------------------


def test_relative_volume_metadata_and_math() -> None:
    """RVOL matches volume divided by rolling mean."""
    factor = RelativeVolumeFactor()
    assert factor.name == "relative_volume"
    assert factor.lookback == 20
    assert RelativeVolumeFactor is RelativeVolumeFactorDirect

    volumes = [10.0, 20.0, 30.0, 40.0]
    values = (
        RelativeVolumeFactor(lookback=2)
        .compute(_volume_frame(volumes))
        .get_column("relative_volume")
        .to_list()
    )
    assert values[0] is None
    assert values[1] == pytest.approx(20.0 / 15.0)
    assert values[2] == pytest.approx(30.0 / 25.0)
    assert values[3] == pytest.approx(40.0 / 35.0)


def test_relative_volume_validation_and_edges() -> None:
    """RVOL handles zero mean, validation, and schema contracts."""
    with pytest.raises(ValidationError, match="lookback must be an integer greater than 0"):
        RelativeVolumeFactor(lookback=0)

    zero_mean = (
        RelativeVolumeFactor(lookback=2)
        .compute(_volume_frame([0.0, 0.0, 5.0]))
        .get_column("relative_volume")
        .to_list()
    )
    assert zero_mean[1] is None

    rising = (
        RelativeVolumeFactor(lookback=3)
        .compute(_volume_frame([1.0, 2.0, 3.0, 4.0]))
        .get_column("relative_volume")
        .to_list()
    )
    falling = (
        RelativeVolumeFactor(lookback=3)
        .compute(_volume_frame([4.0, 3.0, 2.0, 1.0]))
        .get_column("relative_volume")
        .to_list()
    )
    assert rising[3] > 1.0
    assert falling[3] < 1.0

    factor = RelativeVolumeFactor(lookback=2)
    _assert_protocol_and_immutability(
        factor,
        output_column="relative_volume",
        frame=_volume_frame([1.0, 2.0, 3.0]),
    )
    _assert_empty_and_single_row(factor, output_column="relative_volume", columns=("volume",))
    _assert_determinism(
        lambda: RelativeVolumeFactor(lookback=2),
        output_column="relative_volume",
        frame=_volume_frame([10.0, 20.0, 30.0, 40.0]),
    )


# ---------------------------------------------------------------------------
# VolumeZScoreFactor
# ---------------------------------------------------------------------------


def test_volume_zscore_metadata_and_math() -> None:
    """Volume z-score matches population standardization."""
    factor = VolumeZScoreFactor()
    assert factor.name == "volume_zscore"
    assert factor.lookback == 20
    assert VolumeZScoreFactor is VolumeZScoreFactorDirect

    volumes = [10.0, 12.0, 11.0, 15.0, 14.0]
    values = (
        VolumeZScoreFactor(lookback=3)
        .compute(_volume_frame(volumes))
        .get_column("volume_zscore")
        .to_list()
    )
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx(_population_zscore(volumes[0:3]))
    assert values[3] == pytest.approx(_population_zscore(volumes[1:4]))
    assert values[4] == pytest.approx(_population_zscore(volumes[2:5]))


def test_volume_zscore_trends_and_validation() -> None:
    """Volume z-score covers constant/trend cases and validation."""
    constant = (
        VolumeZScoreFactor(lookback=3)
        .compute(_volume_frame([10.0, 10.0, 10.0, 10.0]))
        .get_column("volume_zscore")
        .to_list()
    )
    assert constant[0] is None
    assert constant[1] is None
    assert constant[2] == pytest.approx(0.0)
    assert constant[3] == pytest.approx(0.0)

    rising = (
        VolumeZScoreFactor(lookback=3)
        .compute(_volume_frame([1.0, 2.0, 3.0, 4.0]))
        .get_column("volume_zscore")
        .to_list()
    )
    falling = (
        VolumeZScoreFactor(lookback=3)
        .compute(_volume_frame([4.0, 3.0, 2.0, 1.0]))
        .get_column("volume_zscore")
        .to_list()
    )
    assert rising[3] > 0.0
    assert falling[3] < 0.0

    with pytest.raises(
        ValidationError,
        match="lookback must be an integer greater than or equal to 2",
    ):
        VolumeZScoreFactor(lookback=1)

    null_values = (
        VolumeZScoreFactor(lookback=3)
        .compute(_volume_frame([10.0, None, 12.0, 13.0, 14.0]))
        .get_column("volume_zscore")
        .to_list()
    )
    assert null_values[2] is None
    assert null_values[3] is None
    assert null_values[4] == pytest.approx(_population_zscore([12.0, 13.0, 14.0]))

    factor = VolumeZScoreFactor(lookback=2)
    _assert_protocol_and_immutability(
        factor,
        output_column="volume_zscore",
        frame=_volume_frame([1.0, 2.0, 3.0, 4.0]),
    )
    _assert_empty_and_single_row(factor, output_column="volume_zscore", columns=("volume",))


# ---------------------------------------------------------------------------
# PriceVolumeTrendFactor
# ---------------------------------------------------------------------------


def test_price_volume_trend_metadata_and_math() -> None:
    """PVT matches cumulative return-scaled volume."""
    factor = PriceVolumeTrendFactor()
    assert factor.name == "price_volume_trend"
    assert factor.lookback == 0
    assert factor.required_features == ("close", "volume")
    assert PriceVolumeTrendFactor is PriceVolumeTrendFactorDirect

    frame = pl.DataFrame(
        {
            "close": [10.0, 11.0, 10.5, 10.5, 12.0],
            "volume": [100.0, 200.0, 150.0, 120.0, 300.0],
        }
    )
    values = factor.compute(frame).get_column("price_volume_trend").to_list()
    expected1 = ((11.0 / 10.0) - 1.0) * 200.0
    expected2 = expected1 + ((10.5 / 11.0) - 1.0) * 150.0
    expected3 = expected2 + ((10.5 / 10.5) - 1.0) * 120.0
    expected4 = expected3 + ((12.0 / 10.5) - 1.0) * 300.0
    assert values[0] is None
    assert values[1] == pytest.approx(expected1)
    assert values[2] == pytest.approx(expected2)
    assert values[3] == pytest.approx(expected3)
    assert values[4] == pytest.approx(expected4)


def test_price_volume_trend_validation_and_edges() -> None:
    """PVT rejects non-zero lookback and covers immutability contracts."""
    with pytest.raises(ValidationError, match="lookback must be 0"):
        PriceVolumeTrendFactor(lookback=3)

    factor = PriceVolumeTrendFactor()
    _assert_missing_column(
        factor,
        missing="volume",
        error_code="FACTOR-PRICE-VOLUME-TREND-002",
        frame=pl.DataFrame({"close": [1.0, 2.0]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="price_volume_trend",
        frame=pl.DataFrame({"close": [1.0, 2.0, 3.0], "volume": [10.0, 20.0, 30.0]}),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="price_volume_trend",
        columns=("close", "volume"),
    )
    _assert_determinism(
        PriceVolumeTrendFactor,
        output_column="price_volume_trend",
        frame=pl.DataFrame({"close": [10.0, 11.0, 12.0], "volume": [100.0, 110.0, 120.0]}),
    )


# ---------------------------------------------------------------------------
# EaseOfMovementFactor
# ---------------------------------------------------------------------------


def test_ease_of_movement_metadata_and_math() -> None:
    """EMV matches smoothed midprice distance scaled by range over volume."""
    factor = EaseOfMovementFactor()
    assert factor.name == "ease_of_movement"
    assert factor.lookback == 14
    assert factor.required_features == ("high", "low", "volume")
    assert EaseOfMovementFactor is EaseOfMovementFactorDirect

    frame = pl.DataFrame(
        {
            "high": [12.0, 14.0, 13.0, 15.0],
            "low": [10.0, 11.0, 11.0, 12.0],
            "volume": [100.0, 200.0, 150.0, 180.0],
        }
    )
    mids = [
        (12.0 + 10.0) / 2.0,
        (14.0 + 11.0) / 2.0,
        (13.0 + 11.0) / 2.0,
        (15.0 + 12.0) / 2.0,
    ]
    one_period = [
        None,
        ((mids[1] - mids[0]) * (14.0 - 11.0)) / 200.0,
        ((mids[2] - mids[1]) * (13.0 - 11.0)) / 150.0,
        ((mids[3] - mids[2]) * (15.0 - 12.0)) / 180.0,
    ]
    values = (
        EaseOfMovementFactor(lookback=2).compute(frame).get_column("ease_of_movement").to_list()
    )
    assert values[0] is None
    assert values[1] is None
    assert one_period[1] is not None and one_period[2] is not None
    assert values[2] == pytest.approx((one_period[1] + one_period[2]) / 2.0)
    assert one_period[3] is not None
    assert values[3] == pytest.approx((one_period[2] + one_period[3]) / 2.0)


def test_ease_of_movement_validation_and_edges() -> None:
    """EMV handles zero volume/range, validation, and schema contracts."""
    with pytest.raises(ValidationError, match="lookback must be an integer greater than 0"):
        EaseOfMovementFactor(lookback=0)

    zero_volume = pl.DataFrame(
        {
            "high": [12.0, 13.0, 14.0],
            "low": [10.0, 11.0, 12.0],
            "volume": [100.0, 0.0, 120.0],
        }
    )
    values = (
        EaseOfMovementFactor(lookback=2)
        .compute(zero_volume)
        .get_column("ease_of_movement")
        .to_list()
    )
    assert values[1] is None
    assert values[2] is None

    zero_range = pl.DataFrame(
        {
            "high": [10.0, 10.0, 10.0],
            "low": [10.0, 10.0, 10.0],
            "volume": [100.0, 200.0, 150.0],
        }
    )
    flat = (
        EaseOfMovementFactor(lookback=2)
        .compute(zero_range)
        .get_column("ease_of_movement")
        .to_list()
    )
    assert flat[0] is None
    assert flat[1] is None
    assert flat[2] is None

    factor = EaseOfMovementFactor(lookback=2)
    _assert_missing_column(
        factor,
        missing="high",
        error_code="FACTOR-EASE-OF-MOVEMENT-002",
        frame=pl.DataFrame({"low": [1.0], "volume": [1.0]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="ease_of_movement",
        frame=_ohlcv_frame().select(["high", "low", "volume"]),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="ease_of_movement",
        columns=("high", "low", "volume"),
    )


def test_ease_of_movement_zero_volume_returns_null() -> None:
    """EMV one-period value is null when volume is zero."""
    frame = pl.DataFrame(
        {
            "high": [12.0, 14.0, 13.0],
            "low": [10.0, 11.0, 11.0],
            "volume": [100.0, 0.0, 150.0],
        }
    )
    # lookback=1 exposes the one-period value without SMA masking.
    values = (
        EaseOfMovementFactor(lookback=1).compute(frame).get_column("ease_of_movement").to_list()
    )
    assert values[0] is None  # warmup: no previous mid
    assert values[1] is None  # volume == 0
    assert values[2] == pytest.approx(((12.0 - 12.5) * (13.0 - 11.0)) / 150.0)


def test_ease_of_movement_high_equals_low_returns_null() -> None:
    """EMV is null when high == low (zero range / box-ratio denominator)."""
    frame = pl.DataFrame(
        {
            "high": [12.0, 11.0, 13.0],
            "low": [10.0, 11.0, 11.0],
            "volume": [100.0, 200.0, 150.0],
        }
    )
    values = (
        EaseOfMovementFactor(lookback=1).compute(frame).get_column("ease_of_movement").to_list()
    )
    assert values[0] is None
    assert values[1] is None  # high == low
    assert values[2] == pytest.approx(((12.0 - 11.0) * (13.0 - 11.0)) / 150.0)


def test_ease_of_movement_zero_denominator_returns_null() -> None:
    """EMV is null for any zero denominator, including combined edge cases."""
    frame = pl.DataFrame(
        {
            "high": [10.0, 10.0, 12.0],
            "low": [10.0, 10.0, 11.0],
            "volume": [0.0, 100.0, 0.0],
        }
    )
    values = (
        EaseOfMovementFactor(lookback=1).compute(frame).get_column("ease_of_movement").to_list()
    )
    assert values[0] is None  # volume == 0 and high == low
    assert values[1] is None  # high == low
    assert values[2] is None  # volume == 0


def test_ease_of_movement_never_emits_inf_or_nan() -> None:
    """EMV never emits Inf or NaN under zero-denominator market edge cases."""
    frame = pl.DataFrame(
        {
            "high": [12.0, 10.0, 10.0, 14.0, 13.0],
            "low": [10.0, 10.0, 10.0, 11.0, 11.0],
            "volume": [100.0, 0.0, 200.0, 0.0, 150.0],
        }
    )
    values = (
        EaseOfMovementFactor(lookback=2).compute(frame).get_column("ease_of_movement").to_list()
    )
    for value in values:
        if value is None:
            continue
        assert math.isfinite(value)
        assert not math.isnan(value)
        assert not math.isinf(value)


# ---------------------------------------------------------------------------
# VolumeTrendFactor
# ---------------------------------------------------------------------------


def test_volume_trend_metadata_and_math() -> None:
    """Volume trend matches rolling OLS slope of volume."""
    factor = VolumeTrendFactor()
    assert factor.name == "volume_trend"
    assert factor.lookback == 20
    assert factor.required_features == ("volume",)
    assert VolumeTrendFactor is VolumeTrendFactorDirect

    volumes = [10.0, 12.0, 14.0, 16.0, 18.0]
    values = (
        VolumeTrendFactor(lookback=3)
        .compute(_volume_frame(volumes))
        .get_column("volume_trend")
        .to_list()
    )
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx(_ols_slope(volumes[0:3]))
    assert values[3] == pytest.approx(_ols_slope(volumes[1:4]))
    assert values[4] == pytest.approx(_ols_slope(volumes[2:5]))


def test_volume_trend_trends_and_validation() -> None:
    """Volume trend covers increasing/decreasing/constant volume and validation."""
    rising = (
        VolumeTrendFactor(lookback=3)
        .compute(_volume_frame([1.0, 2.0, 3.0, 4.0]))
        .get_column("volume_trend")
        .to_list()
    )
    falling = (
        VolumeTrendFactor(lookback=3)
        .compute(_volume_frame([4.0, 3.0, 2.0, 1.0]))
        .get_column("volume_trend")
        .to_list()
    )
    constant = (
        VolumeTrendFactor(lookback=3)
        .compute(_volume_frame([5.0, 5.0, 5.0, 5.0]))
        .get_column("volume_trend")
        .to_list()
    )
    assert rising[3] > 0.0
    assert falling[3] < 0.0
    assert constant[3] == pytest.approx(0.0)

    with pytest.raises(
        ValidationError,
        match="lookback must be an integer greater than or equal to 2",
    ):
        VolumeTrendFactor(lookback=1)

    factor = VolumeTrendFactor(lookback=2)
    _assert_missing_column(
        factor,
        missing="volume",
        error_code="FACTOR-VOLUME-TREND-002",
        frame=pl.DataFrame({"close": [1.0, 2.0]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="volume_trend",
        frame=_volume_frame([1.0, 2.0, 3.0, 4.0]),
    )
    _assert_empty_and_single_row(factor, output_column="volume_trend", columns=("volume",))
    _assert_determinism(
        lambda: VolumeTrendFactor(lookback=3),
        output_column="volume_trend",
        frame=_volume_frame([10.0, 12.0, 11.0, 15.0]),
    )
    large = VolumeTrendFactor(lookback=50).compute(_volume_frame([float(i) for i in range(1, 11)]))
    assert all(value is None for value in large.get_column("volume_trend").to_list())


def test_batch4_package_exports() -> None:
    """Volume package exports all ten Batch-4 factors."""
    exported = {
        AccumulationDistributionFactor,
        ChaikinMoneyFlowFactor,
        EaseOfMovementFactor,
        MoneyFlowIndexFactor,
        OnBalanceVolumeFactor,
        PriceVolumeTrendFactor,
        RelativeVolumeFactor,
        VolumeRateOfChangeFactor,
        VolumeTrendFactor,
        VolumeZScoreFactor,
    }
    assert len(exported) == 10
