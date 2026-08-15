"""Unit tests for CQROS Batch-8 liquidation and leverage factors."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import polars as pl
import pytest

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError
from cqros.factors.interfaces import Factor
from cqros.factors.liquidation import (
    LeverageChangeFactor,
    LeveragePressureFactor,
    LiquidationImbalanceFactor,
    LiquidationIntensityFactor,
    LiquidationMomentumFactor,
    LiquidationSpikeFactor,
    LiquidationTrendFactor,
    LiquidationZScoreFactor,
    LongLiquidationPressureFactor,
    ShortLiquidationPressureFactor,
)
from cqros.factors.liquidation.leverage_change import (
    LeverageChangeFactor as LeverageChangeFactorDirect,
)
from cqros.factors.liquidation.leverage_pressure import (
    LeveragePressureFactor as LeveragePressureFactorDirect,
)
from cqros.factors.liquidation.liquidation_imbalance import (
    LiquidationImbalanceFactor as LiquidationImbalanceFactorDirect,
)
from cqros.factors.liquidation.liquidation_intensity import (
    LiquidationIntensityFactor as LiquidationIntensityFactorDirect,
)
from cqros.factors.liquidation.liquidation_momentum import (
    LiquidationMomentumFactor as LiquidationMomentumFactorDirect,
)
from cqros.factors.liquidation.liquidation_spike import (
    LiquidationSpikeFactor as LiquidationSpikeFactorDirect,
)
from cqros.factors.liquidation.liquidation_trend import (
    LiquidationTrendFactor as LiquidationTrendFactorDirect,
)
from cqros.factors.liquidation.liquidation_zscore import (
    LiquidationZScoreFactor as LiquidationZScoreFactorDirect,
)
from cqros.factors.liquidation.long_liquidation_pressure import (
    LongLiquidationPressureFactor as LongLiquidationPressureFactorDirect,
)
from cqros.factors.liquidation.short_liquidation_pressure import (
    ShortLiquidationPressureFactor as ShortLiquidationPressureFactorDirect,
)


def _long_frame(values: Sequence[float | None]) -> pl.DataFrame:
    """Return a long-liquidation-only DataFrame."""
    return pl.DataFrame({"long_liquidation_volume": list(values)})


def _short_frame(values: Sequence[float | None]) -> pl.DataFrame:
    """Return a short-liquidation-only DataFrame."""
    return pl.DataFrame({"short_liquidation_volume": list(values)})


def _total_frame(values: Sequence[float | None]) -> pl.DataFrame:
    """Return a total-liquidation-only DataFrame."""
    return pl.DataFrame({"total_liquidation_volume": list(values)})


def _oi_frame(values: Sequence[float | None]) -> pl.DataFrame:
    """Return an open-interest-only DataFrame."""
    return pl.DataFrame({"open_interest": list(values)})


def _long_short_frame(
    long_values: Sequence[float | None],
    short_values: Sequence[float | None],
) -> pl.DataFrame:
    """Return a long/short liquidation DataFrame."""
    return pl.DataFrame(
        {
            "long_liquidation_volume": list(long_values),
            "short_liquidation_volume": list(short_values),
        }
    )


def _oi_liq_frame(
    oi: Sequence[float | None],
    liq: Sequence[float | None],
) -> pl.DataFrame:
    """Return an open interest / total liquidation DataFrame."""
    return pl.DataFrame(
        {
            "open_interest": list(oi),
            "total_liquidation_volume": list(liq),
        }
    )


def _liq_volume_frame(
    liq: Sequence[float | None],
    volumes: Sequence[float | None],
) -> pl.DataFrame:
    """Return a total liquidation / volume DataFrame."""
    return pl.DataFrame(
        {
            "total_liquidation_volume": list(liq),
            "volume": list(volumes),
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
    factory: Callable[[], BaseFactor],
    *,
    output_column: str,
    frame: pl.DataFrame,
) -> None:
    """Assert identical inputs produce identical outputs."""
    first = factory().compute(frame).get_column(output_column).to_list()
    second = factory().compute(frame).get_column(output_column).to_list()
    assert first == second


def _population_zscore(values: Sequence[float]) -> float | None:
    """Return population z-score for a fully observed window."""
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    if std == 0.0:
        return 0.0
    return (values[-1] - mean) / std


def _ols_slope(values: Sequence[float]) -> float:
    """Return OLS slope of values against 0..n-1."""
    n = len(values)
    xs = list(range(n))
    sum_x = sum(xs)
    sum_y = sum(values)
    sum_xy = sum(x * y for x, y in zip(xs, values, strict=True))
    sum_x2 = sum(x * x for x in xs)
    return (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)


# ---------------------------------------------------------------------------
# LongLiquidationPressureFactor
# ---------------------------------------------------------------------------


def test_long_liquidation_pressure_metadata_and_math() -> None:
    """Long liquidation pressure matches rolling mean of long liquidations."""
    factor = LongLiquidationPressureFactor()
    assert factor.name == "long_liquidation_pressure"
    assert factor.version == "1.0.0"
    assert factor.category == "liquidation"
    assert factor.lookback == 20
    assert factor.required_features == ("long_liquidation_volume",)
    assert factor.produced_columns == ("long_liquidation_pressure",)
    assert factor.metadata.name == "long_liquidation_pressure"
    assert LongLiquidationPressureFactor is LongLiquidationPressureFactorDirect

    values = (
        LongLiquidationPressureFactor(lookback=2)
        .compute(_long_frame([10.0, 20.0, 30.0, 40.0]))
        .get_column("long_liquidation_pressure")
        .to_list()
    )
    assert values[0] is None
    assert values[1] == pytest.approx(15.0)
    assert values[2] == pytest.approx(25.0)
    assert values[3] == pytest.approx(35.0)


def test_long_liquidation_pressure_validation_and_edges() -> None:
    """Long liquidation pressure covers validation, nulls, and contracts."""
    with pytest.raises(ValidationError, match="lookback must be an integer greater than 0"):
        LongLiquidationPressureFactor(lookback=0)

    null_values = (
        LongLiquidationPressureFactor(lookback=2)
        .compute(_long_frame([10.0, None, 30.0]))
        .get_column("long_liquidation_pressure")
        .to_list()
    )
    assert null_values[0] is None
    assert null_values[1] is None
    assert null_values[2] is None

    zero_values = (
        LongLiquidationPressureFactor(lookback=2)
        .compute(_long_frame([0.0, 0.0, 0.0]))
        .get_column("long_liquidation_pressure")
        .to_list()
    )
    assert zero_values[1] == pytest.approx(0.0)

    factor = LongLiquidationPressureFactor(lookback=2)
    _assert_missing_column(
        factor,
        missing="long_liquidation_volume",
        error_code="FACTOR-LONG-LIQUIDATION-PRESSURE-002",
        frame=pl.DataFrame({"close": [1.0, 2.0]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="long_liquidation_pressure",
        frame=_long_frame([10.0, 20.0, 30.0]),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="long_liquidation_pressure",
        columns=("long_liquidation_volume",),
    )
    _assert_determinism(
        lambda: LongLiquidationPressureFactor(lookback=2),
        output_column="long_liquidation_pressure",
        frame=_long_frame([10.0, 20.0, 30.0]),
    )


# ---------------------------------------------------------------------------
# ShortLiquidationPressureFactor
# ---------------------------------------------------------------------------


def test_short_liquidation_pressure_metadata_and_math() -> None:
    """Short liquidation pressure matches rolling mean of short liquidations."""
    factor = ShortLiquidationPressureFactor()
    assert factor.name == "short_liquidation_pressure"
    assert factor.version == "1.0.0"
    assert factor.category == "liquidation"
    assert factor.lookback == 20
    assert factor.required_features == ("short_liquidation_volume",)
    assert factor.produced_columns == ("short_liquidation_pressure",)
    assert ShortLiquidationPressureFactor is ShortLiquidationPressureFactorDirect

    values = (
        ShortLiquidationPressureFactor(lookback=2)
        .compute(_short_frame([5.0, 15.0, 25.0, 35.0]))
        .get_column("short_liquidation_pressure")
        .to_list()
    )
    assert values[0] is None
    assert values[1] == pytest.approx(10.0)
    assert values[2] == pytest.approx(20.0)
    assert values[3] == pytest.approx(30.0)


def test_short_liquidation_pressure_validation_and_edges() -> None:
    """Short liquidation pressure covers validation and contracts."""
    with pytest.raises(ValidationError, match="lookback must be an integer greater than 0"):
        ShortLiquidationPressureFactor(lookback=0)

    factor = ShortLiquidationPressureFactor(lookback=2)
    _assert_missing_column(
        factor,
        missing="short_liquidation_volume",
        error_code="FACTOR-SHORT-LIQUIDATION-PRESSURE-002",
        frame=pl.DataFrame({"close": [1.0, 2.0]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="short_liquidation_pressure",
        frame=_short_frame([5.0, 15.0, 25.0]),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="short_liquidation_pressure",
        columns=("short_liquidation_volume",),
    )
    _assert_determinism(
        lambda: ShortLiquidationPressureFactor(lookback=2),
        output_column="short_liquidation_pressure",
        frame=_short_frame([5.0, 15.0, 25.0]),
    )


# ---------------------------------------------------------------------------
# LiquidationImbalanceFactor
# ---------------------------------------------------------------------------


def test_liquidation_imbalance_metadata_and_math() -> None:
    """Liquidation imbalance matches rolling long/short volume imbalance."""
    factor = LiquidationImbalanceFactor()
    assert factor.name == "liquidation_imbalance"
    assert factor.lookback == 20
    assert factor.required_features == (
        "long_liquidation_volume",
        "short_liquidation_volume",
    )
    assert factor.produced_columns == ("liquidation_imbalance",)
    assert LiquidationImbalanceFactor is LiquidationImbalanceFactorDirect

    values = (
        LiquidationImbalanceFactor(lookback=2)
        .compute(_long_short_frame([10.0, 30.0, 20.0], [10.0, 10.0, 40.0]))
        .get_column("liquidation_imbalance")
        .to_list()
    )
    assert values[0] is None
    # Window [10,30] / [10,10] → (40-20)/(40+20) = 20/60
    assert values[1] == pytest.approx(20.0 / 60.0)
    # Window [30,20] / [10,40] → (50-50)/(50+50) = 0
    assert values[2] == pytest.approx(0.0)


def test_liquidation_imbalance_validation_and_edges() -> None:
    """Liquidation imbalance covers zero total, nulls, and contracts."""
    with pytest.raises(ValidationError, match="lookback must be an integer greater than 0"):
        LiquidationImbalanceFactor(lookback=0)

    zero_total = (
        LiquidationImbalanceFactor(lookback=2)
        .compute(_long_short_frame([0.0, 0.0], [0.0, 0.0]))
        .get_column("liquidation_imbalance")
        .to_list()
    )
    assert zero_total[1] is None

    long_heavy = (
        LiquidationImbalanceFactor(lookback=2)
        .compute(_long_short_frame([40.0, 60.0], [10.0, 10.0]))
        .get_column("liquidation_imbalance")
        .to_list()
    )
    assert long_heavy[1] is not None
    assert long_heavy[1] > 0.0

    short_heavy = (
        LiquidationImbalanceFactor(lookback=2)
        .compute(_long_short_frame([10.0, 10.0], [40.0, 60.0]))
        .get_column("liquidation_imbalance")
        .to_list()
    )
    assert short_heavy[1] is not None
    assert short_heavy[1] < 0.0

    factor = LiquidationImbalanceFactor(lookback=2)
    _assert_missing_column(
        factor,
        missing="short_liquidation_volume",
        error_code="FACTOR-LIQUIDATION-IMBALANCE-002",
        frame=pl.DataFrame({"long_liquidation_volume": [1.0, 2.0]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="liquidation_imbalance",
        frame=_long_short_frame([10.0, 20.0, 30.0], [5.0, 15.0, 25.0]),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="liquidation_imbalance",
        columns=("long_liquidation_volume", "short_liquidation_volume"),
    )
    _assert_determinism(
        lambda: LiquidationImbalanceFactor(lookback=2),
        output_column="liquidation_imbalance",
        frame=_long_short_frame([10.0, 20.0, 30.0], [5.0, 15.0, 25.0]),
    )


# ---------------------------------------------------------------------------
# LiquidationMomentumFactor
# ---------------------------------------------------------------------------


def test_liquidation_momentum_metadata_and_math() -> None:
    """Liquidation momentum matches absolute change over lookback."""
    factor = LiquidationMomentumFactor()
    assert factor.name == "liquidation_momentum"
    assert factor.lookback == 20
    assert LiquidationMomentumFactor is LiquidationMomentumFactorDirect

    values = (
        LiquidationMomentumFactor(lookback=2)
        .compute(_total_frame([100.0, 110.0, 130.0, 125.0]))
        .get_column("liquidation_momentum")
        .to_list()
    )
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx(30.0)
    assert values[3] == pytest.approx(15.0)


def test_liquidation_momentum_validation_and_edges() -> None:
    """Liquidation momentum covers trends, zero liquidations, and contracts."""
    with pytest.raises(ValidationError, match="lookback must be an integer greater than 0"):
        LiquidationMomentumFactor(lookback=0)

    rising = (
        LiquidationMomentumFactor(lookback=2)
        .compute(_total_frame([100.0, 110.0, 130.0, 160.0]))
        .get_column("liquidation_momentum")
        .to_list()
    )
    falling = (
        LiquidationMomentumFactor(lookback=2)
        .compute(_total_frame([160.0, 130.0, 110.0, 100.0]))
        .get_column("liquidation_momentum")
        .to_list()
    )
    zero = (
        LiquidationMomentumFactor(lookback=2)
        .compute(_total_frame([0.0, 0.0, 0.0, 0.0]))
        .get_column("liquidation_momentum")
        .to_list()
    )
    assert rising[3] == pytest.approx(50.0)
    assert falling[3] == pytest.approx(-30.0)
    assert zero[3] == pytest.approx(0.0)

    null_values = (
        LiquidationMomentumFactor(lookback=2)
        .compute(_total_frame([100.0, None, 130.0, 140.0]))
        .get_column("liquidation_momentum")
        .to_list()
    )
    assert null_values[2] == pytest.approx(30.0)
    assert null_values[3] is None

    factor = LiquidationMomentumFactor(lookback=2)
    _assert_missing_column(
        factor,
        missing="total_liquidation_volume",
        error_code="FACTOR-LIQUIDATION-MOMENTUM-002",
        frame=pl.DataFrame({"close": [1.0, 2.0]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="liquidation_momentum",
        frame=_total_frame([100.0, 110.0, 120.0, 130.0]),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="liquidation_momentum",
        columns=("total_liquidation_volume",),
    )
    _assert_determinism(
        lambda: LiquidationMomentumFactor(lookback=2),
        output_column="liquidation_momentum",
        frame=_total_frame([100.0, 110.0, 120.0, 130.0]),
    )


# ---------------------------------------------------------------------------
# LiquidationZScoreFactor
# ---------------------------------------------------------------------------


def test_liquidation_zscore_metadata_and_math() -> None:
    """Liquidation z-score matches population z-score over lookback."""
    factor = LiquidationZScoreFactor()
    assert factor.name == "liquidation_zscore"
    assert factor.lookback == 20
    assert LiquidationZScoreFactor is LiquidationZScoreFactorDirect

    liq = [100.0, 120.0, 110.0, 160.0]
    values = (
        LiquidationZScoreFactor(lookback=2)
        .compute(_total_frame(liq))
        .get_column("liquidation_zscore")
        .to_list()
    )
    assert values[0] is None
    assert values[1] == pytest.approx(_population_zscore(liq[0:2]))
    assert values[2] == pytest.approx(_population_zscore(liq[1:3]))
    assert values[3] == pytest.approx(_population_zscore(liq[2:4]))


def test_liquidation_zscore_validation_and_edges() -> None:
    """Liquidation z-score covers zero std and contracts."""
    with pytest.raises(
        ValidationError,
        match="lookback must be an integer greater than or equal to 2",
    ):
        LiquidationZScoreFactor(lookback=1)

    constant = (
        LiquidationZScoreFactor(lookback=2)
        .compute(_total_frame([100.0, 100.0, 100.0]))
        .get_column("liquidation_zscore")
        .to_list()
    )
    assert constant[0] is None
    assert constant[1] == pytest.approx(0.0)
    assert constant[2] == pytest.approx(0.0)

    factor = LiquidationZScoreFactor(lookback=2)
    _assert_missing_column(
        factor,
        missing="total_liquidation_volume",
        error_code="FACTOR-LIQUIDATION-ZSCORE-002",
        frame=pl.DataFrame({"close": [1.0, 2.0]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="liquidation_zscore",
        frame=_total_frame([100.0, 110.0, 120.0, 130.0]),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="liquidation_zscore",
        columns=("total_liquidation_volume",),
    )
    _assert_determinism(
        lambda: LiquidationZScoreFactor(lookback=2),
        output_column="liquidation_zscore",
        frame=_total_frame([100.0, 110.0, 120.0, 130.0]),
    )


# ---------------------------------------------------------------------------
# LiquidationSpikeFactor
# ---------------------------------------------------------------------------


def test_liquidation_spike_metadata_and_math() -> None:
    """Liquidation spike matches current liquidations over rolling mean."""
    factor = LiquidationSpikeFactor()
    assert factor.name == "liquidation_spike"
    assert factor.lookback == 20
    assert LiquidationSpikeFactor is LiquidationSpikeFactorDirect

    values = (
        LiquidationSpikeFactor(lookback=2)
        .compute(_total_frame([10.0, 30.0, 20.0, 60.0]))
        .get_column("liquidation_spike")
        .to_list()
    )
    assert values[0] is None
    assert values[1] == pytest.approx(30.0 / 20.0)
    assert values[2] == pytest.approx(20.0 / 25.0)
    assert values[3] == pytest.approx(60.0 / 40.0)


def test_liquidation_spike_validation_and_edges() -> None:
    """Liquidation spike covers zero mean, spikes, and contracts."""
    with pytest.raises(ValidationError, match="lookback must be an integer greater than 0"):
        LiquidationSpikeFactor(lookback=0)

    zero_mean = (
        LiquidationSpikeFactor(lookback=2)
        .compute(_total_frame([0.0, 0.0, 0.0]))
        .get_column("liquidation_spike")
        .to_list()
    )
    assert zero_mean[1] is None

    spike = (
        LiquidationSpikeFactor(lookback=2)
        .compute(_total_frame([10.0, 10.0, 100.0]))
        .get_column("liquidation_spike")
        .to_list()
    )
    assert spike[2] == pytest.approx(100.0 / 55.0)
    assert spike[2] is not None
    assert spike[2] > 1.0

    factor = LiquidationSpikeFactor(lookback=2)
    _assert_missing_column(
        factor,
        missing="total_liquidation_volume",
        error_code="FACTOR-LIQUIDATION-SPIKE-002",
        frame=pl.DataFrame({"close": [1.0, 2.0]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="liquidation_spike",
        frame=_total_frame([10.0, 20.0, 30.0]),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="liquidation_spike",
        columns=("total_liquidation_volume",),
    )
    _assert_determinism(
        lambda: LiquidationSpikeFactor(lookback=2),
        output_column="liquidation_spike",
        frame=_total_frame([10.0, 20.0, 30.0]),
    )


# ---------------------------------------------------------------------------
# LeveragePressureFactor
# ---------------------------------------------------------------------------


def test_leverage_pressure_metadata_and_math() -> None:
    """Leverage pressure matches rolling mean liquidations over mean OI."""
    factor = LeveragePressureFactor()
    assert factor.name == "leverage_pressure"
    assert factor.lookback == 20
    assert factor.required_features == ("open_interest", "total_liquidation_volume")
    assert LeveragePressureFactor is LeveragePressureFactorDirect

    values = (
        LeveragePressureFactor(lookback=2)
        .compute(_oi_liq_frame([100.0, 200.0, 300.0, 400.0], [10.0, 20.0, 30.0, 40.0]))
        .get_column("leverage_pressure")
        .to_list()
    )
    assert values[0] is None
    assert values[1] == pytest.approx(15.0 / 150.0)
    assert values[2] == pytest.approx(25.0 / 250.0)
    assert values[3] == pytest.approx(35.0 / 350.0)


def test_leverage_pressure_validation_and_edges() -> None:
    """Leverage pressure covers zero OI mean and contracts."""
    with pytest.raises(ValidationError, match="lookback must be an integer greater than 0"):
        LeveragePressureFactor(lookback=0)

    zero_oi = (
        LeveragePressureFactor(lookback=2)
        .compute(_oi_liq_frame([0.0, 0.0, 0.0], [10.0, 20.0, 30.0]))
        .get_column("leverage_pressure")
        .to_list()
    )
    assert zero_oi[1] is None

    factor = LeveragePressureFactor(lookback=2)
    _assert_missing_column(
        factor,
        missing="total_liquidation_volume",
        error_code="FACTOR-LEVERAGE-PRESSURE-002",
        frame=pl.DataFrame({"open_interest": [1.0, 2.0]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="leverage_pressure",
        frame=_oi_liq_frame([100.0, 200.0, 300.0], [10.0, 20.0, 30.0]),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="leverage_pressure",
        columns=("open_interest", "total_liquidation_volume"),
    )
    _assert_determinism(
        lambda: LeveragePressureFactor(lookback=2),
        output_column="leverage_pressure",
        frame=_oi_liq_frame([100.0, 200.0, 300.0], [10.0, 20.0, 30.0]),
    )


# ---------------------------------------------------------------------------
# LeverageChangeFactor
# ---------------------------------------------------------------------------


def test_leverage_change_metadata_and_math() -> None:
    """Leverage change matches fractional open interest change."""
    factor = LeverageChangeFactor()
    assert factor.name == "leverage_change"
    assert factor.lookback == 20
    assert LeverageChangeFactor is LeverageChangeFactorDirect

    values = (
        LeverageChangeFactor(lookback=2)
        .compute(_oi_frame([100.0, 110.0, 130.0, 125.0]))
        .get_column("leverage_change")
        .to_list()
    )
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx(0.3)
    assert values[3] == pytest.approx(125.0 / 110.0 - 1.0)


def test_leverage_change_validation_and_edges() -> None:
    """Leverage change covers zero lagged OI and contracts."""
    with pytest.raises(ValidationError, match="lookback must be an integer greater than 0"):
        LeverageChangeFactor(lookback=0)

    zero_lag = (
        LeverageChangeFactor(lookback=2)
        .compute(_oi_frame([0.0, 10.0, 20.0]))
        .get_column("leverage_change")
        .to_list()
    )
    assert zero_lag[2] is None

    rising = (
        LeverageChangeFactor(lookback=2)
        .compute(_oi_frame([100.0, 110.0, 150.0]))
        .get_column("leverage_change")
        .to_list()
    )
    assert rising[2] == pytest.approx(0.5)

    factor = LeverageChangeFactor(lookback=2)
    _assert_missing_column(
        factor,
        missing="open_interest",
        error_code="FACTOR-LEVERAGE-CHANGE-002",
        frame=pl.DataFrame({"close": [1.0, 2.0]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="leverage_change",
        frame=_oi_frame([100.0, 110.0, 120.0, 130.0]),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="leverage_change",
        columns=("open_interest",),
    )
    _assert_determinism(
        lambda: LeverageChangeFactor(lookback=2),
        output_column="leverage_change",
        frame=_oi_frame([100.0, 110.0, 120.0, 130.0]),
    )


# ---------------------------------------------------------------------------
# LiquidationIntensityFactor
# ---------------------------------------------------------------------------


def test_liquidation_intensity_metadata_and_math() -> None:
    """Liquidation intensity matches rolling mean liquidations over volume."""
    factor = LiquidationIntensityFactor()
    assert factor.name == "liquidation_intensity"
    assert factor.lookback == 20
    assert factor.required_features == ("total_liquidation_volume", "volume")
    assert LiquidationIntensityFactor is LiquidationIntensityFactorDirect

    values = (
        LiquidationIntensityFactor(lookback=2)
        .compute(_liq_volume_frame([10.0, 20.0, 30.0, 40.0], [100.0, 200.0, 300.0, 400.0]))
        .get_column("liquidation_intensity")
        .to_list()
    )
    assert values[0] is None
    assert values[1] == pytest.approx(15.0 / 150.0)
    assert values[2] == pytest.approx(25.0 / 250.0)
    assert values[3] == pytest.approx(35.0 / 350.0)


def test_liquidation_intensity_validation_and_edges() -> None:
    """Liquidation intensity covers zero volume mean and contracts."""
    with pytest.raises(ValidationError, match="lookback must be an integer greater than 0"):
        LiquidationIntensityFactor(lookback=0)

    zero_volume = (
        LiquidationIntensityFactor(lookback=2)
        .compute(_liq_volume_frame([10.0, 20.0, 30.0], [0.0, 0.0, 0.0]))
        .get_column("liquidation_intensity")
        .to_list()
    )
    assert zero_volume[1] is None

    factor = LiquidationIntensityFactor(lookback=2)
    _assert_missing_column(
        factor,
        missing="volume",
        error_code="FACTOR-LIQUIDATION-INTENSITY-002",
        frame=pl.DataFrame({"total_liquidation_volume": [1.0, 2.0]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="liquidation_intensity",
        frame=_liq_volume_frame([10.0, 20.0, 30.0], [100.0, 200.0, 300.0]),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="liquidation_intensity",
        columns=("total_liquidation_volume", "volume"),
    )
    _assert_determinism(
        lambda: LiquidationIntensityFactor(lookback=2),
        output_column="liquidation_intensity",
        frame=_liq_volume_frame([10.0, 20.0, 30.0], [100.0, 200.0, 300.0]),
    )


# ---------------------------------------------------------------------------
# LiquidationTrendFactor
# ---------------------------------------------------------------------------


def test_liquidation_trend_metadata_and_math() -> None:
    """Liquidation trend matches rolling OLS slope of total liquidations."""
    factor = LiquidationTrendFactor()
    assert factor.name == "liquidation_trend"
    assert factor.lookback == 20
    assert LiquidationTrendFactor is LiquidationTrendFactorDirect

    liq = [10.0, 20.0, 30.0, 50.0]
    values = (
        LiquidationTrendFactor(lookback=3)
        .compute(_total_frame(liq))
        .get_column("liquidation_trend")
        .to_list()
    )
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx(_ols_slope(liq[0:3]))
    assert values[3] == pytest.approx(_ols_slope(liq[1:4]))


def test_liquidation_trend_validation_and_edges() -> None:
    """Liquidation trend covers rising/falling series and contracts."""
    with pytest.raises(
        ValidationError,
        match="lookback must be an integer greater than or equal to 2",
    ):
        LiquidationTrendFactor(lookback=1)

    rising = (
        LiquidationTrendFactor(lookback=3)
        .compute(_total_frame([10.0, 20.0, 30.0, 40.0]))
        .get_column("liquidation_trend")
        .to_list()
    )
    falling = (
        LiquidationTrendFactor(lookback=3)
        .compute(_total_frame([40.0, 30.0, 20.0, 10.0]))
        .get_column("liquidation_trend")
        .to_list()
    )
    assert rising[2] is not None and rising[2] > 0.0
    assert falling[2] is not None and falling[2] < 0.0

    flat = (
        LiquidationTrendFactor(lookback=3)
        .compute(_total_frame([10.0, 10.0, 10.0]))
        .get_column("liquidation_trend")
        .to_list()
    )
    assert flat[2] == pytest.approx(0.0)

    factor = LiquidationTrendFactor(lookback=3)
    _assert_missing_column(
        factor,
        missing="total_liquidation_volume",
        error_code="FACTOR-LIQUIDATION-TREND-002",
        frame=pl.DataFrame({"close": [1.0, 2.0, 3.0]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="liquidation_trend",
        frame=_total_frame([10.0, 20.0, 30.0, 40.0]),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="liquidation_trend",
        columns=("total_liquidation_volume",),
    )
    _assert_determinism(
        lambda: LiquidationTrendFactor(lookback=3),
        output_column="liquidation_trend",
        frame=_total_frame([10.0, 20.0, 30.0, 40.0]),
    )
