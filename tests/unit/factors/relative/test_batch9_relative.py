"""Unit tests for CQROS Batch-9 cross-asset and relative-value factors."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import polars as pl
import pytest

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError
from cqros.factors.interfaces import Factor
from cqros.factors.relative import (
    BetaToBTCFactor,
    BetaToETHFactor,
    CorrelationBTCFactor,
    CorrelationETHFactor,
    RelativeMomentumBTCFactor,
    RelativeMomentumETHFactor,
    RelativeStrengthBTCFactor,
    RelativeStrengthETHFactor,
    RelativeVolatilityFactor,
    TrackingErrorFactor,
)
from cqros.factors.relative.beta_to_btc import BetaToBTCFactor as BetaToBTCFactorDirect
from cqros.factors.relative.beta_to_eth import BetaToETHFactor as BetaToETHFactorDirect
from cqros.factors.relative.correlation_btc import (
    CorrelationBTCFactor as CorrelationBTCFactorDirect,
)
from cqros.factors.relative.correlation_eth import (
    CorrelationETHFactor as CorrelationETHFactorDirect,
)
from cqros.factors.relative.relative_momentum_btc import (
    RelativeMomentumBTCFactor as RelativeMomentumBTCFactorDirect,
)
from cqros.factors.relative.relative_momentum_eth import (
    RelativeMomentumETHFactor as RelativeMomentumETHFactorDirect,
)
from cqros.factors.relative.relative_strength_btc import (
    RelativeStrengthBTCFactor as RelativeStrengthBTCFactorDirect,
)
from cqros.factors.relative.relative_strength_eth import (
    RelativeStrengthETHFactor as RelativeStrengthETHFactorDirect,
)
from cqros.factors.relative.relative_volatility import (
    RelativeVolatilityFactor as RelativeVolatilityFactorDirect,
)
from cqros.factors.relative.tracking_error import (
    TrackingErrorFactor as TrackingErrorFactorDirect,
)


def _btc_frame(
    asset: Sequence[float | None],
    btc: Sequence[float | None],
) -> pl.DataFrame:
    """Return an asset/BTC return DataFrame."""
    return pl.DataFrame({"asset_return": list(asset), "btc_return": list(btc)})


def _eth_frame(
    asset: Sequence[float | None],
    eth: Sequence[float | None],
) -> pl.DataFrame:
    """Return an asset/ETH return DataFrame."""
    return pl.DataFrame({"asset_return": list(asset), "eth_return": list(eth)})


def _benchmark_frame(
    asset: Sequence[float | None],
    benchmark: Sequence[float | None],
) -> pl.DataFrame:
    """Return an asset/benchmark return DataFrame."""
    return pl.DataFrame({"asset_return": list(asset), "benchmark_return": list(benchmark)})


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


def _cumulative_return(values: Sequence[float]) -> float:
    """Return compounded cumulative return for a fully observed window."""
    product = 1.0
    for value in values:
        product *= 1.0 + value
    return product - 1.0


def _population_cov(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Return population covariance for a fully observed window."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / n


def _population_std(values: Sequence[float]) -> float:
    """Return population standard deviation for a fully observed window."""
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


# ---------------------------------------------------------------------------
# RelativeStrengthBTCFactor
# ---------------------------------------------------------------------------


def test_relative_strength_btc_metadata_and_math() -> None:
    """BTC relative strength matches cumulative-return difference."""
    factor = RelativeStrengthBTCFactor()
    assert factor.name == "relative_strength_btc"
    assert factor.version == "1.0.0"
    assert factor.category == "relative"
    assert factor.lookback == 20
    assert factor.required_features == ("asset_return", "btc_return")
    assert factor.produced_columns == ("relative_strength_btc",)
    assert factor.metadata.name == "relative_strength_btc"
    assert RelativeStrengthBTCFactor is RelativeStrengthBTCFactorDirect

    asset = [0.01, 0.02, -0.01, 0.03]
    btc = [0.005, 0.01, 0.0, 0.02]
    values = (
        RelativeStrengthBTCFactor(lookback=3)
        .compute(_btc_frame(asset, btc))
        .get_column("relative_strength_btc")
        .to_list()
    )
    assert values[0] is None
    assert values[1] is None
    expected = _cumulative_return(asset[0:3]) - _cumulative_return(btc[0:3])
    assert values[2] == pytest.approx(expected)
    expected_last = _cumulative_return(asset[1:4]) - _cumulative_return(btc[1:4])
    assert values[3] == pytest.approx(expected_last)


def test_relative_strength_btc_validation_and_edges() -> None:
    """BTC relative strength covers validation, nulls, and contracts."""
    with pytest.raises(ValidationError, match="lookback must be an integer greater than 0"):
        RelativeStrengthBTCFactor(lookback=0)

    factor = RelativeStrengthBTCFactor(lookback=2)
    _assert_missing_column(
        factor,
        missing="asset_return",
        error_code="FACTOR-RELATIVE-STRENGTH-BTC-002",
        frame=pl.DataFrame({"btc_return": [0.01]}),
    )
    _assert_missing_column(
        factor,
        missing="btc_return",
        error_code="FACTOR-RELATIVE-STRENGTH-BTC-002",
        frame=pl.DataFrame({"asset_return": [0.01]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="relative_strength_btc",
        frame=_btc_frame([0.01, 0.02, -0.01], [0.0, 0.01, 0.005]),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="relative_strength_btc",
        columns=("asset_return", "btc_return"),
        single_expected=None,
    )
    null_values = (
        factor.compute(_btc_frame([0.01, None, 0.02], [0.0, 0.01, 0.005]))
        .get_column("relative_strength_btc")
        .to_list()
    )
    assert null_values[0] is None
    assert null_values[1] is None
    assert null_values[2] is None
    _assert_determinism(
        lambda: RelativeStrengthBTCFactor(lookback=2),
        output_column="relative_strength_btc",
        frame=_btc_frame([0.01, 0.02, -0.01], [0.0, 0.01, 0.005]),
    )


# ---------------------------------------------------------------------------
# RelativeStrengthETHFactor
# ---------------------------------------------------------------------------


def test_relative_strength_eth_metadata_and_math() -> None:
    """ETH relative strength matches cumulative-return difference."""
    factor = RelativeStrengthETHFactor()
    assert factor.name == "relative_strength_eth"
    assert factor.category == "relative"
    assert factor.lookback == 20
    assert factor.required_features == ("asset_return", "eth_return")
    assert RelativeStrengthETHFactor is RelativeStrengthETHFactorDirect

    asset = [0.01, 0.02, -0.01, 0.03]
    eth = [0.004, 0.008, 0.001, 0.015]
    values = (
        RelativeStrengthETHFactor(lookback=3)
        .compute(_eth_frame(asset, eth))
        .get_column("relative_strength_eth")
        .to_list()
    )
    assert values[0] is None
    assert values[1] is None
    expected = _cumulative_return(asset[0:3]) - _cumulative_return(eth[0:3])
    assert values[2] == pytest.approx(expected)


def test_relative_strength_eth_validation_and_edges() -> None:
    """ETH relative strength covers validation and contracts."""
    with pytest.raises(ValidationError, match="lookback must be an integer greater than 0"):
        RelativeStrengthETHFactor(lookback=0)

    factor = RelativeStrengthETHFactor(lookback=2)
    _assert_missing_column(
        factor,
        missing="eth_return",
        error_code="FACTOR-RELATIVE-STRENGTH-ETH-002",
        frame=pl.DataFrame({"asset_return": [0.01]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="relative_strength_eth",
        frame=_eth_frame([0.01, 0.02], [0.0, 0.01]),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="relative_strength_eth",
        columns=("asset_return", "eth_return"),
        single_expected=None,
    )
    _assert_determinism(
        lambda: RelativeStrengthETHFactor(lookback=2),
        output_column="relative_strength_eth",
        frame=_eth_frame([0.01, 0.02, -0.01], [0.0, 0.01, 0.005]),
    )


# ---------------------------------------------------------------------------
# RelativeMomentumBTCFactor
# ---------------------------------------------------------------------------


def test_relative_momentum_btc_metadata_and_math() -> None:
    """BTC relative momentum matches rolling return-sum difference."""
    factor = RelativeMomentumBTCFactor()
    assert factor.name == "relative_momentum_btc"
    assert factor.category == "relative"
    assert factor.lookback == 20
    assert RelativeMomentumBTCFactor is RelativeMomentumBTCFactorDirect

    asset = [0.01, 0.02, -0.01, 0.03]
    btc = [0.005, 0.01, 0.0, 0.02]
    values = (
        RelativeMomentumBTCFactor(lookback=3)
        .compute(_btc_frame(asset, btc))
        .get_column("relative_momentum_btc")
        .to_list()
    )
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx(sum(asset[0:3]) - sum(btc[0:3]))
    assert values[3] == pytest.approx(sum(asset[1:4]) - sum(btc[1:4]))


def test_relative_momentum_btc_validation_and_edges() -> None:
    """BTC relative momentum covers validation and contracts."""
    with pytest.raises(ValidationError, match="lookback must be an integer greater than 0"):
        RelativeMomentumBTCFactor(lookback=0)

    factor = RelativeMomentumBTCFactor(lookback=2)
    _assert_missing_column(
        factor,
        missing="btc_return",
        error_code="FACTOR-RELATIVE-MOMENTUM-BTC-002",
        frame=pl.DataFrame({"asset_return": [0.01]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="relative_momentum_btc",
        frame=_btc_frame([0.01, 0.02], [0.0, 0.01]),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="relative_momentum_btc",
        columns=("asset_return", "btc_return"),
        single_expected=None,
    )
    null_values = (
        factor.compute(_btc_frame([0.01, None], [0.0, 0.01]))
        .get_column("relative_momentum_btc")
        .to_list()
    )
    assert null_values[1] is None
    _assert_determinism(
        lambda: RelativeMomentumBTCFactor(lookback=2),
        output_column="relative_momentum_btc",
        frame=_btc_frame([0.01, 0.02, -0.01], [0.0, 0.01, 0.005]),
    )


# ---------------------------------------------------------------------------
# RelativeMomentumETHFactor
# ---------------------------------------------------------------------------


def test_relative_momentum_eth_metadata_and_math() -> None:
    """ETH relative momentum matches rolling return-sum difference."""
    factor = RelativeMomentumETHFactor()
    assert factor.name == "relative_momentum_eth"
    assert factor.category == "relative"
    assert RelativeMomentumETHFactor is RelativeMomentumETHFactorDirect

    asset = [0.01, 0.02, -0.01]
    eth = [0.004, 0.008, 0.001]
    values = (
        RelativeMomentumETHFactor(lookback=2)
        .compute(_eth_frame(asset, eth))
        .get_column("relative_momentum_eth")
        .to_list()
    )
    assert values[0] is None
    assert values[1] == pytest.approx(sum(asset[0:2]) - sum(eth[0:2]))
    assert values[2] == pytest.approx(sum(asset[1:3]) - sum(eth[1:3]))


def test_relative_momentum_eth_validation_and_edges() -> None:
    """ETH relative momentum covers validation and contracts."""
    with pytest.raises(ValidationError, match="lookback must be an integer greater than 0"):
        RelativeMomentumETHFactor(lookback=0)

    factor = RelativeMomentumETHFactor(lookback=2)
    _assert_missing_column(
        factor,
        missing="eth_return",
        error_code="FACTOR-RELATIVE-MOMENTUM-ETH-002",
        frame=pl.DataFrame({"asset_return": [0.01]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="relative_momentum_eth",
        frame=_eth_frame([0.01, 0.02], [0.0, 0.01]),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="relative_momentum_eth",
        columns=("asset_return", "eth_return"),
        single_expected=None,
    )
    _assert_determinism(
        lambda: RelativeMomentumETHFactor(lookback=2),
        output_column="relative_momentum_eth",
        frame=_eth_frame([0.01, 0.02, -0.01], [0.0, 0.01, 0.005]),
    )


# ---------------------------------------------------------------------------
# BetaToBTCFactor
# ---------------------------------------------------------------------------


def test_beta_to_btc_metadata_and_math() -> None:
    """Beta-to-BTC matches population covariance over variance."""
    factor = BetaToBTCFactor()
    assert factor.name == "beta_to_btc"
    assert factor.category == "relative"
    assert factor.lookback == 20
    assert BetaToBTCFactor is BetaToBTCFactorDirect

    asset = [0.01, 0.02, -0.01, 0.03]
    btc = [0.005, 0.01, 0.0, 0.02]
    values = (
        BetaToBTCFactor(lookback=3)
        .compute(_btc_frame(asset, btc))
        .get_column("beta_to_btc")
        .to_list()
    )
    assert values[0] is None
    assert values[1] is None
    cov = _population_cov(asset[0:3], btc[0:3])
    var = _population_std(btc[0:3]) ** 2
    assert values[2] == pytest.approx(cov / var)


def test_beta_to_btc_validation_and_edges() -> None:
    """Beta-to-BTC covers zero variance, validation, and contracts."""
    with pytest.raises(
        ValidationError,
        match="lookback must be an integer greater than or equal to 2",
    ):
        BetaToBTCFactor(lookback=1)

    factor = BetaToBTCFactor(lookback=2)
    _assert_missing_column(
        factor,
        missing="btc_return",
        error_code="FACTOR-BETA-TO-BTC-002",
        frame=pl.DataFrame({"asset_return": [0.01]}),
    )
    zero_var = (
        factor.compute(_btc_frame([0.01, 0.02], [0.01, 0.01])).get_column("beta_to_btc").to_list()
    )
    assert zero_var[1] is None
    _assert_protocol_and_immutability(
        factor,
        output_column="beta_to_btc",
        frame=_btc_frame([0.01, 0.02, -0.01], [0.0, 0.01, 0.005]),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="beta_to_btc",
        columns=("asset_return", "btc_return"),
        single_expected=None,
    )
    _assert_determinism(
        lambda: BetaToBTCFactor(lookback=2),
        output_column="beta_to_btc",
        frame=_btc_frame([0.01, 0.02, -0.01], [0.0, 0.01, 0.005]),
    )


# ---------------------------------------------------------------------------
# BetaToETHFactor
# ---------------------------------------------------------------------------


def test_beta_to_eth_metadata_and_math() -> None:
    """Beta-to-ETH matches population covariance over variance."""
    factor = BetaToETHFactor()
    assert factor.name == "beta_to_eth"
    assert factor.category == "relative"
    assert BetaToETHFactor is BetaToETHFactorDirect

    asset = [0.01, 0.02, -0.01, 0.03]
    eth = [0.004, 0.008, 0.001, 0.015]
    values = (
        BetaToETHFactor(lookback=3)
        .compute(_eth_frame(asset, eth))
        .get_column("beta_to_eth")
        .to_list()
    )
    assert values[0] is None
    assert values[1] is None
    cov = _population_cov(asset[0:3], eth[0:3])
    var = _population_std(eth[0:3]) ** 2
    assert values[2] == pytest.approx(cov / var)


def test_beta_to_eth_validation_and_edges() -> None:
    """Beta-to-ETH covers validation and contracts."""
    with pytest.raises(
        ValidationError,
        match="lookback must be an integer greater than or equal to 2",
    ):
        BetaToETHFactor(lookback=1)

    factor = BetaToETHFactor(lookback=2)
    _assert_missing_column(
        factor,
        missing="eth_return",
        error_code="FACTOR-BETA-TO-ETH-002",
        frame=pl.DataFrame({"asset_return": [0.01]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="beta_to_eth",
        frame=_eth_frame([0.01, 0.02, -0.01], [0.0, 0.01, 0.005]),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="beta_to_eth",
        columns=("asset_return", "eth_return"),
        single_expected=None,
    )
    _assert_determinism(
        lambda: BetaToETHFactor(lookback=2),
        output_column="beta_to_eth",
        frame=_eth_frame([0.01, 0.02, -0.01], [0.0, 0.01, 0.005]),
    )


# ---------------------------------------------------------------------------
# CorrelationBTCFactor
# ---------------------------------------------------------------------------


def test_correlation_btc_metadata_and_math() -> None:
    """BTC correlation matches population Pearson correlation."""
    factor = CorrelationBTCFactor()
    assert factor.name == "correlation_btc"
    assert factor.category == "relative"
    assert factor.lookback == 20
    assert CorrelationBTCFactor is CorrelationBTCFactorDirect

    asset = [0.01, 0.02, -0.01, 0.03]
    btc = [0.005, 0.01, 0.0, 0.02]
    values = (
        CorrelationBTCFactor(lookback=3)
        .compute(_btc_frame(asset, btc))
        .get_column("correlation_btc")
        .to_list()
    )
    assert values[0] is None
    assert values[1] is None
    cov = _population_cov(asset[0:3], btc[0:3])
    expected = cov / (_population_std(asset[0:3]) * _population_std(btc[0:3]))
    assert values[2] == pytest.approx(expected)


def test_correlation_btc_validation_and_edges() -> None:
    """BTC correlation covers zero std, validation, and contracts."""
    with pytest.raises(
        ValidationError,
        match="lookback must be an integer greater than or equal to 2",
    ):
        CorrelationBTCFactor(lookback=1)

    factor = CorrelationBTCFactor(lookback=2)
    _assert_missing_column(
        factor,
        missing="btc_return",
        error_code="FACTOR-CORRELATION-BTC-002",
        frame=pl.DataFrame({"asset_return": [0.01]}),
    )
    zero_std = (
        factor.compute(_btc_frame([0.01, 0.01], [0.02, 0.03]))
        .get_column("correlation_btc")
        .to_list()
    )
    assert zero_std[1] is None
    _assert_protocol_and_immutability(
        factor,
        output_column="correlation_btc",
        frame=_btc_frame([0.01, 0.02, -0.01], [0.0, 0.01, 0.005]),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="correlation_btc",
        columns=("asset_return", "btc_return"),
        single_expected=None,
    )
    _assert_determinism(
        lambda: CorrelationBTCFactor(lookback=2),
        output_column="correlation_btc",
        frame=_btc_frame([0.01, 0.02, -0.01], [0.0, 0.01, 0.005]),
    )


# ---------------------------------------------------------------------------
# CorrelationETHFactor
# ---------------------------------------------------------------------------


def test_correlation_eth_metadata_and_math() -> None:
    """ETH correlation matches population Pearson correlation."""
    factor = CorrelationETHFactor()
    assert factor.name == "correlation_eth"
    assert factor.category == "relative"
    assert CorrelationETHFactor is CorrelationETHFactorDirect

    asset = [0.01, 0.02, -0.01, 0.03]
    eth = [0.004, 0.008, 0.001, 0.015]
    values = (
        CorrelationETHFactor(lookback=3)
        .compute(_eth_frame(asset, eth))
        .get_column("correlation_eth")
        .to_list()
    )
    assert values[0] is None
    assert values[1] is None
    cov = _population_cov(asset[0:3], eth[0:3])
    expected = cov / (_population_std(asset[0:3]) * _population_std(eth[0:3]))
    assert values[2] == pytest.approx(expected)


def test_correlation_eth_validation_and_edges() -> None:
    """ETH correlation covers validation and contracts."""
    with pytest.raises(
        ValidationError,
        match="lookback must be an integer greater than or equal to 2",
    ):
        CorrelationETHFactor(lookback=1)

    factor = CorrelationETHFactor(lookback=2)
    _assert_missing_column(
        factor,
        missing="eth_return",
        error_code="FACTOR-CORRELATION-ETH-002",
        frame=pl.DataFrame({"asset_return": [0.01]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="correlation_eth",
        frame=_eth_frame([0.01, 0.02, -0.01], [0.0, 0.01, 0.005]),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="correlation_eth",
        columns=("asset_return", "eth_return"),
        single_expected=None,
    )
    _assert_determinism(
        lambda: CorrelationETHFactor(lookback=2),
        output_column="correlation_eth",
        frame=_eth_frame([0.01, 0.02, -0.01], [0.0, 0.01, 0.005]),
    )


# ---------------------------------------------------------------------------
# TrackingErrorFactor
# ---------------------------------------------------------------------------


def test_tracking_error_metadata_and_math() -> None:
    """Tracking error matches rolling std of active returns."""
    factor = TrackingErrorFactor()
    assert factor.name == "tracking_error"
    assert factor.category == "relative"
    assert factor.lookback == 20
    assert factor.required_features == ("asset_return", "benchmark_return")
    assert TrackingErrorFactor is TrackingErrorFactorDirect

    asset = [0.01, 0.02, -0.01, 0.03]
    bench = [0.005, 0.01, 0.0, 0.02]
    values = (
        TrackingErrorFactor(lookback=3)
        .compute(_benchmark_frame(asset, bench))
        .get_column("tracking_error")
        .to_list()
    )
    assert values[0] is None
    assert values[1] is None
    active = [a - b for a, b in zip(asset[0:3], bench[0:3], strict=True)]
    assert values[2] == pytest.approx(_population_std(active))


def test_tracking_error_validation_and_edges() -> None:
    """Tracking error covers validation and contracts."""
    with pytest.raises(
        ValidationError,
        match="lookback must be an integer greater than or equal to 2",
    ):
        TrackingErrorFactor(lookback=1)

    factor = TrackingErrorFactor(lookback=2)
    _assert_missing_column(
        factor,
        missing="benchmark_return",
        error_code="FACTOR-TRACKING-ERROR-002",
        frame=pl.DataFrame({"asset_return": [0.01]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="tracking_error",
        frame=_benchmark_frame([0.01, 0.02, -0.01], [0.0, 0.01, 0.005]),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="tracking_error",
        columns=("asset_return", "benchmark_return"),
        single_expected=None,
    )
    null_values = (
        factor.compute(_benchmark_frame([0.01, None], [0.0, 0.01]))
        .get_column("tracking_error")
        .to_list()
    )
    assert null_values[1] is None
    _assert_determinism(
        lambda: TrackingErrorFactor(lookback=2),
        output_column="tracking_error",
        frame=_benchmark_frame([0.01, 0.02, -0.01], [0.0, 0.01, 0.005]),
    )


# ---------------------------------------------------------------------------
# RelativeVolatilityFactor
# ---------------------------------------------------------------------------


def test_relative_volatility_metadata_and_math() -> None:
    """Relative volatility matches asset std divided by benchmark std."""
    factor = RelativeVolatilityFactor()
    assert factor.name == "relative_volatility"
    assert factor.category == "relative"
    assert factor.lookback == 20
    assert RelativeVolatilityFactor is RelativeVolatilityFactorDirect

    asset = [0.01, 0.02, -0.01, 0.03]
    bench = [0.005, 0.01, 0.0, 0.02]
    values = (
        RelativeVolatilityFactor(lookback=3)
        .compute(_benchmark_frame(asset, bench))
        .get_column("relative_volatility")
        .to_list()
    )
    assert values[0] is None
    assert values[1] is None
    expected = _population_std(asset[0:3]) / _population_std(bench[0:3])
    assert values[2] == pytest.approx(expected)


def test_relative_volatility_validation_and_edges() -> None:
    """Relative volatility covers zero denom, validation, and contracts."""
    with pytest.raises(
        ValidationError,
        match="lookback must be an integer greater than or equal to 2",
    ):
        RelativeVolatilityFactor(lookback=1)

    factor = RelativeVolatilityFactor(lookback=2)
    _assert_missing_column(
        factor,
        missing="benchmark_return",
        error_code="FACTOR-RELATIVE-VOLATILITY-002",
        frame=pl.DataFrame({"asset_return": [0.01]}),
    )
    zero_denom = (
        factor.compute(_benchmark_frame([0.01, 0.02], [0.01, 0.01]))
        .get_column("relative_volatility")
        .to_list()
    )
    assert zero_denom[1] is None
    _assert_protocol_and_immutability(
        factor,
        output_column="relative_volatility",
        frame=_benchmark_frame([0.01, 0.02, -0.01], [0.0, 0.01, 0.005]),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="relative_volatility",
        columns=("asset_return", "benchmark_return"),
        single_expected=None,
    )
    _assert_determinism(
        lambda: RelativeVolatilityFactor(lookback=2),
        output_column="relative_volatility",
        frame=_benchmark_frame([0.01, 0.02, -0.01], [0.0, 0.01, 0.005]),
    )
