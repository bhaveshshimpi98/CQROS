"""Unit tests for CQROS Batch-7 open interest and positioning factors."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import polars as pl
import pytest

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError
from cqros.factors.interfaces import Factor
from cqros.factors.open_interest import (
    OpenInterestAccelerationFactor,
    OpenInterestFundingDivergenceFactor,
    OpenInterestIntensityFactor,
    OpenInterestLevelFactor,
    OpenInterestMomentumFactor,
    OpenInterestPriceDivergenceFactor,
    OpenInterestTrendFactor,
    OpenInterestVolatilityFactor,
    OpenInterestVolumeRatioFactor,
    OpenInterestZScoreFactor,
)
from cqros.factors.open_interest.open_interest_acceleration import (
    OpenInterestAccelerationFactor as OpenInterestAccelerationFactorDirect,
)
from cqros.factors.open_interest.open_interest_funding_divergence import (
    OpenInterestFundingDivergenceFactor as OpenInterestFundingDivergenceFactorDirect,
)
from cqros.factors.open_interest.open_interest_intensity import (
    OpenInterestIntensityFactor as OpenInterestIntensityFactorDirect,
)
from cqros.factors.open_interest.open_interest_level import (
    OpenInterestLevelFactor as OpenInterestLevelFactorDirect,
)
from cqros.factors.open_interest.open_interest_momentum import (
    OpenInterestMomentumFactor as OpenInterestMomentumFactorDirect,
)
from cqros.factors.open_interest.open_interest_price_divergence import (
    OpenInterestPriceDivergenceFactor as OpenInterestPriceDivergenceFactorDirect,
)
from cqros.factors.open_interest.open_interest_trend import (
    OpenInterestTrendFactor as OpenInterestTrendFactorDirect,
)
from cqros.factors.open_interest.open_interest_volatility import (
    OpenInterestVolatilityFactor as OpenInterestVolatilityFactorDirect,
)
from cqros.factors.open_interest.open_interest_volume_ratio import (
    OpenInterestVolumeRatioFactor as OpenInterestVolumeRatioFactorDirect,
)
from cqros.factors.open_interest.open_interest_zscore import (
    OpenInterestZScoreFactor as OpenInterestZScoreFactorDirect,
)


def _oi_frame(values: Sequence[float | None]) -> pl.DataFrame:
    """Return an open-interest-only DataFrame."""
    return pl.DataFrame({"open_interest": list(values)})


def _oi_close_frame(
    oi: Sequence[float | None],
    closes: Sequence[float | None],
) -> pl.DataFrame:
    """Return an open interest / close DataFrame."""
    return pl.DataFrame({"open_interest": list(oi), "close": list(closes)})


def _oi_funding_frame(
    oi: Sequence[float | None],
    rates: Sequence[float | None],
) -> pl.DataFrame:
    """Return an open interest / funding DataFrame."""
    return pl.DataFrame({"open_interest": list(oi), "funding_rate": list(rates)})


def _oi_volume_frame(
    oi: Sequence[float | None],
    volumes: Sequence[float | None],
) -> pl.DataFrame:
    """Return an open interest / volume DataFrame."""
    return pl.DataFrame({"open_interest": list(oi), "volume": list(volumes)})


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


def _population_std(values: Sequence[float]) -> float:
    """Return population standard deviation for a fully observed window."""
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


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
# OpenInterestLevelFactor
# ---------------------------------------------------------------------------


def test_open_interest_level_metadata_and_math() -> None:
    """Open interest level mirrors the open_interest column."""
    factor = OpenInterestLevelFactor()
    assert factor.name == "open_interest_level"
    assert factor.version == "1.0.0"
    assert factor.category == "open_interest"
    assert factor.lookback == 0
    assert factor.required_features == ("open_interest",)
    assert factor.produced_columns == ("open_interest_level",)
    assert factor.metadata.name == "open_interest_level"
    assert OpenInterestLevelFactor is OpenInterestLevelFactorDirect

    frame = _oi_frame([1000.0, 0.0, 1500.0, None])
    values = factor.compute(frame).get_column("open_interest_level").to_list()
    assert values == [1000.0, 0.0, 1500.0, None]


def test_open_interest_level_validation_and_edges() -> None:
    """Open interest level covers validation and contracts."""
    with pytest.raises(ValidationError, match="lookback must be 0"):
        OpenInterestLevelFactor(lookback=1)

    factor = OpenInterestLevelFactor()
    _assert_missing_column(
        factor,
        missing="open_interest",
        error_code="FACTOR-OPEN-INTEREST-LEVEL-002",
        frame=pl.DataFrame({"close": [1.0]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="open_interest_level",
        frame=_oi_frame([1000.0, 0.0, 1200.0]),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="open_interest_level",
        columns=("open_interest",),
        single_expected=10.0,
    )
    _assert_determinism(
        OpenInterestLevelFactor,
        output_column="open_interest_level",
        frame=_oi_frame([1000.0, 0.0, 1200.0]),
    )


# ---------------------------------------------------------------------------
# OpenInterestMomentumFactor
# ---------------------------------------------------------------------------


def test_open_interest_momentum_metadata_and_math() -> None:
    """Open interest momentum matches absolute change over lookback."""
    factor = OpenInterestMomentumFactor()
    assert factor.name == "open_interest_momentum"
    assert factor.lookback == 20
    assert OpenInterestMomentumFactor is OpenInterestMomentumFactorDirect

    values = (
        OpenInterestMomentumFactor(lookback=2)
        .compute(_oi_frame([100.0, 110.0, 130.0, 125.0]))
        .get_column("open_interest_momentum")
        .to_list()
    )
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx(30.0)
    assert values[3] == pytest.approx(15.0)


def test_open_interest_momentum_validation_and_edges() -> None:
    """Open interest momentum covers trends, nulls, and contracts."""
    with pytest.raises(ValidationError, match="lookback must be an integer greater than 0"):
        OpenInterestMomentumFactor(lookback=0)

    rising = (
        OpenInterestMomentumFactor(lookback=2)
        .compute(_oi_frame([100.0, 110.0, 130.0, 160.0]))
        .get_column("open_interest_momentum")
        .to_list()
    )
    falling = (
        OpenInterestMomentumFactor(lookback=2)
        .compute(_oi_frame([160.0, 130.0, 110.0, 100.0]))
        .get_column("open_interest_momentum")
        .to_list()
    )
    zero = (
        OpenInterestMomentumFactor(lookback=2)
        .compute(_oi_frame([0.0, 0.0, 0.0, 0.0]))
        .get_column("open_interest_momentum")
        .to_list()
    )
    assert rising[3] == pytest.approx(50.0)
    assert falling[3] == pytest.approx(-30.0)
    assert zero[3] == pytest.approx(0.0)

    null_values = (
        OpenInterestMomentumFactor(lookback=2)
        .compute(_oi_frame([100.0, None, 130.0, 140.0]))
        .get_column("open_interest_momentum")
        .to_list()
    )
    assert null_values[2] == pytest.approx(30.0)
    assert null_values[3] is None

    factor = OpenInterestMomentumFactor(lookback=2)
    _assert_missing_column(
        factor,
        missing="open_interest",
        error_code="FACTOR-OPEN-INTEREST-MOMENTUM-002",
        frame=pl.DataFrame({"close": [1.0, 2.0]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="open_interest_momentum",
        frame=_oi_frame([100.0, 110.0, 120.0, 130.0]),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="open_interest_momentum",
        columns=("open_interest",),
    )
    _assert_determinism(
        lambda: OpenInterestMomentumFactor(lookback=2),
        output_column="open_interest_momentum",
        frame=_oi_frame([100.0, 110.0, 120.0, 130.0]),
    )


# ---------------------------------------------------------------------------
# OpenInterestZScoreFactor
# ---------------------------------------------------------------------------


def test_open_interest_zscore_metadata_and_math() -> None:
    """Open interest z-score matches population z-score over lookback."""
    factor = OpenInterestZScoreFactor()
    assert factor.name == "open_interest_zscore"
    assert factor.lookback == 20
    assert OpenInterestZScoreFactor is OpenInterestZScoreFactorDirect

    oi = [100.0, 120.0, 110.0, 160.0]
    values = (
        OpenInterestZScoreFactor(lookback=2)
        .compute(_oi_frame(oi))
        .get_column("open_interest_zscore")
        .to_list()
    )
    assert values[0] is None
    assert values[1] == pytest.approx(_population_zscore(oi[0:2]))
    assert values[2] == pytest.approx(_population_zscore(oi[1:3]))
    assert values[3] == pytest.approx(_population_zscore(oi[2:4]))


def test_open_interest_zscore_validation_and_edges() -> None:
    """Open interest z-score covers zero std and contracts."""
    with pytest.raises(
        ValidationError,
        match="lookback must be an integer greater than or equal to 2",
    ):
        OpenInterestZScoreFactor(lookback=1)

    constant = (
        OpenInterestZScoreFactor(lookback=2)
        .compute(_oi_frame([100.0, 100.0, 100.0]))
        .get_column("open_interest_zscore")
        .to_list()
    )
    assert constant[0] is None
    assert constant[1] == pytest.approx(0.0)

    factor = OpenInterestZScoreFactor(lookback=2)
    _assert_missing_column(
        factor,
        missing="open_interest",
        error_code="FACTOR-OPEN-INTEREST-ZSCORE-002",
        frame=pl.DataFrame({"close": [1.0, 2.0]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="open_interest_zscore",
        frame=_oi_frame([100.0, 110.0, 120.0, 130.0]),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="open_interest_zscore",
        columns=("open_interest",),
    )
    _assert_determinism(
        lambda: OpenInterestZScoreFactor(lookback=2),
        output_column="open_interest_zscore",
        frame=_oi_frame([100.0, 110.0, 120.0, 130.0]),
    )


# ---------------------------------------------------------------------------
# OpenInterestAccelerationFactor
# ---------------------------------------------------------------------------


def test_open_interest_acceleration_metadata_and_math() -> None:
    """Open interest acceleration is the first difference of momentum."""
    factor = OpenInterestAccelerationFactor()
    assert factor.name == "open_interest_acceleration"
    assert factor.lookback == 20
    assert OpenInterestAccelerationFactor is OpenInterestAccelerationFactorDirect

    values = (
        OpenInterestAccelerationFactor(lookback=2)
        .compute(_oi_frame([100.0, 110.0, 130.0, 160.0, 180.0]))
        .get_column("open_interest_acceleration")
        .to_list()
    )
    assert values[0] is None
    assert values[1] is None
    assert values[2] is None
    assert values[3] == pytest.approx(50.0 - 30.0)
    assert values[4] == pytest.approx(50.0 - 50.0)


def test_open_interest_acceleration_validation_and_edges() -> None:
    """Open interest acceleration covers warm-up and contracts."""
    with pytest.raises(ValidationError, match="lookback must be an integer greater than 0"):
        OpenInterestAccelerationFactor(lookback=0)

    factor = OpenInterestAccelerationFactor(lookback=2)
    _assert_missing_column(
        factor,
        missing="open_interest",
        error_code="FACTOR-OPEN-INTEREST-ACCELERATION-002",
        frame=pl.DataFrame({"close": [1.0, 2.0]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="open_interest_acceleration",
        frame=_oi_frame([100.0, 110.0, 120.0, 130.0, 140.0]),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="open_interest_acceleration",
        columns=("open_interest",),
    )
    _assert_determinism(
        lambda: OpenInterestAccelerationFactor(lookback=2),
        output_column="open_interest_acceleration",
        frame=_oi_frame([100.0, 110.0, 120.0, 130.0, 140.0]),
    )


# ---------------------------------------------------------------------------
# OpenInterestVolatilityFactor
# ---------------------------------------------------------------------------


def test_open_interest_volatility_metadata_and_math() -> None:
    """Open interest volatility matches population std over lookback."""
    factor = OpenInterestVolatilityFactor()
    assert factor.name == "open_interest_volatility"
    assert factor.lookback == 20
    assert OpenInterestVolatilityFactor is OpenInterestVolatilityFactorDirect

    oi = [100.0, 120.0, 110.0, 160.0]
    values = (
        OpenInterestVolatilityFactor(lookback=2)
        .compute(_oi_frame(oi))
        .get_column("open_interest_volatility")
        .to_list()
    )
    assert values[0] is None
    assert values[1] == pytest.approx(_population_std(oi[0:2]))
    assert values[2] == pytest.approx(_population_std(oi[1:3]))
    assert values[3] == pytest.approx(_population_std(oi[2:4]))


def test_open_interest_volatility_validation_and_edges() -> None:
    """Open interest volatility covers zero variance and contracts."""
    with pytest.raises(
        ValidationError,
        match="lookback must be an integer greater than or equal to 2",
    ):
        OpenInterestVolatilityFactor(lookback=1)

    constant = (
        OpenInterestVolatilityFactor(lookback=2)
        .compute(_oi_frame([100.0, 100.0, 100.0]))
        .get_column("open_interest_volatility")
        .to_list()
    )
    assert constant[1] == pytest.approx(0.0)

    factor = OpenInterestVolatilityFactor(lookback=2)
    _assert_missing_column(
        factor,
        missing="open_interest",
        error_code="FACTOR-OPEN-INTEREST-VOLATILITY-002",
        frame=pl.DataFrame({"close": [1.0, 2.0]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="open_interest_volatility",
        frame=_oi_frame([100.0, 110.0, 120.0, 130.0]),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="open_interest_volatility",
        columns=("open_interest",),
    )
    _assert_determinism(
        lambda: OpenInterestVolatilityFactor(lookback=2),
        output_column="open_interest_volatility",
        frame=_oi_frame([100.0, 110.0, 120.0, 130.0]),
    )


# ---------------------------------------------------------------------------
# OpenInterestPriceDivergenceFactor
# ---------------------------------------------------------------------------


def test_open_interest_price_divergence_metadata_and_math() -> None:
    """OI/price divergence matches fractional momentum difference."""
    factor = OpenInterestPriceDivergenceFactor()
    assert factor.name == "open_interest_price_divergence"
    assert factor.lookback == 20
    assert factor.required_features == ("open_interest", "close")
    assert OpenInterestPriceDivergenceFactor is OpenInterestPriceDivergenceFactorDirect

    oi = [100.0, 110.0, 130.0, 140.0]
    closes = [10.0, 11.0, 12.0, 11.0]
    values = (
        OpenInterestPriceDivergenceFactor(lookback=2)
        .compute(_oi_close_frame(oi, closes))
        .get_column("open_interest_price_divergence")
        .to_list()
    )
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx((130.0 / 100.0 - 1.0) - (12.0 / 10.0 - 1.0))
    assert values[3] == pytest.approx((140.0 / 110.0 - 1.0) - (11.0 / 11.0 - 1.0))


def test_open_interest_price_divergence_validation_and_edges() -> None:
    """OI/price divergence covers scenarios, zeros, and contracts."""
    with pytest.raises(ValidationError, match="lookback must be an integer greater than 0"):
        OpenInterestPriceDivergenceFactor(lookback=0)

    # OI rising faster than price → positive divergence.
    rising_div = (
        OpenInterestPriceDivergenceFactor(lookback=1)
        .compute(_oi_close_frame([100.0, 150.0], [10.0, 11.0]))
        .get_column("open_interest_price_divergence")
        .to_list()
    )
    assert rising_div[1] == pytest.approx(0.5 - 0.1)

    # Price rising while OI falls → negative divergence.
    falling_div = (
        OpenInterestPriceDivergenceFactor(lookback=1)
        .compute(_oi_close_frame([100.0, 90.0], [10.0, 12.0]))
        .get_column("open_interest_price_divergence")
        .to_list()
    )
    assert falling_div[1] == pytest.approx(-0.1 - 0.2)

    zero_prior = (
        OpenInterestPriceDivergenceFactor(lookback=1)
        .compute(_oi_close_frame([0.0, 10.0], [10.0, 11.0]))
        .get_column("open_interest_price_divergence")
        .to_list()
    )
    assert zero_prior[1] is None

    factor = OpenInterestPriceDivergenceFactor(lookback=2)
    _assert_missing_column(
        factor,
        missing="close",
        error_code="FACTOR-OPEN-INTEREST-PRICE-DIVERGENCE-002",
        frame=pl.DataFrame({"open_interest": [1.0, 2.0]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="open_interest_price_divergence",
        frame=_oi_close_frame([100.0, 110.0, 120.0, 130.0], [10.0, 11.0, 12.0, 13.0]),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="open_interest_price_divergence",
        columns=("open_interest", "close"),
    )
    _assert_determinism(
        lambda: OpenInterestPriceDivergenceFactor(lookback=2),
        output_column="open_interest_price_divergence",
        frame=_oi_close_frame([100.0, 110.0, 120.0, 130.0], [10.0, 11.0, 12.0, 13.0]),
    )


# ---------------------------------------------------------------------------
# OpenInterestFundingDivergenceFactor
# ---------------------------------------------------------------------------


def test_open_interest_funding_divergence_metadata_and_math() -> None:
    """OI/funding divergence matches z-score difference."""
    factor = OpenInterestFundingDivergenceFactor()
    assert factor.name == "open_interest_funding_divergence"
    assert factor.lookback == 20
    assert factor.required_features == ("open_interest", "funding_rate")
    assert OpenInterestFundingDivergenceFactor is OpenInterestFundingDivergenceFactorDirect

    oi = [100.0, 120.0, 110.0, 160.0]
    rates = [0.0001, 0.0003, 0.0002, 0.0004]
    values = (
        OpenInterestFundingDivergenceFactor(lookback=2)
        .compute(_oi_funding_frame(oi, rates))
        .get_column("open_interest_funding_divergence")
        .to_list()
    )
    assert values[0] is None
    expected_1 = _population_zscore(oi[0:2])
    funding_1 = _population_zscore(rates[0:2])
    assert expected_1 is not None and funding_1 is not None
    assert values[1] == pytest.approx(expected_1 - funding_1)
    expected_3 = _population_zscore(oi[2:4])
    funding_3 = _population_zscore(rates[2:4])
    assert expected_3 is not None and funding_3 is not None
    assert values[3] == pytest.approx(expected_3 - funding_3)


def test_open_interest_funding_divergence_validation_and_edges() -> None:
    """OI/funding divergence covers zero std, scenarios, and contracts."""
    with pytest.raises(
        ValidationError,
        match="lookback must be an integer greater than or equal to 2",
    ):
        OpenInterestFundingDivergenceFactor(lookback=1)

    constant_oi = (
        OpenInterestFundingDivergenceFactor(lookback=2)
        .compute(_oi_funding_frame([100.0, 100.0, 100.0], [0.0001, 0.0002, 0.0003]))
        .get_column("open_interest_funding_divergence")
        .to_list()
    )
    assert constant_oi[0] is None
    assert constant_oi[1] == pytest.approx(0.0 - float(_population_zscore([0.0001, 0.0002])))
    assert constant_oi[2] == pytest.approx(0.0 - float(_population_zscore([0.0002, 0.0003])))

    # Rising OI with falling funding → positive divergence at end.
    divergent = (
        OpenInterestFundingDivergenceFactor(lookback=2)
        .compute(_oi_funding_frame([100.0, 150.0], [0.0004, 0.0001]))
        .get_column("open_interest_funding_divergence")
        .to_list()
    )
    assert divergent[1] is not None
    assert divergent[1] > 0.0

    factor = OpenInterestFundingDivergenceFactor(lookback=2)
    _assert_missing_column(
        factor,
        missing="funding_rate",
        error_code="FACTOR-OPEN-INTEREST-FUNDING-DIVERGENCE-002",
        frame=pl.DataFrame({"open_interest": [1.0, 2.0]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="open_interest_funding_divergence",
        frame=_oi_funding_frame(
            [100.0, 110.0, 120.0, 130.0],
            [0.0001, 0.0002, 0.0003, 0.0004],
        ),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="open_interest_funding_divergence",
        columns=("open_interest", "funding_rate"),
    )
    _assert_determinism(
        lambda: OpenInterestFundingDivergenceFactor(lookback=2),
        output_column="open_interest_funding_divergence",
        frame=_oi_funding_frame(
            [100.0, 110.0, 120.0, 130.0],
            [0.0001, 0.0002, 0.0003, 0.0004],
        ),
    )


# ---------------------------------------------------------------------------
# OpenInterestVolumeRatioFactor
# ---------------------------------------------------------------------------


def test_open_interest_volume_ratio_metadata_and_math() -> None:
    """OI/volume ratio matches rolling mean OI over rolling mean volume."""
    factor = OpenInterestVolumeRatioFactor()
    assert factor.name == "open_interest_volume_ratio"
    assert factor.lookback == 20
    assert OpenInterestVolumeRatioFactor is OpenInterestVolumeRatioFactorDirect

    oi = [100.0, 200.0, 300.0, 400.0]
    volumes = [10.0, 20.0, 30.0, 40.0]
    values = (
        OpenInterestVolumeRatioFactor(lookback=2)
        .compute(_oi_volume_frame(oi, volumes))
        .get_column("open_interest_volume_ratio")
        .to_list()
    )
    assert values[0] is None
    assert values[1] == pytest.approx(150.0 / 15.0)
    assert values[2] == pytest.approx(250.0 / 25.0)
    assert values[3] == pytest.approx(350.0 / 35.0)


def test_open_interest_volume_ratio_validation_and_edges() -> None:
    """OI/volume ratio covers zero volume mean and contracts."""
    with pytest.raises(ValidationError, match="lookback must be an integer greater than 0"):
        OpenInterestVolumeRatioFactor(lookback=0)

    zero_vol = (
        OpenInterestVolumeRatioFactor(lookback=2)
        .compute(_oi_volume_frame([100.0, 200.0], [0.0, 0.0]))
        .get_column("open_interest_volume_ratio")
        .to_list()
    )
    assert zero_vol[1] is None

    factor = OpenInterestVolumeRatioFactor(lookback=2)
    _assert_missing_column(
        factor,
        missing="volume",
        error_code="FACTOR-OPEN-INTEREST-VOLUME-RATIO-002",
        frame=pl.DataFrame({"open_interest": [1.0, 2.0]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="open_interest_volume_ratio",
        frame=_oi_volume_frame([100.0, 200.0, 300.0], [10.0, 20.0, 30.0]),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="open_interest_volume_ratio",
        columns=("open_interest", "volume"),
    )
    _assert_determinism(
        lambda: OpenInterestVolumeRatioFactor(lookback=2),
        output_column="open_interest_volume_ratio",
        frame=_oi_volume_frame([100.0, 200.0, 300.0], [10.0, 20.0, 30.0]),
    )


# ---------------------------------------------------------------------------
# OpenInterestIntensityFactor
# ---------------------------------------------------------------------------


def test_open_interest_intensity_metadata_and_math() -> None:
    """OI intensity matches current OI over rolling mean volume."""
    factor = OpenInterestIntensityFactor()
    assert factor.name == "open_interest_intensity"
    assert factor.lookback == 20
    assert OpenInterestIntensityFactor is OpenInterestIntensityFactorDirect

    oi = [100.0, 200.0, 300.0, 400.0]
    volumes = [10.0, 20.0, 30.0, 40.0]
    values = (
        OpenInterestIntensityFactor(lookback=2)
        .compute(_oi_volume_frame(oi, volumes))
        .get_column("open_interest_intensity")
        .to_list()
    )
    assert values[0] is None
    assert values[1] == pytest.approx(200.0 / 15.0)
    assert values[2] == pytest.approx(300.0 / 25.0)
    assert values[3] == pytest.approx(400.0 / 35.0)


def test_open_interest_intensity_validation_and_edges() -> None:
    """OI intensity covers zero volume mean, zero OI, and contracts."""
    with pytest.raises(ValidationError, match="lookback must be an integer greater than 0"):
        OpenInterestIntensityFactor(lookback=0)

    zero_vol = (
        OpenInterestIntensityFactor(lookback=2)
        .compute(_oi_volume_frame([100.0, 200.0], [0.0, 0.0]))
        .get_column("open_interest_intensity")
        .to_list()
    )
    assert zero_vol[1] is None

    zero_oi = (
        OpenInterestIntensityFactor(lookback=2)
        .compute(_oi_volume_frame([0.0, 0.0], [10.0, 20.0]))
        .get_column("open_interest_intensity")
        .to_list()
    )
    assert zero_oi[1] == pytest.approx(0.0)

    factor = OpenInterestIntensityFactor(lookback=2)
    _assert_missing_column(
        factor,
        missing="open_interest",
        error_code="FACTOR-OPEN-INTEREST-INTENSITY-002",
        frame=pl.DataFrame({"volume": [1.0, 2.0]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="open_interest_intensity",
        frame=_oi_volume_frame([100.0, 200.0, 300.0], [10.0, 20.0, 30.0]),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="open_interest_intensity",
        columns=("open_interest", "volume"),
    )
    _assert_determinism(
        lambda: OpenInterestIntensityFactor(lookback=2),
        output_column="open_interest_intensity",
        frame=_oi_volume_frame([100.0, 200.0, 300.0], [10.0, 20.0, 30.0]),
    )


# ---------------------------------------------------------------------------
# OpenInterestTrendFactor
# ---------------------------------------------------------------------------


def test_open_interest_trend_metadata_and_math() -> None:
    """Open interest trend matches rolling OLS slope."""
    factor = OpenInterestTrendFactor()
    assert factor.name == "open_interest_trend"
    assert factor.lookback == 20
    assert OpenInterestTrendFactor is OpenInterestTrendFactorDirect

    oi = [100.0, 110.0, 130.0, 160.0]
    values = (
        OpenInterestTrendFactor(lookback=3)
        .compute(_oi_frame(oi))
        .get_column("open_interest_trend")
        .to_list()
    )
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx(_ols_slope(oi[0:3]))
    assert values[3] == pytest.approx(_ols_slope(oi[1:4]))


def test_open_interest_trend_validation_and_edges() -> None:
    """Open interest trend covers rising/falling slopes and contracts."""
    with pytest.raises(
        ValidationError,
        match="lookback must be an integer greater than or equal to 2",
    ):
        OpenInterestTrendFactor(lookback=1)

    rising = (
        OpenInterestTrendFactor(lookback=3)
        .compute(_oi_frame([100.0, 110.0, 120.0, 130.0]))
        .get_column("open_interest_trend")
        .to_list()
    )
    falling = (
        OpenInterestTrendFactor(lookback=3)
        .compute(_oi_frame([130.0, 120.0, 110.0, 100.0]))
        .get_column("open_interest_trend")
        .to_list()
    )
    assert rising[3] is not None and rising[3] > 0.0
    assert falling[3] is not None and falling[3] < 0.0

    factor = OpenInterestTrendFactor(lookback=3)
    _assert_missing_column(
        factor,
        missing="open_interest",
        error_code="FACTOR-OPEN-INTEREST-TREND-002",
        frame=pl.DataFrame({"close": [1.0, 2.0, 3.0]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="open_interest_trend",
        frame=_oi_frame([100.0, 110.0, 120.0, 130.0]),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="open_interest_trend",
        columns=("open_interest",),
    )
    _assert_determinism(
        lambda: OpenInterestTrendFactor(lookback=3),
        output_column="open_interest_trend",
        frame=_oi_frame([100.0, 110.0, 120.0, 130.0]),
    )
