"""Unit tests for CQROS Batch-6 funding and derivatives factors."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import polars as pl
import pytest

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError
from cqros.factors.funding import (
    BasisFactor,
    BasisMomentumFactor,
    BasisZScoreFactor,
    CarryFactor,
    FundingAccelerationFactor,
    FundingRateLevelFactor,
    FundingRateMomentumFactor,
    FundingRateZScoreFactor,
    FundingVolatilityFactor,
    PremiumIndexFactor,
)
from cqros.factors.funding.basis import BasisFactor as BasisFactorDirect
from cqros.factors.funding.basis_momentum import (
    BasisMomentumFactor as BasisMomentumFactorDirect,
)
from cqros.factors.funding.basis_zscore import BasisZScoreFactor as BasisZScoreFactorDirect
from cqros.factors.funding.carry import CarryFactor as CarryFactorDirect
from cqros.factors.funding.funding_acceleration import (
    FundingAccelerationFactor as FundingAccelerationFactorDirect,
)
from cqros.factors.funding.funding_rate_level import (
    FundingRateLevelFactor as FundingRateLevelFactorDirect,
)
from cqros.factors.funding.funding_rate_momentum import (
    FundingRateMomentumFactor as FundingRateMomentumFactorDirect,
)
from cqros.factors.funding.funding_rate_zscore import (
    FundingRateZScoreFactor as FundingRateZScoreFactorDirect,
)
from cqros.factors.funding.funding_volatility import (
    FundingVolatilityFactor as FundingVolatilityFactorDirect,
)
from cqros.factors.funding.premium_index import (
    PremiumIndexFactor as PremiumIndexFactorDirect,
)
from cqros.factors.interfaces import Factor


def _funding_frame(rates: Sequence[float | None]) -> pl.DataFrame:
    """Return a funding-rate-only DataFrame."""
    return pl.DataFrame({"funding_rate": list(rates)})


def _basis_frame(
    marks: Sequence[float | None],
    indexes: Sequence[float | None],
) -> pl.DataFrame:
    """Return a mark/index price DataFrame."""
    return pl.DataFrame({"mark_price": list(marks), "index_price": list(indexes)})


def _carry_frame(
    rates: Sequence[float | None],
    marks: Sequence[float | None],
    indexes: Sequence[float | None],
) -> pl.DataFrame:
    """Return a funding/mark/index DataFrame."""
    return pl.DataFrame(
        {
            "funding_rate": list(rates),
            "mark_price": list(marks),
            "index_price": list(indexes),
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


def _population_zscore(values: list[float]) -> float | None:
    """Return population z-score for a fully observed window."""
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    if std == 0.0:
        return 0.0
    return (values[-1] - mean) / std


def _population_std(values: list[float]) -> float:
    """Return population standard deviation for a fully observed window."""
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


# ---------------------------------------------------------------------------
# FundingRateLevelFactor
# ---------------------------------------------------------------------------


def test_funding_rate_level_metadata_and_math() -> None:
    """Funding rate level mirrors the funding_rate column."""
    factor = FundingRateLevelFactor()
    assert factor.name == "funding_rate_level"
    assert factor.version == "1.0.0"
    assert factor.category == "funding"
    assert factor.lookback == 0
    assert factor.required_features == ("funding_rate",)
    assert factor.produced_columns == ("funding_rate_level",)
    assert factor.metadata.name == "funding_rate_level"
    assert FundingRateLevelFactor is FundingRateLevelFactorDirect

    frame = _funding_frame([0.0001, 0.0, -0.0002, None])
    values = factor.compute(frame).get_column("funding_rate_level").to_list()
    assert values == [0.0001, 0.0, -0.0002, None]


def test_funding_rate_level_validation_and_edges() -> None:
    """Funding rate level covers validation, signs, and contracts."""
    with pytest.raises(ValidationError, match="lookback must be 0"):
        FundingRateLevelFactor(lookback=1)

    factor = FundingRateLevelFactor()
    _assert_missing_column(
        factor,
        missing="funding_rate",
        error_code="FACTOR-FUNDING-RATE-LEVEL-002",
        frame=pl.DataFrame({"close": [1.0]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="funding_rate_level",
        frame=_funding_frame([0.0001, -0.0002, 0.0]),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="funding_rate_level",
        columns=("funding_rate",),
        single_expected=10.0,
    )
    _assert_determinism(
        FundingRateLevelFactor,
        output_column="funding_rate_level",
        frame=_funding_frame([0.0001, -0.0002, 0.0]),
    )


# ---------------------------------------------------------------------------
# FundingRateMomentumFactor
# ---------------------------------------------------------------------------


def test_funding_rate_momentum_metadata_and_math() -> None:
    """Funding momentum matches absolute change over lookback."""
    factor = FundingRateMomentumFactor()
    assert factor.name == "funding_rate_momentum"
    assert factor.lookback == 20
    assert factor.category == "funding"
    assert FundingRateMomentumFactor is FundingRateMomentumFactorDirect

    rates = [0.0001, 0.0002, 0.0004, 0.0003]
    values = (
        FundingRateMomentumFactor(lookback=2)
        .compute(_funding_frame(rates))
        .get_column("funding_rate_momentum")
        .to_list()
    )
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx(0.0004 - 0.0001)
    assert values[3] == pytest.approx(0.0003 - 0.0002)


def test_funding_rate_momentum_validation_and_edges() -> None:
    """Funding momentum covers signs, warm-up, nulls, and contracts."""
    with pytest.raises(ValidationError, match="lookback must be an integer greater than 0"):
        FundingRateMomentumFactor(lookback=0)

    rising = (
        FundingRateMomentumFactor(lookback=2)
        .compute(_funding_frame([0.0001, 0.0002, 0.0004, 0.0008]))
        .get_column("funding_rate_momentum")
        .to_list()
    )
    falling = (
        FundingRateMomentumFactor(lookback=2)
        .compute(_funding_frame([0.0008, 0.0004, 0.0002, 0.0001]))
        .get_column("funding_rate_momentum")
        .to_list()
    )
    zero = (
        FundingRateMomentumFactor(lookback=2)
        .compute(_funding_frame([0.0, 0.0, 0.0, 0.0]))
        .get_column("funding_rate_momentum")
        .to_list()
    )
    assert rising[3] == pytest.approx(0.0006)
    assert falling[3] == pytest.approx(-0.0003)
    assert zero[3] == pytest.approx(0.0)

    null_values = (
        FundingRateMomentumFactor(lookback=2)
        .compute(_funding_frame([0.0001, None, 0.0003, 0.0004]))
        .get_column("funding_rate_momentum")
        .to_list()
    )
    assert null_values[0] is None
    assert null_values[1] is None
    assert null_values[2] == pytest.approx(0.0002)
    assert null_values[3] is None

    factor = FundingRateMomentumFactor(lookback=2)
    _assert_missing_column(
        factor,
        missing="funding_rate",
        error_code="FACTOR-FUNDING-RATE-MOMENTUM-002",
        frame=pl.DataFrame({"close": [1.0, 2.0]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="funding_rate_momentum",
        frame=_funding_frame([0.0001, 0.0002, 0.0003, 0.0004]),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="funding_rate_momentum",
        columns=("funding_rate",),
    )
    _assert_determinism(
        lambda: FundingRateMomentumFactor(lookback=2),
        output_column="funding_rate_momentum",
        frame=_funding_frame([0.0001, 0.0002, 0.0003, 0.0004]),
    )


# ---------------------------------------------------------------------------
# FundingRateZScoreFactor
# ---------------------------------------------------------------------------


def test_funding_rate_zscore_metadata_and_math() -> None:
    """Funding z-score matches population z-score over lookback."""
    factor = FundingRateZScoreFactor()
    assert factor.name == "funding_rate_zscore"
    assert factor.lookback == 20
    assert FundingRateZScoreFactor is FundingRateZScoreFactorDirect

    rates = [0.0001, 0.0003, 0.0002, 0.0006]
    values = (
        FundingRateZScoreFactor(lookback=2)
        .compute(_funding_frame(rates))
        .get_column("funding_rate_zscore")
        .to_list()
    )
    assert values[0] is None
    assert values[1] == pytest.approx(_population_zscore(rates[0:2]))
    assert values[2] == pytest.approx(_population_zscore(rates[1:3]))
    assert values[3] == pytest.approx(_population_zscore(rates[2:4]))


def test_funding_rate_zscore_validation_and_edges() -> None:
    """Funding z-score covers zero std, validation, and contracts."""
    with pytest.raises(
        ValidationError,
        match="lookback must be an integer greater than or equal to 2",
    ):
        FundingRateZScoreFactor(lookback=1)

    constant = (
        FundingRateZScoreFactor(lookback=2)
        .compute(_funding_frame([0.0001, 0.0001, 0.0001]))
        .get_column("funding_rate_zscore")
        .to_list()
    )
    assert constant[0] is None
    assert constant[1] == pytest.approx(0.0)
    assert constant[2] == pytest.approx(0.0)

    negative = (
        FundingRateZScoreFactor(lookback=2)
        .compute(_funding_frame([-0.0004, -0.0001, -0.0003]))
        .get_column("funding_rate_zscore")
        .to_list()
    )
    assert negative[1] == pytest.approx(_population_zscore([-0.0004, -0.0001]))

    factor = FundingRateZScoreFactor(lookback=2)
    _assert_missing_column(
        factor,
        missing="funding_rate",
        error_code="FACTOR-FUNDING-RATE-ZSCORE-002",
        frame=pl.DataFrame({"close": [1.0, 2.0]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="funding_rate_zscore",
        frame=_funding_frame([0.0001, 0.0002, 0.0003, 0.0004]),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="funding_rate_zscore",
        columns=("funding_rate",),
    )
    _assert_determinism(
        lambda: FundingRateZScoreFactor(lookback=2),
        output_column="funding_rate_zscore",
        frame=_funding_frame([0.0001, 0.0002, 0.0003, 0.0004]),
    )


# ---------------------------------------------------------------------------
# FundingAccelerationFactor
# ---------------------------------------------------------------------------


def test_funding_acceleration_metadata_and_math() -> None:
    """Funding acceleration is the first difference of momentum."""
    factor = FundingAccelerationFactor()
    assert factor.name == "funding_acceleration"
    assert factor.lookback == 20
    assert FundingAccelerationFactor is FundingAccelerationFactorDirect

    rates = [0.0001, 0.0002, 0.0004, 0.0007, 0.0009]
    values = (
        FundingAccelerationFactor(lookback=2)
        .compute(_funding_frame(rates))
        .get_column("funding_acceleration")
        .to_list()
    )
    assert values[0] is None
    assert values[1] is None
    assert values[2] is None
    assert values[3] == pytest.approx(0.0005 - 0.0003)
    assert values[4] == pytest.approx(0.0005 - 0.0005)


def test_funding_acceleration_validation_and_edges() -> None:
    """Funding acceleration covers warm-up, signs, and contracts."""
    with pytest.raises(ValidationError, match="lookback must be an integer greater than 0"):
        FundingAccelerationFactor(lookback=0)

    rising = (
        FundingAccelerationFactor(lookback=1)
        .compute(_funding_frame([0.0001, 0.0002, 0.0004, 0.0008]))
        .get_column("funding_acceleration")
        .to_list()
    )
    # momentum: None, 0.0001, 0.0002, 0.0004 → acceleration: None, None, 0.0001, 0.0002
    assert rising[0] is None
    assert rising[1] is None
    assert rising[2] == pytest.approx(0.0001)
    assert rising[3] == pytest.approx(0.0002)

    factor = FundingAccelerationFactor(lookback=2)
    _assert_missing_column(
        factor,
        missing="funding_rate",
        error_code="FACTOR-FUNDING-ACCELERATION-002",
        frame=pl.DataFrame({"close": [1.0, 2.0]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="funding_acceleration",
        frame=_funding_frame([0.0001, 0.0002, 0.0003, 0.0004, 0.0005]),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="funding_acceleration",
        columns=("funding_rate",),
    )
    _assert_determinism(
        lambda: FundingAccelerationFactor(lookback=2),
        output_column="funding_acceleration",
        frame=_funding_frame([0.0001, 0.0002, 0.0003, 0.0004, 0.0005]),
    )


# ---------------------------------------------------------------------------
# FundingVolatilityFactor
# ---------------------------------------------------------------------------


def test_funding_volatility_metadata_and_math() -> None:
    """Funding volatility matches population std over lookback."""
    factor = FundingVolatilityFactor()
    assert factor.name == "funding_volatility"
    assert factor.lookback == 20
    assert FundingVolatilityFactor is FundingVolatilityFactorDirect

    rates = [0.0001, 0.0003, 0.0002, 0.0006]
    values = (
        FundingVolatilityFactor(lookback=2)
        .compute(_funding_frame(rates))
        .get_column("funding_volatility")
        .to_list()
    )
    assert values[0] is None
    assert values[1] == pytest.approx(_population_std(rates[0:2]))
    assert values[2] == pytest.approx(_population_std(rates[1:3]))
    assert values[3] == pytest.approx(_population_std(rates[2:4]))


def test_funding_volatility_validation_and_edges() -> None:
    """Funding volatility covers zero variance, signs, and contracts."""
    with pytest.raises(
        ValidationError,
        match="lookback must be an integer greater than or equal to 2",
    ):
        FundingVolatilityFactor(lookback=1)

    constant = (
        FundingVolatilityFactor(lookback=2)
        .compute(_funding_frame([0.0001, 0.0001, 0.0001]))
        .get_column("funding_volatility")
        .to_list()
    )
    assert constant[1] == pytest.approx(0.0)

    mixed = (
        FundingVolatilityFactor(lookback=2)
        .compute(_funding_frame([-0.0002, 0.0, 0.0002]))
        .get_column("funding_volatility")
        .to_list()
    )
    assert mixed[1] == pytest.approx(_population_std([-0.0002, 0.0]))
    assert mixed[2] == pytest.approx(_population_std([0.0, 0.0002]))

    factor = FundingVolatilityFactor(lookback=2)
    _assert_missing_column(
        factor,
        missing="funding_rate",
        error_code="FACTOR-FUNDING-VOLATILITY-002",
        frame=pl.DataFrame({"close": [1.0, 2.0]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="funding_volatility",
        frame=_funding_frame([0.0001, 0.0002, 0.0003, 0.0004]),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="funding_volatility",
        columns=("funding_rate",),
    )
    _assert_determinism(
        lambda: FundingVolatilityFactor(lookback=2),
        output_column="funding_volatility",
        frame=_funding_frame([0.0001, 0.0002, 0.0003, 0.0004]),
    )


# ---------------------------------------------------------------------------
# PremiumIndexFactor
# ---------------------------------------------------------------------------


def test_premium_index_metadata_and_math() -> None:
    """Premium index factor mirrors the premium_index column."""
    factor = PremiumIndexFactor()
    assert factor.name == "premium_index_factor"
    assert factor.lookback == 0
    assert factor.category == "funding"
    assert factor.required_features == ("premium_index",)
    assert factor.produced_columns == ("premium_index_factor",)
    assert PremiumIndexFactor is PremiumIndexFactorDirect

    frame = pl.DataFrame({"premium_index": [0.001, 0.0, -0.002, None]})
    values = factor.compute(frame).get_column("premium_index_factor").to_list()
    assert values == [0.001, 0.0, -0.002, None]


def test_premium_index_validation_and_edges() -> None:
    """Premium index covers validation, nulls, and contracts."""
    with pytest.raises(ValidationError, match="lookback must be 0"):
        PremiumIndexFactor(lookback=5)

    factor = PremiumIndexFactor()
    _assert_missing_column(
        factor,
        missing="premium_index",
        error_code="FACTOR-PREMIUM-INDEX-002",
        frame=pl.DataFrame({"funding_rate": [0.001]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="premium_index_factor",
        frame=pl.DataFrame({"premium_index": [0.001, -0.002, 0.0]}),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="premium_index_factor",
        columns=("premium_index",),
        single_expected=10.0,
    )
    _assert_determinism(
        PremiumIndexFactor,
        output_column="premium_index_factor",
        frame=pl.DataFrame({"premium_index": [0.001, -0.002, 0.0]}),
    )


# ---------------------------------------------------------------------------
# BasisFactor
# ---------------------------------------------------------------------------


def test_basis_metadata_and_math() -> None:
    """Basis matches (mark - index) / index with zero-index nulls."""
    factor = BasisFactor()
    assert factor.name == "basis"
    assert factor.lookback == 0
    assert factor.required_features == ("mark_price", "index_price")
    assert factor.produced_columns == ("basis",)
    assert BasisFactor is BasisFactorDirect

    frame = _basis_frame([101.0, 100.0, 99.0, 50.0], [100.0, 100.0, 100.0, 0.0])
    values = factor.compute(frame).get_column("basis").to_list()
    assert values[0] == pytest.approx(0.01)
    assert values[1] == pytest.approx(0.0)
    assert values[2] == pytest.approx(-0.01)
    assert values[3] is None


def test_basis_validation_and_edges() -> None:
    """Basis covers validation, null inputs, and contracts."""
    with pytest.raises(ValidationError, match="lookback must be 0"):
        BasisFactor(lookback=1)

    null_frame = _basis_frame([101.0, None, 99.0], [100.0, 100.0, None])
    null_values = BasisFactor().compute(null_frame).get_column("basis").to_list()
    assert null_values[0] == pytest.approx(0.01)
    assert null_values[1] is None
    assert null_values[2] is None

    factor = BasisFactor()
    _assert_missing_column(
        factor,
        missing="mark_price",
        error_code="FACTOR-BASIS-002",
        frame=pl.DataFrame({"index_price": [100.0]}),
    )
    _assert_missing_column(
        factor,
        missing="index_price",
        error_code="FACTOR-BASIS-002",
        frame=pl.DataFrame({"mark_price": [100.0]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="basis",
        frame=_basis_frame([101.0, 99.0], [100.0, 100.0]),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="basis",
        columns=("mark_price", "index_price"),
        single_expected=0.0,
    )
    _assert_determinism(
        BasisFactor,
        output_column="basis",
        frame=_basis_frame([101.0, 99.0, 100.0], [100.0, 100.0, 100.0]),
    )


# ---------------------------------------------------------------------------
# BasisMomentumFactor
# ---------------------------------------------------------------------------


def test_basis_momentum_metadata_and_math() -> None:
    """Basis momentum matches absolute change in basis over lookback."""
    factor = BasisMomentumFactor()
    assert factor.name == "basis_momentum"
    assert factor.lookback == 20
    assert BasisMomentumFactor is BasisMomentumFactorDirect

    marks = [101.0, 102.0, 104.0, 103.0]
    indexes = [100.0, 100.0, 100.0, 100.0]
    bases = [(mark - index) / index for mark, index in zip(marks, indexes, strict=True)]
    values = (
        BasisMomentumFactor(lookback=2)
        .compute(_basis_frame(marks, indexes))
        .get_column("basis_momentum")
        .to_list()
    )
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx(bases[2] - bases[0])
    assert values[3] == pytest.approx(bases[3] - bases[1])


def test_basis_momentum_validation_and_edges() -> None:
    """Basis momentum covers signs, zero index, and contracts."""
    with pytest.raises(ValidationError, match="lookback must be an integer greater than 0"):
        BasisMomentumFactor(lookback=0)

    zero_index = (
        BasisMomentumFactor(lookback=1)
        .compute(_basis_frame([101.0, 102.0], [100.0, 0.0]))
        .get_column("basis_momentum")
        .to_list()
    )
    assert zero_index[0] is None
    assert zero_index[1] is None

    positive = (
        BasisMomentumFactor(lookback=1)
        .compute(_basis_frame([101.0, 103.0], [100.0, 100.0]))
        .get_column("basis_momentum")
        .to_list()
    )
    negative = (
        BasisMomentumFactor(lookback=1)
        .compute(_basis_frame([103.0, 101.0], [100.0, 100.0]))
        .get_column("basis_momentum")
        .to_list()
    )
    assert positive[1] == pytest.approx(0.02)
    assert negative[1] == pytest.approx(-0.02)

    factor = BasisMomentumFactor(lookback=2)
    _assert_missing_column(
        factor,
        missing="index_price",
        error_code="FACTOR-BASIS-MOMENTUM-002",
        frame=pl.DataFrame({"mark_price": [1.0, 2.0]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="basis_momentum",
        frame=_basis_frame([101.0, 102.0, 103.0, 104.0], [100.0, 100.0, 100.0, 100.0]),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="basis_momentum",
        columns=("mark_price", "index_price"),
    )
    _assert_determinism(
        lambda: BasisMomentumFactor(lookback=2),
        output_column="basis_momentum",
        frame=_basis_frame([101.0, 102.0, 103.0, 104.0], [100.0, 100.0, 100.0, 100.0]),
    )


# ---------------------------------------------------------------------------
# BasisZScoreFactor
# ---------------------------------------------------------------------------


def test_basis_zscore_metadata_and_math() -> None:
    """Basis z-score matches population z-score of basis over lookback."""
    factor = BasisZScoreFactor()
    assert factor.name == "basis_zscore"
    assert factor.lookback == 20
    assert BasisZScoreFactor is BasisZScoreFactorDirect

    marks = [101.0, 103.0, 102.0, 106.0]
    indexes = [100.0, 100.0, 100.0, 100.0]
    basis_values = [(mark - index) / index for mark, index in zip(marks, indexes, strict=True)]
    values = (
        BasisZScoreFactor(lookback=2)
        .compute(_basis_frame(marks, indexes))
        .get_column("basis_zscore")
        .to_list()
    )
    assert values[0] is None
    assert values[1] == pytest.approx(_population_zscore(basis_values[0:2]))
    assert values[2] == pytest.approx(_population_zscore(basis_values[1:3]))
    assert values[3] == pytest.approx(_population_zscore(basis_values[2:4]))


def test_basis_zscore_validation_and_edges() -> None:
    """Basis z-score covers zero std, zero index, and contracts."""
    with pytest.raises(
        ValidationError,
        match="lookback must be an integer greater than or equal to 2",
    ):
        BasisZScoreFactor(lookback=1)

    constant = (
        BasisZScoreFactor(lookback=2)
        .compute(_basis_frame([101.0, 101.0, 101.0], [100.0, 100.0, 100.0]))
        .get_column("basis_zscore")
        .to_list()
    )
    assert constant[0] is None
    assert constant[1] == pytest.approx(0.0)
    assert constant[2] == pytest.approx(0.0)

    zero_index = (
        BasisZScoreFactor(lookback=2)
        .compute(_basis_frame([101.0, 102.0, 103.0], [100.0, 0.0, 100.0]))
        .get_column("basis_zscore")
        .to_list()
    )
    assert zero_index[1] is None
    assert zero_index[2] is None

    factor = BasisZScoreFactor(lookback=2)
    _assert_missing_column(
        factor,
        missing="mark_price",
        error_code="FACTOR-BASIS-ZSCORE-002",
        frame=pl.DataFrame({"index_price": [1.0, 2.0]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="basis_zscore",
        frame=_basis_frame([101.0, 102.0, 103.0, 104.0], [100.0, 100.0, 100.0, 100.0]),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="basis_zscore",
        columns=("mark_price", "index_price"),
    )
    _assert_determinism(
        lambda: BasisZScoreFactor(lookback=2),
        output_column="basis_zscore",
        frame=_basis_frame([101.0, 102.0, 103.0, 104.0], [100.0, 100.0, 100.0, 100.0]),
    )


# ---------------------------------------------------------------------------
# CarryFactor
# ---------------------------------------------------------------------------


def test_carry_metadata_and_math() -> None:
    """Carry matches rolling-mean funding rate times point-in-time basis."""
    factor = CarryFactor()
    assert factor.name == "carry"
    assert factor.lookback == 20
    assert factor.required_features == ("funding_rate", "mark_price", "index_price")
    assert factor.produced_columns == ("carry",)
    assert CarryFactor is CarryFactorDirect

    rates = [0.0002, 0.0004, 0.0006, 0.0008]
    marks = [101.0, 102.0, 103.0, 104.0]
    indexes = [100.0, 100.0, 100.0, 100.0]
    values = (
        CarryFactor(lookback=2)
        .compute(_carry_frame(rates, marks, indexes))
        .get_column("carry")
        .to_list()
    )
    assert values[0] is None
    expected_1 = ((0.0002 + 0.0004) / 2.0) * ((102.0 - 100.0) / 100.0)
    expected_2 = ((0.0004 + 0.0006) / 2.0) * ((103.0 - 100.0) / 100.0)
    expected_3 = ((0.0006 + 0.0008) / 2.0) * ((104.0 - 100.0) / 100.0)
    assert values[1] == pytest.approx(expected_1)
    assert values[2] == pytest.approx(expected_2)
    assert values[3] == pytest.approx(expected_3)


def test_carry_validation_and_edges() -> None:
    """Carry covers zero basis, signs, zero index, and contracts."""
    with pytest.raises(ValidationError, match="lookback must be an integer greater than 0"):
        CarryFactor(lookback=0)

    zero_basis = (
        CarryFactor(lookback=2)
        .compute(_carry_frame([0.0002, 0.0004], [100.0, 100.0], [100.0, 100.0]))
        .get_column("carry")
        .to_list()
    )
    assert zero_basis[0] is None
    assert zero_basis[1] == pytest.approx(0.0)

    positive = (
        CarryFactor(lookback=2)
        .compute(_carry_frame([0.0002, 0.0004], [101.0, 102.0], [100.0, 100.0]))
        .get_column("carry")
        .to_list()
    )
    negative_funding = (
        CarryFactor(lookback=2)
        .compute(_carry_frame([-0.0002, -0.0004], [101.0, 102.0], [100.0, 100.0]))
        .get_column("carry")
        .to_list()
    )
    negative_basis = (
        CarryFactor(lookback=2)
        .compute(_carry_frame([0.0002, 0.0004], [99.0, 98.0], [100.0, 100.0]))
        .get_column("carry")
        .to_list()
    )
    assert positive[1] > 0.0
    assert negative_funding[1] < 0.0
    assert negative_basis[1] < 0.0

    zero_index = (
        CarryFactor(lookback=2)
        .compute(_carry_frame([0.0002, 0.0004], [101.0, 102.0], [100.0, 0.0]))
        .get_column("carry")
        .to_list()
    )
    assert zero_index[1] is None

    factor = CarryFactor(lookback=2)
    _assert_missing_column(
        factor,
        missing="funding_rate",
        error_code="FACTOR-CARRY-002",
        frame=pl.DataFrame({"mark_price": [1.0], "index_price": [1.0]}),
    )
    _assert_protocol_and_immutability(
        factor,
        output_column="carry",
        frame=_carry_frame(
            [0.0001, 0.0002, 0.0003, 0.0004],
            [101.0, 102.0, 103.0, 104.0],
            [100.0, 100.0, 100.0, 100.0],
        ),
    )
    _assert_empty_and_single_row(
        factor,
        output_column="carry",
        columns=("funding_rate", "mark_price", "index_price"),
    )
    _assert_determinism(
        lambda: CarryFactor(lookback=2),
        output_column="carry",
        frame=_carry_frame(
            [0.0001, 0.0002, 0.0003, 0.0004],
            [101.0, 102.0, 103.0, 104.0],
            [100.0, 100.0, 100.0, 100.0],
        ),
    )
