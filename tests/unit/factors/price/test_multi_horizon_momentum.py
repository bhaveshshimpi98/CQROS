"""Unit tests for CQROS ``MultiHorizonMomentumFactor``."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.core.exceptions import ValidationError
from cqros.factors.price import MultiHorizonMomentumFactor
from cqros.factors.price.multi_horizon_momentum import (
    MultiHorizonMomentumFactor as MultiHorizonMomentumFactorDirect,
)
from tests.unit.factors.price._helpers import (
    assert_missing_close_raises,
    assert_preserves_columns,
    assert_protocol_and_immutability,
)


def _factor(
    *,
    horizons: tuple[int, ...] = (5, 10, 20, 50),
) -> MultiHorizonMomentumFactor:
    """Build a multi-horizon momentum factor with optional horizons."""
    return MultiHorizonMomentumFactor(horizons=horizons)


def test_multi_horizon_momentum_metadata() -> None:
    """MultiHorizonMomentumFactor exposes the fixed production metadata contract."""
    factor = _factor()
    assert factor.name == "multi_horizon_momentum"
    assert factor.version == "1.0.0"
    assert factor.category == "price"
    assert factor.required_features == ("close",)
    assert factor.produced_columns == ("multi_horizon_momentum",)
    assert factor.horizons == (5, 10, 20, 50)
    assert factor.lookback == 50
    assert factor.metadata.lookback == 50


def test_lookback_derived_from_horizons() -> None:
    """lookback is always max(horizons), ignoring constructor lookback."""
    factor = MultiHorizonMomentumFactor(horizons=(2, 7, 3), lookback=99)
    assert factor.horizons == (2, 7, 3)
    assert factor.lookback == 7


def test_multi_horizon_momentum_calculation_correctness() -> None:
    """Output equals the equal-weighted mean of per-horizon momentum."""
    frame = pl.DataFrame({"close": [100.0, 110.0, 121.0, 133.1, 146.41]})
    result = _factor(horizons=(1, 2)).compute(frame)
    values = result.get_column("multi_horizon_momentum").to_list()
    assert values[0] is None
    assert values[1] is None
    mom1 = (121.0 / 110.0) - 1.0
    mom2 = (121.0 / 100.0) - 1.0
    assert values[2] == pytest.approx((mom1 + mom2) / 2.0)
    mom1 = (146.41 / 133.1) - 1.0
    mom2 = (146.41 / 121.0) - 1.0
    assert values[4] == pytest.approx((mom1 + mom2) / 2.0)


def test_insufficient_history_is_null() -> None:
    """Rows before the longest horizon remain null."""
    frame = pl.DataFrame({"close": [1.0, 2.0, 3.0]})
    values = _factor(horizons=(2, 5)).compute(frame).get_column("multi_horizon_momentum").to_list()
    assert values == [None, None, None]


def test_empty_horizons_raises() -> None:
    """Empty horizons are rejected."""
    with pytest.raises(ValidationError, match="horizons must contain at least one entry") as (
        exc_info
    ):
        MultiHorizonMomentumFactor(horizons=())
    assert exc_info.value.error_code == "FACTOR-MULTI-HORIZON-MOMENTUM-001"


def test_non_positive_horizon_raises() -> None:
    """Non-positive horizon entries are rejected."""
    with pytest.raises(
        ValidationError,
        match="horizons entries must be integers greater than 0",
    ) as exc_info:
        MultiHorizonMomentumFactor(horizons=(5, 0, 10))
    assert exc_info.value.error_code == "FACTOR-MULTI-HORIZON-MOMENTUM-002"
    assert exc_info.value.details["index"] == 1


def test_invalid_horizons_type_raises() -> None:
    """Non-sequence horizons are rejected."""
    with pytest.raises(
        ValidationError,
        match="horizons must be a sequence of positive integers",
    ) as exc_info:
        MultiHorizonMomentumFactor(horizons=5)  # type: ignore[arg-type]
    assert exc_info.value.error_code == "FACTOR-MULTI-HORIZON-MOMENTUM-003"


def test_missing_close_immutability_and_exports() -> None:
    """Missing close, immutability, protocol, and package exports."""
    assert_missing_close_raises(
        lambda: _factor(horizons=(1, 2)),
        error_code="FACTOR-MULTI-HORIZON-MOMENTUM-004",
        factor_name="multi_horizon_momentum",
    )
    factor = _factor(horizons=(1, 2))
    assert_protocol_and_immutability(factor, output_column="multi_horizon_momentum")
    assert_preserves_columns(factor, output_column="multi_horizon_momentum")
    assert MultiHorizonMomentumFactor is MultiHorizonMomentumFactorDirect
    import cqros.factors as factors_package
    import cqros.factors.price as price_package

    assert "MultiHorizonMomentumFactor" in price_package.__all__
    assert "MultiHorizonMomentumFactor" in factors_package.__all__
