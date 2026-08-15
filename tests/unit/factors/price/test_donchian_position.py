"""Unit tests for CQROS ``DonchianPositionFactor``."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.core.exceptions import ValidationError
from cqros.factors.exceptions import FactorError
from cqros.factors.price import DonchianPositionFactor
from cqros.factors.price.donchian_position import (
    DonchianPositionFactor as DonchianPositionFactorDirect,
)
from tests.unit.factors.price._helpers import (
    assert_determinism,
    assert_empty_and_single_row,
    assert_lookback_below_two_raises,
    assert_lookback_negative_raises,
    assert_output_float64_nullable,
    assert_preserves_columns,
    assert_protocol_and_immutability,
)


def _factor(*, lookback: int = 20) -> DonchianPositionFactor:
    """Build a Donchian position factor with an optional lookback override."""
    return DonchianPositionFactor(lookback=lookback)


def _ohlc_frame() -> pl.DataFrame:
    """Return a small OHLC fixture for Donchian tests."""
    return pl.DataFrame(
        {
            "high": [11.0, 13.0, 12.0, 15.0, 14.0],
            "low": [9.0, 10.0, 8.0, 11.0, 10.0],
            "close": [10.0, 12.0, 9.0, 14.0, 13.0],
            "volume": [1, 2, 3, 4, 5],
        }
    )


def test_donchian_position_metadata() -> None:
    """DonchianPositionFactor exposes the fixed production metadata contract."""
    factor = _factor()
    assert factor.name == "donchian_position"
    assert factor.version == "1.0.0"
    assert factor.category == "price"
    assert factor.required_features == ("high", "low", "close")
    assert factor.produced_columns == ("donchian_position",)
    assert factor.lookback == 20
    assert factor.metadata.required_features == ("high", "low", "close")


def test_donchian_position_calculation_correctness() -> None:
    """Donchian position matches (close - low_min) / (high_max - low_min)."""
    frame = _ohlc_frame()
    values = _factor(lookback=3).compute(frame).get_column("donchian_position").to_list()
    assert values[0] is None
    assert values[1] is None
    # window highs [11,13,12] -> 13; lows [9,10,8] -> 8; close 9
    assert values[2] == pytest.approx((9.0 - 8.0) / (13.0 - 8.0))
    # highs [13,12,15] -> 15; lows [10,8,11] -> 8; close 14
    assert values[3] == pytest.approx((14.0 - 8.0) / (15.0 - 8.0))
    # highs [12,15,14] -> 15; lows [8,11,10] -> 8; close 13
    assert values[4] == pytest.approx((13.0 - 8.0) / (15.0 - 8.0))


def test_zero_channel_width_is_null() -> None:
    """Zero Donchian width returns null."""
    frame = pl.DataFrame(
        {
            "high": [10.0, 10.0, 10.0],
            "low": [10.0, 10.0, 10.0],
            "close": [10.0, 10.0, 10.0],
        }
    )
    values = _factor(lookback=3).compute(frame).get_column("donchian_position").to_list()
    assert values[2] is None


def test_position_at_channel_extremes() -> None:
    """Close at low is 0 and close at high is 1."""
    frame = pl.DataFrame(
        {
            "high": [12.0, 14.0, 16.0],
            "low": [10.0, 11.0, 12.0],
            "close": [10.0, 12.0, 16.0],
        }
    )
    values = _factor(lookback=3).compute(frame).get_column("donchian_position").to_list()
    assert values[2] == pytest.approx(1.0)

    frame_low = pl.DataFrame(
        {
            "high": [12.0, 14.0, 16.0],
            "low": [10.0, 11.0, 9.0],
            "close": [11.0, 12.0, 9.0],
        }
    )
    values_low = _factor(lookback=3).compute(frame_low).get_column("donchian_position").to_list()
    assert values_low[2] == pytest.approx(0.0)


def test_missing_required_columns_raise() -> None:
    """Missing high, low, or close raises FactorError."""
    with pytest.raises(FactorError, match="required column missing: high") as exc_info:
        _factor(lookback=2).compute(pl.DataFrame({"low": [1.0], "close": [1.0]}))
    assert exc_info.value.error_code == "FACTOR-DONCHIAN-POSITION-002"

    with pytest.raises(FactorError, match="required column missing: low") as exc_info:
        _factor(lookback=2).compute(pl.DataFrame({"high": [1.0], "close": [1.0]}))
    assert exc_info.value.error_code == "FACTOR-DONCHIAN-POSITION-002"

    with pytest.raises(FactorError, match="required column missing: close") as exc_info:
        _factor(lookback=2).compute(pl.DataFrame({"high": [1.0], "low": [1.0]}))
    assert exc_info.value.error_code == "FACTOR-DONCHIAN-POSITION-002"
    assert exc_info.value.details["factor"] == "donchian_position"


def test_validation_immutability_schema_and_exports() -> None:
    """Validation, schema, determinism, and package export contracts."""
    assert_lookback_below_two_raises(
        lambda value: _factor(lookback=value),
        error_code="FACTOR-DONCHIAN-POSITION-001",
    )
    assert_lookback_negative_raises(lambda value: _factor(lookback=value))
    with pytest.raises(ValidationError):
        DonchianPositionFactor(lookback=0)

    factor = _factor(lookback=2)
    frame = _ohlc_frame()
    assert_protocol_and_immutability(factor, output_column="donchian_position", frame=frame)
    assert_preserves_columns(
        factor,
        output_column="donchian_position",
        frame=frame.select(["high", "low", "close", "volume"]),
    )
    assert_output_float64_nullable(factor, output_column="donchian_position")
    assert_empty_and_single_row(
        lambda: _factor(lookback=2),
        output_column="donchian_position",
        columns={"high": [], "low": [], "close": []},
    )
    assert_determinism(
        lambda: _factor(lookback=3),
        output_column="donchian_position",
        frame=frame,
    )
    assert DonchianPositionFactor is DonchianPositionFactorDirect
