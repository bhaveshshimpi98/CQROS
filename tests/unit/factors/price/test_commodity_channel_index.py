"""Unit tests for CQROS ``CommodityChannelIndexFactor``."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.core.exceptions import ValidationError
from cqros.factors.exceptions import FactorError
from cqros.factors.price import CommodityChannelIndexFactor
from cqros.factors.price.commodity_channel_index import (
    CommodityChannelIndexFactor as CommodityChannelIndexFactorDirect,
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


def _factor(*, lookback: int = 20) -> CommodityChannelIndexFactor:
    """Build a CCI factor with an optional lookback override."""
    return CommodityChannelIndexFactor(lookback=lookback)


def _ohlc_frame() -> pl.DataFrame:
    """Return a small OHLC fixture for CCI tests."""
    return pl.DataFrame(
        {
            "high": [11.0, 13.0, 12.0, 15.0, 14.0],
            "low": [9.0, 10.0, 8.0, 11.0, 10.0],
            "close": [10.0, 12.0, 9.0, 14.0, 13.0],
            "volume": [1, 2, 3, 4, 5],
        }
    )


def _cci(highs: list[float], lows: list[float], closes: list[float]) -> float | None:
    """Return CCI for a fully observed typical-price window."""
    typical = [
        (high + low + close) / 3.0 for high, low, close in zip(highs, lows, closes, strict=True)
    ]
    mean = sum(typical) / len(typical)
    mad = sum(abs(value - mean) for value in typical) / len(typical)
    if mad == 0.0:
        return None
    return (typical[-1] - mean) / (0.015 * mad)


def test_commodity_channel_index_metadata() -> None:
    """CommodityChannelIndexFactor exposes the fixed production metadata contract."""
    factor = _factor()
    assert factor.name == "commodity_channel_index"
    assert factor.version == "1.0.0"
    assert factor.category == "price"
    assert factor.required_features == ("high", "low", "close")
    assert factor.produced_columns == ("commodity_channel_index",)
    assert factor.lookback == 20
    assert factor.metadata.name == "commodity_channel_index"


def test_commodity_channel_index_calculation_correctness() -> None:
    """CCI matches typical-price mean absolute deviation formulation."""
    frame = _ohlc_frame()
    values = _factor(lookback=3).compute(frame).get_column("commodity_channel_index").to_list()
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx(
        _cci(
            frame["high"].to_list()[0:3],
            frame["low"].to_list()[0:3],
            frame["close"].to_list()[0:3],
        )
    )
    assert values[3] == pytest.approx(
        _cci(
            frame["high"].to_list()[1:4],
            frame["low"].to_list()[1:4],
            frame["close"].to_list()[1:4],
        )
    )
    assert values[4] == pytest.approx(
        _cci(
            frame["high"].to_list()[2:5],
            frame["low"].to_list()[2:5],
            frame["close"].to_list()[2:5],
        )
    )


def test_constant_prices_are_null() -> None:
    """Constant prices make MAD zero and CCI null."""
    frame = pl.DataFrame(
        {
            "high": [10.0, 10.0, 10.0, 10.0],
            "low": [10.0, 10.0, 10.0, 10.0],
            "close": [10.0, 10.0, 10.0, 10.0],
        }
    )
    values = _factor(lookback=3).compute(frame).get_column("commodity_channel_index").to_list()
    assert values[3] is None


def test_increasing_and_decreasing_prices() -> None:
    """Rising and falling markets produce finite CCI values."""
    rising = pl.DataFrame(
        {
            "high": [2.0, 3.0, 4.0, 5.0],
            "low": [1.0, 2.0, 3.0, 4.0],
            "close": [1.5, 2.5, 3.5, 4.5],
        }
    )
    falling = pl.DataFrame(
        {
            "high": [5.0, 4.0, 3.0, 2.0],
            "low": [4.0, 3.0, 2.0, 1.0],
            "close": [4.5, 3.5, 2.5, 1.5],
        }
    )
    assert (
        _factor(lookback=3).compute(rising).get_column("commodity_channel_index").to_list()[3]
        is not None
    )
    assert (
        _factor(lookback=3).compute(falling).get_column("commodity_channel_index").to_list()[3]
        is not None
    )


def test_missing_required_columns_raise() -> None:
    """Missing high, low, or close raises FactorError."""
    with pytest.raises(FactorError, match="required column missing: high") as exc_info:
        _factor(lookback=2).compute(pl.DataFrame({"low": [1.0], "close": [1.0]}))
    assert exc_info.value.error_code == "FACTOR-COMMODITY-CHANNEL-INDEX-002"

    with pytest.raises(FactorError, match="required column missing: close") as exc_info:
        _factor(lookback=2).compute(pl.DataFrame({"high": [1.0], "low": [1.0]}))
    assert exc_info.value.error_code == "FACTOR-COMMODITY-CHANNEL-INDEX-002"


def test_validation_immutability_schema_and_exports() -> None:
    """Validation, schema, determinism, and package export contracts."""
    assert_lookback_below_two_raises(
        lambda value: _factor(lookback=value),
        error_code="FACTOR-COMMODITY-CHANNEL-INDEX-001",
    )
    assert_lookback_negative_raises(lambda value: _factor(lookback=value))
    with pytest.raises(ValidationError):
        CommodityChannelIndexFactor(lookback=0)

    factor = _factor(lookback=2)
    frame = _ohlc_frame()
    assert_protocol_and_immutability(
        factor,
        output_column="commodity_channel_index",
        frame=frame,
    )
    assert_preserves_columns(
        factor,
        output_column="commodity_channel_index",
        frame=frame.select(["high", "low", "close", "volume"]),
    )
    assert_output_float64_nullable(factor, output_column="commodity_channel_index")
    assert_empty_and_single_row(
        lambda: _factor(lookback=2),
        output_column="commodity_channel_index",
        columns={"high": [], "low": [], "close": []},
    )
    assert_determinism(
        lambda: _factor(lookback=3),
        output_column="commodity_channel_index",
        frame=frame,
    )
    large = _factor(lookback=50).compute(
        pl.DataFrame(
            {
                "high": [float(i + 1) for i in range(10)],
                "low": [float(i) for i in range(10)],
                "close": [float(i) + 0.5 for i in range(10)],
            }
        )
    )
    assert all(value is None for value in large.get_column("commodity_channel_index").to_list())
    assert CommodityChannelIndexFactor is CommodityChannelIndexFactorDirect
