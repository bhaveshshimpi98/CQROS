"""Unit tests for CQROS leveraged long and short build-up factors."""

from __future__ import annotations

import pytest

from cqros.factors.composite import (
    LeveragedLongBuildUpFactor,
    LeveragedShortBuildUpFactor,
)
from tests.unit.factors.composite._helpers import (
    assert_missing_feature_raises,
    assert_null_propagation,
    assert_preserves_columns,
    assert_protocol_and_immutability,
    feature_frame,
)


def test_leveraged_long_build_up_metadata() -> None:
    """LeveragedLongBuildUpFactor exposes the fixed metadata contract."""
    factor = LeveragedLongBuildUpFactor()
    assert factor.name == "leveraged_long_build_up"
    assert factor.version == "1.0.0"
    assert factor.category == "composite"
    assert factor.required_features == (
        "funding_zscore",
        "oi_momentum",
        "buy_pressure",
    )
    assert factor.produced_columns == ("leveraged_long_build_up",)
    assert factor.lookback == 0


def test_leveraged_long_build_up_calculation_correctness() -> None:
    """Leveraged long build-up matches funding_zscore * oi_momentum * buy_pressure."""
    frame = feature_frame(
        {
            "funding_zscore": [1.5, -0.5],
            "oi_momentum": [0.1, 0.2],
            "buy_pressure": [0.6, 0.4],
        }
    )
    values = (
        LeveragedLongBuildUpFactor().compute(frame).get_column("leveraged_long_build_up").to_list()
    )
    assert values[0] == pytest.approx(1.5 * 0.1 * 0.6)
    assert values[1] == pytest.approx((-0.5) * 0.2 * 0.4)


def test_leveraged_long_missing_feature_and_immutability() -> None:
    """Leveraged long build-up fails fast and does not mutate inputs."""
    assert_missing_feature_raises(
        LeveragedLongBuildUpFactor,
        error_code="FACTOR-LEVERAGED-LONG-BUILD-UP-001",
        factor_name="leveraged_long_build_up",
        missing_feature="buy_pressure",
        present_features={"funding_zscore": [1.0], "oi_momentum": [0.1]},
    )
    frame = feature_frame(
        {
            "funding_zscore": [1.0],
            "oi_momentum": [0.1],
            "buy_pressure": [0.5],
            "extra": [2.0],
        }
    )
    factor = LeveragedLongBuildUpFactor()
    assert_protocol_and_immutability(
        factor,
        output_column="leveraged_long_build_up",
        frame=frame,
    )
    assert_preserves_columns(
        factor,
        output_column="leveraged_long_build_up",
        frame=frame,
    )
    assert_null_propagation(
        factor,
        output_column="leveraged_long_build_up",
        frame=feature_frame(
            {
                "funding_zscore": [None],
                "oi_momentum": [0.1],
                "buy_pressure": [0.5],
            }
        ),
    )


def test_leveraged_short_build_up_metadata() -> None:
    """LeveragedShortBuildUpFactor exposes the fixed metadata contract."""
    factor = LeveragedShortBuildUpFactor()
    assert factor.name == "leveraged_short_build_up"
    assert factor.version == "1.0.0"
    assert factor.category == "composite"
    assert factor.required_features == (
        "funding_zscore",
        "oi_momentum",
        "sell_pressure",
    )
    assert factor.produced_columns == ("leveraged_short_build_up",)
    assert factor.lookback == 0


def test_leveraged_short_build_up_calculation_correctness() -> None:
    """Leveraged short build-up matches (-funding_zscore) * oi_momentum * sell_pressure."""
    frame = feature_frame(
        {
            "funding_zscore": [-1.2, 0.8],
            "oi_momentum": [0.25, 0.1],
            "sell_pressure": [0.7, 0.3],
        }
    )
    values = (
        LeveragedShortBuildUpFactor()
        .compute(frame)
        .get_column("leveraged_short_build_up")
        .to_list()
    )
    assert values[0] == pytest.approx(1.2 * 0.25 * 0.7)
    assert values[1] == pytest.approx((-0.8) * 0.1 * 0.3)


def test_leveraged_short_missing_feature_and_immutability() -> None:
    """Leveraged short build-up fails fast and does not mutate inputs."""
    assert_missing_feature_raises(
        LeveragedShortBuildUpFactor,
        error_code="FACTOR-LEVERAGED-SHORT-BUILD-UP-001",
        factor_name="leveraged_short_build_up",
        missing_feature="sell_pressure",
        present_features={"funding_zscore": [-1.0], "oi_momentum": [0.1]},
    )
    frame = feature_frame(
        {
            "funding_zscore": [-1.0],
            "oi_momentum": [0.1],
            "sell_pressure": [0.6],
        }
    )
    assert_protocol_and_immutability(
        LeveragedShortBuildUpFactor(),
        output_column="leveraged_short_build_up",
        frame=frame,
    )
