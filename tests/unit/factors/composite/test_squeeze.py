"""Unit tests for CQROS short and long squeeze composite factors."""

from __future__ import annotations

import pytest

from cqros.factors.composite import LongSqueezeFactor, ShortSqueezeFactor
from tests.unit.factors.composite._helpers import (
    assert_missing_feature_raises,
    assert_null_propagation,
    assert_preserves_columns,
    assert_protocol_and_immutability,
    feature_frame,
)


def test_short_squeeze_metadata() -> None:
    """ShortSqueezeFactor exposes the fixed metadata contract."""
    factor = ShortSqueezeFactor()
    assert factor.name == "short_squeeze"
    assert factor.version == "1.0.0"
    assert factor.category == "composite"
    assert factor.required_features == (
        "crowding_score",
        "returns",
        "flow_imbalance",
    )
    assert factor.produced_columns == ("short_squeeze",)
    assert factor.lookback == 0


def test_short_squeeze_calculation_correctness() -> None:
    """Short squeeze matches (-crowding_score) * returns * flow_imbalance."""
    frame = feature_frame(
        {
            "crowding_score": [-2.0, 1.0],
            "returns": [0.03, 0.02],
            "flow_imbalance": [0.4, -0.1],
        }
    )
    values = ShortSqueezeFactor().compute(frame).get_column("short_squeeze").to_list()
    assert values[0] == pytest.approx(2.0 * 0.03 * 0.4)
    assert values[1] == pytest.approx((-1.0) * 0.02 * (-0.1))


def test_short_squeeze_missing_feature_and_immutability() -> None:
    """Short squeeze fails fast and does not mutate inputs."""
    assert_missing_feature_raises(
        ShortSqueezeFactor,
        error_code="FACTOR-SHORT-SQUEEZE-001",
        factor_name="short_squeeze",
        missing_feature="flow_imbalance",
        present_features={"crowding_score": [-1.0], "returns": [0.01]},
    )
    frame = feature_frame(
        {
            "crowding_score": [-1.0],
            "returns": [0.01],
            "flow_imbalance": [0.2],
        }
    )
    assert_protocol_and_immutability(
        ShortSqueezeFactor(),
        output_column="short_squeeze",
        frame=frame,
    )
    assert_null_propagation(
        ShortSqueezeFactor(),
        output_column="short_squeeze",
        frame=feature_frame(
            {
                "crowding_score": [None],
                "returns": [0.01],
                "flow_imbalance": [0.2],
            }
        ),
    )


def test_long_squeeze_metadata() -> None:
    """LongSqueezeFactor exposes the fixed metadata contract."""
    factor = LongSqueezeFactor()
    assert factor.name == "long_squeeze"
    assert factor.version == "1.0.0"
    assert factor.category == "composite"
    assert factor.required_features == (
        "crowding_score",
        "returns",
        "sell_pressure",
    )
    assert factor.produced_columns == ("long_squeeze",)
    assert factor.lookback == 0


def test_long_squeeze_calculation_correctness() -> None:
    """Long squeeze matches crowding_score * (-returns) * sell_pressure."""
    frame = feature_frame(
        {
            "crowding_score": [2.0, -0.5],
            "returns": [-0.04, 0.01],
            "sell_pressure": [0.8, 0.3],
        }
    )
    values = LongSqueezeFactor().compute(frame).get_column("long_squeeze").to_list()
    assert values[0] == pytest.approx(2.0 * 0.04 * 0.8)
    assert values[1] == pytest.approx((-0.5) * (-0.01) * 0.3)


def test_long_squeeze_missing_feature_and_column_preservation() -> None:
    """Long squeeze fails fast and preserves existing columns."""
    assert_missing_feature_raises(
        LongSqueezeFactor,
        error_code="FACTOR-LONG-SQUEEZE-001",
        factor_name="long_squeeze",
        missing_feature="sell_pressure",
        present_features={"crowding_score": [1.0], "returns": [-0.01]},
    )
    frame = feature_frame(
        {
            "crowding_score": [1.0],
            "returns": [-0.01],
            "sell_pressure": [0.5],
            "extra": [4.0],
        }
    )
    assert_preserves_columns(
        LongSqueezeFactor(),
        output_column="long_squeeze",
        frame=frame,
    )
