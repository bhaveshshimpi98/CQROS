"""Unit tests for CQROS ``CrowdingFactor``."""

from __future__ import annotations

import pytest

from cqros.factors.composite import CrowdingFactor
from tests.unit.factors.composite._helpers import (
    assert_missing_feature_raises,
    assert_null_propagation,
    assert_preserves_columns,
    assert_protocol_and_immutability,
    feature_frame,
)


def _factor() -> CrowdingFactor:
    """Build the default crowding factor."""
    return CrowdingFactor()


def test_crowding_metadata() -> None:
    """CrowdingFactor exposes the fixed production metadata contract."""
    factor = _factor()
    assert factor.name == "crowding"
    assert factor.version == "1.0.0"
    assert factor.category == "composite"
    assert factor.required_features == ("crowding_score", "funding_zscore")
    assert factor.produced_columns == ("crowding",)
    assert factor.lookback == 0
    assert factor.metadata.produced_columns == ("crowding",)


def test_crowding_calculation_correctness() -> None:
    """Crowding matches crowding_score * funding_zscore."""
    frame = feature_frame(
        {
            "crowding_score": [2.0, -1.5],
            "funding_zscore": [1.0, -0.5],
        }
    )
    values = _factor().compute(frame).get_column("crowding").to_list()
    assert values[0] == pytest.approx(2.0 * 1.0)
    assert values[1] == pytest.approx((-1.5) * (-0.5))


def test_null_propagation_missing_feature_and_immutability() -> None:
    """Nulls propagate, missing features fail, and inputs stay immutable."""
    assert_null_propagation(
        _factor(),
        output_column="crowding",
        frame=feature_frame(
            {
                "crowding_score": [None, 1.0],
                "funding_zscore": [0.5, 0.5],
            }
        ),
    )
    assert_missing_feature_raises(
        _factor,
        error_code="FACTOR-CROWDING-001",
        factor_name="crowding",
        missing_feature="funding_zscore",
        present_features={"crowding_score": [1.0]},
    )
    frame = feature_frame(
        {
            "crowding_score": [1.0],
            "funding_zscore": [0.5],
            "extra": [3.0],
        }
    )
    assert_protocol_and_immutability(_factor(), output_column="crowding", frame=frame)
    assert_preserves_columns(_factor(), output_column="crowding", frame=frame)
