"""Unit tests for CQROS flow confirmation and funding divergence factors."""

from __future__ import annotations

import pytest

from cqros.factors.composite import FlowConfirmationFactor, FundingDivergenceFactor
from tests.unit.factors.composite._helpers import (
    assert_missing_feature_raises,
    assert_null_propagation,
    assert_preserves_columns,
    assert_protocol_and_immutability,
    feature_frame,
)


def test_flow_confirmation_metadata() -> None:
    """FlowConfirmationFactor exposes the fixed metadata contract."""
    factor = FlowConfirmationFactor()
    assert factor.name == "flow_confirmation"
    assert factor.version == "1.0.0"
    assert factor.category == "composite"
    assert factor.required_features == ("returns", "flow_imbalance")
    assert factor.produced_columns == ("flow_confirmation",)
    assert factor.lookback == 0


def test_flow_confirmation_calculation_correctness() -> None:
    """Flow confirmation matches returns * flow_imbalance."""
    frame = feature_frame(
        {
            "returns": [0.02, -0.03],
            "flow_imbalance": [0.5, -0.4],
        }
    )
    values = FlowConfirmationFactor().compute(frame).get_column("flow_confirmation").to_list()
    assert values[0] == pytest.approx(0.02 * 0.5)
    assert values[1] == pytest.approx((-0.03) * (-0.4))


def test_flow_confirmation_missing_feature_and_immutability() -> None:
    """Flow confirmation fails fast and does not mutate inputs."""
    assert_missing_feature_raises(
        FlowConfirmationFactor,
        error_code="FACTOR-FLOW-CONFIRMATION-001",
        factor_name="flow_confirmation",
        missing_feature="flow_imbalance",
        present_features={"returns": [0.01]},
    )
    frame = feature_frame(
        {
            "returns": [0.01],
            "flow_imbalance": [0.2],
            "extra": [5.0],
        }
    )
    factor = FlowConfirmationFactor()
    assert_protocol_and_immutability(
        factor,
        output_column="flow_confirmation",
        frame=frame,
    )
    assert_preserves_columns(
        factor,
        output_column="flow_confirmation",
        frame=frame,
    )
    assert_null_propagation(
        factor,
        output_column="flow_confirmation",
        frame=feature_frame(
            {
                "returns": [None],
                "flow_imbalance": [0.2],
            }
        ),
    )


def test_funding_divergence_metadata() -> None:
    """FundingDivergenceFactor exposes the fixed metadata contract."""
    factor = FundingDivergenceFactor()
    assert factor.name == "funding_divergence"
    assert factor.version == "1.0.0"
    assert factor.category == "composite"
    assert factor.required_features == ("returns", "funding_zscore")
    assert factor.produced_columns == ("funding_divergence",)
    assert factor.lookback == 0


def test_funding_divergence_calculation_correctness() -> None:
    """Funding divergence matches returns * (-funding_zscore)."""
    frame = feature_frame(
        {
            "returns": [0.04, -0.02],
            "funding_zscore": [-1.0, 1.5],
        }
    )
    values = FundingDivergenceFactor().compute(frame).get_column("funding_divergence").to_list()
    assert values[0] == pytest.approx(0.04 * 1.0)
    assert values[1] == pytest.approx((-0.02) * (-1.5))


def test_funding_divergence_missing_feature_and_immutability() -> None:
    """Funding divergence fails fast and does not mutate inputs."""
    assert_missing_feature_raises(
        FundingDivergenceFactor,
        error_code="FACTOR-FUNDING-DIVERGENCE-001",
        factor_name="funding_divergence",
        missing_feature="funding_zscore",
        present_features={"returns": [0.01]},
    )
    frame = feature_frame(
        {
            "returns": [0.01],
            "funding_zscore": [-0.5],
        }
    )
    assert_protocol_and_immutability(
        FundingDivergenceFactor(),
        output_column="funding_divergence",
        frame=frame,
    )
