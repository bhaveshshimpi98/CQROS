"""Unit tests for CQROS ``PositionBuildUpFactor``."""

from __future__ import annotations

import pytest

from cqros.factors.base import BaseFactor
from cqros.factors.composite import PositionBuildUpFactor
from cqros.factors.interfaces import Factor
from tests.unit.factors.composite._helpers import (
    assert_missing_feature_raises,
    assert_null_propagation,
    assert_preserves_columns,
    assert_protocol_and_immutability,
    feature_frame,
)


def _factor() -> PositionBuildUpFactor:
    """Build the default position build-up factor."""
    return PositionBuildUpFactor()


def test_position_build_up_metadata() -> None:
    """PositionBuildUpFactor exposes the fixed production metadata contract."""
    factor = _factor()
    assert isinstance(factor, BaseFactor)
    assert isinstance(factor, Factor)
    assert factor.name == "position_build_up"
    assert factor.version == "1.0.0"
    assert factor.category == "composite"
    assert factor.required_features == ("oi_momentum", "ratio_momentum")
    assert factor.produced_columns == ("position_build_up",)
    assert factor.lookback == 0


def test_position_build_up_calculation_correctness() -> None:
    """Position build-up matches oi_momentum * ratio_momentum."""
    frame = feature_frame(
        {
            "oi_momentum": [0.15, -0.05],
            "ratio_momentum": [0.2, 0.3],
        }
    )
    values = _factor().compute(frame).get_column("position_build_up").to_list()
    assert values[0] == pytest.approx(0.15 * 0.2)
    assert values[1] == pytest.approx((-0.05) * 0.3)


def test_null_propagation() -> None:
    """Null feature inputs propagate into the composite output."""
    frame = feature_frame(
        {
            "oi_momentum": [None, 0.1],
            "ratio_momentum": [0.2, 0.2],
        }
    )
    assert_null_propagation(_factor(), output_column="position_build_up", frame=frame)


def test_missing_feature_raises() -> None:
    """Missing required feature raises FactorError."""
    assert_missing_feature_raises(
        _factor,
        error_code="FACTOR-POSITION-BUILD-UP-001",
        factor_name="position_build_up",
        missing_feature="ratio_momentum",
        present_features={"oi_momentum": [0.1]},
    )


def test_input_immutability_and_column_preservation() -> None:
    """compute is immutable and preserves existing columns."""
    frame = feature_frame(
        {
            "oi_momentum": [0.1],
            "ratio_momentum": [0.2],
            "extra": [1.0],
        }
    )
    assert_protocol_and_immutability(
        _factor(),
        output_column="position_build_up",
        frame=frame,
    )
    assert_preserves_columns(
        _factor(),
        output_column="position_build_up",
        frame=frame,
    )
