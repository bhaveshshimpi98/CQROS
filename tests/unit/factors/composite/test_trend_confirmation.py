"""Unit tests for CQROS ``TrendConfirmationFactor``."""

from __future__ import annotations

import pytest

from cqros.factors.base import BaseFactor
from cqros.factors.composite import TrendConfirmationFactor
from cqros.factors.composite.trend_confirmation import (
    TrendConfirmationFactor as TrendConfirmationFactorDirect,
)
from cqros.factors.interfaces import Factor
from tests.unit.factors.composite._helpers import (
    assert_missing_feature_raises,
    assert_null_propagation,
    assert_preserves_columns,
    assert_protocol_and_immutability,
    feature_frame,
)


def _factor() -> TrendConfirmationFactor:
    """Build the default trend confirmation factor."""
    return TrendConfirmationFactor()


def test_trend_confirmation_metadata() -> None:
    """TrendConfirmationFactor exposes the fixed production metadata contract."""
    factor = _factor()
    assert isinstance(factor, BaseFactor)
    assert isinstance(factor, Factor)
    assert factor.name == "trend_confirmation"
    assert factor.version == "1.0.0"
    assert factor.category == "composite"
    assert factor.required_features == ("returns", "flow_imbalance", "oi_momentum")
    assert factor.produced_columns == ("trend_confirmation",)
    assert factor.lookback == 0
    meta = factor.metadata
    assert meta.name == "trend_confirmation"
    assert meta.category == "composite"
    assert meta.required_features == factor.required_features
    assert meta.produced_columns == ("trend_confirmation",)


def test_trend_confirmation_calculation_correctness() -> None:
    """Trend confirmation matches returns * flow_imbalance * oi_momentum."""
    frame = feature_frame(
        {
            "returns": [0.02, -0.01, 0.03],
            "flow_imbalance": [0.5, -0.4, 0.2],
            "oi_momentum": [0.1, 0.2, -0.5],
        }
    )
    values = _factor().compute(frame).get_column("trend_confirmation").to_list()
    assert values[0] == pytest.approx(0.02 * 0.5 * 0.1)
    assert values[1] == pytest.approx((-0.01) * (-0.4) * 0.2)
    assert values[2] == pytest.approx(0.03 * 0.2 * (-0.5))


def test_null_propagation() -> None:
    """Null feature inputs propagate into the composite output."""
    frame = feature_frame(
        {
            "returns": [None, 0.02],
            "flow_imbalance": [0.5, 0.5],
            "oi_momentum": [0.1, 0.1],
        }
    )
    assert_null_propagation(_factor(), output_column="trend_confirmation", frame=frame)
    values = _factor().compute(frame).get_column("trend_confirmation").to_list()
    assert values[0] is None
    assert values[1] == pytest.approx(0.02 * 0.5 * 0.1)


def test_missing_feature_raises() -> None:
    """Missing required feature raises FactorError."""
    assert_missing_feature_raises(
        _factor,
        error_code="FACTOR-TREND-CONFIRMATION-001",
        factor_name="trend_confirmation",
        missing_feature="oi_momentum",
        present_features={"returns": [0.01], "flow_imbalance": [0.2]},
    )


def test_input_immutability() -> None:
    """compute does not mutate the caller-supplied DataFrame."""
    frame = feature_frame(
        {
            "returns": [0.01, 0.02],
            "flow_imbalance": [0.1, 0.2],
            "oi_momentum": [0.3, 0.4],
        }
    )
    assert_protocol_and_immutability(
        _factor(),
        output_column="trend_confirmation",
        frame=frame,
    )


def test_preserves_existing_columns() -> None:
    """Existing columns are preserved alongside the composite output."""
    frame = feature_frame(
        {
            "returns": [0.01],
            "flow_imbalance": [0.2],
            "oi_momentum": [0.3],
            "symbol": [1.0],
        }
    )
    assert_preserves_columns(
        _factor(),
        output_column="trend_confirmation",
        frame=frame,
    )


def test_package_exports_trend_confirmation_factor() -> None:
    """TrendConfirmationFactor is exported from the composite package."""
    assert TrendConfirmationFactor is TrendConfirmationFactorDirect
