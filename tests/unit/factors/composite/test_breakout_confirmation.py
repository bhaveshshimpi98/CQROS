"""Unit tests for CQROS ``BreakoutConfirmationFactor``."""

from __future__ import annotations

import pytest

from cqros.factors.base import BaseFactor
from cqros.factors.composite import BreakoutConfirmationFactor
from cqros.factors.interfaces import Factor
from tests.unit.factors.composite._helpers import (
    assert_missing_feature_raises,
    assert_null_propagation,
    assert_preserves_columns,
    assert_protocol_and_immutability,
    feature_frame,
)


def _factor() -> BreakoutConfirmationFactor:
    """Build the default breakout confirmation factor."""
    return BreakoutConfirmationFactor()


def test_breakout_confirmation_metadata() -> None:
    """BreakoutConfirmationFactor exposes the fixed production metadata contract."""
    factor = _factor()
    assert isinstance(factor, BaseFactor)
    assert isinstance(factor, Factor)
    assert factor.name == "breakout_confirmation"
    assert factor.version == "1.0.0"
    assert factor.category == "composite"
    assert factor.required_features == ("returns", "oi_momentum", "buy_pressure")
    assert factor.produced_columns == ("breakout_confirmation",)
    assert factor.lookback == 0


def test_breakout_confirmation_calculation_correctness() -> None:
    """Breakout confirmation matches returns * oi_momentum * buy_pressure."""
    frame = feature_frame(
        {
            "returns": [0.05, -0.02],
            "oi_momentum": [0.2, 0.1],
            "buy_pressure": [0.7, 0.4],
        }
    )
    values = _factor().compute(frame).get_column("breakout_confirmation").to_list()
    assert values[0] == pytest.approx(0.05 * 0.2 * 0.7)
    assert values[1] == pytest.approx((-0.02) * 0.1 * 0.4)


def test_null_propagation() -> None:
    """Null feature inputs propagate into the composite output."""
    frame = feature_frame(
        {
            "returns": [0.05, None],
            "oi_momentum": [0.2, 0.1],
            "buy_pressure": [0.7, 0.4],
        }
    )
    assert_null_propagation(
        _factor(),
        output_column="breakout_confirmation",
        frame=frame,
    )


def test_missing_feature_raises() -> None:
    """Missing required feature raises FactorError."""
    assert_missing_feature_raises(
        _factor,
        error_code="FACTOR-BREAKOUT-CONFIRMATION-001",
        factor_name="breakout_confirmation",
        missing_feature="buy_pressure",
        present_features={"returns": [0.01], "oi_momentum": [0.2]},
    )


def test_input_immutability_and_column_preservation() -> None:
    """compute is immutable and preserves existing columns."""
    frame = feature_frame(
        {
            "returns": [0.01],
            "oi_momentum": [0.2],
            "buy_pressure": [0.6],
            "extra": [9.0],
        }
    )
    assert_protocol_and_immutability(
        _factor(),
        output_column="breakout_confirmation",
        frame=frame,
    )
    assert_preserves_columns(
        _factor(),
        output_column="breakout_confirmation",
        frame=frame,
    )
