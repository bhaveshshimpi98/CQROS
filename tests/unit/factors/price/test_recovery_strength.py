"""Unit tests for CQROS ``RecoveryStrengthFactor``."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.factors.price import RecoveryStrengthFactor
from cqros.factors.price.recovery_strength import (
    RecoveryStrengthFactor as RecoveryStrengthFactorDirect,
)
from tests.unit.factors.price._helpers import (
    assert_lookback_negative_raises,
    assert_lookback_zero_raises,
    assert_missing_close_raises,
    assert_preserves_columns,
    assert_protocol_and_immutability,
)


def _factor(*, lookback: int = 20) -> RecoveryStrengthFactor:
    """Build a recovery strength factor with an optional lookback override."""
    return RecoveryStrengthFactor(lookback=lookback)


def test_recovery_strength_metadata() -> None:
    """RecoveryStrengthFactor exposes the fixed production metadata contract."""
    factor = _factor()
    assert factor.name == "recovery_strength"
    assert factor.version == "1.0.0"
    assert factor.category == "price"
    assert factor.required_features == ("close",)
    assert factor.produced_columns == ("recovery_strength",)
    assert factor.lookback == 20
    assert factor.metadata.lookback == 20


def test_recovery_strength_calculation_correctness() -> None:
    """Recovery strength matches position within the rolling high-low range."""
    frame = pl.DataFrame({"close": [10.0, 14.0, 12.0]})
    result = _factor(lookback=3).compute(frame)
    values = result.get_column("recovery_strength").to_list()
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx((12.0 - 10.0) / (14.0 - 10.0))


def test_zero_range_is_null() -> None:
    """Flat windows produce null recovery strength."""
    frame = pl.DataFrame({"close": [5.0, 5.0, 5.0]})
    values = _factor(lookback=3).compute(frame).get_column("recovery_strength").to_list()
    assert values[2] is None


def test_lookback_validation_missing_close_and_immutability() -> None:
    """Validation, missing close, immutability, and exports."""
    assert_lookback_zero_raises(
        lambda value: _factor(lookback=value),
        error_code="FACTOR-RECOVERY-STRENGTH-001",
    )
    assert_lookback_negative_raises(lambda value: _factor(lookback=value))
    assert_missing_close_raises(
        lambda: _factor(lookback=1),
        error_code="FACTOR-RECOVERY-STRENGTH-002",
        factor_name="recovery_strength",
    )
    factor = _factor(lookback=1)
    assert_protocol_and_immutability(factor, output_column="recovery_strength")
    assert_preserves_columns(factor, output_column="recovery_strength")
    assert RecoveryStrengthFactor is RecoveryStrengthFactorDirect
