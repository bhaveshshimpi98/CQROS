"""Unit tests for CQROS ``MomentumFactor``."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError
from cqros.factors.interfaces import Factor
from cqros.factors.price import MomentumFactor
from cqros.factors.price.momentum import MomentumFactor as MomentumFactorDirect


def _factor(*, lookback: int = 20) -> MomentumFactor:
    """Build a momentum factor with an optional lookback override."""
    return MomentumFactor(lookback=lookback)


def test_momentum_factor_metadata() -> None:
    """MomentumFactor exposes the fixed production metadata contract."""
    factor = _factor()
    assert isinstance(factor, BaseFactor)
    assert isinstance(factor, Factor)
    assert factor.name == "momentum"
    assert factor.version == "1.0.0"
    assert factor.category == "price"
    assert (
        factor.description
        == "Pure price momentum as normalized cumulative return over a lookback window."
    )
    assert factor.required_features == ("close",)
    assert factor.produced_columns == ("momentum",)
    assert factor.lookback == 20
    meta = factor.metadata
    assert meta.name == "momentum"
    assert meta.version == "1.0.0"
    assert meta.category == "price"
    assert meta.required_features == ("close",)
    assert meta.produced_columns == ("momentum",)
    assert meta.lookback == 20


def test_momentum_calculation_correctness() -> None:
    """Momentum matches (close / close.shift(lookback)) - 1 after warm-up."""
    frame = pl.DataFrame({"close": [100.0, 110.0, 121.0, 133.1, 146.41]})
    result = _factor(lookback=2).compute(frame)
    values = result.get_column("momentum").to_list()
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx((121.0 / 100.0) - 1.0)
    assert values[3] == pytest.approx((133.1 / 110.0) - 1.0)
    assert values[4] == pytest.approx((146.41 / 121.0) - 1.0)


def test_null_head_rows_match_lookback() -> None:
    """The first lookback momentum values are null and never filled."""
    frame = pl.DataFrame({"close": [10.0, 20.0, 30.0, 40.0, 50.0]})
    result = _factor(lookback=3).compute(frame)
    values = result.get_column("momentum").to_list()
    assert values[0] is None
    assert values[1] is None
    assert values[2] is None
    assert values[3] == pytest.approx((40.0 / 10.0) - 1.0)
    assert values[4] == pytest.approx((50.0 / 20.0) - 1.0)


def test_default_lookback_is_twenty() -> None:
    """Default constructor lookback is 20."""
    assert MomentumFactor().lookback == 20


def test_lookback_zero_raises() -> None:
    """Zero lookback is rejected by MomentumFactor."""
    with pytest.raises(ValidationError, match="lookback must be an integer greater than 0") as (
        exc_info
    ):
        MomentumFactor(lookback=0)
    assert exc_info.value.error_code == "FACTOR-MOMENTUM-001"
    assert exc_info.value.details["parameter"] == "lookback"
    assert exc_info.value.details["value"] == 0


@pytest.mark.parametrize("lookback", [-1, -100])
def test_lookback_negative_raises(lookback: int) -> None:
    """Negative lookback is rejected by BaseFactor validation."""
    with pytest.raises(
        ValidationError,
        match="lookback must be an integer greater than or equal to 0",
    ) as exc_info:
        MomentumFactor(lookback=lookback)
    assert exc_info.value.error_code == "FACTOR-BASE-008"
    assert exc_info.value.details["parameter"] == "lookback"
    assert exc_info.value.details["value"] == lookback


def test_missing_close_column_raises() -> None:
    """Missing close column raises FactorError."""
    frame = pl.DataFrame({"open": [1.0, 2.0, 3.0]})
    with pytest.raises(FactorError, match="required column missing: close") as exc_info:
        _factor(lookback=1).compute(frame)
    error = exc_info.value
    assert error.error_code == "FACTOR-MOMENTUM-002"
    assert error.details["factor"] == "momentum"
    assert error.details["required_column"] == "close"
    assert error.details["available_columns"] == ("open",)


def test_input_immutability() -> None:
    """compute does not mutate the caller-supplied DataFrame."""
    frame = pl.DataFrame({"close": [1.0, 2.0, 3.0]})
    original_columns = list(frame.columns)
    original_values = frame.get_column("close").to_list()
    result = _factor(lookback=1).compute(frame)
    assert list(frame.columns) == original_columns
    assert frame.get_column("close").to_list() == original_values
    assert "momentum" not in frame.columns
    assert "momentum" in result.columns
    assert result is not frame


def test_preserves_existing_columns() -> None:
    """Existing non-close columns are preserved alongside momentum."""
    frame = pl.DataFrame({"close": [1.0, 2.0, 3.0], "volume": [10, 20, 30]})
    result = _factor(lookback=1).compute(frame)
    assert result.columns == ["close", "volume", "momentum"]


def test_package_exports_momentum_factor() -> None:
    """MomentumFactor is exported from the price package."""
    assert MomentumFactor is MomentumFactorDirect
    import cqros.factors.price as price_package

    assert "MomentumFactor" in price_package.__all__
