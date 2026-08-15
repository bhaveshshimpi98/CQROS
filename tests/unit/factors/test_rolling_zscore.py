"""Unit tests for shared rolling z-score zero-variance behavior."""

from __future__ import annotations

import math

import polars as pl
import pytest

from cqros.factors.funding import FundingRateZScoreFactor
from cqros.factors.price import PriceZScoreFactor
from cqros.factors.rolling_zscore import rolling_zscore_expr


def _population_zscore(values: list[float]) -> float:
    """Return population z-score for a fully observed window."""
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    if std == 0.0:
        return 0.0
    return (values[-1] - mean) / std


def test_rolling_zscore_expr_normal_case() -> None:
    """Positive rolling std yields the standard z-score."""
    values = [10.0, 12.0, 11.0, 15.0]
    frame = pl.DataFrame({"x": values})
    result = frame.select(rolling_zscore_expr(pl.col("x"), window_size=3).alias("z")).get_column(
        "z"
    )
    assert result[2] == pytest.approx(_population_zscore(values[0:3]))
    assert result[3] == pytest.approx(_population_zscore(values[1:4]))


def test_rolling_zscore_expr_warmup_is_null() -> None:
    """Incomplete windows remain null."""
    frame = pl.DataFrame({"x": [10.0, 12.0, 11.0]})
    result = frame.select(rolling_zscore_expr(pl.col("x"), window_size=3).alias("z")).get_column(
        "z"
    )
    assert result[0] is None
    assert result[1] is None
    assert result[2] is not None


def test_rolling_zscore_expr_flat_series_is_zero() -> None:
    """Complete zero-variance windows return 0.0, not null."""
    frame = pl.DataFrame({"x": [10.0, 10.0, 10.0, 10.0, 10.0]})
    result = (
        frame.select(rolling_zscore_expr(pl.col("x"), window_size=3).alias("z"))
        .get_column("z")
        .to_list()
    )
    assert result[0] is None
    assert result[1] is None
    assert result[2] == pytest.approx(0.0)
    assert result[3] == pytest.approx(0.0)
    assert result[4] == pytest.approx(0.0)


def test_price_zscore_flat_series_is_zero_after_warmup() -> None:
    """PriceZScoreFactor emits 0.0 on flat series after warmup."""
    values = (
        PriceZScoreFactor(lookback=3)
        .compute(pl.DataFrame({"close": [10.0] * 6}))
        .get_column("price_zscore")
        .to_list()
    )
    assert values[0] is None
    assert values[1] is None
    assert all(value == pytest.approx(0.0) for value in values[2:])


def test_funding_rate_zscore_plateau_has_no_unexpected_nulls() -> None:
    """Repeated funding values never emit unexpected NULLs after warmup."""
    plateau = [0.0001] * 24
    values = (
        FundingRateZScoreFactor(lookback=20)
        .compute(pl.DataFrame({"funding_rate": plateau}))
        .get_column("funding_rate_zscore")
        .to_list()
    )
    assert all(value is None for value in values[:19])
    assert all(value == pytest.approx(0.0) for value in values[19:])
    assert None not in values[19:]
