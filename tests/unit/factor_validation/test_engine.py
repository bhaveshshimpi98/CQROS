"""Unit tests for CQROS ``SimpleFactorValidationEngine`` Phase-1 statistics."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from math import sqrt

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.factor_validation import (
    FACTOR_VALIDATION_SCHEMA,
    FactorValidationError,
    FactorValidationStatus,
    SimpleFactorValidationEngine,
)
from cqros.factor_validation.engine import FACTOR_INPUT_COLUMNS, validate_factor_frame
from cqros.factor_validation.schema import CANONICAL_COLUMN_ORDER

_TIMEFRAME = "1h"
_SYMBOL = "BTCUSDT"
_FACTOR_NAME = "momentum"
_FACTOR_VERSION = "1.0.0"
_FACTOR_CATEGORY = "price"


def _open_time(index: int = 0) -> datetime:
    """Build a deterministic UTC open_time for row ``index``."""
    return datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=index)


def _epoch_ms(value: datetime) -> int:
    """Convert a UTC datetime to epoch milliseconds."""
    return int(value.timestamp() * 1000.0)


def _factors_frame(
    *,
    open_times: list[datetime] | list[int] | None = None,
    factor_names: list[str] | None = None,
    factor_versions: list[str] | None = None,
    factor_categories: list[str] | None = None,
    timeframes: list[str] | None = None,
    factor_values: list[float | None] | None = None,
    future_returns: list[float | None] | None = None,
    symbols: list[str] | None = None,
    symbol: str = _SYMBOL,
) -> pl.DataFrame:
    """Build a Factors frame with Phase-1 IC input columns for engine tests."""
    open_times = open_times if open_times is not None else [_open_time(0), _open_time(1)]
    row_count = len(open_times)
    factor_names = factor_names if factor_names is not None else [_FACTOR_NAME] * row_count
    factor_versions = (
        factor_versions if factor_versions is not None else [_FACTOR_VERSION] * row_count
    )
    factor_categories = (
        factor_categories if factor_categories is not None else [_FACTOR_CATEGORY] * row_count
    )
    timeframes = timeframes if timeframes is not None else [_TIMEFRAME] * row_count
    factor_values = (
        factor_values
        if factor_values is not None
        else [float(index + 1) for index in range(row_count)]
    )
    future_returns = (
        future_returns
        if future_returns is not None
        else [float(index + 1) for index in range(row_count)]
    )
    symbol_values = symbols if symbols is not None else [symbol] * row_count
    return pl.DataFrame(
        {
            "symbol": symbol_values,
            "timeframe": timeframes,
            "open_time": open_times,
            "factor_name": factor_names,
            "factor_version": factor_versions,
            "factor_category": factor_categories,
            "factor_group": ["alpha"] * row_count,
            "factor_value": factor_values,
            "future_return_1": future_returns,
            "lookback": [20] * row_count,
            "prediction_horizon": [1] * row_count,
            "enabled": [True] * row_count,
            "status": ["ACTIVE"] * row_count,
        }
    )


def _cross_section_panel(
    *,
    panels: list[tuple[datetime, list[tuple[float | None, float | None]]]],
) -> pl.DataFrame:
    """Build a multi-asset Factors frame from per-timestamp cross-sections.

    Args:
        panels: Chronological panels as ``(open_time, [(factor, return), ...])``.
            Asset symbols are assigned deterministically as ``S00``, ``S01``, …
    """
    open_times: list[datetime] = []
    factor_values: list[float | None] = []
    future_returns: list[float | None] = []
    symbols: list[str] = []
    for open_time, pairs in panels:
        for asset_index, (factor_value, future_return) in enumerate(pairs):
            open_times.append(open_time)
            factor_values.append(factor_value)
            future_returns.append(future_return)
            symbols.append(f"S{asset_index:02d}")
    return _factors_frame(
        open_times=open_times,
        factor_values=factor_values,
        future_returns=future_returns,
        symbols=symbols,
    )


def _multi_horizon_panel(
    *,
    panels: list[tuple[datetime, list[tuple[float | None, ...]]]],
    horizon_columns: tuple[str, ...],
) -> pl.DataFrame:
    """Build a multi-asset Factors frame with multiple forward-return horizons.

    Args:
        panels: Chronological panels as
            ``(open_time, [(factor, ret_h1, ret_h2, ...), ...])``.
            Return values align with ``horizon_columns`` in order.
        horizon_columns: Forward-return column names such as
            ``("future_return_1", "future_return_5")``.
    """
    if not horizon_columns:
        raise ValueError("horizon_columns must contain at least one column")
    if "future_return_1" not in horizon_columns:
        raise ValueError("horizon_columns must include future_return_1")

    open_times: list[datetime] = []
    factor_values: list[float | None] = []
    symbols: list[str] = []
    horizon_values: dict[str, list[float | None]] = {column: [] for column in horizon_columns}
    expected_width = 1 + len(horizon_columns)
    for open_time, rows in panels:
        for asset_index, row in enumerate(rows):
            if len(row) != expected_width:
                raise ValueError(f"expected {expected_width} values per asset, got {len(row)}")
            open_times.append(open_time)
            factor_values.append(row[0])
            symbols.append(f"S{asset_index:02d}")
            for column_index, column in enumerate(horizon_columns):
                horizon_values[column].append(row[column_index + 1])

    frame = _factors_frame(
        open_times=open_times,
        factor_values=factor_values,
        future_returns=horizon_values["future_return_1"],
        symbols=symbols,
    )
    extra_columns = [
        pl.Series(column, horizon_values[column], dtype=pl.Float64)
        for column in horizon_columns
        if column != "future_return_1"
    ]
    if not extra_columns:
        return frame
    return frame.with_columns(extra_columns)


def _quintile_panel(
    *,
    panels: list[tuple[datetime, list[float], list[float]]],
) -> pl.DataFrame:
    """Build five-or-more-asset panels as ``(open_time, factors, returns)``."""
    return _cross_section_panel(
        panels=[
            (open_time, list(zip(factor_values, returns, strict=True)))
            for open_time, factor_values, returns in panels
        ]
    )


def _build(
    engine: SimpleFactorValidationEngine,
    *,
    factors: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build factor validation metrics with a default Factors frame."""
    return engine.build(factors if factors is not None else _factors_frame())


# ---------------------------------------------------------------------------
# Input column contracts
# ---------------------------------------------------------------------------


def test_input_columns_contract() -> None:
    """FACTOR_INPUT_COLUMNS enumerates every column the engine consumes."""
    for column in (
        "factor_name",
        "factor_version",
        "factor_category",
        "timeframe",
        "open_time",
        "symbol",
        "factor_value",
        "future_return_1",
    ):
        assert column in FACTOR_INPUT_COLUMNS


# ---------------------------------------------------------------------------
# Frame validator tests
# ---------------------------------------------------------------------------


def test_validate_factor_frame_rejects_non_dataframe() -> None:
    """validate_factor_frame rejects non-DataFrame inputs with FVAL_FRAME_TYPE."""
    with pytest.raises(FactorValidationError) as exc_info:
        validate_factor_frame("not-a-frame")  # type: ignore[arg-type]
    assert exc_info.value.error_code == "FVAL_FRAME_TYPE"


def test_validate_factor_frame_rejects_empty_dataframe() -> None:
    """validate_factor_frame rejects DataFrames with zero rows."""
    empty = pl.DataFrame({"factor_name": []})
    with pytest.raises(FactorValidationError) as exc_info:
        validate_factor_frame(empty)
    assert exc_info.value.error_code == "FVAL_FRAME_EMPTY"


def test_build_rejects_empty_dataframe() -> None:
    """build rejects empty Factors frames."""
    empty = pl.DataFrame(schema={column: pl.String for column in ("factor_name",)}).clear()
    with pytest.raises(FactorValidationError) as exc_info:
        SimpleFactorValidationEngine().build(empty)
    assert exc_info.value.error_code == "FVAL_FRAME_EMPTY"


# ---------------------------------------------------------------------------
# Missing column validation
# ---------------------------------------------------------------------------


def test_build_rejects_missing_factor_columns() -> None:
    """Missing required Factors columns raise FVAL_MISSING_COLUMNS."""
    engine = SimpleFactorValidationEngine()
    with pytest.raises(FactorValidationError) as exc_info:
        _build(engine, factors=_factors_frame().drop("factor_category"))
    assert exc_info.value.error_code == "FVAL_MISSING_COLUMNS"


def test_build_rejects_missing_factor_value() -> None:
    """Missing factor_value raises FVAL_MISSING_COLUMNS."""
    engine = SimpleFactorValidationEngine()
    with pytest.raises(FactorValidationError) as exc_info:
        _build(engine, factors=_factors_frame().drop("factor_value"))
    assert exc_info.value.error_code == "FVAL_MISSING_COLUMNS"


def test_build_rejects_missing_future_return() -> None:
    """Missing future_return_1 raises FVAL_MISSING_COLUMNS."""
    engine = SimpleFactorValidationEngine()
    with pytest.raises(FactorValidationError) as exc_info:
        _build(engine, factors=_factors_frame().drop("future_return_1"))
    assert exc_info.value.error_code == "FVAL_MISSING_COLUMNS"


# ---------------------------------------------------------------------------
# Identity, grouping, and metadata
# ---------------------------------------------------------------------------


def test_one_validation_row_per_factor() -> None:
    """Engine emits exactly one validation row for each unique factor identity."""
    t0, t1 = _open_time(0), _open_time(1)
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_factors_frame(
            open_times=[t0, t1, t0, t1],
            factor_names=["momentum", "momentum", "rsi", "rsi"],
            factor_versions=["1.0.0", "1.0.0", "1.0.0", "1.0.0"],
            factor_categories=["price", "price", "momentum", "momentum"],
            factor_values=[1.0, 2.0, 1.0, 2.0],
            future_returns=[1.0, 2.0, 2.0, 1.0],
        ),
    )
    assert result.height == 2
    assert result["factor_name"].to_list() == ["momentum", "rsi"]


def test_preserves_factor_identity_and_metadata() -> None:
    """factor_name, factor_version, factor_category, and timeframe are preserved."""
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_factors_frame(
            factor_names=["breakout", "breakout"],
            factor_versions=["2.1.0", "2.1.0"],
            factor_categories=["composite", "composite"],
            timeframes=["4h", "4h"],
            factor_values=[1.0, 2.0],
            future_returns=[1.0, 2.0],
        ),
    )
    assert result["factor_name"].to_list() == ["breakout"]
    assert result["factor_version"].to_list() == ["2.1.0"]
    assert result["factor_category"].to_list() == ["composite"]
    assert result["timeframe"].to_list() == ["4h"]


def test_validation_time_is_current_factor_timestamp() -> None:
    """validation_time uses the latest open_time for each factor."""
    t0, t1, t2 = _open_time(0), _open_time(1), _open_time(2)
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_factors_frame(
            open_times=[t0, t1, t2],
            factor_values=[1.0, 2.0, 3.0],
            future_returns=[1.0, 2.0, 3.0],
        ),
    )
    assert result["validation_time"].to_list() == [_epoch_ms(t2)]


# ---------------------------------------------------------------------------
# Cross-sectional Pearson IC
# ---------------------------------------------------------------------------


def test_perfect_positive_cross_sectional_ic_mean() -> None:
    """information_coefficient is 1.0 when every cross-section has Pearson IC 1."""
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_cross_section_panel(
            panels=[
                (_open_time(0), [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]),
                (_open_time(1), [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]),
            ]
        ),
    )
    assert result["information_coefficient"].to_list()[0] == pytest.approx(1.0)


def test_perfect_negative_cross_sectional_ic_mean() -> None:
    """information_coefficient is -1.0 when every cross-section has Pearson IC -1."""
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_cross_section_panel(
            panels=[
                (_open_time(0), [(1.0, 3.0), (2.0, 2.0), (3.0, 1.0)]),
                (_open_time(1), [(1.0, 3.0), (2.0, 2.0), (3.0, 1.0)]),
            ]
        ),
    )
    assert result["information_coefficient"].to_list()[0] == pytest.approx(-1.0)


def test_zero_cross_sectional_ic_mean() -> None:
    """information_coefficient is 0.0 when every cross-section has Pearson IC 0."""
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_cross_section_panel(
            panels=[
                (_open_time(0), [(1.0, 1.0), (2.0, 0.0), (3.0, 1.0)]),
                (_open_time(1), [(1.0, 1.0), (2.0, 0.0), (3.0, 1.0)]),
            ]
        ),
    )
    assert result["information_coefficient"].to_list()[0] == pytest.approx(0.0)


def test_varying_ic_series_mean() -> None:
    """information_coefficient equals the mean of a mixed cross-sectional IC series."""
    # IC series = [1.0, -1.0] => mean = 0.0
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_cross_section_panel(
            panels=[
                (_open_time(0), [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]),
                (_open_time(1), [(1.0, 3.0), (2.0, 2.0), (3.0, 1.0)]),
            ]
        ),
    )
    assert result["information_coefficient"].to_list()[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Spearman Rank IC (pooled)
# ---------------------------------------------------------------------------


def test_perfect_positive_spearman_correlation() -> None:
    """Rank IC is 1.0 for a strictly monotone positive factor/return relationship."""
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_factors_frame(
            open_times=[_open_time(0), _open_time(1), _open_time(2)],
            factor_values=[1.0, 2.0, 3.0],
            future_returns=[1.0, 4.0, 9.0],
        ),
    )
    assert result["rank_information_coefficient"].to_list()[0] == pytest.approx(1.0)


def test_perfect_negative_spearman_correlation() -> None:
    """Rank IC is -1.0 for a strictly monotone negative factor/return relationship."""
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_factors_frame(
            open_times=[_open_time(0), _open_time(1), _open_time(2)],
            factor_values=[1.0, 2.0, 3.0],
            future_returns=[9.0, 4.0, 1.0],
        ),
    )
    assert result["rank_information_coefficient"].to_list()[0] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# ICIR
# ---------------------------------------------------------------------------


def test_constant_ic_series_yields_null_icir() -> None:
    """Constant IC series has zero sample std and therefore null ICIR."""
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_cross_section_panel(
            panels=[
                (_open_time(0), [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]),
                (_open_time(1), [(2.0, 2.0), (4.0, 4.0), (6.0, 6.0)]),
            ]
        ),
    )
    assert result["information_coefficient"].to_list()[0] == pytest.approx(1.0)
    assert result["ic_information_ratio"].to_list() == [None]
    assert result["ic_t_stat"].to_list() == [None]
    assert result["ic_p_value"].to_list() == [None]


def test_varying_ic_series_positive_icir() -> None:
    """Varying positive IC series produces a positive ICIR."""
    # IC series = [1.0, 0.0] => mean = 0.5, sample std = sqrt(0.5), ICIR = sqrt(0.5)
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_cross_section_panel(
            panels=[
                (_open_time(0), [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]),
                (_open_time(1), [(1.0, 1.0), (2.0, 0.0), (3.0, 1.0)]),
            ]
        ),
    )
    expected_icir = 0.5 / sqrt(0.5)
    assert result["information_coefficient"].to_list()[0] == pytest.approx(0.5)
    assert result["ic_information_ratio"].to_list()[0] == pytest.approx(expected_icir)
    assert result["ic_information_ratio"].to_list()[0] > 0.0


def test_insufficient_ic_history_yields_null_icir() -> None:
    """Fewer than two valid IC timestamps yields null ICIR."""
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_cross_section_panel(
            panels=[
                (_open_time(0), [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]),
            ]
        ),
    )
    assert result["information_coefficient"].to_list()[0] == pytest.approx(1.0)
    assert result["ic_information_ratio"].to_list() == [None]
    assert result["ic_t_stat"].to_list() == [None]
    assert result["ic_p_value"].to_list() == [None]
    assert result["status"].to_list() == [FactorValidationStatus.PASS.value]


def test_timestamps_without_enough_observations_are_skipped_for_icir() -> None:
    """Timestamps with fewer than two valid rows are excluded from the IC series."""
    # t0 skipped (1 valid row after nulls); t1 and t2 form IC series [1.0, 1.0]
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_cross_section_panel(
            panels=[
                (_open_time(0), [(1.0, 1.0), (None, 2.0), (3.0, None)]),
                (_open_time(1), [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]),
                (_open_time(2), [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]),
            ]
        ),
    )
    assert result["information_coefficient"].to_list()[0] == pytest.approx(1.0)
    assert result["ic_information_ratio"].to_list() == [None]
    assert result["ic_t_stat"].to_list() == [None]
    assert result["ic_p_value"].to_list() == [None]


# ---------------------------------------------------------------------------
# IC t-statistic
# ---------------------------------------------------------------------------


def test_positive_ic_t_stat() -> None:
    """IC t-stat is positive for a positively biased IC series."""
    # IC series = [1.0, 0.0] => mean = 0.5, std = sqrt(0.5), N = 2
    # t = 0.5 / (sqrt(0.5) / sqrt(2)) = 1.0
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_cross_section_panel(
            panels=[
                (_open_time(0), [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]),
                (_open_time(1), [(1.0, 1.0), (2.0, 0.0), (3.0, 1.0)]),
            ]
        ),
    )
    assert result["ic_t_stat"].to_list()[0] == pytest.approx(1.0)
    assert result["ic_t_stat"].to_list()[0] > 0.0
    assert result["status"].to_list() == [FactorValidationStatus.PASS.value]


def test_negative_ic_t_stat() -> None:
    """IC t-stat is negative for a negatively biased IC series."""
    # IC series = [-1.0, 0.0] => mean = -0.5, std = sqrt(0.5), N = 2
    # t = -0.5 / (sqrt(0.5) / sqrt(2)) = -1.0
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_cross_section_panel(
            panels=[
                (_open_time(0), [(1.0, 3.0), (2.0, 2.0), (3.0, 1.0)]),
                (_open_time(1), [(1.0, 1.0), (2.0, 0.0), (3.0, 1.0)]),
            ]
        ),
    )
    assert result["ic_t_stat"].to_list()[0] == pytest.approx(-1.0)
    assert result["ic_t_stat"].to_list()[0] < 0.0


def test_zero_variance_yields_null_ic_t_stat() -> None:
    """Zero IC-series sample std yields null IC t-stat."""
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_cross_section_panel(
            panels=[
                (_open_time(0), [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]),
                (_open_time(1), [(2.0, 2.0), (4.0, 4.0), (6.0, 6.0)]),
            ]
        ),
    )
    assert result["ic_t_stat"].to_list() == [None]


def test_insufficient_ic_history_yields_null_ic_t_stat() -> None:
    """Fewer than two valid IC timestamps yields null IC t-stat."""
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_cross_section_panel(
            panels=[
                (_open_time(0), [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]),
            ]
        ),
    )
    assert result["ic_t_stat"].to_list() == [None]


# ---------------------------------------------------------------------------
# IC p-value
# ---------------------------------------------------------------------------


def test_highly_significant_ic_series_has_small_p_value() -> None:
    """A strongly positive multi-period IC series yields a small two-sided p-value."""
    # Four cross-sections with ICs near 1.0 produce a large |t| and tiny p-value.
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_cross_section_panel(
            panels=[
                (_open_time(0), [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]),
                (_open_time(1), [(1.0, 1.1), (2.0, 2.0), (3.0, 2.9)]),
                (_open_time(2), [(1.0, 0.9), (2.0, 2.1), (3.0, 3.0)]),
                (_open_time(3), [(1.0, 1.0), (2.0, 1.9), (3.0, 3.1)]),
            ]
        ),
    )
    assert result["ic_t_stat"].to_list()[0] > 0.0
    assert result["ic_p_value"].to_list()[0] == pytest.approx(0.0, abs=1e-3)
    assert result["ic_p_value"].to_list()[0] < 0.05
    assert result["status"].to_list() == [FactorValidationStatus.PASS.value]


def test_weak_ic_series_has_large_p_value() -> None:
    """A mean-zero IC series yields a two-sided p-value of 1.0."""
    # IC series = [1.0, -1.0] => mean = 0, t = 0, p = 1
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_cross_section_panel(
            panels=[
                (_open_time(0), [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]),
                (_open_time(1), [(1.0, 3.0), (2.0, 2.0), (3.0, 1.0)]),
            ]
        ),
    )
    assert result["information_coefficient"].to_list()[0] == pytest.approx(0.0)
    assert result["ic_t_stat"].to_list()[0] == pytest.approx(0.0)
    assert result["ic_p_value"].to_list()[0] == pytest.approx(1.0)


def test_positive_and_negative_t_stats_share_two_sided_p_value() -> None:
    """Two-sided p-value depends on |t| and is identical for mirrored IC series."""
    positive = _build(
        SimpleFactorValidationEngine(),
        factors=_cross_section_panel(
            panels=[
                (_open_time(0), [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]),
                (_open_time(1), [(1.0, 1.0), (2.0, 0.0), (3.0, 1.0)]),
            ]
        ),
    )
    negative = _build(
        SimpleFactorValidationEngine(),
        factors=_cross_section_panel(
            panels=[
                (_open_time(0), [(1.0, 3.0), (2.0, 2.0), (3.0, 1.0)]),
                (_open_time(1), [(1.0, 1.0), (2.0, 0.0), (3.0, 1.0)]),
            ]
        ),
    )
    assert positive["ic_t_stat"].to_list()[0] == pytest.approx(1.0)
    assert negative["ic_t_stat"].to_list()[0] == pytest.approx(-1.0)
    assert positive["ic_p_value"].to_list()[0] == pytest.approx(negative["ic_p_value"].to_list()[0])
    # df = 1, |t| = 1 => two-sided p = 0.5
    assert positive["ic_p_value"].to_list()[0] == pytest.approx(0.5)


def test_insufficient_ic_history_yields_null_ic_p_value() -> None:
    """Fewer than two valid IC timestamps yields null IC p-value."""
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_cross_section_panel(
            panels=[
                (_open_time(0), [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]),
            ]
        ),
    )
    assert result["ic_p_value"].to_list() == [None]


def test_zero_variance_yields_null_ic_p_value() -> None:
    """Zero IC-series sample std yields null IC p-value via null t-stat."""
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_cross_section_panel(
            panels=[
                (_open_time(0), [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]),
                (_open_time(1), [(2.0, 2.0), (4.0, 4.0), (6.0, 6.0)]),
            ]
        ),
    )
    assert result["ic_t_stat"].to_list() == [None]
    assert result["ic_p_value"].to_list() == [None]


# ---------------------------------------------------------------------------
# Null handling, observations, and status
# ---------------------------------------------------------------------------


def test_null_values_are_excluded_from_cross_sectional_ic() -> None:
    """Null factor/return pairs are ignored inside each cross-section."""
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_cross_section_panel(
            panels=[
                (_open_time(0), [(1.0, 1.0), (None, 2.0), (3.0, 3.0)]),
                (_open_time(1), [(1.0, 1.0), (2.0, None), (4.0, 4.0)]),
            ]
        ),
    )
    assert result["observations"].to_list() == [4]
    assert result["information_coefficient"].to_list()[0] == pytest.approx(1.0)
    assert result["ic_t_stat"].to_list() == [None]
    assert result["status"].to_list() == [FactorValidationStatus.PASS.value]


def test_insufficient_observations_yield_null_ics_and_fail() -> None:
    """Fewer than two valid observations yields null ICs and FAIL status."""
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_factors_frame(
            open_times=[_open_time(0)],
            factor_values=[1.0],
            future_returns=[2.0],
        ),
    )
    assert result["observations"].to_list() == [1]
    assert result["information_coefficient"].to_list() == [None]
    assert result["rank_information_coefficient"].to_list() == [None]
    assert result["ic_information_ratio"].to_list() == [None]
    assert result["ic_t_stat"].to_list() == [None]
    assert result["ic_p_value"].to_list() == [None]
    assert result["status"].to_list() == [FactorValidationStatus.FAIL.value]


def test_null_pair_reduces_observations_to_insufficient() -> None:
    """A single remaining valid pair after null removal yields FAIL."""
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_factors_frame(
            open_times=[_open_time(0), _open_time(1)],
            factor_values=[1.0, None],
            future_returns=[1.0, 2.0],
        ),
    )
    assert result["observations"].to_list() == [1]
    assert result["information_coefficient"].to_list() == [None]
    assert result["rank_information_coefficient"].to_list() == [None]
    assert result["ic_information_ratio"].to_list() == [None]
    assert result["ic_t_stat"].to_list() == [None]
    assert result["ic_p_value"].to_list() == [None]
    assert result["status"].to_list() == [FactorValidationStatus.FAIL.value]


def test_observation_count_matches_valid_pairs() -> None:
    """observations equals the number of non-null factor/return pairs."""
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_factors_frame(
            open_times=[_open_time(i) for i in range(5)],
            factor_values=[1.0, 2.0, None, 4.0, 5.0],
            future_returns=[1.0, None, 3.0, 4.0, 5.0],
        ),
    )
    assert result["observations"].to_list() == [3]


def test_pass_status_when_observations_sufficient() -> None:
    """status is PASS when at least two valid observations remain."""
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_factors_frame(
            open_times=[_open_time(0), _open_time(1)],
            factor_values=[1.0, 2.0],
            future_returns=[1.0, 2.0],
        ),
    )
    assert result["status"].to_list() == [FactorValidationStatus.PASS.value]


def test_fail_status_when_observations_insufficient() -> None:
    """status is FAIL when fewer than two valid observations remain."""
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_factors_frame(
            open_times=[_open_time(0), _open_time(1)],
            factor_values=[None, None],
            future_returns=[1.0, 2.0],
        ),
    )
    assert result["observations"].to_list() == [0]
    assert result["status"].to_list() == [FactorValidationStatus.FAIL.value]


# ---------------------------------------------------------------------------
# IC Decay
# ---------------------------------------------------------------------------


def test_constant_ic_across_horizons_yields_unit_decay() -> None:
    """Identical mean IC at first and last horizons yields ic_decay = 1.0."""
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_multi_horizon_panel(
            horizon_columns=("future_return_1", "future_return_5"),
            panels=[
                (
                    _open_time(0),
                    [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (3.0, 3.0, 3.0)],
                ),
                (
                    _open_time(1),
                    [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (3.0, 3.0, 3.0)],
                ),
            ],
        ),
    )
    assert result["ic_decay"].to_list()[0] == pytest.approx(1.0)
    assert result["status"].to_list() == [FactorValidationStatus.PASS.value]


def test_decaying_ic_across_horizons() -> None:
    """Weaker terminal-horizon IC yields ic_decay between 0 and 1."""
    # Horizon 1 mean IC = 1.0; horizon 5 mean IC = 0.5 => decay = 0.5
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_multi_horizon_panel(
            horizon_columns=("future_return_1", "future_return_5"),
            panels=[
                (
                    _open_time(0),
                    [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (3.0, 3.0, 3.0)],
                ),
                (
                    _open_time(1),
                    [(1.0, 1.0, 1.0), (2.0, 2.0, 0.0), (3.0, 3.0, 1.0)],
                ),
            ],
        ),
    )
    assert result["ic_decay"].to_list()[0] == pytest.approx(0.5)
    assert 0.0 < result["ic_decay"].to_list()[0] < 1.0


def test_increasing_ic_across_horizons() -> None:
    """Stronger terminal-horizon IC yields ic_decay greater than 1.0."""
    # Horizon 1 mean IC = 0.5; horizon 5 mean IC = 1.0 => decay = 2.0
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_multi_horizon_panel(
            horizon_columns=("future_return_1", "future_return_5"),
            panels=[
                (
                    _open_time(0),
                    [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (3.0, 3.0, 3.0)],
                ),
                (
                    _open_time(1),
                    [(1.0, 1.0, 1.0), (2.0, 0.0, 2.0), (3.0, 1.0, 3.0)],
                ),
            ],
        ),
    )
    assert result["ic_decay"].to_list()[0] == pytest.approx(2.0)
    assert result["ic_decay"].to_list()[0] > 1.0


def test_reversing_ic_across_horizons() -> None:
    """Opposite-signed terminal IC yields negative ic_decay."""
    # Horizon 1 mean IC = 1.0; horizon 5 mean IC = -1.0 => decay = -1.0
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_multi_horizon_panel(
            horizon_columns=("future_return_1", "future_return_5"),
            panels=[
                (
                    _open_time(0),
                    [(1.0, 1.0, 3.0), (2.0, 2.0, 2.0), (3.0, 3.0, 1.0)],
                ),
                (
                    _open_time(1),
                    [(1.0, 1.0, 3.0), (2.0, 2.0, 2.0), (3.0, 3.0, 1.0)],
                ),
            ],
        ),
    )
    assert result["ic_decay"].to_list()[0] == pytest.approx(-1.0)
    assert result["ic_decay"].to_list()[0] < 0.0


def test_missing_intermediate_horizons_use_present_endpoints() -> None:
    """IC Decay uses first and last present horizons when intermediates are absent."""
    # Present: 1 and 10 only. Both mean IC = 1.0 => decay = 1.0
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_multi_horizon_panel(
            horizon_columns=("future_return_1", "future_return_10"),
            panels=[
                (
                    _open_time(0),
                    [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (3.0, 3.0, 3.0)],
                ),
                (
                    _open_time(1),
                    [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (3.0, 3.0, 3.0)],
                ),
            ],
        ),
    )
    assert result["ic_decay"].to_list()[0] == pytest.approx(1.0)


def test_insufficient_horizons_yield_null_ic_decay() -> None:
    """A single present horizon yields null ic_decay."""
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_cross_section_panel(
            panels=[
                (_open_time(0), [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]),
                (_open_time(1), [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]),
            ]
        ),
    )
    assert result["ic_decay"].to_list() == [None]
    assert result["status"].to_list() == [FactorValidationStatus.PASS.value]


def test_zero_first_horizon_ic_yields_null_ic_decay() -> None:
    """Zero mean IC at the first horizon yields null ic_decay."""
    # Horizon 1 mean IC = 0.0; horizon 5 mean IC = 1.0 => undefined ratio
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_multi_horizon_panel(
            horizon_columns=("future_return_1", "future_return_5"),
            panels=[
                (
                    _open_time(0),
                    [(1.0, 1.0, 1.0), (2.0, 0.0, 2.0), (3.0, 1.0, 3.0)],
                ),
                (
                    _open_time(1),
                    [(1.0, 1.0, 1.0), (2.0, 0.0, 2.0), (3.0, 1.0, 3.0)],
                ),
            ],
        ),
    )
    assert result["ic_decay"].to_list() == [None]


def test_null_values_are_excluded_from_ic_decay_horizons() -> None:
    """Null factor/return pairs are ignored inside each horizon cross-section."""
    # After null removal each horizon still has perfect positive IC => decay = 1
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_multi_horizon_panel(
            horizon_columns=("future_return_1", "future_return_5"),
            panels=[
                (
                    _open_time(0),
                    [
                        (1.0, 1.0, 1.0),
                        (None, 9.0, 9.0),
                        (2.0, 2.0, None),
                        (3.0, 3.0, 3.0),
                    ],
                ),
                (
                    _open_time(1),
                    [
                        (1.0, 1.0, 1.0),
                        (2.0, None, 2.0),
                        (3.0, 3.0, 3.0),
                    ],
                ),
            ],
        ),
    )
    assert result["ic_decay"].to_list()[0] == pytest.approx(1.0)


def test_ic_decay_output_is_deterministic() -> None:
    """Identical multi-horizon inputs produce identical ic_decay values."""
    factors = _multi_horizon_panel(
        horizon_columns=("future_return_1", "future_return_2", "future_return_5"),
        panels=[
            (
                _open_time(0),
                [(1.0, 1.0, 1.0, 3.0), (2.0, 2.0, 0.0, 2.0), (3.0, 3.0, 1.0, 1.0)],
            ),
            (
                _open_time(1),
                [(1.0, 1.0, 1.0, 3.0), (2.0, 2.0, 2.0, 2.0), (3.0, 3.0, 3.0, 1.0)],
            ),
        ],
    )
    engine = SimpleFactorValidationEngine()
    first = engine.build(factors)
    second = engine.build(factors)
    assert_frame_equal(first, second)
    assert first["ic_decay"].to_list()[0] == pytest.approx(-1.0)


def test_ic_decay_preserves_schema_and_pass_status() -> None:
    """Multi-horizon IC Decay preserves schema and PASS status."""
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_multi_horizon_panel(
            horizon_columns=("future_return_1", "future_return_3", "future_return_20"),
            panels=[
                (
                    _open_time(0),
                    [(1.0, 1.0, 1.0, 1.0), (2.0, 2.0, 2.0, 2.0), (3.0, 3.0, 3.0, 3.0)],
                ),
                (
                    _open_time(1),
                    [(1.0, 1.0, 1.0, 1.0), (2.0, 2.0, 2.0, 2.0), (3.0, 3.0, 3.0, 3.0)],
                ),
            ],
        ),
    )
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result.schema == FACTOR_VALIDATION_SCHEMA
    assert result.schema["ic_decay"] == pl.Float64
    assert result["ic_decay"].to_list()[0] == pytest.approx(1.0)
    assert result["status"].to_list() == [FactorValidationStatus.PASS.value]


def test_fail_status_keeps_null_ic_decay_with_single_horizon() -> None:
    """FAIL rows with only future_return_1 still emit null ic_decay."""
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_factors_frame(
            open_times=[_open_time(0)],
            factor_values=[1.0],
            future_returns=[2.0],
        ),
    )
    assert result["status"].to_list() == [FactorValidationStatus.FAIL.value]
    assert result["ic_decay"].to_list() == [None]


# ---------------------------------------------------------------------------
# Quantile return spread
# ---------------------------------------------------------------------------


def test_positive_quantile_spread() -> None:
    """Q5−Q1 is positive when higher factor values earn higher returns."""
    # Q1 return = 1.0, Q5 return = 5.0 => spread = 4.0
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_quintile_panel(
            panels=[
                (
                    _open_time(0),
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                ),
                (
                    _open_time(1),
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                ),
            ]
        ),
    )
    assert result["quantile_spread"].to_list()[0] == pytest.approx(4.0)
    assert result["status"].to_list() == [FactorValidationStatus.PASS.value]


def test_negative_quantile_spread() -> None:
    """Q5−Q1 is negative when higher factor values earn lower returns."""
    # Q1 return = 5.0, Q5 return = 1.0 => spread = -4.0
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_quintile_panel(
            panels=[
                (
                    _open_time(0),
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                    [5.0, 4.0, 3.0, 2.0, 1.0],
                ),
            ]
        ),
    )
    assert result["quantile_spread"].to_list()[0] == pytest.approx(-4.0)


def test_zero_quantile_spread() -> None:
    """Q5−Q1 is zero when top and bottom quantile mean returns match."""
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_quintile_panel(
            panels=[
                (
                    _open_time(0),
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                    [7.0, 0.0, 0.0, 0.0, 7.0],
                ),
            ]
        ),
    )
    assert result["quantile_spread"].to_list()[0] == pytest.approx(0.0)


def test_insufficient_assets_yield_null_quantile_spread() -> None:
    """Timestamps with fewer than five valid assets are skipped for spreads."""
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_cross_section_panel(
            panels=[
                (_open_time(0), [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)]),
                (_open_time(1), [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)]),
            ]
        ),
    )
    assert result["quantile_spread"].to_list() == [None]
    assert result["status"].to_list() == [FactorValidationStatus.PASS.value]


def test_null_values_are_excluded_from_quantile_spread() -> None:
    """Null factor/return pairs are ignored before forming quantiles."""
    # Six rows with one null pair => five valid assets => Q1=1, Q5=5, spread=4
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_cross_section_panel(
            panels=[
                (
                    _open_time(0),
                    [
                        (1.0, 1.0),
                        (2.0, 2.0),
                        (None, 9.0),
                        (3.0, 3.0),
                        (4.0, 4.0),
                        (5.0, 5.0),
                    ],
                ),
            ]
        ),
    )
    assert result["observations"].to_list() == [5]
    assert result["quantile_spread"].to_list()[0] == pytest.approx(4.0)


def test_quantile_spread_averages_across_timestamps() -> None:
    """quantile_spread equals the mean of the chronological spread series."""
    # t0 spread = 4.0, t1 spread = -4.0 => mean = 0.0
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_quintile_panel(
            panels=[
                (
                    _open_time(0),
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                ),
                (
                    _open_time(1),
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                    [5.0, 4.0, 3.0, 2.0, 1.0],
                ),
            ]
        ),
    )
    assert result["quantile_spread"].to_list()[0] == pytest.approx(0.0)


def test_fail_status_keeps_null_quantile_spread() -> None:
    """FAIL rows with insufficient observations emit null quantile_spread."""
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_factors_frame(
            open_times=[_open_time(0)],
            factor_values=[1.0],
            future_returns=[2.0],
        ),
    )
    assert result["status"].to_list() == [FactorValidationStatus.FAIL.value]
    assert result["quantile_spread"].to_list() == [None]
    assert result["monotonicity_score"].to_list() == [None]


# ---------------------------------------------------------------------------
# Monotonicity score
# ---------------------------------------------------------------------------


def test_perfectly_increasing_quantiles_yield_monotonicity_one() -> None:
    """Non-decreasing Q1–Q5 mean returns yield monotonicity_score = 1.0."""
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_quintile_panel(
            panels=[
                (
                    _open_time(0),
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                ),
                (
                    _open_time(1),
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                    [10.0, 20.0, 30.0, 40.0, 50.0],
                ),
            ]
        ),
    )
    assert result["monotonicity_score"].to_list()[0] == pytest.approx(1.0)
    assert result["status"].to_list() == [FactorValidationStatus.PASS.value]


def test_perfectly_decreasing_quantiles_yield_monotonicity_one() -> None:
    """Non-increasing Q1–Q5 mean returns yield monotonicity_score = 1.0."""
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_quintile_panel(
            panels=[
                (
                    _open_time(0),
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                    [5.0, 4.0, 3.0, 2.0, 1.0],
                ),
            ]
        ),
    )
    assert result["monotonicity_score"].to_list()[0] == pytest.approx(1.0)


def test_non_monotonic_quantiles_yield_monotonicity_zero() -> None:
    """A non-monotonic quantile return pattern yields monotonicity_score = 0.0."""
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_quintile_panel(
            panels=[
                (
                    _open_time(0),
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                    [1.0, 5.0, 2.0, 4.0, 3.0],
                ),
            ]
        ),
    )
    assert result["monotonicity_score"].to_list()[0] == pytest.approx(0.0)


def test_monotonicity_averages_across_timestamps() -> None:
    """monotonicity_score averages the chronological monotonicity indicators."""
    # t0 monotonic (1), t1 non-monotonic (0) => mean = 0.5
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_quintile_panel(
            panels=[
                (
                    _open_time(0),
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                ),
                (
                    _open_time(1),
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                    [1.0, 5.0, 2.0, 4.0, 3.0],
                ),
            ]
        ),
    )
    assert result["monotonicity_score"].to_list()[0] == pytest.approx(0.5)


def test_insufficient_assets_yield_null_monotonicity() -> None:
    """Fewer than five valid assets yields null monotonicity_score."""
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_cross_section_panel(
            panels=[
                (_open_time(0), [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)]),
            ]
        ),
    )
    assert result["monotonicity_score"].to_list() == [None]
    assert result["status"].to_list() == [FactorValidationStatus.PASS.value]


def test_null_values_are_excluded_from_monotonicity() -> None:
    """Null pairs are ignored before monotonicity quantile construction."""
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_cross_section_panel(
            panels=[
                (
                    _open_time(0),
                    [
                        (1.0, 1.0),
                        (2.0, 2.0),
                        (None, 9.0),
                        (3.0, 3.0),
                        (4.0, 4.0),
                        (5.0, 5.0),
                    ],
                ),
            ]
        ),
    )
    assert result["monotonicity_score"].to_list()[0] == pytest.approx(1.0)


def test_fail_status_keeps_null_monotonicity() -> None:
    """FAIL rows with insufficient observations emit null monotonicity_score."""
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_factors_frame(
            open_times=[_open_time(0)],
            factor_values=[1.0],
            future_returns=[2.0],
        ),
    )
    assert result["status"].to_list() == [FactorValidationStatus.FAIL.value]
    assert result["monotonicity_score"].to_list() == [None]


# ---------------------------------------------------------------------------
# Factor turnover
# ---------------------------------------------------------------------------


def test_identical_rankings_yield_zero_turnover() -> None:
    """Identical Top-20% membership across timestamps yields turnover = 0."""
    factors = [1.0, 2.0, 3.0, 4.0, 5.0]
    returns = [0.0, 0.0, 0.0, 0.0, 0.0]
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_quintile_panel(
            panels=[
                (_open_time(0), factors, returns),
                (_open_time(1), factors, returns),
            ]
        ),
    )
    assert result["turnover"].to_list()[0] == pytest.approx(0.0)
    assert result["status"].to_list() == [FactorValidationStatus.PASS.value]


def test_complete_replacement_yields_unit_turnover() -> None:
    """Disjoint Top-20% portfolios yield turnover = 1.0."""
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_quintile_panel(
            panels=[
                (
                    _open_time(0),
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ),
                (
                    _open_time(1),
                    [5.0, 4.0, 3.0, 2.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ),
            ]
        ),
    )
    # t0 top = S04 (factor 5); t1 top = S00 (factor 5) => no overlap
    assert result["turnover"].to_list()[0] == pytest.approx(1.0)


def test_partial_overlap_turnover() -> None:
    """Partial Top-20% overlap yields turnover between 0 and 1."""
    # 10 assets => top_k = 2
    # t0 ranks high: S08,S09 ; t1 keeps S09 and replaces S08 with S07
    # overlap = 1/2 => turnover = 0.5
    factors_t0 = [float(index) for index in range(10)]
    factors_t1 = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0, 7.0, 8.0]
    zeros = [0.0] * 10
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_quintile_panel(
            panels=[
                (_open_time(0), factors_t0, zeros),
                (_open_time(1), factors_t1, zeros),
            ]
        ),
    )
    assert result["turnover"].to_list()[0] == pytest.approx(0.5)


def test_missing_symbols_use_common_universe_only() -> None:
    """Turnover uses only symbols present in both consecutive timestamps."""
    # t0 has S00-S05; t1 drops S05 and keeps S00-S04 with same relative ranks
    # common = 5 assets, identical top => turnover 0
    t0 = _cross_section_panel(
        panels=[
            (
                _open_time(0),
                [
                    (1.0, 0.0),
                    (2.0, 0.0),
                    (3.0, 0.0),
                    (4.0, 0.0),
                    (5.0, 0.0),
                    (6.0, 0.0),
                ],
            ),
        ]
    )
    t1 = _cross_section_panel(
        panels=[
            (
                _open_time(1),
                [
                    (1.0, 0.0),
                    (2.0, 0.0),
                    (3.0, 0.0),
                    (4.0, 0.0),
                    (5.0, 0.0),
                ],
            ),
        ]
    )
    result = _build(SimpleFactorValidationEngine(), factors=pl.concat([t0, t1]))
    assert result["turnover"].to_list()[0] == pytest.approx(0.0)


def test_single_timestamp_yields_null_turnover() -> None:
    """Fewer than two timestamps yields null turnover."""
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_quintile_panel(
            panels=[
                (
                    _open_time(0),
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ),
            ]
        ),
    )
    assert result["turnover"].to_list() == [None]
    assert result["status"].to_list() == [FactorValidationStatus.PASS.value]


def test_insufficient_common_assets_skip_turnover_transition() -> None:
    """Transitions with fewer than five common assets are skipped."""
    # Only 4 overlapping symbols between timestamps
    t0 = _cross_section_panel(
        panels=[
            (
                _open_time(0),
                [(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0), (5.0, 0.0)],
            ),
        ]
    )
    t1 = _cross_section_panel(
        panels=[
            (
                _open_time(1),
                [(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)],
            ),
        ]
    )
    result = _build(SimpleFactorValidationEngine(), factors=pl.concat([t0, t1]))
    assert result["turnover"].to_list() == [None]


def test_fail_status_keeps_null_turnover() -> None:
    """FAIL rows with insufficient observations emit null turnover."""
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_factors_frame(
            open_times=[_open_time(0)],
            factor_values=[1.0],
            future_returns=[2.0],
        ),
    )
    assert result["status"].to_list() == [FactorValidationStatus.FAIL.value]
    assert result["turnover"].to_list() == [None]


# ---------------------------------------------------------------------------
# Output schema, invariants, and immutability
# ---------------------------------------------------------------------------


def test_output_canonical_ordering_and_dtype_schema() -> None:
    """Engine output enforces canonical column order and schema dtypes."""
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_cross_section_panel(
            panels=[
                (_open_time(0), [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]),
                (_open_time(1), [(1.0, 1.0), (2.0, 0.0), (3.0, 1.0)]),
            ]
        ),
    )
    assert len(result.columns) == 22
    assert len(FACTOR_VALIDATION_SCHEMA) == 22
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result.schema == FACTOR_VALIDATION_SCHEMA
    assert "monotonicity_score" in result.columns
    assert "ic_std" in result.columns
    assert "ic_observations" in result.columns
    assert "dataset_version" in result.columns
    assert "label_version" in result.columns
    assert "validation_start_time" in result.columns
    assert "validation_end_time" in result.columns
    assert "observations" in result.columns
    assert "status" in result.columns
    assert result.schema["validation_time"] == pl.Int64
    assert result.schema["dataset_version"] == pl.String
    assert result.schema["label_version"] == pl.String
    assert result.schema["validation_start_time"] == pl.Int64
    assert result.schema["validation_end_time"] == pl.Int64
    assert result.schema["information_coefficient"] == pl.Float64
    assert result.schema["rank_information_coefficient"] == pl.Float64
    assert result.schema["ic_information_ratio"] == pl.Float64
    assert result.schema["ic_std"] == pl.Float64
    assert result.schema["ic_t_stat"] == pl.Float64
    assert result.schema["ic_p_value"] == pl.Float64
    assert result.schema["ic_decay"] == pl.Float64
    assert result.schema["quantile_spread"] == pl.Float64
    assert result.schema["monotonicity_score"] == pl.Float64
    assert result.schema["turnover"] == pl.Float64
    assert result.schema["observations"] == pl.Int64
    assert result.schema["ic_observations"] == pl.Int64
    assert result.schema["status"] == pl.String


def test_inputs_are_immutable() -> None:
    """build must not mutate the caller-supplied Factors frame."""
    factors = _cross_section_panel(
        panels=[
            (_open_time(0), [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]),
            (_open_time(1), [(1.0, 1.0), (2.0, 0.0), (3.0, 1.0)]),
        ]
    )
    before = factors.clone()
    SimpleFactorValidationEngine().build(factors)
    assert_frame_equal(factors, before)


def test_output_is_deterministic() -> None:
    """Identical Factors inputs produce identical validation outputs."""
    factors = _cross_section_panel(
        panels=[
            (_open_time(0), [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]),
            (_open_time(1), [(1.0, 3.0), (2.0, 2.0), (3.0, 1.0)]),
            (_open_time(2), [(1.0, 1.0), (2.0, 0.0), (3.0, 1.0)]),
        ]
    )
    engine = SimpleFactorValidationEngine()
    first = engine.build(factors)
    second = engine.build(factors)
    assert_frame_equal(first, second)


def test_integer_open_time_is_preserved_as_validation_time() -> None:
    """Integer epoch open_time values are emitted directly as validation_time."""
    result = _build(
        SimpleFactorValidationEngine(),
        factors=_factors_frame(
            open_times=[1_700_000_000_000, 1_700_000_003_600_000],
            factor_values=[1.0, 2.0],
            future_returns=[1.0, 2.0],
        ),
    )
    assert result["validation_time"].to_list() == [1_700_000_003_600_000]
