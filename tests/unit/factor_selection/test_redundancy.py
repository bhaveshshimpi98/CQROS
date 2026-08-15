"""Unit tests for CQROS Factor Selection redundancy filtering."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from cqros.factor_selection import (
    FactorSelectionError,
    SimpleFactorSelectionEngine,
)
from cqros.factor_selection.redundancy import (
    DEFAULT_CANDIDATE_N,
    DEFAULT_MAX_FACTOR_CORRELATION,
    DEFAULT_MIN_CORRELATION_OVERLAP,
    REASON_OUTSIDE_CANDIDATE_N,
    REASON_OUTSIDE_TOP_N,
    REASON_REDUNDANT,
    REASON_TOP_N,
    RedundancyConfig,
    apply_greedy_redundancy_filter,
    require_redundancy_config,
)

_TIMEFRAME = "1h"
_VERSION = "1.0.0"
_CATEGORY = "price"
_VALIDATION_TIME = 1_700_000_000_000
_START = 1_699_000_000_000
_END = 1_700_000_000_000


class _StaticObservationSource:
    """In-memory observation source for deterministic redundancy tests."""

    def __init__(self, frame: pl.DataFrame) -> None:
        self._frame = frame

    def load_panel(
        self,
        *,
        timeframe: str,
        factor_names: Sequence[str],
        factor_versions: Sequence[str],
        start_time: int,
        end_time: int,
    ) -> pl.DataFrame:
        _ = timeframe
        return (
            self._frame.filter(pl.col("factor_name").is_in(list(factor_names)))
            .filter(pl.col("factor_version").is_in(list(factor_versions)))
            .filter(pl.col("open_time") >= start_time)
            .filter(pl.col("open_time") <= end_time)
        )


def _validation_frame(
    *,
    names: list[str],
    ics: list[float] | None = None,
) -> pl.DataFrame:
    row_count = len(names)
    ics = ics if ics is not None else [0.20 - (0.01 * index) for index in range(row_count)]
    return pl.DataFrame(
        {
            "factor_name": names,
            "factor_version": [_VERSION] * row_count,
            "timeframe": [_TIMEFRAME] * row_count,
            "validation_time": [_VALIDATION_TIME] * row_count,
            "factor_category": [_CATEGORY] * row_count,
            "dataset_version": ["default"] * row_count,
            "label_version": ["default"] * row_count,
            "validation_start_time": [_START] * row_count,
            "validation_end_time": [_END] * row_count,
            "information_coefficient": ics,
            "rank_information_coefficient": ics,
            "ic_information_ratio": [0.5] * row_count,
            "ic_p_value": [0.01] * row_count,
            "ic_decay": [0.5] * row_count,
            "turnover": [0.2] * row_count,
            "monotonicity_score": [0.5] * row_count,
            "quantile_spread": [0.05] * row_count,
            "observations": [200] * row_count,
            "status": ["PASS"] * row_count,
        }
    )


def _observations_for_clones(
    *,
    names: list[str],
    series: dict[str, list[float]],
    n_points: int = 600,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(n_points):
        open_time = _START + index
        symbol = f"S{index % 3}"
        for name in names:
            rows.append(
                {
                    "symbol": symbol,
                    "open_time": open_time,
                    "factor_name": name,
                    "factor_version": _VERSION,
                    "factor_value": series[name][index],
                }
            )
    return pl.DataFrame(rows)


def test_require_redundancy_config_rejects_candidate_lt_top_n() -> None:
    """candidate_n < top_n raises FactorSelectionError."""
    with pytest.raises(FactorSelectionError) as exc_info:
        require_redundancy_config(top_n=20, candidate_n=10)
    assert exc_info.value.error_code == "FSEL_CANDIDATE_N_LT_TOP_N"


def test_require_redundancy_config_rejects_invalid_correlation() -> None:
    """max_factor_correlation outside (0, 1) is rejected."""
    with pytest.raises(FactorSelectionError):
        require_redundancy_config(top_n=5, max_factor_correlation=1.0)
    with pytest.raises(FactorSelectionError):
        require_redundancy_config(top_n=5, max_factor_correlation=0.0)


def test_require_redundancy_config_rejects_invalid_min_overlap() -> None:
    """Non-positive min_overlap is rejected."""
    with pytest.raises(FactorSelectionError):
        require_redundancy_config(top_n=5, min_overlap=0)


def test_exact_duplicate_factors_are_redundant() -> None:
    """|rho|=1.0 clones are rejected with reason redundant."""
    names = ["alpha", "alpha_clone", "beta"]
    base = np.linspace(-1.0, 1.0, 600).tolist()
    series = {
        "alpha": base,
        "alpha_clone": base,
        "beta": [math.sin(index / 7.0) for index in range(600)],
    }
    source = _StaticObservationSource(_observations_for_clones(names=names, series=series))
    engine = SimpleFactorSelectionEngine(
        top_n=2,
        candidate_n=3,
        max_factor_correlation=0.90,
        min_overlap=500,
        observation_source=source,
    )
    result = engine.build(_validation_frame(names=names, ics=[0.30, 0.29, 0.10]))
    by_name = {row["factor_name"]: row for row in result.to_dicts()}
    assert by_name["alpha"]["selected"] is True
    assert by_name["alpha"]["selection_reason"] == REASON_TOP_N
    assert by_name["alpha_clone"]["selected"] is False
    assert by_name["alpha_clone"]["selection_reason"] == REASON_REDUNDANT
    assert by_name["beta"]["selected"] is True


def test_correlation_just_below_threshold_is_kept() -> None:
    """Correlation just below the threshold does not trigger redundancy."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=600)
    y = 0.5 * x + rng.normal(scale=1.2, size=600)
    corr = abs(float(np.corrcoef(x, y)[0, 1]))
    assert corr < 0.90
    names = ["a", "b"]
    source = _StaticObservationSource(
        _observations_for_clones(
            names=names,
            series={"a": [float(v) for v in x], "b": [float(v) for v in y]},
        )
    )
    engine = SimpleFactorSelectionEngine(
        top_n=2,
        candidate_n=2,
        max_factor_correlation=0.90,
        min_overlap=500,
        observation_source=source,
    )
    result = engine.build(_validation_frame(names=names, ics=[0.2, 0.1]))
    assert result.filter(pl.col("selected")).height == 2


def test_overlap_below_minimum_does_not_reject() -> None:
    """Pairs with overlap below min_overlap are treated as non-redundant."""
    names = ["a", "b"]
    # Only 100 overlapping complete rows after nulling most of b.
    values_b: list[float | None] = [float(index) if index < 100 else None for index in range(600)]
    rows: list[dict[str, object]] = []
    for index in range(600):
        rows.append(
            {
                "symbol": "S0",
                "open_time": _START + index,
                "factor_name": "a",
                "factor_version": _VERSION,
                "factor_value": float(index),
            }
        )
        rows.append(
            {
                "symbol": "S0",
                "open_time": _START + index,
                "factor_name": "b",
                "factor_version": _VERSION,
                "factor_value": values_b[index],
            }
        )
    source = _StaticObservationSource(pl.DataFrame(rows))
    engine = SimpleFactorSelectionEngine(
        top_n=2,
        candidate_n=2,
        max_factor_correlation=0.90,
        min_overlap=500,
        observation_source=source,
    )
    result = engine.build(_validation_frame(names=names, ics=[0.2, 0.1]))
    assert set(result.filter(pl.col("selected"))["factor_name"].to_list()) == {"a", "b"}


def test_constant_factor_undefined_pearson_is_not_rejected() -> None:
    """Constant series yielding undefined Pearson are not treated as rho=1."""
    names = ["const", "vary"]
    series = {
        "const": [1.0] * 600,
        "vary": [float(value) for value in np.linspace(-1.0, 1.0, 600)],
    }
    source = _StaticObservationSource(_observations_for_clones(names=names, series=series))
    engine = SimpleFactorSelectionEngine(
        top_n=2,
        candidate_n=2,
        max_factor_correlation=0.90,
        min_overlap=500,
        observation_source=source,
    )
    result = engine.build(_validation_frame(names=names, ics=[0.2, 0.1]))
    assert result.filter(pl.col("selected")).height == 2


def test_outside_candidate_n_reason() -> None:
    """Ranks beyond candidate_n receive outside_candidate_n."""
    names = [f"f{index}" for index in range(5)]
    engine = SimpleFactorSelectionEngine(top_n=2, candidate_n=3)
    result = engine.build(_validation_frame(names=names))
    outside = result.filter(pl.col("selection_rank") > 3)
    assert outside.height == 2
    assert set(outside["selection_reason"].to_list()) == {REASON_OUTSIDE_CANDIDATE_N}
    assert outside["selected"].to_list() == [False, False]


def test_validation_window_excludes_future_observations() -> None:
    """Observations after validation_end_time do not affect redundancy."""
    names = ["a", "b"]
    # Inside window: uncorrelated. Outside window: perfect clones.
    inside_a = [float(value) for value in np.linspace(-1.0, 1.0, 600)]
    inside_b = [float(value) for value in np.sin(np.linspace(0.0, 20.0, 600))]
    rows: list[dict[str, object]] = []
    for index in range(600):
        rows.append(
            {
                "symbol": "S0",
                "open_time": _START + index,
                "factor_name": "a",
                "factor_version": _VERSION,
                "factor_value": inside_a[index],
            }
        )
        rows.append(
            {
                "symbol": "S0",
                "open_time": _START + index,
                "factor_name": "b",
                "factor_version": _VERSION,
                "factor_value": inside_b[index],
            }
        )
    for index in range(600):
        value = float(index)
        rows.append(
            {
                "symbol": "S0",
                "open_time": _END + 1 + index,
                "factor_name": "a",
                "factor_version": _VERSION,
                "factor_value": value,
            }
        )
        rows.append(
            {
                "symbol": "S0",
                "open_time": _END + 1 + index,
                "factor_name": "b",
                "factor_version": _VERSION,
                "factor_value": value,
            }
        )
    source = _StaticObservationSource(pl.DataFrame(rows))
    engine = SimpleFactorSelectionEngine(
        top_n=2,
        candidate_n=2,
        max_factor_correlation=0.90,
        min_overlap=500,
        observation_source=source,
    )
    result = engine.build(_validation_frame(names=names, ics=[0.2, 0.1]))
    assert result.filter(pl.col("selection_reason") == REASON_REDUNDANT).height == 0


def test_defaults_match_phase3a_recommendation() -> None:
    """Defaults match Phase 3A recommended configuration."""
    assert DEFAULT_CANDIDATE_N == 40
    assert DEFAULT_MAX_FACTOR_CORRELATION == 0.90
    assert DEFAULT_MIN_CORRELATION_OVERLAP == 500


def test_fewer_than_candidate_n_uses_all_available() -> None:
    """When fewer than candidate_n factors exist, all ranks enter filtering."""
    names = ["a", "b", "c"]
    engine = SimpleFactorSelectionEngine(top_n=2, candidate_n=40)
    result = engine.build(_validation_frame(names=names))
    assert result.height == 3
    assert result.filter(pl.col("selection_reason") == REASON_OUTSIDE_CANDIDATE_N).height == 0
    assert result.filter(pl.col("selected")).height == 2


def test_fewer_than_top_n_selects_all_survivors() -> None:
    """When fewer than top_n factors exist, all survivors are selected."""
    names = ["a", "b"]
    engine = SimpleFactorSelectionEngine(top_n=20, candidate_n=40)
    result = engine.build(_validation_frame(names=names))
    assert result.filter(pl.col("selected")).height == 2
    assert set(result["selection_reason"].to_list()) == {REASON_TOP_N}


def test_candidate_n_equals_top_n() -> None:
    """candidate_n == top_n is valid and selects without outside_top_n room."""
    names = [f"f{index}" for index in range(5)]
    engine = SimpleFactorSelectionEngine(top_n=3, candidate_n=3)
    result = engine.build(_validation_frame(names=names))
    assert result.filter(pl.col("selected")).height == 3
    assert result.filter(pl.col("selection_reason") == REASON_OUTSIDE_CANDIDATE_N).height == 2
    assert result.filter(pl.col("selection_reason") == REASON_OUTSIDE_TOP_N).height == 0


def test_negative_correlation_absolute_value_rejects() -> None:
    """Absolute Pearson of -1.0 is treated as redundant."""
    names = ["a", "b", "c"]
    base = np.linspace(-1.0, 1.0, 600)
    series = {
        "a": [float(value) for value in base],
        "b": [float(value) for value in (-base)],
        "c": [math.sin(index / 5.0) for index in range(600)],
    }
    source = _StaticObservationSource(_observations_for_clones(names=names, series=series))
    engine = SimpleFactorSelectionEngine(
        top_n=2,
        candidate_n=3,
        max_factor_correlation=0.90,
        min_overlap=500,
        observation_source=source,
    )
    result = engine.build(_validation_frame(names=names, ics=[0.30, 0.29, 0.10]))
    by_name = {row["factor_name"]: row for row in result.to_dicts()}
    assert by_name["a"]["selected"] is True
    assert by_name["b"]["selection_reason"] == REASON_REDUNDANT
    assert by_name["c"]["selected"] is True


def test_correlation_exactly_at_threshold_is_redundant() -> None:
    """|rho| exactly equal to max_factor_correlation rejects as redundant."""
    n_points = 600
    x = np.linspace(-1.0, 1.0, n_points)
    # Construct y with Pearson correlation exactly 0.90 vs x.
    noise: npt.NDArray[np.floating] = np.random.default_rng(0).normal(size=n_points)
    # Orthogonalize noise against x then mix for target correlation.
    noise = noise - (np.dot(noise, x) / np.dot(x, x)) * x
    noise = noise / np.linalg.norm(noise)
    x_unit = x / np.linalg.norm(x)
    y: npt.NDArray[np.floating] = 0.90 * x_unit + math.sqrt(1.0 - 0.90**2) * noise
    y = y * np.linalg.norm(x)
    corr = abs(float(np.corrcoef(x, y)[0, 1]))
    assert abs(corr - 0.90) < 1e-3
    names = ["a", "b"]
    source = _StaticObservationSource(
        _observations_for_clones(
            names=names,
            series={"a": [float(v) for v in x], "b": [float(v) for v in y]},
        )
    )
    # Use the realized correlation as the threshold so the pair sits exactly
    # on the redundancy boundary (abs(rho) >= threshold).
    engine = SimpleFactorSelectionEngine(
        top_n=1,
        candidate_n=2,
        max_factor_correlation=corr,
        min_overlap=500,
        observation_source=source,
    )
    result = engine.build(_validation_frame(names=names, ics=[0.2, 0.1]))
    by_name = {row["factor_name"]: row for row in result.to_dicts()}
    assert by_name["a"]["selected"] is True
    assert by_name["b"]["selection_reason"] == REASON_REDUNDANT


def test_factor_version_identity_is_not_merged() -> None:
    """Different factor_version values are distinct identities."""
    rows: list[dict[str, object]] = []
    base = np.linspace(-1.0, 1.0, 600)
    for index in range(600):
        open_time = _START + index
        rows.append(
            {
                "symbol": "S0",
                "open_time": open_time,
                "factor_name": "alpha",
                "factor_version": "1.0.0",
                "factor_value": float(base[index]),
            }
        )
        rows.append(
            {
                "symbol": "S0",
                "open_time": open_time,
                "factor_name": "alpha",
                "factor_version": "2.0.0",
                "factor_value": float(base[index]),
            }
        )
    validation = pl.DataFrame(
        {
            "factor_name": ["alpha", "alpha"],
            "factor_version": ["1.0.0", "2.0.0"],
            "timeframe": [_TIMEFRAME, _TIMEFRAME],
            "validation_time": [_VALIDATION_TIME, _VALIDATION_TIME],
            "factor_category": [_CATEGORY, _CATEGORY],
            "dataset_version": ["default", "default"],
            "label_version": ["default", "default"],
            "validation_start_time": [_START, _START],
            "validation_end_time": [_END, _END],
            "information_coefficient": [0.30, 0.29],
            "rank_information_coefficient": [0.30, 0.29],
            "ic_information_ratio": [0.5, 0.5],
            "ic_p_value": [0.01, 0.01],
            "ic_decay": [0.5, 0.5],
            "turnover": [0.2, 0.2],
            "monotonicity_score": [0.5, 0.5],
            "quantile_spread": [0.05, 0.05],
            "observations": [200, 200],
            "status": ["PASS", "PASS"],
        }
    )
    source = _StaticObservationSource(pl.DataFrame(rows))
    engine = SimpleFactorSelectionEngine(
        top_n=1,
        candidate_n=2,
        observation_source=source,
    )
    result = engine.build(validation)
    assert result.height == 2
    selected = result.filter(pl.col("selected"))
    assert selected.height == 1
    assert selected["factor_version"].item() == "1.0.0"
    rejected = result.filter(pl.col("selection_reason") == REASON_REDUNDANT)
    assert rejected.height == 1
    assert rejected["factor_version"].item() == "2.0.0"


def test_candidate_correlated_with_multiple_accepted_uses_first_reference() -> None:
    """Redundancy reference is the first already-accepted qualifying factor."""
    names = ["keep_a", "keep_b", "clone"]
    base = np.linspace(-1.0, 1.0, 600)
    other = np.sin(np.linspace(0.0, 12.0, 600))
    series = {
        "keep_a": [float(value) for value in base],
        "keep_b": [float(value) for value in other],
        "clone": [float(value) for value in base],
    }
    source = _StaticObservationSource(_observations_for_clones(names=names, series=series))
    engine = SimpleFactorSelectionEngine(
        top_n=2,
        candidate_n=3,
        observation_source=source,
    )
    _, audit = engine.build_with_audit(_validation_frame(names=names, ics=[0.30, 0.20, 0.10]))
    clone = audit.filter(pl.col("factor_name") == "clone").to_dicts()[0]
    assert clone["selection_reason"] == REASON_REDUNDANT
    assert clone["redundancy_reference_factor"] == "keep_a"
    assert clone["redundancy_rejected"] is True


def test_outside_top_n_after_surviving_redundancy() -> None:
    """Survivors beyond final top_n receive outside_top_n."""
    names = ["a", "b", "c"]
    # Uncorrelated series so none are redundant.
    series = {
        "a": [float(value) for value in np.linspace(-1.0, 1.0, 600)],
        "b": [float(value) for value in np.sin(np.linspace(0.0, 20.0, 600))],
        "c": [float(value) for value in np.cos(np.linspace(0.0, 13.0, 600))],
    }
    source = _StaticObservationSource(_observations_for_clones(names=names, series=series))
    engine = SimpleFactorSelectionEngine(
        top_n=2,
        candidate_n=3,
        observation_source=source,
    )
    result = engine.build(_validation_frame(names=names, ics=[0.30, 0.20, 0.10]))
    by_name = {row["factor_name"]: row for row in result.to_dicts()}
    assert by_name["a"]["selection_reason"] == REASON_TOP_N
    assert by_name["b"]["selection_reason"] == REASON_TOP_N
    assert by_name["c"]["selection_reason"] == REASON_OUTSIDE_TOP_N
    assert by_name["c"]["selected"] is False


def test_missing_observations_do_not_force_redundancy() -> None:
    """Missing observation panels leave candidates non-redundant."""
    names = ["a", "b"]
    source = _StaticObservationSource(
        pl.DataFrame(
            schema={
                "symbol": pl.String,
                "open_time": pl.Int64,
                "factor_name": pl.String,
                "factor_version": pl.String,
                "factor_value": pl.Float64,
            }
        )
    )
    engine = SimpleFactorSelectionEngine(
        top_n=2,
        candidate_n=2,
        observation_source=source,
    )
    result = engine.build(_validation_frame(names=names, ics=[0.2, 0.1]))
    assert result.filter(pl.col("selected")).height == 2
    assert result.filter(pl.col("selection_reason") == REASON_REDUNDANT).height == 0


def test_top_n_variants_are_configurable() -> None:
    """top_n of 10/20/30 and candidate_n=40 remain configurable."""
    names = [f"f{index:02d}" for index in range(45)]
    for top_n in (10, 20, 30):
        engine = SimpleFactorSelectionEngine(top_n=top_n, candidate_n=40)
        result = engine.build(_validation_frame(names=names))
        assert result.filter(pl.col("selected")).height == top_n
        assert result.filter(pl.col("selection_reason") == REASON_OUTSIDE_CANDIDATE_N).height == 5
        assert engine.candidate_n == 40


def test_apply_greedy_filter_preserves_rank_order() -> None:
    """Greedy filter processes candidates strictly by selection_rank."""
    ranked = pl.DataFrame(
        {
            "factor_name": ["b", "a", "c"],
            "factor_version": [_VERSION] * 3,
            "selection_rank": [2, 1, 3],
            "selection_score": [0.2, 0.3, 0.1],
            "timeframe": [_TIMEFRAME] * 3,
        }
    )
    base = np.linspace(-1.0, 1.0, 600)
    observations = _observations_for_clones(
        names=["a", "b", "c"],
        series={
            "a": [float(value) for value in base],
            "b": [float(value) for value in base],
            "c": [float(value) for value in np.sin(np.linspace(0.0, 9.0, 600))],
        },
    )
    config = RedundancyConfig(
        top_n=2,
        candidate_n=3,
        max_factor_correlation=0.90,
        min_overlap=500,
    )
    result = apply_greedy_redundancy_filter(ranked, observations, config)
    by_name = {row["factor_name"]: row for row in result.to_dicts()}
    assert by_name["a"]["selected"] is True
    assert by_name["b"]["selection_reason"] == REASON_REDUNDANT
    assert by_name["c"]["selected"] is True
