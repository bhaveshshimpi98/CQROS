"""Unit tests for CQROS rolling factor stability analysis."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass
from math import isfinite

import polars as pl
import pytest

from cqros.core.exceptions import ResearchError
from cqros.research.factor_stability import (
    FactorStabilityAnalyzer,
    FactorStabilityResult,
    StabilityWindow,
)
from cqros.research.information_coefficient import InformationCoefficient


def _analyzer(method: str = "spearman") -> FactorStabilityAnalyzer:
    """Build a factor stability analyzer."""
    return FactorStabilityAnalyzer(method=method)


def _aligned_frame(rows: int) -> pl.DataFrame:
    """Build a frame where factor equals target (stable perfect IC)."""
    values = [float(index) for index in range(rows)]
    return pl.DataFrame({"factor": values, "target": values})


def _unstable_frame(window_size: int, windows: int = 4) -> pl.DataFrame:
    """Build a frame with opposing IC regimes across windows."""
    factors: list[float] = []
    targets: list[float] = []
    for window_index in range(windows):
        for offset in range(window_size):
            value = float(offset + 1)
            factors.append(value)
            # Alternate perfect positive and perfect negative association.
            targets.append(value if window_index % 2 == 0 else -value)
    return pl.DataFrame({"factor": factors, "target": targets})


# --- metadata ---


def test_default_method_is_spearman() -> None:
    """Constructor defaults to Spearman IC."""
    assert FactorStabilityAnalyzer().method == "spearman"


def test_result_types_are_frozen() -> None:
    """Result dataclasses are immutable."""
    result = _analyzer(method="pearson").analyze(
        _aligned_frame(20),
        "factor",
        "target",
        window_size=5,
    )
    assert is_dataclass(result)
    assert isinstance(result, FactorStabilityResult)
    assert isinstance(result.windows[0], StabilityWindow)
    with pytest.raises(FrozenInstanceError):
        result.stability_score = 0.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.windows[0].coefficient = 0.0  # type: ignore[misc]


# --- multiple windows / stable / unstable ---


def test_multiple_non_overlapping_windows() -> None:
    """Consecutive non-overlapping windows are indexed with correct bounds."""
    result = _analyzer(method="pearson").analyze(
        _aligned_frame(20),
        "factor",
        "target",
        window_size=5,
    )
    assert result.window_size == 5
    assert len(result.windows) == 4
    assert tuple(window.window_index for window in result.windows) == (0, 1, 2, 3)
    assert tuple(window.start_row for window in result.windows) == (0, 5, 10, 15)
    assert tuple(window.end_row for window in result.windows) == (5, 10, 15, 20)
    assert all(window.observations == 5 for window in result.windows)


def test_trailing_incomplete_window_is_ignored() -> None:
    """Rows that do not fill a complete window are excluded."""
    result = _analyzer(method="pearson").analyze(
        _aligned_frame(23),
        "factor",
        "target",
        window_size=5,
    )
    assert len(result.windows) == 4
    assert result.windows[-1].end_row == 20


def test_stable_factor_has_high_stability_score() -> None:
    """A consistently aligned factor yields high mean IC and stability."""
    result = _analyzer(method="pearson").analyze(
        _aligned_frame(40),
        "factor",
        "target",
        window_size=10,
    )
    assert all(window.coefficient == pytest.approx(1.0) for window in result.windows)
    assert result.mean_ic == pytest.approx(1.0)
    assert result.std_ic == pytest.approx(0.0)
    assert result.min_ic == pytest.approx(1.0)
    assert result.max_ic == pytest.approx(1.0)
    assert result.stability_score == pytest.approx(1.0)


def test_unstable_factor_has_lower_stability_score() -> None:
    """Regime-switching association reduces the stability score."""
    stable = _analyzer(method="pearson").analyze(
        _aligned_frame(40),
        "factor",
        "target",
        window_size=10,
    )
    unstable = _analyzer(method="pearson").analyze(
        _unstable_frame(window_size=10, windows=4),
        "factor",
        "target",
        window_size=10,
    )
    assert unstable.std_ic > stable.std_ic
    assert unstable.stability_score < stable.stability_score
    assert unstable.min_ic < 0.0
    assert unstable.max_ic > 0.0


def test_summary_statistics_match_window_coefficients() -> None:
    """Summary IC statistics are derived from the per-window coefficients."""
    result = _analyzer(method="pearson").analyze(
        _unstable_frame(window_size=8, windows=4),
        "factor",
        "target",
        window_size=8,
    )
    coefficients = [window.coefficient for window in result.windows]
    assert result.mean_ic == pytest.approx(sum(coefficients) / len(coefficients))
    assert result.min_ic == pytest.approx(min(coefficients))
    assert result.max_ic == pytest.approx(max(coefficients))
    assert isfinite(result.std_ic)
    assert 0.0 <= result.stability_score <= 1.0


def test_stability_score_is_zero_when_mean_ic_is_zero() -> None:
    """A zero mean IC forces stability_score to zero."""
    # Two windows with opposing perfect IC cancel in the mean.
    frame = _unstable_frame(window_size=6, windows=2)
    result = _analyzer(method="pearson").analyze(
        frame,
        "factor",
        "target",
        window_size=6,
    )
    assert result.mean_ic == pytest.approx(0.0)
    assert result.stability_score == 0.0


# --- null handling / immutability ---


def test_nulls_inside_windows_are_dropped_by_ic() -> None:
    """Null pairs reduce observations but still allow window IC estimation."""
    values = [float(index) for index in range(20)]
    factor: list[float | None] = list(values)
    target = list(values)
    for index in (1, 7, 12):
        factor[index] = None
    frame = pl.DataFrame({"factor": factor, "target": target})
    result = _analyzer(method="pearson").analyze(
        frame,
        "factor",
        "target",
        window_size=5,
    )
    assert len(result.windows) == 4
    assert result.windows[0].observations == 4
    assert result.windows[1].observations == 4
    assert result.windows[2].observations == 4
    assert result.windows[3].observations == 5


def test_input_frame_is_not_mutated() -> None:
    """analyze never mutates the caller-supplied DataFrame."""
    frame = _aligned_frame(20)
    original = frame.clone()
    _ = _analyzer(method="pearson").analyze(frame, "factor", "target", window_size=5)
    assert frame.equals(original)


# --- validation ---


@pytest.mark.parametrize("window_size", [0, 1, -1, True, 1.5, "5", None])
def test_invalid_window_size_raises(window_size: object) -> None:
    """window_size must be an integer greater than or equal to 2."""
    with pytest.raises(ResearchError, match="window_size must be an integer") as exc_info:
        _analyzer().analyze(
            _aligned_frame(20),
            "factor",
            "target",
            window_size=window_size,  # type: ignore[arg-type]
        )
    assert exc_info.value.error_code == "RESEARCH-STABILITY-001"


def test_missing_factor_column_raises() -> None:
    """Missing factor column raises ResearchError."""
    frame = pl.DataFrame({"target": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(ResearchError, match="required column missing: factor") as exc_info:
        _analyzer().analyze(frame, "factor", "target", window_size=2)
    assert exc_info.value.error_code == "RESEARCH-STABILITY-002"


def test_missing_target_column_raises() -> None:
    """Missing target column raises ResearchError."""
    frame = pl.DataFrame({"factor": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(ResearchError, match="required column missing: target") as exc_info:
        _analyzer().analyze(frame, "factor", "target", window_size=2)
    assert exc_info.value.error_code == "RESEARCH-STABILITY-003"


def test_insufficient_observations_raises() -> None:
    """Fewer rows than window_size raises ResearchError."""
    with pytest.raises(ResearchError, match="insufficient observations") as exc_info:
        _analyzer().analyze(_aligned_frame(5), "factor", "target", window_size=10)
    assert exc_info.value.error_code == "RESEARCH-STABILITY-004"
    assert exc_info.value.details["observations"] == 5
    assert exc_info.value.details["minimum_observations"] == 10


def test_reuses_information_coefficient_method() -> None:
    """Configured method matches InformationCoefficient's supported surface."""
    analyzer = FactorStabilityAnalyzer(method="pearson")
    assert analyzer.method == InformationCoefficient(method="pearson").method


def test_package_exports_factor_stability() -> None:
    """Factor stability symbols are exported from the research package."""
    import cqros.research as research_package

    assert "FactorStabilityAnalyzer" in research_package.__all__
    assert "FactorStabilityResult" in research_package.__all__
    assert "StabilityWindow" in research_package.__all__
    assert research_package.FactorStabilityAnalyzer is FactorStabilityAnalyzer
