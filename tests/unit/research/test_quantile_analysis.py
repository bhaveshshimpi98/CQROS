"""Unit tests for CQROS cross-sectional quantile analysis."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass
from math import isfinite, isnan

import polars as pl
import pytest

from cqros.core.exceptions import ResearchError
from cqros.research.quantile_analysis import (
    QuantileAnalysisResult,
    QuantileAnalyzer,
    QuantileStatistics,
)


def _analyzer(quantiles: int = 5) -> QuantileAnalyzer:
    """Build a quantile analyzer."""
    return QuantileAnalyzer(quantiles=quantiles)


# --- construction / metadata ---


def test_default_quantiles_is_five() -> None:
    """Constructor defaults to five quantiles."""
    assert QuantileAnalyzer().quantiles == 5


@pytest.mark.parametrize("quantiles", [0, 1, -1, True, False, 1.5, "5", None])
def test_invalid_quantiles_raises(quantiles: object) -> None:
    """quantiles must be an integer greater than or equal to 2."""
    with pytest.raises(ResearchError, match="quantiles must be an integer") as exc_info:
        QuantileAnalyzer(quantiles=quantiles)  # type: ignore[arg-type]
    assert exc_info.value.error_code == "RESEARCH-QA-001"


def test_result_types_are_frozen() -> None:
    """Result dataclasses are immutable."""
    frame = pl.DataFrame(
        {
            "factor": list(range(10)),
            "target": [float(value) for value in range(10)],
        }
    )
    result = _analyzer(5).analyze(frame, "factor", "target")
    assert is_dataclass(result)
    assert isinstance(result, QuantileAnalysisResult)
    assert isinstance(result.statistics[0], QuantileStatistics)
    with pytest.raises(FrozenInstanceError):
        result.top_minus_bottom = 0.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.statistics[0].mean_return = 0.0  # type: ignore[misc]


# --- 5 and 10 quantiles ---


def test_five_quantiles_equal_frequency() -> None:
    """Ten observations into five quantiles yield two rows per bucket."""
    frame = pl.DataFrame(
        {
            "factor": list(range(10)),
            "target": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        }
    )
    result = _analyzer(5).analyze(frame, "factor", "target")
    assert result.quantiles == 5
    assert len(result.statistics) == 5
    assert tuple(item.quantile for item in result.statistics) == (1, 2, 3, 4, 5)
    assert all(item.count == 2 for item in result.statistics)
    assert result.statistics[0].mean_return == pytest.approx(0.5)
    assert result.statistics[-1].mean_return == pytest.approx(8.5)


def test_ten_quantiles_one_observation_each() -> None:
    """Ten observations into ten quantiles yield one row per bucket."""
    frame = pl.DataFrame(
        {
            "factor": list(range(10)),
            "target": [float(value) for value in range(10)],
        }
    )
    result = _analyzer(10).analyze(frame, "factor", "target")
    assert result.quantiles == 10
    assert len(result.statistics) == 10
    assert all(item.count == 1 for item in result.statistics)
    assert isnan(result.statistics[0].std_return)


# --- null handling / immutability ---


def test_nulls_are_dropped_before_analysis() -> None:
    """Rows with null factor or target values are excluded."""
    frame = pl.DataFrame(
        {
            "factor": [0.0, None, 2.0, 3.0, 4.0, None, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            "target": [0.0, 1.0, None, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        }
    )
    result = _analyzer(5).analyze(frame, "factor", "target")
    assert sum(item.count for item in result.statistics) == 9


def test_input_frame_is_not_mutated() -> None:
    """analyze never mutates the caller-supplied DataFrame."""
    frame = pl.DataFrame(
        {
            "factor": list(range(10)),
            "target": [float(value) for value in range(10)],
        }
    )
    original = frame.clone()
    _ = _analyzer(5).analyze(frame, "factor", "target")
    assert frame.equals(original)


# --- top-minus-bottom / monotonic ---


def test_top_minus_bottom_for_monotonic_factor() -> None:
    """Top-minus-bottom equals highest-quantile mean minus lowest-quantile mean."""
    frame = pl.DataFrame(
        {
            "factor": list(range(10)),
            "target": [float(value) for value in range(10)],
        }
    )
    result = _analyzer(5).analyze(frame, "factor", "target")
    expected = result.statistics[-1].mean_return - result.statistics[0].mean_return
    assert result.top_minus_bottom == pytest.approx(expected)
    assert result.top_minus_bottom == pytest.approx(8.0)
    assert result.monotonic is True


def test_monotonic_factor_is_detected() -> None:
    """Increasing factor-return alignment marks the result monotonic."""
    frame = pl.DataFrame(
        {
            "factor": list(range(20)),
            "target": [float(value) for value in range(20)],
        }
    )
    result = _analyzer(5).analyze(frame, "factor", "target")
    means = [item.mean_return for item in result.statistics]
    assert means == sorted(means)
    assert result.monotonic is True


def test_non_monotonic_factor_is_detected() -> None:
    """Inverted factor-return alignment marks the result non-monotonic."""
    frame = pl.DataFrame(
        {
            "factor": list(range(10)),
            "target": [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
        }
    )
    result = _analyzer(5).analyze(frame, "factor", "target")
    means = [item.mean_return for item in result.statistics]
    assert means == sorted(means, reverse=True)
    assert result.monotonic is False
    assert result.top_minus_bottom < 0.0


def test_statistics_include_requested_aggregates() -> None:
    """Each quantile reports count, mean, median, std, min, and max."""
    frame = pl.DataFrame(
        {
            "factor": [1.0, 2.0, 3.0, 4.0],
            "target": [10.0, 20.0, 30.0, 40.0],
        }
    )
    result = _analyzer(2).analyze(frame, "factor", "target")
    bottom = result.statistics[0]
    top = result.statistics[1]
    assert bottom.count == 2
    assert bottom.mean_return == pytest.approx(15.0)
    assert bottom.median_return == pytest.approx(15.0)
    assert isfinite(bottom.std_return)
    assert bottom.min_return == pytest.approx(10.0)
    assert bottom.max_return == pytest.approx(20.0)
    assert top.mean_return == pytest.approx(35.0)
    assert result.top_minus_bottom == pytest.approx(20.0)


# --- validation ---


def test_missing_factor_column_raises() -> None:
    """Missing factor column raises ResearchError."""
    frame = pl.DataFrame({"target": [1.0, 2.0, 3.0, 4.0, 5.0]})
    with pytest.raises(ResearchError, match="required column missing: factor") as exc_info:
        _analyzer(5).analyze(frame, "factor", "target")
    assert exc_info.value.error_code == "RESEARCH-QA-002"


def test_missing_target_column_raises() -> None:
    """Missing target column raises ResearchError."""
    frame = pl.DataFrame({"factor": [1.0, 2.0, 3.0, 4.0, 5.0]})
    with pytest.raises(ResearchError, match="required column missing: target") as exc_info:
        _analyzer(5).analyze(frame, "factor", "target")
    assert exc_info.value.error_code == "RESEARCH-QA-003"


def test_insufficient_observations_raises() -> None:
    """Fewer observations than quantiles raises ResearchError."""
    frame = pl.DataFrame(
        {
            "factor": [1.0, 2.0, 3.0],
            "target": [0.1, 0.2, 0.3],
        }
    )
    with pytest.raises(ResearchError, match="insufficient observations") as exc_info:
        _analyzer(5).analyze(frame, "factor", "target")
    assert exc_info.value.error_code == "RESEARCH-QA-004"
    assert exc_info.value.details["observations"] == 3
    assert exc_info.value.details["minimum_observations"] == 5


def test_insufficient_observations_after_null_drop() -> None:
    """Null dropping can reduce the sample below the quantile requirement."""
    frame = pl.DataFrame(
        {
            "factor": [1.0, None, 3.0, None, 5.0],
            "target": [0.1, 0.2, None, 0.4, 0.5],
        }
    )
    with pytest.raises(ResearchError, match="insufficient observations"):
        _analyzer(5).analyze(frame, "factor", "target")


def test_package_exports_quantile_analysis() -> None:
    """Quantile analysis symbols are exported from the research package."""
    import cqros.research as research_package

    assert "QuantileAnalyzer" in research_package.__all__
    assert "QuantileAnalysisResult" in research_package.__all__
    assert "QuantileStatistics" in research_package.__all__
    assert research_package.QuantileAnalyzer is QuantileAnalyzer
