"""Unit tests for CQROS Rank Information Coefficient evaluator."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass
from math import isfinite

import polars as pl
import pytest

from cqros.core.exceptions import ResearchError
from cqros.research.rank_ic import RankICResult, RankInformationCoefficient


def _frame(
    factor: list[float | None],
    target: list[float | None],
    *,
    factor_column: str = "factor",
    target_column: str = "target",
) -> pl.DataFrame:
    """Build a two-column research frame for Rank IC tests."""
    return pl.DataFrame({factor_column: factor, target_column: target})


def _rank_ic() -> RankInformationCoefficient:
    """Build a Rank IC calculator."""
    return RankInformationCoefficient()


# --- metadata / immutability ---


def test_result_is_frozen_dataclass() -> None:
    """RankICResult is an immutable slotted dataclass."""
    result = _rank_ic().compute(
        _frame([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]),
        "factor",
        "target",
    )
    assert is_dataclass(result)
    assert isinstance(result, RankICResult)
    with pytest.raises(FrozenInstanceError):
        result.coefficient = 0.0  # type: ignore[misc]


def test_result_records_columns_and_observations() -> None:
    """Result metadata mirrors compute arguments and observation count."""
    result = _rank_ic().compute(
        _frame([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]),
        "factor",
        "target",
    )
    assert result.factor_column == "factor"
    assert result.target_column == "target"
    assert result.observations == 4


def test_input_frame_is_not_mutated() -> None:
    """compute never mutates the caller-supplied DataFrame."""
    frame = _frame([1.0, None, 3.0], [2.0, 4.0, 6.0])
    original = frame.clone()
    _ = _rank_ic().compute(frame, "factor", "target")
    assert frame.equals(original)


# --- correlation ---


def test_perfect_positive_rank_correlation() -> None:
    """Monotone increasing ranks yield Rank IC of 1."""
    result = _rank_ic().compute(
        _frame([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 40.0, 80.0]),
        "factor",
        "target",
    )
    assert result.coefficient == pytest.approx(1.0)
    assert result.p_value == pytest.approx(0.0)


def test_perfect_negative_rank_correlation() -> None:
    """Strictly opposing ranks yield Rank IC of -1."""
    result = _rank_ic().compute(
        _frame([1.0, 2.0, 3.0, 4.0], [80.0, 40.0, 20.0, 10.0]),
        "factor",
        "target",
    )
    assert result.coefficient == pytest.approx(-1.0)
    assert result.p_value == pytest.approx(0.0)


def test_nonlinear_monotone_still_perfect_rank_ic() -> None:
    """Nonlinear but monotone association still yields Rank IC of 1."""
    result = _rank_ic().compute(
        _frame([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 4.0, 9.0, 16.0, 25.0]),
        "factor",
        "target",
    )
    assert result.coefficient == pytest.approx(1.0)


# --- null handling ---


def test_nulls_are_dropped_before_estimation() -> None:
    """Rows with null factor or target values are excluded."""
    frame = _frame(
        [1.0, None, 3.0, 4.0, None],
        [2.0, 4.0, None, 8.0, 10.0],
    )
    result = _rank_ic().compute(frame, "factor", "target")
    assert result.observations == 2
    assert result.coefficient == pytest.approx(1.0)


def test_null_only_in_target_is_dropped() -> None:
    """A null target with a valid factor does not enter the estimate."""
    frame = _frame([1.0, 2.0, 3.0], [1.0, None, 3.0])
    result = _rank_ic().compute(frame, "factor", "target")
    assert result.observations == 2
    assert result.coefficient == pytest.approx(1.0)


# --- validation ---


def test_missing_factor_column_raises() -> None:
    """Missing factor column raises ResearchError."""
    frame = pl.DataFrame({"target": [1.0, 2.0, 3.0]})
    with pytest.raises(ResearchError, match="required column missing: factor") as exc_info:
        _rank_ic().compute(frame, "factor", "target")
    assert exc_info.value.error_code == "RESEARCH-IC-002"
    assert exc_info.value.details["role"] == "factor"


def test_missing_target_column_raises() -> None:
    """Missing target column raises ResearchError."""
    frame = pl.DataFrame({"factor": [1.0, 2.0, 3.0]})
    with pytest.raises(ResearchError, match="required column missing: target") as exc_info:
        _rank_ic().compute(frame, "factor", "target")
    assert exc_info.value.error_code == "RESEARCH-IC-003"
    assert exc_info.value.details["role"] == "target"


def test_insufficient_observations_after_null_drop() -> None:
    """Fewer than two paired observations raise ResearchError."""
    frame = _frame([1.0, None], [2.0, 3.0])
    with pytest.raises(ResearchError, match="insufficient observations") as exc_info:
        _rank_ic().compute(frame, "factor", "target")
    assert exc_info.value.error_code == "RESEARCH-IC-004"
    assert exc_info.value.details["observations"] == 1


def test_single_observation_raises() -> None:
    """A single paired observation is insufficient."""
    with pytest.raises(ResearchError, match="insufficient observations"):
        _rank_ic().compute(_frame([1.0], [2.0]), "factor", "target")


# --- p-value ---


def test_p_value_is_present_and_finite() -> None:
    """p-value is a finite float for non-degenerate rank associations."""
    result = _rank_ic().compute(
        _frame([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 4.0, 3.0, 5.0]),
        "factor",
        "target",
    )
    assert isinstance(result.p_value, float)
    assert isfinite(result.p_value)
    assert 0.0 <= result.p_value <= 1.0


def test_p_value_near_zero_for_perfect_association() -> None:
    """Perfect rank association yields a near-zero two-sided p-value."""
    result = _rank_ic().compute(
        _frame([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]),
        "factor",
        "target",
    )
    assert result.p_value == pytest.approx(0.0)


def test_package_exports_rank_ic() -> None:
    """Rank IC symbols are exported from the research package."""
    import cqros.research as research_package

    assert "RankInformationCoefficient" in research_package.__all__
    assert "RankICResult" in research_package.__all__
    assert research_package.RankInformationCoefficient is RankInformationCoefficient
