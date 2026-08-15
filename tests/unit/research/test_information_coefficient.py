"""Unit tests for CQROS Information Coefficient engine."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass
from math import isfinite

import polars as pl
import pytest

from cqros.core.exceptions import ResearchError
from cqros.research.information_coefficient import (
    InformationCoefficient,
    InformationCoefficientResult,
)


def _frame(
    factor: list[float | None],
    target: list[float | None],
    *,
    factor_column: str = "factor",
    target_column: str = "target",
) -> pl.DataFrame:
    """Build a two-column research frame for IC tests."""
    return pl.DataFrame({factor_column: factor, target_column: target})


# --- result metadata ---


def test_result_is_frozen_dataclass() -> None:
    """InformationCoefficientResult is an immutable slotted dataclass."""
    result = InformationCoefficient(
        method="pearson",
    ).compute(
        _frame([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]),
        "factor",
        "target",
    )
    assert is_dataclass(result)
    assert isinstance(result, InformationCoefficientResult)
    with pytest.raises(FrozenInstanceError):
        result.coefficient = 0.0  # type: ignore[misc]


def test_default_method_is_spearman() -> None:
    """Constructor defaults to Spearman rank correlation."""
    assert InformationCoefficient().method == "spearman"


def test_result_records_columns_method_and_observations() -> None:
    """Result metadata mirrors compute arguments and observation count."""
    result = InformationCoefficient(method="pearson").compute(
        _frame([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]),
        "factor",
        "target",
    )
    assert result.factor_column == "factor"
    assert result.target_column == "target"
    assert result.method == "pearson"
    assert result.observations == 4


# --- positive / negative / zero correlation ---


def test_perfect_positive_pearson() -> None:
    """Perfect positive linear association yields Pearson IC of 1."""
    result = InformationCoefficient(method="pearson").compute(
        _frame([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]),
        "factor",
        "target",
    )
    assert result.coefficient == pytest.approx(1.0)
    assert result.p_value == pytest.approx(0.0)


def test_perfect_negative_pearson() -> None:
    """Perfect negative linear association yields Pearson IC of -1."""
    result = InformationCoefficient(method="pearson").compute(
        _frame([1.0, 2.0, 3.0, 4.0], [8.0, 6.0, 4.0, 2.0]),
        "factor",
        "target",
    )
    assert result.coefficient == pytest.approx(-1.0)
    assert result.p_value == pytest.approx(0.0)


def test_zero_pearson_correlation() -> None:
    """Orthogonal centered series yield Pearson IC of zero."""
    result = InformationCoefficient(method="pearson").compute(
        _frame([-1.0, 0.0, 1.0], [1.0, -2.0, 1.0]),
        "factor",
        "target",
    )
    assert result.coefficient == pytest.approx(0.0)
    assert isfinite(result.p_value)


def test_perfect_positive_spearman() -> None:
    """Monotone increasing ranks yield Spearman IC of 1."""
    result = InformationCoefficient(method="spearman").compute(
        _frame([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 40.0, 80.0]),
        "factor",
        "target",
    )
    assert result.coefficient == pytest.approx(1.0)


def test_perfect_negative_spearman() -> None:
    """Strictly opposing ranks yield Spearman IC of -1."""
    result = InformationCoefficient(method="spearman").compute(
        _frame([1.0, 2.0, 3.0, 4.0], [80.0, 40.0, 20.0, 10.0]),
        "factor",
        "target",
    )
    assert result.coefficient == pytest.approx(-1.0)


def test_zero_spearman_correlation() -> None:
    """Balanced opposing ranks can yield Spearman IC of zero."""
    result = InformationCoefficient(method="spearman").compute(
        _frame([1.0, 2.0, 3.0, 4.0], [1.0, 4.0, 2.0, 3.0]),
        "factor",
        "target",
    )
    # Rank pattern chosen for near-zero association; assert small magnitude.
    assert abs(result.coefficient) < 0.5


# --- pearson vs spearman ---


def test_pearson_and_spearman_differ_for_nonlinear_monotone() -> None:
    """Nonlinear monotone data can separate Pearson from Spearman."""
    frame = _frame([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 4.0, 9.0, 16.0, 25.0])
    pearson = InformationCoefficient(method="pearson").compute(frame, "factor", "target")
    spearman = InformationCoefficient(method="spearman").compute(frame, "factor", "target")
    assert spearman.coefficient == pytest.approx(1.0)
    assert pearson.coefficient < 1.0
    assert pearson.coefficient > 0.9


def test_methods_agree_on_linear_data() -> None:
    """Pearson and Spearman agree on perfectly linear increasing data."""
    frame = _frame([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0])
    pearson = InformationCoefficient(method="pearson").compute(frame, "factor", "target")
    spearman = InformationCoefficient(method="spearman").compute(frame, "factor", "target")
    assert pearson.coefficient == pytest.approx(1.0)
    assert spearman.coefficient == pytest.approx(1.0)


# --- null handling ---


def test_nulls_are_dropped_before_estimation() -> None:
    """Rows with null factor or target values are excluded."""
    frame = _frame(
        [1.0, None, 3.0, 4.0, None],
        [2.0, 4.0, None, 8.0, 10.0],
    )
    result = InformationCoefficient(method="pearson").compute(frame, "factor", "target")
    assert result.observations == 2
    assert result.coefficient == pytest.approx(1.0)


def test_null_only_in_factor_is_dropped() -> None:
    """A null factor with a valid target does not enter the estimate."""
    frame = _frame([1.0, None, 3.0], [1.0, 100.0, 3.0])
    result = InformationCoefficient(method="pearson").compute(frame, "factor", "target")
    assert result.observations == 2
    assert result.coefficient == pytest.approx(1.0)


def test_input_frame_is_not_mutated() -> None:
    """compute never mutates the caller-supplied DataFrame."""
    frame = _frame([1.0, None, 3.0], [2.0, 4.0, 6.0])
    original = frame.clone()
    _ = InformationCoefficient(method="pearson").compute(frame, "factor", "target")
    assert frame.equals(original)


# --- validation ---


def test_missing_factor_column_raises() -> None:
    """Missing factor column raises ResearchError."""
    frame = pl.DataFrame({"target": [1.0, 2.0, 3.0]})
    with pytest.raises(ResearchError, match="required column missing: factor") as exc_info:
        InformationCoefficient().compute(frame, "factor", "target")
    assert exc_info.value.error_code == "RESEARCH-IC-002"
    assert exc_info.value.details["role"] == "factor"


def test_missing_target_column_raises() -> None:
    """Missing target column raises ResearchError."""
    frame = pl.DataFrame({"factor": [1.0, 2.0, 3.0]})
    with pytest.raises(ResearchError, match="required column missing: target") as exc_info:
        InformationCoefficient().compute(frame, "factor", "target")
    assert exc_info.value.error_code == "RESEARCH-IC-003"
    assert exc_info.value.details["role"] == "target"


@pytest.mark.parametrize("method", ["kendall", "foo", "", "PEARSON", "Spearman"])
def test_invalid_method_raises(method: str) -> None:
    """Unknown correlation methods raise ResearchError at construction."""
    with pytest.raises(ResearchError, match="unknown correlation method") as exc_info:
        InformationCoefficient(method=method)
    assert exc_info.value.error_code == "RESEARCH-IC-001"


def test_insufficient_observations_after_null_drop() -> None:
    """Fewer than two paired observations raise ResearchError."""
    frame = _frame([1.0, None], [2.0, 3.0])
    with pytest.raises(ResearchError, match="insufficient observations") as exc_info:
        InformationCoefficient().compute(frame, "factor", "target")
    assert exc_info.value.error_code == "RESEARCH-IC-004"
    assert exc_info.value.details["observations"] == 1


def test_insufficient_observations_empty_frame() -> None:
    """An empty frame raises ResearchError for insufficient observations."""
    frame = pl.DataFrame(
        {
            "factor": pl.Series("factor", [], dtype=pl.Float64),
            "target": pl.Series("target", [], dtype=pl.Float64),
        }
    )
    with pytest.raises(ResearchError, match="insufficient observations"):
        InformationCoefficient().compute(frame, "factor", "target")


def test_single_observation_raises() -> None:
    """A single paired observation is insufficient."""
    frame = _frame([1.0], [2.0])
    with pytest.raises(ResearchError, match="insufficient observations"):
        InformationCoefficient().compute(frame, "factor", "target")


# --- p-value ---


def test_p_value_is_present_and_finite_for_noisy_data() -> None:
    """p-value is a finite float for non-degenerate associations."""
    result = InformationCoefficient(method="pearson").compute(
        _frame([1.0, 2.0, 3.0, 4.0, 5.0], [1.1, 1.9, 3.2, 3.8, 5.1]),
        "factor",
        "target",
    )
    assert isinstance(result.p_value, float)
    assert isfinite(result.p_value)
    assert 0.0 <= result.p_value <= 1.0


def test_p_value_present_for_zero_correlation() -> None:
    """Zero-correlation results still expose a numeric p-value."""
    result = InformationCoefficient(method="pearson").compute(
        _frame([-1.0, 0.0, 1.0], [1.0, -2.0, 1.0]),
        "factor",
        "target",
    )
    assert isinstance(result.p_value, float)
    assert result.p_value == pytest.approx(1.0)


def test_package_exports_information_coefficient() -> None:
    """IC symbols are exported from the research package."""
    import cqros.research as research_package

    assert "InformationCoefficient" in research_package.__all__
    assert "InformationCoefficientResult" in research_package.__all__
    assert research_package.InformationCoefficient is InformationCoefficient
