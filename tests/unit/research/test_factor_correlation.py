"""Unit tests for CQROS cross-factor correlation analysis."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass

import polars as pl
import pytest

from cqros.core.exceptions import ResearchError
from cqros.research.factor_correlation import (
    FactorCorrelationAnalyzer,
    FactorCorrelationResult,
    find_highly_correlated,
)


def _analyzer(method: str = "spearman") -> FactorCorrelationAnalyzer:
    """Build a factor correlation analyzer."""
    return FactorCorrelationAnalyzer(method=method)


def _frame() -> pl.DataFrame:
    """Build a frame with independent, identical, and ranked factors."""
    return pl.DataFrame(
        {
            "alpha": [1.0, 2.0, 3.0, 4.0, 5.0],
            "beta": [2.0, 4.0, 6.0, 8.0, 10.0],  # perfect linear with alpha
            "gamma": [5.0, 4.0, 3.0, 2.0, 1.0],  # perfect inverse of alpha
            "delta": [1.0, 4.0, 9.0, 16.0, 25.0],  # monotone nonlinear in alpha
        }
    )


# --- metadata ---


def test_default_method_is_spearman() -> None:
    """Constructor defaults to Spearman correlation."""
    assert FactorCorrelationAnalyzer().method == "spearman"


def test_result_is_frozen() -> None:
    """FactorCorrelationResult is an immutable slotted dataclass."""
    result = _analyzer(method="pearson").analyze(_frame(), ("alpha", "beta"))
    assert is_dataclass(result)
    assert isinstance(result, FactorCorrelationResult)
    with pytest.raises(FrozenInstanceError):
        result.method = "spearman"  # type: ignore[misc]


# --- pearson / spearman ---


def test_pearson_perfect_positive_and_negative() -> None:
    """Pearson recovers perfect positive and negative linear relationships."""
    result = _analyzer(method="pearson").analyze(_frame(), ("alpha", "beta", "gamma"))
    assert result.method == "pearson"
    assert result.factor_names == ("alpha", "beta", "gamma")
    assert result.matrix[0][1] == pytest.approx(1.0)
    assert result.matrix[0][2] == pytest.approx(-1.0)
    assert result.matrix[1][2] == pytest.approx(-1.0)


def test_spearman_perfect_for_monotone_nonlinear() -> None:
    """Spearman is perfect for monotone nonlinear factor pairs."""
    result = _analyzer(method="spearman").analyze(_frame(), ("alpha", "delta"))
    assert result.method == "spearman"
    assert result.matrix[0][1] == pytest.approx(1.0)
    assert result.matrix[1][0] == pytest.approx(1.0)


def test_pearson_below_one_for_nonlinear_monotone() -> None:
    """Pearson is below one for a nonlinear monotone relationship."""
    result = _analyzer(method="pearson").analyze(_frame(), ("alpha", "delta"))
    assert result.matrix[0][1] < 1.0
    assert result.matrix[0][1] > 0.9


# --- matrix properties ---


def test_matrix_is_symmetric() -> None:
    """The correlation matrix is symmetric."""
    result = _analyzer(method="pearson").analyze(
        _frame(),
        ("alpha", "beta", "gamma", "delta"),
    )
    size = len(result.factor_names)
    for row in range(size):
        for column in range(size):
            assert result.matrix[row][column] == pytest.approx(result.matrix[column][row])


def test_diagonal_equals_one() -> None:
    """Diagonal entries are exactly one."""
    result = _analyzer(method="spearman").analyze(
        _frame(),
        ("alpha", "beta", "gamma"),
    )
    for index in range(len(result.factor_names)):
        assert result.matrix[index][index] == 1.0


def test_matrix_shape_matches_factor_count() -> None:
    """Matrix dimensions match the number of requested factors."""
    result = _analyzer().analyze(_frame(), ("alpha", "beta", "gamma"))
    assert len(result.matrix) == 3
    assert all(len(row) == 3 for row in result.matrix)


# --- highly correlated pairs ---


def test_find_highly_correlated_pairs() -> None:
    """Highly correlated unique pairs are returned with coefficients."""
    result = _analyzer(method="pearson").analyze(
        _frame(),
        ("alpha", "beta", "gamma", "delta"),
    )
    pairs = find_highly_correlated(result, threshold=0.90)
    pair_map = {(left, right): value for left, right, value in pairs}
    assert ("alpha", "beta") in pair_map
    assert pair_map[("alpha", "beta")] == pytest.approx(1.0)
    assert ("beta", "alpha") not in pair_map


def test_threshold_filtering() -> None:
    """Raising the threshold filters weaker absolute associations."""
    result = _analyzer(method="pearson").analyze(
        _frame(),
        ("alpha", "beta", "gamma", "delta"),
    )
    strict = find_highly_correlated(result, threshold=0.999)
    loose = find_highly_correlated(result, threshold=0.90)
    assert len(strict) <= len(loose)
    assert ("alpha", "beta") in {(left, right) for left, right, _ in strict}


def test_threshold_includes_negative_correlations() -> None:
    """Absolute-value threshold includes strongly negative pairs."""
    result = _analyzer(method="pearson").analyze(_frame(), ("alpha", "gamma"))
    pairs = find_highly_correlated(result, threshold=0.90)
    assert len(pairs) == 1
    assert pairs[0][0] == "alpha"
    assert pairs[0][1] == "gamma"
    assert pairs[0][2] == pytest.approx(-1.0)


# --- null handling / immutability ---


def test_nulls_are_dropped_across_selected_factors() -> None:
    """Rows with any selected-factor null are excluded before estimation."""
    frame = pl.DataFrame(
        {
            "alpha": [1.0, 2.0, None, 4.0, 5.0],
            "beta": [2.0, 4.0, 6.0, 8.0, 10.0],
            "gamma": [1.0, 2.0, 3.0, 4.0, None],
        }
    )
    result = _analyzer(method="pearson").analyze(frame, ("alpha", "beta"))
    # Rows 0,1,3 remain for alpha/beta (row 2 dropped); gamma null irrelevant.
    assert result.matrix[0][1] == pytest.approx(1.0)


def test_input_frame_is_not_mutated() -> None:
    """analyze never mutates the caller-supplied DataFrame."""
    frame = _frame()
    original = frame.clone()
    _ = _analyzer(method="pearson").analyze(frame, ("alpha", "beta"))
    assert frame.equals(original)


# --- validation ---


def test_unknown_method_raises() -> None:
    """Unsupported correlation methods raise ResearchError."""
    with pytest.raises(ResearchError, match="unknown correlation method") as exc_info:
        FactorCorrelationAnalyzer(method="kendall")
    assert exc_info.value.error_code == "RESEARCH-CORR-001"


def test_less_than_two_factors_raises() -> None:
    """Fewer than two factor columns raise ResearchError."""
    with pytest.raises(ResearchError, match="at least two factor names") as exc_info:
        _analyzer().analyze(_frame(), ("alpha",))
    assert exc_info.value.error_code == "RESEARCH-CORR-002"


def test_missing_column_raises() -> None:
    """Missing factor columns raise ResearchError."""
    with pytest.raises(ResearchError, match="required column missing: missing") as exc_info:
        _analyzer().analyze(_frame(), ("alpha", "missing"))
    assert exc_info.value.error_code == "RESEARCH-CORR-003"


@pytest.mark.parametrize("threshold", [-0.1, 1.1, True, "0.9", None])
def test_invalid_threshold_raises(threshold: object) -> None:
    """Threshold values outside [0, 1] raise ResearchError."""
    result = _analyzer(method="pearson").analyze(_frame(), ("alpha", "beta"))
    with pytest.raises(ResearchError, match="threshold must be a number") as exc_info:
        find_highly_correlated(result, threshold=threshold)  # type: ignore[arg-type]
    assert exc_info.value.error_code == "RESEARCH-CORR-005"


@pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
def test_valid_threshold_boundary_accepted(threshold: float) -> None:
    """Boundary thresholds in [0, 1] are accepted."""
    result = _analyzer(method="pearson").analyze(_frame(), ("alpha", "beta"))
    pairs = find_highly_correlated(result, threshold=threshold)
    assert isinstance(pairs, tuple)


def test_insufficient_observations_after_null_drop_raises() -> None:
    """Fewer than two complete rows raise ResearchError."""
    frame = pl.DataFrame(
        {
            "alpha": [1.0, None],
            "beta": [2.0, 4.0],
        }
    )
    with pytest.raises(ResearchError, match="insufficient observations") as exc_info:
        _analyzer().analyze(frame, ("alpha", "beta"))
    assert exc_info.value.error_code == "RESEARCH-CORR-004"


def test_package_exports_factor_correlation() -> None:
    """Factor correlation symbols are exported from the research package."""
    import cqros.research as research_package

    assert "FactorCorrelationAnalyzer" in research_package.__all__
    assert "FactorCorrelationResult" in research_package.__all__
    assert "find_highly_correlated" in research_package.__all__
    assert research_package.FactorCorrelationAnalyzer is FactorCorrelationAnalyzer
