"""Unit tests for CQROS factor decay analysis."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass

import polars as pl
import pytest

from cqros.core.exceptions import ResearchError
from cqros.research.factor_decay import (
    DecayPoint,
    FactorDecayAnalyzer,
    FactorDecayResult,
)
from cqros.research.information_coefficient import InformationCoefficient


def _analyzer(method: str = "spearman") -> FactorDecayAnalyzer:
    """Build a factor decay analyzer."""
    return FactorDecayAnalyzer(method=method)


def _noisy_price_frame(rows: int = 128) -> pl.DataFrame:
    """Build a noisy price series with a one-step predictive factor.

    The factor equals the one-step simple return shock used to evolve
    prices (lookahead allowed only in tests). Horizon-1 IC is therefore
    perfect, while longer compounded horizons decay.
    """
    shocks = [(((index * 47) % 13) - 6) / 50.0 for index in range(rows - 1)]
    close: list[float] = [100.0]
    for shock in shocks:
        close.append(close[-1] * (1.0 + shock))
    factor: list[float | None] = [*shocks, None]
    return pl.DataFrame({"factor": factor, "close": close})


# --- metadata ---


def test_default_method_is_spearman() -> None:
    """Constructor defaults to Spearman IC."""
    assert FactorDecayAnalyzer().method == "spearman"


def test_result_types_are_frozen() -> None:
    """Result dataclasses are immutable."""
    result = _analyzer().analyze(_noisy_price_frame(), "factor", horizons=(1, 2))
    assert is_dataclass(result)
    assert isinstance(result, FactorDecayResult)
    assert isinstance(result.points[0], DecayPoint)
    with pytest.raises(FrozenInstanceError):
        result.half_life = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.points[0].coefficient = 0.0  # type: ignore[misc]


# --- multiple horizons / declining IC ---


def test_multiple_horizons_are_evaluated_in_order() -> None:
    """Each requested horizon produces one ordered decay point."""
    horizons = (1, 2, 4, 8)
    result = _analyzer(method="pearson").analyze(
        _noisy_price_frame(),
        "factor",
        horizons=horizons,
    )
    assert result.factor_column == "factor"
    assert result.price_column == "close"
    assert result.method == "pearson"
    assert tuple(point.horizon for point in result.points) == horizons
    assert len(result.points) == len(horizons)
    assert all(point.observations >= 2 for point in result.points)


def test_declining_ic_across_horizons() -> None:
    """Absolute IC declines as the forward horizon increases."""
    result = _analyzer(method="pearson").analyze(
        _noisy_price_frame(160),
        "factor",
        horizons=(1, 2, 4, 8, 12),
    )
    coefficients = [abs(point.coefficient) for point in result.points]
    assert coefficients[0] == pytest.approx(1.0)
    assert coefficients[0] > coefficients[-1]


def test_default_horizons_are_used_when_omitted() -> None:
    """analyze uses the default horizon schedule when none is supplied."""
    result = _analyzer(method="pearson").analyze(_noisy_price_frame(80), "factor")
    assert tuple(point.horizon for point in result.points) == (1, 2, 4, 8, 12, 24)


# --- half-life ---


def test_half_life_detection() -> None:
    """Half-life is the first horizon below 50% of the first absolute IC."""
    result = _analyzer(method="pearson").analyze(
        _noisy_price_frame(200),
        "factor",
        horizons=(1, 2, 4, 8, 12, 24),
    )
    baseline = abs(result.points[0].coefficient)
    threshold = 0.5 * baseline
    assert result.half_life is not None
    half_life_point = next(point for point in result.points if point.horizon == result.half_life)
    assert abs(half_life_point.coefficient) < threshold
    prior = [point for point in result.points if point.horizon < result.half_life]
    assert all(abs(point.coefficient) >= threshold for point in prior)


def test_no_half_life_when_ic_stays_strong() -> None:
    """Half-life is None when absolute IC never falls below the threshold."""
    rows = 40
    close = [100.0 * (1.01**index) for index in range(rows)]
    factor = [float(value) for value in range(rows)]
    frame = pl.DataFrame({"factor": factor, "close": close})
    result = _analyzer(method="spearman").analyze(frame, "factor", horizons=(1, 2))
    assert result.half_life is None
    baseline = abs(result.points[0].coefficient)
    assert all(abs(point.coefficient) >= 0.5 * baseline for point in result.points)


# --- null handling / immutability ---


def test_null_factor_values_are_handled() -> None:
    """Null factor rows are dropped inside IC estimation per horizon."""
    base = _noisy_price_frame(40)
    factor_values = base.get_column("factor").to_list()
    for index in range(0, len(factor_values), 7):
        factor_values[index] = None
    frame = base.with_columns(pl.Series("factor", factor_values))
    result = _analyzer(method="pearson").analyze(frame, "factor", horizons=(1, 2))
    assert len(result.points) == 2
    assert all(point.observations >= 2 for point in result.points)


def test_input_frame_is_not_mutated() -> None:
    """analyze never mutates the caller-supplied DataFrame."""
    frame = _noisy_price_frame(32)
    original = frame.clone()
    _ = _analyzer(method="pearson").analyze(frame, "factor", horizons=(1, 2))
    assert frame.equals(original)
    assert "__cqros_decay_forward_return" not in frame.columns


# --- validation ---


def test_missing_factor_column_raises() -> None:
    """Missing factor column raises ResearchError."""
    frame = pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(ResearchError, match="required column missing: factor") as exc_info:
        _analyzer().analyze(frame, "factor", horizons=(1, 2))
    assert exc_info.value.error_code == "RESEARCH-DECAY-003"


def test_missing_price_column_raises() -> None:
    """Missing price column raises ResearchError."""
    frame = pl.DataFrame({"factor": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(ResearchError, match="required column missing: close") as exc_info:
        _analyzer().analyze(frame, "factor", horizons=(1, 2))
    assert exc_info.value.error_code == "RESEARCH-DECAY-004"


@pytest.mark.parametrize("horizon", [0, -1, True, 1.5, "1", None])
def test_invalid_horizon_entries_raise(horizon: object) -> None:
    """Non-positive or non-integer horizons raise ResearchError."""
    frame = _noisy_price_frame(16)
    with pytest.raises(ResearchError, match="horizons entries must be integers") as exc_info:
        _analyzer().analyze(frame, "factor", horizons=(1, horizon))  # type: ignore[arg-type]
    assert exc_info.value.error_code == "RESEARCH-DECAY-002"


def test_empty_horizons_raise() -> None:
    """An empty horizons sequence raises ResearchError."""
    with pytest.raises(ResearchError, match="horizons must contain at least one") as exc_info:
        _analyzer().analyze(_noisy_price_frame(16), "factor", horizons=())
    assert exc_info.value.error_code == "RESEARCH-DECAY-001"


def test_invalid_method_raises_at_construction() -> None:
    """Unsupported IC methods are rejected by the analyzer constructor."""
    with pytest.raises(ResearchError, match="unknown correlation method"):
        FactorDecayAnalyzer(method="kendall")


def test_reuses_information_coefficient_method() -> None:
    """Configured method matches InformationCoefficient's supported surface."""
    analyzer = FactorDecayAnalyzer(method="pearson")
    assert analyzer.method == InformationCoefficient(method="pearson").method


def test_package_exports_factor_decay() -> None:
    """Factor decay symbols are exported from the research package."""
    import cqros.research as research_package

    assert "FactorDecayAnalyzer" in research_package.__all__
    assert "FactorDecayResult" in research_package.__all__
    assert "DecayPoint" in research_package.__all__
    assert research_package.FactorDecayAnalyzer is FactorDecayAnalyzer
