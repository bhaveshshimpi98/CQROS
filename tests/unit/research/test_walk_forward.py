"""Unit tests for CQROS walk-forward factor research validation."""

from __future__ import annotations

import statistics
from dataclasses import FrozenInstanceError, is_dataclass
from datetime import UTC, datetime
from unittest.mock import MagicMock

import polars as pl
import pytest

from cqros.core.exceptions import ResearchError
from cqros.research.experiment import (
    ExperimentDefinition,
    ExperimentResult,
    ResearchExperiment,
)
from cqros.research.factor_correlation import FactorCorrelationAnalyzer
from cqros.research.factor_decay import FactorDecayAnalyzer
from cqros.research.factor_stability import FactorStabilityAnalyzer
from cqros.research.information_coefficient import (
    InformationCoefficient,
    InformationCoefficientResult,
)
from cqros.research.quantile_analysis import QuantileAnalysisResult, QuantileAnalyzer
from cqros.research.rank_ic import RankICResult, RankInformationCoefficient
from cqros.research.target import ForwardReturnTarget, TargetDefinition
from cqros.research.walk_forward import (
    WalkForwardResult,
    WalkForwardValidator,
    WalkForwardWindow,
)


def _definition(**overrides: object) -> ExperimentDefinition:
    """Build an experiment definition with optional overrides."""
    values: dict[str, object] = {
        "name": "walk_forward_experiment",
        "description": "Walk-forward factor evaluation",
        "factor_columns": ("factor_a",),
        "price_column": "close",
        "target_horizon": 1,
        "ic_method": "pearson",
        "quantiles": 5,
        "decay_horizons": (1, 2),
        "stability_window": 10,
        "correlation_threshold": 0.90,
    }
    values.update(overrides)
    return ExperimentDefinition(**values)  # type: ignore[arg-type]


def _research_frame(rows: int = 80, *, factors: tuple[str, ...] = ("factor_a",)) -> pl.DataFrame:
    """Build a deterministic research frame for walk-forward tests."""
    close = [100.0]
    for index in range(rows - 1):
        shock = (((index * 47) % 13) - 6) / 50.0
        close.append(close[-1] * (1.0 + shock))
    data: dict[str, object] = {"close": close}
    for position, name in enumerate(factors):
        data[name] = [
            (
                (close[index + 1] / close[index]) - 1.0 + (0.01 * position)
                if index < rows - 1
                else None
            )
            for index in range(rows)
        ]
    return pl.DataFrame(data)


def _forward_target(definition: ExperimentDefinition) -> ForwardReturnTarget:
    """Build a forward-return target aligned to the experiment definition."""
    return ForwardReturnTarget(
        TargetDefinition(
            name="walk_forward_forward_return",
            horizon=definition.target_horizon,
            price_column=definition.price_column,
            output_column="forward_return",
        )
    )


def _real_experiment(definition: ExperimentDefinition) -> ResearchExperiment:
    """Build a ResearchExperiment with real injected collaborators."""
    return ResearchExperiment(
        forward_return_target=_forward_target(definition),
        information_coefficient=InformationCoefficient(method="pearson"),
        rank_information_coefficient=RankInformationCoefficient(),
        quantile_analyzer=QuantileAnalyzer(quantiles=definition.quantiles),
        factor_decay_analyzer=FactorDecayAnalyzer(method="pearson"),
        factor_stability_analyzer=FactorStabilityAnalyzer(method="pearson"),
        factor_correlation_analyzer=FactorCorrelationAnalyzer(method="pearson"),
    )


def _validator(definition: ExperimentDefinition) -> WalkForwardValidator:
    """Build a walk-forward validator with a real experiment engine."""
    return WalkForwardValidator(_real_experiment(definition))


def _mock_experiment_result(
    *,
    definition: ExperimentDefinition,
    ic: float,
    rank_ic: float,
    spread: float,
) -> ExperimentResult:
    """Build a minimal experiment result for mocked walk-forward runs."""
    factor = definition.factor_columns[0]
    return ExperimentResult(
        definition=definition,
        target=TargetDefinition(
            name="mock_target",
            horizon=definition.target_horizon,
            price_column=definition.price_column,
            output_column="forward_return",
        ),
        information_coefficients=(
            InformationCoefficientResult(
                factor_column=factor,
                target_column="forward_return",
                method="pearson",
                observations=10,
                coefficient=ic,
                p_value=0.1,
            ),
        ),
        rank_information_coefficients=(
            RankICResult(
                factor_column=factor,
                target_column="forward_return",
                observations=10,
                coefficient=rank_ic,
                p_value=0.1,
            ),
        ),
        quantile_results=(
            QuantileAnalysisResult(
                factor_column=factor,
                target_column="forward_return",
                quantiles=5,
                statistics=(),
                top_minus_bottom=spread,
                monotonic=True,
            ),
        ),
        decay_results=(),
        stability_results=(),
        correlation_result=None,
        highly_correlated_pairs=(),
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        completed_at=datetime(2026, 1, 1, tzinfo=UTC),
        duration_seconds=0.0,
    )


# --- immutability ---


def test_walk_forward_types_are_frozen() -> None:
    """Walk-forward result dataclasses are immutable."""
    definition = _definition(stability_window=10, decay_horizons=(1, 2))
    result = _validator(definition).run(
        _research_frame(60),
        definition,
        train_size=20,
        test_size=20,
        step_size=20,
    )
    assert is_dataclass(result)
    assert isinstance(result, WalkForwardResult)
    assert isinstance(result.windows[0], WalkForwardWindow)
    with pytest.raises(FrozenInstanceError):
        result.mean_ic = 0.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.windows[0].window_index = 99  # type: ignore[misc]


def test_input_frame_is_not_mutated() -> None:
    """run never mutates the caller-supplied DataFrame."""
    definition = _definition(stability_window=10, decay_horizons=(1, 2))
    frame = _research_frame(60)
    original = frame.clone()
    _ = _validator(definition).run(
        frame,
        definition,
        train_size=20,
        test_size=20,
        step_size=20,
    )
    assert frame.equals(original)
    assert "forward_return" not in frame.columns


# --- multiple / rolling windows ---


def test_multiple_non_overlapping_windows() -> None:
    """Consecutive non-overlapping windows are indexed with correct bounds."""
    definition = _definition(stability_window=10, decay_horizons=(1, 2))
    result = _validator(definition).run(
        _research_frame(60),
        definition,
        train_size=20,
        test_size=10,
        step_size=20,
    )
    assert len(result.windows) == 2
    assert result.windows[0].window_index == 0
    assert result.windows[0].train_start == 0
    assert result.windows[0].train_end == 20
    assert result.windows[0].test_start == 20
    assert result.windows[0].test_end == 30
    assert result.windows[1].window_index == 1
    assert result.windows[1].train_start == 20
    assert result.windows[1].train_end == 40
    assert result.windows[1].test_start == 40
    assert result.windows[1].test_end == 50


def test_rolling_windows_advance_by_step_size() -> None:
    """Overlapping rolling windows advance train_start by step_size."""
    definition = _definition(stability_window=10, decay_horizons=(1, 2))
    result = _validator(definition).run(
        _research_frame(55),
        definition,
        train_size=20,
        test_size=10,
        step_size=5,
    )
    # Windows start at train_start = 0, 5, 10, 15, 20, 25
    # Last complete: train_start=25 -> test_end=55
    assert len(result.windows) == 6
    for index, window in enumerate(result.windows):
        assert window.window_index == index
        assert window.train_start == index * 5
        assert window.train_end == window.train_start + 20
        assert window.test_start == window.train_end
        assert window.test_end == window.test_start + 10


def test_experiment_runs_on_each_test_slice() -> None:
    """ResearchExperiment is invoked once per window on the test slice only."""
    definition = _definition(stability_window=5, decay_horizons=(1,))
    frame = _research_frame(30)
    experiment = MagicMock(spec=ResearchExperiment)
    experiment.run.side_effect = [
        _mock_experiment_result(definition=definition, ic=0.4, rank_ic=0.3, spread=0.01),
        _mock_experiment_result(definition=definition, ic=0.6, rank_ic=0.5, spread=0.03),
    ]
    result = WalkForwardValidator(experiment).run(
        frame,
        definition,
        train_size=10,
        test_size=10,
        step_size=10,
    )
    assert experiment.run.call_count == 2
    first_frame = experiment.run.call_args_list[0].args[0]
    second_frame = experiment.run.call_args_list[1].args[0]
    assert first_frame.equals(frame.slice(10, 10))
    assert second_frame.equals(frame.slice(20, 10))
    assert result.windows[0].experiment_result.information_coefficients[0].coefficient == 0.4
    assert result.windows[1].experiment_result.information_coefficients[0].coefficient == 0.6


# --- summary statistics ---


def test_summary_statistics_average_window_metrics() -> None:
    """Summary fields average per-window IC, Rank IC, and quantile spread."""
    definition = _definition(stability_window=5, decay_horizons=(1,))
    experiment = MagicMock(spec=ResearchExperiment)
    experiment.run.side_effect = [
        _mock_experiment_result(definition=definition, ic=0.2, rank_ic=0.4, spread=0.01),
        _mock_experiment_result(definition=definition, ic=0.6, rank_ic=0.8, spread=0.03),
    ]
    result = WalkForwardValidator(experiment).run(
        _research_frame(30),
        definition,
        train_size=10,
        test_size=10,
        step_size=10,
    )
    assert result.mean_ic == pytest.approx(0.4)
    assert result.mean_rank_ic == pytest.approx(0.6)
    assert result.mean_quantile_spread == pytest.approx(0.02)
    expected_stability = 1.0 - (statistics.stdev([0.2, 0.6]) / 0.4)
    assert result.stability == pytest.approx(expected_stability)
    assert result.definition is definition


def test_stability_is_one_when_ic_is_constant() -> None:
    """Constant window IC yields perfect stability."""
    definition = _definition(stability_window=5, decay_horizons=(1,))
    experiment = MagicMock(spec=ResearchExperiment)
    experiment.run.side_effect = [
        _mock_experiment_result(definition=definition, ic=0.5, rank_ic=0.5, spread=0.02),
        _mock_experiment_result(definition=definition, ic=0.5, rank_ic=0.5, spread=0.02),
    ]
    result = WalkForwardValidator(experiment).run(
        _research_frame(30),
        definition,
        train_size=10,
        test_size=10,
        step_size=10,
    )
    assert result.stability == pytest.approx(1.0)


# --- validation ---


@pytest.mark.parametrize(
    ("kwargs", "error_code", "match"),
    [
        (
            {"train_size": 0, "test_size": 10, "step_size": 5},
            "RESEARCH-WF-001",
            "train_size must be an integer greater than 0",
        ),
        (
            {"train_size": 10, "test_size": 0, "step_size": 5},
            "RESEARCH-WF-002",
            "test_size must be an integer greater than 0",
        ),
        (
            {"train_size": 10, "test_size": 10, "step_size": 0},
            "RESEARCH-WF-003",
            "step_size must be an integer greater than 0",
        ),
        (
            {"train_size": -1, "test_size": 10, "step_size": 5},
            "RESEARCH-WF-001",
            "train_size must be an integer greater than 0",
        ),
    ],
)
def test_invalid_window_sizes_raise(
    kwargs: dict[str, int],
    error_code: str,
    match: str,
) -> None:
    """Non-positive window sizes raise ResearchError."""
    definition = _definition()
    with pytest.raises(ResearchError, match=match) as exc_info:
        _validator(definition).run(_research_frame(40), definition, **kwargs)
    assert exc_info.value.error_code == error_code


def test_insufficient_rows_raise() -> None:
    """Frames shorter than one train/test window raise ResearchError."""
    definition = _definition()
    with pytest.raises(ResearchError, match="insufficient rows") as exc_info:
        _validator(definition).run(
            _research_frame(25),
            definition,
            train_size=20,
            test_size=10,
            step_size=5,
        )
    assert exc_info.value.error_code == "RESEARCH-WF-004"
    assert exc_info.value.details["minimum_rows"] == 30


def test_bool_sizes_are_rejected() -> None:
    """Boolean sizes are rejected even though bool subclasses int."""
    definition = _definition()
    with pytest.raises(ResearchError, match="train_size") as exc_info:
        _validator(definition).run(
            _research_frame(40),
            definition,
            train_size=True,  # type: ignore[arg-type]
            test_size=10,
            step_size=5,
        )
    assert exc_info.value.error_code == "RESEARCH-WF-001"


# --- package export ---


def test_package_exports_walk_forward() -> None:
    """Walk-forward symbols are exported from the research package."""
    import cqros.research as research_package

    assert "WalkForwardValidator" in research_package.__all__
    assert "WalkForwardResult" in research_package.__all__
    assert "WalkForwardWindow" in research_package.__all__
    assert research_package.WalkForwardValidator is WalkForwardValidator
