"""Unit tests for CQROS Research Experiment Engine."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass
from unittest.mock import MagicMock

import polars as pl
import pytest

from cqros.core.exceptions import ResearchError
from cqros.research.experiment import (
    ExperimentDefinition,
    ExperimentResult,
    ResearchExperiment,
)
from cqros.research.factor_correlation import (
    FactorCorrelationAnalyzer,
    FactorCorrelationResult,
)
from cqros.research.factor_decay import FactorDecayAnalyzer, FactorDecayResult
from cqros.research.factor_stability import FactorStabilityAnalyzer, FactorStabilityResult
from cqros.research.information_coefficient import (
    InformationCoefficient,
    InformationCoefficientResult,
)
from cqros.research.quantile_analysis import QuantileAnalysisResult, QuantileAnalyzer
from cqros.research.rank_ic import RankICResult, RankInformationCoefficient
from cqros.research.target import ForwardReturnTarget, TargetDefinition


def _definition(**overrides: object) -> ExperimentDefinition:
    """Build an experiment definition with optional overrides."""
    values: dict[str, object] = {
        "name": "momentum_experiment",
        "description": "Evaluate momentum factors",
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


def _research_frame(rows: int = 40, *, factors: tuple[str, ...] = ("factor_a",)) -> pl.DataFrame:
    """Build a deterministic research frame for experiment tests."""
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
            name="experiment_forward_return",
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


# --- definition / immutability ---


def test_experiment_definition_is_frozen() -> None:
    """ExperimentDefinition is an immutable slotted dataclass."""
    definition = _definition()
    assert is_dataclass(definition)
    with pytest.raises(FrozenInstanceError):
        definition.name = "other"  # type: ignore[misc]


def test_experiment_definition_requires_at_least_one_factor() -> None:
    """Empty factor_columns raise ResearchError."""
    with pytest.raises(ResearchError, match="at least one factor") as exc_info:
        _definition(factor_columns=())
    assert exc_info.value.error_code == "RESEARCH-EXPERIMENT-003"


def test_experiment_result_is_frozen() -> None:
    """ExperimentResult produced by a run is immutable."""
    definition = _definition(stability_window=10, decay_horizons=(1, 2))
    result = _real_experiment(definition).run(_research_frame(40), definition)
    assert isinstance(result, ExperimentResult)
    with pytest.raises(FrozenInstanceError):
        result.duration_seconds = 0.0  # type: ignore[misc]


# --- single / multiple factors ---


def test_single_factor_experiment_populates_per_factor_results() -> None:
    """A single-factor experiment fills all per-factor result collections."""
    definition = _definition(
        factor_columns=("factor_a",),
        stability_window=10,
        decay_horizons=(1, 2),
        quantiles=5,
    )
    result = _real_experiment(definition).run(_research_frame(40), definition)
    assert result.definition is definition
    assert result.target.horizon == 1
    assert result.target.price_column == "close"
    assert len(result.information_coefficients) == 1
    assert len(result.rank_information_coefficients) == 1
    assert len(result.quantile_results) == 1
    assert len(result.decay_results) == 1
    assert len(result.stability_results) == 1
    assert result.correlation_result is None
    assert result.highly_correlated_pairs == ()
    assert result.completed_at >= result.started_at
    assert result.duration_seconds >= 0.0


def test_multiple_factors_compute_correlation() -> None:
    """Multiple factors produce a correlation matrix and pair scan."""
    definition = _definition(
        factor_columns=("factor_a", "factor_b"),
        stability_window=10,
        decay_horizons=(1, 2),
        correlation_threshold=0.0,
    )
    frame = _research_frame(40, factors=("factor_a", "factor_b"))
    result = _real_experiment(definition).run(frame, definition)
    assert len(result.information_coefficients) == 2
    assert len(result.decay_results) == 2
    assert isinstance(result.correlation_result, FactorCorrelationResult)
    assert result.correlation_result.factor_names == ("factor_a", "factor_b")
    assert len(result.highly_correlated_pairs) >= 1


# --- injected analyzers called ---


def test_all_injected_analyzers_are_called_for_each_factor() -> None:
    """Every injected collaborator is invoked during run."""
    definition = _definition(
        factor_columns=("factor_a", "factor_b"),
        stability_window=10,
        decay_horizons=(1, 2),
    )
    frame = _research_frame(30, factors=("factor_a", "factor_b"))
    evaluated = frame.with_columns(pl.Series("forward_return", [0.01] * 30))

    forward = MagicMock(spec=ForwardReturnTarget)
    forward.horizon = definition.target_horizon
    forward.price_column = definition.price_column
    forward.output_column = "forward_return"
    forward.definition = TargetDefinition(
        name="mock_target",
        horizon=definition.target_horizon,
        price_column=definition.price_column,
        output_column="forward_return",
    )
    forward.transform.return_value = evaluated

    ic = MagicMock(spec=InformationCoefficient)
    ic.compute.return_value = InformationCoefficientResult(
        factor_column="factor_a",
        target_column="forward_return",
        method="pearson",
        observations=10,
        coefficient=0.5,
        p_value=0.1,
    )
    rank_ic = MagicMock(spec=RankInformationCoefficient)
    rank_ic.compute.return_value = RankICResult(
        factor_column="factor_a",
        target_column="forward_return",
        observations=10,
        coefficient=0.5,
        p_value=0.1,
    )
    quantiles = MagicMock(spec=QuantileAnalyzer)
    quantiles.analyze.return_value = MagicMock(spec=QuantileAnalysisResult)
    decay = MagicMock(spec=FactorDecayAnalyzer)
    decay.analyze.return_value = MagicMock(spec=FactorDecayResult)
    stability = MagicMock(spec=FactorStabilityAnalyzer)
    stability.analyze.return_value = MagicMock(spec=FactorStabilityResult)
    correlation = MagicMock(spec=FactorCorrelationAnalyzer)
    correlation.analyze.return_value = FactorCorrelationResult(
        factor_names=("factor_a", "factor_b"),
        method="pearson",
        matrix=((1.0, 0.95), (0.95, 1.0)),
    )

    experiment = ResearchExperiment(
        forward_return_target=forward,
        information_coefficient=ic,
        rank_information_coefficient=rank_ic,
        quantile_analyzer=quantiles,
        factor_decay_analyzer=decay,
        factor_stability_analyzer=stability,
        factor_correlation_analyzer=correlation,
    )
    result = experiment.run(frame, definition)

    forward.transform.assert_called_once_with(frame)
    assert ic.compute.call_count == 2
    assert rank_ic.compute.call_count == 2
    assert quantiles.analyze.call_count == 2
    assert decay.analyze.call_count == 2
    assert stability.analyze.call_count == 2
    correlation.analyze.assert_called_once()
    assert result.correlation_result is not None
    assert result.highly_correlated_pairs == (("factor_a", "factor_b", 0.95),)


def test_correlation_analyzer_not_called_for_single_factor() -> None:
    """Correlation analysis is skipped when only one factor is present."""
    definition = _definition(factor_columns=("factor_a",), stability_window=10)
    frame = _research_frame(30)
    evaluated = frame.with_columns(pl.Series("forward_return", [0.01] * 30))

    forward = MagicMock(spec=ForwardReturnTarget)
    forward.horizon = 1
    forward.price_column = "close"
    forward.output_column = "forward_return"
    forward.definition = TargetDefinition(
        name="mock_target",
        horizon=1,
        price_column="close",
        output_column="forward_return",
    )
    forward.transform.return_value = evaluated

    correlation = MagicMock(spec=FactorCorrelationAnalyzer)
    experiment = ResearchExperiment(
        forward_return_target=forward,
        information_coefficient=MagicMock(
            spec=InformationCoefficient,
            compute=MagicMock(
                return_value=InformationCoefficientResult(
                    factor_column="factor_a",
                    target_column="forward_return",
                    method="pearson",
                    observations=10,
                    coefficient=0.5,
                    p_value=0.1,
                )
            ),
        ),
        rank_information_coefficient=MagicMock(
            spec=RankInformationCoefficient,
            compute=MagicMock(
                return_value=RankICResult(
                    factor_column="factor_a",
                    target_column="forward_return",
                    observations=10,
                    coefficient=0.5,
                    p_value=0.1,
                )
            ),
        ),
        quantile_analyzer=MagicMock(
            spec=QuantileAnalyzer,
            analyze=MagicMock(return_value=MagicMock(spec=QuantileAnalysisResult)),
        ),
        factor_decay_analyzer=MagicMock(
            spec=FactorDecayAnalyzer,
            analyze=MagicMock(return_value=MagicMock(spec=FactorDecayResult)),
        ),
        factor_stability_analyzer=MagicMock(
            spec=FactorStabilityAnalyzer,
            analyze=MagicMock(return_value=MagicMock(spec=FactorStabilityResult)),
        ),
        factor_correlation_analyzer=correlation,
    )
    result = experiment.run(frame, definition)
    correlation.analyze.assert_not_called()
    assert result.correlation_result is None
    assert result.highly_correlated_pairs == ()


# --- validation / immutability ---


def test_missing_factor_column_raises() -> None:
    """Missing factor columns raise ResearchError before analysis."""
    definition = _definition(factor_columns=("missing",))
    with pytest.raises(ResearchError, match="required column missing: missing") as exc_info:
        _real_experiment(definition).run(_research_frame(20), definition)
    assert exc_info.value.error_code == "RESEARCH-EXPERIMENT-004"


def test_missing_price_column_raises() -> None:
    """Missing price column raises ResearchError."""
    definition = _definition(price_column="close")
    frame = pl.DataFrame({"factor_a": [1.0, 2.0, 3.0]})
    with pytest.raises(ResearchError, match="required column missing: close") as exc_info:
        _real_experiment(definition).run(frame, definition)
    assert exc_info.value.error_code == "RESEARCH-EXPERIMENT-005"


def test_injected_target_mismatch_raises() -> None:
    """Injected forward-return target must match definition horizon/price."""
    definition = _definition(target_horizon=1, price_column="close")
    mismatched = ForwardReturnTarget(
        TargetDefinition(
            name="mismatch",
            horizon=2,
            price_column="close",
            output_column="forward_return",
        )
    )
    experiment = ResearchExperiment(
        forward_return_target=mismatched,
        information_coefficient=InformationCoefficient(method="pearson"),
        rank_information_coefficient=RankInformationCoefficient(),
        quantile_analyzer=QuantileAnalyzer(quantiles=5),
        factor_decay_analyzer=FactorDecayAnalyzer(method="pearson"),
        factor_stability_analyzer=FactorStabilityAnalyzer(method="pearson"),
        factor_correlation_analyzer=FactorCorrelationAnalyzer(method="pearson"),
    )
    with pytest.raises(ResearchError, match="does not match experiment definition"):
        experiment.run(_research_frame(20), definition)


def test_input_frame_is_not_mutated() -> None:
    """run never mutates the caller-supplied DataFrame."""
    definition = _definition(stability_window=10, decay_horizons=(1, 2))
    frame = _research_frame(40)
    original = frame.clone()
    _ = _real_experiment(definition).run(frame, definition)
    assert frame.equals(original)
    assert "forward_return" not in frame.columns


def test_package_exports_experiment() -> None:
    """Experiment symbols are exported from the research package."""
    import cqros.research as research_package

    assert "ResearchExperiment" in research_package.__all__
    assert "ExperimentDefinition" in research_package.__all__
    assert "ExperimentResult" in research_package.__all__
    assert research_package.ResearchExperiment is ResearchExperiment
